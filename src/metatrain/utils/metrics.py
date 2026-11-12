import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed
from metatensor.torch import LabelsEntry, TensorMap


def _block_missing_from_prediction(
    prediction: TensorMap, block_key: Any, key: str
) -> bool:
    """Check (and warn) whether a target block is missing from the predictions.

    The model may legitimately not predict every block of the target (e.g. a
    composition model only fits invariant blocks of a spherical target), so such
    blocks are skipped when computing the error metrics. warnings.warn (rather
    than logging.warning) so this is only reported once, instead of on every
    batch.

    :param prediction: the prediction TensorMap for the target named ``key``.
    :param block_key: the target block key to look for in the predictions.
    :param key: the name of the target, used in the warning message.
    :return: whether the block is missing from the predictions.
    """
    if block_key not in prediction.keys:
        warnings.warn(
            f"Block {block_key} of target '{key}' is not present in "
            "the model's predictions. It will be skipped when "
            "computing the error metrics, which will be computed "
            "over fewer blocks than the full target.",
            stacklevel=3,
        )
        return True
    return False


def _separate_block_key(key: str, block_key: LabelsEntry) -> str:
    """Label ``key`` with the values of ``block_key``, e.g. ``"energy (_=0)"``.

    :param key: The name of the target.
    :param block_key: The key of one of its blocks.

    :return: The labeled key.
    """
    values = ",".join(
        f"{name}={int(value)}"
        for name, value in zip(block_key.names, block_key.values, strict=True)
    )
    return f"{key} ({values})"


def _get_global_keys(keys: List[str]) -> List[str]:
    # Collect all keys across ranks, in case some ranks have seen different keys than
    # others
    local_keys = list(keys)
    world_size = torch.distributed.get_world_size()
    gathered_keys: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_keys, local_keys)
    global_keys: set[str] = set()
    for keys_from_rank in gathered_keys:
        if keys_from_rank is not None:
            global_keys.update(keys_from_rank)
    return sorted(list(global_keys))


def _all_reduce_information(
    information: Dict[str, Tuple[float, int]], device: torch.device
) -> None:
    """Sum, in place, the accumulated values and numbers of elements of each key
    over all ranks, including the keys that this rank has not seen.

    :param information: The ``information`` of an accumulator.
    :param device: The local device to use for the communication.
    """
    for key in _get_global_keys(list(information.keys())):
        value, n_elems = information.get(key, (0.0, 0))
        value_tensor = torch.tensor(value, device=device)
        n_elems_tensor = torch.tensor(n_elems, device=device)
        torch.distributed.all_reduce(value_tensor)
        torch.distributed.all_reduce(n_elems_tensor)
        information[key] = (value_tensor.item(), n_elems_tensor.item())


class RMSEAccumulator:
    """Accumulates the RMSE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target.

    :param separate_blocks: if true, the RMSE will be computed separately for each
        block in the target and prediction ``TensorMap`` objects.
    """

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information: Dict[str, Tuple[float, int]] = {}
        """A dictionary mapping each target key to a tuple containing the sum of
        squared errors and the number of elements for which the error has been
        computed."""

        self.separate_blocks = separate_blocks
        """Whether the RMSE should be computed separately for each block in the
        target and prediction ``TensorMap`` objects."""

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.
        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        :param extra_data: A dictionary of extra data, where the keys correspond to
            mask keys (i.e. "{target_key}_mask"), and the values are the masks to apply
            when computing the RMSE.
        """

        for key, target in targets.items():
            prediction = predictions[key]

            # Get the mask from extra data if present
            mask = None
            if extra_data is not None:
                mask_key = f"{key}_mask"
                if mask_key in extra_data:
                    mask = extra_data[mask_key]

            for block_key in target.keys:
                if _block_missing_from_prediction(prediction, block_key, key):
                    continue

                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write = _separate_block_key(key, block_key)

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                if mask is None:
                    # Get a mask that ignores NaN values in the target
                    mask_as_tensor = ~torch.isnan(target_block.values)
                    rmse_value = (
                        (
                            (
                                prediction_block.values[mask_as_tensor]
                                - target_block.values[mask_as_tensor]
                            )
                            ** 2
                        )
                        .sum()
                        .item()
                    )
                else:
                    mask_as_tensor = mask.block(block_key).values
                    rmse_value = (
                        (
                            (
                                prediction_block.values[mask_as_tensor]
                                - target_block.values[mask_as_tensor]
                            )
                            ** 2
                        )
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + rmse_value,
                    self.information[key_to_write][1] + mask_as_tensor.sum().item(),
                )

                for gradient_name, target_gradient in target_block.gradients():
                    if (
                        f"{key_to_write}_{gradient_name}_gradients"
                        not in self.information
                    ):
                        self.information[
                            f"{key_to_write}_{gradient_name}_gradients"
                        ] = (0.0, 0)
                    prediction_gradient = prediction_block.gradient(gradient_name)

                    if mask is None:
                        # Get a mask that ignores NaN values in the target
                        mask_as_tensor = ~torch.isnan(target_gradient.values)
                        gradient_rmse_value = (
                            (
                                (
                                    prediction_gradient.values[mask_as_tensor]
                                    - target_gradient.values[mask_as_tensor]
                                )
                                ** 2
                            )
                            .sum()
                            .item()
                        )
                    else:
                        mask_as_tensor = (
                            mask.block(block_key).gradient(gradient_name).values
                        )
                        gradient_rmse_value = (
                            (
                                (
                                    prediction_gradient.values[mask_as_tensor]
                                    - target_gradient.values[mask_as_tensor]
                                )
                                ** 2
                            )
                            .sum()
                            .item()
                        )
                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + gradient_rmse_value,
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][1]
                        + mask_as_tensor.sum().item(),
                    )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """Finalizes the accumulator and returns the RMSE for each key.

        All keys will be returned as "{key} RMSE (per atom)" in the output dictionary,
        unless ``key`` contains one or more of the strings in ``not_per_atom``,
        in which case "{key} RMSE" will be returned.

        :param not_per_atom: a list of strings. If any of these strings are present in
            a key, the RMSE key will not be labeled as "(per atom)".
        :param is_distributed: if true, the RMSE will be computed across all ranks
            of the distributed system.
        :param device: the local device to use for the computation. Only needed if
            ``is_distributed`` is :obj:`python:True`.

        :return: The RMSE for each key.
        """

        if is_distributed:
            _all_reduce_information(self.information, device)

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} RMSE"
            else:
                out_key = f"{key} RMSE (per atom)"
            finalized_info[out_key] = (value[0] / value[1]) ** 0.5

        return finalized_info


class MAEAccumulator:
    """Accumulates the MAE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target.

    :param separate_blocks: if true, the RMSE will be computed separately for each
        block in the target and prediction ``TensorMap`` objects.
    """

    information: Dict[str, Tuple[float, int]]
    separate_blocks: bool

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information = {}
        """A dictionary mapping each target key to a tuple containing the sum of
        absolute errors and the number of elements for which the error has been
        computed."""

        self.separate_blocks = separate_blocks
        """Whether the MAE should be computed separately for each block in the
        target and prediction ``TensorMap`` objects."""

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.
        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        :param extra_data: A dictionary of extra data, where the keys correspond to
            mask keys (i.e. "{target_key}_mask"), and the values are the masks to apply
            when computing the MAE.
        """

        for key, target in targets.items():
            prediction = predictions[key]

            # Get the mask from extra data if present
            mask = None
            if extra_data is not None:
                mask_key = f"{key}_mask"
                if mask_key in extra_data:
                    mask = extra_data[mask_key]

            for block_key in target.keys:
                if _block_missing_from_prediction(prediction, block_key, key):
                    continue

                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write = _separate_block_key(key, block_key)

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                if mask is None:
                    # Get a mask that ignores NaN values in the target
                    mask_as_tensor = ~torch.isnan(target_block.values)
                    mae_value = (
                        (
                            prediction_block.values[mask_as_tensor]
                            - target_block.values[mask_as_tensor]
                        )
                        .abs()
                        .sum()
                        .item()
                    )
                else:
                    mask_as_tensor = mask.block(block_key).values
                    mae_value = (
                        (
                            prediction_block.values[mask_as_tensor]
                            - target_block.values[mask_as_tensor]
                        )
                        .abs()
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + mae_value,
                    self.information[key_to_write][1] + mask_as_tensor.sum().item(),
                )

                for gradient_name, target_gradient in target_block.gradients():
                    if (
                        f"{key_to_write}_{gradient_name}_gradients"
                        not in self.information
                    ):
                        self.information[
                            f"{key_to_write}_{gradient_name}_gradients"
                        ] = (0.0, 0)
                    prediction_gradient = prediction_block.gradient(gradient_name)

                    if mask is None:
                        # Get a mask that ignores NaN values in the target
                        mask_as_tensor = ~torch.isnan(target_gradient.values)
                        gradient_mae_value = (
                            (
                                prediction_gradient.values[mask_as_tensor]
                                - target_gradient.values[mask_as_tensor]
                            )
                            .abs()
                            .sum()
                            .item()
                        )
                    else:
                        mask_as_tensor = (
                            mask.block(block_key).gradient(gradient_name).values
                        )
                        gradient_mae_value = (
                            (
                                prediction_gradient.values[mask_as_tensor]
                                - target_gradient.values[mask_as_tensor]
                            )
                            .abs()
                            .sum()
                            .item()
                        )

                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + gradient_mae_value,
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][1]
                        + mask_as_tensor.sum().item(),
                    )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """Finalizes the accumulator and returns the MAE for each key.

        All keys will be returned as "{key} MAE (per atom)" in the output dictionary,
        unless ``key`` contains one or more of the strings in ``not_per_atom``,
        in which case "{key} MAE" will be returned.

        :param not_per_atom: a list of strings. If any of these strings are present in
            a key, the MAE key will not be labeled as "(per atom)".
        :param is_distributed: if true, the MAE will be computed across all ranks
            of the distributed system.
        :param device: the local device to use for the computation. Only needed if
            ``is_distributed`` is :obj:`python:True`.

        :return: The MAE for each key.
        """

        if is_distributed:
            _all_reduce_information(self.information, device)

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} MAE"
            else:
                out_key = f"{key} MAE (per atom)"
            finalized_info[out_key] = value[0] / value[1]

        return finalized_info


class EquivarianceAccumulator:
    """Accumulates the O(3) equivariance error of a model over a dataset.

    This consumes the ``o3::variance::<name>`` outputs of a metatomic
    ``SymmetrizedModel``, which hold, for every sample, the variance of the
    back-rotated predictions over a rotation grid, averaged over the components
    of each block. Pooling them with the same normalization as
    :class:`RMSEAccumulator` gives an error in the same units as the accuracy
    RMSE of the target, and a lower bound on it.

    :param separate_blocks: as in :class:`RMSEAccumulator`. The blocks are then not
        weighted by ``2 * o3_lambda + 1``.
    """

    def __init__(self, separate_blocks: bool = False) -> None:
        self.information: Dict[str, Tuple[float, int]] = {}
        """A dictionary mapping each target key to a tuple containing the sum of
        the variances and the number of elements over which it was accumulated."""

        self.separate_blocks = separate_blocks
        """Whether the equivariance error should be computed separately for each
        block of each target."""

    def update(self, variances: Dict[str, TensorMap]) -> None:
        """Updates the accumulator with the variances of a new batch.

        :param variances: a dictionary mapping each target name to the
            ``o3::variance::<name>`` output of a ``SymmetrizedModel``, already
            normalized per atom like the predictions.
        """
        for key, variance in variances.items():
            # the irrep columns are appended after any pre-existing ones, so the
            # position of "o3_lambda" is not fixed and it is absent altogether
            # for targets metatomic does not decompose
            key_names = list(variance.keys.names)
            lambda_index = (
                key_names.index("o3_lambda") if "o3_lambda" in key_names else None
            )

            for block_key, block in variance.items():
                if self.separate_blocks:
                    key_to_write = _separate_block_key(key, block_key)
                    multiplicity = 1
                elif lambda_index is None:
                    # every block of an undecomposed target shares the same
                    # number of components, so the weight cancels out
                    key_to_write = key
                    multiplicity = 1
                else:
                    key_to_write = key
                    # the spherical decomposition is orthonormal, so weighting each
                    # block by its number of components recovers the squared error
                    # summed over the components of the original (e.g. Cartesian)
                    # target
                    multiplicity = 2 * int(block_key.values[lambda_index]) + 1

                values = block.values
                sum_of_variances, n_elems = self.information.get(key_to_write, (0.0, 0))
                self.information[key_to_write] = (
                    sum_of_variances + multiplicity * values.sum().item(),
                    n_elems + multiplicity * values.shape[0] * values.shape[-1],
                )

    def finalize(
        self,
        not_per_atom: List[str],
        is_distributed: bool = False,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """Finalizes the accumulator and returns the equivariance error per key.

        All keys will be returned as "{key} equivariance RMSE (per atom)" in the
        output dictionary, unless ``key`` contains one or more of the strings in
        ``not_per_atom``, in which case "{key} equivariance RMSE" will be
        returned.

        :param not_per_atom: a list of strings. If any of these strings are
            present in a key, the metric will not be labeled as "(per atom)".
        :param is_distributed: if true, the equivariance error will be computed
            across all ranks of the distributed system.
        :param device: the local device to use for the computation. Only needed if
            ``is_distributed`` is :obj:`python:True`.

        :return: The equivariance error for each key.
        """
        if is_distributed:
            _all_reduce_information(self.information, device)

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} equivariance RMSE"
            else:
                out_key = f"{key} equivariance RMSE (per atom)"
            # the variance of a nearly-equivariant model is dominated by the
            # round-off of the quadrature and can come out slightly negative
            finalized_info[out_key] = (max(value[0], 0.0) / value[1]) ** 0.5

        return finalized_info


def get_selected_metric(metric_dict: Dict[str, float], selected_metric: str) -> float:
    """
    Selects and/or calculates a (user-)selected metric from a dictionary of metrics.

    This is useful when choosing the best model from a training run.

    :param metric_dict: A dictionary of metrics, where the keys are the names of the
        metrics and the values are the corresponding values.
    :param selected_metric: The metric to return. This can be one of the following:
        - "loss": return the loss value
        - "rmse_prod": return the product of all RMSEs
        - "mae_prod": return the product of all MAEs

    :return: The value of the selected metric.
    """
    if selected_metric == "loss":
        metric = metric_dict["loss"]
    elif selected_metric == "rmse_prod":
        metric = 1
        for key in metric_dict:
            if "RMSE" in key:
                metric *= metric_dict[key]
    elif selected_metric == "mae_prod":
        metric = 1
        for key in metric_dict:
            if "MAE" in key:
                metric *= metric_dict[key]
    else:
        raise ValueError(
            f"Selected metric {selected_metric} not recognized. "
            "Please select from 'loss', 'rmse_prod', or 'mae_prod'."
        )
    return metric
