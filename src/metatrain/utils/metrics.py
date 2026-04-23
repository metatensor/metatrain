import copy
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch.distributed
from metatensor.torch import TensorMap


_ZERO_SUPPORT_WARNED_METRICS: set[str] = set()


def _format_block_key(block_key) -> str:
    if len(block_key.names) == 0:
        return "<default>"
    return ", ".join(
        f"{name}={int(value)}"
        for name, value in zip(block_key.names, block_key.values, strict=True)
    )


def _warn_zero_support_metrics(omitted_metrics: List[str]) -> None:
    new_metrics = [key for key in omitted_metrics if key not in _ZERO_SUPPORT_WARNED_METRICS]
    if not new_metrics:
        return

    _ZERO_SUPPORT_WARNED_METRICS.update(new_metrics)
    warnings.warn(
        "Omitting metrics with zero valid support: " + ", ".join(sorted(new_metrics)),
        RuntimeWarning,
        stacklevel=2,
    )


def _raise_if_nonfinite_metric_contribution(
    value: float,
    *,
    target_name: str,
    block_key,
    gradient_name: Optional[str],
) -> None:
    if math.isfinite(value):
        return

    channel = "values" if gradient_name is None else f"gradient '{gradient_name}'"
    raise ValueError(
        "Encountered a non-finite metric contribution on supported entries for "
        f"target '{target_name}', block {_format_block_key(block_key)}, {channel}. "
        "This usually indicates non-finite predictions on valid supervision."
    )


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
                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += " ("
                    for name, value in zip(
                        block_key.names, block_key.values, strict=True
                    ):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

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

                if mask_as_tensor.any():
                    _raise_if_nonfinite_metric_contribution(
                        rmse_value,
                        target_name=key,
                        block_key=block_key,
                        gradient_name=None,
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
                    if mask_as_tensor.any():
                        _raise_if_nonfinite_metric_contribution(
                            gradient_rmse_value,
                            target_name=key,
                            block_key=block_key,
                            gradient_name=gradient_name,
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
            for key in sorted(self.information.keys()):
                sse = torch.tensor(self.information[key][0], device=device)
                n_elems = torch.tensor(self.information[key][1], device=device)
                torch.distributed.all_reduce(sse)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sse.item(), n_elems.item())

        finalized_info = {}
        omitted_metrics = []
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} RMSE"
            else:
                out_key = f"{key} RMSE (per atom)"
            if value[1] == 0:
                omitted_metrics.append(out_key)
                continue
            finalized_info[out_key] = (value[0] / value[1]) ** 0.5

        _warn_zero_support_metrics(omitted_metrics)

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
                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += " ("
                    for name, value in zip(
                        block_key.names, block_key.values, strict=True
                    ):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

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

                if mask_as_tensor.any():
                    _raise_if_nonfinite_metric_contribution(
                        mae_value,
                        target_name=key,
                        block_key=block_key,
                        gradient_name=None,
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

                    if mask_as_tensor.any():
                        _raise_if_nonfinite_metric_contribution(
                            gradient_mae_value,
                            target_name=key,
                            block_key=block_key,
                            gradient_name=gradient_name,
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
            for key in sorted(self.information.keys()):
                sae = torch.tensor(self.information[key][0], device=device)
                n_elems = torch.tensor(self.information[key][1], device=device)
                torch.distributed.all_reduce(sae)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sae.item(), n_elems.item())

        finalized_info = {}
        omitted_metrics = []
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} MAE"
            else:
                out_key = f"{key} MAE (per atom)"
            if value[1] == 0:
                omitted_metrics.append(out_key)
                continue
            finalized_info[out_key] = value[0] / value[1]

        _warn_zero_support_metrics(omitted_metrics)

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
