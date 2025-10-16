import copy
from typing import Dict, List, Optional, Tuple

import torch.distributed
from metatensor.torch import TensorMap


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
                    rmse_value = (
                        ((prediction_block.values - target_block.values) ** 2)
                        .sum()
                        .item()
                    )
                else:
                    mask_block = mask.block(block_key)
                    rmse_value = (
                        (
                            (
                                prediction_block.values[mask_block.values]
                                - target_block.values[mask_block.values]
                            )
                            ** 2
                        )
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + rmse_value,
                    self.information[key_to_write][1] + prediction_block.values.numel(),
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
                        gradient_rmse_value = (
                            ((prediction_gradient.values - target_gradient.values) ** 2)
                            .sum()
                            .item()
                        )
                    else:
                        mask_gradient = mask_block.gradient(gradient_name)
                        gradient_rmse_value = (
                            (
                                (
                                    prediction_gradient.values[mask_gradient.values]
                                    - target_gradient.values[mask_gradient.values]
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
                        + prediction_gradient.values.numel(),
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
            for key, value in self.information.items():
                sse = torch.tensor(value[0]).to(device)
                n_elems = torch.tensor(value[1]).to(device)
                torch.distributed.all_reduce(sse)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sse.item(), n_elems.item())  # type: ignore

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
                    mae_value = (
                        (prediction_block.values - target_block.values)
                        .abs()
                        .sum()
                        .item()
                    )
                else:
                    mask_block = mask.block(block_key)
                    mae_value = (
                        (
                            prediction_block.values[mask_block.values]
                            - target_block.values[mask_block.values]
                        )
                        .abs()
                        .sum()
                        .item()
                    )

                self.information[key_to_write] = (
                    self.information[key_to_write][0] + mae_value,
                    self.information[key_to_write][1] + prediction_block.values.numel(),
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
                        gradient_mae_value = (
                            (prediction_gradient.values - target_gradient.values)
                            .abs()
                            .sum()
                            .item()
                        )
                    else:
                        mask_gradient = mask_block.gradient(gradient_name)
                        gradient_mae_value = (
                            (
                                prediction_gradient.values[mask_gradient.values]
                                - target_gradient.values[mask_gradient.values]
                            )
                            .abs()
                            .sum()
                            .item()
                        )

                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + gradient_mae_value,
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][1]
                        + prediction_gradient.values.numel(),
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
            for key, value in self.information.items():
                sae = torch.tensor(value[0]).to(device)
                n_elems = torch.tensor(value[1]).to(device)
                torch.distributed.all_reduce(sae)
                torch.distributed.all_reduce(n_elems)
                self.information[key] = (sae.item(), n_elems.item())  # type: ignore

        finalized_info = {}
        for key, value in self.information.items():
            if any([s in key for s in not_per_atom]):
                out_key = f"{key} MAE"
            else:
                out_key = f"{key} MAE (per atom)"
            finalized_info[out_key] = value[0] / value[1]

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
