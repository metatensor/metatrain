import copy
from typing import Dict, List, Tuple

import torch.distributed
from metatensor.torch import TensorMap


class RMSEAccumulator:
    """Accumulates the RMSE between predictions and targets for an arbitrary
    number of keys, each corresponding to one target.

    :param separate_blocks: if true, the RMSE will be computed separately for each
        block in the target and prediction ``TensorMap`` objects.
    """

    def __init__(self, separate_blocks: bool = False) -> None:
        """Initialize the accumulator."""
        self.information: Dict[str, Tuple[float, int]] = {}
        self.separate_blocks = separate_blocks

    def update(self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]):
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.

        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        """

        for key, target in targets.items():
            prediction = predictions[key]
            for block_key in target.keys:
                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += "("
                    for name, value in zip(block_key.names, block_key.values):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                self.information[key_to_write] = (
                    self.information[key_to_write][0]
                    + ((prediction_block.values - target_block.values) ** 2)
                    .sum()
                    .item(),
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
                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + ((prediction_gradient.values - target_gradient.values) ** 2)
                        .sum()
                        .item(),
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

    def __init__(self, separate_blocks: bool = False) -> None:
        """Initialize the accumulator."""
        self.information: Dict[str, Tuple[float, int]] = {}
        self.separate_blocks = separate_blocks

    def update(self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]):
        """Updates the accumulator with new predictions and targets.

        :param predictions: A dictionary of predictions, where the keys correspond
            to the keys in the targets dictionary, and the values are the predictions.

        :param targets: A dictionary of targets, where the keys correspond to the keys
            in the predictions dictionary, and the values are the targets.
        """

        for key, target in targets.items():
            prediction = predictions[key]
            for block_key in target.keys:
                target_block = target.block(block_key)
                prediction_block = prediction.block(block_key)

                key_to_write = copy.deepcopy(key)
                if self.separate_blocks:
                    key_to_write += "("
                    for name, value in zip(block_key.names, block_key.values):
                        key_to_write += f"{name}={int(value)},"
                    key_to_write = key_to_write[:-1]
                    key_to_write += ")"

                if key_to_write not in self.information:  # create key if not present
                    self.information[key_to_write] = (0.0, 0)

                self.information[key_to_write] = (
                    self.information[key_to_write][0]
                    + (prediction_block.values - target_block.values)
                    .abs()
                    .sum()
                    .item(),
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
                    self.information[f"{key_to_write}_{gradient_name}_gradients"] = (
                        self.information[f"{key_to_write}_{gradient_name}_gradients"][0]
                        + (prediction_gradient.values - target_gradient.values)
                        .abs()
                        .sum()
                        .item(),
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
