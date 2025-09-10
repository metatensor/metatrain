from typing import Dict, List, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
)

from ._base_scaler import BaseScaler
from .additive.remove import remove_additive
from .data import DatasetInfo, TargetInfo
from .jsonschema import validate
from .transfer import batch_to


class Scaler(torch.nn.Module):
    """
    A simple model that calculates the per-species contributions to targets
    based on the stoichiometry in a system.

    :param hypers: A dictionary of model hyperparameters. This parameter is ignored and
        is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    # Needed for torchscript compatibility
    outputs: Dict[str, ModelOutput]

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `hypers` should be an empty dictionary
        validate(
            instance=hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)
        self.target_infos = {
            target_name: target_info
            for target_name, target_info in dataset_info.targets.items()
        }

        # Initialize the scaler model
        self.model = BaseScaler(
            atomic_types=self.atomic_types,
            layouts={
                target_name: target_info.layout
                for target_name, target_info in self.target_infos.items()
            },
        )
        self.outputs: Dict[str, ModelOutput] = {}

        # keeps track of dtype and device of the composition model
        self.register_buffer("dummy_buffer", torch.randn(1))

        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

    def _get_dataloader(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> DataLoader:
        """
        Create a DataLoader for the provided datasets. As the dataloader is only used to
        accumulate the quanitites needed for fitting the composition weights, there is
        no need to shuffle or drop the last non-full batch. Distributed sampling can be
        used or not, based on the `is_distributed` argument, and training with double
        precision is enforced.
        """
        # Create the collate function
        targets_keys = list(self.dataset_info.targets.keys())
        collate_fn = CollateFn(target_keys=targets_keys)

        dtype = datasets[0][0]["system"].positions.dtype
        if dtype != torch.float64:
            raise ValueError(
                "The composition model only supports float64 during training. "
                f"Got dtype: {dtype}."
            )

        # Build the dataloaders
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            samplers = [
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for dataset in datasets
            ]
        else:
            samplers = [None] * len(datasets)

        dataloaders = []
        for dataset, sampler in zip(datasets, samplers):
            if len(dataset) < batch_size:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(dataset)}) than the batch size "
                    f"({batch_size}). "
                    "Please reduce the batch size."
                )
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=None if sampler else False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )

        return CombinedDataLoader(dataloaders, shuffle=False)

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        batch_size: int,
        is_distributed: bool,
    ) -> None:
        """
        Train the composition model on the provided training data in the ``datasets``.

        Assumes the systems are stored in the ``system`` attribute of each sample, with
        targets expected to be stored as well, with keys corresponding to the target
        names defined in the dataset info.

        Any additive contributions from the provided ``additive_models`` will be removed
        from the targets before training. The `fixed_weights` argument can be used to
        specify which targets should be treated as fixed weights during training.
        """

        if not isinstance(datasets, list):
            datasets = [datasets]

        if len(self.target_infos) == 0:  # no (new) targets to fit
            return

        # Create dataloader for the training datasets
        dataloader = self._get_dataloader(
            datasets, batch_size, is_distributed=is_distributed
        )

        device = self.dummy_buffer.device

        # accumulate
        for batch in dataloader:
            systems, targets, extra_data = batch
            systems, targets, extra_data = batch_to(
                systems, targets, extra_data, device=device
            )
            if len(targets) == 0:
                break

            # remove additive contributions from these targets
            for additive_model in additive_models:
                targets = remove_additive(
                    systems,
                    targets,
                    additive_model,
                    {
                        target_name: self.target_infos[target_name]
                        for target_name in targets
                    },
                )
            self.model.accumulate(systems, targets, extra_data)

        if is_distributed:
            torch.distributed.barrier()
            # All-reduce the accumulated TensorMaps across all processes
            for target_name in self.N.keys():
                for atomic_type in self.atomic_types:
                    for N_block, Y_block, Y2_block in zip(
                        self.model.N[target_name][atomic_type],
                        self.model.Y[target_name][atomic_type],
                        self.model.Y2[target_name][atomic_type],
                        strict=True,
                    ):
                        torch.distributed.all_reduce(N_block.values)
                        torch.distributed.all_reduce(Y_block.values)
                        torch.distributed.all_reduce(Y2_block.values)

        # Compute the scales on all ranks
        self.model.fit()

        # update the buffer weights now they are fitted
        for target_name in self.model.scales.keys():
            for atomic_type in self.atomic_types:
                self.register_buffer(
                    target_name + f"_{atomic_type}" + "_scaler_buffer",
                    mts.save_buffer(
                        mts.make_contiguous(
                            self.model.scales[target_name][atomic_type].to(
                                "cpu", torch.float64
                            )
                        )
                    ).to(device),
                )

    def restart(self, dataset_info: DatasetInfo) -> "Scaler":
        """
        Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)

        self.target_infos = {
            target_name: target_info
            for target_name, target_info in merged_info.targets.items()
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        # register new outputs
        for target_name, target_info in self.target_infos.items():
            self._add_output(target_name, target_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool,
    ) -> Dict[str, TensorMap]:
        """Scales the outputs based on the stored standard deviations.

        :param outputs: Dictionary containing the output TensorMaps.
        :returns: A dictionary with the scaled outputs.

        :raises ValueError: If no scales have been computed or if `outputs` keys
            contain unsupported keys.
        """
        device = list(outputs.values())[0][0].values.device
        dtype = list(outputs.values())[0][0].values.dtype

        self.scales_to(device, dtype)

        scaled_outputs = self.model.forward(systems, outputs, remove)

        return scaled_outputs

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        layout = target_info.layout

        for atomic_type in self.atomic_types:
            fake_scales = TensorMap(
                keys=layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            1,
                            *[len(comp) for comp in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            names=["sample"],
                            values=torch.tensor([[0]], dtype=torch.int),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in layout.blocks()
                ],
            )
            self.register_buffer(
                target_name + f"_{atomic_type}" + "_scaler_buffer",
                mts.save_buffer(mts.make_contiguous(fake_scales)),
            )

    def scales_to(self, device: torch.device, dtype: torch.dtype):
        if len(self.model.scales) != 0:
            if (
                self.model.scales[list(self.model.scales.keys())[0]][
                    self.atomic_types[0]
                ].device
                != device
            ):
                scales: Dict[str, Dict[int, TensorMap]] = {}
                for target_name in self.target_infos.keys():
                    scales[target_name] = {}
                    for atomic_type in self.atomic_types:
                        scales[target_name][atomic_type] = self.model.scales[
                            target_name
                        ][atomic_type].to(device)
                self.model.scales = scales

            if (
                self.model.scales[list(self.model.scales.keys())[0]][
                    self.atomic_types[0]
                ].dtype
                != dtype
            ):
                scales = {}
                for target_name in self.target_infos.keys():
                    scales[target_name] = {}
                    for atomic_type in self.atomic_types:
                        scales[target_name][atomic_type] = self.model.scales[
                            target_name
                        ][atomic_type].to(dtype)
                self.model.scales = scales

        self.model._sync_device_dtype(device, dtype)

    def sync_tensor_maps(self):
        # Reload the scales of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            for atomic_type in self.atomic_types:
                self.model.scales[k][atomic_type] = mts.load_buffer(
                    self.__getattr__(k + f"_{atomic_type}" + "_scaler_buffer")
                )

    def get_scales_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Return a dictionary mapping each target (and atomic type, block, property) to
        its scale.
        """
        scales_dict: Dict[str, Dict[str, float]] = {}
        for target_name, atomic_scales in self.model.scales.items():
            scales_dict[target_name] = {}
            for atomic_type, tensor_map in atomic_scales.items():
                for block_i, block in enumerate(tensor_map.blocks()):
                    for prop_i, _ in enumerate(block.properties.values):
                        key = f"type_{atomic_type}_block_{block_i}_prop_{prop_i}"
                        scale_value = float(block.values.flatten()[prop_i].item())
                        scales_dict[target_name][key] = scale_value
        return scales_dict


# ===== Remove scale =====


def remove_scale(
    systems: List[System],
    targets: Dict[str, TensorMap],
    scaler: Scaler,
):
    """
    Scale all targets to a standard deviation of one.

    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    """
    return scaler(systems, targets, remove=True)
