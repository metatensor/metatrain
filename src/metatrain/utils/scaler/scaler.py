from typing import Dict, List, Optional, Union

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
from metatrain.utils.per_atom import average_by_num_atoms

from ..additive import remove_additive
from ..data import DatasetInfo, TargetInfo, unpack_batch
from ..jsonschema import validate
from ..transfer import batch_to
from ._base_scaler import BaseScaler


class Scaler(torch.nn.Module):
    """
    Placeholder docs.

    :param hypers: Hyperparameters for the scaler. Should be an empty dictionary.
    :param dataset_info: Information about the dataset used to initialize the scaler.
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

        self.new_outputs = []
        for target_name, target_info in self.dataset_info.targets.items():
            self.new_outputs.append(target_name)
            self._add_output(target_name, target_info)

    def _get_dataloader(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
    ) -> DataLoader:
        """
        Create a DataLoader for the provided datasets. As the dataloader is only used to
        accumulate the quanitites needed for fitting the scales, there is no need to
        shuffle or drop the last non-full batch. Distributed sampling can be used or
        not, based on the `is_distributed` argument, and training with double precision
        is enforced.

        :param datasets: List of datasets to create the dataloader from.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :return: The created DataLoader.
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
        for dataset, sampler in zip(datasets, samplers, strict=False):
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
        fixed_weights: Optional[Dict[str, Union[float, Dict[int, float]]]] = None,
    ) -> None:
        """
        Placeholder docs.

        :param datasets: List of datasets to use for training the scaler.
        :param additive_models: List of additive models to remove from the targets
            before accumulating the quantities needed for fitting the scales.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :param fixed_weights: Optional dict of fixed weights to apply to the scales
            of each target. The keys of the dict are the target names, and the values
            are either a single float value to be applied to all atomic types, or a
            dict mapping atomic type (int) to weight (float). If not provided, all
            scales will be computed based on the accumulated quantities.
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
            systems, targets, extra_data = unpack_batch(batch)
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
            targets = average_by_num_atoms(targets, systems, [])
            self.model.accumulate(systems, targets, extra_data)

        if is_distributed:
            torch.distributed.barrier()
            # All-reduce the accumulated TensorMaps across all processes
            for target_name in self.new_outputs:
                for N_block, Y2_block in zip(
                    self.model.N[target_name],
                    self.model.Y2[target_name],
                    strict=True,
                ):
                    torch.distributed.all_reduce(N_block.values)
                    torch.distributed.all_reduce(Y2_block.values)

        # Compute the scales on all ranks
        self.model.fit(fixed_weights=fixed_weights, targets_to_fit=self.new_outputs)

        # update the buffer scales now they are fitted
        for target_name in self.model.scales.keys():
            self.register_buffer(
                target_name + "_scaler_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        self.model.scales[target_name].to("cpu", torch.float64)
                    )
                ).to(device),
            )

    def restart(self, dataset_info: DatasetInfo) -> "Scaler":
        """
        Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :return: The restarted Scaler.
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
        remove: bool = False,
    ) -> Dict[str, TensorMap]:
        """Scales the outputs based on the stored standard deviations.

        :param systems: List of systems for which the outputs were computed.
        :param outputs: Dictionary containing the output TensorMaps.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
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

        valid_sample_names = [
            ["system"],
            ["system", "atom"],
        ]

        if layout.sample_names == valid_sample_names[0]:
            samples = Labels(["atomic_type"], torch.tensor([[-1]]))

        elif layout.sample_names == valid_sample_names[1]:
            samples = Labels(
                ["atomic_type"], torch.arange(len(self.atomic_types)).reshape(-1, 1)
            )

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        fake_scales = TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(  # important when scale_targets=False
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout.blocks()
            ],
        )
        self.register_buffer(
            target_name + "_scaler_buffer",
            mts.save_buffer(mts.make_contiguous(fake_scales)),
        )

    def scales_to(self, device: torch.device, dtype: torch.dtype) -> None:
        if len(self.model.scales) != 0:
            if self.model.scales[list(self.model.scales.keys())[0]].device != device:
                self.model.scales = {
                    k: v.to(device) for k, v in self.model.scales.items()
                }
            if self.model.scales[list(self.model.scales.keys())[0]].dtype != dtype:
                self.model.scales = {
                    k: v.to(dtype) for k, v in self.model.scales.items()
                }

        self.model._sync_device_dtype(device, dtype)

    def sync_tensor_maps(self) -> None:
        # Reload the scales of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.model.scales[k] = mts.load_buffer(
                self.__getattr__(k + "_scaler_buffer")
            )
