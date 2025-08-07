import logging
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

from ..data import DatasetInfo, TargetInfo
from ..jsonschema import validate
from ..transfer import batch_to
from ._base_composition import BaseCompositionModel, _include_key
from .remove import remove_additive


class CompositionModel(torch.nn.Module):
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

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target quantity "
                    f"{target_info.quantity}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.target_infos = {
            target_name: target_info
            for target_name, target_info in dataset_info.targets.items()
        }

        # Initialize the composition model
        self.model = BaseCompositionModel(
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
        no need to shuffle or drop the last non-full batch. Distributed sampling is also
        not used, and training with double precision is enforced.
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
        fixed_weights: Optional[Dict[str, Dict[int, float]]] = None,
        is_distributed: bool = False,
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

        if fixed_weights is None:
            fixed_weights = {}

        device = self.dummy_buffer.device

        # accumulate
        for batch in dataloader:
            systems, targets, _ = batch
            systems, targets, _ = batch_to(systems, targets, device=device)
            # only accumulate the targets that do not use fixed weights
            targets = {
                target_name: targets[target_name]
                for target_name in self.target_infos.keys()
                if target_name not in fixed_weights
            }
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
            self.model.accumulate(systems, targets)

        if is_distributed:
            torch.distributed.barrier()
            # All-reduce the accumulated TensorMaps across all processes
            for target_name in self.model.XTX.keys():
                for XTX_block, XTY_block in zip(
                    self.model.XTX[target_name],
                    self.model.XTY[target_name],
                    strict=True,
                ):
                    torch.distributed.all_reduce(XTX_block.values)
                    torch.distributed.all_reduce(XTY_block.values)

        # Fit the model on all ranks
        self.model.fit(fixed_weights)

        # update the buffer weights now they are fitted
        for target_name in self.model.weights.keys():
            self.register_buffer(
                target_name + "_composition_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        self.model.weights[target_name].to("cpu", torch.float64)
                    )
                ).to(device),
            )

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """
        Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The composition model does not support adding new atomic types."
            )

        self.target_infos = {
            target_name: target_info
            for target_name, target_info in merged_info.targets.items()
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        # register new outputs
        for target_name, target_info in self.target_infos.items():
            self.model.add_output(target_name, target_info.layout)
            self._add_output(target_name, target_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of samples for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device

        self.weights_to(device, dtype)

        for output_name in outputs.keys():
            if output_name not in self.outputs:
                raise ValueError(
                    f"Output {output_name} is not supported by the "
                    "composition model. Supported outputs are: "
                    f"{list(self.outputs.keys())}"
                )

        pred = self.model.forward(
            systems,
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return pred

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        # Create a fake weights buffer for the target, filtering the blocks that will
        # not be fitted
        layout = mts.filter_blocks(
            target_info.layout,
            Labels(
                target_info.layout.keys.names,
                torch.vstack(
                    [key.values for key in target_info.layout.keys if _include_key(key)]
                ),
            ),
        )

        fake_weights = TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        (len(self.atomic_types),) + b.values.shape[1:],
                        dtype=torch.float64,
                    ),
                    samples=Labels(
                        names=["center_type"],
                        values=torch.tensor(self.atomic_types, dtype=torch.int).reshape(
                            -1, 1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                )
                for b in layout.blocks()
            ],
        )
        self.register_buffer(
            target_name + "_composition_buffer",
            mts.save_buffer(mts.make_contiguous(fake_weights)),
        )

    def weights_to(self, device: torch.device, dtype: torch.dtype):
        if len(self.model.weights) != 0:
            if self.model.weights[list(self.model.weights.keys())[0]].device != device:
                self.model.weights = {
                    k: v.to(device) for k, v in self.model.weights.items()
                }
            if self.model.weights[list(self.model.weights.keys())[0]].dtype != dtype:
                self.model.weights = {
                    k: v.to(dtype) for k, v in self.model.weights.items()
                }

        self.model._sync_device_dtype(device, dtype)

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with a composition model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        # only scalars can have composition contributions
        if not target_info.is_scalar and not target_info.is_spherical:
            logging.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False
        if (
            target_info.is_spherical
            and len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0
        ):
            logging.debug(
                f"Composition model does not support spherical target {target_name} "
                "since it does not have any invariant blocks."
            )
            return False
        return True

    def sync_tensor_maps(self):
        # Reload the weights of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.model.weights[k] = mts.load_buffer(
                self.__getattr__(k + "_composition_buffer")
            )
