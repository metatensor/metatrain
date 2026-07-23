import logging
import warnings
from typing import Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
    sparsify_atomic_basis_target,
)
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from . import checkpoints
from ._base_scaler import BaseScaler
from .documentation import FixedScalerWeights, ModelHypers
from .utils.samples import get_samples_labels


class Scaler(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": ["Scaler: per species, target and property scaling."]
        }
    )

    outputs: Dict[str, ModelOutput]
    atomic_types: List[int]
    target_infos: Dict[str, TargetInfo]
    new_outputs: List[str]
    # model: BaseScaler

    @staticmethod
    def requested_neighbor_lists() -> List[NeighborListOptions]:
        return []

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.atomic_types = sorted(dataset_info.atomic_types)
        self.densify_atomic_basis = self.hypers.get("densify_atomic_basis", True)

        if self.densify_atomic_basis:
            dataset_info = densify_atomic_basis_dataset_info(dataset_info)

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
        for target_name, target_info in self.target_infos.items():
            self.new_outputs.append(target_name)
            self._add_output(target_name, target_info)

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        batch_size: int,
        is_distributed: bool,
        fixed_weights: Optional[FixedScalerWeights] = None,
    ) -> None:
        warnings.warn(
            "train_model is deprecated, use Trainer.train() instead.",
            FutureWarning,
            stacklevel=2,
        )
        from . import train_or_load_scaler

        if not isinstance(datasets, list):
            datasets = [datasets]

        train_or_load_scaler(
            scaler=self,
            fixed_weights=fixed_weights if fixed_weights is not None else {},
            train_datasets=datasets,
            additive_models=additive_models,
            batch_size=batch_size,
            is_distributed=is_distributed,
        )

    def restart(self, dataset_info: DatasetInfo) -> "Scaler":
        """
        Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :return: The restarted Scaler.
        """

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)

        dense_new_targets = densify_atomic_basis_dataset_info(
            DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=merged_info.atomic_types,
                targets=dataset_info.targets,
            )
        ).targets

        self.target_infos = {
            target_name: dense_new_targets[target_name]
            for target_name in merged_info.targets
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        # register new outputs
        self.new_outputs = []
        buffer_names = [n for n, _ in self.named_buffers()]
        for target_name, target_info in self.target_infos.items():
            if target_name + "_scaler_buffer" in buffer_names:
                continue
            self.new_outputs.append(target_name)
            self.model.add_output(target_name, target_info.layout)
            self._add_output(target_name, target_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Get the scales for the given outputs.

        It also sparsifies atomic basis targets in evaluation mode.

        :param systems: List of systems for which to provide the scales.
        :param outputs: Dictionary containing the requested outputs.
        :param selected_atoms: Optional labels for selected atoms.
        :returns: A dictionary with the scales.
        """

        scales = self.get_scales(
            systems,
            outputs,
            selected_atoms=selected_atoms,
        )

        if not self.training and self.densify_atomic_basis:
            # For atomic basis targets, sparsify to create blocks with "atom_type"
            # in the key dimensions, and ensure properties are unpadded. In training
            # mode, predictions stay dense: remove_additive subtracts them from
            # transform-densified targets.
            targets = self.dataset_info.targets
            for k, v in scales.items():
                if k in targets and targets[k].is_atomic_basis:
                    scales[k] = sparsify_atomic_basis_target(
                        systems,
                        v,
                        targets[k].layout,
                    )

        return scales

    def apply_scales(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool = False,
        selected_atoms: Optional[Labels] = None,
        use_per_target_scales: bool = True,
        use_per_property_scales: bool = True,
    ) -> Dict[str, TensorMap]:
        """Scales the outputs based on the stored standard deviations.

        :param systems: List of systems for which the outputs were computed.
        :param outputs: Dictionary containing the output TensorMaps.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
        :param selected_atoms: Optional labels for selected atoms.
        :param use_per_target_scales: If True, applies/removes per-target scales.
        :param use_per_property_scales: If True, applies/removes per-block, per-property
            scales. This only applies to targets with multiple blocks or multiple
            properties. When combined with `use_per_target_scales`, this is equivalent
            to applying/removing the full scales.
        :returns: A dictionary with the scaled outputs.

        :raises ValueError: If no scales have been computed or if `outputs` keys
            contain unsupported keys.
        """
        if len(outputs) == 0:
            return {}

        device = list(outputs.values())[0][0].values.device
        dtype = list(outputs.values())[0][0].values.dtype

        self.scales_to(device, dtype)

        scaled_outputs = self.model.forward(
            systems,
            outputs,
            remove,
            use_per_target_scales,
            use_per_property_scales,
            selected_atoms,
        )

        return scaled_outputs

    def get_scales(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
        use_per_target_scales: bool = True,
        use_per_property_scales: bool = True,
    ) -> Dict[str, TensorMap]:
        """Get the scales for the given outputs.

        :param systems: List of systems for which to provide the scales.
        :param outputs: Dictionary containing the requested outputs.
        :param selected_atoms: Optional labels for selected atoms.
        :param use_per_target_scales: If True, get per-target scales.
        :param use_per_property_scales: If True, get per-block, per-property
            scales. This only applies to targets with multiple blocks or multiple
            properties. When combined with `use_per_target_scales`, this is equivalent
            to getting the full scales.
        :returns: A dictionary with the scales.
        """
        ones: dict[str, TensorMap] = {}
        device = systems[0].positions.device
        dtype = systems[0].positions.dtype

        sample_labels = get_samples_labels(
            systems,
            selected_atoms=selected_atoms,
        )

        for k, output in outputs.items():
            if k not in self.target_infos:
                raise ValueError(
                    f"Output key {k} is not in the dataset info targets. "
                    f"Available targets: {list(self.dataset_info.targets.keys())}"
                )

            target = self.target_infos[k]

            samples = sample_labels[output.sample_kind]

            # Create a TensorMap full of ones with the corresponing structure
            ones[k] = TensorMap(
                keys=target.layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.ones(
                            (samples.values.shape[0],) + block.values.shape[1:],
                            device=device,
                            dtype=dtype,
                        ),
                        samples=samples,
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in target.layout.blocks()
                ],
            )

        scales = self.apply_scales(
            systems,
            ones,
            remove=False,
            selected_atoms=selected_atoms,
            use_per_target_scales=use_per_target_scales,
            use_per_property_scales=use_per_property_scales,
        )

        return scales

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            sample_kind="atom",
            description=target_info.description,
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

        fake_scales_per_target = TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
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
            target_name + "_per_target_scaler_buffer",
            mts.save_buffer(mts.make_contiguous(fake_scales_per_target)),
        )

        fake_scales_per_property = TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
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
            target_name + "_per_property_scaler_buffer",
            mts.save_buffer(mts.make_contiguous(fake_scales_per_property)),
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
        if len(self.model.per_target_scales) != 0:
            if (
                self.model.per_target_scales[
                    list(self.model.per_target_scales.keys())[0]
                ].device
                != device
            ):
                self.model.per_target_scales = {
                    k: v.to(device) for k, v in self.model.per_target_scales.items()
                }
            if (
                self.model.per_target_scales[
                    list(self.model.per_target_scales.keys())[0]
                ].dtype
                != dtype
            ):
                self.model.per_target_scales = {
                    k: v.to(dtype) for k, v in self.model.per_target_scales.items()
                }
        if len(self.model.per_property_scales) != 0:
            if (
                self.model.per_property_scales[
                    list(self.model.per_property_scales.keys())[0]
                ].device
                != device
            ):
                self.model.per_property_scales = {
                    k: v.to(device) for k, v in self.model.per_property_scales.items()
                }
            if (
                self.model.per_property_scales[
                    list(self.model.per_property_scales.keys())[0]
                ].dtype
                != dtype
            ):
                self.model.per_property_scales = {
                    k: v.to(dtype) for k, v in self.model.per_property_scales.items()
                }

        self.model._sync_device_dtype(device, dtype)

    def sync_tensor_maps(self) -> None:
        # Reload the scales of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.model.scales[k] = mts.load_buffer(
                self.__getattr__(k + "_scaler_buffer")
            )
            self.model.per_target_scales[k] = mts.load_buffer(
                self.__getattr__(k + "_per_target_scaler_buffer")
            )

            self.model.per_property_scales[k] = mts.load_buffer(
                self.__getattr__(k + "_per_property_scaler_buffer")
            )

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        checkpoint = {
            "architecture_name": "scaler",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": model_state_dict,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict,
        context: Literal["restart", "finetune", "export"],
    ) -> "Scaler":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint.get('epoch')}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint.get('best_epoch')}")
            model_state_dict = checkpoint.get(
                "best_model_state_dict", checkpoint["model_state_dict"]
            )
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model_data = checkpoint["model_data"]
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )

        dtype = model_state_dict["dummy_buffer"].dtype
        model.to(dtype)

        model.load_state_dict(model_state_dict)
        model.sync_tensor_maps()

        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint.get("model_ckpt_version") == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        version = checkpoint.get("model_ckpt_version")
        if version != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {version}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = self.dummy_buffer.dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for scaler")

        self.to(dtype)
        self.scales_to(torch.device("cpu"), torch.float64)

        interaction_range = 0.0

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)
