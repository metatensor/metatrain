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
from ._base_composition import (
    BaseCompositionModel,
    FixedCompositionWeights,
    _include_key,
)
from .documentation import ModelHypers


class CompositionModel(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "Composition model: per-species linear fit to training targets"
            ]
        }
    )

    outputs: Dict[str, ModelOutput]
    atomic_types: List[int]
    target_infos: Dict[str, TargetInfo]
    _new_outputs: List[str]

    @staticmethod
    def requested_neighbor_lists() -> List[NeighborListOptions]:
        return []

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        if not (isinstance(hypers, dict) and len(hypers) == 0):
            raise ValueError(
                f"{self.__class__.__name__} hypers takes an empty dictionary. "
                f"Got: {hypers}."
            )

        self.atomic_types = sorted(dataset_info.atomic_types)

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target quantity "
                    f"{target_info.quantity}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # The composition model always fits and stores dense weights (a single
        # block per o3_lambda/o3_sigma, with properties padded to the union
        # across atomic types), regardless of whether the target's native
        # layout is sparse (atom_type as a key dimension). This keeps
        # checkpoints portable across every architecture: standalone training
        # and training embedded in e.g. PET both produce the same layout.
        dense_dataset_info = densify_atomic_basis_dataset_info(dataset_info)
        self.target_infos = {
            target_name: target_info
            for target_name, target_info in dense_dataset_info.targets.items()
        }

        self.model: BaseCompositionModel = BaseCompositionModel(
            atomic_types=self.atomic_types,
            layouts={
                target_name: target_info.layout
                for target_name, target_info in self.target_infos.items()
            },
        )

        self.outputs: Dict[str, ModelOutput] = {}

        self.register_buffer("dummy_buffer", torch.randn(1, dtype=torch.float64))

        self._new_outputs = []
        for target_name in self.dataset_info.targets:
            self._new_outputs.append(target_name)
            self._add_output(target_name, self.target_infos[target_name])

    @classmethod
    def from_valid_targets(
        cls,
        dataset_info: DatasetInfo,
        atomic_types: List[int],
    ) -> "CompositionModel":
        """Build a composition model from the valid targets of ``dataset_info``.

        Targets the composition model cannot handle (see ``is_valid_target``)
        are dropped.

        :param dataset_info: dataset information to build the model from.
        :param atomic_types: atomic types the model should support. Passed
            separately because it can be wider than
            ``dataset_info.atomic_types``, e.g. all the types known to a
            pretrained backbone (MACE).
        :return: a composition model for the valid targets.
        """
        targets = {
            name: info
            for name, info in dataset_info.targets.items()
            if cls.is_valid_target(name, info)
        }
        return cls(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=sorted(atomic_types),
                targets=targets,
                extra_data=dataset_info.extra_data,
            ),
        )

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        batch_size: int,
        is_distributed: bool,
        fixed_weights: Optional[FixedCompositionWeights] = None,
    ) -> None:
        warnings.warn(
            "train_model is deprecated, use Trainer.train() instead.",
            FutureWarning,
            stacklevel=2,
        )
        from .trainer import Trainer as CompositionTrainer

        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(self.target_infos) == 0:
            return

        hypers: dict = {}
        if fixed_weights is not None:
            hypers["atomic_baseline"] = fixed_weights
        hypers["batch_size"] = batch_size
        trainer = CompositionTrainer(hypers=hypers)  # type: ignore[arg-type]
        trainer._additive_models = additive_models
        trainer._is_distributed = is_distributed
        trainer.train(
            model=self,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=datasets,
            val_datasets=datasets,
            checkpoint_dir="",
        )

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        raw_targets = {}
        for target_name in dataset_info.targets:
            target_info = dataset_info.targets[target_name]
            if self.is_valid_target(target_name, target_info):
                raw_targets[target_name] = target_info
            else:
                logging.debug(
                    f"Composition model does not support target "
                    f"'{target_name}', skipping."
                )

        if len(raw_targets) == 0:
            # No new targets to fit: reset so a subsequent Trainer.train() does not
            # refit (and zero out) the already-fitted targets from __init__.
            self.target_infos = {}
            self._new_outputs = []
            return self

        merged_info = self.dataset_info.union(
            DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=dataset_info.atomic_types,
                targets=raw_targets,
            )
        )
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The composition model does not support adding new atomic types."
            )

        dense_new_targets = densify_atomic_basis_dataset_info(
            DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=merged_info.atomic_types,
                targets=raw_targets,
            )
        ).targets
        self.target_infos = {
            target_name: dense_new_targets[target_name]
            for target_name in raw_targets
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        self._new_outputs = []
        buffer_names = [n for n, _ in self.named_buffers()]
        for target_name, target_info in self.target_infos.items():
            if target_name + "_composition_buffer" in buffer_names:
                continue
            self._new_outputs.append(target_name)
            self.model.add_output(target_name, target_info.layout)
            self._add_output(target_name, target_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
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

        if not self.training:
            # For atomic basis targets, sparsify to create blocks with "atom_type"
            # in the key dimensions, and ensure properties are unpadded. In training
            # mode, predictions stay dense: remove_additive subtracts them from
            # transform-densified targets.
            targets = self.dataset_info.targets
            for k, v in pred.items():
                if k in targets and targets[k].is_atomic_basis:
                    pred[k] = sparsify_atomic_basis_target(
                        systems,
                        v,
                        targets[k].layout,
                    )

        return pred

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            sample_kind="atom",
            description=target_info.description,
        )

        layout = mts.filter_blocks(
            target_info.layout,
            Labels(
                target_info.layout.keys.names,
                torch.vstack(
                    [key.values for key in target_info.layout.keys if _include_key(key)]
                ),
                assume_unique=True,
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
                        assume_unique=True,
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

    def remove_output(self, target_name: str) -> None:
        """
        Remove a previously registered output target, mirroring ``_add_output``.

        :param target_name: Name of the target to remove.
        """
        self.outputs.pop(target_name, None)
        self.dataset_info.targets.pop(target_name, None)
        self.model.remove_output(target_name)
        buffer_name = target_name + "_composition_buffer"
        if hasattr(self, buffer_name):
            delattr(self, buffer_name)

    def weights_to(self, device: torch.device, dtype: torch.dtype) -> None:
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
        if not target_info.is_scalar and not target_info.is_spherical:
            logging.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False

        if target_info.is_spherical:
            if "o3_lambda" in target_info.layout.keys.names:
                if len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0:
                    logging.debug(
                        "Composition model does not support spherical "
                        f"target {target_name} since it does not have "
                        "any invariant blocks."
                    )
                    return False
            elif "o3_lambda_2" in target_info.layout.keys.names:
                if not any(
                    key["o3_lambda_1"] == key["o3_lambda_2"]
                    and key["o3_sigma_1"] == key["o3_sigma_2"]
                    for key in target_info.layout.keys
                ):
                    logging.debug(
                        "Composition model does not support spherical "
                        f"target {target_name} since it does not have "
                        "any invariant contribution."
                    )
                    return False

        return True

    def sync_tensor_maps(self) -> None:
        for k in self.dataset_info.targets:
            self.model.weights[k] = mts.load_buffer(
                self.__getattr__(k + "_composition_buffer")
            )

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        checkpoint = {
            "architecture_name": "composition",
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
    ) -> "CompositionModel":
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
            raise ValueError(f"unsupported dtype {dtype} for composition model")

        self.to(dtype)
        self.weights_to(torch.device("cpu"), torch.float64)

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
