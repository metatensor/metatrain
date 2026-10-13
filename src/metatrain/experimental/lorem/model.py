import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

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

from metatrain.utils.abc import ModelInterface, common_upgrade_checkpoint
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.hypers import raise_if_hypers_mismatch
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers
from .modules.bec import BecPredictor, apply_acoustic_sum_rule
from .modules.lorem import LongRange, ShortRange


def _tensor_map(values: torch.Tensor, samples: Labels, layout: TensorMap) -> TensorMap:
    """Build one output block from raw values and the target's layout."""
    block = layout.block(0)
    return TensorMap(
        keys=layout.keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=samples,
                components=block.components,
                properties=block.properties,
            )
        ],
    )


class LOREM(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 2
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "lorem-jax: https://github.com/lab-cosmo/lorem-jax",
                "torch-pme: https://github.com/lab-cosmo/torch-pme",
            ],
            "architecture": [
                "LOREM: https://arxiv.org/abs/2507.19382",
            ],
        }
    )

    layouts: Dict[str, TensorMap]
    bec_targets: List[str]

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)
        self.atomic_types = dataset_info.atomic_types
        self.has_new_targets = False

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )
        self.num_features = int(self.hypers["num_features"])
        self.max_degree_lr = int(self.hypers["max_degree_lr"])
        max_degree = int(self.hypers["max_degree"])
        if self.max_degree_lr > max_degree:
            raise ValueError(
                f"max_degree_lr ({self.max_degree_lr}) cannot exceed "
                f"max_degree ({max_degree})."
            )
        num_spherical = int(self.hypers["num_spherical_features"])
        # Ewald width follows the cutoff, the same way lorem-jax derives it
        # when it prepares a dataset: smearing = cutoff / 4, wavelength = cutoff / 8.
        cutoff = float(self.hypers["cutoff"])
        self.sr = ShortRange(
            cutoff=cutoff,
            max_degree=max_degree,
            num_features=self.num_features,
            num_radial=int(self.hypers["num_radial"]),
            num_spherical_features=num_spherical,
            num_species=int(self.hypers["num_species"]),
            atomic_types=self.atomic_types,
            neighbor_list_options=self.requested_nl,
            num_message_passing=int(self.hypers["num_message_passing"]),
            equivariant_message_passing=bool(
                self.hypers["equivariant_message_passing"]
            ),
            initialize_node_features=bool(self.hypers["initialize_node_features"]),
        )
        self.lr = LongRange(
            feature_dim=self.num_features,
            num_spherical_features=num_spherical,
            max_degree=max_degree,
            max_degree_lr=self.max_degree_lr,
            neighbor_list_options=self.requested_nl,
            smearing=cutoff / 4.0,
            kspace_resolution=cutoff / 8.0,
        )

        self.outputs: Dict[str, ModelOutput] = {}
        self.bec_heads = torch.nn.ModuleDict({})
        self.bec_targets = []
        self.layouts: Dict[str, TensorMap] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        n_properties = len(target.layout.block().properties)
        # The rank-2 head is a Born effective charge: the acoustic sum rule
        # is part of the prediction. Another per-atom 3x3 tensor (an NMR
        # shielding, for example) has the same layout and must not be
        # accepted just because of that.
        is_bec = (
            target.is_cartesian
            and len(target.layout.block().components) == 2
            and target.sample_kind == "atom"
            and target.quantity == "bec"
        )
        if target.is_scalar:
            if n_properties != 1:
                raise ValueError(
                    "LOREM's energy head has one output. "
                    f"Target '{target_name}' has {n_properties} properties."
                )
            # One shared energy head. A second scalar target would be a copy
            # of the first, including across a restart.
            existing = [name for name in self.outputs if name not in self.bec_targets]
            if existing:
                raise ValueError(
                    "LOREM has one energy head and accepts a single scalar "
                    f"target. '{existing[0]}' is already registered; refusing "
                    f"'{target_name}'."
                )
        elif is_bec:
            if n_properties != 1:
                raise ValueError(
                    "LOREM's Born-charge head has one output. "
                    f"Target '{target_name}' has {n_properties} properties."
                )
            if int(self.hypers["max_degree"]) < 2:
                raise ValueError(
                    "Cartesian rank-2 targets (Born effective charges) require "
                    f"max_degree >= 2, got {self.hypers['max_degree']}."
                )
        else:
            raise ValueError(
                "LOREM predicts one scalar target and per-atom Born effective "
                "charges (Cartesian rank 2, quantity 'bec', one property). "
                f"Unsupported target '{target_name}'."
            )

        self.layouts[target_name] = target.layout
        self.outputs[target_name] = ModelOutput(
            unit=target.unit,
            sample_kind="atom",
            description=target.description,
        )
        if is_bec:
            self.bec_targets.append(target_name)
            self.bec_heads[target_name] = BecPredictor(
                n_stages=1 + int(self.hypers["num_message_passing"]),
                in_features=int(self.hypers["num_spherical_features"]),
                hidden_features=self.num_features,
                in_max_degree=int(self.hypers["max_degree"]),
            )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}
        for name in outputs:
            if name not in self.outputs:
                raise ValueError(
                    f"Output {name} is not supported by this LOREM model. "
                    f"Supported outputs are: {list(self.outputs.keys())}"
                )

        device = systems[0].positions.device
        self.layouts = {
            name: layout.to(device) for name, layout in self.layouts.items()
        }

        (
            _nodes_scalar,
            neighbor_distances,
            _nodes_spherical,
            sr_energy,
            snapshots,
        ) = self.sr(systems)
        lr_energy, spherical_updates = self.lr(
            systems, _nodes_scalar, neighbor_distances, _nodes_spherical
        )
        atomic_energy = (sr_energy + lr_energy).unsqueeze(-1)

        system_sizes = [len(system) for system in systems]
        system_sizes_tensor = torch.tensor(system_sizes, device=device)
        n_atoms = int(sum(system_sizes))
        system_indices = torch.repeat_interleave(
            torch.arange(len(systems), device=device),
            system_sizes_tensor,
            output_size=n_atoms,
        )
        first_atoms = torch.cumsum(system_sizes_tensor, 0) - system_sizes_tensor
        atom_indices = (
            torch.arange(n_atoms, device=device) - first_atoms[system_indices]
        )
        sample_values = torch.stack([system_indices, atom_indices], dim=1)
        samples = Labels(names=["system", "atom"], values=sample_values.to(torch.int32))

        return_dict: Dict[str, TensorMap] = {}
        for bec_name, bec_head in self.bec_heads.items():
            if bec_name in outputs:
                apt = apply_acoustic_sum_rule(
                    bec_head(snapshots, spherical_updates), system_sizes
                )
                atomic_property = _tensor_map(
                    apt.unsqueeze(-1), samples, self.layouts[bec_name]
                )
                if selected_atoms is not None:
                    atomic_property = mts.slice(
                        atomic_property, axis="samples", selection=selected_atoms
                    )
                if outputs[bec_name].sample_kind == "atom":
                    return_dict[bec_name] = atomic_property
                else:
                    return_dict[bec_name] = sum_over_atoms(atomic_property)

        for target_name, output in outputs.items():
            if target_name in self.bec_targets:
                continue
            atomic_property = _tensor_map(
                atomic_energy, samples, self.layouts[target_name]
            )
            if selected_atoms is not None:
                atomic_property = mts.slice(
                    atomic_property, axis="samples", selection=selected_atoms
                )
            if output.sample_kind == "atom":
                return_dict[target_name] = atomic_property
            else:
                return_dict[target_name] = sum_over_atoms(atomic_property)

        return return_dict

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> "LOREM":
        if model_hypers is not None:
            raise_if_hypers_mismatch(
                self.hypers,
                model_hypers,
                default_hypers=get_default_hypers("experimental.lorem")["model"],
            )

        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The LOREM model does not support adding new atomic types."
            )

        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info
        return self

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LOREM":
        model_data = checkpoint["model_data"]

        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        # A freshly built model is float32. The checkpoint carries the
        # training dtype; taking it from the new model downcasts float64.
        dtype = next(
            tensor.dtype
            for tensor in model_state_dict.values()
            if torch.is_tensor(tensor) and tensor.is_floating_point()
        )
        model.to(dtype).load_state_dict(model_state_dict)

        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.metadata = merge_metadata(model.metadata, metadata)

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for LOREM")

        self.to(dtype)

        # Long-range Coulomb interactions are always on, so the interaction
        # range is unbounded.
        interaction_range = torch.inf

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict[str, Any], version: Optional[int] = None
    ) -> dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "model", checkpoints
        )

    def get_checkpoint(
        self, best_model_state_dict: Optional[dict[str, Any]] = None
    ) -> Dict:
        model_state_dict = self.state_dict()
        return {
            "architecture_name": "experimental.lorem",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": dict(self.hypers),
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": best_model_state_dict or model_state_dict,
        }
