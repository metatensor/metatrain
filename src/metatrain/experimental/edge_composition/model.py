import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

import torch
from elearn.interface.metatensor.couple import couple_tensor_blocks
from featomic.torch.clebsch_gordan._coefficients import calculate_cg_coefficients
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from sphericart.torch import SphericalHarmonics
from torch import Tensor

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from . import checkpoints
from .documentation import ModelHypers
from .radials import Exponential, Tabulated
from .utils.transforms import batch_neighborlist, radial_to_spherical_harmonics


# For each l, norm of the tensor returned by sphericart
SPHERICART_NORMS = torch.tensor(
    [
        0.2821,
        0.4886,
        0.6308,
        0.7464,
        0.8463,
        0.9356,
        1.0171,
        1.0925,
        1.1631,
        1.2296,
        1.2927,
    ]
)


class UnitarySphericalHarmonics(torch.nn.Module):
    """
    Wrapper around sphericart's SphericalHarmonics that returns
    spherical harmonics with norm 1, instead of the default norm
    used by sphericart.
    """

    def __init__(self, max_l: int):
        super().__init__()
        # Initialize the SphericalHarmonics calculator
        self._sh = SphericalHarmonics(max_l)

        # We want spherical harmonics to have norm 1, so we need to
        # compute a correction from the convention used by the
        # calculator.
        factors = torch.ones((max_l + 1) ** 2)
        for l in range(max_l + 1):
            start, end = l**2, l**2 + 2 * l + 1
            factors[start:end] = 1 / SPHERICART_NORMS[l]

        self.register_buffer("_factors", factors)

    def forward(self, x: Tensor) -> Tensor:
        return self._sh.compute(x) * self._factors


class EdgeCompositionModel(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.atom_type_pairs = []
        for type1 in self.dataset_info.atomic_types:
            for type2 in self.dataset_info.atomic_types:
                self.atom_type_pairs.append((type1, type2))

        self.radials = torch.nn.ModuleDict()
        self.layouts = {}
        self.max_l = 0
        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

        self.spherical_harmonics = UnitarySphericalHarmonics(self.max_l)

        self.dense_cg_coeffs = calculate_cg_coefficients(
            self.max_l,
            cg_backend="python-dense",
            arrays_backend="torch",
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        targets = self.dataset_info.targets
        self.outputs = {
            k: ModelOutput(
                quantity=targets[k].quantity if k in targets else "",
                unit=targets[k].unit if k in targets else "",
                sample_kind="atom_pair",
            )
            for k in self.layouts
        }

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        """
        Register a new output target.
        """
        if target_info.sample_kind != "atom_pair":
            raise ValueError(
                f"EdgeCompositionModel only supports targets with sample_kind 'atom_pair'. "
                f"{target_name} has sample_kind '{target_info.sample_kind}'."
            )

        is_coupled = self.hypers["sph_basis"] == "coupled"

        layout = target_info.layout
        if is_coupled:
            layout = couple_tensor_blocks(layout)
        self.layouts[target_name] = layout

        self.radials[target_name] = torch.nn.ModuleDict()

        # Count number of properties per pair of atom types,
        # and store the pointers to the properties of each block
        n_props_per_type_pair = defaultdict(int)
        for block_key, block in layout.items():
            if is_coupled:
                l, o3_sigma, type1, type2 = block_key
                self.max_l = max(self.max_l, l)
            else:
                l1, l2, o3_sigma_1, o3_sigma_2, type1, type2 = block_key
                self.max_l = max(self.max_l, l1, l2)

            n_props = block.properties.values.shape[0]
            n_props_per_type_pair[type1, type2] = (
                n_props_per_type_pair[type1, type2] + n_props
            )

        radial_cls = {
            "exponential": Exponential,
            "tabulated": Tabulated
        }[self.hypers["radial_basis"]]

        for (type1, type2), n_props in n_props_per_type_pair.items():
            self.radials[target_name][str((type1, type2))] = radial_cls(n_props)

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def capabilities(self) -> ModelCapabilities:
        return self._get_capabilities()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:

        # Quick exit if outputs is empty
        if not outputs:
            return {}

        # --------------------------
        # Moving to device and dtype
        # --------------------------
        # We can't overwrite the to() method because this does not work with
        # torchscript, so we do the necessary operations here.
        # Get device and dtype from the first system
        device = systems[0].device
        # Move layouts to the correct device
        self.layouts = {k: v.to(device=device) for k, v in self.layouts.items()}

        # ---------------------------------
        #   Prepare the input data
        # ---------------------------------
        if not torch.jit.is_scripting():
            if (
                hasattr(self, "batched_neighborlist")
                and self.batched_neighborlist is not None
            ):
                batched_neighborlist = self.batched_neighborlist
            else:
                batched_neighborlist = batch_neighborlist(systems, self.requested_nl)
        else:
            batched_neighborlist = batch_neighborlist(systems, self.requested_nl)

        all_ds: dict[str, Tensor] = {}
        for key, block in batched_neighborlist.items():
            type1, type2 = int(key[0]), int(key[1])
            vs = block.values.reshape(-1, 3)
            ds = torch.linalg.norm(vs, dim=-1)
            all_ds[str((type1, type2))] = ds

        # ---------------------------------
        #       Run radial models
        # ---------------------------------
        radial_outputs: dict[str, dict[str, Tensor]] = {}
        for target_name, target_radials in self.radials.items():
            if target_name in outputs:
                radial_outputs[target_name] = {}
            for key, radial_model in target_radials.items():
                if key in all_ds and len(all_ds[key]) > 0 and target_name in outputs:
                    radial_outputs[target_name][key] = radial_model(all_ds[key])

        # ---------------------------------
        #  Collect outputs into TensorMaps
        # ---------------------------------
        return_dict: dict[str, TensorMap] = {}
        for target_name, target_radial_outs in radial_outputs.items():
            target_layout = self.layouts[target_name]
            if self.training:
                blocks: list[TensorBlock] = []
                keys: list[list[int]] = []
                for key, radial_out in target_radial_outs.items():
                    type1, type2 = key.replace("(", "").replace(")", "").split(", ")
                    type1, type2 = int(type1), int(type2)
                    keys.append([type1, type2])
                    block = TensorBlock(
                        values=radial_out,
                        samples=batched_neighborlist.block(
                            dict(first_atom_type=type1, second_atom_type=type2)
                        ).samples,
                        components=[],
                        properties=Labels(
                            names=["_"],
                            values=torch.arange(
                                radial_out.shape[1], device=device
                            ).reshape(-1, 1),
                        ),
                    )
                    blocks.append(block)
                return_dict[target_name] = TensorMap(
                    keys=Labels(
                        names=["first_atom_type", "second_atom_type"],
                        values=torch.tensor(keys, device=device),
                    ),
                    blocks=blocks,
                )
            else:
                all_shs = {}
                for block_key, block in batched_neighborlist.items():
                    type1, type2 = int(block_key[0]), int(block_key[1])
                    vs = block.values.reshape(-1, 3)
                    shs = self.spherical_harmonics(vs)
                    all_shs[str((type1, type2))] = shs

                return_dict[target_name] = radial_to_spherical_harmonics(
                    target_radial_outs,
                    all_shs,
                    target_layout,
                    batched_neighborlist,
                    self.dense_cg_coeffs,
                )

        return return_dict

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "EdgeCompositionModel":
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

        # Create the model
        model_data = checkpoint["model_data"]
        if "sph_basis" not in model_data["hypers"]:
            model_data["hypers"]["sph_basis"] = "uncoupled"
        if "radial_basis" not in model_data["hypers"]:
            model_data["hypers"]["radial_basis"] = "exponential"
        model = cls(**model_data)

        if "spherical_harmonics._factors" not in checkpoint["model_state_dict"]:
            old_factors = model.spherical_harmonics._factors.clone()
            old_factors[1:] = old_factors[1:] ** 2
            checkpoint["model_state_dict"]["spherical_harmonics._factors"] = old_factors

        # Infer dtype
        model_state_dict = checkpoint["model_state_dict"]
        state_dict_iter = iter(model_state_dict.values())
        dtype = next(state_dict_iter).dtype
        # Load the state dict.
        model.to(dtype).load_state_dict(model_state_dict)

        # Loading the metadata from the checkpoint
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    def _get_capabilities(self) -> ModelCapabilities:
        dtype = next(self.parameters()).dtype

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=self.hypers["cutoff"],
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return capabilities

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for MACE")

        # Make sure the model is all in the same dtype
        self.to(dtype)

        capabilities = self._get_capabilities()

        metadata = merge_metadata(self.metadata, metadata)

        model = AtomisticModel(self.eval(), metadata, capabilities)

        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()

        checkpoint = {
            "architecture_name": "experimental.edge_composition",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info.to(device="cpu"),
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": model_state_dict,
        }
        return checkpoint

    def restart(self, dataset_info: DatasetInfo) -> "EdgeCompositionModel":
        # Check that the new dataset info does not contain new atomic types
        if new_atomic_types := set(dataset_info.atomic_types) - set(
            self.dataset_info.atomic_types
        ):
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "EdgeCompositionModel does not support adding new atomic types."
            )

        # Merge the old dataset info with the new one
        merged_info = self.dataset_info.union(dataset_info)

        # Check if there are new targets
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        # Add extra heads for the new targets
        for target_name in new_targets:
            self._add_output(target_name, dataset_info.targets[target_name])

        self.dataset_info = merged_info

        return self
