import contextlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import ase.data
import metatensor.torch as mts
import torch
from deepmd.pt.model.model import get_standard_model
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
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
from metatrain.utils.data.target_info import get_features_info
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.hypers import raise_if_hypers_mismatch
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers
from .modules.structures import concatenate_structures


_PRECISION_INT_TO_DTYPE = {32: torch.float32, 64: torch.float64}
_INT_TO_DEEPMD_PREC = {32: "float32", 64: "float64"}
_DEEPMD_PREC_TO_INT = {v: k for k, v in _INT_TO_DEEPMD_PREC.items()}
_ARCHITECTURE_NAME = "experimental.se_e2_a"


@contextlib.contextmanager
def _build_on_cpu():
    """Force deepmd-kit module-level device constants to CPU during model
    construction, ensuring deterministic (CPU) RNG for weight initialization
    regardless of CUDA availability."""
    cpu = torch.device("cpu")
    saved = {}
    for name, mod in sys.modules.items():
        if mod is None or not name.startswith("deepmd.pt"):
            continue
        for attr in ("device", "DEVICE"):
            val = getattr(mod, attr, None)
            if isinstance(val, torch.device) and val.type == "cuda":
                saved[(name, attr)] = val
                setattr(mod, attr, cpu)
    try:
        yield
    finally:
        for (name, attr), original in saved.items():
            mod = sys.modules.get(name)
            if mod is not None:
                setattr(mod, attr, original)


def _register_untracked_tensors(model: torch.nn.Module) -> None:
    """Register plain tensor attributes as non-persistent buffers.

    deepmd-kit stores some tensors (e.g. type_mask) as plain attributes
    rather than parameters or buffers.  These are invisible to .to(device)
    and .to(dtype).  Converting them to non-persistent buffers makes them
    move correctly without affecting state_dict (checkpoint compatibility)."""
    for module in model.modules():
        registered = set()
        for n, _ in module.named_parameters(recurse=False):
            registered.add(n)
        for n, _ in module.named_buffers(recurse=False):
            registered.add(n)
        for attr_name in list(vars(module).keys()):
            val = getattr(module, attr_name)
            if isinstance(val, torch.Tensor) and attr_name not in registered:
                delattr(module, attr_name)
                module.register_buffer(attr_name, val, persistent=False)


def _resolve_sel(sel: Union[int, List[int]], n_types: int) -> List[int]:
    """Expand ``descriptor.sel`` to one positive count per element type."""
    if isinstance(sel, bool) or isinstance(sel, int):
        if isinstance(sel, bool) or sel < 1:
            raise ValueError(f"descriptor.sel must be an integer >= 1, got {sel!r}.")
        return [int(sel)] * n_types
    try:
        resolved = [int(value) for value in sel]
    except TypeError as err:
        raise ValueError(
            "descriptor.sel must be a positive integer or a list of positive "
            f"integers, got {sel!r}."
        ) from err
    if len(resolved) != n_types:
        raise ValueError(
            f"descriptor.sel has {len(resolved)} entries but this dataset has "
            f"{n_types} atomic types. Pass an integer to use the same neighbor "
            "count for every type, or a list with one entry per type in "
            "type_map order."
        )
    if any(value < 1 for value in resolved):
        raise ValueError(f"Every descriptor.sel entry must be >= 1, got {resolved}.")
    return resolved


def _deepmd_config(hypers: Mapping[str, Any], type_map: List[str]) -> Dict[str, Any]:
    """Translate metatrain hypers into a deepmd-kit ``se_e2_a`` configuration."""
    descriptor = hypers["descriptor"]
    fitting = hypers["fitting_net"]
    desc_prec = descriptor["precision"]
    fit_prec = fitting["precision"]
    if desc_prec not in _INT_TO_DEEPMD_PREC:
        raise ValueError(
            f"Unsupported descriptor precision: {desc_prec}. "
            f"Must be one of {list(_INT_TO_DEEPMD_PREC.keys())}."
        )
    if fit_prec not in _INT_TO_DEEPMD_PREC:
        raise ValueError(
            f"Unsupported fitting_net precision: {fit_prec}. "
            f"Must be one of {list(_INT_TO_DEEPMD_PREC.keys())}."
        )
    neuron = [int(value) for value in descriptor["neuron"]]
    if len(neuron) == 0:
        raise ValueError("descriptor.neuron must contain at least one width.")
    axis_neuron = int(descriptor["axis_neuron"])
    if axis_neuron < 1 or axis_neuron > neuron[-1]:
        raise ValueError(
            f"descriptor.axis_neuron ({axis_neuron}) must be between 1 and the "
            f"last descriptor neuron ({neuron[-1]})."
        )
    rcut = float(descriptor["rcut"])
    rcut_smth = float(descriptor["rcut_smth"])
    if rcut_smth >= rcut:
        raise ValueError(
            f"descriptor.rcut_smth ({rcut_smth}) must be smaller than "
            f"descriptor.rcut ({rcut})."
        )
    fitting_neuron = [int(value) for value in fitting["neuron"]]
    if len(fitting_neuron) == 0:
        raise ValueError("fitting_net.neuron must contain at least one width.")

    descriptor_config: Dict[str, Any] = {
        "type": "se_e2_a",
        "sel": _resolve_sel(descriptor["sel"], len(type_map)),
        "rcut": rcut,
        "rcut_smth": rcut_smth,
        "neuron": neuron,
        "axis_neuron": axis_neuron,
        "resnet_dt": bool(descriptor["resnet_dt"]),
        "activation_function": descriptor["activation_function"],
        "precision": _INT_TO_DEEPMD_PREC[desc_prec],
        "type_one_side": bool(descriptor["type_one_side"]),
        "set_davg_zero": bool(descriptor["set_davg_zero"]),
        "env_protection": float(descriptor["env_protection"]),
        "trainable": bool(descriptor["trainable"]),
        "seed": descriptor["seed"],
    }
    exclude_types = descriptor.get("exclude_types") or []
    if len(exclude_types) > 0:
        descriptor_config["exclude_types"] = [
            [int(pair[0]), int(pair[1])] for pair in exclude_types
        ]

    fitting_config: Dict[str, Any] = {
        "neuron": fitting_neuron,
        "resnet_dt": bool(fitting["resnet_dt"]),
        "seed": fitting["seed"],
        "precision": _INT_TO_DEEPMD_PREC[fit_prec],
        "activation_function": fitting["activation_function"],
        "type": fitting["type"],
        "numb_fparam": int(fitting["numb_fparam"]),
        "numb_aparam": int(fitting["numb_aparam"]),
        "dim_case_embd": int(fitting["dim_case_embd"]),
        "trainable": bool(fitting["trainable"]),
        "use_aparam_as_mask": bool(fitting["use_aparam_as_mask"]),
    }
    if fitting.get("rcond") is not None:
        fitting_config["rcond"] = float(fitting["rcond"])

    return {
        "type_map": list(type_map),
        "descriptor": descriptor_config,
        "fitting_net": fitting_config,
    }


class SeE2A(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "https://github.com/deepmodeling/deepmd-kit",
            ],
            "architecture": [
                "se_e2_a: https://papers.nips.cc/paper/7696",
            ],
        }
    )

    key_labels: Dict[str, Labels]
    component_labels: Dict[str, List[List[Labels]]]
    property_labels: Dict[str, List[Labels]]
    n_types: int
    type_specific_features: bool

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)
        self.atomic_types = list(dataset_info.atomic_types)
        if len(self.atomic_types) == 0:
            raise ValueError("se_e2_a needs at least one atomic type.")
        if len(dataset_info.targets) == 0:
            raise ValueError("se_e2_a needs at least one target.")

        desc_prec = self.hypers["descriptor"]["precision"]
        fit_prec = self.hypers["fitting_net"]["precision"]
        if desc_prec not in _PRECISION_INT_TO_DTYPE:
            raise ValueError(
                f"Unsupported descriptor precision: {desc_prec}. "
                f"Must be one of {list(_PRECISION_INT_TO_DTYPE.keys())}."
            )
        if fit_prec not in _PRECISION_INT_TO_DTYPE:
            raise ValueError(
                f"Unsupported fitting_net precision: {fit_prec}. "
                f"Must be one of {list(_PRECISION_INT_TO_DTYPE.keys())}."
            )
        self.dtype: torch.dtype = _PRECISION_INT_TO_DTYPE[desc_prec]
        self.targets_keys = list(dataset_info.targets.keys())[0]

        self.loaded_se_e2_a = self.hypers.get("se_e2_a_model") is not None
        self._loaded_out_bias: Optional[torch.Tensor] = None
        self._loaded_out_std: Optional[torch.Tensor] = None
        if self.loaded_se_e2_a:
            self._load_deepmd_model(dataset_info)
        else:
            type_map = [ase.data.chemical_symbols[z] for z in self.atomic_types]
            with _build_on_cpu():
                self.model = get_standard_model(_deepmd_config(self.hypers, type_map))
            _register_untracked_tensors(self.model)

        self._set_feature_layout()

        self.outputs: Dict[str, TargetInfo] = {}
        self.num_properties: Dict[str, Dict[str, int]] = {}
        self.register_buffer("key_labels", {})
        self.register_buffer("component_labels", {})
        self.register_buffer("property_labels", {})
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

        self.requested_nl = NeighborListOptions(
            cutoff=float(self.hypers["descriptor"]["rcut"]),
            full_list=True,
            strict=True,
        )

    def _load_deepmd_model(self, dataset_info: DatasetInfo) -> None:
        """Build ``self.model`` from a checkpoint, module, or deepmd-kit config."""
        se_e2_a_model = self.hypers["se_e2_a_model"]
        if isinstance(se_e2_a_model, (str, Path)):
            loaded = torch.load(
                str(se_e2_a_model),
                weights_only=False,
                map_location=torch.device("cpu"),
            )
            state_dict = loaded.get("model") if isinstance(loaded, dict) else None
            if not isinstance(state_dict, dict) or "_extra_state" not in state_dict:
                raise ValueError(
                    f"'{se_e2_a_model}' is not a deepmd-kit training checkpoint. "
                    "Use the checkpoint written by deepmd-kit training directly, "
                    "without freezing it via 'dp --pt freeze'."
                )

            params = state_dict["_extra_state"]["model_params"]
            if "model_dict" not in params:
                cfg, prefix = params, "model.Default."
                branch = None
            else:
                branch = self.hypers["se_e2_a_model_branch"]
                available = sorted(params["model_dict"])
                if branch is None:
                    raise ValueError(
                        "The pretrained se_e2_a model has multiple branches. "
                        "Please specify the branch to load via the "
                        "'se_e2_a_model_branch' hyperparameter. Available "
                        f"branches: {', '.join(available)}"
                    )
                if branch not in params["model_dict"]:
                    raise ValueError(
                        f"The specified branch '{branch}' is not available in "
                        "the pretrained se_e2_a model. "
                        f"Available branches: {', '.join(available)}"
                    )
                cfg = params["model_dict"][branch]
                prefix = f"model.{branch}."
            with _build_on_cpu():
                self.model = get_standard_model(cfg)
            branch_state_dict = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            missing, unexpected = self.model.load_state_dict(
                branch_state_dict, strict=False
            )
            real = [key for key in missing + unexpected if "_extra_state" not in key]
            if real:
                where = f"branch {branch}" if branch is not None else "checkpoint"
                raise RuntimeError(f"state_dict mismatch for {where}: {real[:5]}")
        elif isinstance(se_e2_a_model, torch.nn.Module):
            self.model = se_e2_a_model.cpu()
        elif isinstance(se_e2_a_model, dict):
            with _build_on_cpu():
                self.model = get_standard_model(se_e2_a_model)
            # The weights that go with this config are loaded afterwards from
            # the metatrain checkpoint, which already has the bias handled.
            self.model._metatrain_extracted_scaleshift = True
        else:
            raise ValueError(
                "The 'se_e2_a_model' hyper must be a file path, a "
                "torch.nn.Module, or a dict."
            )

        self._deepmd_cfg = json.loads(self.model.get_model_def_script())
        cfg = self._deepmd_cfg
        self.hypers["descriptor"]["precision"] = _DEEPMD_PREC_TO_INT[
            cfg["descriptor"]["precision"]
        ]
        self.hypers["fitting_net"]["precision"] = _DEEPMD_PREC_TO_INT[
            cfg["fitting_net"]["precision"]
        ]
        self.hypers["descriptor"]["rcut"] = float(cfg["descriptor"]["rcut"])
        self.hypers["descriptor"]["rcut_smth"] = float(cfg["descriptor"]["rcut_smth"])
        self.hypers["descriptor"]["sel"] = [
            int(value) for value in cfg["descriptor"]["sel"]
        ]
        self.hypers["descriptor"]["neuron"] = list(cfg["descriptor"]["neuron"])
        self.hypers["descriptor"]["axis_neuron"] = int(cfg["descriptor"]["axis_neuron"])
        self.hypers["fitting_net"]["neuron"] = list(cfg["fitting_net"]["neuron"])
        self.dtype = _PRECISION_INT_TO_DTYPE[self.hypers["descriptor"]["precision"]]
        self.atomic_types = [ase.data.atomic_numbers[s] for s in cfg["type_map"]]
        dataset_info.atomic_types = self.atomic_types
        _register_untracked_tensors(self.model)

        if not getattr(self.model, "_metatrain_extracted_scaleshift", False):
            atomic_model = self.model.atomic_model
            if hasattr(atomic_model, "out_bias"):
                self._loaded_out_bias = atomic_model.out_bias.clone()
                atomic_model.out_bias.zero_()
            if hasattr(atomic_model, "out_std"):
                self._loaded_out_std = atomic_model.out_std.clone()
                atomic_model.out_std.fill_(1.0)
            self.model._metatrain_extracted_scaleshift = True

    def _set_feature_layout(self) -> None:
        """Record how many last-layer features one atom produces.

        ``se_e2_a`` fits each element with its own network. The feature that
        LLPR sees is that network's last hidden activation, written into the
        element's own block, followed by the element one-hot that carries the
        per-element output bias.
        """
        hidden = int(self.model.atomic_model.fitting_net.neuron[-1])
        self.n_types = len(self.atomic_types)
        mixed_types = bool(self.model.atomic_model.fitting_net.mixed_types)
        self.type_specific_features = not mixed_types
        if self.type_specific_features:
            self._last_layer_feature_size = self.n_types * (hidden + 1)
        else:
            self._last_layer_feature_size = hidden

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        if not target.is_scalar:
            raise ValueError("The se_e2_a architecture can only predict scalars.")
        self.num_properties[target_name] = {}
        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]
        self.outputs[target_name] = target
        features_name = self._last_layer_features_name(target_name)
        self.outputs[features_name] = get_features_info(
            self._last_layer_feature_size,
            sample_kind="atom",
            description=f"last-layer features for {target_name}",
        )

    @staticmethod
    def _last_layer_features_name(target_name: str) -> str:
        return f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"

    def get_rcut(self):
        return self.model.atomic_model.get_rcut()

    def get_sel(self):
        return self.model.atomic_model.get_sel()

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def _prepare_atype(self, species: torch.Tensor) -> torch.Tensor:
        """Map padded species tensor to deepmd-kit type indices."""
        max_z = max(max(self.atomic_types), 0) + 1
        lookup = torch.full(
            (max_z + 1,), -1, dtype=species.dtype, device=species.device
        )
        for idx, z in enumerate(self.atomic_types):
            lookup[z] = idx
        valid_mask = species >= 0
        safe_species = species.clamp(min=0)
        result = lookup[safe_species]
        result[~valid_mask] = -1
        return result

    def _forward_from_batch(
        self,
        positions: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pure-tensor forward pass.

        :param positions: Padded positions [batch, max_atoms, 3].
        :param atype: Deepmd-kit type indices [batch, max_atoms].
        :param box: Cell tensor [batch, 3, 3], or None when every cell is zero.
        :return: ``"atom_energy"`` and ``"energy"`` as returned by deepmd-kit.
        """
        model_ret = self.model.forward_common(
            positions,
            atype,
            box,
            fparam=None,
            aparam=None,
            do_atomic_virial=False,
        )
        result: Dict[str, torch.Tensor] = {}
        if self.model.get_fitting_net() is not None:
            result["atom_energy"] = model_ret["energy"]
            result["energy"] = model_ret["energy_redu"]
        else:
            result["atom_energy"] = model_ret.get("energy", model_ret["updated_coord"])
            result["energy"] = model_ret.get("energy_redu", torch.zeros(1))
        return result

    @torch.jit.unused
    def _forward_from_batch_with_grad(
        self,
        positions: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run deepmd-kit with autograd enabled in eager no-grad contexts."""
        with torch.enable_grad():
            return self._forward_from_batch(positions, atype, box)

    def _forward_embedding_from_batch(
        self,
        positions: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return fitting features without changing the model mode.

        deepmd-kit's public ``model.forward_embedding`` refuses training mode.
        metatrain still has to return last-layer features when ``forward`` is
        called while training, so the neighbor list is built here and the
        atomic-model embedding is called directly.
        """
        coord, cast_box, fparam, aparam, _ = self.model._input_type_cast(
            positions,
            box=box,
            fparam=None,
            aparam=None,
        )
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.model.get_rcut(),
            self.model.get_sel(),
            # ``format_nlist`` splits this mixed list by element for se_e2_a.
            mixed_types=True,
            box=cast_box,
        )
        extended_coord = extended_coord.view(extended_atype.shape[0], -1, 3)
        nlist = self.model.format_nlist(
            extended_coord,
            extended_atype,
            nlist,
        )
        return self.model.atomic_model.forward_embedding(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
        )

    def _project_last_layer_features(
        self, atomic_feature: torch.Tensor, atype: torch.Tensor
    ) -> torch.Tensor:
        """Map per-atom fitting activations to the LLPR feature of each atom."""
        if not self.type_specific_features:
            return atomic_feature
        one_hot = torch.nn.functional.one_hot(atype.long(), num_classes=self.n_types)
        one_hot = one_hot.to(dtype=atomic_feature.dtype)
        gated = one_hot.unsqueeze(-1) * atomic_feature.unsqueeze(1)
        n_atoms = atomic_feature.shape[0]
        hidden = atomic_feature.shape[-1]
        return torch.cat(
            [gated.reshape(n_atoms, self.n_types * hidden), one_hot], dim=1
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}

        device = systems[0].positions.device
        return_dict: Dict[str, TensorMap] = {}
        (positions, species, cells, atom_index, system_index) = concatenate_structures(
            systems
        )
        atype = self._prepare_atype(species)
        if torch.all(cells == 0).item():
            box = None
        else:
            box = cells

        values = torch.stack([system_index, atom_index], dim=0).transpose(0, 1)
        invariant_coefficients = Labels(
            names=["system", "atom"], values=values.to(device)
        )
        # Real atoms have a species >= 0. Padding added by the batch collate is -1.
        mask = species >= 0

        if self.targets_keys in outputs:
            # deepmd-kit differentiates positions inside its forward to build
            # forces. LLPR collects features under ``no_grad``, so re-enable
            # gradients for the energy path when not scripting.
            if torch.jit.is_scripting():
                raw = self._forward_from_batch(positions, atype, box)
            else:
                raw = self._forward_from_batch_with_grad(positions, atype, box)

            batch_size, max_atoms = species.shape
            atom_energy = raw["atom_energy"].reshape(batch_size, max_atoms)
            atomic_property = TensorMap(
                self.key_labels[self.targets_keys],
                [
                    TensorBlock(
                        values=atom_energy[mask].unsqueeze(-1),
                        samples=invariant_coefficients,
                        components=self.component_labels[self.targets_keys][0],
                        properties=self.property_labels[self.targets_keys][0],
                    )
                ],
            )
            if selected_atoms is not None:
                atomic_property = mts.slice(
                    atomic_property, axis="samples", selection=selected_atoms
                )
            if outputs[self.targets_keys].sample_kind == "atom":
                return_dict[self.targets_keys] = atomic_property
            else:
                return_dict[self.targets_keys] = sum_over_atoms(atomic_property)

        features_name = self._last_layer_features_name(self.targets_keys)
        if features_name in outputs:
            # The embedding path returns no energy, so the two outputs are two
            # forwards and the neighbor list is built twice.
            embedding = self._forward_embedding_from_batch(positions, atype, box)
            atomic_features = self._project_last_layer_features(
                embedding["atomic_feature"][mask],
                atype[mask],
            )
            features_layout = self.outputs[features_name].layout
            features_block = features_layout.block(0)
            features = TensorMap(
                features_layout.keys.to(device),
                [
                    TensorBlock(
                        values=atomic_features,
                        samples=invariant_coefficients,
                        components=[],
                        properties=features_block.properties.to(device),
                    )
                ],
            )
            if selected_atoms is not None:
                features = mts.slice(features, axis="samples", selection=selected_atoms)
            if outputs[features_name].sample_kind == "system":
                features = sum_over_atoms(features)
            return_dict[features_name] = features

        return return_dict

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> "SeE2A":
        if model_hypers is not None:
            default_hypers = get_default_hypers(_ARCHITECTURE_NAME)["model"]
            raise_if_hypers_mismatch(
                self.hypers, model_hypers, default_hypers=default_hypers
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
                "The se_e2_a model does not support adding new atomic types."
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
    ) -> "SeE2A":
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
        # deepmd-kit stores its precision on the module at construction time.
        # .to(dtype) does not update it, so the dtype to load is the one the
        # rebuilt module already has.
        dtype = next(model.model.parameters()).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata
        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for se_e2_a")
        self.to(dtype)
        capabilities = ModelCapabilities(
            outputs={
                name: info.to_model_output(sample_kind="atom")
                for name, info in self.outputs.items()
            },
            atomic_types=self.atomic_types,
            interaction_range=float(self.hypers["descriptor"]["rcut"]),
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
        hypers = dict(self.hypers)
        # deepmd-kit modules contain locally defined classes that cannot be
        # pickled. The configuration dict rebuilds the same module; the
        # state_dict carries the weights.
        if hasattr(self, "_deepmd_cfg"):
            hypers["se_e2_a_model"] = self._deepmd_cfg
        else:
            hypers.pop("se_e2_a_model", None)
        return {
            "architecture_name": _ARCHITECTURE_NAME,
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": best_model_state_dict or self.state_dict(),
        }

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return self.outputs

    def get_fixed_composition_weights(self) -> dict[str, dict[int, float]]:
        """Per-element energy biases extracted from a loaded deepmd-kit model.

        Passed to ``CompositionModel.train_model`` as ``fixed_weights`` so a
        fine-tune keeps the pretrained bias instead of fitting a new one.
        """
        if self._loaded_out_bias is None:
            return {}
        bias = self._loaded_out_bias[0, :, 0]
        return {
            self.targets_keys: {
                z: bias[i].item() for i, z in enumerate(self.atomic_types)
            }
        }

    def get_fixed_scaling_weights(self) -> dict[str, Union[float, dict[int, float]]]:
        """Per-element output scales extracted from a loaded deepmd-kit model.

        The scaler only accepts one scalar for a per-structure target, so a
        model whose elements were stored with different scales is rejected.
        """
        if self._loaded_out_std is None:
            return {}
        std = self._loaded_out_std[0, :, 0]
        if not torch.all(std == std[0]):
            raise NotImplementedError(
                "Loaded se_e2_a models with non-uniform per-type 'out_std' are "
                "not supported."
            )
        return {self.targets_keys: std[0].item()}
