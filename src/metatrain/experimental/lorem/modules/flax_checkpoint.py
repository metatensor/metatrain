"""Load lorem-jax parameters into a :class:`~metatrain.experimental.lorem.LOREM`.

lorem-jax exports (lab-cosmo/lorem-jax#39) are a folder with the flax
parameters flattened to ``/``-joined paths (``.npz`` or ``.safetensors``),
plus ``model.yaml`` and ``baseline.yaml``. :func:`read_export` reads one, and
:func:`load_checkpoint` copies the parameters into a model built from the
returned hypers. Neither needs JAX.

Flax names submodules ``{Type}_{n}``, counting each type in call order, so the
loader walks the model in lorem-jax's call order and asks for the next name of
each type. Three leaves are not a plain copy:

- ``Dense`` kernels are transposed.
- e3x layers have one kernel per degree, named ``{l}+`` or ``{l}-``.
- Clebsch-Gordan weights are a dense ``(l1, l2, parity, L)`` grid, including
  invalid triples. Only the triples used here are kept, with
  :func:`~.e3x_compat.cg_phase_correction`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Set, Tuple

import numpy as np
import torch

from .e3x_compat import cg_phase_correction


if TYPE_CHECKING:
    from ..model import LOREM


# lorem-jax model classes and the field defaults that differ between them
_MODEL_CLASSES = {
    "lorem.Lorem": {
        "equivariant_message_passing": True,
        "initialize_node_features": True,
    },
    "lorem.LoremBEC": {
        "equivariant_message_passing": False,
        "initialize_node_features": False,
    },
}
_FIXED_FIELDS = {
    "cutoff_fn": "cosine_cutoff",
    "radial_basis": "basic_bernstein",
    "lr": True,
}


class _Scope:
    """A flax scope: hands out ``{Type}_{n}`` names and records used leaves."""

    def __init__(self, params: Mapping[str, Any], prefix: str, used: Set[str]) -> None:
        self.params, self.prefix, self.used = params, prefix, used
        self.counts: Dict[str, int] = {}

    def next(self, kind: str) -> "_Scope":
        n = self.counts.get(kind, 0)
        self.counts[kind] = n + 1
        return self.child(f"{kind}_{n}")

    def child(self, name: str) -> "_Scope":
        return _Scope(self.params, f"{self.prefix}{name}/", self.used)

    def has(self, leaf: str) -> bool:
        return self.prefix + leaf in self.params

    def __getitem__(self, leaf: str) -> torch.Tensor:
        name = self.prefix + leaf
        if name not in self.params:
            raise KeyError(f"missing lorem-jax parameter '{name}'")
        self.used.add(name)
        return torch.as_tensor(np.asarray(self.params[name]))


def _copy(target: torch.Tensor, value: torch.Tensor, name: str) -> None:
    if target.shape != value.shape:
        raise ValueError(
            f"shape mismatch for '{name}': {tuple(value.shape)} in the checkpoint, "
            f"{tuple(target.shape)} in the model. Do the hypers match model.yaml?"
        )
    target.copy_(value)


def _linear(layer: torch.nn.Linear, scope: _Scope) -> None:
    _copy(layer.weight, scope["kernel"].T, scope.prefix + "kernel")
    if layer.bias is not None:
        _copy(layer.bias, scope["bias"], scope.prefix + "bias")


def _mlp(mlp: torch.nn.Sequential, scope: _Scope) -> None:
    layers = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
    for index, layer in enumerate(layers):
        _linear(layer, scope.child(f"Dense_{index}"))


def _update(update: torch.nn.Module, scope: _Scope) -> None:
    for index in (0, 1):
        _mlp(getattr(update, f"mlp{index}"), scope.child(f"MLP_{index}"))
        norm, norm_scope = (
            getattr(update, f"norm{index}"),
            scope.child(f"LayerNorm_{index}"),
        )
        _copy(norm.weight, norm_scope["scale"], norm_scope.prefix + "scale")
        _copy(norm.bias, norm_scope["bias"], norm_scope.prefix + "bias")


def _degree_wise(dense: torch.nn.Module, scope: _Scope) -> None:
    for ell, layer in enumerate(dense.layers):
        _linear(layer, scope.child(f"{ell}{'+' if ell % 2 == 0 else '-'}"))


def _tensor(product: torch.nn.Module, scope: _Scope) -> None:
    """``(1, l1, 1, l2, parity, L, features)`` grid to one row per coupling.

    Both inputs are proper tensors, so parity slot 0 is the channel kept here:
    parity ``(-1)**L`` without pseudotensors, ``+1`` with them.
    """
    grid = scope["kernel"]
    l1_max, l2_max, n_parity, out_max = (
        grid.shape[1],
        grid.shape[3],
        grid.shape[4],
        grid.shape[5],
    )
    grid = grid.reshape(l1_max, l2_max, n_parity, out_max, -1)[:, :, 0]
    for index, coupling in enumerate(product.couplings):
        l1, l2, L = coupling.l1, coupling.l2, coupling.L
        value = grid[l1, l2, L] * cg_phase_correction(l1, l2, L)
        _copy(product.tensor_weight[index], value, scope.prefix + "kernel")


def _tensor_dense(tensor_dense: torch.nn.Module, scope: _Scope) -> None:
    _degree_wise(tensor_dense.dense, scope.child("dense"))
    _tensor(tensor_dense, scope.child("tensor"))


def _bec_head(head: torch.nn.Module, scope: _Scope) -> None:
    for index in range(3):
        _degree_wise(getattr(head, f"dense{index}"), scope.child(f"Dense_{index}"))
    _tensor_dense(head.tensor_dense, scope.child("TensorDense_0"))


def load_checkpoint(model: "LOREM", params: Mapping[str, Any]) -> None:
    """Copy flat lorem-jax parameters into ``model``, in place.

    :param model: A LOREM built with the hypers from the checkpoint's
        ``model.yaml`` (see :func:`read_export`), with a Born-effective-charge
        target when the checkpoint is a ``LoremBEC``.
    :param params: ``{"Initial_0/ChemicalEmbedding_0/Embed_0/embedding": array,
        ...}``, flax paths without the leading ``params/``.
    :raises KeyError: if the model expects a parameter the checkpoint lacks.
    :raises ValueError: if a shape differs, or the checkpoint has parameters
        the model does not use.
    """
    used: Set[str] = set()
    root = _Scope(params, "", used)
    sr, lr = model.sr, model.lr
    bec_heads = [model.bec_heads[name] for name in model.bec_targets]
    bec_stages = [iter([*head.stages, head.long_range]) for head in bec_heads]

    def bec_stage() -> None:
        for head_stages in bec_stages:
            _bec_head(next(head_stages), root.next("PerParticleTensorPredictor"))

    with torch.no_grad():
        embedding = (
            root.child("Initial_0").child("ChemicalEmbedding_0").child("Embed_0")
        )
        _copy(sr.chemical_embedding.weight, embedding["embedding"], "embedding")
        _mlp(sr.radial_coefficients, root.next("RadialCoefficients").child("MLP_0"))
        for layer in sr.dense0:
            _linear(layer, root.next("Dense"))
        _linear(sr.dense1, root.next("Dense"))
        _update(sr.update0, root.next("Update"))
        _linear(sr.dense2, root.next("Dense"))
        _tensor_dense(sr.tensor_dense, root.next("TensorDense"))
        _update(sr.update1, root.next("Update"))
        _mlp(sr.energy_mlp, root.next("MLP"))
        bec_stage()

        for step in sr.message_passing:
            _mlp(
                step.radial_coefficients, root.next("RadialCoefficients").child("MLP_0")
            )
            _linear(step.edge_dense, root.next("Dense"))
            _update(step.update_edges, root.next("Update"))
            if step.equivariant:
                _linear(step.coeff_dense, root.next("Dense"))
                message_pass = root.next("MessagePass")
                _degree_wise(step.message_pass.filter, message_pass.child("filter"))
                _tensor(step.message_pass.message_tensor, message_pass.child("tensor"))
                _degree_wise(step.message_pass.combine_dense_x, root.next("Dense"))
                _degree_wise(step.message_pass.combine_dense_m, root.next("Dense"))
                _tensor(step.message_pass.combine_tensor, root.next("Tensor"))
                _update(step.update_norms, root.next("Update"))
            _mlp(step.energy_mlp, root.next("MLP"))
            bec_stage()

        _mlp(lr.scalar_charge_mlp, root.next("MLP"))
        _tensor_dense(lr.spherical_charge_dense, root.next("TensorDense"))
        _degree_wise(lr.potential_to_features, root.next("Dense"))
        _tensor(lr.potential_product, root.next("Tensor"))
        _update(lr.update2, root.next("Update"))
        _mlp(lr.energy_mlp, root.next("MLP"))
        bec_stage()

    unused = sorted(set(params) - used)
    if unused:
        raise ValueError(
            f"{len(unused)} lorem-jax parameters were not used, starting with "
            f"'{unused[0]}'. Do the hypers and targets match the checkpoint?"
        )


def model_hypers_from_yaml(model_yaml: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """LOREM model hypers for a lorem-jax ``model.yaml``.

    :return: The lorem-jax class name (``lorem.Lorem`` or ``lorem.LoremBEC``)
        and the complete model hypers, defaults included.
    """
    from metatrain.utils.architectures import get_default_hypers

    ((name, fields),) = model_yaml["model"].items()
    if name not in _MODEL_CLASSES:
        raise ValueError(f"unsupported lorem-jax model '{name}'")
    fields = dict(fields or {})
    for field, supported in _FIXED_FIELDS.items():
        if (value := fields.pop(field, supported)) != supported:
            raise ValueError(f"only {field}={supported!r} is supported, got {value!r}")

    hypers = dict(get_default_hypers("experimental.lorem")["model"])
    hypers.update(_MODEL_CLASSES[name])
    unknown = set(fields) - set(hypers)
    if unknown:
        raise ValueError(f"unknown lorem-jax model fields: {sorted(unknown)}")
    hypers.update(fields)
    return name, hypers


def read_export(
    folder: str | Path,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[int, float]]:
    """Read a lorem-jax export folder.

    :param folder: Holds exactly one ``*.npz`` or ``*.safetensors`` file of flat
        parameters, ``model.yaml`` and ``baseline.yaml``.
    :return: The lorem-jax class name, the LOREM model hypers, the flat
        parameters for :func:`load_checkpoint`, and the per-element energy
        baseline (usable as ``atomic_baseline: {"energy": ...}``).
    """
    import yaml

    folder = Path(folder)
    files = sorted([*folder.glob("*.npz"), *folder.glob("*.safetensors")])
    if len(files) != 1:
        raise ValueError(
            f"expected one .npz or .safetensors file in {folder}, found {files}"
        )
    if files[0].suffix == ".npz":
        with np.load(files[0]) as data:
            params = {key: data[key] for key in data.files}
    else:
        from safetensors.numpy import load_file

        params = load_file(files[0])

    name, hypers = model_hypers_from_yaml(
        yaml.safe_load((folder / "model.yaml").read_text())
    )
    baseline = yaml.safe_load((folder / "baseline.yaml").read_text())
    elemental = {int(z): float(weight) for z, weight in baseline["elemental"].items()}
    return name, hypers, params, elemental
