"""Load lorem-jax parameters into a :class:`~metatrain.experimental.lorem.LOREM`.

lorem-jax exports (``scripts/export.py``, lab-cosmo/lorem-jax#42) are a folder
with the flax parameters flattened to ``/``-joined paths in ``params.npz``,
the checkpoint's ``model.yaml`` and ``baseline.yaml``, and ``export.yaml``
with the format version. :func:`read_export` reads one, and
:func:`load_checkpoint` copies the parameters into a model built from the
returned hypers. Neither needs JAX.

Flax names submodules ``{Type}_{n}``, counting each type in call order, so the
loader walks the model in lorem-jax's call order and asks for the next name of
each type. Every leaf is a copy, up to layout:

- flax ``Dense`` kernels are transposed into ``torch.nn.Linear`` weights.
- e3x ``Dense`` has one kernel per degree, named ``{l}+`` or ``{l}-``, stacked
  here into one ``(max_degree + 1, in, out)`` weight.
- e3x ``Tensor`` kernels are a ``(l1, l2, parity, L)`` grid; the parity slot
  used here is copied whole. The Clebsch-Gordan buffers carry e3x's signs,
  so no per-triple correction is needed.

Parameter paths may keep flax's leading ``params/`` or drop it.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Set, Tuple

import numpy as np
import torch


if TYPE_CHECKING:
    from ..model import LOREM


_MODEL_CLASSES = ("lorem.models.mlip.Lorem", "lorem.models.bec.LoremBEC")
_EXPORT_FORMAT = 1
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
    for ell in range(dense.weight.shape[0]):
        degree = scope.child(f"{ell}{'+' if ell % 2 == 0 else '-'}")
        _copy(dense.weight[ell], degree["kernel"], degree.prefix + "kernel")
    if dense.bias is not None:
        scalar = scope.child("0+")
        _copy(dense.bias, scalar["bias"], scalar.prefix + "bias")


def _tensor(product: torch.nn.Module, scope: _Scope) -> None:
    """``(1, l1, 1, l2, parity, L, features)`` kernel to ``(l1, l2, L, features)``.

    Both inputs are proper tensors, so parity slot 0 is the channel kept here:
    parity ``(-1)**L`` without pseudotensors, ``+1`` with them.
    """
    kernel = scope["kernel"]
    _copy(product.tensor_weight, kernel[0, :, 0, :, 0], scope.prefix + "kernel")


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
        ...}``, flax paths with or without the leading ``params/``.
    :raises KeyError: if the model expects a parameter the checkpoint lacks.
    :raises ValueError: if a shape differs, or the checkpoint has parameters
        the model does not use.
    """
    params = {name.removeprefix("params/"): value for name, value in params.items()}
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
    """LOREM model hypers for a lorem-jax checkpoint's ``model.yaml``.

    :param model_yaml: ``{"lorem.models.mlip.Lorem": {field: value, ...}}``, the
        model's spec dict with every field written out (``LoremBEC`` likewise).
    :return: The lorem-jax class name and the LOREM model hypers.
    """
    from metatrain.utils.architectures import get_default_hypers

    ((name, fields),) = model_yaml.items()
    if name not in _MODEL_CLASSES:
        raise ValueError(f"unsupported lorem-jax model '{name}'")
    fields = dict(fields)
    if inputs := fields.pop("inputs", []):
        raise ValueError(f"conditioning on inputs is not supported, got {inputs}")
    for field, supported in _FIXED_FIELDS.items():
        if (value := fields.pop(field, supported)) != supported:
            raise ValueError(f"only {field}={supported!r} is supported, got {value!r}")

    expected = set(get_default_hypers("experimental.lorem")["model"])
    if set(fields) != expected:
        raise ValueError(
            f"lorem-jax model fields {sorted(set(fields) - expected)} are unknown "
            f"and {sorted(expected - set(fields))} are missing"
        )
    return name, fields


def read_export(
    folder: str | Path,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[int, float]]:
    """Read a lorem-jax export folder.

    :param folder: Holds ``params.npz``, ``model.yaml``, ``baseline.yaml`` and
        ``export.yaml``, as written by lorem-jax's ``scripts/export.py``.
    :return: The lorem-jax class name, the LOREM model hypers, the flat
        parameters for :func:`load_checkpoint`, and the per-element energy
        baseline (usable as ``atomic_baseline: {"energy": ...}``).
    """
    import yaml

    folder = Path(folder)
    export_format = yaml.safe_load((folder / "export.yaml").read_text())["format"]
    if export_format != _EXPORT_FORMAT:
        raise ValueError(f"unsupported lorem-jax export format {export_format}")
    with np.load(folder / "params.npz") as data:
        params = {key: data[key] for key in data.files}

    name, hypers = model_hypers_from_yaml(
        yaml.safe_load((folder / "model.yaml").read_text())
    )
    baseline = yaml.safe_load((folder / "baseline.yaml").read_text())
    elemental = {int(z): float(weight) for z, weight in baseline["elemental"].items()}
    return name, hypers, params, elemental
