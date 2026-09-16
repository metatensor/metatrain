"""Load a shipped ``lorem-jax`` flax checkpoint into :class:`JaxParityBackbone`
/ :class:`JaxParityLongRange`, leaf by leaf.

Unlike :func:`~metatrain.experimental.lorem.flax_io.apply_flax_params` (which
shape-matches leaves heuristically, for best-effort warm-starting of the
*default* ``experimental.lorem`` architecture), this module assumes the exact
1:1 correspondence documented in :mod:`.jax_parity`'s module docstring and
copies every leaf explicitly. No JAX or flax import is needed here: the
checkpoint's parameter tree must already be a flat ``{"path/to/leaf": array}``
mapping of array-likes (e.g. read via ``flax.serialization.msgpack_restore``
in a separate JAX environment, then flattened and saved to a plain ``.npz`` --
see ``etc/lorem-parity/jax_checkpoint_parity/`` in the
`metawork <https://github.com/EricBoittier/metawork>`_ workspace for a
worked example, including the JAX-side dump script).

Two pieces of the transfer need more than a reshape/transpose:

- The Clebsch-Gordan ``tensor_weight`` (:class:`.tensor_dense.TensorDense` /
  :class:`.tensor_dense.TensorProduct`): flax stores a dense
  ``(l1_max + 1, l2_max + 1, out_max + 1, channels)`` grid (including entries
  for invalid ``(l1, l2, L)`` triples); this port only stores the valid
  triples, in :func:`~.tensor_dense._build_couplings` order, each phase
  -corrected from e3x's CG convention to this codebase's own
  (:func:`~.e3x_compat.cg_phase_correction`).
- ``e3x.nn.Dense`` (:class:`.tensor_dense._DegreeWiseLinear`): one weight
  matrix *per degree*, stored under flax names like ``.../0+/kernel``,
  ``.../1-/kernel``, alternating ``+``/``-`` parity suffixes by degree parity.
"""

from typing import Any, Mapping

import numpy as np
import torch

from .e3x_compat import cg_phase_correction
from .jax_parity import JaxParityBackbone, JaxParityLongRange
from .tensor_dense import _build_couplings


def _t(flax: Mapping[str, Any], name: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(flax[name]))


def _load_dense_wise(
    module: torch.nn.Module, flax: Mapping[str, Any], prefix: str, max_degree: int
) -> None:
    """``_DegreeWiseLinear``: per-degree ``Linear`` from flax
    ``{prefix}/{l}{parity}/kernel`` (``+`` for even ``l``, ``-`` for odd)."""
    for ell in range(max_degree + 1):
        parity = "+" if ell % 2 == 0 else "-"
        kernel = _t(flax, f"{prefix}/{ell}{parity}/kernel")  # (in, out)
        module.layers[ell].weight.data.copy_(kernel.transpose(0, 1))
        bias_name = f"{prefix}/{ell}{parity}/bias"
        if bias_name in flax and module.layers[ell].bias is not None:
            module.layers[ell].bias.data.copy_(_t(flax, bias_name))


def _load_tensor_weight(
    module: torch.nn.Module,
    flax: Mapping[str, Any],
    prefix: str,
    l1_max: int,
    l2_max: int,
    out_max: int,
) -> None:
    """``tensor_weight``: flax's dense ``(l1_max+1, l2_max+1, out_max+1,
    channels)`` grid -> this port's flat ``(n_couplings, channels)``, one row
    per valid ``(l1, l2, L)`` triple in ``_build_couplings`` order,
    phase-corrected from e3x's CG convention to this codebase's own.
    """
    grid = _t(flax, f"{prefix}/kernel")
    # flax stores a rank-6+channel tensor with singleton axes interspersed
    # (e.g. (1, l1_max+1, 1, l2_max+1, 1, out_max+1, channels)); reshape
    # squeezes them out while keeping axis order, since total element count
    # and relative axis order match.
    grid = grid.reshape(l1_max + 1, l2_max + 1, out_max + 1, -1)
    _, l1_list, l2_list, L_list = _build_couplings(
        l1_max, l2_max, out_max, include_pseudotensors=False
    )
    for index, (l1, l2, L) in enumerate(zip(l1_list, l2_list, L_list, strict=True)):
        value = grid[l1, l2, L] * cg_phase_correction(l1, l2, L)
        module.tensor_weight.data[index].copy_(value)


def _load_update(module: torch.nn.Module, flax: Mapping[str, Any], prefix: str) -> None:
    """``Update``: two gated ``MLP([2f, f])`` + ``LayerNorm`` residuals."""
    module.mlp0[0].weight.data.copy_(
        _t(flax, f"{prefix}/MLP_0/Dense_0/kernel").transpose(0, 1)
    )
    module.mlp0[0].bias.data.copy_(_t(flax, f"{prefix}/MLP_0/Dense_0/bias"))
    module.mlp0[2].weight.data.copy_(
        _t(flax, f"{prefix}/MLP_0/Dense_1/kernel").transpose(0, 1)
    )
    module.mlp0[2].bias.data.copy_(_t(flax, f"{prefix}/MLP_0/Dense_1/bias"))
    module.norm0.weight.data.copy_(_t(flax, f"{prefix}/LayerNorm_0/scale"))
    module.norm0.bias.data.copy_(_t(flax, f"{prefix}/LayerNorm_0/bias"))
    module.mlp1[0].weight.data.copy_(
        _t(flax, f"{prefix}/MLP_1/Dense_0/kernel").transpose(0, 1)
    )
    module.mlp1[0].bias.data.copy_(_t(flax, f"{prefix}/MLP_1/Dense_0/bias"))
    module.mlp1[2].weight.data.copy_(
        _t(flax, f"{prefix}/MLP_1/Dense_1/kernel").transpose(0, 1)
    )
    module.mlp1[2].bias.data.copy_(_t(flax, f"{prefix}/MLP_1/Dense_1/bias"))
    module.norm1.weight.data.copy_(_t(flax, f"{prefix}/LayerNorm_1/scale"))
    module.norm1.bias.data.copy_(_t(flax, f"{prefix}/LayerNorm_1/bias"))


def _load_mlp(
    module: torch.nn.Sequential, flax: Mapping[str, Any], prefix: str
) -> None:
    """3-layer ``MLP(features=[d, d, 1])`` (``MLP_0`` / ``MLP_2``)."""
    module[0].weight.data.copy_(_t(flax, f"{prefix}/Dense_0/kernel").transpose(0, 1))
    module[0].bias.data.copy_(_t(flax, f"{prefix}/Dense_0/bias"))
    module[2].weight.data.copy_(_t(flax, f"{prefix}/Dense_1/kernel").transpose(0, 1))
    module[2].bias.data.copy_(_t(flax, f"{prefix}/Dense_1/bias"))
    module[4].weight.data.copy_(_t(flax, f"{prefix}/Dense_2/kernel").transpose(0, 1))
    module[4].bias.data.copy_(_t(flax, f"{prefix}/Dense_2/bias"))


def _load_mlp2(
    module: torch.nn.Sequential, flax: Mapping[str, Any], prefix: str
) -> None:
    """2-layer ``MLP(features=[2 * d, 1])`` (``MLP_1``, the scalar charge head)."""
    module[0].weight.data.copy_(_t(flax, f"{prefix}/Dense_0/kernel").transpose(0, 1))
    module[0].bias.data.copy_(_t(flax, f"{prefix}/Dense_0/bias"))
    module[2].weight.data.copy_(_t(flax, f"{prefix}/Dense_1/kernel").transpose(0, 1))
    module[2].bias.data.copy_(_t(flax, f"{prefix}/Dense_1/bias"))


def load_checkpoint(
    backbone: JaxParityBackbone,
    long_range: JaxParityLongRange,
    flax: Mapping[str, Any],
) -> None:
    """Copy every leaf of a flattened lorem-jax checkpoint into ``backbone``
    and ``long_range``, in place.

    :param backbone: A :class:`JaxParityBackbone` built with hypers matching
        the checkpoint's ``model.yaml`` (``cutoff``, ``max_degree``,
        ``num_features``, ``num_radial``, ``num_spherical_features``).
    :param long_range: The matching :class:`JaxParityLongRange`
        (``max_degree_lr`` matching ``model.yaml``).
    :param flax: A flat mapping from flax parameter path (``/``-separated,
        no leading ``params/``) to array-like, e.g. produced by
        ``flatten_flax_tree`` from a msgpack-restored parameter tree, or
        loaded back from an ``.npz`` written that way.
    """
    max_degree = backbone.max_degree
    max_degree_lr = long_range.max_degree_lr

    with torch.no_grad():
        # -- Initial_0 / ChemicalEmbedding --
        backbone.chemical_embedding.weight.data.copy_(
            _t(flax, "Initial_0/ChemicalEmbedding_0/Embed_0/embedding")
        )

        # -- Dense_0 (species -> nodes_scalar init, with bias) --
        backbone.dense0.weight.data.copy_(_t(flax, "Dense_0/kernel").transpose(0, 1))
        backbone.dense0.bias.data.copy_(_t(flax, "Dense_0/bias"))

        # -- Dense_1 (edges_scalar -> node update, no bias) --
        backbone.dense1.weight.data.copy_(_t(flax, "Dense_1/kernel").transpose(0, 1))

        # -- RadialCoefficients_0/MLP_0 (2 dense layers, SiLU between) --
        backbone.radial_coefficients[0].weight.data.copy_(
            _t(flax, "RadialCoefficients_0/MLP_0/Dense_0/kernel").transpose(0, 1)
        )
        backbone.radial_coefficients[0].bias.data.copy_(
            _t(flax, "RadialCoefficients_0/MLP_0/Dense_0/bias")
        )
        backbone.radial_coefficients[2].weight.data.copy_(
            _t(flax, "RadialCoefficients_0/MLP_0/Dense_1/kernel").transpose(0, 1)
        )
        backbone.radial_coefficients[2].bias.data.copy_(
            _t(flax, "RadialCoefficients_0/MLP_0/Dense_1/bias")
        )

        # -- Update_0 --
        _load_update(backbone.update0, flax, "Update_0")

        # -- Dense_2 (edges_scalar -> per-degree spherical coefficients) --
        backbone.dense2.weight.data.copy_(_t(flax, "Dense_2/kernel").transpose(0, 1))

        # -- TensorDense_0 --
        _load_dense_wise(
            backbone.tensor_dense.dense, flax, "TensorDense_0/dense", max_degree
        )
        _load_tensor_weight(
            backbone.tensor_dense,
            flax,
            "TensorDense_0/tensor",
            max_degree,
            max_degree,
            max_degree,
        )

        # -- Update_1 --
        _load_update(backbone.update1, flax, "Update_1")

        # -- MLP_0 (SR-only energy head) --
        _load_mlp(backbone.energy_mlp, flax, "MLP_0")

        # -- MLP_1 (scalar charge head, 2 dense layers) --
        _load_mlp2(long_range.scalar_charge_mlp, flax, "MLP_1")

        # -- TensorDense_1 (spherical charge head) --
        _load_dense_wise(
            long_range.spherical_charge_dense.dense,
            flax,
            "TensorDense_1/dense",
            max_degree,
        )
        _load_tensor_weight(
            long_range.spherical_charge_dense,
            flax,
            "TensorDense_1/tensor",
            max_degree,
            max_degree,
            max_degree_lr,
        )

        # -- Dense_3 (per-degree potential -> feature-width dense) --
        _load_dense_wise(
            long_range.potential_to_features, flax, "Dense_3", max_degree_lr
        )

        # -- Tensor_0 (potential x nodes_spherical CG mix) --
        _load_tensor_weight(
            long_range.potential_product,
            flax,
            "Tensor_0",
            max_degree_lr,
            max_degree,
            max_degree,
        )

        # -- Update_2 --
        _load_update(long_range.update2, flax, "Update_2")

        # -- MLP_2 (LR energy head) --
        _load_mlp(long_range.energy_mlp, flax, "MLP_2")
