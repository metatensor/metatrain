"""Paper / lorem-jax contracts that ``experimental.lorem`` pins in CI.

This file is the checklist. These tests do **not** import JAX and do **not**
claim bit-exact energies. They check the same equations and knobs as
``lorem.Lorem`` / ``lorem.LoremBEC`` (and the ``sr`` / ``lr`` / ``lr_scale``
module scopes, which are covered elsewhere).

Already tested (do not duplicate):

- acoustic sum rule — ``test_bec.test_bec_acoustic_sum_rule``
- ``lr_scale == 0`` is a no-op — ``test_long_range.test_lr_scale_zero_is_noop``
- children named ``sr`` / ``lr`` — ``test_long_range.test_sr_lr_module_scopes``
- energy rotation invariance —
  ``test_long_range.test_long_range_energy_rotation_invariant``
- ℓ=1 charges rotate as vectors —
  ``test_long_range.test_spherical_charges_rotate_as_vectors``
- ``TensorDense`` scalar rotation invariance —
  ``test_tensor_dense.test_tensor_dense_scalar_is_rotation_invariant``
"""

import pytest
import torch

from metatrain.experimental.lorem.modules.backbone import (
    _cosine_cutoff,
    _degree_norms,
)
from metatrain.experimental.lorem.modules.clebsch_gordan import (
    ClebschGordanReal,
    cg_combine_features,
)
from metatrain.utils.architectures import get_default_hypers

from . import MODEL_HYPERS


# Defaults copied from lorem-jax ``src/lorem/models/mlip.py`` class ``Lorem``.
# CI does not import JAX; keep this table in sync by hand.
LOREM_JAX_LOREM_DEFAULTS = {
    "cutoff": 5.0,
    "max_degree": 6,
    "max_degree_lr": 2,
    "num_features": 128,
    "num_radial": 32,
    "num_species": 8,
    "num_spherical_features": 8,
    "cutoff_fn": "cosine_cutoff",
    "radial_basis": "basic_bernstein",
    "lr": True,
    "num_message_passing": 0,
    "equivariant_message_passing": True,
    "initialize_node_features": True,
}

# Same names and meaning on both sides.
SHARED_HYPER_KEYS = (
    "cutoff",
    "max_degree",
    "max_degree_lr",
    "num_features",
    "num_radial",
    "num_spherical_features",
    "radial_basis",
    "num_message_passing",
)

# lorem-jax name → experimental.lorem path (same role, different spelling).
RENAMED_HYPER_KEYS = {
    "lr": "always on; long-range is a structural part of this architecture, "
    "not a toggle-able hyper",
    "cutoff_fn": "cutoff_width (cosine envelope; onset = cutoff - width)",
}


def _nested_keys(tree, prefix=""):
    keys = []
    if not isinstance(tree, dict):
        return keys
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            keys.extend(_nested_keys(value, name))
        else:
            keys.append(name)
    return keys


def _print_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(row[0]) for row in rows)
    for left, right in rows:
        print(f"  {left:<{width}}  {right}")


def test_default_hypers_match_paper():
    """Paper / ``documentation.py`` defaults (lorem-jax ``Lorem`` knobs)."""
    hypers = get_default_hypers("experimental.lorem")["model"]
    assert hypers == MODEL_HYPERS
    assert hypers["cutoff"] == 5.0
    assert hypers["max_degree"] == 6
    assert hypers["max_degree_lr"] == 2
    assert hypers["num_features"] == 128
    assert hypers["num_radial"] == 32
    assert hypers["num_spherical_features"] == 8
    assert hypers["num_message_passing"] == 0
    assert hypers["radial_basis"] == "basic_bernstein"
    assert hypers["sh_convention"] == "e3x"
    assert hypers["trunk"] == "spherical"


def test_hypers_keys_overlap_lorem_jax():
    """Print the two param dicts. Shared knobs must exist and match defaults."""
    hypers = get_default_hypers("experimental.lorem")["model"]
    torch_keys = set(_nested_keys(hypers))
    jax_keys = set(LOREM_JAX_LOREM_DEFAULTS)

    _print_table(
        "lorem-jax Lorem fields (params dict keys / defaults)",
        [(key, repr(value)) for key, value in LOREM_JAX_LOREM_DEFAULTS.items()],
    )
    _print_table(
        "experimental.lorem model hypers (params dict keys / defaults)",
        [
            (
                key,
                repr(hypers[key]) if not isinstance(hypers[key], dict) else "{...}",
            )
            for key in hypers
        ],
    )
    _print_table(
        "nested experimental.lorem keys",
        [(key, "") for key in sorted(torch_keys)],
    )

    rows = []
    for key in SHARED_HYPER_KEYS:
        jax_value = LOREM_JAX_LOREM_DEFAULTS[key]
        torch_value = hypers[key]
        mark = "match" if jax_value == torch_value else "DIFF"
        rows.append((key, f"jax={jax_value!r}  torch={torch_value!r}  [{mark}]"))
    _print_table("shared architecture knobs", rows)

    jax_only = sorted(jax_keys - set(SHARED_HYPER_KEYS))
    _print_table(
        "lorem-jax-only knobs (role covered, different name or inferred)",
        [(key, RENAMED_HYPER_KEYS.get(key, "see README")) for key in jax_only],
    )

    missing = [key for key in SHARED_HYPER_KEYS if key not in hypers]
    assert missing == [], f"experimental.lorem missing lorem-jax knobs: {missing}"
    for key in SHARED_HYPER_KEYS:
        assert hypers[key] == LOREM_JAX_LOREM_DEFAULTS[key], key


def test_torch_param_keys_follow_sr_lr_scopes():
    """Print named_parameters keys: ``sr`` (short-range) and ``lr``
    (long-range) top-level scopes."""
    pytest.importorskip("torchpme")

    from metatrain.experimental.lorem import LOREM

    from .test_long_range import _energy_dataset_info, _small_lr_hypers

    model = LOREM(
        _small_lr_hypers(max_degree=1, max_degree_lr=1),
        _energy_dataset_info(),
    )
    names = [name for name, _ in model.named_parameters()]
    prefixes = sorted({name.split(".")[0] for name in names})
    _print_table(
        "experimental.lorem top-level param scopes",
        [
            (
                prefix,
                f"{sum(n.startswith(prefix + '.') or n == prefix for n in names)} "
                "leaves",
            )
            for prefix in prefixes
        ],
    )
    _print_table(
        "experimental.lorem named_parameters keys",
        [(name, str(tuple(param.shape))) for name, param in model.named_parameters()],
    )
    assert "sr" in prefixes
    assert "lr" in prefixes
    assert any(name.startswith("lr.scalar_charge_mlp") for name in names)
    assert any(name.startswith("lr.spherical_charge_dense") for name in names)
    assert "lr.lr_scale" in names


def test_degree_norm_factor_is_two_ell_plus_one_to_the_quarter():
    """``l_factors = (2ℓ+1)^{1/4}`` as in lorem-jax ``LoremBEC`` / ``Lorem``."""
    max_degree = 2
    spherical = torch.zeros(1, (max_degree + 1) ** 2, 1)
    # Unit mass in the ℓ=1 block so the raw ℓ-norm is 1.
    spherical[0, 2, 0] = 1.0
    norms = _degree_norms(spherical, max_degree)
    assert norms.shape == (1, max_degree + 1)
    expected = torch.tensor(
        [
            0.0,
            (2.0 * 1.0 + 1.0) ** 0.25,
            0.0,
        ]
    )
    # atol=2e-6: the expected-zero l=2 block hits `_safe_vector_norm`'s eps=1e-12
    # safety term (sqrt(0 + eps) = 1e-6), scaled by that degree's (2*2+1)**0.25
    # factor -- a deterministic ~1.495e-6 floor, not platform noise.
    torch.testing.assert_close(norms[0], expected, atol=2e-6, rtol=1e-6)


def test_cosine_cutoff_is_one_inside_and_zero_at_cutoff():
    """``e3x.nn.functions.cosine_cutoff``: 1 below onset, 0 at the cutoff."""
    cutoff = 5.0
    width = 0.5
    distances = torch.tensor([0.0, 4.4, 4.5, 5.0, 5.2])
    weights = _cosine_cutoff(distances, cutoff, width)
    torch.testing.assert_close(weights[:3], torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights[3:], torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_charge_layout_is_scalar_plus_spherical_lm():
    """lorem-jax equivariant charges: 1 scalar + ``(max_degree_lr+1)²`` (ℓ, m)."""
    pytest.importorskip("torchpme")

    from metatrain.experimental.lorem import LOREM

    from .test_long_range import _energy_dataset_info, _small_lr_hypers

    hypers = _small_lr_hypers(max_degree=1, max_degree_lr=1)
    model = LOREM(hypers, _energy_dataset_info())
    n_atoms = 3
    features = torch.randn(n_atoms, model.num_features)
    n_lm = (int(model.hypers["max_degree"]) + 1) ** 2
    spherical = torch.randn(n_atoms, n_lm, int(model.hypers["num_spherical_features"]))
    charges = model.lr.map_charges(features, spherical)
    max_degree_lr = int(model.hypers["max_degree_lr"])
    assert charges.shape == (n_atoms, 1 + (max_degree_lr + 1) ** 2)


def test_cg_one_otimes_one_to_scalar_is_dot_product():
    """e3x / lorem-jax ``TensorDense``: ``1 ⊗ 1 → 0`` is the invariant."""
    cg = ClebschGordanReal()
    coeff = cg.get((1, 1, 0)).to(torch.float64)
    vectors = torch.tensor(
        [
            [[1.0], [0.0], [0.0]],
            [[0.0], [2.0], [0.0]],
            [[0.0], [0.0], [3.0]],
            [[1.0], [1.0], [1.0]],
        ],
        dtype=torch.float64,
    )
    coupled = cg_combine_features(vectors, vectors, coeff)[:, 0, 0]
    dots = (vectors[:, :, 0] ** 2).sum(dim=-1)
    ratio = coupled / dots
    torch.testing.assert_close(ratio, ratio[0].expand_as(ratio), atol=1e-8, rtol=1e-8)
    assert torch.all(ratio.abs() > 0)
