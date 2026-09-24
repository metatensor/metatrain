# experimental.lorem developer notes

Developer notes for `experimental.lorem`. User-facing hypers and install
instructions stay in [`documentation.py`](documentation.py).

This architecture is a paper-shaped PyTorch / TorchScript port of LOREM
(*Learning Long-Range Representations with Equivariant Messages*,
[arXiv:2507.19382](https://arxiv.org/abs/2507.19382)). The official JAX
reference implementation is
[lorem-jax](https://github.com/lab-cosmo/lorem-jax) (`lorem.Lorem`,
`lorem.LoremBEC`, `e3x.nn.TensorDense`), available as a git submodule for
local development:

```bash
git submodule update --init lorem-jax
```

## What this package implements

- [`LoremBackbone`](modules/backbone.py) (`sr`): `basic_bernstein` radial
  basis x real spherical harmonics (lorem-jax defaults; `bessel` /
  `orthonormal` remain selectable), then
  [`TensorDense`](modules/tensor_dense.py) (the paper's Clebsch-Gordan
  self-product). Optional equivariant message passing. `trunk: pet` mounts
  metatrain's own `PETBackend` as the short-range trunk instead, with a
  spherical sidecar for equivariant charges / BEC.
- [`LoremLongRangeFeaturizer`](modules/long_range.py) (`lr`): scalar
  charge MLP + `TensorDense` spherical charges; torch-pme Ewald / P3M /
  direct Coulomb evaluation; CG [`TensorProduct`](modules/tensor_dense.py)
  mix back into node features, gated by a learnable `lr_scale`. This is
  always on: it is a core part of the architecture, not an optional
  add-on.
- Dipole head: PhysNet-style :math:`\mu = \sum_i q_i r_i` from a learned
  per-atom charge times position (system or per-atom Cartesian rank-1).
- [`BornEffectiveChargeHead`](modules/bec.py): `LoremBEC` /
  `PerParticleTensorPredictor` — Dense+SiLU, `TensorDense` to
  :math:`\ell \le 2`, CG reconstruction to a 3x3, acoustic sum rule.
  Requires `max_degree >= 2` and a per-atom Cartesian rank-2 target.
- Composition + `Scaler` additives on scalar targets.

This is a metatrain-native port: same equations and knobs as the paper /
lorem-jax, not a bit-exact JAX clone (different spherical-harmonics
convention, PME implementation, and RNG). It uses the paper's equations
and knobs, plus metatrain-style `sr` / `lr` module scopes.

## Checking parity

Three layers:

1. **In-repo contracts** (this package's tests, no JAX). Checklist:
   [`tests/test_paper_contracts.py`](tests/test_paper_contracts.py).
   `test_hypers_keys_overlap_lorem_jax` prints both param-dict key
   tables; `test_torch_param_keys_follow_sr_lr_scopes` prints the
   `named_parameters` tree (`sr` / `lr`).
2. **External train / eval** (optional, not in this repository): run
   `mtt` on one side and the `lorem-jax` examples (in a separate JAX
   venv) on the other, and compare.
3. **This README** — what is and is not claimed.

| Symbol | Test | Source |
| --- | --- | --- |
| paper default hypers | `test_default_hypers_match_paper` | `documentation.py` |
| printed lorem-jax vs torch keys | `test_hypers_keys_overlap_lorem_jax` | `lorem.Lorem` fields |
| printed `sr` / `lr` param tree | `test_torch_param_keys_follow_sr_lr_scopes` | `named_parameters` scopes |
| one train step, no NaNs | `test_one_training_step_is_finite` | energy / forces / stress / dipole / BEC x batch 1 and 4 |
| isolated-atom force step | `test_force_step_with_isolated_atom_is_finite` | empty neighbor list + forces |
| `l_factors = (2l+1)^{1/4}` | `test_degree_norm_factor_is_two_ell_plus_one_to_the_quarter` | lorem-jax `Lorem` / `LoremBEC` |
| cosine cutoff | `test_cosine_cutoff_is_one_inside_and_zero_at_cutoff` | e3x `cosine_cutoff` |
| `1 + (L_lr+1)^2` charges | `test_charge_layout_is_scalar_plus_spherical_lm` | lorem-jax equivariant charges |
| `1 tensor 1 -> 0` | `test_cg_one_otimes_one_to_scalar_is_dot_product` | e3x `TensorDense` |
| acoustic sum rule | `test_bec_acoustic_sum_rule` | `lorem.LoremBEC` |
| `lr_scale == 0` no-op | `test_lr_scale_zero_is_noop` | zero perturbation warm-start |
| `sr` / `lr` scopes | `test_sr_lr_module_scopes` | top-level module scopes |
| energy / l=1 rotation | `test_long_range_energy_rotation_invariant`, `test_spherical_charges_rotate_as_vectors` | paper equivariance |
| `TensorDense` scalars | `test_tensor_dense_scalar_is_rotation_invariant` | e3x `TensorDense` |

CI does not import JAX, does not store golden JAX energies, and does not
claim bit-exact numerical parity with the JAX reference.

## Stack

`experimental.lorem` needs `pip install 'metatrain[lorem]'`
(`torch-pme`, `sphericart-torch`, `wigners`). `lorem-jax` is a separate
JAX install (see its own README); do not pip-install JAX packages into
the shared metatrain venv.
