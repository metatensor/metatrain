# experimental.lorem

Developer notes. User-facing hypers live in [`documentation.py`](documentation.py).

LOREM (*Learning Long-Range Representations with Equivariant Messages*,
[arXiv:2507.19382](https://arxiv.org/abs/2507.19382)) is a short-range
spherical model plus long-range Coulomb features from learned charges
(`torch-pme`).

- [`modules/lorem.py`](modules/lorem.py): short-range density
  (`sr`) and long-range Coulomb block (`lr`), including message passing
  when `num_message_passing > 0`. Long-range is always on.
- [`modules/bec.py`](modules/bec.py): per-atom 3×3 Born effective charges,
  one head per short-range stage plus one on the long-range spherical mix.
  Needs `max_degree >= 2`.
- [`modules/flax_checkpoint.py`](modules/flax_checkpoint.py): loads a
  lorem-jax export (flat flax parameters, `model.yaml`, `baseline.yaml`;
  lab-cosmo/lorem-jax#39) into a LOREM, without JAX. Its tests compare
  against lorem-jax outputs in `tests/assets/lorem_jax/`, written by
  `generate.py` there.

`tests/` checks behavior that should stay stable (defaults, equivariance,
one training step, TorchScript, regression numbers). Install with
`pip install 'metatrain[lorem]'`.
