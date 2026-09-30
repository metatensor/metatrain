# experimental.lorem

Developer notes. User-facing hypers live in [`documentation.py`](documentation.py).

LOREM (*Learning Long-Range Representations with Equivariant Messages*,
[arXiv:2507.19382](https://arxiv.org/abs/2507.19382)) is a short-range
spherical model plus long-range Coulomb features from learned charges
(`torch-pme`).

- [`modules/jax_parity.py`](modules/jax_parity.py): short-range density
  (`sr`) and long-range Coulomb block (`lr`), including message passing
  when `num_message_passing > 0`. Long-range is always on.
- [`modules/bec.py`](modules/bec.py): per-atom 3×3 Born effective charges,
  one head per short-range stage plus one on the long-range spherical mix.
  Needs `max_degree >= 2`.

`tests/` checks behavior that should stay stable (defaults, equivariance,
one training step, TorchScript, regression numbers). Install with
`pip install 'metatrain[lorem]'`.
