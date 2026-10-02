"""``TensorDense``/``TensorProduct`` are bit-identical to ``e3x``'s.

Loads a real ``e3x.nn.TensorDense`` / ``e3x.nn.Tensor`` parameter tree and
its output for a fixed random input (dumped once from a genuine ``e3x``
install into ``assets/e3x_tensor_dense/``; this test does **not** need
JAX/e3x installed to run), transfers the weights into this codebase's
``TensorDense``/``TensorProduct``, and asserts the two agree to float32
precision.

This exercises the two convention differences between e3x and this codebase:

1. These fixtures were dumped with e3x's ``Config.cartesian_order``, whose
   component order within each degree differs from LOREM's ``m = -l ... +l``
   by a fixed per-degree permutation (``_to_e3x`` / ``_from_e3x`` below).
   lorem-jax also runs in that order, but its kernels are per degree and its
   spherical harmonics and Clebsch-Gordan coefficients share the permutation,
   so whole-model outputs (``test_flax_checkpoint.py``) need no reordering.
2. The ``wigners``-based Clebsch-Gordan coefficients differ from e3x's by a
   sign ``(-1)**((l1+l2-L)//2)`` (``cg_phase_correction``), which
   ``dense_clebsch_gordan`` folds into the buffers, so e3x kernels copy as is.

Checkpoint-parity in spirit: load a checkpoint dumped from the reference
implementation, run both on identical inputs, assert the outputs agree --
adapted here to not require the reference framework (JAX/e3x) to be
installed at test time, since the fixture is dumped ahead of time instead
of generated on the fly.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from metatrain.experimental.lorem.modules.clebsch_gordan import cg_phase_correction
from metatrain.experimental.lorem.modules.tensor_dense import TensorDense, TensorProduct


ASSETS = Path(__file__).parent / "assets" / "e3x_tensor_dense"


def _load(name):
    return np.load(ASSETS / name)


def _e3x_order(max_degree):
    """Position in e3x's cartesian order of each LOREM ``(ℓ, m)`` row: degree
    1 is ``(y, z, x)`` -> ``(x, y, z)``, and higher degrees follow the same
    interleaving."""
    order = []
    for ell in range(max_degree + 1):
        within = [*range(1, 2 * ell, 2), 2 * ell, *range(2 * ell - 2, -1, -2)]
        order.extend(ell * ell + i for i in within)
    return torch.tensor(order)


def _from_e3x(x, max_degree):
    return x[:, _e3x_order(max_degree), :]


def _to_e3x(x, max_degree):
    out = torch.empty_like(x)
    out[:, _e3x_order(max_degree), :] = x
    return out


@pytest.mark.skipif(not ASSETS.is_dir(), reason=f"missing fixtures in {ASSETS}")
class TestTensorDenseParity:
    """``max_degree=2``, ``in_features=4``, ``out_features=3``."""

    @pytest.fixture
    def data(self):
        return _load("tensor_dense_reference.npz")

    @pytest.fixture
    def max_degree(self):
        return 2

    @pytest.fixture
    def torch_output(self, data, max_degree):
        td = TensorDense(
            in_features=4,
            out_features=3,
            in_max_degree=max_degree,
            out_max_degree=max_degree,
            use_bias=False,
        )
        with torch.no_grad():
            for ell, key in enumerate(("dense_0", "dense_1", "dense_2")):
                td.dense.weight[ell].copy_(torch.from_numpy(data[key]).float())
            kernel = data["tensor_kernel"][0, :, 0, :, 0]
            td.tensor_weight.copy_(torch.from_numpy(kernel).float())

        x_lorem = _from_e3x(torch.from_numpy(data["input"]).float(), max_degree)
        return _to_e3x(td(x_lorem), max_degree).detach().numpy()

    def test_couplings_are_the_proper_triples(self, max_degree):
        """Nonzero Clebsch-Gordan blocks are exactly the couplings of two proper
        tensors into a proper one: ``|l1 - l2| <= L <= l1 + l2``, ``l1+l2+L`` even."""
        td = TensorDense(4, 3, max_degree, max_degree)
        n = max_degree + 1
        blocks = {
            (l1, l2, L)
            for l1 in range(n)
            for l2 in range(n)
            for L in range(n)
            if td.cg[
                l1**2 : (l1 + 1) ** 2, l2**2 : (l2 + 1) ** 2, L**2 : (L + 1) ** 2
            ].any()
        }
        expected = {
            (l1, l2, L)
            for l1 in range(n)
            for l2 in range(n)
            for L in range(abs(l1 - l2), min(l1 + l2, max_degree) + 1)
            if (l1 + l2 + L) % 2 == 0
        }
        assert blocks == expected

    def test_output_matches_e3x(self, data, torch_output):
        np.testing.assert_allclose(torch_output, data["output"], atol=2e-5, rtol=1e-4)


@pytest.mark.skipif(not ASSETS.is_dir(), reason=f"missing fixtures in {ASSETS}")
def test_tensor_product_matches_e3x_with_nontrivial_phase():
    """Two ``max_degree=1`` inputs coupled down to ``L=0``: the couplings are
    ``(0,0,0)`` (phase +1) and ``(1,1,0)`` (phase -1, confirmed above) --
    catches a regression that silently drops the phase correction, since
    getting it wrong on just the ``(1,1,0)`` term still leaves the ``(0,0,0)``
    term matching and would pass a less targeted test."""
    data = _load("tensor_product_1_1_0_reference.npz")
    assert cg_phase_correction(1, 1, 0) == -1.0
    max_degree = 1

    tp = TensorProduct(
        left_max_degree=max_degree,
        right_max_degree=max_degree,
        out_max_degree=0,
        n_features=2,
    )
    with torch.no_grad():
        tp.tensor_weight.copy_(torch.from_numpy(data["kernel"][0, :, 0, :, 0]).float())

    left = _from_e3x(torch.from_numpy(data["left"]).float(), max_degree)
    right = _from_e3x(torch.from_numpy(data["right"]).float(), max_degree)
    out = _to_e3x(tp(left, right), 0).detach().numpy()
    np.testing.assert_allclose(out, data["output"], atol=2e-5, rtol=1e-4)


class TestPhaseCorrection:
    def test_known_values(self):
        # From the exhaustive check (42/42 triples up to degree 4) that
        # produced this formula -- a fixed set of regression anchors.
        assert cg_phase_correction(0, 0, 0) == 1.0
        assert cg_phase_correction(1, 1, 0) == -1.0
        assert cg_phase_correction(1, 1, 2) == 1.0
        assert cg_phase_correction(1, 2, 1) == -1.0
        assert cg_phase_correction(2, 2, 0) == 1.0
        assert cg_phase_correction(2, 2, 2) == -1.0

    def test_rejects_invalid_triple(self):
        with pytest.raises(ValueError):
            cg_phase_correction(1, 0, 0)  # L < |l1 - l2|
