"""``TensorDense``/``TensorProduct`` are bit-identical to ``e3x``'s.

Loads a real ``e3x.nn.TensorDense`` / ``e3x.nn.Tensor`` parameter tree and
its output for a fixed random input (dumped once from a genuine ``e3x``
install into ``assets/e3x_tensor_dense/`` -- see the generation script in
that fixture's own header comment; this test does **not** need JAX/e3x
installed to run), transfers the weights into this codebase's
``TensorDense``/``TensorProduct``, and asserts the two agree to float32
precision.

This exercises the two real convention differences between lorem-jax (e3x)
and this codebase, both handled by ``e3x_compat.py``:

1. e3x's ``Config.cartesian_order`` component ordering within each degree
   block differs from LOREM's own ``m = -l ... +l`` ordering by a fixed,
   per-degree signed permutation (``to_e3x_convention`` / ``from_e3x_convention``).
2. LOREM's own (``wigners``-based) Clebsch-Gordan coefficients differ from
   e3x's own by an extra sign ``(-1)**((l1+l2-L)//2)`` once both are expressed
   in the same basis (``cg_phase_correction``) -- a residual real/complex
   phase-convention difference between the two independently-built CG
   generators, unrelated to the permutation.

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

from metatrain.experimental.lorem.modules import e3x_compat
from metatrain.experimental.lorem.modules.tensor_dense import TensorDense, TensorProduct


ASSETS = Path(__file__).parent / "assets" / "e3x_tensor_dense"


def _load(name):
    return np.load(ASSETS / name)


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
            include_pseudotensors=False,
            use_bias=False,
        )
        with torch.no_grad():
            for ell, key in enumerate(("dense_0", "dense_1", "dense_2")):
                # e3x Dense kernel is (in, out); torch Linear.weight is (out, in).
                td.dense.layers[ell].weight.copy_(torch.from_numpy(data[key].T).float())
            for index, coupling in enumerate(td.couplings):
                l1, l2, L = coupling.l1, coupling.l2, coupling.L
                weight = data["tensor_kernel"][0, l1, 0, l2, 0, L, :]
                td.tensor_weight[index].copy_(torch.from_numpy(weight).float())
                sign = e3x_compat.cg_phase_correction(l1, l2, L)
                if sign < 0:
                    coupling.cg.mul_(-1.0)

        x_lorem = e3x_compat.from_e3x_convention(
            torch.from_numpy(data["input"]).float(), max_degree
        )
        return e3x_compat.to_e3x_convention(td(x_lorem), max_degree).detach().numpy()

    def test_couplings_cover_every_valid_triple(self, max_degree):
        td = TensorDense(4, 3, max_degree, max_degree, include_pseudotensors=False)
        found = {(c.l1, c.l2, c.L) for c in td.couplings}
        assert found == set(e3x_compat.cg_couplings(max_degree, max_degree))

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
    assert e3x_compat.cg_phase_correction(1, 1, 0) == -1.0
    max_degree = 1

    tp = TensorProduct(
        left_max_degree=max_degree,
        right_max_degree=max_degree,
        out_max_degree=0,
        include_pseudotensors=False,
        n_features=2,
    )
    with torch.no_grad():
        for index, coupling in enumerate(tp.couplings):
            l1, l2, L = coupling.l1, coupling.l2, coupling.L
            weight = data["kernel"][0, l1, 0, l2, 0, L, :]
            tp.tensor_weight[index].copy_(torch.from_numpy(weight).float())
            if e3x_compat.cg_phase_correction(l1, l2, L) < 0:
                coupling.cg.mul_(-1.0)

    left = e3x_compat.from_e3x_convention(
        torch.from_numpy(data["left"]).float(), max_degree
    )
    right = e3x_compat.from_e3x_convention(
        torch.from_numpy(data["right"]).float(), max_degree
    )
    out = e3x_compat.to_e3x_convention(tp(left, right), 0).detach().numpy()
    np.testing.assert_allclose(out, data["output"], atol=2e-5, rtol=1e-4)


class TestE3xCompatUtilities:
    """Pure-Python checks on the hardcoded permutation / phase tables."""

    @pytest.mark.parametrize("max_degree", range(e3x_compat.max_supported_degree() + 1))
    def test_permutation_is_involution(self, max_degree):
        """``to_e3x_convention`` and ``from_e3x_convention`` must invert each other."""
        x = torch.randn(2, (max_degree + 1) ** 2, 3)
        roundtrip = e3x_compat.from_e3x_convention(
            e3x_compat.to_e3x_convention(x, max_degree), max_degree
        )
        torch.testing.assert_close(roundtrip, x)

    def test_cg_phase_correction_known_values(self):
        # From the exhaustive check (42/42 triples up to degree 4) that
        # produced this formula -- a fixed set of regression anchors.
        assert e3x_compat.cg_phase_correction(0, 0, 0) == 1.0
        assert e3x_compat.cg_phase_correction(1, 1, 0) == -1.0
        assert e3x_compat.cg_phase_correction(1, 1, 2) == 1.0
        assert e3x_compat.cg_phase_correction(1, 2, 1) == -1.0
        assert e3x_compat.cg_phase_correction(2, 2, 0) == 1.0
        assert e3x_compat.cg_phase_correction(2, 2, 2) == -1.0

    def test_cg_phase_correction_rejects_invalid_triple(self):
        with pytest.raises(ValueError):
            e3x_compat.cg_phase_correction(1, 0, 0)  # l1+l2+L odd
