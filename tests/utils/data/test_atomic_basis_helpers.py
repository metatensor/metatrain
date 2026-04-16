import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_target,
    sparsify_atomic_basis_target,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_layout_coupled_per_atom():
    """
    Layout TensorMap (zero-sample blocks) defining the coupled-spherical-harmonic basis
    for H (type 1) and C (type 6).
    """
    empty = Labels(
        names=["system", "atom"],
        values=torch.empty((0, 2), dtype=torch.int32),
    )
    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor(
                [[0, 1, 1], [0, 1, 6], [1, 1, 1], [1, 1, 6]], dtype=torch.int32
            ),
        ),
        blocks=[
            TensorBlock(  # l=0, H: n=[0,1] - seeds the union at positions 0,1
                values=torch.empty((0, 1, 2), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=0, C: n=[2,3] - ends up at positions 2,3 in union
                values=torch.empty((0, 1, 2), dtype=torch.float64),
                samples=empty,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[2], [3]], dtype=torch.int32),
                ),
            ),
            TensorBlock(  # l=1, H: n=[0] - seeds the union at position 0
                values=torch.empty((0, 3, 1), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(  # l=1, C: n=[1] - disjoint; ends up at position 1 in union
                values=torch.empty((0, 3, 1), dtype=torch.float64),
                samples=empty,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[1]], dtype=torch.int32)
                ),
            ),
        ],
    )


def _make_systems():
    """
    Two systems:
      - System 0: 2 atoms  [H (atom 0), C (atom 1)]
      - System 1: 3 atoms  [H (atom 0), H (atom 1), C (atom 2)]
    """
    cell = torch.zeros((3, 3), dtype=torch.float64)
    pbc = torch.zeros(3, dtype=torch.bool)
    return [
        System(
            types=torch.tensor([1, 6]),
            positions=torch.zeros((2, 3), dtype=torch.float64),
            cell=cell,
            pbc=pbc,
        ),
        System(
            types=torch.tensor([1, 1, 6]),
            positions=torch.zeros((3, 3), dtype=torch.float64),
            cell=cell,
            pbc=pbc,
        ),
    ]


def _make_sparse_tensor():
    """
    Construct an atomic basis spherical target in the coupled basis, per-atom, for the
    two systems from ``_make_systems()``.

    Samples within each block are sorted by (system, atom) – this is the ordering that
    densification produces, so starting from sorted samples ensures the round-trip
    comparison is straightforward.
    """
    h_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32),
    )
    c_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    l0_comp = [Labels(names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32))]
    l1_comp = [
        Labels(
            names=["o3_mu"],
            values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
        )
    ]

    # Use easily distinguishable values to make assertion failures informative. H and C
    # have disjoint property indices, matching the layout above.
    h_l0 = torch.arange(6, dtype=torch.float64).reshape(
        3, 1, 2
    )  # (3 atoms, 1 comp, n=[0,1])
    c_l0 = torch.arange(100, 104, dtype=torch.float64).reshape(
        2, 1, 2
    )  # (2 atoms, 1 comp, n=[2,3])
    h_l1 = (
        torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) + 200.0
    )  # (3 atoms, 3 comps, n=[0])
    c_l1 = (
        torch.arange(6, dtype=torch.float64).reshape(2, 3, 1) + 300.0
    )  # (2 atoms, 3 comps, n=[1])

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor(
                [[0, 1, 1], [0, 1, 6], [1, 1, 1], [1, 1, 6]], dtype=torch.int32
            ),
        ),
        blocks=[
            TensorBlock(
                values=h_l0,
                samples=h_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0], [1]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=c_l0,
                samples=c_samples,
                components=l0_comp,
                properties=Labels(
                    names=["n"],
                    values=torch.tensor([[2], [3]], dtype=torch.int32),
                ),
            ),
            TensorBlock(
                values=h_l1,
                samples=h_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[0]], dtype=torch.int32)
                ),
            ),
            TensorBlock(
                values=c_l1,
                samples=c_samples,
                components=l1_comp,
                properties=Labels(
                    names=["n"], values=torch.tensor([[1]], dtype=torch.int32)
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_densify_sparsify_round_trip_():
    """
    Densifying then sparsifying (in atom type) a sparse atomic basis TensorMap should
    exactly recover the original TensorMap.

    This test is the primary regression guard for a class of bug where the arguments to
    ``Labels.select`` are swapped in the sparsification step. With the bug, ``sparsify``
    returns the first *n* columns of the union-property block instead of the columns
    that actually belong to each atom type.  For atom types whose properties do not
    begin at position 0 in the union (here, all C blocks and all H covariant blocks),
    the round-trip would silently return the wrong values, causing the characteristic
    crossing pattern in parity plots.
    """
    layout = _make_layout_coupled_per_atom()
    sparse = _make_sparse_tensor()
    systems = _make_systems()

    dense = densify_atomic_basis_target(sparse, layout)
    recovered = sparsify_atomic_basis_target(systems, dense, layout)

    mts.allclose_raise(recovered, sparse, atol=0.0, rtol=0.0)


def test_densify_nan_padding_structure():
    """
    The intermediate densified TensorMap must have the correct NaN-padded structure:
    property slots belonging to an atom type carry the original values, while property
    slots that *don't* belong to that type are NaN.

    After densification, samples are sorted by (system, atom).  For our two systems [H,
    C] and [H, H, C] this gives:

      index 0: (sys=0, atom=0) → H
      index 1: (sys=0, atom=1) → C
      index 2: (sys=1, atom=0) → H
      index 3: (sys=1, atom=1) → H
      index 4: (sys=1, atom=2) → C

    Scalar block (l=0), union n=[0,1,2,3]:
      - H atoms (layout n=[0,1]): real values at positions 0,1 ; NaN at positions 2,3
      - C atoms (layout n=[2,3]): NaN at positions 0,1 ; real values at positions 2,3

    Covariant block (l=1), union n=[0,1]:
      - H atoms (layout n=[0]): real value at position 0 ; NaN at position 1
      - C atoms (layout n=[1]): NaN at position 0 ; real value at position 1

    The disjoint property sets mean C's real values are placed at the *end* of the
    union, not the start.  This is precisely the structural condition that the
    ``Labels.select`` argument-swap bug fails to handle: it would return positions [0,1]
    for C (the indices within C's own label set) rather than [2,3] (the positions of C's
    labels within the union), overwriting C's values with H's.

    This test validates the densification direction independently of sparsification and
    confirms that property padding is both correct in placement and preserves the
    original values at the occupied positions.
    """
    layout = _make_layout_coupled_per_atom()
    sparse = _make_sparse_tensor()

    dense = densify_atomic_basis_target(sparse, layout)

    # Keys no longer contain atom_type after densification
    assert "atom_type" not in dense.keys.names
    assert dense.keys.names == ["o3_lambda", "o3_sigma"]

    # Sample index sets (see docstring for ordering rationale)
    h_idx = torch.tensor([0, 2, 3])
    c_idx = torch.tensor([1, 4])

    # --- scalar block (l=0): union properties n=[0,1,2,3] ---
    l0 = dense.block({"o3_lambda": 0, "o3_sigma": 1})
    assert l0.values.shape == (5, 1, 4)

    # H atoms: real at union positions 0,1 (n=0,1); NaN at positions 2,3 (n=2,3)
    h_l0_orig = torch.arange(6, dtype=torch.float64).reshape(3, 1, 2)
    torch.testing.assert_close(l0.values[h_idx][..., :2], h_l0_orig)
    assert torch.all(torch.isnan(l0.values[h_idx][..., 2:]))

    # C atoms: NaN at union positions 0,1 (n=0,1); real at positions 2,3 (n=2,3)
    c_l0_orig = torch.arange(100, 104, dtype=torch.float64).reshape(2, 1, 2)
    assert torch.all(torch.isnan(l0.values[c_idx][..., :2]))
    torch.testing.assert_close(l0.values[c_idx][..., 2:], c_l0_orig)

    # --- covariant block (l=1): union properties n=[0,1] ---
    l1 = dense.block({"o3_lambda": 1, "o3_sigma": 1})
    assert l1.values.shape == (5, 3, 2)

    # H atoms: real at union position 0 (n=0); NaN at position 1 (n=1)
    h_l1_orig = torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) + 200.0
    torch.testing.assert_close(l1.values[h_idx][..., :1], h_l1_orig)
    assert torch.all(torch.isnan(l1.values[h_idx][..., 1:]))

    # C atoms: NaN at union position 0 (n=0); real at position 1 (n=1)
    c_l1_orig = torch.arange(6, dtype=torch.float64).reshape(2, 3, 1) + 300.0
    assert torch.all(torch.isnan(l1.values[c_idx][..., :1]))
    torch.testing.assert_close(l1.values[c_idx][..., 1:], c_l1_orig)
