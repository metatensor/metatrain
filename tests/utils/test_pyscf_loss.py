import importlib

import pytest
import torch
from metatomic.torch import System

from metatrain.utils.pyscf_loss import (
    _load_auxiliary_basis,
    build_auxiliary_molecule,
    compute_batched_coulomb_matrices,
    compute_batched_overlap_matrices,
    compute_coulomb_matrix,
    compute_overlap_matrix,
    metric_matrix_name,
    pack_two_center_matrices,
    resolve_ri_aux_basis,
    unpack_two_center_matrices,
)


pytest.importorskip("pyscf")


def _make_systems() -> list[System]:
    cell = torch.zeros((3, 3), dtype=torch.float64)
    pbc = torch.zeros(3, dtype=torch.bool)
    return [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([1], dtype=torch.int32),
            cell=cell,
            pbc=pbc,
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                dtype=torch.float64,
            ),
            types=torch.tensor([8, 1, 1], dtype=torch.int32),
            cell=cell,
            pbc=pbc,
        ),
    ]


def test_build_auxiliary_molecule_and_overlap_matrix():
    system = _make_systems()[1]

    auxmol = build_auxiliary_molecule(system, "def2-svp-jkfit")
    S = compute_overlap_matrix(system, "def2-svp-jkfit")

    assert auxmol.nao == S.shape[0] == S.shape[1]
    torch.testing.assert_close(S, S.T)
    assert torch.linalg.eigvalsh(S).min().item() > 0.0


def test_build_auxiliary_molecule_and_coulomb_matrix():
    system = _make_systems()[1]

    auxmol = build_auxiliary_molecule(system, "def2-svp-jkfit")
    J = compute_coulomb_matrix(system, "def2-svp-jkfit")

    assert auxmol.nao == J.shape[0] == J.shape[1]
    torch.testing.assert_close(J, J.T)
    assert torch.linalg.eigvalsh(J).min().item() > 0.0


def test_pack_unpack_two_center_matrices_round_trip():
    matrices = [
        torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64),
        torch.tensor([[3.0]], dtype=torch.float64),
    ]

    packed = pack_two_center_matrices(matrices)
    torch.testing.assert_close(
        packed.block().samples.values[:, 1], torch.tensor([2, 1], dtype=torch.int32)
    )
    unpacked = unpack_two_center_matrices(packed, [2, 1])

    for expected, actual in zip(matrices, unpacked, strict=True):
        torch.testing.assert_close(actual, expected)


def test_unpack_two_center_matrices_rejects_basis_mismatch():
    packed = pack_two_center_matrices([torch.eye(2, dtype=torch.float64)])

    with pytest.raises(ValueError, match="RI target size does not match"):
        unpack_two_center_matrices(packed, [1])


def test_compute_batched_overlap_matrices_matches_individual_results():
    systems = _make_systems()

    packed = compute_batched_overlap_matrices(systems, "def2-svp-jkfit")
    expected = [compute_overlap_matrix(system, "def2-svp-jkfit") for system in systems]
    actual = unpack_two_center_matrices(
        packed, [matrix.shape[0] for matrix in expected]
    )

    for expected_matrix, actual_matrix in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_matrix, expected_matrix)


def test_compute_batched_coulomb_matrices_matches_individual_results():
    systems = _make_systems()

    packed = compute_batched_coulomb_matrices(systems, "def2-svp-jkfit")
    expected = [compute_coulomb_matrix(system, "def2-svp-jkfit") for system in systems]
    actual = unpack_two_center_matrices(
        packed, [matrix.shape[0] for matrix in expected]
    )

    for expected_matrix, actual_matrix in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_matrix, expected_matrix)


def test_metric_matrix_name_dispatches_correctly():
    assert metric_matrix_name("tgt", "overlap") == "tgt_overlap_matrix"
    assert metric_matrix_name("tgt", "coulomb") == "tgt_coulomb_matrix"

    with pytest.raises(ValueError, match="Unknown RI metric"):
        metric_matrix_name("tgt", "euclidean")


def test_auxiliary_basis_cache_reuses_parsed_basis(monkeypatch):
    gto, _ = (
        importlib.import_module("pyscf.gto"),
        importlib.import_module("pyscf.data.elements"),
    )
    original_load = gto.basis.load
    calls: list[tuple[str, str]] = []

    def counted_load(aux_basis: str, symbol: str):
        calls.append((aux_basis, symbol))
        return original_load(aux_basis, symbol)

    _load_auxiliary_basis.cache_clear()
    monkeypatch.setattr(gto.basis, "load", counted_load)

    system = _make_systems()[1]
    compute_overlap_matrix(system, "def2-svp-jkfit")
    compute_overlap_matrix(system, "def2-svp-jkfit")

    assert sorted(calls) == [("def2-svp-jkfit", "H"), ("def2-svp-jkfit", "O")]


def test_resolve_ri_aux_basis():
    assert resolve_ri_aux_basis("target", "basis") == "basis"
    assert resolve_ri_aux_basis("target", {"target": "basis"}) == "basis"

    with pytest.raises(ValueError, match="No RI auxiliary basis configured"):
        resolve_ri_aux_basis("target", {"other": "basis"})


def test_missing_pyscf_dependency(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name.startswith("pyscf"):
            raise ModuleNotFoundError(name)
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="require `pyscf`"):
        compute_overlap_matrix(_make_systems()[0], "def2-svp-jkfit")
