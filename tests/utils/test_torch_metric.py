"""
Tests for the torch metric backend of the density losses.

The backend re-derives everything PySCF-conventional numerically (basis fit,
AO ordering) and rebuilds the metric matrices from closed forms, so the tests
here are agreement tests against the PySCF reference path: matrices, packed
geometry plumbing, and the losses evaluated end to end through both backends.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.density_hooks import get_density_hooks
from metatrain.utils.loss import DensityMSELossViaC, DensityMSELossViaW
from metatrain.utils.pyscf_loss import (
    DENSITY_GEOMETRY_NAME,
    compute_coulomb_matrix,
    compute_metric_matrix,
    compute_overlap_matrix,
    get_density_geometry_transform,
    pack_density_geometry,
    ri_density_fit_constant_name,
    ri_projections_name,
    unpack_density_geometry,
)
from metatrain.utils.torch_metric import TorchMetricBuilder

from .test_pyscf_loss import (
    AUX_BASIS,
    TARGET,
    O3Augmenter,
    _batch,
    _ethane,
    _extra,
    _flatten_to_pyscf_order,
    _random_target,
    _rotation,
    _system,
    _target_info,
    _unflatten_like,
)


pyscf = pytest.importorskip("pyscf")

CPU = torch.device("cpu")


def _geometry(system) -> tuple:
    return (system.types, system.positions)


def _relative_error(ours: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.norm(ours - reference) / torch.linalg.norm(reference))


# ── matrices against PySCF ────────────────────────────────────────────────────


@pytest.mark.parametrize("metric", ["overlap", "coulomb"])
def test_matrices_match_pyscf(metric):
    builder = TorchMetricBuilder(AUX_BASIS, metric)
    for system in (_system(), _ethane()):
        reference = (
            compute_overlap_matrix(system, AUX_BASIS)
            if metric == "overlap"
            else compute_coulomb_matrix(system, AUX_BASIS)
        )
        ours = builder.compute_batch([_geometry(system)], CPU)[0]
        assert _relative_error(ours, reference) < 1e-10


def test_even_tempered_basis_matches_pyscf():
    aux_basis = "etb:def2-svp:2.0"
    builder = TorchMetricBuilder(aux_basis, "coulomb")
    system = _system()
    reference = compute_coulomb_matrix(system, aux_basis)
    ours = builder.compute_batch([_geometry(system)], CPU)[0]
    assert _relative_error(ours, reference) < 1e-10


def test_batched_systems_match_per_system_assembly():
    builder = TorchMetricBuilder(AUX_BASIS, "coulomb")
    systems = [_system(), _ethane()]
    matrices = builder.compute_batch([_geometry(s) for s in systems], CPU)
    for system, matrix in zip(systems, matrices, strict=True):
        reference = compute_coulomb_matrix(system, AUX_BASIS)
        assert _relative_error(matrix, reference) < 1e-10


def test_float32_assembly_is_close():
    """float32 assembly matches the float64 matrix to single precision -- the
    same resolution the PySCF path has after ``batch_to`` casts it down."""
    builder = TorchMetricBuilder(AUX_BASIS, "coulomb")
    system = _system()
    exact = builder.compute_batch([_geometry(system)], CPU)[0]
    single = builder.compute_batch([_geometry(system)], CPU, torch.float32)[0]
    assert single.dtype == torch.float32
    assert _relative_error(single.to(torch.float64), exact) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_assembly_matches_cpu():
    builder = TorchMetricBuilder(AUX_BASIS, "coulomb")
    system = _system()
    cpu = builder.compute_batch([_geometry(system)], CPU)[0]
    cuda = builder.compute_batch([_geometry(system)], torch.device("cuda"))[0]
    assert _relative_error(cuda.cpu(), cpu) < 1e-12


# ── geometry plumbing ─────────────────────────────────────────────────────────


def test_geometry_round_trips():
    systems = [_system(), _ethane()]
    unpacked = unpack_density_geometry(pack_density_geometry(systems))
    for system, (types, positions) in zip(systems, unpacked, strict=True):
        assert torch.equal(types, system.types.to(torch.int64))
        torch.testing.assert_close(positions, system.positions)


def test_geometry_survives_the_augmenter():
    systems = [_system(), _ethane()]
    target = _batch([_random_target(s, seed=21 + i) for i, s in enumerate(systems)])
    packed = pack_density_geometry(systems)
    augmenter = O3Augmenter(target_info_dict=_target_info(target))
    _, _, extra = augmenter.apply_augmentations(
        systems,
        {TARGET: target},
        [_rotation(0.4), _rotation(0.9)],
        extra_data={DENSITY_GEOMETRY_NAME: packed},
    )
    for i in range(len(systems)):
        torch.testing.assert_close(
            extra[DENSITY_GEOMETRY_NAME].block(i).values, packed.block(i).values
        )


def test_hooks_ship_geometry_instead_of_matrices():
    spec = {
        TARGET: {
            "type": "density_mse_via_c",
            "weight": 1.0,
            "reduction": "sum",
            "metric": "coulomb",
            "aux_basis": AUX_BASIS,
            "backend": "torch",
        }
    }
    hooks = get_density_hooks(spec)
    for transforms in (
        hooks.training_collate_transforms(),
        hooks.validation_collate_transforms(),
    ):
        assert len(transforms) == 1
        _, _, extra = transforms[0]([_system()], {}, {})
        assert list(extra) == [DENSITY_GEOMETRY_NAME]


# ── losses through both backends ──────────────────────────────────────────────


def _via_c(backend: str, metric: str = "coulomb") -> DensityMSELossViaC:
    return DensityMSELossViaC(
        TARGET,
        None,
        weight=1.0,
        reduction="sum",
        metric=metric,
        aux_basis=AUX_BASIS,
        backend=backend,
    )


def _torch_backend_extra(systems) -> dict:
    _, _, extra = get_density_geometry_transform()(list(systems), {}, {})
    return extra


@pytest.mark.parametrize("metric", ["overlap", "coulomb"])
def test_via_c_backends_agree(metric):
    systems = [_system(), _ethane()]
    target = _batch([_random_target(s, seed=31 + i) for i, s in enumerate(systems)])
    prediction = _batch([_random_target(s, seed=41 + i) for i, s in enumerate(systems)])

    from metatrain.utils.pyscf_loss import (
        metric_matrix_name,
        pack_metric_matrices,
    )

    pyscf_extra = {
        metric_matrix_name(TARGET, metric): pack_metric_matrices(
            [compute_metric_matrix(s, AUX_BASIS, metric) for s in systems]
        )
    }
    reference = float(
        _via_c("pyscf", metric).compute(
            {TARGET: prediction}, {TARGET: target}, pyscf_extra
        )
    )
    value = float(
        _via_c("torch", metric).compute(
            {TARGET: prediction}, {TARGET: target}, _torch_backend_extra(systems)
        )
    )
    assert value == pytest.approx(reference, rel=1e-10)


def test_via_w_backends_agree():
    system = _system()
    target = _random_target(system, seed=51)
    prediction = _random_target(system, seed=52)

    matrix = compute_metric_matrix(system, AUX_BASIS, "coulomb")
    reference_flat, _ = _flatten_to_pyscf_order(target)
    projections = _unflatten_like(target, matrix @ reference_flat)
    constant = float(reference_flat @ matrix @ reference_flat)

    shared = {
        ri_projections_name(TARGET): projections,
        ri_density_fit_constant_name(TARGET): _constant_map(constant),
    }
    reference = float(
        DensityMSELossViaW(
            TARGET,
            None,
            weight=1.0,
            reduction="sum",
            metric="coulomb",
            aux_basis=AUX_BASIS,
        ).compute(
            {TARGET: prediction},
            {TARGET: target},
            {**_extra(system, "coulomb"), **shared},
        )
    )
    value = float(
        DensityMSELossViaW(
            TARGET,
            None,
            weight=1.0,
            reduction="sum",
            metric="coulomb",
            aux_basis=AUX_BASIS,
            backend="torch",
        ).compute(
            {TARGET: prediction},
            {TARGET: target},
            {**_torch_backend_extra([system]), **shared},
        )
    )
    assert value == pytest.approx(reference, rel=1e-9)


def _constant_map(value: float) -> TensorMap:
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor([[value]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32)),
                components=[],
                properties=Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )


def test_missing_geometry_is_reported():
    system = _system()
    target = _random_target(system, seed=61)
    with pytest.raises(RuntimeError, match=DENSITY_GEOMETRY_NAME):
        _via_c("torch").compute({TARGET: target}, {TARGET: target}, {})


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        _via_c("numpy")
