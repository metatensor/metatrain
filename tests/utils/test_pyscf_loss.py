"""
Tests for the density (RI-coefficient) losses and their auxiliary-basis metrics.

Two checks carry most of the weight:

* **O(3) invariance** of the loss. The density error cannot change when the molecule
  is rotated, because ``Δc → D Δc`` and ``M → D M D^T`` for the block-diagonal
  Wigner-D matrix ``D``, which is orthogonal. This exercises the whole convention
  stack at once: a flattening that walked the basis in a different order than PySCF,
  or got the ``l=1`` component permutation wrong, would break it.
* **The original-frame path**. Metrics are built on the *unaugmented* geometry, the
  frame the reference coefficients were fitted in, so the trainer augments only the
  systems and maps the predictions back before the loss. That must give exactly the
  same number as augmenting everything and building the metric on the augmented
  geometry, which is what ``test_original_frame_evaluation_matches_rotated_metric``
  pins down.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.augmentation import (
    AUGMENTATION_NAME,
    O3Augmenter,
    _pack_transformations,
    _unpack_transformations,
)
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.loss import (
    DensityMSELossViaC,
    DensityMSELossViaW,
    LossType,
    _flatten_to_pyscf_order,
)
from metatrain.utils.pyscf_loss import (
    METRICS,
    build_auxiliary_molecule,
    compute_metric_matrix,
    get_metric_matrices_transform,
    metric_matrix_name,
    pack_metric_matrices,
    resolve_aux_basis,
    ri_density_fit_constant_name,
    ri_projections_name,
)
from metatrain.utils.scaler.remove import removed_scale_name


pyscf = pytest.importorskip("pyscf")

AUX_BASIS = "def2-universal-jfit"
TARGET = "mtt::density"


def _system() -> System:
    """A small molecule with two elements, so the property padding is exercised."""
    return System(
        types=torch.tensor([6, 1, 1, 1, 1]),
        positions=torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [0.63, -0.63, -0.63],
                [-0.63, 0.63, -0.63],
            ],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _radial_counts(mol) -> dict:
    """Map ``(atomic_number, l)`` to the number of radial functions per atom.

    Counted on one representative atom per element: shells repeat for every atom of
    the same element, so accumulating over all of them would overcount.
    """
    charges = mol.atom_charges()
    representative: dict = {}
    for i_atom, charge in enumerate(charges):
        representative.setdefault(int(charge), i_atom)

    counts: dict = {}
    for shell in range(mol.nbas):
        i_atom = mol.bas_atom(shell)
        charge = int(charges[i_atom])
        if representative[charge] != i_atom:
            continue
        key = (charge, int(mol.bas_angular(shell)))
        counts[key] = counts.get(key, 0) + int(mol.bas_nctr(shell))
    return counts


def _densified_target(system: System, values_fn) -> TensorMap:
    """
    Build a densified atomic-basis TensorMap matching the auxiliary basis.

    Mirrors what metatrain's atomic-basis transform produces at collate time: one
    block per ``o3_lambda``, samples ``(system, atom)``, values ``(atom, m, n)``
    with the property axis padded to the largest per-element count and NaN
    elsewhere.
    """
    mol = build_auxiliary_molecule(system, AUX_BASIS)
    counts = _radial_counts(mol)
    types = [int(t) for t in system.types]
    l_values = sorted({key[1] for key in counts})

    blocks, keys = [], []
    for angular in l_values:
        n_props = max(counts.get((z, angular), 0) for z in set(types))
        values = torch.full(
            (len(types), 2 * angular + 1, n_props), torch.nan, dtype=torch.float64
        )
        for i_atom, z in enumerate(types):
            n_radial = counts.get((z, angular), 0)
            if n_radial:
                values[i_atom, :, :n_radial] = values_fn(
                    i_atom, angular, n_radial, 2 * angular + 1
                )
        keys.append([angular, 1])
        blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[0, i] for i in range(len(types))], dtype=torch.int32
                    ),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(
                            -angular, angular + 1, dtype=torch.int32
                        ).reshape(-1, 1),
                    )
                ],
                properties=Labels(
                    names=["n"],
                    values=torch.arange(n_props, dtype=torch.int32).reshape(-1, 1),
                ),
            )
        )
    return TensorMap(
        Labels(names=["o3_lambda", "o3_sigma"], values=torch.tensor(keys)), blocks
    )


def _random_target(system: System, seed: int) -> TensorMap:
    generator = torch.Generator().manual_seed(seed)
    return _densified_target(
        system,
        lambda i_atom, angular, n_radial, n_m: torch.randn(
            n_m, n_radial, generator=generator, dtype=torch.float64
        ),
    )


def _unflatten_like(template: TensorMap, flat: torch.Tensor) -> TensorMap:
    """Scatter a flat PySCF-ordered vector back into a densified TensorMap.

    The naive inverse of :py:func:`_flatten_to_pyscf_order`, used to build
    projection vectors for the ``via_w`` tests.
    """
    keys = sorted(template.keys, key=lambda key: int(key[0]))
    values = {int(key[0]): template.block(key).values.clone() for key in keys}
    n_atoms = template.block(keys[0]).values.shape[0]

    cursor = 0
    for i_atom in range(n_atoms):
        for key in keys:
            angular = int(key[0])
            block = values[angular][i_atom]  # (m, n)
            order = [2, 0, 1] if angular == 1 else list(range(block.shape[0]))
            for i_n in range(block.shape[1]):
                for i_m in order:
                    if not torch.isnan(block[i_m, i_n]):
                        block[i_m, i_n] = flat[cursor]
                        cursor += 1
    assert cursor == flat.shape[0]
    return TensorMap(
        template.keys,
        [
            TensorBlock(
                values=values[int(key[0])],
                samples=template.block(key).samples,
                components=template.block(key).components,
                properties=template.block(key).properties,
            )
            for key in template.keys
        ],
    )


def _densify(tensor_map: TensorMap) -> TensorMap:
    """Fill the NaN padding with values, as a model's dense output does.

    The filler is deliberately large: anything that lets it reach the loss shows up
    as an obviously wrong number rather than a small discrepancy.
    """
    return TensorMap(
        tensor_map.keys,
        [
            TensorBlock(
                values=torch.nan_to_num(tensor_map.block(key).values, nan=1.0e3),
                samples=tensor_map.block(key).samples,
                components=tensor_map.block(key).components,
                properties=tensor_map.block(key).properties,
            )
            for key in tensor_map.keys
        ],
    )


def _target_info(tensor_map: TensorMap) -> dict:
    irreps = [
        {
            "o3_lambda": int(key[0]),
            "o3_sigma": int(key[1]),
            "num": len(tensor_map.block(key).properties),
        }
        for key in tensor_map.keys
    ]
    return {
        TARGET: get_generic_target_info(
            TARGET,
            {
                "quantity": "",
                "unit": "",
                "type": {"spherical": {"irreps": irreps}},
                "num_subtargets": 1,
                "sample_kind": "atom",
            },
        )
    }


def _extra(system: System, metric: str, extra=None) -> dict:
    packed = pack_metric_matrices([compute_metric_matrix(system, AUX_BASIS, metric)])
    out = {metric_matrix_name(TARGET, metric): packed}
    if extra:
        out.update(extra)
    return out


def _via_c(metric: str = "overlap") -> DensityMSELossViaC:
    return DensityMSELossViaC(
        TARGET, None, weight=1.0, reduction="sum", metric=metric, aux_basis=AUX_BASIS
    )


def _loss_value(system, prediction, target, metric="overlap", extra=None) -> float:
    return float(
        _via_c(metric).compute(
            {TARGET: prediction}, {TARGET: target}, _extra(system, metric, extra)
        )
    )


# ── flattening / ordering ─────────────────────────────────────────────────────


def test_flattened_size_matches_the_pyscf_basis():
    system = _system()
    flat, counts = _flatten_to_pyscf_order(_random_target(system, seed=0))
    mol = build_auxiliary_molecule(system, AUX_BASIS)
    assert flat.shape[0] == mol.nao_nr()
    assert int(counts.sum()) == mol.nao_nr()


def test_flatten_matches_naive_reference():
    """The vectorised flattening must agree with an obvious per-atom loop."""
    tensor_map = _random_target(_system(), seed=1)
    keys = sorted(tensor_map.keys, key=lambda key: int(key[0]))

    expected = []
    for i_atom in range(len(tensor_map.block(keys[0]).samples)):
        for key in keys:
            values = tensor_map.block(key).values[i_atom]
            if int(key[0]) == 1:
                values = values[[2, 0, 1], :]
            for i_n in range(values.shape[1]):
                for i_m in range(values.shape[0]):
                    if not torch.isnan(values[i_m, i_n]):
                        expected.append(values[i_m, i_n])

    flat, _ = _flatten_to_pyscf_order(tensor_map)
    torch.testing.assert_close(flat, torch.stack(expected))


def test_unflatten_round_trips():
    tensor_map = _random_target(_system(), seed=2)
    flat, _ = _flatten_to_pyscf_order(tensor_map)
    again, _ = _flatten_to_pyscf_order(_unflatten_like(tensor_map, flat))
    torch.testing.assert_close(flat, again)


# ── the loss itself ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("metric", METRICS)
def test_loss_is_zero_for_an_exact_prediction(metric):
    system = _system()
    target = _random_target(system, seed=3)
    assert _loss_value(system, target, target, metric) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("metric", METRICS)
def test_loss_matches_the_quadratic_form(metric):
    """Both metrics are positive definite, and the loss is exactly Δc^T M Δc."""
    system = _system()
    target, prediction = _random_target(system, 4), _random_target(system, 5)

    delta, _ = _flatten_to_pyscf_order(prediction, subtract=target)
    matrix = compute_metric_matrix(system, AUX_BASIS, metric)
    reference = float(delta @ matrix @ delta)

    value = _loss_value(system, prediction, target, metric)
    assert value > 0.0
    assert value == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("metric", METRICS)
def test_metrics_are_different(metric):
    """Overlap and Coulomb must not silently be the same matrix."""
    system = _system()
    overlap = compute_metric_matrix(system, AUX_BASIS, "overlap")
    coulomb = compute_metric_matrix(system, AUX_BASIS, "coulomb")
    assert not torch.allclose(overlap, coulomb)
    # Both are symmetric positive definite.
    matrix = compute_metric_matrix(system, AUX_BASIS, metric)
    torch.testing.assert_close(matrix, matrix.T)
    assert torch.linalg.eigvalsh(matrix).min() > 0.0


def _rotation(angle: float, determinant: float = 1.0) -> torch.Tensor:
    axis = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    axis = axis / axis.norm()
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    theta = torch.tensor(angle, dtype=torch.float64)
    return (
        torch.eye(3, dtype=torch.float64)
        + torch.sin(theta) * cross
        + (1 - torch.cos(theta)) * (cross @ cross)
    ) * determinant


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("determinant", [1.0, -1.0], ids=["rotation", "reflection"])
def test_loss_is_invariant_under_o3(metric, determinant):
    """The density error cannot depend on the orientation of the molecule."""
    system = _system()
    target, prediction = _random_target(system, 6), _random_target(system, 7)
    before = _loss_value(system, prediction, target, metric)

    augmenter = O3Augmenter(target_info_dict=_target_info(target))
    systems, rotated, _ = augmenter.apply_augmentations(
        [system],
        {TARGET: target, "pred": prediction},
        [_rotation(0.7, determinant)],
    )
    after = _loss_value(systems[0], rotated["pred"], rotated[TARGET], metric)
    assert before == pytest.approx(after, rel=1e-8)


def test_system_augmentation_leaves_targets_alone():
    """The systems-only workflow records its transformation and touches nothing else."""
    system = _system()
    target = _random_target(system, 8)
    augmenter = O3Augmenter(target_info_dict=_target_info(target))

    augmented, passed_through, extra = augmenter.apply_random_system_augmentations(
        [system], {TARGET: target}, original_frame={TARGET}
    )
    assert AUGMENTATION_NAME in extra
    assert _unpack_transformations(extra[AUGMENTATION_NAME]).shape == (1, 3, 3)
    # the target must come back in the frame the dataset stores it in, untouched
    assert passed_through[TARGET] is target
    assert not torch.allclose(
        augmented[0].positions, system.positions
    ) or torch.allclose(
        _unpack_transformations(extra[AUGMENTATION_NAME])[0],
        torch.eye(3, dtype=torch.float64),
    )


@pytest.mark.parametrize("determinant", [1.0, -1.0])
@pytest.mark.parametrize("metric", METRICS)
def test_original_frame_evaluation_matches_rotated_metric(metric, determinant):
    """
    The caching path must be exact, for proper *and* improper transformations.

    Metrics are built on the unaugmented geometry, so the trainer augments only the
    systems and maps the predictions back before the loss. That must equal augmenting
    everything and building the metric on the augmented geometry.

    ``determinant=-1`` is the case that matters: a Wigner-D matrix carries only the
    proper rotation, and the ``(-1)**ell`` parity of an improper one lives in the
    block key. A back-transform that applied the Wigner-D alone would silently leave
    the odd-``ell`` blocks with the wrong sign, and only here would it show.
    """
    system = _system()
    target, prediction = _random_target(system, 8), _random_target(system, 9)
    augmenter = O3Augmenter(target_info_dict=_target_info(target))

    matrix = _rotation(1.1, determinant)
    extra = {AUGMENTATION_NAME: _pack_transformations([matrix])}

    # Reference: the same transformation applied to everything, metric built on the
    # augmented geometry, loss taken there.
    reference_systems, rotated, _ = augmenter.apply_augmentations(
        [system], {TARGET: target, "pred": prediction}, [matrix]
    )
    reference = _loss_value(
        reference_systems[0], rotated["pred"], rotated[TARGET], metric
    )

    # Original-frame path: back-transform the prediction made in the augmented frame
    # and compare it against the untouched target.
    back = augmenter.undo_augmentation(
        {TARGET: rotated["pred"]}, reference_systems, extra, original_frame={TARGET}
    )
    packed = pack_metric_matrices([compute_metric_matrix(system, AUX_BASIS, metric)])
    value = float(
        _via_c(metric).compute(
            {TARGET: back[TARGET]},
            {TARGET: target},
            {metric_matrix_name(TARGET, metric): packed},
        )
    )
    assert value == pytest.approx(reference, rel=1e-8)


# ── via_w ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("metric", METRICS)
def test_via_w_equals_via_c(metric):
    """
    ``via_w`` is an algebraic rearrangement of ``via_c`` and must agree exactly.

    With ``w = M c_ref`` and the constant ``c_ref^T w`` supplied, the two forms are
    the same number; the difference is only that ``via_w`` never forms ``c_ref``.
    """
    system = _system()
    target, prediction = _random_target(system, 10), _random_target(system, 11)

    reference_flat, _ = _flatten_to_pyscf_order(target)
    matrix = compute_metric_matrix(system, AUX_BASIS, metric)
    projections_flat = matrix @ reference_flat
    constant = float(reference_flat @ projections_flat)

    projections = _unflatten_like(target, projections_flat)
    constant_map = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor([[constant]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32)),
                components=[],
                properties=Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )

    via_c = _loss_value(system, prediction, target, metric)
    via_w_loss = DensityMSELossViaW(
        TARGET, None, weight=1.0, reduction="sum", metric=metric, aux_basis=AUX_BASIS
    )
    via_w = float(
        via_w_loss.compute(
            {TARGET: prediction},
            {TARGET: target},
            _extra(
                system,
                metric,
                {
                    ri_projections_name(TARGET): projections,
                    ri_density_fit_constant_name(TARGET): constant_map,
                },
            ),
        )
    )
    assert via_w == pytest.approx(via_c, rel=1e-8)


def test_via_w_without_constant_is_shifted_but_tracks():
    """Without the constant the loss is offset by exactly c_ref^T w."""
    system = _system()
    target, prediction = _random_target(system, 12), _random_target(system, 13)
    reference_flat, _ = _flatten_to_pyscf_order(target)
    matrix = compute_metric_matrix(system, AUX_BASIS, "overlap")
    projections = _unflatten_like(target, matrix @ reference_flat)
    constant = float(reference_flat @ (matrix @ reference_flat))

    loss = DensityMSELossViaW(
        TARGET, None, weight=1.0, reduction="sum", aux_basis=AUX_BASIS
    )
    value = float(
        loss.compute(
            {TARGET: prediction},
            {TARGET: target},
            _extra(system, "overlap", {ri_projections_name(TARGET): projections}),
        )
    )
    via_c = _loss_value(system, prediction, target)
    assert value == pytest.approx(via_c - constant, rel=1e-8)


def test_via_w_accepts_a_dense_prediction():
    """
    The padding lives in the reference data, not in the model's output.

    A prediction flattened against its own values carries no NaN, so every padded
    slot counts as a real coefficient and the layout check rejects any batch mixing
    elements of different basis sizes -- which is every batch of a real dataset.
    Taking the padding from the projections instead fixes the layout, and the
    padded slots must not reach the number either.
    """
    system = _system()
    target, prediction = _random_target(system, 20), _random_target(system, 21)
    reference_flat, _ = _flatten_to_pyscf_order(target)
    matrix = compute_metric_matrix(system, AUX_BASIS, "overlap")
    projections = _unflatten_like(target, matrix @ reference_flat)

    loss = DensityMSELossViaW(
        TARGET, None, weight=1.0, reduction="sum", aux_basis=AUX_BASIS
    )
    extra = _extra(system, "overlap", {ri_projections_name(TARGET): projections})

    padded = float(loss.compute({TARGET: prediction}, {TARGET: target}, extra))
    dense = float(loss.compute({TARGET: _densify(prediction)}, {TARGET: target}, extra))
    assert dense == pytest.approx(padded, rel=1e-12)


def test_via_w_requires_projections():
    loss = DensityMSELossViaW(
        TARGET, None, weight=1.0, reduction="sum", aux_basis=AUX_BASIS
    )
    system = _system()
    target = _random_target(system, 14)
    with pytest.raises(RuntimeError, match="requires"):
        loss.compute({TARGET: target}, {TARGET: target}, _extra(system, "overlap"))


# ── the scale convention ──────────────────────────────────────────────────────


#: Deliberately far apart, and different per element: a scale that is uniform
#: across species would factor out of the quadratic form and hide the bug.
_SCALES = {6: 4.0, 1: 0.25}


def _per_type_map(template: TensorMap, system: System, factors: dict) -> TensorMap:
    """A map shaped like ``template`` holding ``factors[z]`` on every real entry."""
    blocks = []
    for key in template.keys:
        block = template.block(key)
        per_atom = torch.tensor(
            [factors[int(z)] for z in system.types], dtype=block.values.dtype
        ).reshape(-1, 1, 1)
        blocks.append(
            TensorBlock(
                values=(0.0 * block.values + 1.0) * per_atom,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(template.keys, blocks)


def _multiply(tensor_map: TensorMap, factor_map: TensorMap) -> TensorMap:
    blocks = [
        TensorBlock(
            values=tensor_map.block(key).values * factor_map.block(key).values,
            samples=tensor_map.block(key).samples,
            components=tensor_map.block(key).components,
            properties=tensor_map.block(key).properties,
        )
        for key in tensor_map.keys
    ]
    return TensorMap(tensor_map.keys, blocks)


@pytest.mark.parametrize("metric", METRICS)
def test_via_c_undoes_a_per_species_scale(metric):
    """
    The trainer's scale removal must not leak into the metric.

    What it divides out is one scale *per atomic type*, i.e. a diagonal ``D``, so
    a loss that consumed the scaled residual would evaluate ``Δc^T D^-1 M D^-1
    Δc`` and silently reweight the metric per species. Feeding the loss the
    scaled coefficients together with the record of what was removed must
    reproduce the physical value exactly.
    """
    system = _system()
    target, prediction = _random_target(system, 20), _random_target(system, 21)

    physical = _loss_value(system, prediction, target, metric)

    inverse = _per_type_map(target, system, {z: 1.0 / s for z, s in _SCALES.items()})
    scaled = _loss_value(
        system,
        _multiply(prediction, inverse),
        _multiply(target, inverse),
        metric,
        {removed_scale_name(TARGET): inverse},
    )
    assert scaled == pytest.approx(physical, rel=1e-10)


def test_via_c_without_a_record_takes_the_coefficients_as_physical():
    system = _system()
    target, prediction = _random_target(system, 22), _random_target(system, 23)
    inverse = _per_type_map(target, system, {z: 1.0 / s for z, s in _SCALES.items()})

    # Same scaled inputs as above, but with the record withheld: the loss has no
    # way to undo the scaling, so it must differ from the physical value.
    assert _loss_value(
        system, _multiply(prediction, inverse), _multiply(target, inverse)
    ) != pytest.approx(_loss_value(system, prediction, target), rel=1e-6)


def test_via_w_undoes_the_scale_but_not_on_the_projections():
    """
    ``via_w`` mixes scaled predictions with unscaled dataset projections.

    ``w`` and the constant come straight from the dataset, so the linear and
    quadratic terms would carry different powers of the scale. Only the
    coefficients may be rescaled.
    """
    system = _system()
    target, prediction = _random_target(system, 24), _random_target(system, 25)

    reference_flat, _ = _flatten_to_pyscf_order(target)
    matrix = compute_metric_matrix(system, AUX_BASIS, "overlap")
    projections_flat = matrix @ reference_flat
    projections = _unflatten_like(target, projections_flat)
    constant_map = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(
                    [[float(reference_flat @ projections_flat)]], dtype=torch.float64
                ),
                samples=Labels(["system"], torch.tensor([[0]], dtype=torch.int32)),
                components=[],
                properties=Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
            )
        ],
    )

    loss = DensityMSELossViaW(
        TARGET, None, weight=1.0, reduction="sum", aux_basis=AUX_BASIS
    )
    inverse = _per_type_map(target, system, {z: 1.0 / s for z, s in _SCALES.items()})

    def value(predicted, extra):
        return float(
            loss.compute(
                {TARGET: predicted},
                {TARGET: target},
                _extra(
                    system,
                    "overlap",
                    {
                        ri_projections_name(TARGET): projections,
                        ri_density_fit_constant_name(TARGET): constant_map,
                        **extra,
                    },
                ),
            )
        )

    assert value(
        _multiply(prediction, inverse), {removed_scale_name(TARGET): inverse}
    ) == pytest.approx(value(prediction, {}), rel=1e-8)


def test_a_mismatched_scale_record_is_rejected():
    system = _system()
    target = _random_target(system, 26)
    # A record covering only some of the coefficients: the layouts disagree, and
    # applying it entry by entry would silently misalign the two vectors.
    truncated = TensorMap(
        target.keys,
        [
            TensorBlock(
                values=0.0 * target.block(key).values[:, :, :1] + 1.0,
                samples=target.block(key).samples,
                components=target.block(key).components,
                properties=Labels(["n"], torch.tensor([[0]], dtype=torch.int32)),
            )
            for key in target.keys
        ],
    )
    with pytest.raises(ValueError, match="does not share the coefficients"):
        _loss_value(
            system, target, target, extra={removed_scale_name(TARGET): truncated}
        )


# ── caching ───────────────────────────────────────────────────────────────────


def test_transform_recomputes_every_batch(monkeypatch):
    """Metric matrices are rebuilt per batch; nothing is carried between them."""
    import metatrain.utils.pyscf_loss as module

    calls = {"n": 0}
    original = module.compute_metric_matrix

    def counting(system, aux_basis, metric):
        calls["n"] += 1
        return original(system, aux_basis, metric)

    monkeypatch.setattr(module, "compute_metric_matrix", counting)
    transform = get_metric_matrices_transform({TARGET: AUX_BASIS}, "overlap")
    transform([_system()], {}, {})
    transform([_system()], {}, {})
    assert calls["n"] == 2


def test_packed_metric_survives_the_augmenter():
    """
    The packed metric must pass through O(3) augmentation untouched.

    This is why it is stored without a component axis: the augmenter infers tensor
    character from component names and raises on anything it cannot transform, which
    would otherwise forbid computing the metric before augmentation -- and hence
    forbid caching it.
    """
    # Must be more than one system: the augmenter only demands a "system" sample
    # dimension when routing several, so a single-system check passes vacuously.
    systems = [_system(), _ethane()]
    target = _batch([_random_target(s, seed=15 + i) for i, s in enumerate(systems)])
    packed = pack_metric_matrices(
        [compute_metric_matrix(s, AUX_BASIS, "overlap") for s in systems]
    )
    augmenter = O3Augmenter(target_info_dict=_target_info(target))
    _, _, extra = augmenter.apply_augmentations(
        systems,
        {TARGET: target},
        [_rotation(0.4), _rotation(0.9)],
        extra_data={"m": packed},
    )
    for i in range(len(systems)):
        torch.testing.assert_close(extra["m"].block(i).values, packed.block(i).values)


def test_system_augmentation_respects_the_augmenter_group():
    """
    The systems-only workflow must augment with the group the trainer asked for.

    SPACE is rotation-equivariant by construction and so augments with inversions
    only; drawing full O(3) rotations here would silently change its training.
    """
    system = _system()
    target = _random_target(system, seed=21)
    augmenter = O3Augmenter(target_info_dict=_target_info(target), group="inversions")

    identity = torch.eye(3, dtype=torch.float64)
    # Several draws, since the inversion is sampled and half of them are the identity.
    for _ in range(10):
        _, _, extra = augmenter.apply_random_system_augmentations(
            [system], {}, original_frame={TARGET}
        )
        matrix = _unpack_transformations(extra[AUGMENTATION_NAME])[0]
        assert torch.allclose(matrix, identity) or torch.allclose(matrix, -identity), (
            "not a signed identity, so a rotation was applied"
        )


# ── configuration and errors ──────────────────────────────────────────────────


def test_losses_are_registered():
    assert LossType.from_key("density_mse_via_c").cls is DensityMSELossViaC
    assert LossType.from_key("density_mse_via_w").cls is DensityMSELossViaW


def test_basis_mismatch_is_reported():
    system = _system()
    target = _random_target(system, seed=16)
    n = compute_metric_matrix(system, AUX_BASIS, "overlap").shape[0]
    packed = pack_metric_matrices([torch.eye(n + 5, dtype=torch.float64)])
    with pytest.raises(ValueError, match="does not match"):
        _via_c().compute(
            {TARGET: target},
            {TARGET: target},
            {metric_matrix_name(TARGET, "overlap"): packed},
        )


def test_missing_metric_matrix_is_reported():
    target = _random_target(_system(), seed=17)
    with pytest.raises(RuntimeError, match="requires"):
        _via_c().compute({TARGET: target}, {TARGET: target}, {})


def test_aux_basis_is_required():
    with pytest.raises(ValueError, match="require 'aux_basis'"):
        DensityMSELossViaC(TARGET, None, weight=1.0, reduction="mean")


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        DensityMSELossViaC(
            TARGET,
            None,
            weight=1.0,
            reduction="mean",
            metric="nonsense",
            aux_basis=AUX_BASIS,
        )
    with pytest.raises(ValueError, match="unknown metric"):
        get_metric_matrices_transform({TARGET: AUX_BASIS}, "nonsense")


def test_gradients_are_rejected():
    with pytest.raises(NotImplementedError, match="gradients"):
        DensityMSELossViaC(
            TARGET, "positions", weight=1.0, reduction="mean", aux_basis=AUX_BASIS
        )


def test_resolve_aux_basis():
    assert resolve_aux_basis(TARGET, AUX_BASIS) == AUX_BASIS
    assert resolve_aux_basis(TARGET, {TARGET: AUX_BASIS}) == AUX_BASIS
    with pytest.raises(ValueError, match="no auxiliary basis"):
        resolve_aux_basis(TARGET, {"other": AUX_BASIS})


def test_even_tempered_basis_builds():
    overlap = compute_metric_matrix(_system(), "etb:def2-svp:2.0", "overlap")
    assert overlap.shape[0] == overlap.shape[1] > 0
    torch.testing.assert_close(overlap, overlap.T)
    assert torch.linalg.eigvalsh(overlap).min() > 0.0


def test_gradients_flow_to_the_prediction():
    system = _system()
    target, prediction = _random_target(system, 18), _random_target(system, 19)
    blocks = [
        TensorBlock(
            values=prediction.block(key).values.clone().requires_grad_(True),
            samples=prediction.block(key).samples,
            components=prediction.block(key).components,
            properties=prediction.block(key).properties,
        )
        for key in prediction.keys
    ]
    differentiable = TensorMap(prediction.keys, blocks)

    value = _via_c().compute(
        {TARGET: differentiable}, {TARGET: target}, _extra(system, "overlap")
    )
    value.backward()

    grads = [block.values.grad for block in differentiable.blocks()]
    assert all(g is not None for g in grads)
    assert any(torch.any(torch.nan_to_num(g) != 0.0) for g in grads)


# ── heterogeneous batches and dtype ───────────────────────────────────────


def _ethane() -> System:
    """A second molecule with the same elements but more atoms than ``_system``."""
    return System(
        types=torch.tensor([6, 6, 1, 1, 1, 1, 1, 1]),
        positions=torch.tensor(
            [
                [0.00, 0.00, 0.77],
                [0.00, 0.00, -0.77],
                [0.00, 1.02, 1.16],
                [0.88, -0.51, 1.16],
                [-0.88, -0.51, 1.16],
                [0.00, -1.02, -1.16],
                [0.88, 0.51, -1.16],
                [-0.88, 0.51, -1.16],
            ],
            dtype=torch.float64,
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _batch(tensor_maps: list) -> TensorMap:
    """Concatenate per-system densified TensorMaps into one batch."""
    keys = tensor_maps[0].keys
    blocks = []
    for key in keys:
        values, samples = [], []
        for i_system, tensor_map in enumerate(tensor_maps):
            block = tensor_map.block(key)
            values.append(block.values)
            sample_values = block.samples.values.clone()
            sample_values[:, 0] = i_system
            samples.append(sample_values)
        blocks.append(
            TensorBlock(
                values=torch.cat(values),
                samples=Labels(["system", "atom"], torch.cat(samples)),
                components=tensor_maps[0].block(key).components,
                properties=tensor_maps[0].block(key).properties,
            )
        )
    return TensorMap(keys, blocks)


def test_heterogeneous_batch_matches_per_system_evaluation():
    """
    Systems of different sizes must be handled without padding to the batch max.

    The batched loss must equal the sum of the individually evaluated losses, which
    is only true if each system is contracted against its own metric matrix.
    """
    systems = [_system(), _ethane()]
    targets = [_random_target(s, seed=100 + i) for i, s in enumerate(systems)]
    predictions = [_random_target(s, seed=200 + i) for i, s in enumerate(systems)]

    sizes = [compute_metric_matrix(s, AUX_BASIS, "overlap").shape[0] for s in systems]
    assert sizes[0] != sizes[1], "this test needs differently sized systems"

    separate = sum(
        _loss_value(s, p, t)
        for s, p, t in zip(systems, predictions, targets, strict=True)
    )

    packed = pack_metric_matrices(
        [compute_metric_matrix(s, AUX_BASIS, "overlap") for s in systems]
    )
    batched = float(
        _via_c().compute(
            {TARGET: _batch(predictions)},
            {TARGET: _batch(targets)},
            {metric_matrix_name(TARGET, "overlap"): packed},
        )
    )
    assert batched == pytest.approx(separate, rel=1e-9)


def test_ragged_storage_is_not_padded():
    """Each system's block is its own size, so no batch-max padding is stored."""
    systems = [_system(), _ethane()]
    matrices = [compute_metric_matrix(s, AUX_BASIS, "overlap") for s in systems]
    packed = pack_metric_matrices(matrices)

    assert len(packed) == len(systems)
    stored = sum(packed.block(i).values.numel() for i in range(len(packed)))
    n_max = max(m.shape[0] for m in matrices)
    assert stored == sum(m.shape[0] ** 2 for m in matrices)
    assert stored < len(systems) * n_max**2, "storage should not be padded to n_max"


def test_integrals_are_computed_in_double():
    """
    The integrals are always evaluated in double precision.

    What reaches the loss is whatever ``batch_to`` casts them to, which is the
    model's dtype -- the same treatment as the targets. The loss has no say in it.
    """
    _, _, extra = get_metric_matrices_transform({TARGET: AUX_BASIS}, "overlap")(
        [_system()], {}, {}
    )
    name = metric_matrix_name(TARGET, "overlap")
    assert extra[name].block(0).values.dtype == torch.float64


def test_loss_is_stable_in_single_precision():
    """
    Evaluating in the model's precision must not meaningfully change the value.

    Models train in float32, so this is the precision the loss actually runs in; the
    metric is ill-conditioned, so it is worth pinning that this is safe.
    """
    system = _system()
    target, prediction = _random_target(system, 300), _random_target(system, 301)
    name = metric_matrix_name(TARGET, "overlap")

    values = {}
    for dtype in (torch.float64, torch.float32):
        _, _, extra = get_metric_matrices_transform({TARGET: AUX_BASIS}, "overlap")(
            [system], {}, {}
        )
        # what `batch_to` does to every tensor in the batch, targets included
        extra = {name: extra[name].to(dtype=dtype)}
        values[dtype] = float(
            _via_c().compute(
                {TARGET: prediction.to(dtype=dtype)},
                {TARGET: target.to(dtype=dtype)},
                extra,
            )
        )
    assert values[torch.float32] == pytest.approx(values[torch.float64], rel=1e-4)


def test_periodic_systems_are_rejected():
    """
    The metric is molecular, so a periodic system must fail loudly.

    The cell is absent from the integral construction; that is
    self-consistent, but it would silently score a periodic system against a
    non-periodic metric.
    """
    periodic = System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64) * 8.0,
        pbc=torch.tensor([True, True, True]),
    )
    with pytest.raises(NotImplementedError, match="non-periodic"):
        compute_metric_matrix(periodic, AUX_BASIS, "overlap")


def test_eval_options_accept_a_metrics_block():
    """``mtt eval`` takes the same ``metrics`` block that ``mtt train`` does."""
    from metatrain.utils.pydantic import validate_eval_options

    options = validate_eval_options(
        {
            "systems": "systems.xyz",
            "metrics": {
                TARGET: [
                    "rmse",
                    {"type": "density_mse_via_c", "aux_basis": AUX_BASIS},
                ]
            },
        }
    )
    assert options["metrics"][TARGET][1]["aux_basis"] == AUX_BASIS
