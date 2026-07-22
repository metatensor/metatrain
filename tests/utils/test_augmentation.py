import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from metatomic.torch.o3 import (
    random_transformations,
    transform_system,
    transform_tensor,
)

from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.data.target_info import get_generic_target_info

from ..conftest import RESOURCES_PATH


# Hard-coded rotation matrix used to generate the rotated DFT dataset
_R = np.array(
    [
        [0.22922512, -0.15149287, 0.96151222],
        [0.66175278, -0.70015582, -0.26807664],
        [0.71382008, 0.69773328, -0.06024247],
    ]
)

# Irreps of the spherical targets in the test datasets
_ELECTRON_DENSITY_IRREPS = [
    {"o3_lambda": 0, "o3_sigma": 1},
    {"o3_lambda": 1, "o3_sigma": 1},
    {"o3_lambda": 2, "o3_sigma": 1},
    {"o3_lambda": 3, "o3_sigma": 1},
]

# Per-atom-type irreps of the atomic-basis Hamiltonian targets in the test datasets
_HAMILTONIAN_IRREPS = {
    1: [  # H
        {"o3_lambda": 0, "o3_sigma": 1, "num": 3},
        {"o3_lambda": 1, "o3_sigma": 1, "num": 1},
    ],
    8: [  # O
        {"o3_lambda": 0, "o3_sigma": 1, "num": 5},
        {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
        {"o3_lambda": 2, "o3_sigma": 1, "num": 2},
        {"o3_lambda": 3, "o3_sigma": 1, "num": 1},
    ],
}


def _transformation(batch_size: int) -> list:
    # apply_augmentations expects List[torch.Tensor] where each matrix R satisfies
    # positions_rotated = positions @ R.T, matching how the test datasets were built
    t = torch.tensor(_R.T, dtype=torch.float64)
    return [t] * batch_size


def _relabel_system_samples(tensor: TensorMap, system_ids: torch.Tensor) -> TensorMap:
    """Replace a TensorMap's "system" sample values (assumed to be dataset positions
    0, ..., n-1) with the given absolute dataset ids, simulating a batch drawn from a
    larger dataset than itself."""
    blocks = []
    for block in tensor.blocks():
        position = block.samples.names.index("system")
        values = block.samples.values.clone()
        values[:, position] = system_ids[values[:, position]]
        blocks.append(
            TensorBlock(
                values=block.values,
                samples=Labels(block.samples.names, values),
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


def _dataset_info(target_name: str, target_type: dict, sample_kind: str) -> DatasetInfo:
    """Build a single-target ``DatasetInfo`` matching the test datasets."""
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            target_name: get_generic_target_info(
                target_name,
                {
                    "quantity": "",
                    "unit": "",
                    "type": target_type,
                    "sample_kind": sample_kind,
                    "num_subtargets": 1,
                },
            )
        },
    )


def _load_systems_and_targets(
    target_name: str, batch_size: int
) -> tuple[list[System], TensorMap, TensorMap]:
    """Load the first ``batch_size`` systems together with their unrotated and
    DFT-rotated ``target_name`` references, joined along samples."""
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")
    X = [dataset_unrotated[i]["system"].to(torch.float64) for i in range(batch_size)]
    fX = mts.join(
        [dataset_unrotated[i][target_name] for i in range(batch_size)], axis="samples"
    )
    fRX = mts.join(
        [dataset_rotated[i][target_name] for i in range(batch_size)], axis="samples"
    )
    return X, fX, fRX


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_structure_spherical(batch_size):
    """Tests that the rotational augmenter rotates a dipole moment consistent with
    targets computed from DFT"""

    target_name = "mtt::dipole_moment"
    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)

    dataset_info = _dataset_info(
        target_name,
        {"spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}},
        "system",
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


def test_apply_augmentations_extra_data():
    """Tests that spherical extra data is rotated exactly like a target, while extra
    data whose name ends in ``_mask`` is passed through untouched (loss masks are not
    physical quantities and must not be rotated)."""
    target_name = "mtt::dipole_moment"
    batch_size = 2
    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)

    target_type = {"spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}}
    target_info = _dataset_info(target_name, target_type, "system")
    extra_name = "mtt::dipole_extra"
    extra_info = _dataset_info(extra_name, target_type, "system")
    rotational_augmenter = O3Augmenter(target_info.targets, extra_info.targets)

    mask = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
                samples=Labels(["system"], torch.tensor([[0], [1]])),
                components=[],
                properties=Labels(["_"], torch.tensor([[0]])),
            )
        ],
    )

    _, _, new_extra_data = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={extra_name: fX, "mtt::dipole_moment_mask": mask},
    )

    # the spherical extra entry rotates to the DFT-rotated reference, just like a target
    mts.allclose_raise(new_extra_data[extra_name], fRX, atol=1e-5)
    # the mask comes back bit-for-bit unchanged
    torch.testing.assert_close(
        new_extra_data["mtt::dipole_moment_mask"].block().values, mask.block().values
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical(batch_size):
    """Tests that the rotational augmenter rotates electron density projections
    consistent with targets computed from DFT"""

    target_name = "mtt::electron_density_basis_projs"
    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)

    dataset_info = _dataset_info(
        target_name, {"spherical": {"irreps": _ELECTRON_DENSITY_IRREPS}}, "atom"
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


def test_distinct_transformations_per_system():
    """Tests that each system's target rows are transformed by that system's own
    matrix. metatomic's ``transform_tensor`` pairs ``system_ids[i]`` positionally
    with ``transformations[i]``, so this only holds if the ids recovered from the
    tensor come out in the same order as the ``systems`` list. Every other test
    applies the same matrix to all systems and would not notice a mix-up.
    """
    target_name = "mtt::electron_density_basis_projs"

    # Absolute dataset ids, deliberately not 0, ..., n-1
    system_ids = torch.tensor([3, 8])

    X, fX, _ = _load_systems_and_targets(target_name, 2)
    fX = _relabel_system_samples(fX, system_ids)

    # system 0 is left alone while system 1 is rotated, so the expected result
    # mixes the unrotated and the DFT-rotated references
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")
    expected = _relabel_system_samples(
        mts.join(
            [dataset_unrotated[0][target_name], dataset_rotated[1][target_name]],
            axis="samples",
        ),
        system_ids,
    )

    dataset_info = _dataset_info(
        target_name, {"spherical": {"irreps": _ELECTRON_DENSITY_IRREPS}}, "atom"
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    transformations = [
        torch.eye(3, dtype=torch.float64),
        torch.tensor(_R.T, dtype=torch.float64),
    ]
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X, {target_name: fX}, transformations, extra_data={}
    )

    mts.allclose_raise(RfX[target_name], expected, atol=1e-5)


def test_apply_random_augmentations():
    """Tests the entry point the trainers actually use. Seeding ``torch`` and
    replaying the same ``random_transformations`` draw must reproduce exactly what
    ``apply_random_augmentations`` does: the augmented systems and targets equal
    ``transform_system``/``transform_tensor`` under the reconstructed transformations,
    and the augmented targets keep their metadata and leave invariant
    (``o3_lambda=0``) blocks untouched."""
    target_name = "mtt::electron_density_basis_projs"
    batch_size = 2

    X, fX, _ = _load_systems_and_targets(target_name, batch_size)

    dataset_info = _dataset_info(
        target_name, {"spherical": {"irreps": _ELECTRON_DENSITY_IRREPS}}, "atom"
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # reconstruct the exact draw (max ell is 3 for this target), then reseed so the
    # augmenter draws the same transformations from the same generator state
    torch.manual_seed(42)
    transformations = random_transformations(
        len(X),
        3,
        device=torch.device("cpu"),
        dtype=torch.float64,
        include_inversions=True,
    )
    torch.manual_seed(42)
    new_systems, new_targets, _ = rotational_augmenter.apply_random_augmentations(
        X, {target_name: fX}
    )
    RfX = new_targets[target_name]

    # the augmented systems/targets match the reconstructed transformations exactly
    for system, new_system, transformation in zip(
        X, new_systems, transformations, strict=True
    ):
        torch.testing.assert_close(
            new_system.positions, transform_system(system, transformation).positions
        )
    mts.allclose_raise(RfX, transform_tensor(fX, X, transformations))

    assert RfX.keys == fX.keys
    for key, block in fX.items():
        new_block = RfX.block(key)
        assert new_block.samples == block.samples
        if int(key["o3_lambda"]) == 0:
            # invariant blocks must come back bit-compatible
            torch.testing.assert_close(new_block.values, block.values)


def test_inversion_parity():
    """Tests that a pure inversion flips a polar vector (``o3_sigma=1``) but leaves
    a pseudovector (``o3_sigma=-1``) unchanged: spherical blocks pick up
    ``sigma * (-1)**lambda`` under improper transformations. Proper rotations
    cannot distinguish the two, so this is the only check of the parity factor."""
    target_name = "mtt::vectors"

    system = System(
        positions=torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1, 8]),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )

    vector_values = torch.tensor([[[0.5], [1.0], [-2.0]]], dtype=torch.float64)
    pseudo_values = torch.tensor([[[1.5], [-0.5], [0.25]]], dtype=torch.float64)

    def _spherical_block(values: torch.Tensor) -> TensorBlock:
        return TensorBlock(
            values=values,
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[Labels(["o3_mu"], torch.tensor([[-1], [0], [1]]))],
            properties=Labels(["properties"], torch.tensor([[0]])),
        )

    target = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1], [1, -1]])),
        blocks=[_spherical_block(vector_values), _spherical_block(pseudo_values)],
    )

    dataset_info = _dataset_info(
        target_name,
        {
            "spherical": {
                "irreps": [
                    {"o3_lambda": 1, "o3_sigma": 1},
                    {"o3_lambda": 1, "o3_sigma": -1},
                ]
            }
        },
        "system",
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    inversion = -torch.eye(3, dtype=torch.float64)
    new_systems, new_targets, _ = rotational_augmenter.apply_augmentations(
        [system], {target_name: target}, [inversion], extra_data={}
    )

    torch.testing.assert_close(new_systems[0].positions, -system.positions)
    out = new_targets[target_name]
    torch.testing.assert_close(
        out.block({"o3_lambda": 1, "o3_sigma": 1}).values, -vector_values
    )
    torch.testing.assert_close(
        out.block({"o3_lambda": 1, "o3_sigma": -1}).values, pseudo_values
    )


def test_apply_random_inversions():
    """Tests that ``group="inversions"`` draws only the identity and the inversion:
    each system's positions come back either exactly unchanged or exactly negated,
    its polar vector (``o3_sigma=1``) target rows flip with the positions, and its
    pseudovector (``o3_sigma=-1``) rows stay put (``sigma * (-1)**lambda`` is +1 for
    a pseudovector under inversion). Seed 0 gives a batch exercising both signs."""
    target_name = "mtt::vectors"
    batch_size = 4

    positions = torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.0, -0.5]], dtype=torch.float64)
    systems = [
        System(
            positions=positions,
            types=torch.tensor([1, 8]),
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        )
        for _ in range(batch_size)
    ]

    torch.manual_seed(0)
    vector_values = torch.randn(batch_size, 3, 1, dtype=torch.float64)
    pseudo_values = torch.randn(batch_size, 3, 1, dtype=torch.float64)

    def _spherical_block(values: torch.Tensor) -> TensorBlock:
        return TensorBlock(
            values=values.clone(),
            samples=Labels(["system"], torch.arange(batch_size).reshape(-1, 1)),
            components=[Labels(["o3_mu"], torch.tensor([[-1], [0], [1]]))],
            properties=Labels(["properties"], torch.tensor([[0]])),
        )

    target = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[1, 1], [1, -1]])),
        blocks=[_spherical_block(vector_values), _spherical_block(pseudo_values)],
    )

    dataset_info = _dataset_info(
        target_name,
        {
            "spherical": {
                "irreps": [
                    {"o3_lambda": 1, "o3_sigma": 1},
                    {"o3_lambda": 1, "o3_sigma": -1},
                ]
            }
        },
        "system",
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {}, group="inversions")

    # reconstruct the +-1 signs the augmenter draws, and check the seed hits both
    torch.manual_seed(0)
    signs = torch.randint(0, 2, (batch_size,)) * 2 - 1
    assert (signs == 1).any() and (signs == -1).any()

    torch.manual_seed(0)
    new_systems, new_targets, _ = rotational_augmenter.apply_random_augmentations(
        systems, {target_name: target}
    )
    out = new_targets[target_name]
    out_vector = out.block({"o3_lambda": 1, "o3_sigma": 1}).values
    out_pseudo = out.block({"o3_lambda": 1, "o3_sigma": -1}).values
    for i, sign in enumerate(signs.tolist()):
        torch.testing.assert_close(new_systems[i].positions, sign * positions)
        torch.testing.assert_close(out_vector[i], sign * vector_values[i])
        torch.testing.assert_close(out_pseudo[i], pseudo_values[i])


def test_unknown_group_raises():
    """Tests that constructing an augmenter with an unknown transformation group is
    rejected rather than silently ignored."""
    with pytest.raises(ValueError, match="unknown transformation group"):
        O3Augmenter({}, {}, group="bogus")


def test_cartesian_rank3():
    """Tests that Cartesian targets of rank > 2 are accepted and rotated one axis
    at a time (the old hand-rolled implementation rejected them; the delegated
    transformation supports arbitrary rank)."""
    target_name = "mtt::rank3"

    system = System(
        positions=torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1, 8]),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )

    values = torch.arange(27, dtype=torch.float64).reshape(1, 3, 3, 3, 1)
    target = TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels(["system"], torch.tensor([[0]])),
                components=[
                    Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
                    Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
                    Labels(["xyz_3"], torch.arange(3).reshape(-1, 1)),
                ],
                properties=Labels(["properties"], torch.tensor([[0]])),
            )
        ],
    )

    dataset_info = _dataset_info(target_name, {"cartesian": {"rank": 3}}, "system")
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    matrix = torch.tensor(_R, dtype=torch.float64)
    _, new_targets, _ = rotational_augmenter.apply_augmentations(
        [system], {target_name: target}, [matrix], extra_data={}
    )

    expected = torch.einsum("ai,bj,ck,sijkp->sabcp", matrix, matrix, matrix, values)
    torch.testing.assert_close(new_targets[target_name].block().values, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical_atomicbasis(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian in the coupled basis
    (rank-1 tensors with an atomic basis) consistent with targets computed from DFT.

    Previously this raised ValueError; metatomic's transform_tensor now handles
    atomic basis targets via per-block row-index indexing.
    """
    target_name = "mtt::hamiltonian_nodes"
    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)

    dataset_info = _dataset_info(
        target_name,
        {"spherical": {"product": "coupled", "irreps": _HAMILTONIAN_IRREPS}},
        "atom",
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


def test_rotation_after_atomic_basis_prepare_transform():
    """Regression test for a real collate-order interaction bug: the atomic-basis
    "prepare" transform used to reindex targets to a batch-local 0, ..., n-1 numbering
    before augmentation ran, while everything else kept the absolute dataset "system"
    ids untouched. When a batch was drawn from a dataset larger than the batch (so
    absolute ids != local positions), this mismatch made ``O3Augmenter`` raise
    a ``ValueError``.

    This reproduces the exact PET/space collate order (atomic-basis "prepare"
    transform, then augmentation, then the reverse transform) with non-trivial,
    non-contiguous absolute "system" ids, and checks the final result against the
    DFT-rotated reference.
    """
    target_name = "mtt::hamiltonian_nodes"
    batch_size = 2

    # Absolute dataset ids this batch is drawn from: deliberately not 0, ..., n-1, to
    # catch any code that (still) assumes local batch numbering.
    system_ids = torch.tensor([3, 8])

    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)
    fX = _relabel_system_samples(fX, system_ids)
    fRX = _relabel_system_samples(fRX, system_ids)

    system_index_extra = TensorMap(
        keys=Labels(names=["_"], values=torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=system_ids.reshape(-1, 1).to(torch.float64),
                samples=Labels(names=["system"], values=system_ids.reshape(-1, 1)),
                components=[],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            )
        ],
    )

    dataset_info = _dataset_info(
        target_name,
        {"spherical": {"product": "coupled", "irreps": _HAMILTONIAN_IRREPS}},
        "atom",
    )

    atomic_basis_transform, atomic_basis_reverse_transform = (
        get_prepare_atomic_basis_targets_transform(dataset_info.targets, {})
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # Mirror the PET/space CollateFn order: prepare atomic-basis targets first, then
    # augment, then reverse the preparation.
    _, prepared_targets, _ = atomic_basis_transform(
        X, {target_name: fX}, {"mtt::aux::system_index": system_index_extra}
    )
    _, augmented_targets, _ = rotational_augmenter.apply_augmentations(
        X, prepared_targets, _transformation(batch_size), extra_data={}
    )
    _, reversed_targets, _ = atomic_basis_reverse_transform(X, augmented_targets, {})

    # Check that the rotated target matches the reference
    mts.allclose_raise(reversed_targets[target_name], fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical_rank2(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian in the uncoupled basis
    (rank-2 tensors with an atomic basis) consistent with targets computed from DFT.

    Previously this raised ValueError; metatomic's transform_tensor now handles
    atomic basis targets via per-block row-index indexing.
    """
    target_name = "mtt::hamiltonian_nodes_uncoupled"
    X, fX, fRX = _load_systems_and_targets(target_name, batch_size)

    dataset_info = _dataset_info(
        target_name,
        {"spherical": {"product": "cartesian", "irreps": _HAMILTONIAN_IRREPS}},
        "atom",
    )
    rotational_augmenter = O3Augmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)
