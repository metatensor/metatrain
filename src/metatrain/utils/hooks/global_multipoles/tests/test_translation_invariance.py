"""Origin handling of the global multipoles hook.

The hook builds the global dipole as :math:`\\sum_i q_i \\mathbf{r}_i + \\sum_i
\\mathbf{p}_i`, which is origin-dependent whenever the charges do not sum to zero. With
``origin="center_of_charge"`` the positions are referred to each system's centre of
nuclear charge, which removes that dependence; with ``origin="absolute"`` they are not,
and the dependence is kept.

These tests feed the hook random charges and local dipoles directly, standing in for the
per-atom output of a translationally invariant architecture: such a model returns the
same per-atom values for a system and for a rigidly translated copy of it, so the same
inputs are reused for both and only the positions change.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hooks.helpers import load_hook


OUTPUT_NAME = "mtt::global_dipole"
INPUT_NAME = "mtt::aux::local_multipoles::global_dipole"

ATOMIC_TYPES = [1, 6, 7, 8]
SYSTEM_SIZES = [4, 11, 7]


def _dataset_info() -> DatasetInfo:
    """A dataset info holding a single global dipole target."""
    global_dipole = get_generic_target_info(
        OUTPUT_NAME,
        dict(
            sample_kind="system",
            unit="",
            quantity="",
            num_subtargets=1,
            type=dict(spherical=dict(irreps=[{"o3_lambda": 1, "o3_sigma": 1}])),
        ),
    )
    return DatasetInfo(
        length_unit="angstrom",
        atomic_types=ATOMIC_TYPES,
        targets={OUTPUT_NAME: global_dipole},
    )


def _random_systems(generator: torch.Generator) -> list[System]:
    """Random systems, deliberately far from the origin of their frame."""
    systems = []
    for size in SYSTEM_SIZES:
        types = torch.tensor(ATOMIC_TYPES, dtype=torch.int32)[
            torch.randint(len(ATOMIC_TYPES), (size,), generator=generator)
        ]
        positions = (
            torch.rand((size, 3), generator=generator, dtype=torch.float64) * 4.0
            + torch.tensor([7.0, -3.0, 11.0], dtype=torch.float64)
        )
        systems.append(
            System(
                types=types,
                positions=positions,
                cell=torch.zeros((3, 3), dtype=torch.float64),
                pbc=torch.zeros(3, dtype=torch.bool),
            )
        )
    return systems


def _random_local_multipoles(
    systems: list[System], generator: torch.Generator
) -> dict[str, TensorMap]:
    """Random per-atom charges and local dipoles.

    These stand in for what a translationally invariant architecture would
    predict: they depend on the systems only through their internal geometry,
    so a rigid translation leaves them unchanged.
    """
    samples = Labels(
        names=["system", "atom"],
        values=torch.tensor(
            [
                [system_index, atom_index]
                for system_index, system in enumerate(systems)
                for atom_index in range(len(system))
            ]
        ),
    )
    total_atoms = sum(len(system) for system in systems)

    # Charges deliberately not centred on zero, so that the sum over each
    # system is far from neutral and the origin genuinely matters.
    charges = (
        torch.rand((total_atoms, 1, 1), generator=generator, dtype=torch.float64) + 0.5
    )
    local_dipoles = torch.rand(
        (total_atoms, 3, 1), generator=generator, dtype=torch.float64
    )

    keys = Labels(
        names=["o3_lambda", "o3_sigma"], values=torch.tensor([[0, 1], [1, 1]])
    )
    blocks = [
        TensorBlock(
            values=charges,
            samples=samples,
            components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
            properties=Labels(names=["properties"], values=torch.tensor([[0]])),
        ),
        TensorBlock(
            values=local_dipoles,
            samples=samples,
            components=[
                Labels(names=["o3_mu"], values=torch.tensor([[-1], [0], [1]]))
            ],
            properties=Labels(names=["properties"], values=torch.tensor([[0]])),
        ),
    ]
    return {INPUT_NAME: TensorMap(keys=keys, blocks=blocks)}


def _translate(systems: list[System], shift: torch.Tensor) -> list[System]:
    """Rigidly translate every system by the same vector."""
    return [
        System(
            types=system.types,
            positions=system.positions + shift,
            cell=system.cell,
            pbc=system.pbc,
        )
        for system in systems
    ]


def _make_hook(origin: str = "center_of_charge"):
    """Build the hook with a given origin convention."""
    hypers = {"outputs": OUTPUT_NAME}
    if origin is not None:
        hypers["origin"] = origin
    return load_hook("global_multipoles")(hypers, _dataset_info())


@pytest.fixture
def hook():
    return _make_hook("center_of_charge")


@pytest.fixture
def absolute_hook():
    return _make_hook("absolute")


@pytest.fixture
def outputs():
    return {OUTPUT_NAME: ModelOutput(quantity="", unit="", sample_kind="system")}


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


def test_charges_are_not_neutral(generator):
    """The test is only meaningful if the origin actually matters.

    A translation changes the raw sum by ``sum_i q_i * a``, so if the random
    charges happened to sum to zero the invariance check below would pass even
    without an origin being subtracted.
    """
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)
    charges = inputs[INPUT_NAME].block(0).values.reshape(-1)

    start = 0
    for system in systems:
        total = charges[start : start + len(system)].sum()
        assert abs(float(total)) > 1.0
        start += len(system)


@pytest.mark.parametrize(
    "shift",
    [
        [100.0, 0.0, 0.0],
        [-13.0, 7.0, -21.0],
        [1e4, 1e4, 1e4],
    ],
)
def test_translation_invariance(hook, outputs, generator, shift):
    """Translating every system must leave the global dipole unchanged."""
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)
    translated = _translate(systems, torch.tensor(shift, dtype=torch.float64))

    original = hook(systems, outputs, inputs)[OUTPUT_NAME].block(0).values
    moved = hook(translated, outputs, inputs)[OUTPUT_NAME].block(0).values

    torch.testing.assert_close(original, moved, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("origin", ["center_of_charge", "absolute"])
def test_independent_of_batching(outputs, generator, origin):
    """The origin is per system, so batching must not change the result."""
    hook = _make_hook(origin)
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)

    batched = hook(systems, outputs, inputs)[OUTPUT_NAME].block(0).values

    one_at_a_time = []
    start = 0
    for index, system in enumerate(systems):
        block = inputs[INPUT_NAME]
        stop = start + len(system)
        single = {
            INPUT_NAME: TensorMap(
                keys=block.keys,
                blocks=[
                    TensorBlock(
                        values=b.values[start:stop],
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor(
                                [[0, atom] for atom in range(len(system))]
                            ),
                        ),
                        components=b.components,
                        properties=b.properties,
                    )
                    for b in block.blocks()
                ],
            )
        }
        one_at_a_time.append(
            hook([system], outputs, single)[OUTPUT_NAME].block(0).values
        )
        start = stop

    torch.testing.assert_close(
        batched, torch.cat(one_at_a_time, dim=0), rtol=0.0, atol=1e-9
    )


def test_matches_centre_of_nuclear_charge(hook, outputs, generator):
    """The global dipole is built about each system's centre of nuclear charge."""
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)

    predicted = hook(systems, outputs, inputs)[OUTPUT_NAME].block(0).values

    block = inputs[INPUT_NAME]
    charges = block.block(0).values.reshape(-1)
    local_dipoles = block.block(1).values.squeeze(-1)

    expected, start = [], 0
    for system in systems:
        stop = start + len(system)
        positions = system.positions
        nuclear_charges = system.types.to(positions.dtype)
        origin = (nuclear_charges.unsqueeze(1) * positions).sum(0) / nuclear_charges.sum()
        # (x, y, z) -> (y, z, x), the spherical harmonics convention
        shifted = (positions - origin)[:, [1, 2, 0]]
        expected.append(
            (charges[start:stop].unsqueeze(1) * shifted).sum(0)
            + local_dipoles[start:stop].sum(0)
        )
        start = stop

    torch.testing.assert_close(
        predicted.squeeze(-1), torch.stack(expected), rtol=1e-10, atol=1e-10
    )


def test_center_of_charge_is_the_default(outputs, generator):
    """Omitting the hyperparameter must give the origin-independent behaviour."""
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)

    default = _make_hook(None)(systems, outputs, inputs)[OUTPUT_NAME].block(0).values
    explicit = (
        _make_hook("center_of_charge")(systems, outputs, inputs)[OUTPUT_NAME]
        .block(0)
        .values
    )

    torch.testing.assert_close(default, explicit, rtol=0.0, atol=0.0)


def test_absolute_matches_raw_positions(absolute_hook, outputs, generator):
    """With ``origin="absolute"`` the positions are used exactly as stored."""
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)

    predicted = absolute_hook(systems, outputs, inputs)[OUTPUT_NAME].block(0).values

    block = inputs[INPUT_NAME]
    charges = block.block(0).values.reshape(-1)
    local_dipoles = block.block(1).values.squeeze(-1)

    expected, start = [], 0
    for system in systems:
        stop = start + len(system)
        # (x, y, z) -> (y, z, x), the spherical harmonics convention
        positions = system.positions[:, [1, 2, 0]]
        expected.append(
            (charges[start:stop].unsqueeze(1) * positions).sum(0)
            + local_dipoles[start:stop].sum(0)
        )
        start = stop

    torch.testing.assert_close(
        predicted.squeeze(-1), torch.stack(expected), rtol=1e-10, atol=1e-10
    )


def test_absolute_shifts_by_the_total_charge(absolute_hook, outputs, generator):
    """Without an origin the prediction moves by exactly ``sum_i q_i * a``.

    This pins down what ``origin="absolute"`` gives up: the residual is not
    merely non-zero, it is the total charge times the translation.
    """
    shift = torch.tensor([-13.0, 7.0, -21.0], dtype=torch.float64)
    systems = _random_systems(generator)
    inputs = _random_local_multipoles(systems, generator)
    translated = _translate(systems, shift)

    original = absolute_hook(systems, outputs, inputs)[OUTPUT_NAME].block(0).values
    moved = absolute_hook(translated, outputs, inputs)[OUTPUT_NAME].block(0).values

    charges = inputs[INPUT_NAME].block(0).values.reshape(-1)
    totals, start = [], 0
    for system in systems:
        stop = start + len(system)
        totals.append(charges[start:stop].sum())
        start = stop

    # (x, y, z) -> (y, z, x), the spherical harmonics convention
    expected = torch.stack(totals).unsqueeze(1) * shift[[1, 2, 0]].unsqueeze(0)

    torch.testing.assert_close(
        (moved - original).squeeze(-1), expected, rtol=1e-9, atol=1e-9
    )
