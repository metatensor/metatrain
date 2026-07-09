import copy

import metatensor.torch as mts
import pytest
import torch
from metatomic.torch import (
    ModelOutput,
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)

from metatrain.pet import PET
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _make_model(hypers=MODEL_HYPERS):
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    return PET(hypers, dataset_info).eval()


def _make_periodic_system():
    return System(
        types=torch.tensor([8, 1, 1, 6]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.9, 1.1],
                [2.1, 1.9, 0.3],
                [1.2, 3.1, 2.4],
            ]
        ),
        cell=4.0 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )


def _attach_non_strict_neighbor_list(system, nl_options):
    """The list is computed with a larger cutoff, so it contains pairs beyond
    ``nl_options.cutoff``
    """
    larger_options = NeighborListOptions(
        cutoff=nl_options.cutoff + 3.0,
        full_list=nl_options.full_list,
        strict=False,
    )
    scratch = _make_periodic_system()
    scratch = get_system_with_neighbor_lists(scratch, [larger_options])
    padded_nl = mts.detach_block(scratch.get_neighbor_list(larger_options))

    register_autograd_neighbors(system, padded_nl)
    system.add_neighbor_list(nl_options, padded_nl)
    return system


def test_requested_neighbor_list_is_not_strict():
    """By default PET requests a non-strict NL"""
    model = _make_model()
    assert model.requested_neighbor_lists()[0].strict is False


def test_requested_neighbor_list_is_strict_with_long_range():
    """With long-range features enabled, PET requests a strict NL"""
    pytest.importorskip("torchpme")
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["long_range"]["enable"] = True
    hypers["long_range"]["use_ewald"] = True
    model = _make_model(hypers)
    assert model.requested_neighbor_lists()[0].strict is True


def test_non_strict_nl_matches_strict_nl():
    """Predictions from a non-strict NL (with extra pairs beyond the cutoff) match those
    from a strict one, for both energies and forces"""
    model = _make_model()
    nl_options = model.requested_neighbor_lists()[0]

    strict_system = get_system_with_neighbor_lists(
        _make_periodic_system(), [nl_options]
    )
    non_strict_system = _attach_non_strict_neighbor_list(
        _make_periodic_system(), nl_options
    )

    n_strict_pairs = len(strict_system.get_neighbor_list(nl_options).samples)
    n_non_strict_pairs = len(non_strict_system.get_neighbor_list(nl_options).samples)
    assert n_non_strict_pairs > n_strict_pairs

    outputs = {"energy": ModelOutput(sample_kind="atom")}
    energies = []
    forces = []
    for system, nl_is_strict in [(strict_system, True), (non_strict_system, False)]:
        model.backend.nl_is_strict = nl_is_strict
        positions = system.positions.detach().clone().requires_grad_(True)
        new_system = System(
            types=system.types,
            positions=positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        for options in system.known_neighbor_lists():
            neighbors = mts.detach_block(system.get_neighbor_list(options))
            register_autograd_neighbors(new_system, neighbors)
            new_system.add_neighbor_list(options, neighbors)

        energy = model([new_system], outputs)["energy"].block().values
        gradient = torch.autograd.grad(energy.sum(), positions)[0]
        energies.append(energy.detach())
        forces.append(-gradient)

    torch.testing.assert_close(energies[0], energies[1])
    torch.testing.assert_close(forces[0], forces[1])


def test_non_strict_nl_does_not_inflate_nef_tensors():
    """The out-of-cutoff pairs of a non-strict NL are filtered out"""
    model = _make_model()
    nl_options = model.requested_neighbor_lists()[0]

    strict_system = get_system_with_neighbor_lists(
        _make_periodic_system(), [nl_options]
    )
    non_strict_system = _attach_non_strict_neighbor_list(
        _make_periodic_system(), nl_options
    )

    batch_data = []
    for system, nl_is_strict in [(strict_system, True), (non_strict_system, False)]:
        model.backend.nl_is_strict = nl_is_strict
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            _sample_labels,
        ) = concatenate_structures([system], nl_options)
        batch_data.append(
            model.backend.preprocess(
                positions,
                centers,
                neighbors,
                species,
                cells,
                cell_shifts,
                system_indices,
                model.cutoff_width_adaptive,
            )
        )
    strict_data, non_strict_data = batch_data

    for key in strict_data:
        assert strict_data[key].shape == non_strict_data[key].shape, key

    real_distances = non_strict_data["edge_distances"][non_strict_data["padding_mask"]]
    assert bool((real_distances <= model.cutoff + 1e-10).all())
