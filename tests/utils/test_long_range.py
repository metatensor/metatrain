import pytest
import torch
from metatomic.torch import System, systems_to_torch

from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.long_range import LongRangeFeaturizer
from metatrain.utils.neighbor_lists import (
    NeighborListOptions,
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from ..conftest import RESOURCES_PATH


@pytest.mark.parametrize("periodicity", [True, False])
@pytest.mark.parametrize("architecture_name", ["pet", "soap_bpnn"])
def test_long_range(periodicity, architecture_name, tmpdir):
    """Tests that the long-range module can predict successfully."""

    if periodicity:
        filename = "carbon_reduced_100.xyz"
    else:
        filename = "ethanol_reduced_100.xyz"

    structures = read(RESOURCES_PATH / filename, ":10")
    systems = systems_to_torch(structures)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    hypers = get_default_hypers(architecture_name)
    hypers["model"]["long_range"]["enable"] = True
    hypers["model"]["long_range"]["use_ewald"] = True

    architecture = import_architecture(architecture_name)
    Model = architecture.__model__
    model = Model(hypers["model"], dataset_info)
    requested_nls = get_requested_neighbor_lists(model)

    systems = [
        get_system_with_neighbor_lists(system, requested_nls) for system in systems
    ]

    model(systems, {"energy": model.outputs["energy"]})

    # now torchscripted
    model = torch.jit.script(model)
    model(systems, {"energy": model.outputs["energy"]})

    # torch.jit.save and torch.jit.load
    with tmpdir.as_cwd():
        torch.jit.save(model, "model.pt")
        model = torch.jit.load("model.pt")
        model(systems, {"energy": model.outputs["energy"]})


@pytest.mark.parametrize("architecture_name", ["pet", "soap_bpnn"])
def test_long_range_1d_structure(architecture_name):
    """Tests that the long-range module can't handle reduced periodicity."""
    structures = read(RESOURCES_PATH / "carbon_reduced_100.xyz", ":1")
    structures[0].pbc = [True, False, False]
    structures[0].cell = [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    systems = systems_to_torch(structures)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    hypers = get_default_hypers(architecture_name)
    hypers["model"]["long_range"]["enable"] = True
    hypers["model"]["long_range"]["use_ewald"] = True

    architecture = import_architecture(architecture_name)
    Model = architecture.__model__
    model = Model(hypers["model"], dataset_info)
    requested_nls = get_requested_neighbor_lists(model)

    systems = [
        get_system_with_neighbor_lists(system, requested_nls) for system in systems
    ]

    with pytest.raises(
        NotImplementedError,
        match=(
            "Long-range featurizer does not support systems with only one or two "
            "periodic dimensions."
        ),
    ):
        model(systems, {"energy": model.outputs["energy"]})


def test_batched_long_range_matches_loop():
    """The eager periodic Ewald path should match the original per-system loop."""

    pytest.importorskip("torchpme")

    nl_options = NeighborListOptions(cutoff=4.4, full_list=True, strict=True)
    featurizer = LongRangeFeaturizer(
        hypers={
            "enable": True,
            "use_ewald": True,
            "smearing": 1.4,
            "kspace_resolution": 1.33,
            "interpolation_nodes": 5,
            "num_charges": None,
            "prefactor": 1.5,
            "every_layer": False,
        },
        feature_dim=8,
        neighbor_list_options=nl_options,
    )
    featurizer.train()

    systems = [
        System(
            types=torch.tensor([11, 17]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
            cell=torch.eye(3) * 6.0,
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            types=torch.tensor([19, 35, 35]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.8, 1.8, 1.8], [0.9, 0.9, 0.9]]),
            cell=torch.eye(3) * 7.0,
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    systems = [
        get_system_with_neighbor_lists(system, [nl_options]) for system in systems
    ]

    features = torch.randn(sum(len(system) for system in systems), 8)

    normalized_features = featurizer.norm(features)
    charges = featurizer.charges_map(normalized_features)
    expected = normalized_features + featurizer.out_projection(
        featurizer.prefactor * featurizer._loop_forward(systems, charges)
    )
    actual = featurizer(systems, features, torch.empty(0))

    torch.testing.assert_close(actual, expected)


def test_partitioned_batched_long_range_matches_loop():
    """Mixed periodic and non-periodic batches should match the loop fallback."""

    pytest.importorskip("torchpme")

    nl_options = NeighborListOptions(cutoff=4.4, full_list=True, strict=True)
    featurizer = LongRangeFeaturizer(
        hypers={
            "enable": True,
            "use_ewald": True,
            "smearing": 1.4,
            "kspace_resolution": 1.33,
            "interpolation_nodes": 5,
            "num_charges": None,
            "prefactor": 1.5,
            "every_layer": False,
        },
        feature_dim=8,
        neighbor_list_options=nl_options,
    )
    featurizer.train()

    systems = [
        System(
            types=torch.tensor([11, 17]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
            cell=torch.eye(3) * 6.0,
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            types=torch.tensor([6, 8, 1]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]]),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        ),
        System(
            types=torch.tensor([19, 35, 35]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.8, 1.8, 1.8], [0.9, 0.9, 0.9]]),
            cell=torch.eye(3) * 7.0,
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    systems = [
        get_system_with_neighbor_lists(system, [nl_options]) for system in systems
    ]

    features = torch.randn(sum(len(system) for system in systems), 8)

    normalized_features = featurizer.norm(features)
    charges = featurizer.charges_map(normalized_features)
    expected = normalized_features + featurizer.out_projection(
        featurizer.prefactor * featurizer._loop_forward(systems, charges)
    )
    actual = featurizer(systems, features, torch.empty(0))

    torch.testing.assert_close(actual, expected)


def test_long_range_2d_structure_raises():
    """The long-range featurizer should reject 2D-periodic systems."""

    pytest.importorskip("torchpme")

    nl_options = NeighborListOptions(cutoff=4.4, full_list=True, strict=True)
    featurizer = LongRangeFeaturizer(
        hypers={
            "enable": True,
            "use_ewald": True,
            "smearing": 1.4,
            "kspace_resolution": 1.33,
            "interpolation_nodes": 5,
            "num_charges": None,
            "prefactor": 1.5,
            "every_layer": False,
        },
        feature_dim=8,
        neighbor_list_options=nl_options,
    )
    featurizer.train()

    system = System(
        types=torch.tensor([6, 8, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]]),
        cell=torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]),
        pbc=torch.tensor([True, True, False]),
    )
    system = get_system_with_neighbor_lists(system, [nl_options])
    features = torch.randn(len(system), 8)
    with pytest.raises(
        NotImplementedError,
        match=(
            "Long-range featurizer does not support systems with only one or two "
            "periodic dimensions."
        ),
    ):
        featurizer([system], features, torch.empty(0))


def test_long_range_num_charges_hyperparameter():
    """The latent charge width should control the charge and output projections."""

    pytest.importorskip("torchpme")

    featurizer = LongRangeFeaturizer(
        hypers={
            "enable": True,
            "use_ewald": True,
            "smearing": 1.4,
            "kspace_resolution": 1.33,
            "interpolation_nodes": 5,
            "num_charges": 3,
            "prefactor": 1.5,
            "every_layer": False,
        },
        feature_dim=8,
        neighbor_list_options=NeighborListOptions(
            cutoff=4.4, full_list=True, strict=True
        ),
    )

    assert featurizer.charges_map[2].out_features == 3
    assert featurizer.out_projection[0].in_features == 3
    assert featurizer.out_projection[0].out_features == 8
