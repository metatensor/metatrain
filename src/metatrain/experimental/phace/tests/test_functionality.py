import metatensor.torch
import pytest
import torch
from jsonschema.exceptions import ValidationError
from metatensor.torch.atomistic import ModelOutput, NeighborListOptions, System
from omegaconf import OmegaConf

from metatrain.experimental.phace import PhACE
from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.data.readers import read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DEFAULT_HYPERS


@pytest.mark.parametrize("use_sphericart", [True, False])
@pytest.mark.parametrize("use_mops", [True, False])
def test_batched_prediction(use_sphericart, use_mops):
    """Test that predictions are the same no matter the batch size."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )

    DEFAULT_HYPERS["model"]["use_sphericart"] = use_sphericart
    DEFAULT_HYPERS["model"]["use_mops"] = use_mops
    model = PhACE(DEFAULT_HYPERS["model"], dataset_info)
    model = torch.jit.script(model)

    systems = read_systems(DATASET_PATH)[:8]
    systems = [system.to(torch.float32) for system in systems]
    nl_options = NeighborListOptions(cutoff=5.0, full_list=True)
    systems = [
        get_system_with_neighbor_lists(system, [nl_options]) for system in systems
    ]

    systems_1 = [
        [systems[0]],
        [systems[1]],
        [systems[2]],
        [systems[3]],
        [systems[4]],
        [systems[5]],
        [systems[6]],
        [systems[7]],
    ]
    systems_2 = [
        [systems[0], systems[1]],
        [systems[2], systems[3]],
        [systems[4], systems[5]],
        [systems[6], systems[7]],
    ]
    systems_4 = [
        [systems[0], systems[1], systems[2], systems[3]],
        [systems[4], systems[5], systems[6], systems[7]],
    ]
    systems_8 = [
        [
            systems[0],
            systems[1],
            systems[2],
            systems[3],
            systems[4],
            systems[5],
            systems[6],
            systems[7],
        ]
    ]

    energies_per_mode = []
    for system_mode in [systems_1, systems_2, systems_4, systems_8]:
        all_energies = []
        for systems in system_mode:
            energies = (
                model(
                    systems,
                    {"energy": model.outputs["energy"]},
                )["energy"]
                .block()
                .values
            )
            all_energies.append(energies)
        all_energies = torch.concatenate(all_energies)
        energies_per_mode.append(all_energies)

    assert torch.allclose(energies_per_mode[0], energies_per_mode[1])
    assert torch.allclose(energies_per_mode[0], energies_per_mode[2])
    assert torch.allclose(energies_per_mode[0], energies_per_mode[3])


@pytest.mark.parametrize("use_sphericart", [True, False])
@pytest.mark.parametrize("use_mops", [True, False])
@pytest.mark.parametrize("disable_nu_0", [True, False])
def test_isolated_atoms(disable_nu_0, use_sphericart, use_mops):
    """Test that predictions are correct for isolated atoms."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )

    DEFAULT_HYPERS["model"]["disable_nu_0"] = disable_nu_0
    DEFAULT_HYPERS["model"]["use_sphericart"] = use_sphericart
    DEFAULT_HYPERS["model"]["use_mops"] = use_mops
    model = PhACE(DEFAULT_HYPERS["model"], dataset_info)
    model.set_composition_weights(
        "energy",
        torch.tensor([123.0, 42.0, -42.0, 123.0]),
        [1, 6, 7, 8],
    )
    model = torch.jit.script(model)

    systems = [
        System(
            positions=torch.zeros((1, 3), dtype=torch.float32),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            types=torch.tensor([1]),
        ),
        System(
            positions=torch.zeros((1, 3), dtype=torch.float32),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            types=torch.tensor([6]),
        ),
        System(
            positions=torch.zeros((1, 3), dtype=torch.float32),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            types=torch.tensor([7]),
        ),
        System(
            positions=torch.zeros((1, 3), dtype=torch.float32),
            cell=torch.zeros((3, 3), dtype=torch.float32),
            types=torch.tensor([8]),
        ),
    ]
    systems = [system.to(torch.float32) for system in systems]
    nl_options = NeighborListOptions(cutoff=5.0, full_list=True)
    systems = [
        get_system_with_neighbor_lists(system, [nl_options]) for system in systems
    ]

    energies = (
        model(
            systems,
            {"energy": model.outputs["energy"]},
        )["energy"]
        .block()
        .values
    )

    allclose = torch.allclose(
        energies.squeeze(1),
        torch.tensor([123.0, 42.0, -42.0, 123.0]),
    )
    if disable_nu_0:
        assert allclose
    else:
        assert not allclose
