import gzip

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.composition import CompositionModel
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


torch.set_default_dtype(torch.float64)


def _make_synthetic_targets(systems, per_species_energies):
    energies = []
    for system in systems:
        e = 0.0
        for species, energy in per_species_energies.items():
            count = int((system.types == species).sum())
            e += count * energy
        energies.append([e])
    return torch.tensor(energies)


def _build_target_tensormaps(systems, target_values):
    target_tensormaps = []
    for i in range(len(systems)):
        block = TensorBlock(
            values=target_values[i].reshape(1, 1),
            samples=Labels(["system"], torch.tensor([[i]])),
            components=[],
            properties=Labels(["_"], torch.tensor([[0]])),
        )
        target_tensormaps.append(TensorMap(Labels(["_"], torch.tensor([[0]])), [block]))
    return target_tensormaps


def _train_composition_model(systems, target_values):
    target_tensormaps = _build_target_tensormaps(systems, target_values)
    dataset = Dataset.from_dict({"system": systems, "energy": target_tensormaps})

    target_info = get_energy_target_info("energy", {"unit": "eV"})
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": target_info},
    )

    model = CompositionModel(hypers={}, dataset_info=dataset_info)
    model.train_model(
        datasets=[dataset],
        additive_models=[],
        batch_size=len(dataset),
        is_distributed=False,
    )
    return model, dataset_info


def test_composition_fitting_exact():
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)

    fitted_weights = model.model.weights["energy"].block().values
    assert fitted_weights.shape == (4, 1)

    fitted = {
        1: fitted_weights[0].item(),
        6: fitted_weights[1].item(),
        7: fitted_weights[2].item(),
        8: fitted_weights[3].item(),
    }

    for species, expected in per_species_energies.items():
        assert fitted[species] == pytest.approx(expected, abs=1e-10)


def test_composition_forward_after_fitting():
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    model.eval()

    output = model(systems[:5], {"energy": model.outputs["energy"]})
    assert "energy" in output
    predicted = output["energy"].block().values
    # model outputs per-atom energies; sum per system
    expected = target_values[:5]
    atom_offset = 0
    for i, system in enumerate(systems[:5]):
        n_atoms = len(system.types)
        system_pred = predicted[atom_offset : atom_offset + n_atoms].sum()
        assert system_pred == pytest.approx(expected[i].item(), abs=1e-10)
        atom_offset += n_atoms


def test_regression_checkpoint():
    with gzip.open("checkpoints/model-v1_trainer-v1.ckpt.gz", "rb") as fd:
        checkpoint = torch.load(fd, weights_only=False)
    model = CompositionModel.load_checkpoint(checkpoint, context="export")
    model.eval()

    systems = read_systems(DATASET_PATH)[:5]
    output = model(systems, {"mtt::U0": model.outputs["mtt::U0"]})

    block = output["mtt::U0"].block(0)
    system_energies = torch.zeros(len(systems))
    for i in range(len(systems)):
        mask = block.samples.column("system") == i
        system_energies[i] = block.values[mask].sum()

    expected = torch.tensor(
        [
            -40.486321238783184,
            -56.562378646683584,
            -76.42868639329494,
            -77.35308964488898,
            -93.42914705278939,
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(repr(system_energies.tolist()))
    torch.testing.assert_close(system_energies, expected, rtol=1e-10, atol=1e-10)
