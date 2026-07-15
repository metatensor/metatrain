import gzip

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.composition import CompositionModel, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)

from . import DATASET_PATH


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


torch.set_default_dtype(torch.float64)  # the composition model only supports float64


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


def test_restart_no_new_targets_preserves_weights():
    """
    restart() with only composition-invalid targets, followed by a trainer run
    on data lacking the fitted target (as happens when finetuning a wrapping
    architecture on a new dataset), must not zero out the fitted weights.
    """
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    weights_before = model.model.weights["energy"].block().values.clone()

    vector_info = get_generic_target_info(
        "vector_target",
        {
            "quantity": "",
            "unit": "",
            "type": {"cartesian": {"rank": 1}},
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    model.restart(
        DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={"vector_target": vector_info},
        )
    )

    dataset_without_energy = Dataset.from_dict({"system": systems[:4]})
    trainer = Trainer(hypers={"batch_size": 2})
    trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset_without_energy],
        val_datasets=[dataset_without_energy],
        checkpoint_dir="",
    )

    torch.testing.assert_close(
        model.model.weights["energy"].block().values, weights_before
    )


def test_multi_dataset_fixed_and_fitted_targets():
    """
    With two datasets in a CombinedDataLoader, a first dataset whose only
    target has fixed weights must not stop the accumulation of the second
    dataset's target. Also exercises the default batch size with datasets of
    different lengths.
    """
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}

    systems_a, systems_b = systems[:6], systems[6:10]
    values_b = _make_synthetic_targets(systems_b, per_species_energies)
    dataset_a = Dataset.from_dict(
        {
            "system": systems_a,
            "energy_a": _build_target_tensormaps(
                systems_a, _make_synthetic_targets(systems_a, per_species_energies)
            ),
        }
    )
    dataset_b = Dataset.from_dict(
        {
            "system": systems_b,
            "energy_b": _build_target_tensormaps(systems_b, values_b),
        }
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy_a": get_energy_target_info("energy_a", {"unit": "eV"}),
            "energy_b": get_energy_target_info("energy_b", {"unit": "eV"}),
        },
    )
    model = CompositionModel(hypers={}, dataset_info=dataset_info)

    trainer = Trainer(hypers={"atomic_baseline": {"energy_a": 0.0}})
    trainer.train(
        model=model,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset_a, dataset_b],
        val_datasets=[dataset_a, dataset_b],
        checkpoint_dir="",
    )

    fitted_weights = model.model.weights["energy_b"].block().values
    fitted = dict(zip([1, 6, 7, 8], fitted_weights.flatten().tolist(), strict=True))
    for species, expected in per_species_energies.items():
        assert fitted[species] == pytest.approx(expected, abs=1e-8)


def test_checkpoint_roundtrip_predictions():
    """get_checkpoint() -> load_checkpoint() preserves predictions."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    model.eval()

    reference_output = model(systems[:5], {"energy": model.outputs["energy"]})

    loaded_model = CompositionModel.load_checkpoint(model.get_checkpoint(), "export")
    loaded_model.eval()

    loaded_output = loaded_model(
        systems[:5], {"energy": loaded_model.outputs["energy"]}
    )
    assert torch.allclose(
        reference_output["energy"].block().values,
        loaded_output["energy"].block().values,
    )


def test_regression_checkpoint():
    with gzip.open("regression_checkpoint.ckpt.gz", "rb") as fd:
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
