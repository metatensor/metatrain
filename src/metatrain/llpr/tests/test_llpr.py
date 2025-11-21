import shutil
import subprocess
import urllib.request
from pathlib import Path

import ase.io
import ase.units
import pytest
import torch
from ase.md import VelocityVerlet
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator


HERE = Path(__file__).parent


torch.manual_seed(42)


def test_llpr(monkeypatch, tmp_path):
    """
    Tests the functionalities of the LLPRUncertaintyModel from the CLI.
    """

    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    shutil.copy(HERE / "options-llpr.yaml", "options-llpr.yaml")
    shutil.copy(HERE / "options-pet-ft.yaml", "options-pet-ft.yaml")

    # 1. Train a PET model on a subset of the QM7 dataset:
    command = ["mtt", "train", "options-pet.yaml"]
    subprocess.check_call(command)

    # 2. Create the LLPR model, using `mtt train`:
    command = ["mtt", "train", "options-llpr.yaml", "-o", "model-llpr.pt"]
    subprocess.check_call(command)

    # 3. Check that the exported LLPR model from this training run works as intended,
    #    from an ASE calculator:
    for individual_outputs in [True, False]:
        for per_atom in [True, False]:
            check_exported_model_predictions(
                "model-llpr.pt", individual_outputs, per_atom
            )

    # 4. Check that a model exported from the checkpoint also works as intended
    command = ["mtt", "export", "model-llpr.ckpt"]
    subprocess.check_call(command)
    for individual_outputs in [True, False]:
        for per_atom in [True, False]:
            check_exported_model_predictions(
                "model-llpr.pt", individual_outputs, per_atom
            )

    # 5. Check that fine-tuning the PET model works through the LLPR wrapper:
    command = ["mtt", "train", "options-pet-ft.yaml"]
    subprocess.check_call(command)


def test_with_old_llpr_checkpoint(monkeypatch, tmp_path):
    """
    Tests the LLPR wrapper with PET-MAD v1.0.2, which was published using a LLPR
    checkpoint version 1 (before refactoring the LLPR into an architecture).
    """

    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet-ft.yaml", "options-pet-ft.yaml")

    # 1. Get the PET-MAD model checkpoint (v1.0.2):
    url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt"
    urllib.request.urlretrieve(url, "model-llpr.ckpt")

    # 2. Check that the LLPR model exported from the checkpoint works as intended
    command = ["mtt", "export", "model-llpr.ckpt"]
    subprocess.check_call(command)
    for individual_outputs in [True, False]:
        for per_atom in [True, False]:
            check_exported_model_predictions(
                "model-llpr.pt", individual_outputs, per_atom
            )

    # 3. Check that the ASE calculator with the exported model works:
    # (automatically checks LLPR uncertainties)
    calculator = MetatomicCalculator("model-llpr.pt")
    structure = ase.io.read("qm9_reduced_100.xyz")
    structure.calc = calculator
    dyn = VelocityVerlet(structure, 0.5 * ase.units.fs)
    dyn.run(10)
    calculator.run_model(structure, {"energy_ensemble": ModelOutput(per_atom=True)})

    # 4. Fine-tune the PET-MAD model through the LLPR wrapper:
    command = ["mtt", "train", "options-pet-ft.yaml"]
    subprocess.check_call(command)


def check_exported_model_predictions(
    exported_model_path: str,
    individual_outputs: bool,
    per_atom: bool,
    check_uncertainty_and_ensemble_consistency: bool = True,
):
    calc = MetatomicCalculator(exported_model_path)
    structures = ase.io.read("qm9_reduced_100.xyz", ":")
    if individual_outputs:
        predictions = {}
        predictions_energy = calc.run_model(
            structures, {"energy": ModelOutput(per_atom=per_atom)}
        )
        predictions_uncertainty = calc.run_model(
            structures, {"energy_uncertainty": ModelOutput(per_atom=per_atom)}
        )
        predictions_ensemble = calc.run_model(
            structures,
            {
                "energy_ensemble": ModelOutput(per_atom=per_atom),
            },
        )
        predictions = {
            **predictions_energy,
            **predictions_uncertainty,
            **predictions_ensemble,
        }
    else:
        outputs = {
            "energy": ModelOutput(per_atom=per_atom),
            "energy_uncertainty": ModelOutput(per_atom=per_atom),
            "energy_ensemble": ModelOutput(per_atom=per_atom),
        }
        predictions = calc.run_model(structures, outputs)

    if check_uncertainty_and_ensemble_consistency:
        energy = predictions["energy"].block().values
        uncertainty = predictions["energy_uncertainty"].block().values
        ensemble = predictions["energy_ensemble"].block().values
        assert torch.allclose(
            energy, ensemble.mean(dim=1, keepdim=True), atol=1e-5, rtol=1e-5
        )
        # require lower precision for PET-MAD which only has 128 ensemble members
        required_precision = 3e-2 if ensemble.shape[1] < 1000 else 1e-2
        assert torch.allclose(
            uncertainty,
            ensemble.std(dim=1, keepdim=True),
            atol=required_precision,
            rtol=required_precision,
        )


@pytest.mark.parametrize("train_backbone", [True, False])
def test_llpr_ensemble_training(monkeypatch, tmp_path, train_backbone):
    """
    Tests the functionalities of the LLPRUncertaintyModel from the CLI, with ensemble
    training enabled.
    """
    # Note that we will use check_uncertainty_and_ensemble_consistency=False since
    # training the ensemble breaks the consistency between LLPR and its ensemble

    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    options_file_name = (
        "options-llpr-ensemble-training-with-backbone.yaml"
        if train_backbone
        else "options-llpr-ensemble-training.yaml"
    )
    shutil.copy(HERE / options_file_name, options_file_name)
    shutil.copy(HERE / "options-pet-ft.yaml", "options-pet-ft.yaml")

    # 1. Train a PET model on a subset of the QM7 dataset:
    command = ["mtt", "train", "options-pet.yaml"]
    subprocess.check_call(command)

    # 2. Create the LLPR model, using `mtt train`:
    command = ["mtt", "train", options_file_name, "-o", "model-llpr.pt"]
    subprocess.check_call(command)

    # 3. Check that training continuation works:
    command = [
        "mtt",
        "train",
        options_file_name,
        "-o",
        "model-llpr.pt",
        "--restart",
        "auto",
    ]
    subprocess.check_call(command)

    # 4. Check that the exported LLPR model from this training run works as intended,
    #    from an ASE calculator:
    for individual_outputs in [True, False]:
        for per_atom in [True, False]:
            check_exported_model_predictions(
                "model-llpr.pt", individual_outputs, per_atom, False
            )

    # 5. Check that a model exported from the checkpoint also works as intended
    command = ["mtt", "export", "model-llpr.ckpt"]
    subprocess.check_call(command)
    for individual_outputs in [True, False]:
        for per_atom in [True, False]:
            check_exported_model_predictions(
                "model-llpr.pt", individual_outputs, per_atom, False
            )

    # 6. Check that fine-tuning the PET model works through the LLPR wrapper:
    command = ["mtt", "train", "options-pet-ft.yaml"]
    subprocess.check_call(command)
