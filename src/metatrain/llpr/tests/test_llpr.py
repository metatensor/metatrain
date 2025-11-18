import shutil
import subprocess
import urllib.request
from pathlib import Path

import ase.io
import torch
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

    # 3. Fine-tune the PET-MAD model through the LLPR wrapper:
    command = ["mtt", "train", "options-pet-ft.yaml"]
    subprocess.check_call(command)


def check_exported_model_predictions(
    exported_model_path: str, individual_outputs: bool, per_atom: bool
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
                "energy": ModelOutput(per_atom=per_atom),
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
