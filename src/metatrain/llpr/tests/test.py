import subprocess

from metatomic.torch import ModelOutput
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.utils.architectures import get_default_hypers, import_architecture
from pathlib import Path
import ase.io

HERE = Path(__file__).parent
import shutil


torch.manual_seed(42)


def test_llpr(monkeypatch, tmp_path):
    """
    Tests the functionalities of the LLPRUncertaintyModel, mainly from the CLI.
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
    calc = MetatomicCalculator("model-llpr.pt")
    structures = ase.io.read("qm9_reduced_100.xyz", ":5")
    outputs = {"energy": ModelOutput(per_atom=False), "energy_uncertainty": ModelOutput(per_atom=False), "energy_ensemble": ModelOutput(per_atom=False)}
    predictions = calc.run_model(structures, outputs)
    energy = predictions["energy"].block().values
    uncertainty = predictions["energy_uncertainty"].block().values
    ensemble = predictions["energy_ensemble"].block().values

    print(uncertainty)
    print(ensemble.std(dim=1, keepdim=True))

    assert torch.allclose(energy, ensemble.mean(dim=1, keepdim=True), atol=1e-5, rtol=1e-5)
    assert torch.allclose(uncertainty, ensemble.std(dim=1, keepdim=True), atol=1e-5, rtol=1e-5)

    raise ValueError()
