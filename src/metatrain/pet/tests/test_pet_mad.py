import shutil
import subprocess
import warnings
from urllib.parse import urlparse
from urllib.request import urlretrieve

import ase.io
import pytest
import torch
from metatomic.torch import ModelOutput

from metatrain.utils.data import read_systems
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH


STABLE_VERSIONS = ["1.0.2", "1.5.0"]
HF_PATH = "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-{size}-v{version}.ckpt"
NUM_SYSTEMS = 5

FINETUNING_OPTIONS = """
architecture:
    name: pet
    training:
        finetune:
            read_from: model.ckpt
        batch_size: 5
        num_epochs: 1

training_set:
    systems:
        read_from: finetune.xyz
        length_unit: angstrom
    targets:
        energy:
            unit: eV

validation_set: 0.1
"""


def _get_expected_output(size, version):
    if size == "s":
        if version == "1.0.2":
            return torch.tensor(
                [
                    [-36.721317291260],
                    [-37.199848175049],
                    [-37.049076080322],
                    [-36.586517333984],
                    [-37.476135253906],
                ]
            )
        elif version == "1.5.0":
            return torch.tensor(
                [
                    [-37.733596801758],
                    [-38.187965393066],
                    [-38.050487518311],
                    [-37.594646453857],
                    [-38.423156738281],
                ]
            )
    elif size == "xs":
        if version == "1.5.0":
            return torch.tensor(
                [
                    [-37.813751220703],
                    [-38.242935180664],
                    [-38.089038848877],
                    [-37.688468933105],
                    [-38.564857482910],
                ]
            )
    else:
        raise ValueError(f"Unknown version: {version} and size: {size}")


@pytest.mark.parametrize("version", STABLE_VERSIONS)
def test_pet_mad_consistency(version, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if version == "1.0.2":
        size = "s"
    elif version == "1.5.0":
        size = "xs"
    path = HF_PATH.format(size=size, version=version)
    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=UserWarning,
        )
        pet_mad_model = load_model(path).model.eval()

    systems = read_systems(DATASET_WITH_FORCES_PATH)[:NUM_SYSTEMS]
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(system, pet_mad_model.requested_neighbor_lists())

    systems = [system.to(torch.float32) for system in systems]

    outputs = {"energy": ModelOutput(sample_kind="system")}

    predictions = pet_mad_model(systems, outputs)

    expected_output = _get_expected_output(size=size, version=version)

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print("VERSION", version)
    # print(predictions["energy"].block().values[:5])

    torch.testing.assert_close(
        predictions["energy"].block().values[:NUM_SYSTEMS], expected_output
    )


@pytest.mark.parametrize("version", STABLE_VERSIONS)
def test_pet_mad_finetuning(version, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if version == "1.0.2":
        size = "s"
    elif version == "1.5.0":
        size = "xs"
    path = HF_PATH.format(size=size, version=version)
    if urlparse(path).scheme:
        path, _ = urlretrieve(path)

    # copy dataset with forces to here
    dataset = ase.io.read(DATASET_WITH_FORCES_PATH, index=f":{NUM_SYSTEMS}")
    ase.io.write(tmp_path / "finetune.xyz", dataset, format="extxyz")
    shutil.copy(path, tmp_path / "model.ckpt")

    with open(tmp_path / "finetune.yaml", "w") as f:
        f.write(FINETUNING_OPTIONS)

    subprocess.run(["mtt", "train", "finetune.yaml"], check=True)
