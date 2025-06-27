import logging
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch import load as metatensor_load
from metatomic.torch import NeighborListOptions, systems_to_torch
from omegaconf import OmegaConf

from metatrain.cli.eval import eval_model
from metatrain.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, DiskDataset, DiskDatasetWriter
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import EVAL_OPTIONS_PATH, MODEL_HYPERS, MODEL_PATH, RESOURCES_PATH


@pytest.fixture
def model():
    return torch.jit.load(MODEL_PATH)


@pytest.fixture
def options():
    return OmegaConf.load(EVAL_OPTIONS_PATH)


def test_eval_cli(monkeypatch, tmp_path):
    """Test succesful run of the eval script via the CLI with default arguments"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    command = [
        "mtt",
        "eval",
        str(MODEL_PATH),
        str(EVAL_OPTIONS_PATH),
        "-e",
        str(RESOURCES_PATH / "extensions"),
        "--check-consistency",
    ]

    output = subprocess.check_output(command, stderr=subprocess.STDOUT)

    assert b"energy RMSE" in output

    assert Path("output.xyz").is_file()


@pytest.mark.parametrize("model_name", ["model-32-bit.pt", "model-64-bit.pt"])
def test_eval(monkeypatch, tmp_path, caplog, model_name, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    model = torch.jit.load(RESOURCES_PATH / model_name)

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    frames[0].info["energy"]


@pytest.mark.parametrize("model_name", ["model-32-bit.pt", "model-64-bit.pt"])
def test_eval_batch_size(monkeypatch, tmp_path, caplog, model_name, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    model = torch.jit.load(RESOURCES_PATH / model_name)

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
        batch_size=13,
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "evaluation time" in log
    assert "ms per atom" in log
    assert "inaccurate average timings" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    frames[0].info["energy"]


def test_eval_export(monkeypatch, tmp_path, options):
    """Test evaluation of a trained model exported but not saved to disk."""
    monkeypatch.chdir(tmp_path)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6, 7, 8},
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    exported_model = model.export()

    eval_model(
        model=exported_model,
        options=options,
        output="foo.xyz",
        check_consistency=True,
    )


def test_eval_multi_dataset(monkeypatch, tmp_path, caplog, model, options):
    """Test that eval runs for multiple datasets should be evaluated."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    eval_model(
        model=model,
        options=OmegaConf.create([options, options]),
        output="foo.xyz",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "index 0" in log
    assert "index 1" in log

    # Test file is written predictions
    for i in range(2):
        frames = read(f"foo_{i}.xyz", ":")
        frames[0].info["energy"]


def test_eval_no_targets(monkeypatch, tmp_path, model, options):
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")

    eval_model(
        model=model,
        options=options,
        check_consistency=True,
    )

    assert Path("output.xyz").is_file()


@pytest.mark.parametrize("suffix", [".zip", ".mts"])
def test_eval_disk_dataset(monkeypatch, tmp_path, caplog, suffix):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    model = torch.jit.load(RESOURCES_PATH / "model-32-bit.pt")

    options = OmegaConf.create(
        {
            "systems": {"read_from": "qm9_reduced_100.zip"},
            "targets": {"energy": {"read_from": "qm9_reduced_100.zip"}},
        }
    )

    # Write a disk dataset
    disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
    for i in range(100):
        frame = read("qm9_reduced_100.xyz", index=i)
        system = systems_to_torch(frame, dtype=torch.float64)
        system = get_system_with_neighbor_lists(
            system,
            [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
        )
        energy = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[frame.info["U0"]]], dtype=torch.float64),
                    samples=Labels(
                        names=["system"],
                        values=torch.tensor([[i]]),
                    ),
                    components=[],
                    properties=Labels("energy", torch.tensor([[0]])),
                )
            ],
        )
        disk_dataset_writer.write_sample(system, {"energy": energy})
    del disk_dataset_writer

    eval_model(
        model=model,
        options=options,
        output=f"foo{suffix}",
        check_consistency=True,
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE (per atom)" in log
    assert "energy MAE (per atom)" in log
    assert "dataset with index" not in log
    assert "evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    if suffix == ".mts":
        pred = metatensor_load("foo_energy.mts")
        assert pred.keys == Labels(["_"], torch.tensor([[0]]))
    else:
        pred = DiskDataset("foo.zip")
        assert pred[0]["energy"].keys == Labels(["_"], torch.tensor([[0]]))
