import glob
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import ase.io
import pytest
import torch
from jsonschema.exceptions import ValidationError
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, systems_to_torch
from omegaconf import OmegaConf

from metatrain import RANDOM_SEED
from metatrain.cli.train import _process_continue_from, train_model
from metatrain.utils.data import DiskDatasetWriter
from metatrain.utils.data.readers.ase import read
from metatrain.utils.errors import ArchitectureError
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import (
    DATASET_PATH_CARBON,
    DATASET_PATH_ETHANOL,
    DATASET_PATH_QM7X,
    DATASET_PATH_QM9,
    MODEL_PATH_64_BIT,
    OPTIONS_PATH,
    RESOURCES_PATH,
)
from .dump_spherical_targets import dump_spherical_targets


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train(capfd, monkeypatch, tmp_path, output):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["mtt", "train", "options.yaml"]

    if output is not None:
        command += ["-o", output]
    else:
        output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if restart_options.yaml file is written
    restart_glob = glob.glob("outputs/*/*/options_restart.yaml")
    assert len(restart_glob) == 1

    # Open restart options an check that default parameters are overwritten
    restart_options = OmegaConf.load(restart_glob[0])
    assert restart_options["architecture"]["training"]["num_epochs"] == 1

    # Test if logfile is written
    log_glob = glob.glob("outputs/*/*/train.log")
    assert len(log_glob) == 1

    model_name = "mymodel" if output == "mymodel.pt" else "model"

    # Test if the model is saved (both .pt and .ckpt)
    pt_glob = glob.glob(f"{model_name}.pt")
    assert len(pt_glob) == 1
    ckpt_glob = glob.glob(f"{model_name}.ckpt")
    assert len(ckpt_glob) == 1

    # Test if they are also saved to the outputs/ directory
    pt_glob = glob.glob(f"outputs/*/*/{model_name}.pt")
    assert len(pt_glob) == 1
    ckpt_glob = glob.glob(f"outputs/*/*/{model_name}.ckpt")
    assert len(ckpt_glob) == 1

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test if training indices are saved
    for subset in ["training", "validation", "test"]:
        subset_glob = glob.glob(f"outputs/*/*/indices/{subset}.txt")
        assert len(subset_glob) == 1

    # Open the log file and check if the logging is correct
    with open(log_glob[0]) as f:
        file_log = f.read()

    stdout_log = capfd.readouterr().out

    assert file_log == stdout_log

    assert "This log is also available" in stdout_log
    assert "Running training for 'soap_bpnn' architecture"
    assert re.search(r"Random seed of this run is [1-9]\d*", stdout_log)
    assert "Training dataset:" in stdout_log
    assert "Validation dataset:" in stdout_log
    assert "Test dataset:" in stdout_log
    assert "50 structures" in stdout_log
    assert "mean " in stdout_log
    assert "std " in stdout_log
    assert "[INFO]" in stdout_log
    assert "Epoch" in stdout_log
    assert "loss" in stdout_log
    assert "validation" in stdout_log
    assert "train" in stdout_log
    assert "energy" in stdout_log
    assert "with index" not in stdout_log  # index only printed for more than 1 dataset
    assert "Running final evaluation with batch size 2" in stdout_log


@pytest.mark.parametrize(
    "overrides",
    [
        ["architecture.training.num_epochs=2"],
        ["architecture.training.num_epochs=2", "architecture.training.batch_size=3"],
    ],
)
def test_command_line_override(monkeypatch, tmp_path, overrides):
    """Test that training options can be overwritten from the command line."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["mtt", "train", "options.yaml"]
    for override in overrides:
        command += ["-r", override]

    subprocess.check_call(command)

    restart_glob = glob.glob("outputs/*/*/options_restart.yaml")
    assert len(restart_glob) == 1

    restart_options = OmegaConf.load(restart_glob[0])
    assert restart_options["architecture"]["training"]["num_epochs"] == 2

    if len(overrides) == 2:
        assert restart_options["architecture"]["training"]["batch_size"] == 3


def test_train_from_options_restart_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    # run training with original options
    options = OmegaConf.load(OPTIONS_PATH)
    train_model(options)

    # run training with options_restart.yaml
    os.mkdir("outputs/")
    options_restart = OmegaConf.load("options_restart.yaml")
    train_model(options_restart, checkpoint_dir="outputs/")


def test_train_unknown_arch_options(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options_str = """
    architecture:
        name: soap_bpnn
        training:
            batch_size: 2
            num_epoch: 1

    training_set:
        systems:
            read_from: qm9_reduced_100.xyz
            length_unit: angstrom
        targets:
            energy:
            key: U0
            unit: eV

    test_set: 0.5
    validation_set: 0.1
    """
    options = OmegaConf.create(options_str)

    match = (
        r"Unrecognized options \('num_epoch' was unexpected\). "
        r"Do you mean 'num_epochs'?"
    )
    with pytest.raises(ValidationError, match=match):
        train_model(options)


@pytest.mark.parametrize("n_datasets", [1, 2])
@pytest.mark.parametrize("test_set_file", (True, False))
@pytest.mark.parametrize("validation_set_file", (True, False))
def test_train_explicit_validation_test(
    monkeypatch,
    tmp_path,
    caplog,
    n_datasets,
    test_set_file,
    validation_set_file,
    options,
):
    """Test that training via the training cli runs without an error raise
    also when the validation and test sets are provided explicitly."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    systems = read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    options["training_set"] = OmegaConf.create(n_datasets * [options["training_set"]])

    if validation_set_file:
        ase.io.write("validation.xyz", systems[50:80])
        options["validation_set"] = options["training_set"][0].copy()
        options["validation_set"]["systems"]["read_from"] = "validation.xyz"
        options["validation_set"] = OmegaConf.create(
            n_datasets * [options["validation_set"]]
        )

    if test_set_file:
        ase.io.write("test.xyz", systems[80:])
        options["test_set"] = options["training_set"][0].copy()
        options["test_set"]["systems"]["read_from"] = "test.xyz"
        options["test_set"] = OmegaConf.create(n_datasets * [options["test_set"]])

    train_model(options)

    # Test log messages which are written to STDOUT
    log = caplog.text
    for set_type in ["training", "test", "validation"]:
        for i in range(n_datasets):
            if n_datasets == 1:
                extra_log_message = ""
            else:
                extra_log_message = f" with index {i}"

            assert f"Evaluating {set_type} dataset{extra_log_message}" in log

    assert Path("model.pt").is_file()


def test_train_multiple_datasets(monkeypatch, tmp_path, options):
    """Test that training via the training cli runs without an error raise
    also when learning on two different datasets."""
    monkeypatch.chdir(tmp_path)

    systems_qm9 = ase.io.read(DATASET_PATH_QM9, ":")
    systems_ethanol = ase.io.read(DATASET_PATH_ETHANOL, ":")

    ase.io.write("qm9_reduced_100.xyz", systems_qm9[:50])
    ase.io.write("ethanol_reduced_100.xyz", systems_ethanol[:50])

    options["training_set"] = OmegaConf.create(2 * [options["training_set"]])
    options["training_set"][1]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"][1]["targets"]["energy"]["key"] = "energy"
    options["training_set"][0]["targets"].pop("energy")
    options["training_set"][0]["targets"]["mtt::U0"] = OmegaConf.create(
        {"key": "U0", "unit": "eV"}
    )

    train_model(options)


def test_empty_training_set(monkeypatch, tmp_path, options):
    """Test that an error is raised if no training set is provided."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.6
    options["test_set"] = 0.4

    with pytest.raises(
        ValueError, match="Fraction of the train set is smaller or equal to 0!"
    ):
        train_model(options)


@pytest.mark.parametrize("split", [-0.1, 1.1])
def test_wrong_test_split_size(split, monkeypatch, tmp_path, options):
    """Test that an error is raised if the test split has the wrong size"""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.1
    options["test_set"] = split

    if split > 1:
        match = rf"{split} is greater than or equal to the maximum of 1"
    if split < 0:
        match = rf"{split} is less than the minimum of 0"

    with pytest.raises(ValidationError, match=match):
        train_model(options)


@pytest.mark.parametrize("split", [0.0, 1.1])
def test_wrong_validation_split_size(split, monkeypatch, tmp_path, options):
    """Test that an error is raised if the validation split has the wrong size"""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = split
    options["test_set"] = 0.1

    if split > 1:
        match = rf"{split} is greater than or equal to the maximum of 1"
    if split <= 0:
        match = rf"{split} is less than or equal to the minimum of 0"

    with pytest.raises(ValidationError, match=match):
        train_model(options)


def test_empty_test_set(caplog, monkeypatch, tmp_path, options):
    """Test that NO error is raised if no test set is provided."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.4
    options["test_set"] = 0.0

    match = "Requested dataset of zero length. This dataset will be empty."
    with pytest.warns(UserWarning, match=match):
        train_model(options)

    # check if the logging is correct
    assert "This dataset is empty. No evaluation" in caplog.text


@pytest.mark.parametrize(
    "test_set_file, validation_set_file", [(True, False), (False, True)]
)
def test_unit_check_is_performed(
    monkeypatch,
    tmp_path,
    test_set_file,
    validation_set_file,
    options,
):
    """Test that error is raised if units are inconsistent between the datasets."""
    monkeypatch.chdir(tmp_path)

    systems = read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    if validation_set_file:
        ase.io.write("test.xyz", systems[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["systems"]["read_from"] = "test.xyz"
        options["validation_set"]["systems"]["length_unit"] = "foo"

    if test_set_file:
        ase.io.write("validation.xyz", systems[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["systems"]["read_from"] = "validation.xyz"
        options["test_set"]["systems"]["length_unit"] = "foo"

    with pytest.raises(ValueError, match="`length_unit`s are inconsistent"):
        train_model(options)


@pytest.mark.parametrize(
    "test_set_file, validation_set_file", [(True, False), (False, True)]
)
def test_inconsistent_number_of_datasets(
    monkeypatch, tmp_path, test_set_file, validation_set_file, options
):
    """Test that error is raised in inconsistent number datasets are provided.

    i.e one train dataset but two validation datasets. Same for the test dataset."""
    monkeypatch.chdir(tmp_path)

    systems = read(DATASET_PATH_QM9, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])

    if validation_set_file:
        ase.io.write("validation.xyz", systems[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["systems"]["read_from"] = "validation.xyz"
        options["validation_set"] = OmegaConf.create(2 * [options["validation_set"]])

    if test_set_file:
        ase.io.write("test.xyz", systems[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["systems"]["read_from"] = "test.xyz"
        options["test_set"] = OmegaConf.create(2 * [options["test_set"]])

    with pytest.raises(ValueError, match="different size than the training datatset"):
        train_model(options)


@pytest.mark.parametrize(
    "taining_set_file, test_set_file, validation_set_file",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_inconsistencies_within_list_datasets(
    monkeypatch,
    tmp_path,
    taining_set_file,
    test_set_file,
    validation_set_file,
    options,
):
    """Test error raise if inconsistency within one datasets config present."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    ref_dataset_conf = OmegaConf.create(2 * [options["training_set"]])
    broken_dataset_conf = ref_dataset_conf.copy()
    broken_dataset_conf[0]["systems"]["length_unit"] = "foo"
    broken_dataset_conf[1]["systems"]["length_unit"] = "bar"

    options["training_set"] = ref_dataset_conf
    options["validation_set"] = ref_dataset_conf
    options["test_set"] = ref_dataset_conf

    if taining_set_file:
        options["training_set"] = broken_dataset_conf
    if test_set_file:
        options["test_set"] = broken_dataset_conf
    if validation_set_file:
        options["validation_set"] = broken_dataset_conf

    with pytest.raises(ValueError, match="`length_unit`s are inconsistent"):
        train_model(options)


def test_continue(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    train_model(options, continue_from=MODEL_PATH_64_BIT)


def test_continue_auto(options, caplog, monkeypatch, tmp_path):
    """Test that continuing with the `auto` keyword results in
    a continuation from the most recent checkpoint."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    caplog.set_level(logging.INFO)

    # Make up an output directory with some checkpoints
    true_checkpoint_dir = Path("outputs/2021-09-02/00-10-05")
    true_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # as well as some lower-priority checkpoints
    fake_checkpoints_dirs = [
        Path("outputs/2021-08-01/00-00-00"),
        Path("outputs/2021-09-01/00-00-00"),
        Path("outputs/2021-09-02/00-00-00"),
        Path("outputs/2021-09-02/00-10-00"),
    ]
    for fake_checkpoint_dir in fake_checkpoints_dirs:
        fake_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 4):
        shutil.copy(MODEL_PATH_64_BIT, true_checkpoint_dir / f"model_{i}.ckpt")
        for fake_checkpoint_dir in fake_checkpoints_dirs:
            shutil.copy(MODEL_PATH_64_BIT, fake_checkpoint_dir / f"model_{i}.ckpt")

    train_model(options, continue_from=_process_continue_from("auto"))

    assert "Loading checkpoint from" in caplog.text
    assert str(true_checkpoint_dir) in caplog.text
    assert "model_3.ckpt" in caplog.text


def test_continue_auto_no_outputs(options, caplog, monkeypatch, tmp_path):
    """Test that continuing with the `auto` keyword results in
    training from scratch if `outputs/` is not present."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    caplog.set_level(logging.INFO)

    train_model(options, continue_from=_process_continue_from("auto"))

    assert "Loading checkpoint from" not in caplog.text


def test_continue_different_dataset(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise
    with a different dataset than the original."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")

    options["training_set"]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"

    train_model(options, continue_from=MODEL_PATH_64_BIT)


@pytest.mark.parametrize("seed", [None, 1234])
def test_model_consistency_with_seed(options, monkeypatch, tmp_path, seed):
    """Checks final model consistency with a fixed seed."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    if seed is not None:
        options["seed"] = seed

    os.mkdir("outputs_1/")
    train_model(options, output="model1.pt", checkpoint_dir="outputs_1/")

    if seed is None:
        options["seed"] = RANDOM_SEED + 1

    os.mkdir("outputs_2/")
    train_model(options, output="model2.pt", checkpoint_dir="outputs_2/")

    m1 = torch.load("model1.ckpt", weights_only=False)
    m2 = torch.load("model2.ckpt", weights_only=False)

    for i in m1["model_state_dict"]:
        tensor1 = m1["model_state_dict"][i]
        tensor2 = m2["model_state_dict"][i]

        if seed is None:
            assert not torch.allclose(tensor1, tensor2)
        else:
            torch.testing.assert_close(tensor1, tensor2)


def test_base_validation(options, monkeypatch, tmp_path):
    """Test that the base options are validated."""
    monkeypatch.chdir(tmp_path)

    options["base_precision"] = 67

    match = r"67 is not one of \[16, 32, 64\]"
    with pytest.raises(ValidationError, match=match):
        train_model(options)


# TODO add parametrize for 16-bit once we have a model that supports this.
@pytest.mark.parametrize("base_precision", [64])
def test_different_base_precision(options, monkeypatch, tmp_path, base_precision):
    """Test different `base_precision`s."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")
    options["base_precision"] = base_precision
    train_model(options)


def test_architecture_error(options, monkeypatch, tmp_path):
    """Test an error raise if there is problem wth the architecture."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    options["architecture"]["model"] = OmegaConf.create(
        {"soap": {"cutoff": {"radius": -1.0}}}
    )

    with pytest.raises(ArchitectureError, match="originates from an architecture"):
        train_model(options)


def test_train_issue_290(monkeypatch, tmp_path):
    """Test the potential problem from issue #290."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_ETHANOL, "ethanol_reduced_100.xyz")

    structures = ase.io.read("ethanol_reduced_100.xyz", ":")
    more_structures = structures * 15 + [structures[0]]
    ase.io.write("ethanol_1501.xyz", more_structures)

    # run training with original options
    options = OmegaConf.load(OPTIONS_PATH)
    options["training_set"]["systems"]["read_from"] = "ethanol_1501.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"
    options["validation_set"] = 0.01
    options["test_set"] = 0.85

    train_model(options)


def test_train_log_order(caplog, monkeypatch, tmp_path, options):
    """Tests that the log is always printed in the same order for forces
    and virials."""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_CARBON, "carbon_reduced_100.xyz")

    options["architecture"]["training"]["num_epochs"] = 5
    options["architecture"]["training"]["log_interval"] = 1

    options["training_set"]["systems"]["read_from"] = str(DATASET_PATH_CARBON)
    options["training_set"]["targets"]["energy"]["read_from"] = str(DATASET_PATH_CARBON)
    options["training_set"]["targets"]["energy"]["key"] = "energy"
    options["training_set"]["targets"]["energy"]["forces"] = {
        "key": "force",
    }
    options["training_set"]["targets"]["energy"]["virial"] = True

    caplog.set_level(logging.INFO)
    train_model(options)
    log_test = caplog.text

    # find all the lines that have "Epoch" in them; these are the lines that
    # contain the training metrics
    epoch_lines = [line for line in log_test.split("\n") if "Epoch" in line]

    # check that "training forces RMSE" comes before "training virial RMSE"
    # in every line
    for line in epoch_lines:
        force_index = line.index("training forces RMSE")
        virial_index = line.index("training virial RMSE")
        assert force_index < virial_index

    # same for validation
    for line in epoch_lines:
        force_index = line.index("validation forces RMSE")
        virial_index = line.index("validation virial RMSE")
        assert force_index < virial_index


def test_train_generic_target(monkeypatch, tmp_path):
    """Test training on a spherical vector target"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_ETHANOL, "ethanol_reduced_100.xyz")

    # run training with original options
    options = OmegaConf.load(OPTIONS_PATH)
    options["training_set"]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["type"] = {
        "spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}
    }
    options["training_set"]["targets"]["energy"]["sample_kind"] = ["atom"]
    options["training_set"]["targets"]["energy"]["key"] = "forces"

    train_model(options)


@pytest.mark.parametrize("with_scalar_part", [False, True])
def test_train_generic_target_metatensor(monkeypatch, tmp_path, with_scalar_part):
    """Test training on a spherical rank-2 tensor target in metatensor format"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM7X, "qm7x_reduced_100.xyz")

    dump_spherical_targets(
        "qm7x_reduced_100.xyz", "qm7x_reduced_100.mts", with_scalar_part
    )

    # run training with original options
    options = OmegaConf.load(OPTIONS_PATH)
    options["architecture"]["name"] = "experimental.nanopet"
    options["training_set"]["systems"]["read_from"] = "qm7x_reduced_100.xyz"
    options["training_set"]["targets"] = {
        "mtt::polarizability": {
            "read_from": "qm7x_reduced_100.mts",
            "type": {
                "spherical": {
                    "irreps": (
                        [{"o3_lambda": 0, "o3_sigma": 1}] if with_scalar_part else []
                    )
                    + [{"o3_lambda": 2, "o3_sigma": 1}]
                }
            },
        }
    }

    train_model(options)


def test_train_disk_dataset(monkeypatch, tmp_path, options):
    """Test that training via the training cli runs without an error raise
    when learning from a `DiskDataset`."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH_QM9, "qm9_reduced_100.xyz")

    disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
    for i in range(100):
        frame = read("qm9_reduced_100.xyz", index=i)
        system = systems_to_torch(frame, dtype=torch.float64)
        system = get_system_with_neighbor_lists(
            system,
            [NeighborListOptions(cutoff=5.0, full_list=False, strict=False)],
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

    options["training_set"]["systems"]["read_from"] = "qm9_reduced_100.zip"
    train_model(options)
