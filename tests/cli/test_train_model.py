import glob
import shutil
import subprocess
from pathlib import Path

import ase.io
import metatensor.torch  # noqa
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from metatensor.models.cli.train import check_architecture_name, train_model
from metatensor.models.utils.errors import ArchitectureError


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"
DATASET_PATH = RESOURCES_PATH / "qm9_reduced_100.xyz"
DATASET_PATH_2 = RESOURCES_PATH / "ethanol_reduced_100.xyz"
OPTIONS_PATH = RESOURCES_PATH / "options.yaml"
MODEL_PATH = RESOURCES_PATH / "model-32-bit.ckpt"


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train(monkeypatch, tmp_path, output):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["metatensor-models", "train", "options.yaml"]

    if output is not None:
        command += ["-o", output]
    else:
        output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if fully expanded options.yaml file is written
    assert len(glob.glob("outputs/*/*/options.yaml")) == 1

    # Test if logfile is written
    assert len(glob.glob("outputs/*/*/train.log")) == 1

    # Open the log file and check if the logging is correct
    with open(glob.glob("outputs/*/*/train.log")[0]) as f:
        log = f.read()

    assert "This log is also available"
    assert "[INFO]" in log
    assert "Epoch" in log
    assert "loss" in log
    assert "validation" in log
    assert "train" in log
    assert "energy" in log
    assert "with index" not in log  # index only printed for more than 1 dataset


@pytest.mark.parametrize("n_datasets", [1, 2])
@pytest.mark.parametrize("test_set_file", (True, False))
@pytest.mark.parametrize("validation_set_file", (True, False))
def test_train_explicit_validation_test(
    monkeypatch,
    tmp_path,
    capsys,
    n_datasets,
    test_set_file,
    validation_set_file,
    options,
):
    """Test that training via the training cli runs without an error raise
    also when the validation and test sets are provided explicitly."""
    monkeypatch.chdir(tmp_path)

    systems = ase.io.read(DATASET_PATH, ":")

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

    # Test log messages which are written by hydra to STDOUT
    log = capsys.readouterr().out
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

    systems = ase.io.read(DATASET_PATH, ":")
    systems_2 = ase.io.read(DATASET_PATH_2, ":")

    ase.io.write("qm9_reduced_100.xyz", systems[:50])
    ase.io.write("ethanol_reduced_100.xyz", systems_2[:50])

    options["training_set"] = OmegaConf.create(2 * [options["training_set"]])
    options["training_set"][1]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"][1]["targets"]["energy"]["key"] = "energy"
    options["training_set"][0]["targets"].pop("energy")
    options["training_set"][0]["targets"]["U0"] = OmegaConf.create({"key": "U0"})

    train_model(options)


def test_empty_training_set(monkeypatch, tmp_path, options):
    """Test that an error is raised if no training set is provided."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.6
    options["test_set"] = 0.4

    with pytest.raises(
        ValueError, match="Fraction of the train set is smaller or equal to 0!"
    ):
        train_model(options)


def test_empty_validation_set(monkeypatch, tmp_path, options):
    """Test that an error is raised if no validation set is provided."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.0
    options["test_set"] = 0.4

    with pytest.raises(ValueError, match="must be greater than 0"):
        train_model(options)


def test_empty_test_set(monkeypatch, tmp_path, options):
    """Test that no error is raised if no test set is provided."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["validation_set"] = 0.4
    options["test_set"] = 0.0

    train_model(options)

    # check if the logging is correct
    with open(glob.glob("outputs/*/*/train.log")[0]) as f:
        log = f.read()
    assert "This dataset is empty. No evaluation" in log


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

    systems = ase.io.read(DATASET_PATH, ":")

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

    systems = ase.io.read(DATASET_PATH, ":")

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

    with pytest.raises(ValueError, match="different size than the train datatset"):
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
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

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
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    train_model(options, continue_from=MODEL_PATH)


def test_continue_different_dataset(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise
    with a different dataset than the original."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")

    options["training_set"]["systems"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"

    train_model(options, continue_from=MODEL_PATH)


def test_hydra_arguments():
    """Test if hydra arguments work."""
    option_path = str(RESOURCES_PATH / "options.yaml")
    out = subprocess.check_output(
        ["metatensor-models", "train", option_path, "--hydra=--help"]
    )
    # Check that num_epochs is override is succesful
    assert "num_epochs: 1" in str(out)


def test_no_architecture_name(options):
    """Test error raise if architecture.name is not set."""
    options["architecture"].pop("name")

    with pytest.raises(ConfigKeyError, match="Architecture name is not defined!"):
        train_model(options)


@pytest.mark.parametrize("seed", [1234, None, 0, -123])
@pytest.mark.parametrize("architecture_name", ["experimental.soap_bpnn"])
def test_model_consistency_with_seed(
    options, monkeypatch, tmp_path, architecture_name, seed
):
    """Checks final model consistency with a fixed seed."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["architecture"]["name"] = architecture_name
    options["seed"] = seed

    if seed is not None and seed < 0:
        with pytest.raises(ValueError, match="should be a positive number or None."):
            train_model(options)
        return

    train_model(options, output="model1.pt")
    train_model(options, output="model2.pt")

    m1 = torch.load("model1.ckpt")
    m2 = torch.load("model2.ckpt")

    for index, i in enumerate(m1["model_state_dict"]):
        tensor1 = m1["model_state_dict"][i]
        tensor2 = m2["model_state_dict"][i]

        # The first tensor only depend on the chemical compositions (not on the
        # seed) and should alwyas be the same.
        if index == 0:
            torch.testing.assert_close(tensor1, tensor2)
        else:
            if seed is None:
                assert not torch.allclose(tensor1, tensor2)
            else:
                torch.testing.assert_close(tensor1, tensor2)


def test_error_base_precision(options, monkeypatch, tmp_path):
    """Test unsupported `base_precision`"""
    monkeypatch.chdir(tmp_path)

    options["base_precision"] = "123"

    with pytest.raises(ValueError, match="Only 64, 32 or 16 are possible values for"):
        train_model(options)


# TODO add parametrize for 16-bit once we have a model that supports this.
@pytest.mark.parametrize("base_precision", [64])
def test_different_base_precision(options, monkeypatch, tmp_path, base_precision):
    """Test different `base_precision`s."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")
    options["base_precision"] = base_precision
    train_model(options)


def test_unsupported_dtype(options):
    options["base_precision"] = 16
    match = (
        r"Requested dtype torch.float16 is not supported. experimental.soap_bpnn "
        r"only supports \[torch.float64, torch.float32\]."
    )
    with pytest.raises(ValueError, match=match):
        train_model(options)


def test_architecture_error(options, monkeypatch, tmp_path):
    """Test an error raise if there is problem wth the architecture."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["architecture"]["model"] = OmegaConf.create({"soap": {"cutoff": -1}})

    with pytest.raises(ArchitectureError, match="originates from an architecture"):
        train_model(options)


def test_check_architecture_name():
    check_architecture_name("experimental.soap_bpnn")


def test_check_architecture_name_suggest():
    name = "soap-bpnn"
    match = f"Architecture {name!r} is not a valid architecture."
    with pytest.raises(ValueError, match=match):
        check_architecture_name(name)


def test_check_architecture_name_experimental():
    with pytest.raises(
        ValueError, match="experimental architecture with the same name"
    ):
        check_architecture_name("soap_bpnn")


def test_check_architecture_name_deprecated():
    # Create once a deprecated architecture exist
    pass
