import copy
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch import load as metatensor_load
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    NeighborListOptions,
    systems_to_torch,
)
from omegaconf import OmegaConf

import metatrain.cli.eval
import metatrain.utils.equivariance
from metatrain.cli.eval import eval_model
from metatrain.pet import PET
from metatrain.soap_bpnn import __model__
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, DiskDataset
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
)
from metatrain.utils.data.dataset import MemmapDataset
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.wrapper import MetatrainModel

from ..conftest import EVAL_OPTIONS_PATH, MODEL_HYPERS, RESOURCES_PATH


@pytest.fixture
def model(MODEL_PATH):
    return torch.jit.load(MODEL_PATH)


@pytest.fixture
def options():
    return OmegaConf.load(EVAL_OPTIONS_PATH)


@pytest.mark.parametrize("warm_up", [True, False])
def test_eval_cli(monkeypatch, tmp_path, MODEL_PATH, warm_up):
    """Test succesful run of the eval script via the CLI with default arguments"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    command = [
        "mtt",
        "eval",
        str(MODEL_PATH),
        str(EVAL_OPTIONS_PATH),
        "-e",
        str(MODEL_PATH.parent / "extensions"),
        "--check-consistency",
    ]
    if not warm_up:
        command.append("--no-warm-up")

    result = subprocess.run(
        command,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )

    if result.returncode != 0:
        print(
            f"Eval logs:\n{result.stdout.decode()}",
            file=sys.stderr,
        )
        raise RuntimeError(
            "Failed to evaluate model via CLI. Logs should be printed above."
        )

    log_text = result.stdout.decode()

    assert "100%|██████████" in log_text
    assert "energy RMSE" in log_text

    # Check that the warm-up flag is correctly used
    warm_up_str = "Warming up the model with 10 batches..."
    no_warm_up_str = "Skipping warm-up of the model."
    if warm_up:
        assert warm_up_str in log_text
        assert no_warm_up_str not in log_text
    else:
        assert warm_up_str not in log_text
        assert no_warm_up_str in log_text

    assert Path("output.xyz").is_file()


@pytest.mark.parametrize("model_type", ["32-bit", "64-bit"])
def test_eval(request, monkeypatch, tmp_path, caplog, model_type, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    fixture_name = {
        "32-bit": "MODEL_PATH",
        "64-bit": "MODEL_PATH_64_BIT",
    }.get(model_type)

    model_path = request.getfixturevalue(fixture_name)

    model = torch.jit.load(model_path)

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
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    frames[0].info["energy"]


@pytest.mark.parametrize("model_type", ["32-bit", "64-bit"])
def test_eval_batch_size(request, monkeypatch, tmp_path, caplog, model_type, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    fixture_name = {
        "32-bit": "MODEL_PATH",
        "64-bit": "MODEL_PATH_64_BIT",
    }.get(model_type)

    model_path = request.getfixturevalue(fixture_name)

    model = torch.jit.load(model_path)

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
    assert "Evaluation time" in log
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
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = __model__(hypers=MODEL_HYPERS, dataset_info=dataset_info)

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


def test_eval_extxyz_input_and_output(monkeypatch, tmp_path, model, options):
    """Test that eval can read systems from and write predictions to extxyz files."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.extxyz")
    options.pop("targets")
    options["systems"] = "qm9_reduced_100.extxyz"

    eval_model(model=model, options=options, output="predictions.extxyz")

    frames = read("predictions.extxyz", ":")
    assert len(frames) == 100
    assert "energy" in frames[0].info


@pytest.mark.parametrize("suffix", [".zip", "/"])
def test_eval_no_targets_disallowed_for_dataset_writers(
    monkeypatch, tmp_path, model, options, suffix
):
    """DiskDataset/MemmapDataset outputs require explicit targets in the input."""
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")

    with pytest.raises(ValueError, match="not allowed without explicitly"):
        eval_model(
            model=model,
            options=options,
            output=f"output{suffix}",
            check_consistency=True,
        )


@pytest.mark.parametrize("suffix", [".zip", ".mts", "/"])
def test_eval_disk_dataset(monkeypatch, tmp_path, caplog, suffix, MODEL_PATH):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    model = torch.jit.load(MODEL_PATH)

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
                        values=torch.tensor([[0]]),
                    ),
                    components=[],
                    properties=Labels("energy", torch.tensor([[0]])),
                )
            ],
        )
        disk_dataset_writer.write([system], {"energy": energy})
    disk_dataset_writer.finish()

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
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    if suffix == ".mts":
        pred = metatensor_load("foo_energy.mts")
        assert pred.keys == Labels(["_"], torch.tensor([[0]]))
    elif suffix == "/":
        target_options = {
            "energy": {
                "key": "energy",
                "quantity": "energy",
                "sample_kind": "system",
                "type": "scalar",
                "num_subtargets": 1,
                "forces": False,
                "stress": False,
                "virial": False,
            }
        }
        pred = MemmapDataset("foo", target_options)
        assert len(pred) == 100
        assert pred[0].energy.keys == Labels(["_"], torch.tensor([[0]]))
    else:
        pred = DiskDataset("foo.zip")
        assert pred[0]["energy"].keys == Labels(["_"], torch.tensor([[0]]))


@pytest.mark.parametrize("indices_mode", ["list", "file"])
def test_eval_indices(monkeypatch, tmp_path, caplog, options, MODEL_PATH, indices_mode):
    """Checks that the indices option is correctly used to evaluate only a subset"""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = options.copy()
    if indices_mode == "file":
        with open("indices.txt", "w") as f:
            f.write("0\n")
        options["indices"] = "indices.txt"
    else:
        options["indices"] = [0]

    model = torch.jit.load(MODEL_PATH)

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
    assert "Evaluation time" in log
    assert "ms per atom" in log

    # Test file is written predictions
    frames = read("foo.xyz", ":")
    assert len(frames) == 1


def test_eval_equivariance(monkeypatch, tmp_path, caplog, MODEL_PATH_64_BIT, options):
    """The O(3)-averaged predictions of an invariant model are its predictions, and
    its equivariance error is zero."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = options.copy()
    options["indices"] = [0, 1, 2]
    options["equivariance"] = {}
    model = torch.jit.load(MODEL_PATH_64_BIT)
    eval_model(model=model, options=options, output="foo.xyz")

    log = "".join([rec.message for rec in caplog.records])
    # the default grid for scalar targets has degree 1, with 12 rotations
    assert "evaluated on 24 rotated and inverted copies" in log
    assert "energy equivariance RMSE (per atom): 0.0000 meV" in log
    assert "energy O3-averaged RMSE (per atom)" in log
    assert "energy O3-averaged MAE (per atom)" in log
    assert "energy oriented RMSE (per atom)" in log

    frames = read("foo.xyz", ":")
    assert len(frames) == 3
    for frame in frames:
        energy = frame.info["energy"]
        assert frame.info["o3::average::energy"] == pytest.approx(energy, rel=1e-12)
        variance = frame.info["o3::variance::energy"]
        assert variance == pytest.approx(0.0, abs=1e-12 * energy**2)


# a scalar, a spherical, a Cartesian and an atomic-basis target
EQUIVARIANCE_TARGETS = {
    "energy": {"quantity": "energy", "unit": "eV"},
    "mtt::spherical": {
        "type": {"spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}},
    },
    "mtt::vector": {"type": {"cartesian": {"rank": 1}}},
    "mtt::basis": {
        "type": {
            "spherical": {
                "irreps": {
                    atomic_type: [
                        {"o3_lambda": 0, "o3_sigma": 1, "num": 2},
                        {"o3_lambda": 1, "o3_sigma": 1},
                    ]
                    for atomic_type in (1, 6, 7, 8, 9)
                }
            }
        },
    },
}


def _logged_metric(log, name):
    return float(re.search(re.escape(name) + r": ([0-9.e+-]+)", log).group(1))


@pytest.mark.parametrize(
    "batch_size, equivariance, expected",
    [(None, False, 1), (None, True, 16), (4, True, 4)],
)
def test_eval_equivariance_batch_size(
    monkeypatch,
    tmp_path,
    MODEL_PATH_64_BIT,
    options,
    batch_size,
    equivariance,
    expected,
):
    """``-b`` batches both the systems and their rotated copies, and defaults to 16
    when measuring the equivariance error."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    used_batch_sizes = []
    eval_targets = metatrain.cli.eval._eval_targets
    symmetrized_model = metatrain.utils.equivariance.SymmetrizedModel

    def spy_eval_targets(*args, **kwargs):
        used_batch_sizes.append(kwargs["batch_size"])
        return eval_targets(*args, **kwargs)

    def spy_symmetrized_model(*args, **kwargs):
        used_batch_sizes.append(kwargs["batch_size"])
        return symmetrized_model(*args, **kwargs)

    monkeypatch.setattr(metatrain.cli.eval, "_eval_targets", spy_eval_targets)
    monkeypatch.setattr(
        metatrain.utils.equivariance, "SymmetrizedModel", spy_symmetrized_model
    )

    options = options.copy()
    options["indices"] = [0]
    if equivariance:
        options["equivariance"] = {}
    model = torch.jit.load(MODEL_PATH_64_BIT)
    eval_model(model=model, options=options, output="foo.xyz", batch_size=batch_size)

    # the symmetrized model is built twice, the second time on a finer grid to check
    # the convergence of the equivariance error
    assert used_batch_sizes == [expected] * (3 if equivariance else 1)


# Atomic-basis targets have empty blocks for the atomic types absent from a system,
# for which ``mtt eval`` warns, with or without the equivariance error.
@pytest.mark.filterwarnings("ignore:Block .* is not present in the model's predictions")
def test_eval_equivariance_all_target_types(monkeypatch, tmp_path, caplog):
    """The equivariance error of a model that is not equivariant (an untrained PET)
    is measured for every kind of target, and per-structure variances are normalized
    by the square of the number of atoms."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8, 9],
        targets={
            name: (
                get_energy_target_info(name, target)
                if name == "energy"
                else get_generic_target_info(
                    name,
                    {
                        "quantity": "",
                        "unit": "",
                        "sample_kind": "atom",
                        "num_subtargets": 1,
                        **target,
                    },
                )
            )
            for name, target in EQUIVARIANCE_TARGETS.items()
        },
    )
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers.update(d_pet=8, d_node=8, d_head=8, d_feedforward=8, num_gnn_layers=1)
    torch.manual_seed(0)
    core = PET(hypers, densify_atomic_basis_dataset_info(dataset_info))
    model = MetatrainModel(core, [], None, dataset_info).to(torch.float64).export()

    # the model's own predictions, as targets
    systems = [
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        for system in systems_to_torch(
            read(RESOURCES_PATH / "qm9_reduced_100.xyz", ":2"), dtype=torch.float64
        )
    ]
    outputs = {
        name: ModelOutput(sample_kind="system" if name == "energy" else "atom")
        for name in EQUIVARIANCE_TARGETS
    }
    predictions = model(
        systems,
        ModelEvaluationOptions(length_unit="angstrom", outputs=outputs),
        check_consistency=True,
    )
    writer = DiskDatasetWriter("data.zip")
    writer.write(systems, predictions)
    writer.finish()

    options = OmegaConf.create(
        {
            "systems": {"read_from": "data.zip", "length_unit": "angstrom"},
            "targets": {
                name: {"quantity": "", "unit": "", "sample_kind": "atom", **target}
                if name != "energy"
                else target
                for name, target in EQUIVARIANCE_TARGETS.items()
            },
            "equivariance": {},
        }
    )
    eval_model(model=model, options=options, output="foo.zip")

    log = "".join([rec.message for rec in caplog.records])
    for name in EQUIVARIANCE_TARGETS:
        assert _logged_metric(log, f"{name} equivariance RMSE (per atom)") > 0.0

    # the outputs of the symmetrized model are written next to the predictions
    symmetrized = [DiskDataset("foo.zip")[i] for i in range(len(systems))]
    for sample in symmetrized:
        for name in EQUIVARIANCE_TARGETS:
            assert {
                name,
                f"o3::average::{name}",
                f"o3::variance::{name}",
            } <= set(sample._fields)

    # the variances are averaged over the components of each block, and the
    # per-structure ones are normalized per atom as the square of the energy
    energy_variances = [
        sample["o3::variance::energy"].block().values.item() / len(system) ** 2
        for sample, system in zip(symmetrized, systems, strict=True)
    ]
    expected = 1e3 * (sum(energy_variances) / len(systems)) ** 0.5  # in meV
    logged = _logged_metric(log, "energy equivariance RMSE (per atom)")
    assert logged == pytest.approx(expected, rel=1e-4)

    vector_variances = torch.cat(
        [s["o3::variance::mtt::vector"].block().values for s in symmetrized]
    )
    expected = vector_variances.mean().item() ** 0.5
    logged = _logged_metric(log, "mtt::vector equivariance RMSE (per atom)")
    assert logged == pytest.approx(expected, rel=1e-4)


def test_eval_equivariance_requires_targets(monkeypatch, tmp_path, model):
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = OmegaConf.create({"systems": "qm9_reduced_100.xyz", "equivariance": {}})
    match = re.escape(
        "measuring the equivariance error requires a `targets` section in the eval "
        "options file"
    )
    with pytest.raises(ValueError, match=match):
        eval_model(model=model, options=options, output="foo.xyz")


@pytest.mark.parametrize("equivariance", [False, True])
def test_eval_log_separate_blocks(
    monkeypatch, tmp_path, caplog, MODEL_PATH_64_BIT, options, equivariance
):
    """Metrics are reported for each block, labeled by its key."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = options.copy()
    options["indices"] = [0, 1]
    options["log_separate_blocks"] = True
    if equivariance:
        options["equivariance"] = {}
    model = torch.jit.load(MODEL_PATH_64_BIT)
    eval_model(model=model, options=options, output="foo.xyz")

    log = "".join([rec.message for rec in caplog.records])
    assert "energy (_=0) RMSE (per atom)" in log
    assert "energy (_=0) MAE (per atom)" in log
    assert "energy RMSE" not in log
    if equivariance:
        assert "energy (_=0) O3-averaged RMSE (per atom)" in log
        # equivariance keys use the O(3) block labels, accuracy keys the target's own
        assert "energy (o3_lambda=0,o3_sigma=1) equivariance RMSE (per atom)" in log
