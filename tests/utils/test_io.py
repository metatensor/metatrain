from pathlib import Path

import metatensor.torch
import pytest
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model
from metatensor.models.utils.data import read_systems
from metatensor.models.utils.export import is_exported
from metatensor.models.utils.io import check_suffix, load, save


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("path", [Path("test_model.ckpt"), "test_model.ckpt"])
def test_save_load_checkpoint(monkeypatch, tmp_path, path):
    """Test that saving and loading a model works and preserves its internal state."""
    monkeypatch.chdir(tmp_path)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )

    model = Model(capabilities)
    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    output_before_save = model(
        systems,
        {"energy": model.capabilities.outputs["energy"]},
    )

    save(model, path)
    loaded_model = load(path)

    output_after_load = loaded_model(
        systems,
        {"energy": model.capabilities.outputs["energy"]},
    )

    assert metatensor.torch.allclose(
        output_before_save["energy"], output_after_load["energy"]
    )


def test_missing_extension(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )

    model = Model(capabilities)

    with pytest.warns(
        match="adding '.ckpt' extension, the file will be saved at 'model.foo.ckpt'"
    ):
        save(model, "model.foo")


@pytest.mark.parametrize(
    "path",
    [RESOURCES_PATH / "model-32-bit.pt", str(RESOURCES_PATH / "model-32-bit.pt")],
)
def test_load_exported_model(path):
    model = load(path)
    assert is_exported(model)


def test_load_no_file():
    with pytest.raises(ValueError, match="foo: no such file or directory"):
        load("foo")


def test_no_checkpoint_no_export():
    path = RESOURCES_PATH / "eval.yaml"
    match = f"{path} is neither a valid 'checkpoint' nor an 'exported' model"
    with pytest.raises(ValueError, match=match):
        load(path)


@pytest.mark.parametrize("filename", ["example.txt", Path("example.txt")])
def test_check_suffix(filename):
    result = check_suffix(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


@pytest.mark.parametrize("filename", ["example", Path("example")])
def test_warning_on_missing_suffix(filename):
    match = r"The file name should have a '\.txt' extension."
    with pytest.warns(UserWarning, match=match):
        result = check_suffix(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))
