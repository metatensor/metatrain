from pathlib import Path

import metatensor.torch
import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import Model
from metatensor.models.utils.data import read_systems
from metatensor.models.utils.io import export, is_exported, load, save


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
    )

    model = Model(capabilities)
    systems = read_systems(
        RESOURCES_PATH / "qm9_reduced_100.xyz", dtype=torch.get_default_dtype()
    )

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
    )

    model = Model(capabilities)

    with pytest.warns(
        match="adding '.ckpt' extension, the file will be saved at 'model.foo.ckpt'"
    ):
        save(model, "model.foo")


@pytest.mark.parametrize(
    "path", [RESOURCES_PATH / "bpnn-model.pt", str(RESOURCES_PATH / "bpnn-model.pt")]
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


@pytest.mark.parametrize("path", [Path("exported.pt"), "exported.pt"])
def test_export(monkeypatch, tmp_path, path):
    """Tests the export function"""
    monkeypatch.chdir(tmp_path)

    model = Model(
        capabilities=ModelCapabilities(atomic_types=[1], length_unit="angstrom")
    )
    export(model, path)

    assert Path(path).is_file()


def test_export_warning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    model = Model(
        capabilities=ModelCapabilities(atomic_types=[1], length_unit="angstrom")
    )

    with pytest.warns(
        match="adding '.pt' extension, the file will be saved at 'model.foo.pt'"
    ):
        export(model, "model.foo")


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    model = Model(
        capabilities=ModelCapabilities(atomic_types=[1], length_unit="angstrom")
    )
    export(model, "exported.pt")

    model_loaded = load("exported.pt")
    export(model_loaded, "exported_new.pt")

    assert Path("exported_new.pt").is_file()


def test_is_exported():
    """Tests the is_exported function"""

    checkpoint = load(RESOURCES_PATH / "bpnn-model.ckpt")
    exported_model = load(RESOURCES_PATH / "bpnn-model.pt")

    assert is_exported(exported_model)
    assert not is_exported(checkpoint)


def test_length_units_warning(monkeypatch, tmp_path):
    model = Model(capabilities=ModelCapabilities(atomic_types=[1]))

    monkeypatch.chdir(tmp_path)
    with pytest.warns(match="No `length_unit` was provided for the model."):
        export(model, "exported.pt")


def test_units_warning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    outputs = {"output": ModelOutput(quantity="energy")}
    model = Model(
        capabilities=ModelCapabilities(
            atomic_types=[1], outputs=outputs, length_unit="angstrom"
        )
    )

    with pytest.warns(match="No target units were provided for output 'output'"):
        export(model, "exported.pt")
