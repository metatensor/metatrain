from pathlib import Path

import pytest
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import Model
from metatensor.models.utils.export import export, is_exported
from metatensor.models.utils.model_io import load_checkpoint, load_exported_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_export(monkeypatch, tmp_path):
    """Tests the export function"""
    monkeypatch.chdir(tmp_path)

    model = Model(
        capabilities=ModelCapabilities(atomic_types=[1], length_unit="angstrom")
    )
    export(model, "exported.pt")

    assert Path("exported.pt").is_file()


def test_is_exported():
    """Tests the is_exported function"""

    checkpoint = load_checkpoint(RESOURCES_PATH / "bpnn-model.ckpt")
    exported_model = load_exported_model(RESOURCES_PATH / "bpnn-model.pt")

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
