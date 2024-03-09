import shutil
from pathlib import Path

import metatensor.torch
import pytest
import rascaline.torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental import soap_bpnn
from metatensor.models.utils.data import read_systems
from metatensor.models.utils.export import is_exported
from metatensor.models.utils.model_io import (
    load_checkpoint,
    load_exported_model,
    save_model,
)


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_save_load_checkpoint(monkeypatch, tmp_path):
    """Test that saving and loading a model works and preserves its internal state."""
    monkeypatch.chdir(tmp_path)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = soap_bpnn.Model(capabilities)
    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    output_before_save = model(
        rascaline.torch.systems_to_torch(systems),
        {"energy": model.capabilities.outputs["energy"]},
    )

    save_model(model, "test_model.ckpt")
    loaded_model = load_checkpoint("test_model.ckpt")

    output_after_load = loaded_model(
        rascaline.torch.systems_to_torch(systems),
        {"energy": model.capabilities.outputs["energy"]},
    )

    assert metatensor.torch.allclose(
        output_before_save["energy"], output_after_load["energy"]
    )


def test_load_checkpoint_wraning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Create a model with "wrong" filending
    shutil.copy(RESOURCES_PATH / "bpnn-model.ckpt", "model.pt")

    with pytest.warns(match="Trying to load a checkpoint from a .pt file."):
        load_checkpoint("model.pt")


def test_load_exported_model():
    model = load_exported_model(RESOURCES_PATH / "bpnn-model.pt")
    assert is_exported(model)


def test_load_exported_model_warning(monkeypatch, tmp_path):
    """Test error raise if filesuffix is not the expected one."""
    monkeypatch.chdir(tmp_path)

    # Create a model with "wrong" filending
    shutil.copy(RESOURCES_PATH / "bpnn-model.pt", "model.ckpt")

    with pytest.warns(match="Trying to load an exported model from a .ckpt file."):
        load_exported_model("model.ckpt")


def test_load_exported_model_error():
    with pytest.warns(match="This is probably not an exported model"):
        with pytest.raises(ValueError, match="is not exported"):
            load_exported_model(RESOURCES_PATH / "bpnn-model.ckpt")
