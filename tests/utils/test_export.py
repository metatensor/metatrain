import os
from pathlib import Path

import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model
from metatensor.models.utils.export import export, is_exported
from metatensor.models.utils.io import load


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_export(tmp_path):
    """Tests the export and is_export function"""
    os.chdir(tmp_path)
    model = Model(
        capabilities=ModelCapabilities(
            atomic_types=[1],
            length_unit="angstrom",
            interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
            dtype="float32",
        )
    )

    # test `export()`
    exported_model = export(model)
    exported_model.export("model.pt")

    # test `is_export()`
    assert not is_exported(model)
    assert is_exported(exported_model)
    assert is_exported(torch.jit.load("model.pt"))


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    model = Model(
        capabilities=ModelCapabilities(
            atomic_types=[1],
            length_unit="angstrom",
            interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
            dtype="float32",
        )
    )
    exported_model = export(model)
    export(exported_model)


def test_is_exported():
    """Tests the is_exported function"""

    checkpoint = load(RESOURCES_PATH / "model-32-bit.ckpt")
    assert not is_exported(checkpoint)

    exported_model = load(RESOURCES_PATH / "model-32-bit.pt")
    assert is_exported(exported_model)


def test_length_units_warning():
    model = Model(
        capabilities=ModelCapabilities(
            atomic_types=[1],
            interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
            dtype="float32",
        )
    )

    with pytest.warns(match="No `length_unit` was provided for the model."):
        export(model)


def test_units_warning():
    outputs = {"mtm::output": ModelOutput(quantity="energy")}
    model = Model(
        capabilities=ModelCapabilities(
            atomic_types=[1],
            outputs=outputs,
            length_unit="angstrom",
            interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
            dtype="float32",
        )
    )

    with pytest.warns(match="No target units were provided for output 'mtm::output'"):
        export(model)
