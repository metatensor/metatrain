import os

import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, is_atomistic_model

from metatrain.experimental.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.export import export

from . import MODEL_HYPERS


def test_export(tmp_path):
    """Tests the export and is_export function"""
    os.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={"energy": TargetInfo(quantity="energy", unit="eV")},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    capabilities = ModelCapabilities(
        length_unit=model.dataset_info.length_unit,
        outputs=model.outputs,
        atomic_types=list(model.dataset_info.atomic_types),
        supported_devices=model.__supported_devices__,
        interaction_range=model.hypers["soap"]["cutoff"],
        dtype="float32",
    )

    # test `export()`
    exported_model = export(model, capabilities)
    exported_model.save("model.pt")

    # test `is_export()`
    assert not is_atomistic_model(model)
    assert is_atomistic_model(exported_model)
    assert is_atomistic_model(torch.jit.load("model.pt"))


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={"energy": TargetInfo(quantity="energy", unit="eV")},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    capabilities = ModelCapabilities(
        length_unit=model.dataset_info.length_unit,
        outputs=model.outputs,
        atomic_types=list(model.dataset_info.atomic_types),
        supported_devices=model.__supported_devices__,
        interaction_range=model.hypers["soap"]["cutoff"],
        dtype="float32",
    )

    exported_model = export(model, capabilities)
    export(exported_model, capabilities)


def test_length_units_warning():
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={"energy": TargetInfo(quantity="energy", unit="eV")},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    capabilities = ModelCapabilities(
        outputs=model.outputs,
        atomic_types=list(model.dataset_info.atomic_types),
        interaction_range=model.hypers["soap"]["cutoff"],
        length_unit="",
        supported_devices=model.__supported_devices__,
        dtype="float32",
    )

    with pytest.warns(match="No `length_unit` was provided for the model."):
        export(model, capabilities)


def test_units_warning():
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={"mtt::output": TargetInfo(quantity="energy")},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    capabilities = ModelCapabilities(
        length_unit=model.dataset_info.length_unit,
        outputs=model.outputs,
        atomic_types=list(model.dataset_info.atomic_types),
        supported_devices=model.__supported_devices__,
        interaction_range=model.hypers["soap"]["cutoff"],
        dtype="float32",
    )

    with pytest.warns(match="No target units were provided for output 'mtt::output'"):
        export(model, capabilities)
