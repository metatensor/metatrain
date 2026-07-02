import tempfile

import pytest
import torch

from metatrain.composition import CompositionModel
from metatrain.utils.data.readers import read_systems

from . import DATASET_PATH
from .test_regression import _make_synthetic_targets, _train_composition_model


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


torch.set_default_dtype(torch.float64)


def test_get_checkpoint_roundtrip():
    """Test that get_checkpoint() -> load_checkpoint() preserves predictions."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, dataset_info = _train_composition_model(systems, target_values)
    model.eval()

    reference_output = model(systems[:5], {"energy": model.outputs["energy"]})

    checkpoint = model.get_checkpoint()
    assert checkpoint["architecture_name"] == "composition"
    assert checkpoint["model_ckpt_version"] == CompositionModel.__checkpoint_version__

    loaded_model = CompositionModel.load_checkpoint(checkpoint, "export")
    loaded_model.eval()

    loaded_output = loaded_model(
        systems[:5], {"energy": loaded_model.outputs["energy"]}
    )
    assert torch.allclose(
        reference_output["energy"].block().values,
        loaded_output["energy"].block().values,
    )


def test_checkpoint_save_load_file():
    """Test saving checkpoint to file and loading it back."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, dataset_info = _train_composition_model(systems, target_values)
    model.eval()

    reference_output = model(systems[:5], {"energy": model.outputs["energy"]})

    checkpoint = model.get_checkpoint()

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as f:
        torch.save(checkpoint, f.name)
        loaded_checkpoint = torch.load(f.name, weights_only=False)

    loaded_model = CompositionModel.load_checkpoint(loaded_checkpoint, "export")
    loaded_model.eval()

    loaded_output = loaded_model(
        systems[:5], {"energy": loaded_model.outputs["energy"]}
    )
    assert torch.allclose(
        reference_output["energy"].block().values,
        loaded_output["energy"].block().values,
    )


def test_upgrade_checkpoint_invalid_version():
    """Test that upgrade_checkpoint raises on invalid version."""
    checkpoint = {"model_ckpt_version": 99999999999999}
    with pytest.raises(
        RuntimeError,
        match="Unable to upgrade the checkpoint",
    ):
        CompositionModel.upgrade_checkpoint(checkpoint)


def test_load_checkpoint_all_contexts():
    """Test load_checkpoint in all contexts (restart, finetune, export)."""
    systems = read_systems(DATASET_PATH)
    per_species_energies = {1: -0.5, 6: -10.0, 7: -15.0, 8: -20.0}
    target_values = _make_synthetic_targets(systems, per_species_energies)

    model, _ = _train_composition_model(systems, target_values)
    checkpoint = model.get_checkpoint()

    for context in ["restart", "finetune", "export"]:
        loaded = CompositionModel.load_checkpoint(checkpoint, context)
        assert isinstance(loaded, CompositionModel)
