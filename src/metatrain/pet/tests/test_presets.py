"""Tests for PET presets functionality."""

import pytest

from metatrain.pet.presets import PETPresets, get_preset_hypers, PRESET_DESCRIPTIONS


def test_preset_names():
    """Test that all expected presets are available."""
    expected_presets = ["default", "fast", "medium", "large"]
    for preset in expected_presets:
        assert preset in PETPresets.__args__, f"Preset '{preset}' not found in PETPresets"


def test_get_default_preset():
    """Test getting the default preset."""
    hypers = get_preset_hypers("default")
    assert hypers == {}, "Default preset should be an empty dict"


def test_get_fast_preset():
    """Test getting the fast preset."""
    hypers = get_preset_hypers("fast")
    assert "model" in hypers
    assert "training" in hypers

    # Check some specific fast preset values
    assert hypers["model"]["cutoff"] == 3.5
    assert hypers["model"]["d_pet"] == 64
    assert hypers["model"]["d_node"] == 128
    assert hypers["model"]["num_gnn_layers"] == 1
    assert hypers["model"]["num_attention_layers"] == 1

    assert hypers["training"]["batch_size"] == 32
    assert hypers["training"]["learning_rate"] == 2e-4


def test_get_medium_preset():
    """Test getting the medium preset."""
    hypers = get_preset_hypers("medium")
    assert "model" in hypers
    assert "training" in hypers

    # Check some specific medium preset values
    assert hypers["model"]["cutoff"] == 4.5
    assert hypers["model"]["d_pet"] == 128
    assert hypers["model"]["d_node"] == 256
    assert hypers["model"]["num_gnn_layers"] == 2
    assert hypers["model"]["num_attention_layers"] == 2

    assert hypers["training"]["batch_size"] == 16
    assert hypers["training"]["learning_rate"] == 1e-4


def test_get_large_preset():
    """Test getting the large preset."""
    hypers = get_preset_hypers("large")
    assert "model" in hypers
    assert "training" in hypers

    # Check some specific large preset values
    assert hypers["model"]["cutoff"] == 5.5
    assert hypers["model"]["d_pet"] == 256
    assert hypers["model"]["d_node"] == 512
    assert hypers["model"]["num_gnn_layers"] == 3
    assert hypers["model"]["num_attention_layers"] == 3

    assert hypers["training"]["batch_size"] == 8
    assert hypers["training"]["learning_rate"] == 5e-5


def test_invalid_preset():
    """Test that requesting an invalid preset raises an error."""
    with pytest.raises(ValueError) as excinfo:
        get_preset_hypers("invalid")

    assert "Unknown preset 'invalid'" in str(excinfo.value)
    assert "Available presets are:" in str(excinfo.value)


def test_preset_descriptions():
    """Test that all presets have descriptions."""
    for preset in ["default", "fast", "medium", "large"]:
        assert preset in PRESET_DESCRIPTIONS
        assert isinstance(PRESET_DESCRIPTIONS[preset], str)
        assert len(PRESET_DESCRIPTIONS[preset]) > 0


def test_preset_progression():
    """Test that presets follow expected progression (small -> medium -> large)."""
    fast = get_preset_hypers("fast")
    medium = get_preset_hypers("medium")
    large = get_preset_hypers("large")

    # Check that dimensions increase
    assert fast["model"]["d_pet"] < medium["model"]["d_pet"] < large["model"]["d_pet"]
    assert fast["model"]["d_node"] < medium["model"]["d_node"] < large["model"]["d_node"]

    # Check that layers increase
    assert fast["model"]["num_gnn_layers"] <= medium["model"]["num_gnn_layers"] <= large["model"]["num_gnn_layers"]
    assert fast["model"]["num_attention_layers"] <= medium["model"]["num_attention_layers"] <= large["model"]["num_attention_layers"]

    # Check that batch size decreases (larger models use smaller batch sizes)
    assert fast["training"]["batch_size"] > medium["training"]["batch_size"] > large["training"]["batch_size"]

    # Check that learning rate decreases
    assert fast["training"]["learning_rate"] > medium["training"]["learning_rate"] > large["training"]["learning_rate"]
