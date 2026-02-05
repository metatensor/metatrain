"""Tests for architecture presets integration."""

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import (
    get_default_hypers,
    get_default_hypers_with_preset,
)


def test_get_default_hypers_without_preset():
    """Test that get_default_hypers_with_preset works without preset."""
    hypers = get_default_hypers_with_preset("pet", None)
    default_hypers = get_default_hypers("pet")

    # Should be the same as get_default_hypers
    assert hypers == default_hypers


def test_get_preset_hypers_for_pet():
    """Test that get_default_hypers_with_preset applies presets for PET."""
    # Get default hypers
    default_hypers = get_default_hypers("pet")

    # Get fast preset hypers
    fast_hypers = get_default_hypers_with_preset("pet", "fast")

    # Check that fast preset overrides defaults
    assert fast_hypers["model"]["cutoff"] == 3.5
    assert fast_hypers["model"]["cutoff"] != default_hypers["model"]["cutoff"]

    assert fast_hypers["model"]["d_pet"] == 64
    assert fast_hypers["model"]["d_pet"] != default_hypers["model"]["d_pet"]

    # Check that non-preset values remain from defaults
    assert fast_hypers["model"]["cutoff_function"] == default_hypers["model"]["cutoff_function"]


def test_get_medium_preset_for_pet():
    """Test getting medium preset for PET."""
    medium_hypers = get_default_hypers_with_preset("pet", "medium")

    assert medium_hypers["model"]["cutoff"] == 4.5
    assert medium_hypers["model"]["d_pet"] == 128
    assert medium_hypers["training"]["batch_size"] == 16


def test_get_large_preset_for_pet():
    """Test getting large preset for PET."""
    large_hypers = get_default_hypers_with_preset("pet", "large")

    assert large_hypers["model"]["cutoff"] == 5.5
    assert large_hypers["model"]["d_pet"] == 256
    assert large_hypers["training"]["batch_size"] == 8


def test_invalid_preset_for_pet():
    """Test that invalid preset raises an error."""
    with pytest.raises(ValueError) as excinfo:
        get_default_hypers_with_preset("pet", "invalid_preset")

    assert "Unknown preset 'invalid_preset'" in str(excinfo.value)


def test_preset_for_architecture_without_presets():
    """Test that requesting a preset for an architecture without presets raises an error."""
    with pytest.raises(ValueError) as excinfo:
        get_default_hypers_with_preset("soap_bpnn", "fast")

    assert "does not support presets" in str(excinfo.value)


def test_preset_merging_with_options():
    """Test that presets can be merged with additional options."""
    # Get preset hypers
    preset_hypers = get_default_hypers_with_preset("pet", "fast")

    # Simulate merging with user options
    user_options = OmegaConf.create({
        "model": {"d_pet": 128}  # Override the d_pet from fast preset
    })

    merged = OmegaConf.merge(preset_hypers, user_options)

    # Check that user option overrides preset
    assert merged["model"]["d_pet"] == 128
    # Check that other preset values remain
    assert merged["model"]["cutoff"] == 3.5


def test_all_pet_presets():
    """Test that all PET presets are accessible and valid."""
    for preset_name in ["default", "fast", "medium", "large"]:
        hypers = get_default_hypers_with_preset("pet", preset_name)
        assert "name" in hypers
        assert "model" in hypers
        assert "training" in hypers
        assert hypers["name"] == "pet"
