"""Tests for loading a pretrained DPA3 model via the ``dpa3_model`` hyper."""

import copy

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.dpa3 import DPA3
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS


def _make_dataset_info():
    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    return DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )


def _build_base_model():
    """Build a DPA3 model from hypers (the normal path)."""
    dataset_info = _make_dataset_info()
    return DPA3(MODEL_HYPERS, dataset_info)


def test_pretrained_module_loading():
    """Loading a deepmd-kit Module via dpa3_model creates a valid model."""
    base = _build_base_model()

    # Pass the inner deepmd-kit Module as dpa3_model
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = base.model

    dataset_info = _make_dataset_info()
    pretrained = DPA3(hypers, dataset_info)

    assert pretrained.loaded_dpa3 is True
    # The inner model should be the exact same object (not a copy)
    assert pretrained.model is base.model


def test_pretrained_extracts_bias_and_std():
    """Bias and std are extracted from the loaded model and zeroed/reset."""
    base = _build_base_model()
    atomic_model = base.model.atomic_model

    # Record the original values before extraction
    has_bias = hasattr(atomic_model, "out_bias")
    has_std = hasattr(atomic_model, "out_std")
    if has_bias:
        orig_bias = atomic_model.out_bias.clone()
    if has_std:
        orig_std = atomic_model.out_std.clone()

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = base.model

    dataset_info = _make_dataset_info()
    pretrained = DPA3(hypers, dataset_info)

    if has_bias:
        # Extracted bias should match the original
        torch.testing.assert_close(pretrained._loaded_out_bias, orig_bias)
        # The model's out_bias should now be zeroed
        assert (atomic_model.out_bias == 0).all()
    if has_std:
        # Extracted std should match the original
        torch.testing.assert_close(pretrained._loaded_out_std, orig_std)
        # The model's out_std should now be 1.0
        assert (atomic_model.out_std == 1.0).all()

    # The flag should prevent re-extraction
    assert getattr(base.model, "_metatrain_extracted_scaleshift", False) is True


def test_pretrained_fixed_weights():
    """get_fixed_composition_weights and get_fixed_scaling_weights return
    non-empty dicts when a pretrained model was loaded."""
    base = _build_base_model()

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = base.model

    dataset_info = _make_dataset_info()
    pretrained = DPA3(hypers, dataset_info)

    comp_weights = pretrained.get_fixed_composition_weights()
    scale_weights = pretrained.get_fixed_scaling_weights()

    has_bias = hasattr(base.model.atomic_model, "out_bias")
    has_std = hasattr(base.model.atomic_model, "out_std")

    if has_bias:
        assert "mtt::U0" in comp_weights
        # Should have one entry per atomic type
        assert set(comp_weights["mtt::U0"].keys()) == {1, 6, 7, 8}
    else:
        assert comp_weights == {}

    if has_std:
        assert "mtt::U0" in scale_weights
        assert set(scale_weights["mtt::U0"].keys()) == {1, 6, 7, 8}
    else:
        assert scale_weights == {}


def test_pretrained_no_fixed_weights_without_loading():
    """Without dpa3_model, fixed weight methods return empty dicts."""
    model = _build_base_model()
    assert model.get_fixed_composition_weights() == {}
    assert model.get_fixed_scaling_weights() == {}


def test_pretrained_forward_matches():
    """A model loaded via dpa3_model produces the same output as the base."""
    torch.manual_seed(42)
    base = _build_base_model().to("cpu")

    # Clone the state before pretrained loading modifies out_bias/out_std
    base_state = copy.deepcopy(base.state_dict())

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = copy.deepcopy(base.model)

    dataset_info = _make_dataset_info()
    pretrained = DPA3(hypers, dataset_info).to("cpu")
    # Restore base's original state so both models have the same weights
    # (pretrained zeroed out_bias/out_std, so load original state back)
    pretrained.load_state_dict(base_state)

    systems = read_systems(DATASET_PATH)[:3]
    systems = [s.to(torch.float64) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, base.requested_neighbor_lists())

    output_request = {
        "mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)
    }
    base_out = base(systems, output_request)
    pretrained_out = pretrained(systems, output_request)

    torch.testing.assert_close(
        base_out["mtt::U0"].block().values,
        pretrained_out["mtt::U0"].block().values,
    )


def test_pretrained_checkpoint_roundtrip():
    """A pretrained model survives get_checkpoint -> load_checkpoint."""
    base = _build_base_model()

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = base.model

    dataset_info = _make_dataset_info()
    pretrained = DPA3(hypers, dataset_info)

    # get_checkpoint should succeed (no pickle errors from dpa3_model)
    checkpoint = pretrained.get_checkpoint()

    # dpa3_model should be stripped from the saved hypers
    assert "dpa3_model" not in checkpoint["model_data"]["model_hypers"]

    # Round-trip: load the checkpoint
    reloaded = DPA3.load_checkpoint(checkpoint, context="restart")

    # The reloaded model should NOT have loaded_dpa3 set (built from hypers)
    assert reloaded.loaded_dpa3 is False

    # Both models should produce the same output
    systems = read_systems(DATASET_PATH)[:2]
    systems = [s.to(torch.float64) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, pretrained.requested_neighbor_lists())

    output_request = {
        "mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)
    }
    orig_out = pretrained(systems, output_request)
    reloaded_out = reloaded(systems, output_request)

    torch.testing.assert_close(
        orig_out["mtt::U0"].block().values,
        reloaded_out["mtt::U0"].block().values,
    )


def test_pretrained_no_double_extraction():
    """If _metatrain_extracted_scaleshift is already set, bias/std are not
    re-extracted (prevents double-zeroing on checkpoint reload)."""
    base = _build_base_model()

    # First load: extracts bias/std
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["dpa3_model"] = base.model
    dataset_info = _make_dataset_info()
    first = DPA3(hypers, dataset_info)

    # Second load with the same model (flag already set)
    hypers2 = copy.deepcopy(MODEL_HYPERS)
    hypers2["dpa3_model"] = base.model  # same Module, flag set
    second = DPA3(hypers2, dataset_info)

    # Second instance should NOT have extracted values (skipped)
    assert second._loaded_out_bias is None
    assert second._loaded_out_std is None
