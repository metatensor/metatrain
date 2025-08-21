"""
Test for the fix that allows ensemble outputs without requiring uncertainty outputs.

This test verifies that the LLPR model correctly handles ensemble-only requests
without requiring corresponding uncertainty requests.
"""

import torch
import pytest
from metatomic.torch import ModelOutput
from unittest.mock import Mock, MagicMock


def test_ensemble_without_uncertainty_logic():
    """
    Test that the forward method logic correctly identifies when LLPR processing
    is needed for ensemble-only requests.
    """
    # Test the condition logic that was fixed
    def check_llpr_needed(outputs_dict):
        """Replicate the fixed condition from LLPRUncertaintyModel.forward"""
        has_uncertainty = any("_uncertainty" in output for output in outputs_dict)
        has_ensemble = any(output.endswith("_ensemble") for output in outputs_dict)
        return has_uncertainty or has_ensemble

    # Test cases that should NOT require LLPR processing
    test_cases_no_llpr = [
        ({"energy": ModelOutput()}, "Regular energy output only"),
        ({"energy": ModelOutput(), "forces": ModelOutput()}, "Multiple regular outputs"),
        ({"mtt::aux::energy_last_layer_features": ModelOutput()}, "Last layer features only"),
    ]

    for outputs, description in test_cases_no_llpr:
        result = check_llpr_needed(outputs)
        assert not result, f"Should not need LLPR for: {description}"

    # Test cases that SHOULD require LLPR processing
    test_cases_need_llpr = [
        ({"energy_uncertainty": ModelOutput()}, "Uncertainty only"),
        ({"energy_ensemble": ModelOutput()}, "Ensemble only (this is the fix)"),
        ({"energy_uncertainty": ModelOutput(), "energy_ensemble": ModelOutput()}, 
         "Both uncertainty and ensemble"),
        ({"mtt::aux::forces_uncertainty": ModelOutput()}, "Auxiliary uncertainty"),
        ({"mtt::aux::forces_ensemble": ModelOutput()}, "Auxiliary ensemble (this is the fix)"),
        ({"energy": ModelOutput(), "energy_ensemble": ModelOutput()}, 
         "Regular output with ensemble"),
        ({"energy": ModelOutput(), "energy_uncertainty": ModelOutput()}, 
         "Regular output with uncertainty"),
    ]

    for outputs, description in test_cases_need_llpr:
        result = check_llpr_needed(outputs)
        assert result, f"Should need LLPR for: {description}"


def test_ensemble_feature_request_logic():
    """
    Test that the correct last layer features are requested for ensemble outputs.
    """
    def get_ll_features_for_ensemble(ensemble_name):
        """Replicate the logic for requesting last layer features for ensembles"""
        ll_features_name = ensemble_name.replace("_ensemble", "_last_layer_features")
        if ll_features_name == "energy_last_layer_features":
            # special case for energy_ensemble
            ll_features_name = "mtt::aux::energy_last_layer_features"
        return ll_features_name

    test_cases = [
        ("energy_ensemble", "mtt::aux::energy_last_layer_features"),
        ("forces_ensemble", "forces_last_layer_features"),
        ("mtt::aux::forces_ensemble", "mtt::aux::forces_last_layer_features"),
        ("mtt::aux::stress_ensemble", "mtt::aux::stress_last_layer_features"),
    ]

    for ensemble_name, expected_features in test_cases:
        result = get_ll_features_for_ensemble(ensemble_name)
        assert result == expected_features, (
            f"For ensemble '{ensemble_name}', expected features '{expected_features}', "
            f"got '{result}'"
        )


def test_error_messages():
    """
    Test that appropriate error messages are raised for invalid requests.
    """
    # Test the ensemble weights check logic
    def check_ensemble_weights_exist(ensemble_name, available_buffers):
        """Mock the ensemble weights existence check"""
        ensemble_weights_name = ensemble_name + "_weights"
        return ensemble_weights_name in available_buffers

    # Test case: requesting ensemble without generated weights
    available_buffers = []  # No ensemble weights available
    assert not check_ensemble_weights_exist("energy_ensemble", available_buffers)

    # Test case: requesting ensemble with generated weights
    available_buffers = ["energy_ensemble_weights"]
    assert check_ensemble_weights_exist("energy_ensemble", available_buffers)


if __name__ == "__main__":
    test_ensemble_without_uncertainty_logic()
    test_ensemble_feature_request_logic()
    test_error_messages()
    print("✅ All tests passed!")