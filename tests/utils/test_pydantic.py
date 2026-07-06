# mypy: ignore-errors
# Satisfying mypy in this file is hard because the fixtures
# define different classes depending on the parametrization.
import pytest
import requests
from pydantic import BaseModel
from typing_extensions import TypedDict

from metatrain.utils.pydantic import (
    MetatrainArchitectureValidationError,
    MetatrainValidationError,
    validate,
    validate_architecture_options,
    validate_base_options,
)


@pytest.fixture(params=["typed_dict", "pydantic_model"])
def simple_hypers_class(request: pytest.FixtureRequest):
    if request.param == "typed_dict":

        class Hypers(TypedDict):
            a: float = 2.0

    elif request.param == "pydantic_model":

        class Hypers(BaseModel):
            a: float = 2.0

    return Hypers


def test_validate_success(simple_hypers_class: type):
    """Test that valid data passes validation.

    :param simple_hypers_class: A simple hypers class, either a TypedDict
    or Pydantic model.
    """
    data = {"a": 3.5}
    # Should not raise an error
    validate(simple_hypers_class, data)


def test_validate_failure(simple_hypers_class: type):
    """Test that invalid data raises a MetatrainValidationError.

    :param simple_hypers_class: A simple hypers class, either a TypedDict or
    Pydantic model.
    """
    data = {"a": "whatever"}
    with pytest.raises(MetatrainValidationError):
        validate(simple_hypers_class, data)


def test_validate_architecture_options(simple_hypers_class: type):
    """Test that architecture options validation works."""

    options = {
        "name": "dummy_architecture",
        "model": {"a": 2.0},
        "training": {"a": 2.0},
    }

    validate_architecture_options(
        options, model_hypers=simple_hypers_class, trainer_hypers=simple_hypers_class
    )


def test_validate_architecture_options_error(simple_hypers_class: type):
    """Test that architecture options validation returns error on wrong hypers."""

    options = {
        "name": "dummy_architecture",
        "model": {"a": "wrong_value"},
        "training": {"a": 2.0},
    }

    with pytest.raises(MetatrainValidationError):
        validate_architecture_options(
            options,
            model_hypers=simple_hypers_class,
            trainer_hypers=simple_hypers_class,
        )


def test_validate_architecture_options_warning(caplog):
    """Test that architecture options are not validated if hypers are not validatable.

    This should not fail, but should issue a warning.
    """

    class Hypers:
        a = 2.0

    options = {
        "name": "dummy_architecture",
        "model": {"a": False},
        "training": {"a": 2.0},
    }

    validate_architecture_options(options, model_hypers=Hypers, trainer_hypers=Hypers)

    assert "Architecture does not provide validation of hyperparameters" in caplog.text


# ============================================================================
# Tests for validate_base_options with indices
# ============================================================================


def test_indices_only_validation_set_list():
    """validation_set accepts indices-only config with list."""
    config = {
        "architecture": {"name": "soap_bpnn"},
        "training_set": {"systems": "data.xyz", "targets": {"energy": {"key": "E"}}},
        "validation_set": {"indices": [0, 1, 2]},
    }
    validate_base_options(config)  # should not raise


def test_indices_only_validation_set_string():
    """validation_set accepts indices-only config with file path."""
    config = {
        "architecture": {"name": "soap_bpnn"},
        "training_set": {"systems": "data.xyz", "targets": {"energy": {"key": "E"}}},
        "validation_set": {"indices": "indices.txt"},
    }
    validate_base_options(config)  # should not raise


def test_indices_in_full_dataset_config():
    """Dataset config accepts optional indices field."""
    config = {
        "architecture": {"name": "soap_bpnn"},
        "training_set": {
            "systems": "data.xyz",
            "targets": {"energy": {"key": "E"}},
            "indices": [0, 1, 2, 3],
        },
        "validation_set": 0.1,
    }
    validate_base_options(config)  # should not raise


def test_validation_error_doc_link():
    error_cls = MetatrainArchitectureValidationError.for_architecture("pet")

    error = error_cls(model=None, errors=[])

    modelhypers_link = error.architecture_link("ModelHypers")
    trainerhypers_link = error.architecture_link("TrainerHypers")

    # Check that links are reachable.
    # This does not check that the fragment identifier (thing after #)
    # exists, but it is the best we can do without parsing the HTML.
    for link in [modelhypers_link, trainerhypers_link]:
        response = requests.head(link)
        assert response.status_code == 200, f"Link {link} is not reachable"


# ============================================================================
# Tests for validate_base_options with final_evaluation
# ============================================================================

_BASE_CONFIG = {
    "architecture": {"name": "soap_bpnn"},
    "training_set": "data.xyz",
    "validation_set": 0.1,
}


@pytest.mark.parametrize("fmt", ["xyz", "memmap"])
def test_final_evaluation_valid_format(fmt):
    """Both 'xyz' and 'memmap' are accepted as final_evaluation.format."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {"write_predictions": True, "format": fmt},
    }
    validate_base_options(config)  # must not raise


def test_final_evaluation_write_predictions_false():
    """write_predictions=False is valid and is the documented default."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {"write_predictions": False},
    }
    validate_base_options(config)  # must not raise


def test_final_evaluation_invalid_format():
    """Unknown format values are rejected with a clear validation error."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {"write_predictions": True, "format": "hdf5"},
    }
    with pytest.raises(MetatrainValidationError, match="format"):
        validate_base_options(config)


def test_final_evaluation_unknown_key():
    """Extra keys inside final_evaluation are rejected (extra='forbid')."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {"write_predictions": True, "unknown_option": 42},
    }
    with pytest.raises(MetatrainValidationError, match="unknown_option"):
        validate_base_options(config)


def test_final_evaluation_omitted():
    """Omitting final_evaluation entirely is valid (it is NotRequired)."""
    validate_base_options(_BASE_CONFIG)  # must not raise


@pytest.mark.parametrize(
    "split", ["write_training_set", "write_validation_set", "write_test_set"]
)
def test_final_evaluation_split_toggle(split):
    """Each split flag inside final_evaluation accepts a bool, default True."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {split: False},
    }
    validate_base_options(config)  # must not raise


def test_final_evaluation_split_toggle_invalid_type():
    """Non-bool values for the split toggles are rejected."""
    config = {
        **_BASE_CONFIG,
        "final_evaluation": {"write_training_set": "yes"},
    }
    with pytest.raises(MetatrainValidationError, match="write_training_set"):
        validate_base_options(config)
