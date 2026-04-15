# mypy: ignore-errors
# Satisfying mypy in this file is hard because the fixtures
# define different classes depending on the parametrization.
import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from metatrain.utils.pydantic import (
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


def test_model_seed_optional():
    """model_seed is optional and accepts non-negative int."""
    config = {
        "architecture": {"name": "soap_bpnn"},
        "training_set": {"systems": "data.xyz", "targets": {"energy": {"key": "E"}}},
        "validation_set": 0.1,
        "seed": 0,
        "model_seed": 42,
    }
    validate_base_options(config)  # should not raise
