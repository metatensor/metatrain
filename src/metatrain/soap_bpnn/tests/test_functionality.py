import copy

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.pydantic import MetatrainValidationError

from . import DEFAULT_HYPERS

def test_fixed_composition_weights():
    """Tests the correctness of the json schema for fixed_composition_weights"""

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["fixed_composition_weights"] = {
        "energy": {
            1: 1.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 3000.0,
        }
    }
    hypers = OmegaConf.create(hypers)
    check_architecture_options(name="soap_bpnn", options=OmegaConf.to_container(hypers))


def test_fixed_composition_weights_error():
    """Test that only input of type Dict[str, Dict[int, float]] are allowed."""
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["fixed_composition_weights"] = {"energy": {"H": 300.0}}
    hypers = OmegaConf.create(hypers)
    with pytest.raises(
        MetatrainValidationError, match=r"Input should be a valid integer"
    ):
        check_architecture_options(
            name="soap_bpnn", options=OmegaConf.to_container(hypers)
        )
