import copy

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.pydantic import MetatrainValidationError

from .base import ArchitectureTests


class InputTests(ArchitectureTests):
    def test_fixed_composition_weights(self, default_hypers):
        """Tests the correctness of the json schema for fixed_composition_weights"""

        if "fixed_composition_weights" not in default_hypers["training"]:
            pytest.skip(
                "The architecture's trainer does not use fixed_composition_weights"
            )

        hypers = copy.deepcopy(default_hypers)
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
        check_architecture_options(
            name=self.architecture, options=OmegaConf.to_container(hypers)
        )

    def test_fixed_composition_weights_error(self, default_hypers):
        """Test that only input of type Dict[str, Dict[int, float]] are allowed."""
        if "fixed_composition_weights" not in default_hypers["training"]:
            pytest.skip(
                "The architecture's trainer does not use fixed_composition_weights"
            )

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["fixed_composition_weights"] = {"energy": {"H": 300.0}}
        hypers = OmegaConf.create(hypers)
        with pytest.raises(
            MetatrainValidationError, match=r"Input should be a valid integer"
        ):
            check_architecture_options(
                name=self.architecture, options=OmegaConf.to_container(hypers)
            )
