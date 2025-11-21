import copy

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.pydantic import MetatrainValidationError

from .architectures import ArchitectureTests


class InputTests(ArchitectureTests):
    """Test suite to check that the model handles inputs correctly."""

    def test_fixed_composition_weights(self, default_hypers: dict) -> None:
        """Test that the trainer can accept fixed composition weights.

        The tests checks that when providing valid fixed composition weights,
        the architecture options are accepted.

        This test is skipped if the architecture's trainer does not use
        ``fixed_composition_weights``.

        If this test is failing you need to add the correct type hint to
        the ``fixed_composition_weights`` field of the trainer hypers.
        I.e., in ``documentation.py`` of your architecture:

        .. code-block:: python

            from typing_extensions import TypedDict

            from metatrain.utils.additive import FixedCompositionWeights


            class TrainerHypers(TypedDict):
                ...  # Rest of hyperparameters
                fixed_composition_weights: FixedCompositionWeights

        with the appropiate documentation and default if applicable.

        :param default_hypers: The default hyperparameters for the architecture.
        """

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

    def test_fixed_composition_weights_error(self, default_hypers: dict) -> None:
        """Test that invalid input is not accepted for ``fixed_composition_weights``.

        The tests checks that when providing invalid fixed composition weights,
        the architecture options raise a validation error.

        This test is skipped if the architecture's trainer does not use
        ``fixed_composition_weights``.

        If this test is failing you need to add the correct type hint to
        the ``fixed_composition_weights`` field of the trainer hypers.
        I.e., in ``documentation.py`` of your architecture:

        .. code-block:: python

            from typing_extensions import TypedDict

            from metatrain.utils.additive import FixedCompositionWeights


            class TrainerHypers(TypedDict):
                ...  # Rest of hyperparameters
                fixed_composition_weights: FixedCompositionWeights

        with the appropiate documentation and default if applicable.

        :param default_hypers: The default hyperparameters for the architecture.
        """
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
