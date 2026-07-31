import copy
from typing import Any

import pytest
from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import DatasetInfo
from metatrain.utils.pydantic import MetatrainValidationError

from .architectures import ArchitectureTests


class InputTests(ArchitectureTests):
    """Test suite to check that the model handles inputs correctly."""

    supports_restart: bool = True
    """Whether the architecture supports restarting training."""

    def test_atomic_baseline(self, default_hypers: dict) -> None:
        """Test that the trainer can accept atomic baselines.

        The tests checks that when providing valid atomic baselines,
        the architecture options are accepted.

        This test is skipped if the architecture's trainer does not use
        ``atomic_baseline``.
        If this test is failing you need to add the correct type hint to
        the ``atomic_baseline`` field of the trainer hypers.
        I.e., in ``documentation.py`` of your architecture:

        .. code-block:: python

            from typing_extensions import TypedDict

            from metatrain.composition.documentation import FixedCompositionWeights


            class TrainerHypers(TypedDict):
                ...  # Rest of hyperparameters
                atomic_baseline: FixedCompositionWeights

        with the appropiate documentation and default if applicable.

        :param default_hypers: The default hyperparameters for the architecture.
        """

        if "atomic_baseline" not in default_hypers["training"]:
            pytest.skip("The architecture's trainer does not use atomic_baseline")

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["atomic_baseline"] = {
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

        hypers["training"]["atomic_baseline"] = {"energy": 0.0}
        hypers = OmegaConf.create(hypers)
        check_architecture_options(
            name=self.architecture, options=OmegaConf.to_container(hypers)
        )

    def test_atomic_baseline_error(self, default_hypers: dict) -> None:
        """Test that invalid input is not accepted for ``atomic_baseline``.

        The tests checks that when providing invalid atomic baselines,
        the architecture options raise a validation error.

        This test is skipped if the architecture's trainer does not use
        ``atomic_baseline``.

        If this test is failing you need to add the correct type hint to
        the ``atomic_baseline`` field of the trainer hypers.
        I.e., in ``documentation.py`` of your architecture:

        .. code-block:: python

            from typing_extensions import TypedDict

            from metatrain.composition.documentation import FixedCompositionWeights


            class TrainerHypers(TypedDict):
                ...  # Rest of hyperparameters
                atomic_baseline: FixedCompositionWeights

        with the appropiate documentation and default if applicable.

        :param default_hypers: The default hyperparameters for the architecture.
        """
        if "atomic_baseline" not in default_hypers["training"]:
            pytest.skip("The architecture's trainer does not use atomic_baseline")

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["atomic_baseline"] = {"energy": {"H": 300.0}}
        hypers = OmegaConf.create(hypers)
        with pytest.raises(
            MetatrainValidationError, match=r"Input should be a valid integer"
        ):
            check_architecture_options(
                name=self.architecture, options=OmegaConf.to_container(hypers)
            )

    def test_restart(
        self, minimal_model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Test that the model can be restarted with the same hyperparameters
        and same dataset information.

        If the model doesn't support restarting (which should be indicated by
        setting ``supports_restart = False``), a call to ``model.restart()``
        is supposed to raise a ``NotImplementedError``.

        If your model supports restarting, but this test is failing, you need
        to make sure that the model's ``restart`` method is implemented
        correctly. Essentially, a call to ``model.restart()`` with the same
        dataset information and same model hyperparameters should
        keep the model unchanged.

        :param minimal_model_hypers: The hyperparameters used to initialize the
            model.
        :param dataset_info: The dataset information used to initialize the model.
        """

        model = self.model_cls(minimal_model_hypers, dataset_info)

        if not self.supports_restart:
            with pytest.raises(NotImplementedError):
                model.restart(dataset_info=dataset_info)
            return

        # This should work, as we are not changing the hypers
        model.restart(dataset_info=dataset_info)
        model.restart(dataset_info=dataset_info, model_hypers={})
        model.restart(dataset_info=dataset_info, model_hypers=minimal_model_hypers)

    def test_restart_hypers_mismatch(
        self,
        default_hypers: dict,
        minimal_model_hypers: dict,
        dataset_info: DatasetInfo,
    ) -> None:
        """Test that the model throws an error when there is an attempt to
        restart training with different model hyperparameters.

        This test is skipped if the architecture does not support restarting
        (which should be indicated by setting ``supports_restart = False``)
        or if the architecture does not have any hyperparameters.

        If this test is failing, you need to make sure that the model's
        ``restart`` method checks that the provided hyperparameters
        match the ones used to initialize the model. An easy way to do this
        is by doing the following in your model's ``restart`` method:

        .. code-block:: python

            from metatrain.utils.hypers import raise_if_hypers_mismatch


            def restart(
                self,
                dataset_info: DatasetInfo,
                model_hypers: Optional[dict[str, Any]] = None,
            ):
                if model_hypers is not None:
                    raise_if_hypers_mismatch(self.hypers, model_hypers)
                # Rest of the restart logic...

        :param default_hypers: The default hyperparameters for the architecture.
        :param minimal_model_hypers: The hyperparameters used to initialize the
            model.
        :param dataset_info: The dataset information used to initialize the model.
        """
        if not self.supports_restart:
            pytest.skip("The architecture does not support restart")
        if len(default_hypers) == 0:
            pytest.skip("The architecture does not have any hyperparameters")

        model = self.model_cls(minimal_model_hypers, dataset_info)

        # Find an input that is a number or string and change it.
        new_hypers: dict[str, Any] = {}
        for k, v in minimal_model_hypers.items():
            if isinstance(v, (int, float)):
                new_hypers[k] = v + 1.0
                break
            elif isinstance(v, str):
                new_hypers[k] = "new_value"
                break
        else:
            # No numbers or strings found, just change any value to a string
            new_hypers[list(minimal_model_hypers.keys())[0]] = "new_value"

        # This shouldn't work, as the hypers have been changed
        with pytest.raises(ValueError):
            model.restart(dataset_info=dataset_info, model_hypers=new_hypers)
