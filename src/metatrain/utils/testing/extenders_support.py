import copy

from omegaconf import OmegaConf

from metatrain.utils.architectures import get_hypers_classes
from metatrain.utils.data import DatasetInfo
from metatrain.utils.extenders import ExtendersHypers, get_default_hypers
from metatrain.utils.extenders.identity import Identity
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from .architectures import ArchitectureTests


class ExtendersSupportTests(ArchitectureTests):
    """Test suite to check that an architecture correctly
    supports extenders."""

    def test_extenders_hyper(self) -> None:
        """Test that the architecture correctly supports extenders."""

        hypers_classes = get_hypers_classes(self.architecture)

        trainer_hypers = hypers_classes["trainer"].__annotations__

        # Check that the extenders hyper exists
        assert "extenders" in trainer_hypers, (
            "The trainer hypers must contain a 'extenders' key."
        )

        # Check that it has the right annotation
        assert trainer_hypers["extenders"] is ExtendersHypers, (
            "The 'extenders' key must be annotated with "
            "metatrain.utils.extenders.ExtendersHypers."
        )

    def test_extenders_setup(
        self,
        dataset_info: DatasetInfo,
        minimal_model_hypers: dict,
        default_hypers: dict,
    ) -> None:
        """Test that the architecture correctly sets up the extenders.

        :param dataset_info: The dataset information.
        :param minimal_model_hypers: Minimal model hyperparameters,
          we won't even use the model, it's just to check the extender
          set up.
        :param default_hypers: The default hyper-parameters for the architecture.
        """

        hypers = copy.deepcopy(default_hypers)
        if "num_epochs" in hypers["training"]:
            hypers["training"]["num_epochs"] = 0
        if "loss" in hypers["training"]:
            loss_conf = OmegaConf.create(
                {k: init_with_defaults(LossSpecification) for k in dataset_info.targets}
            )
            OmegaConf.resolve(loss_conf)
            hypers["training"]["loss"] = loss_conf

        # Ask for an identity extender that handles the first target of the dataset.
        first_target = list(dataset_info.targets)[0]
        extenders_hypers = {"identity": get_default_hypers("identity")}
        extenders_hypers["identity"]["outputs"] = first_target
        hypers["training"]["extenders"] = extenders_hypers

        # Setup model
        trainer = self.trainer_cls(hypers["training"])
        model = trainer.setup(minimal_model_hypers, dataset_info)

        assert len(model.extenders) == 1, (
            "The model should have one forward extender set."
            f" Found {len(model.extenders)} forward extenders instead."
        )

        assert isinstance(model.extenders[0], Identity), (
            "The model's forward extender should be an instance of Identity."
            f" Found {type(model.extenders[0])} instead."
        )

        # Check that the core model does not have the extender's output
        # as an output. Instead, the core model should be able to output
        # whatever the extender needs as input.
        core_outputs = model.core.supported_outputs()
        assert first_target not in core_outputs, (
            "The core model should not have the extender's output as an output."
            f" Found {first_target} in the core model's outputs: {core_outputs}"
        )

        requested_inputs = model.extenders[0].requested_target_infos()
        extender_input = list(requested_inputs)[0]
        assert extender_input in core_outputs, (
            "The core model should have the extender's input as an output."
            f" Found {extender_input} in the core model's outputs: {core_outputs}"
        )

        assert requested_inputs[extender_input] == core_outputs[extender_input], (
            "The core model's output should match the extender's requested input."
            f" Found {core_outputs[extender_input]} for the core model's output and "
            f"{requested_inputs[extender_input]} for the extender's requested input."
        )
