import copy

from omegaconf import OmegaConf

from metatrain.utils.architectures import get_hypers_classes
from metatrain.utils.hooks import HooksHypers, get_default_hypers
from metatrain.utils.hooks.identity import IdentityHook
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from .architectures import ArchitectureTests


class HooksSupportTests(ArchitectureTests):
    """Test suite to check that an architecture correctly
    supports hooks."""

    def test_hooks_hyper(self):
        """Test that the architecture correctly supports hooks."""

        hypers_classes = get_hypers_classes(self.architecture)

        trainer_hypers = hypers_classes["trainer"].__annotations__

        # Check that the hooks hyper exists
        assert "forward_hooks" in trainer_hypers, (
            "The trainer hypers must contain a 'forward_hooks' key."
        )

        # Check that it has the right annotation
        assert trainer_hypers["forward_hooks"] is HooksHypers, (
            "The 'forward_hooks' key must be annotated with "
            "metatrain.utils.hooks.HooksHypers."
        )

    def test_hooks_setup(self, dataset_info, minimal_model_hypers, default_hypers):
        """Test that the architecture correctly sets the hooks."""

        hypers = copy.deepcopy(default_hypers)
        if "num_epochs" in hypers["training"]:
            hypers["training"]["num_epochs"] = 0
        if "loss" in hypers["training"]:
            loss_conf = OmegaConf.create(
                {k: init_with_defaults(LossSpecification) for k in dataset_info.targets}
            )
            OmegaConf.resolve(loss_conf)
            hypers["training"]["loss"] = loss_conf

        # Ask for an identity hook that handles the first target of the dataset.
        first_target = list(dataset_info.targets)[0]
        hooks_hypers = {"identity": get_default_hypers("identity")}
        hooks_hypers["identity"]["outputs"] = first_target
        hypers["training"]["forward_hooks"] = hooks_hypers

        # Setup model
        trainer = self.trainer_cls(hypers["training"])
        model = trainer.setup(minimal_model_hypers, dataset_info)

        assert len(model.forward_hooks) == 1, (
            "The model should have one forward hook set."
            f" Found {len(model.forward_hooks)} forward hooks instead."
        )

        assert isinstance(model.forward_hooks[0], IdentityHook), (
            "The model's forward hook should be an instance of IdentityHook."
            f" Found {type(model.forward_hooks[0])} instead."
        )

        # Check that the core model does not have the hook's output
        # as an output. Instead, the core model should be able to output
        # whatever the hook needs as input.
        core_outputs = model.core.supported_outputs()
        assert first_target not in core_outputs, (
            "The core model should not have the hook's output as an output."
            f" Found {first_target} in the core model's outputs: {core_outputs}"
        )

        requested_inputs = model.forward_hooks[0].requested_target_infos()
        hook_input = list(requested_inputs)[0]
        assert hook_input in core_outputs, (
            "The core model should have the hook's input as an output."
            f" Found {hook_input} in the core model's outputs: {core_outputs}"
        )

        assert requested_inputs[hook_input] == core_outputs[hook_input], (
            "The core model's output should match the hook's requested input."
            f" Found {core_outputs[hook_input]} for the core model's output and "
            f"{requested_inputs[hook_input]} for the hook's requested input."
        )
