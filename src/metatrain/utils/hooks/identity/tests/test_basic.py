import pytest

from metatrain.utils.hooks.helpers import get_default_hypers
from metatrain.utils.hooks.testing.hook import HookTests
from metatrain.utils.hooks.testing.inputs import InputTests
from metatrain.utils.hooks.testing.output import OutputTests
from metatrain.utils.hooks.testing.torchscript import TorchscriptTests


class IdentityTests(HookTests):
    hook = "identity"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.hook)
        hypers["outputs"] = "mtt::energy"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.hook)
        if input is None:
            output_name = output["energy"]
            input_name = f"mtt::hook_aux::identity::{output_name.replace('mtt::', '')}"
            return {**hypers, "outputs": output_name}, [input_name], [output_name]
        elif output is None:
            input_name = input["energy"]
            output_name = f"mtt::hook_aux::identity::{input_name.replace('mtt::', '')}"
            return {**hypers, "inputs": input_name}, [input_name], [output_name]
        else:
            input_name = input["energy"]
            output_name = output["energy"]
            return (
                {**hypers, "inputs": input_name, "outputs": output_name},
                [input_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, IdentityTests): ...


class TestInputs(InputTests, IdentityTests): ...


class TestOutputs(OutputTests, IdentityTests): ...
