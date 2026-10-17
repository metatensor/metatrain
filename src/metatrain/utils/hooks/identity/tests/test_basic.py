import pytest

from metatrain.utils.hooks.testing.hook import HookTests
from metatrain.utils.hooks.testing.inputs import InputTests
from metatrain.utils.hooks.testing.output import OutputTests
from metatrain.utils.hooks.testing.torchscript import TorchscriptTests


class IdentityTests(HookTests):
    hook = "identity"

    @pytest.fixture
    def hypers(self):
        return {"outputs": "mtt::energy"}

    def build_hypers(self, input, output):
        if input is None:
            output_name = output["energy"]
            input_name = f"mtt::aux::identity::{output_name.replace('mtt::', '')}"
            return {"outputs": output_name}, [input_name], [output_name]
        elif output is None:
            input_name = input["energy"]
            output_name = f"mtt::aux::identity::{input_name.replace('mtt::', '')}"
            return {"inputs": input_name}, [input_name], [output_name]
        else:
            input_name = input["energy"]
            output_name = output["energy"]
            return (
                {"inputs": input_name, "outputs": output_name},
                [input_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, IdentityTests): ...


class TestInputs(InputTests, IdentityTests): ...


class TestOutputs(OutputTests, IdentityTests): ...
