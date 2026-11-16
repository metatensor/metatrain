import pytest

from metatrain.utils.extenders.helpers import get_default_hypers
from metatrain.utils.extenders.testing.extender import ExtenderTests
from metatrain.utils.extenders.testing.inputs import InputTests
from metatrain.utils.extenders.testing.output import OutputTests
from metatrain.utils.extenders.testing.torchscript import TorchscriptTests


class IdentityTests(ExtenderTests):
    extender = "rename"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.extender)
        hypers["outputs"] = "mtt::energy"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.extender)
        if input is None:
            output_name = output["energy"]
            input_name = (
                f"mtt::extender_aux::rename::{output_name.replace('mtt::', '')}"
            )
            return {**hypers, "outputs": output_name}, [input_name], [output_name]
        elif output is None:
            input_name = input["energy"]
            output_name = (
                f"mtt::extender_aux::rename::{input_name.replace('mtt::', '')}"
            )
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
