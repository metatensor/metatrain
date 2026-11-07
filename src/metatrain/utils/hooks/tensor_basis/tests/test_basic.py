import pytest

from metatrain.utils.hooks.helpers import get_default_hypers
from metatrain.utils.hooks.testing.hook import HookTests
from metatrain.utils.hooks.testing.inputs import InputTests
from metatrain.utils.hooks.testing.output import OutputTests
from metatrain.utils.hooks.testing.torchscript import TorchscriptTests


class TensorBasisTests(HookTests):
    hook = "tensor_basis"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.hook)
        hypers["outputs"] = "mtt::global_dipole"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.hook)
        if input is None:
            output_name = output["global_dipole"]
            input_name = f"mtt::hook_aux::scalars::{output_name.replace('mtt::', '')}"
            return {**hypers, "outputs": output_name}, [input_name], [output_name]
        elif output is None:
            ...
        else:
            input_name = input["local_dipole_invariant_coeffs"]
            output_name = output["global_dipole"]
            return (
                {**hypers, "inputs": input_name, "outputs": output_name},
                [input_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, TensorBasisTests): ...


class TestInputs(InputTests, TensorBasisTests): ...


class TestOutputs(OutputTests, TensorBasisTests): ...
