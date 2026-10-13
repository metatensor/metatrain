import pytest

from metatrain.utils.hooks.testing.hook import HookTests
from metatrain.utils.hooks.testing.inputs import InputTests
from metatrain.utils.hooks.testing.output import OutputTests
from metatrain.utils.hooks.testing.torchscript import TorchscriptTests


class TensorBasisTests(HookTests):
    hook = "tensor_basis"

    @pytest.fixture
    def hypers(self):
        return {"outputs": "mtt::global_dipole"}

    def build_hypers(self, input, output):
        if input is None:
            output_name = output["global_dipole"]
            input_name = f"mtt::aux::scalars::{output_name.replace('mtt::', '')}"
            return {"outputs": output_name}, [input_name], [output_name]
        elif output is None:
            ...
        else:
            input_name = input["local_energy"]
            output_name = output["global_dipole"]
            return (
                {"inputs": input_name, "outputs": output_name},
                [input_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, TensorBasisTests): ...


class TestInputs(InputTests, TensorBasisTests): ...


class TestOutputs(OutputTests, TensorBasisTests): ...
