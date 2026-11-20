import pytest

from metatrain.utils.extenders.helpers import get_default_hypers
from metatrain.utils.extenders.testing.extender import ExtenderTests
from metatrain.utils.extenders.testing.inputs import InputTests
from metatrain.utils.extenders.testing.output import OutputTests
from metatrain.utils.extenders.testing.torchscript import TorchscriptTests


class TensorBasisTests(ExtenderTests):
    extender = "tensor_basis"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.extender)
        hypers["outputs"] = "mtt::global_dipole"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.extender)
        if input is None:
            output_name = output["global_dipole"]
            input_name = (
                f"mtt::extender_aux::tensor_basis::{output_name.replace('mtt::', '')}"
            )
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
