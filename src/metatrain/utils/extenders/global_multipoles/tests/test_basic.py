import pytest

from metatrain.utils.extenders.helpers import get_default_hypers
from metatrain.utils.extenders.testing.extender import ExtenderTests
from metatrain.utils.extenders.testing.inputs import InputTests
from metatrain.utils.extenders.testing.output import OutputTests
from metatrain.utils.extenders.testing.torchscript import TorchscriptTests


class GlobalMultipolesTests(ExtenderTests):
    extender = "global_multipoles"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.extender)
        hypers["outputs"] = "mtt::global_dipole"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.extender)
        if input is None:
            output_name = output["global_dipole"]
            output_suffix = output_name.replace("mtt::", "")
            input_name = f"mtt::extender_aux::global_multipoles::{output_suffix}"
            return {**hypers, "outputs": output_name}, [input_name], [output_name]
        elif output is None:
            ...
        else:
            input_name = input["local_multipole"]
            output_name = output["global_dipole"]
            return (
                {**hypers, "inputs": input_name, "outputs": output_name},
                [input_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, GlobalMultipolesTests): ...


class TestInputs(InputTests, GlobalMultipolesTests): ...


class TestOutputs(OutputTests, GlobalMultipolesTests): ...
