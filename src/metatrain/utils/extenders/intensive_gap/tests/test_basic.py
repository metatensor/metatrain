import pytest

from metatrain.utils.extenders.helpers import get_default_hypers
from metatrain.utils.extenders.testing.extender import ExtenderTests
from metatrain.utils.extenders.testing.inputs import InputTests
from metatrain.utils.extenders.testing.output import OutputTests
from metatrain.utils.extenders.testing.torchscript import TorchscriptTests


class IntensiveGapTests(ExtenderTests):
    extender = "intensive_gap"

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.extender)
        hypers["outputs"] = "mtt::energy"
        return hypers

    def build_hypers(self, input, output):
        hypers = get_default_hypers(self.extender)
        if input is None:
            output_name = output["energy"]
            input_names = [
                f"mtt::extender_aux::gap_bottom::{output_name.replace('mtt::', '')}",
                f"mtt::extender_aux::gap_top::{output_name.replace('mtt::', '')}",
            ]
            return {**hypers, "outputs": output_name}, input_names, [output_name]
        elif output is None:
            ...
        else:
            bottom_name = input["local_energy"]
            top_name = f"{bottom_name}_1"
            output_name = output["energy"]
            return (
                {
                    **hypers,
                    "inputs": {"bottom": bottom_name, "top": top_name},
                    "outputs": output_name,
                },
                [bottom_name, top_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, IntensiveGapTests): ...


class TestInputs(InputTests, IntensiveGapTests): ...


class TestOutputs(OutputTests, IntensiveGapTests): ...
