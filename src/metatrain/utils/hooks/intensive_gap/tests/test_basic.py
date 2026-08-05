import pytest

from metatrain.utils.hooks.testing.hook import HookTests
from metatrain.utils.hooks.testing.inputs import InputTests
from metatrain.utils.hooks.testing.output import OutputTests
from metatrain.utils.hooks.testing.torchscript import TorchscriptTests


class IntensiveGapTests(HookTests):
    hook = "intensive_gap"

    @pytest.fixture
    def hypers(self):
        return {"outputs": "mtt::energy"}

    def build_hypers(self, input, output):
        if input is None:
            output_name = output["energy"]
            input_names = [
                f"mtt::aux::gap_bottom::{output_name.replace('mtt::', '')}",
                f"mtt::aux::gap_top::{output_name.replace('mtt::', '')}",
            ]
            return {"outputs": output_name}, input_names, [output_name]
        elif output is None:
            ...
        else:
            bottom_name = input["local_energy"]
            top_name = f"{bottom_name}_1"
            output_name = output["energy"]
            return (
                {
                    "inputs": {"bottom": bottom_name, "top": top_name},
                    "outputs": output_name,
                },
                [bottom_name, top_name],
                [output_name],
            )


class TestTorchscript(TorchscriptTests, IntensiveGapTests): ...


class TestInputs(InputTests, IntensiveGapTests): ...


class TestOutputs(OutputTests, IntensiveGapTests): ...
