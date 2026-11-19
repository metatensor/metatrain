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
            output_suffix = output_name.replace("mtt::", "")
            input_names = [
                f"mtt::extender_aux::intensive_gap::bottom::{output_suffix}",
                f"mtt::extender_aux::intensive_gap::top::{output_suffix}",
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


class LearnableIntensiveGapTests(IntensiveGapTests):
    """The same suites with learnable per-atom sharpness (two more per-atom inputs)."""

    @pytest.fixture
    def hypers(self):
        hypers = get_default_hypers(self.extender)
        hypers["outputs"] = "mtt::energy"
        hypers["learnable_sharpness"] = True
        return hypers

    def build_hypers(self, input, output):
        hypers, input_names, output_names = super().build_hypers(input, output)
        hypers["learnable_sharpness"] = True
        if input is None:
            base = output_names[0].replace("mtt::", "")
            input_names = input_names + [
                f"mtt::extender_aux::intensive_gap::log_sharpness_bottom::{base}",
                f"mtt::extender_aux::intensive_gap::log_sharpness_top::{base}",
            ]
        else:
            bottom_name, top_name = input_names
            hypers["inputs"] = {
                **hypers["inputs"],
                "log_sharpness_bottom": bottom_name,
                "log_sharpness_top": top_name,
            }
        return hypers, input_names, output_names


class TestTorchscriptLearnable(TorchscriptTests, LearnableIntensiveGapTests): ...


class TestInputsLearnable(InputTests, LearnableIntensiveGapTests): ...


class TestOutputsLearnable(OutputTests, LearnableIntensiveGapTests): ...
