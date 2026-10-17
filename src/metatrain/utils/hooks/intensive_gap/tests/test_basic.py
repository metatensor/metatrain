import pytest

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
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


LEARNABLE_POOLING = {
    "type": "softmax",
    "alpha_bottom": 10.0,
    "alpha_top": -10.0,
    "learnable_alpha": True,
}


class IntensiveGapLearnableTests(IntensiveGapTests):
    """Same as :py:class:`IntensiveGapTests`, with learnable per-atom sharpness."""

    @pytest.fixture
    def hypers(self):
        return {"outputs": "mtt::energy", "pooling": dict(LEARNABLE_POOLING)}

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:
        global_energy = get_energy_target_info(
            "energy",
            dict(sample_kind="system", unit="eV", quantity="energy", num_subtargets=1),
        )
        local_energy = get_generic_target_info(
            "mtt::local_energy",
            dict(
                sample_kind="atom",
                unit="eV",
                quantity="energy",
                num_subtargets=1,
                type="scalar",
            ),
        )
        return DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={"mtt::energy": global_energy},
            extra_data={
                f"mtt::extra_local_energy{suffix}": local_energy
                for suffix in ["", "_1", "_2", "_3"]
            },
        )

    def build_hypers(self, input, output):
        hypers, input_names, output_names = super().build_hypers(input, output)
        hypers["pooling"] = dict(LEARNABLE_POOLING)
        if input is None:
            base_name = output["energy"].replace("mtt::", "")
            input_names += [
                f"mtt::aux::gap_bottom_logalpha::{base_name}",
                f"mtt::aux::gap_top_logalpha::{base_name}",
            ]
        else:
            alpha_bottom = f"{input['local_energy']}_2"
            alpha_top = f"{input['local_energy']}_3"
            hypers["inputs"].update(alpha_bottom=alpha_bottom, alpha_top=alpha_top)
            input_names += [alpha_bottom, alpha_top]
        return hypers, input_names, output_names


class TestLearnableTorchscript(TorchscriptTests, IntensiveGapLearnableTests): ...


class TestLearnableOutputs(OutputTests, IntensiveGapLearnableTests): ...
