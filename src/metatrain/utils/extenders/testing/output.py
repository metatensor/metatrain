import pytest
import torch
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .extender import ExtenderTests


class OutputTests(ExtenderTests):
    @pytest.fixture(params=[torch.float32, torch.float64])
    def dtype(self, request: pytest.FixtureRequest) -> torch.dtype:
        """Data type in which to run the tests.

        :param request: the pytest request.
        :return: the dtype
        """
        return request.param

    @pytest.fixture(params=["cpu", "cuda"])
    def device(self, request: pytest.FixtureRequest) -> str:
        """Device in which to run the tests.

        :param request: the pytest request.
        :return: the device
        """
        if not torch.cuda.is_available() and request.param == "cuda":
            pytest.skip("CUDA is not available")
        return request.param

    def test_empty(
        self, hypers: dict, dataset_info: DatasetInfo, dtype: torch.dtype, device: str
    ) -> None:
        """Test that the extender returns an empty dictionary
        when no outputs are requested.

        :param hypers: A dictionary with the extender's hyper-parameters.
        :param dataset_info: Information containing details about the dataset, such as
            target quantities and atomic types.
        :param dtype: dtype in which to run the tests.
        :param device: device in which to run the tests.
        """
        extender = self.extender_cls(hypers, dataset_info)

        # We call the forward method of the extender with empty inputs
        # and outputs, and a dummy system.
        outputs: dict = {}
        inputs: dict = {}
        system = System(
            positions=torch.zeros((1, 3)),
            types=torch.tensor([1]),
            cell=torch.eye(3),
            pbc=torch.tensor([True, True, True]),
        )

        # Move everything to the requested dtype and device.
        system = system.to(dtype=dtype, device=device)
        extender = extender.to(dtype=dtype, device=device)

        result = extender(
            systems=[system], outputs=outputs, inputs=inputs, selected_atoms=None
        )

        # Check that the result is an empty dictionary
        assert result == {}

    def test_output(
        self,
        dataset_info: DatasetInfo,
        extra_data: dict,
        targets: dict,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        """Test that the extender returns the correct output.

        :param dataset_info: Information containing details about the dataset,
            such as target quantities and atomic types.
        :param extra_data: A dictionary containing the target names that
            are present in the dataset's extra_data.
        :param targets: A dictionary containing the target names that are
            present in the dataset's targets.
        :param dtype: dtype in which to run the tests.
        :param device: device in which to run the tests.
        """

        input = extra_data
        output = targets

        hypers, expected_inputs, expected_outputs = self.build_hypers(
            input=input, output=output
        )

        extender = self.extender_cls(hypers, dataset_info)

        # We call the forward method with random values
        systems, inputs = self.get_random(dataset_info, expected_inputs)

        systems = [
            get_system_with_neighbor_lists(
                system, get_requested_neighbor_lists(extender)
            )
            for system in systems
        ]

        # Move everything to the requested dtype and device.
        systems = [system.to(dtype=dtype, device=device) for system in systems]
        inputs = {k: v.to(dtype=dtype, device=device) for k, v in inputs.items()}
        extender = extender.to(dtype=dtype, device=device)

        out_targets = {
            out_name: dataset_info.targets[out_name] for out_name in expected_outputs
        }

        result = extender(
            systems=systems,
            outputs=extender.supported_outputs(),
            inputs=inputs,
            selected_atoms=None,
        )

        # Check that the result contains all requested outputs
        assert set(result.keys()) == set(expected_outputs)
        # Check that each output has the expected metadata
        for out_name in expected_outputs:
            out_tmap = result[out_name]
            layout = out_targets[out_name].layout

            assert out_tmap.keys == layout.keys, (
                f"Output {out_name} has keys {out_tmap.keys}, "
                f"but expected {layout.keys}"
            )

            for key, layout_block in layout.items():
                out_block = out_tmap[key]
                assert out_block.samples.names == layout_block.samples.names
                assert out_block.samples.values.shape[0] == 1

                assert out_block.components == layout_block.components, (
                    f"Output {out_name} block {key} has components "
                    f"{out_block.components}, "
                    f"but expected {layout_block.components}"
                )

                assert out_block.properties == layout_block.properties, (
                    f"Output {out_name} block {key} has properties "
                    f"{out_block.properties}, "
                    f"but expected {layout_block.properties}"
                )
