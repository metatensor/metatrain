import torch
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo

from .hook import HookTests


class OutputTests(HookTests):
    def test_empty(self, hypers: dict, dataset_info: DatasetInfo) -> None:
        """Test that the hook returns an empty dictionary
        when no outputs are requested.

        :param hypers: A dictionary with the hook's hyper-parameters.
        :param dataset_info: Information containing details about the dataset, such as
            target quantities and atomic types.
        """
        hook = self.hook_cls(hypers, dataset_info)

        # We call the forward method of the hook with empty inputs
        # and outputs, and a dummy system.
        outputs: dict = {}
        inputs: dict = {}
        system = System(
            positions=torch.zeros((1, 3)),
            types=torch.tensor([1]),
            cell=torch.eye(3),
            pbc=torch.tensor([True, True, True]),
        )

        result = hook(
            systems=[system], outputs=outputs, inputs=inputs, selected_atoms=None
        )

        # Check that the result is an empty dictionary
        assert result == {}

    def test_output(
        self, dataset_info: DatasetInfo, extra_data: dict, targets: dict
    ) -> None:
        """Test that the hook returns the correct output.

        :param dataset_info: Information containing details about the dataset,
            such as target quantities and atomic types.
        :param extra_data: A dictionary containing the target names that
            are present in the dataset's extra_data.
        :param targets: A dictionary containing the target names that are
            present in the dataset's targets.
        """

        input = extra_data
        output = targets

        hypers, expected_inputs, expected_outputs = self.build_hypers(
            input=input, output=output
        )

        hook = self.hook_cls(hypers, dataset_info)

        # We call the forward method with random values
        systems, inputs = self.get_random(dataset_info, expected_inputs)

        out_targets = {
            out_name: dataset_info.targets[out_name] for out_name in expected_outputs
        }

        result = hook(
            systems=systems,
            outputs=hook.supported_outputs(),
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
