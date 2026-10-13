from metatomic.torch import ModelOutput

from metatrain.utils.data import DatasetInfo

from .hook import HookTests


class InputTests(HookTests):
    def test_only_output(self, dataset_info: DatasetInfo, targets: dict) -> None:

        input = None
        output = targets

        hypers, expected_inputs, expected_outputs = self.build_hypers(
            input=input, output=output
        )

        self.check_correctness(
            hypers,
            dataset_info,
            expected_inputs,
            expected_outputs,
            inputs_in_dataset=False,
        )

    def test_output_and_specific_input(
        self, dataset_info: DatasetInfo, intermediate: dict, targets: dict
    ) -> None:

        input = intermediate
        output = targets

        hypers, expected_inputs, expected_outputs = self.build_hypers(
            input=input, output=output
        )

        self.check_correctness(
            hypers,
            dataset_info,
            expected_inputs,
            expected_outputs,
            inputs_in_dataset=False,
        )

    def test_output_and_extradata_input(
        self, dataset_info: DatasetInfo, extra_data: dict, targets: dict
    ) -> None:

        input = extra_data
        output = targets

        hypers, expected_inputs, expected_outputs = self.build_hypers(
            input=input, output=output
        )

        self.check_correctness(
            hypers,
            dataset_info,
            expected_inputs,
            expected_outputs,
        )

    def check_correctness(
        self,
        hypers: dict,
        dataset_info: DatasetInfo,
        expected_inputs: list[str],
        expected_outputs: list[str],
        inputs_in_dataset: bool = True,
        outputs_in_dataset: bool = True,
    ) -> None:
        hook = self.hook_cls(hypers, dataset_info)

        all_target_infos = {**dataset_info.targets, **dataset_info.extra_data}

        requested_target_infos = hook.requested_target_infos()
        requested_inputs = hook.requested_inputs()
        supported_outputs = hook.supported_outputs()

        if inputs_in_dataset:
            expected_inp_targets = {
                in_name: all_target_infos[in_name] for in_name in expected_inputs
            }

            assert requested_target_infos == expected_inp_targets, (
                f"Requested target infos do not match expected inputs.\n"
                f"Expected: {expected_inp_targets}\n"
                f"Got: {requested_target_infos}"
            )

            expected_model_inputs = {
                in_name: ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind=target_info.sample_kind,
                )
                for in_name, target_info in expected_inp_targets.items()
            }

            # Check the ModelOutput objects that the hook requests as inputs.
            assert len(requested_inputs) == len(expected_model_inputs), (
                f"There are more requested inputs than expected. "
                f"Expected: {len(expected_model_inputs)}, Got: {len(requested_inputs)}"
            )
            for in_name, model_output in requested_inputs.items():
                assert in_name in expected_model_inputs, (
                    f"Requested input '{in_name}' is not in the expected inputs."
                )
                expected_output = expected_model_inputs[in_name]
                assert model_output.quantity == expected_output.quantity, (
                    f"Requested input '{in_name}' has quantity "
                    f"'{model_output.quantity}', "
                    f"but expected '{expected_output.quantity}'."
                )
                assert model_output.unit == expected_output.unit, (
                    f"Requested input '{in_name}' has unit '{model_output.unit}', "
                    f"but expected '{expected_output.unit}'."
                )
                assert model_output.sample_kind == expected_output.sample_kind, (
                    f"Requested input '{in_name}' has sample kind "
                    f"'{model_output.sample_kind}', "
                    f"but expected '{expected_output.sample_kind}'."
                )
            else:
                assert len(requested_inputs) == len(expected_model_inputs)
                assert set(expected_inputs) == set(requested_inputs)
                assert len(requested_target_infos) == len(expected_inputs)
                assert set(expected_inputs) == set(requested_target_infos)

        if outputs_in_dataset:
            expected_out_targets = {
                out_name: all_target_infos[out_name] for out_name in expected_outputs
            }
            expected_model_outputs = {
                out_name: ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind=target_info.sample_kind,
                )
                for out_name, target_info in expected_out_targets.items()
            }

            # Check the ModelOutput objects that the hook supports as outputs.
            assert len(supported_outputs) == len(expected_model_outputs), (
                f"There are more supported outputs than expected. "
                f"Expected: {len(expected_model_outputs)}, "
                f"Got: {len(supported_outputs)}"
            )
            for out_name, model_output in supported_outputs.items():
                assert out_name in expected_model_outputs, (
                    f"Supported output '{out_name}' is not in the expected outputs."
                )
                expected_output = expected_model_outputs[out_name]
                assert model_output.quantity == expected_output.quantity, (
                    f"Supported output '{out_name}' has quantity "
                    f"'{model_output.quantity}', "
                    f"but expected '{expected_output.quantity}'."
                )
                assert model_output.unit == expected_output.unit, (
                    f"Supported output '{out_name}' has unit '{model_output.unit}', "
                    f"but expected '{expected_output.unit}'."
                )
                assert model_output.sample_kind == expected_output.sample_kind, (
                    f"Supported output '{out_name}' has sample kind "
                    f"'{model_output.sample_kind}', "
                    f"but expected '{expected_output.sample_kind}'."
                )
        else:
            assert len(supported_outputs) == len(expected_outputs), (
                f"The number of supported outputs does not match the expected number. "
                f"Expected: {len(expected_outputs)}, Got: {len(supported_outputs)}"
            )
            assert set(expected_outputs) == set(supported_outputs), (
                f"The supported outputs are not as expected. "
                f"Expected: {set(expected_outputs)}, Got: {set(supported_outputs)}"
            )
