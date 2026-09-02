from typing import Optional

from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo

from ..abc import HookInterface
from .documentation import Hypers


class Identity(HookInterface[Hypers]):
    """
    Passes the inputs to the outputs without any modification.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        self.hypers = hypers

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            raise ValueError("Identity hook requires at least one output target")

        # Get the information for the input targets.
        input_names = hypers.get("inputs")
        if isinstance(input_names, str):
            input_names = [input_names]
        elif input_names is None:
            input_names = [
                f"mtt::aux::identity::{out_name.replace('mtt::', '')}"
                for out_name in out_names
            ]

        if len(input_names) != len(out_names):
            raise ValueError(
                f"Identity hook expects the same number of input and output "
                f"targets, but got {len(input_names)} inputs and "
                f"{len(out_names)} outputs"
            )

        # Build the target infos that the hook will request.
        self.out_targets = {}
        self._input_target_infos = {}
        targets = dataset_info.targets
        for in_name, out_name in zip(input_names, out_names, strict=True):
            if in_name in targets and out_name in targets:
                if targets[in_name] != targets[out_name]:
                    raise ValueError(
                        f"Identity hook found both the input '{in_name}' "
                        f"and the output '{out_name}' in the dataset targets, "
                        "but they have different layouts. To be able to apply "
                        "the identity hook, the input and output must have the "
                        "same layout."
                    )
                else:
                    target = targets[in_name]
            elif out_name not in targets and in_name not in targets:
                raise ValueError(
                    f"Identity hook expects the output '{out_name}' or the input "
                    f"'{in_name}' to be present in the dataset targets, but neither "
                    f"was found."
                )
            elif out_name in dataset_info.targets:
                target = dataset_info.targets[out_name]
            else:
                target = dataset_info.targets[in_name]

            self._input_target_infos[in_name] = target
            self.out_targets[out_name] = target

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_inputs(self, outputs: Optional[dict[str, ModelOutput]] = None) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
            are handled by other hooks or the main model. In that case, the hook
            ignores those outputs.
        :return: A list of requested input names.
        """
        if outputs is None:
            outputs_names = list(self.out_targets)
        else:
            outputs_names = list(outputs)

        req_inputs: dict[str, ModelOutput] = {}
        for out_name, (in_name, target_info) in zip(
            self.out_targets, self._input_target_infos.items(), strict=True
        ):
            if out_name in outputs_names:
                req_inputs[in_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind=target_info.sample_kind,
                )

        return req_inputs

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {
            out_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind=target_info.sample_kind,
            )
            for out_name, target_info in self.out_targets.items()
        }

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        return_dict: dict[str, TensorMap] = {}
        for in_name, out_name in zip(
            self._input_target_infos, self.out_targets, strict=True
        ):
            if out_name in outputs:
                return_dict[out_name] = inputs[in_name]

        return return_dict
