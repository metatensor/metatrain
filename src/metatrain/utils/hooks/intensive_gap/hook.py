from typing import Any, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo

from ..abc import HookInterface
from ..helpers import UnavailableOutputError
from .documentation import Hypers


# -----------------------------------------
#    Functions to do the pooling
# -----------------------------------------


def _scatter_softmax_pool(
    values: torch.Tensor,
    alphas: torch.Tensor,
    system_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Per-system self-weighted softmax pool: ``sum_i softmax(alpha_i * v_i)_i * v_i``.

    Strictly intensive (the softmax weights sum to 1 within each system). The sign
    of ``alpha`` selects max- vs min-pooling. With a per-atom ``alpha`` the weights
    change under a constant shift of the values. Numerically stable: the logits are
    shifted by their per-system max before exponentiating, which doesn't change the
    result.

    :param values: ``(N,)`` per-atom values.
    :param alphas: ``(N,)`` per-atom sharpness; the sign selects max- vs min-pool.
    :param system_indices: ``(N,)`` system index per atom (in ``[0, num_systems)``).
    :param num_systems: number of systems ``S`` in the batch.
    :return: ``(S,)`` pooled values.
    """
    logits = alphas * values  # (N,)
    neg_inf = torch.full(
        (num_systems,), float("-inf"), dtype=values.dtype, device=values.device
    )
    sys_max = neg_inf.scatter_reduce(
        0, system_indices, logits, reduce="amax", include_self=True
    )
    sys_max = torch.where(torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max)
    exps = torch.exp(logits - sys_max[system_indices])  # (N,)
    denom = torch.zeros(
        num_systems, dtype=values.dtype, device=values.device
    ).scatter_add(0, system_indices, exps)
    weights = exps / denom[system_indices]  # (N,) softmax across each system
    weighted = weights * values
    pooled = torch.zeros(
        num_systems, dtype=values.dtype, device=values.device
    ).scatter_add(0, system_indices, weighted)
    return pooled


# -----------------------------------------
#    The hook itself
# -----------------------------------------


class IntensiveGap(HookInterface[Hypers]):
    """Min/max pooling hook for intensive system-level outputs.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        sharpness_bottom = self.hypers["sharpness_bottom"]
        sharpness_top = self.hypers["sharpness_top"]

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        self.out_targets = {}
        for k in out_names:
            if k not in dataset_info.targets:
                raise UnavailableOutputError(
                    f"IntensiveGap hook requested output {k} but it is not "
                    f"available in the dataset info"
                )
            self.out_targets[k] = dataset_info.targets[k]

        # Get the information for the input targets.
        input_names = hypers["inputs"]
        if isinstance(input_names, dict):
            input_names = [input_names]
        if input_names is None:
            input_names = [{}] * len(out_names)
        input_names = [
            {
                "bottom": in_names.get(
                    "bottom", f"mtt::aux::gap_bottom::{out_name.replace('mtt::', '')}"
                ),
                "top": in_names.get(
                    "top", f"mtt::aux::gap_top::{out_name.replace('mtt::', '')}"
                ),
            }
            for in_names, out_name in zip(input_names, out_names, strict=True)
        ]

        assert len(input_names) == len(out_names), (
            f"IntensiveGap hook expects the same number of input and output "
            f"targets, but got {len(input_names)} inputs and {len(out_names)} outputs"
        )

        # Build the target infos that the hook will request.
        self._input_target_infos = {}
        self._block_shapes = {}
        for in_names, (out_name, out_target) in zip(
            input_names, self.out_targets.items(), strict=True
        ):
            if out_target.sample_kind != "system":
                raise ValueError(
                    f"IntensiveGap hook only supports system-level outputs, "
                    f"but {out_name} has sample kind "
                    f"{out_target.sample_kind}"
                )
            if not out_target.is_scalar:
                raise ValueError(
                    f"IntensiveGap hook only supports scalar outputs, "
                    f"but {out_name} has components "
                    f"{out_target.layout.components}"
                )

            layout = out_target.layout
            per_atom_layout = TensorMap(
                keys=layout.keys,
                blocks=[
                    TensorBlock(
                        values=block.values,
                        samples=Labels(
                            ["system", "atom"], block.samples.values.reshape((0, 2))
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in layout.blocks()
                ],
            )

            for in_name in [in_names["bottom"], in_names["top"]]:
                built_target_info = TargetInfo(
                    per_atom_layout,
                    quantity=out_target.quantity,
                    unit=out_target.unit,
                )
                if (
                    in_name in dataset_info.targets
                    or in_name in dataset_info.extra_data
                ):
                    # The input is present in the dataset info, check if the target
                    # that we built is compatible with the one in the dataset info.
                    # If it is, use it as our input
                    target_info = (
                        dataset_info.targets.get(in_name)
                        or dataset_info.extra_data[in_name]
                    )

                    try:
                        _check_compatible_target_infos(target_info, built_target_info)
                    except AssertionError:
                        raise ValueError(
                            f"IntensiveGap hook requested input {in_name} but we found"
                            f" '{in_name}' in the dataset and it is not compatible as"
                            f" an input for the hook. "
                        )

                    self._input_target_infos[in_name] = target_info
                else:
                    self._input_target_infos[in_name] = built_target_info

            layout.set_info("sharpness_bottom", str(sharpness_bottom))
            layout.set_info("sharpness_top", str(sharpness_top))
            layout.set_info("bottom_name", in_names["bottom"])
            layout.set_info("top_name", in_names["top"])

            self._block_shapes[out_name] = [
                (-1, *block.values.shape[1:]) for block in out_target.layout.blocks()
            ]

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_hook_inputs(
        self, outputs: dict[str, ModelOutput]
    ) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
            are handled by other hooks or the main model. In that case, the hook
            ignores those outputs.
        :return: A list of requested input names.
        """
        req_inputs: dict[str, ModelOutput] = {}
        for out_name, out_info in self.out_targets.items():
            if out_name in outputs:
                info = out_info.layout.info()
                bottom_name = info["bottom_name"]
                top_name = info["top_name"]

                req_inputs[bottom_name] = ModelOutput(
                    quantity=self._input_target_infos[bottom_name].quantity,
                    unit=self._input_target_infos[bottom_name].unit,
                    sample_kind="atom",
                )

                req_inputs[top_name] = ModelOutput(
                    quantity=self._input_target_infos[top_name].quantity,
                    unit=self._input_target_infos[top_name].unit,
                    sample_kind="atom",
                )

        return req_inputs

    def supported_outputs(self) -> dict[str, TargetInfo]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return self.out_targets

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        requested_outs: list[str] = []
        for out_name in self.out_targets:
            if out_name in outputs:
                requested_outs.append(out_name)
        if not requested_outs:
            # Quick exit if no output of this hook is requested
            return {}

        device = systems[0].positions.device

        num_systems = len(systems)
        system_indices = []
        for i, system in enumerate(systems):
            system_indices.append(
                torch.full((len(system),), i, dtype=torch.int32, device=device)
            )
        system_indices = torch.cat(system_indices, dim=0)

        return_dict: dict[str, TensorMap] = {}
        for out_name in requested_outs:
            out_target = self.out_targets[out_name]
            layout = out_target.layout.to(device)

            # Get the parameters for this target
            info = layout.info()
            sharpness_bottom = float(info["sharpness_bottom"])
            sharpness_top = float(info["sharpness_top"])
            input_bottom = info["bottom_name"]
            input_top = info["top_name"]
            block_shapes = self._block_shapes[out_name]

            # Build the output TensorMap by pooling over the per-atom values
            # for each block
            blocks: list[TensorBlock] = []
            for i, layout_block in enumerate(layout.blocks()):
                values_bottom = inputs[input_bottom].block(i).values.ravel()
                values_top = inputs[input_top].block(i).values.ravel()

                alphas_bottom = torch.full_like(values_bottom, sharpness_bottom)
                alphas_top = torch.full_like(values_top, sharpness_top)

                out_bottom = _scatter_softmax_pool(
                    values_bottom, alphas_bottom, system_indices, num_systems
                )
                out_top = _scatter_softmax_pool(
                    values_top, alphas_top, system_indices, num_systems
                )

                gap = out_top - out_bottom

                blocks.append(
                    TensorBlock(
                        values=gap.reshape(block_shapes[i]),
                        samples=Labels(
                            names=["system"],
                            values=torch.arange(
                                num_systems, dtype=torch.int32, device=device
                            ).reshape(-1, 1),
                        ),
                        components=layout_block.components,
                        properties=layout_block.properties,
                    )
                )

            return_dict[out_name] = TensorMap(
                keys=layout.keys,
                blocks=blocks,
            )

        return return_dict

    def get_checkpoint(self, best_state_dict: Optional[dict[str, Any]] = None) -> dict:
        """
        Get the checkpoint of the hook.

        :return: A dictionary containing the state of the hook.
        """
        state_dict = self.state_dict()
        return {
            "hook_name": "intensive_gap",
            "hook_ckpt_version": self.__checkpoint_version__,
            "hook_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": state_dict,
        }

    @classmethod
    def load_checkpoint(
        cls, checkpoint: dict[str, Any], context: str
    ) -> "IntensiveGap":
        """
        Load the checkpoint of the hook.

        :param checkpoint: A dictionary containing the state of the hook.
        """
        if checkpoint["hook_ckpt_version"] != cls.__checkpoint_version__:
            raise ValueError(
                f"Invalid checkpoint version for the hook: "
                f"expected {cls.__checkpoint_version__}, got "
                f"{checkpoint['hook_ckpt_version']}"
            )

        hook = cls(
            hypers=checkpoint["hook_data"]["hypers"],
            dataset_info=checkpoint["hook_data"]["dataset_info"],
        )

        hook.load_state_dict(checkpoint["model_state_dict"])
        return hook


def _check_compatible_target_infos(
    target_info1: TargetInfo, target_info2: TargetInfo
) -> None:
    """Check that two target infos have compatible layouts.

    :param target_info1: The first target info.
    :param target_info2: The second target info.
    :raises ValueError: If the layouts are not compatible.
    """

    assert target_info1.sample_kind == target_info2.sample_kind

    layout1 = target_info1.layout
    layout2 = target_info2.layout

    for (key1, block1), (_, block2) in zip(
        layout1.items(), layout2.items(), strict=True
    ):
        assert block1.samples.values.shape == block2.samples.values.shape, (
            f"Sample shapes do not match for key {key1}: "
            f"{block1.samples.values.shape} vs {block2.samples.values.shape}"
        )
        assert block1.components == block2.components

        assert block1.properties.values.shape == block2.properties.values.shape, (
            f"Property shapes do not match for key {key1}: "
            f"{block1.properties.values.shape} vs {block2.properties.values.shape}"
        )
