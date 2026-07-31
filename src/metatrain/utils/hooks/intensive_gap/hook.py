from typing import Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.hypers import init_with_defaults

from ..abc import HookInterface
from ..helpers import UnavailableOutputError
from .documentation import Hypers, PoolingHypers


# -----------------------------------------
#    Functions to do the pooling
# -----------------------------------------


def _scatter_softmax_pool(
    values: torch.Tensor,
    alpha: float,
    system_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Per-system self-weighted softmax pool: ``sum_i softmax(alpha * v_i)_i * v_i``.

    Numerically stable: shift ``alpha * v`` by per-system max before exponentiating.
    Strictly intensive (softmax weights sum to 1 within each system). The sign of
    ``alpha`` selects max- vs min-pool, exactly as in :func:`_scatter_logsumexp`.

    :param values: ``(N,)`` per-atom values.
    :param alpha: scalar tensor; sign determines max- vs min-pool.
    :param system_indices: ``(N,)`` system index per atom (in ``[0, num_systems)``).
    :param num_systems: number of systems ``S`` in the batch.
    :return: ``(S,)`` pooled values.
    """
    logits = alpha * values  # (N,)
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


def _scatter_logsumexp(
    values: torch.Tensor,
    alpha: float,
    system_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Numerically stable per-system ``(1/alpha) * logsumexp(alpha * values)``.

    Works for ``alpha`` of either sign. Implementation: shift by per-system max
    of ``alpha * values`` for stability, then scatter-add the exponentials.

    :param values: ``(N,)`` per-atom values.
    :param alpha: scalar tensor; sign determines max- vs min-pool.
    :param system_indices: ``(N,)`` system index per atom (in ``[0, num_systems)``).
    :param num_systems: number of systems ``S`` in the batch.
    :return: ``(S,)`` pooled values.
    """
    scaled = alpha * values  # (N,)
    neg_inf = torch.full(
        (num_systems,),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    sys_max = neg_inf.scatter_reduce(
        0, system_indices, scaled, reduce="amax", include_self=True
    )
    sys_max = torch.where(torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max)
    shifted_exp = torch.exp(scaled - sys_max[system_indices])
    sum_exp = torch.zeros(
        num_systems, dtype=values.dtype, device=values.device
    ).scatter_add(0, system_indices, shifted_exp)
    log_sum_exp = sys_max + torch.log(sum_exp)
    return log_sum_exp / alpha


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

        self.hypers = hypers

        pooling_hypers = hypers.get("pooling", init_with_defaults(PoolingHypers))

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
        input_names = hypers.get("inputs")
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
                self._input_target_infos[in_name] = TargetInfo(
                    per_atom_layout,
                    quantity=out_target.quantity,
                    unit=out_target.unit,
                )

            layout.set_info("pooling_type", pooling_hypers["type"])
            layout.set_info(
                "alpha_bottom", str(pooling_hypers.get("alpha_bottom", 20.0))
            )
            layout.set_info("alpha_top", str(pooling_hypers.get("alpha_top", -20.0)))
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

    def requested_inputs(self) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :return: A list of requested input names.
        """
        return {
            k: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind="atom",
            )
            for k, target_info in self._input_target_infos.items()
        }

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {
            out_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind="system",
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
            alpha_bottom = float(info["alpha_bottom"])
            alpha_top = float(info["alpha_top"])
            input_bottom = info["bottom_name"]
            input_top = info["top_name"]
            pooling_type = info["pooling_type"]
            block_shapes = self._block_shapes[out_name]

            # Build the output TensorMap by pooling over the per-atom values
            # for each block
            blocks: list[TensorBlock] = []
            for i, layout_block in enumerate(layout.blocks()):
                values_bottom = inputs[input_bottom].block(i).values.ravel()
                values_top = inputs[input_top].block(i).values.ravel()

                if pooling_type == "softmax":
                    # Self-weighted softmax pool: the softmax weights are computed
                    # directly from the per-atom values themselves, so atoms with the
                    # most extreme contribution dominate. Strictly intensive (weights
                    # sum to 1) and recovers a hard max/min as ``|alpha| -> infinity``.
                    out_bottom = _scatter_softmax_pool(
                        values_bottom, alpha_bottom, system_indices, num_systems
                    )
                    out_top = _scatter_softmax_pool(
                        values_top, alpha_top, system_indices, num_systems
                    )
                else:
                    out_bottom = _scatter_logsumexp(
                        values_bottom, alpha_bottom, system_indices, num_systems
                    )
                    out_top = _scatter_logsumexp(
                        values_top, alpha_top, system_indices, num_systems
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
