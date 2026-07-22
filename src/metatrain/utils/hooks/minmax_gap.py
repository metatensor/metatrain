import torch
from metatensor.torch import TensorBlock, TensorMap, Labels
from metatomic.torch import ModelOutput, System
from typing_extensions import TypedDict, NotRequired
from typing import Literal

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.hypers import init_with_defaults

# -----------------------------------------
#                Hypers
# -----------------------------------------

class PoolingHypers(TypedDict):
    """Hyperparameters for the per-system pooling step.

    Two pooling types are supported, both controlled by the same ``alpha_max``
    / ``alpha_min`` parameters (sign selects max- vs min-pool, magnitude sets
    the sharpness):

    - ``"smoothmax"`` (default): ``E = (1/alpha) log sum_i exp(alpha h_i)``.
      Recovers a hard max/min as ``|alpha| -> infinity``. Size-intensive up to a
      ``log(N)/|alpha|`` residual.
    - ``"softmax"``: ``E = sum_i softmax(alpha * h_i) * h_i``. A self-weighted
      softmax pool: the softmax weights are computed from the per-atom *values
      themselves*, so the pool attends to the most extreme contributions.
      Strictly intensive (softmax weights sum to 1, removing the
      ``log(N)/|alpha|`` residual of the smoothmax pool) and recovers a hard
      max/min as ``|alpha| -> infinity``.
    """

    type: Literal["smoothmax", "softmax"] = "smoothmax"
    """Pooling type. One of ``"smoothmax"`` or ``"softmax"``."""

    alpha_bottom: float = 20.0
    """Max pooling parameter. ``alpha_bottom > 0`` gives a (smooth/soft) max.
    Larger magnitude -> sharper (closer to a hard max). Used by both pooling
    types."""

    alpha_top: float = -20.0
    """Min pooling parameter. ``alpha_top < 0`` gives a (smooth/soft) min.
    Larger magnitude -> sharper (closer to a hard min). Used by both pooling
    types."""

class HookInputs(TypedDict):
    """Inputs for the minmax hook."""
    bottom: NotRequired[str]
    """Name of the target for the bottom of the gap."""
    top: NotRequired[str]
    """Name of the target for the top of the gap."""

class HookHypers(TypedDict):
    """
    Hyperparameters for the global multipole hook.
    """
    pooling: PoolingHypers = init_with_defaults(PoolingHypers)

    inputs: HookInputs = init_with_defaults(HookInputs)

    outputs: str


# -----------------------------------------
#    Functions to do the pooling
# -----------------------------------------

def _scatter_softmax_pool(
    values: torch.Tensor,
    alpha: torch.Tensor,
    system_indices: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Per-system self-weighted softmax pool: ``sum_i softmax(alpha * v_i)_i * v_i``.

    Numerically stable: shift ``alpha * v`` by per-system max before exponentiating.
    Strictly intensive (softmax weights sum to 1 within each system). The sign of
    ``alpha`` selects max- vs min-pool, exactly as in :func:`_scatter_logsumexp`.
    """
    logits = alpha * values  # (N,)
    neg_inf = torch.full(
        (num_systems,), float("-inf"), dtype=values.dtype, device=values.device
    )
    sys_max = neg_inf.scatter_reduce(
        0, system_indices, logits, reduce="amax", include_self=True
    )
    sys_max = torch.where(
        torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max
    )
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
    alpha: torch.Tensor,
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
    sys_max = torch.where(
        torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max
    )
    shifted_exp = torch.exp(scaled - sys_max[system_indices])
    sum_exp = torch.zeros(
        num_systems, dtype=values.dtype, device=values.device
    ).scatter_add(0, system_indices, shifted_exp)
    log_sum_exp = sys_max + torch.log(sum_exp)
    return log_sum_exp / alpha

# -----------------------------------------
#    The hook itself
# -----------------------------------------


class MinMaxGap(torch.nn.Module):
    """
    Does minimum and maximum pooling to get an intensive property.
    """

    def __init__(self, hypers: HookHypers, dataset_info: DatasetInfo):
        super().__init__()

        self.hypers = hypers

        pooling_hypers = hypers.get("pooling", init_with_defaults(PoolingHypers))

        self._pooling_type = pooling_hypers["type"]

        # Get the information about the output target from the dataset info
        self.out_name = hypers["outputs"]
        self.out_target = dataset_info.targets[self.out_name]

        # Build the input target that we will request from the model,
        # which is the local multipoles
        inputs = hypers.get("inputs", init_with_defaults(HookInputs))
        self._input_names = [
            inputs.get("bottom", "mtt::aux::gap_bottom"),
            inputs.get("top", "mtt::aux::gap_top"),
        ]

        if self.out_target.sample_kind != "system":
            raise ValueError(
                f"MinMaxGap hook only supports system-level outputs, "
                f"but {self.out_name} has sample kind "
                f"{self.out_target.sample_kind}"
            )
        if not self.out_target.is_scalar:
            raise ValueError(
                f"MinMaxGap hook only supports scalar outputs, "
                f"but {self.out_name} has components "
                f"{self.out_target.layout.components}"
            )
        
        layout = self.out_target.layout
        per_atom_layout = TensorMap(
            keys=layout.keys,
            blocks=[
                TensorBlock(
                    values=block.values,
                    samples=Labels(["system", "atom"], block.samples.values.reshape((0,2))),
                    components=block.components,
                    properties=block.properties, 
                )
                for block in layout.blocks()
            ]
        )

        self._input_target_infos = {
            input_name: TargetInfo(per_atom_layout)
            for input_name in self._input_names
        }
        self._blocks_shape = [
            (-1, *block.values.shape[1:]) for block in self.out_target.layout.blocks()
        ]

        self.register_buffer(
            "alphas",
            torch.tensor([
                pooling_hypers.get("alpha_bottom", 20.0),
                pooling_hypers.get("alpha_top", -20.0)
            ], dtype=torch.float32),
            persistent=False,
        )

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
                quantity="",
                unit="",
                sample_kind="atom",
            ) for k in self._input_names
        }

    def forward(
        self, systems: list[System], inputs: dict[str, TensorMap]
    ) -> dict[str, TensorMap]:
        """
        Computes the global multipole from the local predictions.
        """
        device = inputs[self._input_names[0]].block(0).values.device
        layout = self.out_target.layout.to(device)

        num_systems = len(systems)
        system_indices = []
        for i, system in enumerate(systems):
            system_indices.append(
                torch.full((len(system),), i, dtype=torch.int32, device=device)
            )
        system_indices = torch.cat(system_indices, dim=0)

        blocks: list[TensorBlock]= []
        for i, layout_block in enumerate(layout.blocks()):

            values_bottom = inputs[self._input_names[0]].block(i).values.ravel()
            values_top = inputs[self._input_names[1]].block(i).values.ravel()
        
            if self._pooling_type == "softmax":
                # Self-weighted softmax pool: the softmax weights are computed
                # directly from the per-atom values themselves, so atoms with the
                # most extreme contribution dominate. Strictly intensive (weights
                # sum to 1) and recovers a hard max/min as ``|alpha| -> infinity``.
                out_bottom = _scatter_softmax_pool(
                    values_bottom, self.alphas[0], system_indices, num_systems
                )
                out_top = _scatter_softmax_pool(
                    values_top, self.alphas[1], system_indices, num_systems
                )
            else:
                out_bottom = _scatter_logsumexp(
                    values_bottom, self.alphas[0], system_indices, num_systems
                )
                out_top = _scatter_logsumexp(
                    values_top, self.alphas[1], system_indices, num_systems
                )

            gap = out_top - out_bottom
            
            blocks.append(
                TensorBlock(
                    values=gap.reshape(self._blocks_shape[i]),
                    samples=Labels(
                        names=["system"],
                        values=torch.arange(num_systems, dtype=torch.int32, device=device).reshape(-1, 1),
                    ),
                    components=layout.components,
                    properties=layout.properties,
                )
            )

        output_tmap = TensorMap(
            keys=layout.keys,
            blocks=blocks,
        )
            
        # Return the global multipole by summing over the atoms in the system
        return {self.out_name: output_tmap}
