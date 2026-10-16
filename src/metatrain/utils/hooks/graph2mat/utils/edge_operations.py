import torch
from e3nn import o3, nn

# Taken directly from the MACE repository (mace.modules.irreps_tools).
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> tuple[o3.Irreps, list]:
    """"""
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: list[tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


class MACEEdgeState(torch.nn.Module):
    """This is basically MACE's RealAgnosticResidualInteractionBlock, but only up to the part
    where it computes the partial mji messages.

    It computes a "message" for each edge in the graph. Note that the message
    is different for the edge (i, j) and the edge (j, i).
    """

    def __init__(
        self,
        irreps: dict[str, o3.Irreps],
    ) -> None:
        super().__init__()

        node_feats_irreps = irreps["node_feats_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        message_irreps = irreps["message_irreps"]
        edge_targets_irreps = irreps["edge_targets_irreps"]

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            message_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        assert (
            edge_feats_irreps.lmax == 0
        ), "Edge features must be a scalar array to preserve equivariance"
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()

        self.linear = o3.Linear(irreps_mid, message_irreps)

        self.message_receiver_tp = o3.FullyConnectedTensorProduct(
            message_irreps,
            node_feats_irreps,
            edge_targets_irreps,
        )



    def forward(
        self,
        data: dict[str, torch.Tensor],
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        sender = data["edge_index"][0]
        receiver = data["edge_index"][1]

        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        mji = self.linear(mji)

        value_ji = self.message_receiver_tp(
            mji, node_feats[receiver]
        )

        return value_ji

        

