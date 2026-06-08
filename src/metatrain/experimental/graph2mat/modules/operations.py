import torch

from e3nn import o3

import e3nn

from graph2mat.bindings.e3nn.modules._utils import tp_out_irreps_with_instructions
    
def find_common_neighbors(data):
    edges = data["edge_index"].T
    shifts = data["shifts"]

    i_edges = torch.arange(edges.shape[0])

    common_neighs = []
    
    for i_edge, (center, neigh) in enumerate(edges):
        # Get all messages that are coming to this center (from anyone)
        mask = (edges[:, 0] == center)
        center_edges = edges[mask]
        center_shifts = shifts[mask]
        center_iedges = i_edges[mask]
        this_edge_shift = shifts[i_edge]

        # Get all messages that are coming to this neighbor (from anyone)
        neigh_mask = (edges[:, 0] == neigh)
        neigh_edges = edges[neigh_mask]
        neighbor_iedges = i_edges[neigh_mask]
        neigh_shifts = shifts[neigh_mask]
        neigh_shifts = neigh_shifts + this_edge_shift

        # Find the messages that come to the center and neighbor from the
        # same third atom
        for i, center_edge, center_shift in zip(center_iedges, center_edges, center_shifts):
            for j, neigh_edge, neigh_shift in zip(neighbor_iedges, neigh_edges, neigh_shifts):
                if neigh_edge[1] == center_edge[1] and torch.all(center_shift == neigh_shift):
                    common_neighs.append([
                        i_edge,
                        i,
                        j
                    ])

    return torch.tensor(common_neighs)
    
class E3nnTwoCenterMessageBlock(torch.nn.Module):
    """This is basically MACE's RealAgnosticResidualInteractionBlock, but only up to the part
    where it computes the partial mji messages.

    It computes a "message" for each edge in the graph. Note that the message
    is different for the edge (i, j) and the edge (j, i).

    This function can be used for the preprocessing step of edges. It has no effect when used
    as the preprocessing step of nodes.
    """

    def __init__(
        self,
        irreps: dict[str, o3.Irreps],
    ) -> None:
        super().__init__()

        self.common_neighs = None
        self.common_neighs_data = None

        node_feats_irreps = irreps["node_feats_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        target_irreps = irreps["edge_hidden_irreps"]

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )

        self.two_center_edge_attrs = o3.FullyConnectedTensorProduct(
            edge_attrs_irreps,
            edge_attrs_irreps,
            edge_attrs_irreps,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
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
        self.conv_tp_weights = e3nn.nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()

        self.linear = o3.Linear(irreps_mid, target_irreps)

        self.irreps_out = (None, target_irreps)

    def forward(
        self,
        data, #: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        """
        This function implements E_ij = sum_k E_ik ⊗ E_jk
        for all k that are neighbors of both i and j.

        It takes into account periodic boundary conditions, managing
        the shifts.
        """
        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        if self.common_neighs is None or not torch.all(data["positions"] == self.common_neighs_data):
            self.common_neighs = find_common_neighbors(data)
            self.common_neighs_data = data["positions"].clone()

        common_neighs = self.common_neighs

        i_edge = common_neighs[:,0]
        first_center_edge = common_neighs[:,1]
        second_center_edge = common_neighs[:,2]
        neighs = data["edge_index"][1, first_center_edge]

        two_center_edge_feats = edge_feats[first_center_edge] * edge_feats[second_center_edge]
        two_center_edge_attrs = self.two_center_edge_attrs(
            edge_attrs[first_center_edge],
            edge_attrs[second_center_edge],
        )

        neigh_feats = torch.concat([
            node_feats[neighs],
            node_feats[data["edge_index"][0, :]],
        ], dim=0)

        i_edge = torch.concat([
            i_edge,
            torch.arange(data["edge_index"].shape[1], device=i_edge.device)
        ], dim=0)

        edge_attrs = torch.concat([
            two_center_edge_attrs,
            edge_attrs,
        ], dim=0)

        edge_feats = torch.concat([
            two_center_edge_feats,
            edge_feats,
        ], dim=0)

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        two_center_msgs = self.conv_tp(
            neigh_feats, edge_attrs, tp_weights
        )

        two_center_msgs = self.linear(two_center_msgs)

        # For one center messages, we aggregate the two directions
        # one_center_msgs = two_center_msgs[-data.edge_index.shape[1]:]
        # one_center_msgs = one_center_msgs.reshape(2, -1, one_center_msgs.shape[-1]).sum(dim=0)
        # two_center_msgs = torch.concat([
        #     two_center_msgs[:-data.edge_index.shape[1]],
        #     one_center_msgs.repeat(2,1),
        # ], dim=0)

        edge_values = torch.zeros((data["edge_index"].shape[1], two_center_msgs.shape[-1]), device=two_center_msgs.device)
        edge_values = edge_values.index_add(0, i_edge, two_center_msgs)

        return None, edge_values

class E3nnLinearEdge(torch.nn.Module):
    _data_get_edge_args = ("edge_feats",)

    def __init__(
        self,
        edge_feats_irreps: o3.Irreps,
        edge_messages_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        # self.linear = o3.Linear(
        #     edge_messages_irreps,
        #     irreps_out,
        # )

        self.linear = o3.TensorSquare(
            edge_messages_irreps,
            irreps_out=irreps_out,
        )

    def forward(
        self,
        edge_messages: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.linear(edge_messages[0])
    
class E3nnLinearNodeBlock(torch.nn.Module):
    """Sums all node features and then passes them to a linear layer."""

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (list, tuple)) and not isinstance(
            irreps_in, o3.Irreps
        ):
            assert all(
                irreps == irreps_in[0] for irreps in irreps_in
            ), "All input irreps must be the same."
            irreps_in = irreps_in[0]

        #self.tsq = o3.TensorSquare(irreps_in, irreps_out)
        self.linear = o3.Linear(irreps_in, irreps_out)

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        node_tensors = iter(node_kwargs.values())

        node_feats = next(node_tensors)
        for other_node_feats in node_tensors:
            node_feats = node_feats + other_node_feats

        #return self.tsq(node_feats)
        return self.linear(node_feats)
    
OPERATIONS_REGISTRY = {
    "node_operation": {
        "linear": E3nnLinearNodeBlock,
    },
    "edge_operation": {
        "linear": E3nnLinearEdge,
    },
    "preprocessing_edges": {
        "two_center_message": E3nnTwoCenterMessageBlock,
    },
    "preprocessing_nodes": {},
}