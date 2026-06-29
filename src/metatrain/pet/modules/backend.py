from math import prod
from typing import Dict, List, Tuple

import torch

from ..documentation import ModelHypers
from .structures import compute_batch_tensors
from .transformer import CartesianTransformer


class PETBackend(torch.nn.Module):
    """
    Pure-PyTorch backend of the PET architecture.

    This module contains the structure-preprocessing, featurization and prediction
    steps of PET, operating purely on plain :class:`torch.Tensor` objects. It does not
    touch any metatensor / metatomic objects (``System``, ``Labels``, ``TensorMap``),
    so that it can be ``torch.compile``-d. The surrounding
    :class:`metatrain.pet.model.PET` module is responsible for extracting the input
    tensors from a list of ``System`` objects and for wrapping the returned predictions
    back into ``TensorMap`` objects.

    The learnable submodules of PET (GNN layers, embedders, heads, last layers, ...)
    are *owned* by this module. The per-output heads and last layers are populated
    lazily by :meth:`metatrain.pet.model.PET._add_output`.

    :param hypers: Hyperparameters for the PET model. See the documentation for details.
    :param atomic_types: Sorted list of atomic types the model supports.
    """

    NUM_FEATURE_TYPES: int = 2  # node + edge features

    def __init__(self, hypers: ModelHypers, atomic_types: List[int]) -> None:
        super().__init__()

        # Cache frequently accessed hyperparameters
        self.cutoff = float(hypers["cutoff"])
        self.cutoff_function = hypers["cutoff_function"]
        self.cutoff_width = float(hypers["cutoff_width"])
        self.num_neighbors_adaptive = (
            float(hypers["num_neighbors_adaptive"])
            if hypers["num_neighbors_adaptive"] is not None
            else None
        )
        self.adaptive_cutoff_method = hypers["adaptive_cutoff_method"]
        self.d_pet = hypers["d_pet"]
        self.d_node = hypers["d_node"]
        self.d_head = hypers["d_head"]
        self.d_feedforward = hypers["d_feedforward"]
        self.num_heads = hypers["num_heads"]
        self.num_gnn_layers = hypers["num_gnn_layers"]
        self.num_attention_layers = hypers["num_attention_layers"]
        self.normalization = hypers["normalization"]
        self.activation = hypers["activation"]
        self.attention_temperature = hypers["attention_temperature"]
        self.transformer_type = hypers["transformer_type"]
        self.featurizer_type = hypers["featurizer_type"]

        num_atomic_species = len(atomic_types)

        # ``species_to_species_index`` is registered first so that it remains the first
        # entry of the ``state_dict`` (an integer buffer), which the checkpoint dtype
        # probe in ``PET.load_checkpoint`` relies on.
        self.register_buffer(
            "species_to_species_index",
            torch.full((max(atomic_types) + 1,), -1),
        )
        for i, species in enumerate(atomic_types):
            self.species_to_species_index[species] = i

        self.gnn_layers = torch.nn.ModuleList(
            [
                CartesianTransformer(
                    self.cutoff,
                    self.cutoff_width,
                    self.d_pet,
                    self.num_heads,
                    self.d_node,
                    self.d_feedforward,
                    self.num_attention_layers,
                    self.normalization,
                    self.activation,
                    self.attention_temperature,
                    self.transformer_type,
                    num_atomic_species,
                    layer_index == 0,  # is first layer
                )
                for layer_index in range(self.num_gnn_layers)
            ]
        )
        if self.featurizer_type == "feedforward":
            self.num_readout_layers = 1
            self.combination_norms = torch.nn.ModuleList(
                [torch.nn.LayerNorm(2 * self.d_pet) for _ in range(self.num_gnn_layers)]
            )
            self.combination_mlps = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(2 * self.d_pet, 2 * self.d_pet),
                        torch.nn.SiLU(),
                        torch.nn.Linear(2 * self.d_pet, self.d_pet),
                    )
                    for _ in range(self.num_gnn_layers)
                ]
            )
        else:
            self.num_readout_layers = self.num_gnn_layers
            self.combination_norms = torch.nn.ModuleList()
            self.combination_mlps = torch.nn.ModuleList()

        self.node_embedders = torch.nn.ModuleList(
            [
                torch.nn.Embedding(num_atomic_species, self.d_node)
                for _ in range(self.num_readout_layers)
            ]
        )
        self.edge_embedder = torch.nn.Embedding(num_atomic_species, self.d_pet)

        # Per-output heads and last layers, populated by ``PET._add_output``.
        self.node_heads = torch.nn.ModuleDict()
        self.edge_heads = torch.nn.ModuleDict()
        self.node_last_layers = torch.nn.ModuleDict()
        self.edge_last_layers = torch.nn.ModuleDict()

        # ===== BEGIN DIAGNOSTIC-RELATED ATTRIBUTES
        # These are used to capture the node and edge features from each GNN layer post
        # message passing, for diagnostic purposes.
        self.gnn_layers_post_mp_node = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(self.num_gnn_layers)]
        )
        self.gnn_layers_post_mp_edge = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(self.num_gnn_layers)]
        )
        # These are used to capture the raw backbone features before they are processed
        # by the featurizer heads, for diagnostic purposes.
        self.node_backbone = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(self.num_readout_layers)]
        )
        self.edge_backbone = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(self.num_readout_layers)]
        )
        # ===== END DIAGNOSTIC-RELATED ATTRIBUTES

    def add_output(self, target_name: str, output_shapes: Dict[str, List[int]]) -> None:
        """
        Create the node/edge heads and last layers for a new output target.

        This is the pure-PyTorch part of registering an output: it builds the learnable
        modules from the plain per-block output shapes. The metatensor-side bookkeeping
        (``ModelOutput``, key / component / property labels) stays on
        :class:`metatrain.pet.model.PET`.

        :param target_name: Name of the target to add.
        :param output_shapes: Mapping from per-block key to the block's shape (the
            component sizes followed by the number of properties), as computed by
            :meth:`metatrain.pet.model.PET._add_output`.
        """
        self.node_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.d_node, self.d_head),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.d_head, self.d_head),
                    torch.nn.SiLU(),
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.edge_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.d_pet, self.d_head),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.d_head, self.d_head),
                    torch.nn.SiLU(),
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.node_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(self.d_head, prod(shape), bias=True)
                        for key, shape in output_shapes.items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.edge_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(self.d_head, prod(shape), bias=True)
                        for key, shape in output_shapes.items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )

    def preprocess(
        self,
        positions: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        cells: torch.Tensor,
        cell_shifts: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run structure preprocessing on plain tensors.

        This computes edge vectors, optional adaptive cutoffs, cutoff factors and the
        NEF reshaping, returning the per-edge batch tensors consumed by
        :meth:`compute_features` and the wrapping logic in :class:`PET`. It is kept
        separate from :meth:`compute_features` so that diagnostic hooks (which need the
        post-cutoff ``centers`` / ``nef_to_edges_neighbor``) can be registered in
        between.

        :param positions: Concatenated atomic positions, shape ``(num_nodes, 3)``.
        :param centers: Flat center atom global indices, shape ``(n_edges,)``.
        :param neighbors: Flat neighbor atom global indices, shape ``(n_edges,)``.
        :param species: Concatenated atomic species, shape ``(num_nodes,)``.
        :param cells: Stacked cell tensors, shape ``(num_systems, 3, 3)``.
        :param cell_shifts: Integer cell shift vectors, shape ``(n_edges, 3)``.
        :param system_indices: System index for each atom, shape ``(num_nodes,)``.
        :return: A dictionary ``aux`` of the intermediate tensors. The keys in
            :data:`metatrain.pet.modules.diagnostic.FEATURIZER_INPUT_NAMES` are the
            inputs to :meth:`compute_features`; the remaining keys
            (``atomic_cutoffs_stats``, ``centers``, ``neighbors``,
            ``nef_to_edges_neighbor``, ``cell_shifts``) are used by :class:`PET` to
            build the output ``TensorMap`` objects.
        """
        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
            atomic_cutoffs_stats,
            centers,
            neighbors,
            nef_to_edges_neighbor,
            cell_shifts,
        ) = compute_batch_tensors(
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            self.species_to_species_index,
            self.cutoff,
            self.cutoff_function,
            self.cutoff_width,
            self.num_neighbors_adaptive,
            self.adaptive_cutoff_method,
        )

        batch_data: Dict[str, torch.Tensor] = {
            "element_indices_nodes": element_indices_nodes,
            "element_indices_neighbors": element_indices_neighbors,
            "edge_vectors": edge_vectors,
            "edge_distances": edge_distances,
            "reverse_neighbor_index": reverse_neighbor_index,
            "padding_mask": padding_mask,
            "cutoff_factors": cutoff_factors,
            "atomic_cutoffs_stats": atomic_cutoffs_stats,
            "centers": centers,
            "neighbors": neighbors,
            "nef_to_edges_neighbor": nef_to_edges_neighbor,
            "cell_shifts": cell_shifts,
        }
        return batch_data

    def predict(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        batch_data: Dict[str, torch.Tensor],
        cells: torch.Tensor,
        system_indices: torch.Tensor,
        requested_output_names: List[str],
    ) -> Tuple[
        Dict[str, List[torch.Tensor]],
        Dict[str, List[torch.Tensor]],
        Dict[str, List[torch.Tensor]],
    ]:
        """
        Compute the per-block atomic predictions and last-layer features.

        :param node_features_list: Per-layer node features (possibly modified by the
            long-range featurizer).
        :param edge_features_list: Per-layer edge features.
        :param batch_data: Dictionary containing input tensors required for feature
            calculation.
        :param cells: Stacked cell tensors, shape ``(num_systems, 3, 3)``, used to
            normalize non-conservative stress predictions by cell volume.
        :param system_indices: System index for each atom, shape ``(num_nodes,)``.
        :param requested_output_names: Names of the target outputs to compute.
        :return: A tuple ``(atomic_predictions, node_ll_features, edge_ll_features)``
            where ``atomic_predictions`` maps each requested output to a list of
            per-block flat prediction tensors, and the last-layer feature dictionaries
            map each output to its per-layer node / edge last-layer features.
        """
        padding_mask = batch_data["padding_mask"]
        cutoff_factors = batch_data["cutoff_factors"]

        node_ll_features, edge_ll_features = self._calculate_last_layer_features(
            node_features_list,
            edge_features_list,
        )

        node_atomic_predictions_dict, edge_atomic_predictions_dict = (
            self._calculate_atomic_predictions(
                node_ll_features,
                edge_ll_features,
                padding_mask,
                cutoff_factors,
                requested_output_names,
            )
        )

        # Sum the node and edge contributions over all GNN layers, block by block.
        atomic_predictions: Dict[str, List[torch.Tensor]] = {}
        for output_name in node_atomic_predictions_dict.keys():
            node_by_layer = node_atomic_predictions_dict[output_name]
            edge_by_layer = edge_atomic_predictions_dict[output_name]
            num_blocks = len(node_by_layer[0])
            block_sums: List[torch.Tensor] = []
            for b in range(num_blocks):
                block_sum = node_by_layer[0][b] + edge_by_layer[0][b]
                for layer in range(1, len(node_by_layer)):
                    block_sum = (
                        block_sum + node_by_layer[layer][b] + edge_by_layer[layer][b]
                    )
                block_sums.append(block_sum)

            if output_name == "non_conservative_stress":  # TODO: variants
                num_properties = block_sums[0].shape[1] // 9
                block_sums[0] = process_non_conservative_stress(
                    block_sums[0],
                    cells,
                    system_indices,
                    num_properties,
                )

            atomic_predictions[output_name] = block_sums

        return atomic_predictions, node_ll_features, edge_ll_features

    def calculate_features(
        self,
        batch_data: Dict[str, torch.Tensor],
        capture_diagnostics: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Calculate node and edge features using the selected featurization strategy.
        Returns lists of feature tensors from GNN layers.

        :param batch_data: Dictionary containing input tensors required for feature
            computation
        :param capture_diagnostics: Whether to capture diagnostic features via temporary
            module hooks. This is only used when diagnostic outputs are requested, and
            it is skipped in TorchScript / tracing mode where hooks are not supported.
        :return: Tuple of two lists:
            - List of node feature tensors
            - List of edge feature tensors
            In the case of feedforward featurization, each list contains a single tensor
            from the final GNN layer. In the case of residual featurization, each list
            contains tensors from all GNN layers.
        """

        featurizer_inputs: Dict[str, torch.Tensor] = {
            "element_indices_nodes": batch_data["element_indices_nodes"],
            "element_indices_neighbors": batch_data["element_indices_neighbors"],
            "edge_vectors": batch_data["edge_vectors"],
            "edge_distances": batch_data["edge_distances"],
            "reverse_neighbor_index": batch_data["reverse_neighbor_index"],
            "padding_mask": batch_data["padding_mask"],
            "cutoff_factors": batch_data["cutoff_factors"],
        }

        # the scaled_dot_product_attention function from torch cannot do
        # double backward, so we will use manual attention if needed
        use_manual_attention = (
            batch_data["edge_vectors"].requires_grad and self.training
        )

        if self.featurizer_type == "feedforward":
            node_features_list, edge_features_list = (
                self._feedforward_featurization_impl(
                    featurizer_inputs, use_manual_attention
                )
            )
        else:
            node_features_list, edge_features_list = self._residual_featurization_impl(
                featurizer_inputs, use_manual_attention
            )

        # ===== BEGIN DIAGNOSTIC-RELATED BLOCK
        # Pass the raw node and edge backbone features through identity modules so
        # that diagnostic hooks on ``node_backbone[i]`` / ``edge_backbone[i]`` fire.
        # Skipped when no diagnostic output was requested, and always skipped in
        # TorchScript / tracing mode where hooks are never registered.
        if (
            capture_diagnostics
            and (not torch.jit.is_scripting())
            and (not torch.jit.is_tracing())
        ):
            new_node_features: List[torch.Tensor] = []
            for i in range(len(node_features_list)):
                new_node_features.append(self.node_backbone[i](node_features_list[i]))
            node_features_list = new_node_features

            new_edge_features: List[torch.Tensor] = []
            for i in range(len(edge_features_list)):
                new_edge_features.append(self.edge_backbone[i](edge_features_list[i]))
            edge_features_list = new_edge_features
        # ===== END DIAGNOSTIC-RELATED BLOCK

        return node_features_list, edge_features_list

    def _feedforward_featurization_impl(
        self, inputs: Dict[str, torch.Tensor], use_manual_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Feedforward featurization: iterates features through all GNN layers,
        returning only the final layer outputs. Uses combination MLPs to mix
        forward and reversed edge messages at each layer.

        :param inputs: Dictionary containing input tensors required for feature
            computation
        :param use_manual_attention: Whether to use manual attention computation
            (required for double backward when edge vectors require gradients)
        :return: Tuple of two lists:
            - List of node feature tensors from the final GNN layer
            - List of edge feature tensors from the final GNN layer
        """
        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []

        input_node_embeddings = self.node_embedders[0](inputs["element_indices_nodes"])
        input_edge_embeddings = self.edge_embedder(inputs["element_indices_neighbors"])
        for (
            combination_norm,
            combination_mlp,
            gnn_layer,
            gnn_layer_post_mp_node,
            gnn_layer_post_mp_edge,
        ) in zip(
            self.combination_norms,
            self.combination_mlps,
            self.gnn_layers,
            self.gnn_layers_post_mp_node,
            self.gnn_layers_post_mp_edge,
            strict=True,
        ):
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                inputs["element_indices_neighbors"],
                inputs["edge_vectors"],
                inputs["padding_mask"],
                inputs["edge_distances"],
                inputs["cutoff_factors"],
                use_manual_attention,
            )

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            input_node_embeddings = output_node_embeddings
            new_input_edge_embeddings = output_edge_embeddings.reshape(
                output_edge_embeddings.shape[0] * output_edge_embeddings.shape[1],
                output_edge_embeddings.shape[2],
            )[inputs["reverse_neighbor_index"]].reshape(
                output_edge_embeddings.shape[0],
                output_edge_embeddings.shape[1],
                output_edge_embeddings.shape[2],
            )
            # input_messages = 0.5 * (output_edge_embeddings + new_input_messages)
            concatenated = torch.cat(
                [output_edge_embeddings, new_input_edge_embeddings], dim=-1
            )
            input_edge_embeddings = (
                input_edge_embeddings
                + output_edge_embeddings
                + combination_mlp(combination_norm(concatenated))
            )

            # ===== BEGIN DIAGNOSTIC-RELATED ATTRIBUTES
            # Capture the node and edge features from this GNN layer post message
            # passing
            input_node_embeddings = gnn_layer_post_mp_node(input_node_embeddings)
            input_edge_embeddings = gnn_layer_post_mp_edge(input_edge_embeddings)

            # ===== END DIAGNOSTIC-RELATED ATTRIBUTES

        node_features_list.append(input_node_embeddings)
        edge_features_list.append(input_edge_embeddings)
        return node_features_list, edge_features_list

    def _residual_featurization_impl(
        self, inputs: Dict[str, torch.Tensor], use_manual_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Residual featurization: saves intermediate features from each GNN layer
        for use in readout. Averages forward and reversed edge messages between layers.

        :param inputs: Dictionary containing input tensors required for feature
            computation
        :param use_manual_attention: Whether to use manual attention computation
            (required for double backward when edge vectors require gradients)
        :return: Tuple of two lists:
            - List of node feature tensors from all GNN layers
            - List of edge feature tensors from all GNN layers
        """
        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []
        input_edge_embeddings = self.edge_embedder(inputs["element_indices_neighbors"])
        for node_embedder, gnn_layer in zip(
            self.node_embedders, self.gnn_layers, strict=True
        ):
            input_node_embeddings = node_embedder(inputs["element_indices_nodes"])
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                inputs["element_indices_neighbors"],
                inputs["edge_vectors"],
                inputs["padding_mask"],
                inputs["edge_distances"],
                inputs["cutoff_factors"],
                use_manual_attention,
            )
            node_features_list.append(output_node_embeddings)
            edge_features_list.append(output_edge_embeddings)

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            # (Flatten, index, and reshape to the original shape)
            new_input_messages = output_edge_embeddings.reshape(
                output_edge_embeddings.shape[0] * output_edge_embeddings.shape[1],
                output_edge_embeddings.shape[2],
            )[inputs["reverse_neighbor_index"]].reshape(
                output_edge_embeddings.shape[0],
                output_edge_embeddings.shape[1],
                output_edge_embeddings.shape[2],
            )
            input_edge_embeddings = 0.5 * (input_edge_embeddings + new_input_messages)
        return node_features_list, edge_features_list

    def _calculate_last_layer_features(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """
        Apply output-specific heads to node and edge features from each GNN layer.
        Returns dictionaries mapping output names to lists of head-transformed features.

        :param node_features_list: List of node feature tensors from each GNN layer.
        :param edge_features_list: List of edge feature tensors from each GNN layer.
        :return: Tuple of two dictionaries:
            - Dictionary mapping output names to lists of node last layer features
            - Dictionary mapping output names to lists of edge last layer features
        """
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}

        # Calculating node last layer features
        for output_name, node_heads in self.node_heads.items():
            if output_name not in node_last_layer_features_dict:
                node_last_layer_features_dict[output_name] = []
            for i, node_head in enumerate(node_heads):
                node_last_layer_features_dict[output_name].append(
                    node_head(node_features_list[i])
                )

        # Calculating edge last layer features
        for output_name, edge_heads in self.edge_heads.items():
            if output_name not in edge_last_layer_features_dict:
                edge_last_layer_features_dict[output_name] = []
            for i, edge_head in enumerate(edge_heads):
                edge_last_layer_features_dict[output_name].append(
                    edge_head(edge_features_list[i])
                )

        return node_last_layer_features_dict, edge_last_layer_features_dict

    def _calculate_atomic_predictions(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        requested_output_names: List[str],
    ) -> Tuple[
        Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]
    ]:
        """
        Apply final linear layers to last layer features to produce
        per-atom predictions. Handles multiple blocks per output and sums
        edge contributions with cutoff weighting.

        :param node_last_layer_features_dict: Dictionary mapping output names to
            lists of node last layer features.
        :param edge_last_layer_features_dict: Dictionary mapping output names to
            lists of edge last layer features.
        :param padding_mask: Boolean mask indicating real vs padded neighbors
            [n_atoms, max_num_neighbors].
        :param cutoff_factors: Tensor of cutoff factors for edge distances
            [n_atoms, max_num_neighbors].
        :param requested_output_names: Names of the target outputs to compute.
        :return: Tuple of two dictionaries:
            - Dictionary mapping output names to lists of lists of node atomic
              prediction tensors (one list per GNN layer, one tensor per block)
            - Dictionary mapping output names to lists of lists of edge atomic
              prediction tensors (one list per GNN layer, one tensor per block)
        """
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}

        # Computing node atomic predictions. Since we have last layer features
        # for each GNN layer, and each last layer can have multiple blocks,
        # we apply each last layer block to each of the last layer features.

        for output_name, node_last_layers in self.node_last_layers.items():
            if output_name in requested_output_names:
                node_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for i, node_last_layer in enumerate(node_last_layers):
                    node_last_layer_features = node_last_layer_features_dict[
                        output_name
                    ][i]
                    node_atomic_predictions_by_block: List[torch.Tensor] = []
                    for node_last_layer_by_block in node_last_layer.values():
                        node_atomic_predictions_by_block.append(
                            node_last_layer_by_block(node_last_layer_features)
                        )
                    node_atomic_predictions_dict[output_name].append(
                        node_atomic_predictions_by_block
                    )

        # Computing edge atomic predictions. Following the same logic as above,
        # we (1) iterate over the last layer features and last layer blocks, and (2)
        # sum the edge features with cutoff factors to get their per-node contribution.

        for output_name, edge_last_layers in self.edge_last_layers.items():
            if output_name in requested_output_names:
                edge_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for i, edge_last_layer in enumerate(edge_last_layers):
                    edge_last_layer_features = edge_last_layer_features_dict[
                        output_name
                    ][i]
                    edge_atomic_predictions_by_block: List[torch.Tensor] = []
                    for edge_last_layer_by_block in edge_last_layer.values():
                        edge_atomic_predictions = edge_last_layer_by_block(
                            edge_last_layer_features
                        )
                        expanded_padding_mask = padding_mask[..., None].repeat(
                            1, 1, edge_atomic_predictions.shape[2]
                        )
                        edge_atomic_predictions = torch.where(
                            ~expanded_padding_mask, 0.0, edge_atomic_predictions
                        )
                        edge_atomic_predictions_by_block.append(
                            (edge_atomic_predictions * cutoff_factors[:, :, None]).sum(
                                dim=1
                            )
                        )
                    edge_atomic_predictions_dict[output_name].append(
                        edge_atomic_predictions_by_block
                    )

        return node_atomic_predictions_dict, edge_atomic_predictions_dict


def process_non_conservative_stress(
    tensor: torch.Tensor,
    cells: torch.Tensor,
    system_indices: torch.Tensor,
    num_properties: int,
) -> torch.Tensor:
    """
    Symmetrizes and normalizes by the volume rank-2 Cartesian tensors that are meant
    to predict the non-conservative stress.

    :param tensor: Tensor of shape [n_atoms, 9 * num_properties].
    :param cells: Stacked cell tensors, shape ``(num_systems, 3, 3)``.
    :param system_indices: Tensor mapping each atom to its system index [n_atoms].
    :param num_properties: Number of properties in the tensor (e.g., 6 for stress).
    :return: Symmetrized tensor of shape [n_atoms, 3, 3, num_properties], divided by the
        cell volume.
    """
    # Reshape to 3x3 matrix per atom
    tensor_as_three_by_three = tensor.reshape(-1, 3, 3, num_properties)

    # Normalize by cell volume
    volumes = torch.abs(torch.det(cells))
    # Zero volume can happen due to metatomic's convention of zero cell
    # vectors for non-periodic directions. The actual volume is +inf
    volumes[volumes == 0.0] = torch.inf
    volumes_by_atom = volumes[system_indices].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    tensor_as_three_by_three = tensor_as_three_by_three / volumes_by_atom

    # Symmetrize
    tensor_as_three_by_three = (
        tensor_as_three_by_three + tensor_as_three_by_three.transpose(1, 2)
    ) / 2.0

    return tensor_as_three_by_three
