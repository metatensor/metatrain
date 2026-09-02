from math import prod
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...utils.ensemble import make_ensemble_members
from ...utils.hypers import resolve_per_target
from ...utils.readout import LinearReadout, MoEReadout
from ..documentation import ModelHypers
from .conditioning import SystemConditioningEmbedding
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
        self.nl_is_strict = bool(hypers["long_range"]["enable"])
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
        # ``d_head`` is either a single int (shared by the node and edge heads) or
        # a dict {node: int, edge: int} setting them independently.
        d_head = hypers["d_head"]
        if isinstance(d_head, dict):
            self.d_head_node = int(d_head["node"])
            self.d_head_edge = int(d_head["edge"])
        else:
            self.d_head_node = int(d_head)
            self.d_head_edge = int(d_head)
        # ``head_type`` and ``readout_type`` are either a single (global) value or
        # a dict keyed by target name, resolved per target in ``add_output``.
        self.head_type = hypers["head_type"]
        self.readout_type = hypers["readout_type"]
        self.num_head_layers = hypers["num_head_layers"]
        if self.num_head_layers < 1:
            raise ValueError(
                f"num_head_layers must be >= 1, got {self.num_head_layers}."
            )
        self.d_feedforward = hypers["d_feedforward"]
        self.num_heads = hypers["num_heads"]
        self.num_gnn_layers = hypers["num_gnn_layers"]
        self.num_attention_layers = hypers["num_attention_layers"]
        self.normalization = hypers["normalization"]
        self.activation = hypers["activation"]
        self.attention_temperature = hypers["attention_temperature"]
        self.transformer_type = hypers["transformer_type"]
        self.featurizer_type = hypers["featurizer_type"]
        self.geometry_embedding_l_max = hypers["geometry_embedding_l_max"]
        # Shallow ensemble: a single global {scope, members} setting applied to
        # every target (unlike ``head_type``/``readout_type``, this is not
        # resolved per target). ``shallow_ensemble_scope`` is ``None`` when
        # ensembling is disabled.
        shallow_ensemble = hypers["shallow_ensemble"]
        self.shallow_ensemble_scope: Optional[str] = (
            shallow_ensemble["scope"] if shallow_ensemble is not None else None
        )
        self.shallow_ensemble_members: int = (
            shallow_ensemble["members"] if shallow_ensemble is not None else 1
        )
        # Read via .get(), not direct indexing: a checkpoint saved before these
        # two hypers were added has a ``shallow_ensemble`` dict without them, and
        # checkpoint loading does not re-run options validation (which is what
        # would otherwise fill in the defaults) -- see ``load_checkpoint``. Their
        # defaults (0.0, 1.0) are themselves the inert "no effect" values, so an
        # old checkpoint loads with exactly its original behavior.
        self.shallow_ensemble_dropout: float = (
            shallow_ensemble.get("dropout", 0.0)
            if shallow_ensemble is not None
            else 0.0
        )
        self.shallow_ensemble_bagging: float = (
            shallow_ensemble.get("bagging", 1.0)
            if shallow_ensemble is not None
            else 1.0
        )

        num_atomic_species = len(atomic_types)
        self.num_atomic_species = num_atomic_species

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
                    self.geometry_embedding_l_max,
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

        if hypers["system_conditioning"]:
            self.system_conditioning: Optional[SystemConditioningEmbedding] = (
                SystemConditioningEmbedding(
                    d_out=self.d_node,
                    max_charge=hypers["max_charge"],
                    max_spin_multiplicity=hypers["max_spin_multiplicity"],
                )
            )
        else:
            self.system_conditioning = None

        # Per-output heads and last layers, populated by ``PET._add_output``. When
        # ``shallow_ensemble_scope`` is set, these plain (non-ensembled) dicts are
        # used only for the parts that stay shared across members: with
        # ``scope="readout"`` the heads are shared, so ``node_heads``/``edge_heads``
        # are still populated normally, but ``node_last_layers``/``edge_last_layers``
        # are not (the readouts live in ``*_last_layers_ensemble`` instead); with
        # ``scope="head"`` neither pair is populated at all (everything lives in the
        # ``*_ensemble`` dicts below). This keeps the non-ensembled code path (the
        # methods below that read these four dicts) completely untouched.
        self.node_heads = torch.nn.ModuleDict()
        self.edge_heads = torch.nn.ModuleDict()
        self.node_last_layers = torch.nn.ModuleDict()
        self.edge_last_layers = torch.nn.ModuleDict()
        # Ensembled heads, only populated for targets when
        # ``shallow_ensemble_scope == "head"``: target -> ``member_{e}`` ->
        # flat, layer-major ``ModuleList`` of heads (same shape ``_make_heads``
        # would return for a single, non-ensembled target).
        self.node_heads_ensemble = torch.nn.ModuleDict()
        self.edge_heads_ensemble = torch.nn.ModuleDict()
        # Ensembled readouts. Although ``shallow_ensemble_scope`` is a single,
        # global setting (so only one of the two pairs below is ever actually
        # populated by a given model), TorchScript compiles every method against
        # the concrete nesting of *both* containers, so the two scopes need
        # separate containers rather than one sharing a scope-dependent nesting
        # order (the unpopulated one just stays an empty ``ModuleDict``):
        # - ``*_ensemble_readout``: target -> readout layer -> block key ->
        #   ``member_{e}`` -> readout (member axis innermost, since the heads
        #   feeding into these readouts are shared across members).
        # - ``*_ensemble_head``: target -> ``member_{e}`` -> readout layer ->
        #   block key -> readout (member axis outermost, matching the per-member
        #   heads in ``node_heads_ensemble``/``edge_heads_ensemble`` above).
        self.node_last_layers_ensemble_readout = torch.nn.ModuleDict()
        self.edge_last_layers_ensemble_readout = torch.nn.ModuleDict()
        self.node_last_layers_ensemble_head = torch.nn.ModuleDict()
        self.edge_last_layers_ensemble_head = torch.nn.ModuleDict()
        # Heads per readout layer for each target: 1 for ``head_type="per_target"``
        # (one head shared by all blocks), or the number of blocks for
        # ``head_type="per_block"``. The heads are stored in a flat, layer-major
        # ``ModuleList``, and this is the stride needed to index into it. Populated
        # for every target, ensembled or not.
        self.heads_per_layer: Dict[str, int] = {}

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
        head_type = resolve_per_target(self.head_type, target_name, "per_target")
        readout_spec = resolve_per_target(
            self.readout_type,
            target_name,
            {"atom_type_gating": False, "hypers": {}},
            spec_keys=("atom_type_gating",),
        )

        # With ``head_type="per_target"`` a single head is shared by all of the
        # target's blocks; with ``"per_block"`` there is one head per block. Either
        # way the heads are stored as a flat ``ModuleList``, ordered layer-major
        # with ``heads_per_layer`` heads per readout layer. For ``"per_target"``
        # that is exactly one head per layer, i.e. the standard PET layout, so
        # parameter names and diagnostic module paths are unchanged.
        heads_per_layer = len(output_shapes) if head_type == "per_block" else 1
        self.heads_per_layer[target_name] = heads_per_layer

        if self.shallow_ensemble_scope is None:
            self.node_heads[target_name] = self._make_heads(
                self.d_node, self.d_head_node, heads_per_layer
            )
            self.edge_heads[target_name] = self._make_heads(
                self.d_pet, self.d_head_edge, heads_per_layer
            )
            # Readouts consume the head output, so their input dimension is d_head.
            self.node_last_layers[target_name] = self._make_readouts_per_layer(
                self.d_head_node, output_shapes, readout_spec
            )
            self.edge_last_layers[target_name] = self._make_readouts_per_layer(
                self.d_head_edge, output_shapes, readout_spec
            )
        elif self.shallow_ensemble_scope == "readout":
            # Heads are shared across members, exactly as in the non-ensembled case.
            self.node_heads[target_name] = self._make_heads(
                self.d_node, self.d_head_node, heads_per_layer
            )
            self.edge_heads[target_name] = self._make_heads(
                self.d_pet, self.d_head_edge, heads_per_layer
            )
            # Only the readouts are replicated per member, member axis innermost.
            self.node_last_layers_ensemble_readout[target_name] = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            key: make_ensemble_members(
                                lambda shape=shape: self._make_readout(  # type: ignore[misc]
                                    self.d_head_node, prod(shape), readout_spec
                                ),
                                self.shallow_ensemble_members,
                            )
                            for key, shape in output_shapes.items()
                        }
                    )
                    for _ in range(self.num_readout_layers)
                ]
            )
            self.edge_last_layers_ensemble_readout[target_name] = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            key: make_ensemble_members(
                                lambda shape=shape: self._make_readout(  # type: ignore[misc]
                                    self.d_head_edge, prod(shape), readout_spec
                                ),
                                self.shallow_ensemble_members,
                            )
                            for key, shape in output_shapes.items()
                        }
                    )
                    for _ in range(self.num_readout_layers)
                ]
            )
        else:
            assert self.shallow_ensemble_scope == "head"
            # Both heads and readouts are fully replicated per member, member axis
            # outermost, so each member is an independent head+readout pipeline.
            self.node_heads_ensemble[target_name] = make_ensemble_members(
                lambda: self._make_heads(
                    self.d_node,
                    self.d_head_node,
                    heads_per_layer,
                    self.shallow_ensemble_dropout,
                ),
                self.shallow_ensemble_members,
            )
            self.edge_heads_ensemble[target_name] = make_ensemble_members(
                lambda: self._make_heads(
                    self.d_pet,
                    self.d_head_edge,
                    heads_per_layer,
                    self.shallow_ensemble_dropout,
                ),
                self.shallow_ensemble_members,
            )
            self.node_last_layers_ensemble_head[target_name] = make_ensemble_members(
                lambda: self._make_readouts_per_layer(
                    self.d_head_node, output_shapes, readout_spec
                ),
                self.shallow_ensemble_members,
            )
            self.edge_last_layers_ensemble_head[target_name] = make_ensemble_members(
                lambda: self._make_readouts_per_layer(
                    self.d_head_edge, output_shapes, readout_spec
                ),
                self.shallow_ensemble_members,
            )

    def remove_output(self, target_name: str) -> None:
        """
        Remove the node/edge heads and last layers for a previously registered output
        target, mirroring :meth:`add_output`.

        ``torch.nn.ModuleDict.pop`` has no ``default`` argument, so presence is
        checked explicitly before deleting.

        :param target_name: Name of the target to remove.
        """
        for module_dict in (
            self.node_heads,
            self.edge_heads,
            self.node_last_layers,
            self.edge_last_layers,
            self.node_heads_ensemble,
            self.edge_heads_ensemble,
            self.node_last_layers_ensemble_readout,
            self.edge_last_layers_ensemble_readout,
            self.node_last_layers_ensemble_head,
            self.edge_last_layers_ensemble_head,
        ):
            if target_name in module_dict:
                del module_dict[target_name]
        self.heads_per_layer.pop(target_name, None)

    def _make_heads(
        self, d_in: int, d_out: int, heads_per_layer: int, dropout: float = 0.0
    ) -> torch.nn.ModuleList:
        """
        Build the flat, layer-major list of head MLPs for one target.

        Each head is ``num_head_layers`` Linear+SiLU layers, the first mapping
        ``d_in`` -> ``d_out`` and the rest ``d_out`` -> ``d_out``. The returned
        list holds ``num_readout_layers * heads_per_layer`` of them: all the heads
        of readout layer 0, then those of layer 1, and so on. Keeping it flat (and
        of uniform element type) is what makes the forward pass TorchScript-able,
        since TorchScript cannot index a ``ModuleList`` dynamically.

        :param d_in: Input feature dimension (``d_node`` for nodes, ``d_pet`` for
            edges).
        :param d_out: Head output dimension (``d_head_node`` / ``d_head_edge``).
        :param heads_per_layer: Number of heads per readout layer.
        :param dropout: Dropout probability inserted after every Linear+SiLU
            pair. ``0`` (the default) inserts no ``Dropout`` module at all,
            rather than one with ``p=0`` acting as a no-op, so that the module
            structure -- and hence the state dict's layer indices -- is
            byte-for-byte identical to before this parameter existed whenever
            it is left at its default. Only :meth:`add_output`'s
            ``scope="head"`` branch ever passes a nonzero value (see
            ``shallow_ensemble.dropout``'s own docstring for why).
        :return: A ``ModuleList`` of head MLPs.
        """
        heads: List[torch.nn.Module] = []
        for _ in range(self.num_readout_layers * heads_per_layer):
            layers: List[torch.nn.Module] = [
                torch.nn.Linear(d_in, d_out),
                torch.nn.SiLU(),
            ]
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))
            for _ in range(self.num_head_layers - 1):
                layers.extend([torch.nn.Linear(d_out, d_out), torch.nn.SiLU()])
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(p=dropout))
            heads.append(torch.nn.Sequential(*layers))
        return torch.nn.ModuleList(heads)

    def _make_readout(
        self, in_features: int, out_features: int, readout_spec: Dict[str, Any]
    ) -> torch.nn.Module:
        """
        Build the (linear) readout for one output block.

        The readout is strictly linear; all nonlinearity lives in the heads. Its
        optional conditioning on the central-atom type is set by
        ``readout_spec["atom_type_gating"]``: ``False`` for a single shared linear
        map (the standard PET readout), ``"one-hot"`` for an independent map per
        atomic type, or ``"moe"`` for a mixture of experts routed by an embedding
        of the atomic type.

        :param in_features: Input feature dimension (the head dimension).
        :param out_features: Output feature dimension for this block.
        :param readout_spec: The per-target readout spec, with keys
            ``atom_type_gating`` and ``hypers``.
        :return: A module with forward signature ``(features, group_idx)``.
        """
        gating = readout_spec.get("atom_type_gating", False)
        hypers = readout_spec.get("hypers", {})

        if not gating:
            return LinearReadout(in_features, out_features, bias=True)
        if gating == "one-hot":
            return LinearReadout(
                in_features,
                out_features,
                n_groups=self.num_atomic_species,
                bias=True,
            )
        if gating == "moe":
            return MoEReadout(
                in_features,
                out_features,
                n_groups=self.num_atomic_species,
                num_experts=hypers["num_experts"],
                num_routed_experts=hypers["num_routed_experts"],
                num_topk_experts=hypers["num_topk_experts"],
                bias=True,
                embedding_dim=hypers.get("embedding_dim", 16),
            )
        raise ValueError(
            f"Unknown atom_type_gating: {gating}. "
            "Available options are: false, 'one-hot' and 'moe'."
        )

    def _make_readouts_per_layer(
        self,
        in_features: int,
        output_shapes: Dict[str, List[int]],
        readout_spec: Dict[str, Any],
    ) -> torch.nn.ModuleList:
        """
        Build the per-readout-layer, per-block readouts for one (non-ensembled, or
        single ensemble member's) target: a ``ModuleList`` of ``num_readout_layers``
        ``ModuleDict``s, each mapping block key to its readout module.

        :param in_features: Input feature dimension (the head dimension).
        :param output_shapes: Mapping from per-block key to the block's shape.
        :param readout_spec: The per-target readout spec, see :meth:`_make_readout`.
        :return: A ``ModuleList`` of ``ModuleDict``s, one per readout layer.
        """
        return torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: self._make_readout(in_features, prod(shape), readout_spec)
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
        cutoff_width_adaptive: float,
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
        :param cutoff_width_adaptive: Width of the smooth cutoff taper used by the
            adaptive cutoff scheme when ``num_neighbors_adaptive`` is set.
        :return: A dictionary ``batch_data`` of the intermediate tensors:
            - `element_indices_nodes`: The atomic species of the central atoms
            - `element_indices_neighbors`: The atomic species of the neighboring atoms
            - `edge_vectors`: The cartesian edge vectors between the central atoms and
                their neighbors
            - `edge_distances`: The distances between the central atoms and their
                neighbors
            - `padding_mask`: A padding mask indicating which neighbors are real, and
                which are padded
            - `reverse_neighbor_index`: The reversed neighbor list for each central atom
            - `cutoff_factors`: The cutoff function values for each edge
            - `atomic_cutoffs_stats`: Diagnostic per-atom cutoff radius (detached
                from the autograd graph). With adaptive cutoff active this is
                the per-atom adapted cutoff; otherwise every entry equals
                ``cutoff``. Always shape ``(num_nodes,)``.
            - `centers`: Flat tensor of center atom global indices for each real
            (non-padded) edge, shape ``(n_edges,)``. Suitable for use with
            :func:`get_pair_sample_labels`.
            - `neighbors`: Flat tensor of neighbor atom global indices for each real
            (non-padded) edge, shape ``(n_edges,)``. Suitable for use with
            :func:`get_pair_sample_labels`.
            - `nef_to_edges_neighbor`: Index tensor of shape ``(n_edges,)`` such that
            `nef_tensor[centers, nef_to_edges_neighbor]` recovers the flat edge array
            from a NEF-format tensor. Needed to flatten 3D (edge-like) hook outputs back
            to per-edge arrays for TensorMap construction.
            - `cell_shifts`: Integer cell shift vectors for each real (non-padded) edge,
            shape ``(n_edges, 3)``. Columns correspond to ``(cell_shift_a, cell_shift_b,
            cell_shift_c)``. Suitable for use with :func:`get_pair_sample_labels`.
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
            cutoff_width_adaptive,
            self.nl_is_strict,
        )

        batch_data: Dict[str, torch.Tensor] = {
            "element_indices_nodes": element_indices_nodes,
            "element_indices_neighbors": element_indices_neighbors,
            "edge_vectors": edge_vectors,
            "edge_distances": edge_distances,
            "padding_mask": padding_mask,
            "reverse_neighbor_index": reverse_neighbor_index,
            "cutoff_factors": cutoff_factors,
            "atomic_cutoffs_stats": atomic_cutoffs_stats,
            "centers": centers,
            "neighbors": neighbors,
            "nef_to_edges_neighbor": nef_to_edges_neighbor,
            "cell_shifts": cell_shifts,
        }
        return batch_data

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
        if self.system_conditioning is not None:
            featurizer_inputs["charge"] = batch_data["charge"]
            featurizer_inputs["spin_multiplicity"] = batch_data["spin_multiplicity"]
            featurizer_inputs["system_indices"] = batch_data["system_indices"]

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

    def _bagged_ensemble_mean(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Member-weighted mean over the ensemble, for online bagging.

        See ``shallow_ensemble.bagging``'s own docstring for the motivation.
        During training, and only when bagging is enabled, each member's row is
        kept with probability ``self.shallow_ensemble_bagging``, independently
        per member and per atom (this block's leading sample axis) -- so the
        set of atoms a given member actually "sees" a nonzero gradient from
        differs, and reshuffles, on every forward pass. An atom for which every
        member is dropped that draw falls back to the plain, uniform mean for
        that atom, so the mean is never left undefined by an unlucky draw. At
        eval time, or with bagging disabled (the default), this is exactly
        ``stacked.mean(dim=0)``, matching the computation before this method
        existed.

        :param stacked: per-member predictions for one block, member axis
            first: ``(n_members, n_atoms, ...)``, any number of trailing
            component/property dimensions.
        :return: the (possibly bagging-weighted) mean over members, shape
            ``stacked.shape[1:]``.
        """
        if not self.training or self.shallow_ensemble_bagging >= 1.0:
            return stacked.mean(dim=0)

        n_members = stacked.shape[0]
        n_atoms = stacked.shape[1]
        # trailing component/property dims, forced to size 1 so `weights`/`total`
        # broadcast against `stacked` regardless of how many such dims it has
        trailing_ones: List[int] = []
        for _ in range(2, stacked.dim()):
            trailing_ones.append(1)

        weights = torch.bernoulli(
            torch.full(
                (n_members, n_atoms),
                self.shallow_ensemble_bagging,
                dtype=stacked.dtype,
                device=stacked.device,
            )
        )
        total = weights.sum(dim=0)  # (n_atoms,): how many members kept this atom
        total_safe = torch.where(total > 0, total, torch.ones_like(total))

        weights_view = weights.view([n_members, n_atoms] + trailing_ones)
        total_view = total_safe.view([n_atoms] + trailing_ones)
        weighted_mean = (weights_view * stacked).sum(dim=0) / total_view

        all_dropped = (total == 0).view([n_atoms] + trailing_ones)
        return torch.where(all_dropped, stacked.mean(dim=0), weighted_mean)

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
        :return: A tuple ``(atomic_predictions, node_ll_features, edge_ll_features,
            uncertainty)``. ``atomic_predictions`` maps each requested output to a
            list of per-block flat prediction tensors -- for a shallow ensemble
            target, this is the *mean* over members, so that every other caller of
            :meth:`predict` keeps seeing exactly one prediction per output, unaware
            that ensembling is active. The last-layer feature dictionaries map each
            output to its per-layer node / edge last-layer features (only for
            non-ensembled and ``scope="readout"`` targets; ``scope="head"`` targets
            have per-member features that are not exposed here). ``uncertainty``
            maps each *ensembled* requested output to a list of per-block
            variance-over-members tensors, same shape as ``atomic_predictions``.
            Empty for outputs that are not ensembled.
        """
        padding_mask = batch_data["padding_mask"]
        cutoff_factors = batch_data["cutoff_factors"]
        element_indices_nodes = batch_data["element_indices_nodes"]

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
                element_indices_nodes,
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

        # Shallow ensemble: compute every member's prediction (summed over GNN
        # layers and node/edge contributions, exactly like above), then write the
        # mean into ``atomic_predictions`` under the same key and return the
        # variance across members separately -- unlike the plain last-layer
        # features above, ``scope="head"`` targets need their own per-member
        # features, computed separately.
        uncertainty: Dict[str, List[torch.Tensor]] = {}
        if self.shallow_ensemble_scope is not None:
            node_ll_features_ens, edge_ll_features_ens = (
                self._calculate_last_layer_features_ensemble(
                    node_features_list, edge_features_list
                )
            )
            node_member_predictions, edge_member_predictions = (
                self._calculate_atomic_predictions_ensemble(
                    node_ll_features,
                    edge_ll_features,
                    node_ll_features_ens,
                    edge_ll_features_ens,
                    padding_mask,
                    cutoff_factors,
                    requested_output_names,
                    element_indices_nodes,
                )
            )

            for output_name in node_member_predictions.keys():
                node_by_member = node_member_predictions[output_name]
                edge_by_member = edge_member_predictions[output_name]
                n_members = len(node_by_member)
                member_block_sums: List[List[torch.Tensor]] = []
                for m in range(n_members):
                    node_by_layer = node_by_member[m]
                    edge_by_layer = edge_by_member[m]
                    num_blocks = len(node_by_layer[0])
                    block_sums = []
                    for b in range(num_blocks):
                        block_sum = node_by_layer[0][b] + edge_by_layer[0][b]
                        for layer in range(1, len(node_by_layer)):
                            block_sum = (
                                block_sum
                                + node_by_layer[layer][b]
                                + edge_by_layer[layer][b]
                            )
                        block_sums.append(block_sum)

                    if output_name == "non_conservative_stress":  # TODO: variants
                        num_properties = block_sums[0].shape[1] // 9
                        block_sums[0] = process_non_conservative_stress(
                            block_sums[0], cells, system_indices, num_properties
                        )

                    member_block_sums.append(block_sums)

                # mean and (unbiased) variance over members, block by block; the
                # mean replaces the value ``atomic_predictions`` would otherwise
                # be missing for this (ensembled) output, since it is absent from
                # ``node_atomic_predictions_dict`` above
                num_blocks = len(member_block_sums[0])
                mean_blocks: List[torch.Tensor] = []
                variance_blocks: List[torch.Tensor] = []
                for b in range(num_blocks):
                    stacked = torch.stack(
                        [member_block_sums[m][b] for m in range(n_members)], dim=0
                    )
                    mean_blocks.append(self._bagged_ensemble_mean(stacked))
                    variance_blocks.append(stacked.var(dim=0, unbiased=True))
                atomic_predictions[output_name] = mean_blocks
                uncertainty[output_name] = variance_blocks

        return (
            atomic_predictions,
            node_ll_features,
            edge_ll_features,
            uncertainty,
        )

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
        # Compute conditioning embedding once: same inputs at every GNN layer.
        cond_embedding: Optional[torch.Tensor] = None
        if self.system_conditioning is not None:
            cond_embedding = self.system_conditioning(
                inputs["charge"],
                inputs["spin_multiplicity"],
                inputs["system_indices"],
            )

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

            if cond_embedding is not None:
                output_node_embeddings = output_node_embeddings + cond_embedding

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
        # Compute conditioning embedding once: same inputs at every GNN layer.
        cond_embedding: Optional[torch.Tensor] = None
        if self.system_conditioning is not None:
            cond_embedding = self.system_conditioning(
                inputs["charge"],
                inputs["spin_multiplicity"],
                inputs["system_indices"],
            )
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
            if cond_embedding is not None:
                output_node_embeddings = output_node_embeddings + cond_embedding

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

        The heads are stored flat and layer-major, with ``heads_per_layer[output]``
        heads per readout layer (1 for ``head_type="per_target"``, one per block for
        ``head_type="per_block"``), so the returned lists are flat and layer-major
        too: head ``j`` of readout layer ``i`` is at index ``i * heads_per_layer +
        j``.

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
            heads_per_layer = self.heads_per_layer[output_name]
            node_features: List[torch.Tensor] = []
            for i, node_head in enumerate(node_heads):
                node_features.append(
                    node_head(node_features_list[i // heads_per_layer])
                )
            node_last_layer_features_dict[output_name] = node_features

        # Calculating edge last layer features
        for output_name, edge_heads in self.edge_heads.items():
            heads_per_layer = self.heads_per_layer[output_name]
            edge_features: List[torch.Tensor] = []
            for i, edge_head in enumerate(edge_heads):
                edge_features.append(
                    edge_head(edge_features_list[i // heads_per_layer])
                )
            edge_last_layer_features_dict[output_name] = edge_features

        return node_last_layer_features_dict, edge_last_layer_features_dict

    def _calculate_last_layer_features_ensemble(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
    ) -> Tuple[
        Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]
    ]:
        """
        Like :meth:`_calculate_last_layer_features`, but for ``scope="head"``
        ensembled targets: each member has its own heads, so this returns, for
        each output, one flat (layer-major) feature list per member.

        Only ever has entries when ``shallow_ensemble_scope == "head"``: with
        ``scope="readout"`` the heads are shared, and last-layer features for
        those targets are computed once by the ordinary (unensembled)
        :meth:`_calculate_last_layer_features`, via the plain ``node_heads``/
        ``edge_heads`` dicts.

        :param node_features_list: List of node feature tensors from each GNN
            layer.
        :param edge_features_list: List of edge feature tensors from each GNN
            layer.
        :return: Tuple of two dictionaries mapping output names to a list of
            per-member, flat (layer-major) last-layer feature lists.
        """
        node_last_layer_features_dict: Dict[str, List[List[torch.Tensor]]] = {}
        edge_last_layer_features_dict: Dict[str, List[List[torch.Tensor]]] = {}

        for output_name, member_heads in self.node_heads_ensemble.items():
            heads_per_layer = self.heads_per_layer[output_name]
            member_node_features: List[List[torch.Tensor]] = []
            for node_heads_m in member_heads.values():
                features: List[torch.Tensor] = []
                for i, node_head in enumerate(node_heads_m):
                    features.append(node_head(node_features_list[i // heads_per_layer]))
                member_node_features.append(features)
            node_last_layer_features_dict[output_name] = member_node_features

        for output_name, member_heads in self.edge_heads_ensemble.items():
            heads_per_layer = self.heads_per_layer[output_name]
            member_edge_features: List[List[torch.Tensor]] = []
            for edge_heads_m in member_heads.values():
                features = []
                for i, edge_head in enumerate(edge_heads_m):
                    features.append(edge_head(edge_features_list[i // heads_per_layer]))
                member_edge_features.append(features)
            edge_last_layer_features_dict[output_name] = member_edge_features

        return node_last_layer_features_dict, edge_last_layer_features_dict

    def _calculate_atomic_predictions_ensemble(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        node_last_layer_features_ensemble_dict: Dict[str, List[List[torch.Tensor]]],
        edge_last_layer_features_ensemble_dict: Dict[str, List[List[torch.Tensor]]],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        requested_output_names: List[str],
        element_indices_nodes: torch.Tensor,
    ) -> Tuple[
        Dict[str, List[List[List[torch.Tensor]]]],
        Dict[str, List[List[List[torch.Tensor]]]],
    ]:
        """
        Ensembled counterpart of :meth:`_calculate_atomic_predictions`: applies the
        per-member readouts, for every ensembled target. Node and edge
        contributions are *not* yet summed together, nor over GNN layers -- that
        still happens once per member in :meth:`predict`, mirroring exactly what
        it already does for non-ensembled targets.

        :param node_last_layer_features_dict: Shared-head node last-layer
            features (from the ordinary, unensembled
            :meth:`_calculate_last_layer_features`), used for ``scope="readout"``
            targets.
        :param edge_last_layer_features_dict: As above, for edges.
        :param node_last_layer_features_ensemble_dict: Per-member node last-layer
            features (from :meth:`_calculate_last_layer_features_ensemble`), used
            for ``scope="head"`` targets.
        :param edge_last_layer_features_ensemble_dict: As above, for edges.
        :param padding_mask: Boolean mask indicating real vs padded neighbors.
        :param cutoff_factors: Cutoff factors for edge distances.
        :param requested_output_names: Names of the target outputs to compute.
        :param element_indices_nodes: Species index of each central atom.
        :return: Tuple of two dictionaries mapping output name to a list (one
            entry per member) of lists (one per GNN readout layer) of per-block
            prediction tensors -- the same shape :meth:`_calculate_atomic_predictions`
            returns for a single target, with an extra outer "member" list.
        """
        node_out: Dict[str, List[List[List[torch.Tensor]]]] = {}
        edge_out: Dict[str, List[List[List[torch.Tensor]]]] = {}

        # ----- scope="readout": member axis innermost, shared last-layer features
        for (
            output_name,
            node_last_layers,
        ) in self.node_last_layers_ensemble_readout.items():
            # TorchScript unrolls iteration over an ``nn.ModuleDict``, and does not
            # allow ``continue``/``break`` inside such an unrolled loop body, so the
            # "is this target requested" check has to wrap the whole body instead
            # (as the non-ensembled methods above also do).
            if (
                output_name in requested_output_names
                and output_name in node_last_layer_features_dict
            ):
                heads_per_layer = self.heads_per_layer[output_name]
                node_features = node_last_layer_features_dict[output_name]
                n_members = self.shallow_ensemble_members
                by_member: List[List[List[torch.Tensor]]] = [
                    torch.jit.annotate(List[List[torch.Tensor]], [])
                    for _ in range(n_members)
                ]
                for i, node_last_layer in enumerate(node_last_layers):
                    by_member_this_layer: List[List[torch.Tensor]] = [
                        torch.jit.annotate(List[torch.Tensor], [])
                        for _ in range(n_members)
                    ]
                    j = 0
                    for block_members in node_last_layer.values():
                        features = node_features[
                            i * heads_per_layer + j % heads_per_layer
                        ]
                        m = 0
                        for readout_m in block_members.values():
                            by_member_this_layer[m].append(
                                readout_m(features, element_indices_nodes)
                            )
                            m += 1
                        j += 1
                    for m in range(n_members):
                        by_member[m].append(by_member_this_layer[m])
                node_out[output_name] = by_member

        for (
            output_name,
            edge_last_layers,
        ) in self.edge_last_layers_ensemble_readout.items():
            if (
                output_name in requested_output_names
                and output_name in edge_last_layer_features_dict
            ):
                heads_per_layer = self.heads_per_layer[output_name]
                edge_features = edge_last_layer_features_dict[output_name]
                n_members = self.shallow_ensemble_members
                by_member = [
                    torch.jit.annotate(List[List[torch.Tensor]], [])
                    for _ in range(n_members)
                ]
                for i, edge_last_layer in enumerate(edge_last_layers):
                    edge_by_member_this_layer: List[List[torch.Tensor]] = [
                        torch.jit.annotate(List[torch.Tensor], [])
                        for _ in range(n_members)
                    ]
                    j = 0
                    for block_members in edge_last_layer.values():
                        features = edge_features[
                            i * heads_per_layer + j % heads_per_layer
                        ]
                        m = 0
                        for readout_m in block_members.values():
                            edge_atomic_predictions = readout_m(
                                features, element_indices_nodes
                            )
                            expanded_padding_mask = padding_mask[..., None].repeat(
                                1, 1, edge_atomic_predictions.shape[2]
                            )
                            edge_atomic_predictions = torch.where(
                                ~expanded_padding_mask, 0.0, edge_atomic_predictions
                            )
                            edge_by_member_this_layer[m].append(
                                (
                                    edge_atomic_predictions * cutoff_factors[:, :, None]
                                ).sum(dim=1)
                            )
                            m += 1
                        j += 1
                    for m in range(n_members):
                        by_member[m].append(edge_by_member_this_layer[m])
                edge_out[output_name] = by_member

        # ----- scope="head": member axis outermost, features already per-member
        for (
            output_name,
            node_last_layers_ens,
        ) in self.node_last_layers_ensemble_head.items():
            if (
                output_name in requested_output_names
                and output_name in node_last_layer_features_ensemble_dict
            ):
                heads_per_layer = self.heads_per_layer[output_name]
                member_features_list = node_last_layer_features_ensemble_dict[
                    output_name
                ]
                by_member_head: List[List[List[torch.Tensor]]] = []
                member_idx = 0
                for node_last_layers_m in node_last_layers_ens.values():
                    node_features_m = member_features_list[member_idx]
                    layer_blocks: List[List[torch.Tensor]] = []
                    for i, node_last_layer in enumerate(node_last_layers_m):
                        block_preds: List[torch.Tensor] = torch.jit.annotate(
                            List[torch.Tensor], []
                        )
                        j = 0
                        for node_last_layer_by_block in node_last_layer.values():
                            features = node_features_m[
                                i * heads_per_layer + j % heads_per_layer
                            ]
                            block_preds.append(
                                node_last_layer_by_block(
                                    features, element_indices_nodes
                                )
                            )
                            j += 1
                        layer_blocks.append(block_preds)
                    by_member_head.append(layer_blocks)
                    member_idx += 1
                node_out[output_name] = by_member_head

        for (
            output_name,
            edge_last_layers_ens,
        ) in self.edge_last_layers_ensemble_head.items():
            if (
                output_name in requested_output_names
                and output_name in edge_last_layer_features_ensemble_dict
            ):
                heads_per_layer = self.heads_per_layer[output_name]
                member_features_list = edge_last_layer_features_ensemble_dict[
                    output_name
                ]
                by_member_head_edge: List[List[List[torch.Tensor]]] = []
                member_idx = 0
                for edge_last_layers_m in edge_last_layers_ens.values():
                    edge_features_m = member_features_list[member_idx]
                    layer_blocks_e: List[List[torch.Tensor]] = []
                    for i, edge_last_layer in enumerate(edge_last_layers_m):
                        block_preds_e: List[torch.Tensor] = torch.jit.annotate(
                            List[torch.Tensor], []
                        )
                        j = 0
                        for edge_last_layer_by_block in edge_last_layer.values():
                            features = edge_features_m[
                                i * heads_per_layer + j % heads_per_layer
                            ]
                            j += 1
                            edge_atomic_predictions = edge_last_layer_by_block(
                                features, element_indices_nodes
                            )
                            expanded_padding_mask = padding_mask[..., None].repeat(
                                1, 1, edge_atomic_predictions.shape[2]
                            )
                            edge_atomic_predictions = torch.where(
                                ~expanded_padding_mask, 0.0, edge_atomic_predictions
                            )
                            block_preds_e.append(
                                (
                                    edge_atomic_predictions * cutoff_factors[:, :, None]
                                ).sum(dim=1)
                            )
                        layer_blocks_e.append(block_preds_e)
                    by_member_head_edge.append(layer_blocks_e)
                    member_idx += 1
                edge_out[output_name] = by_member_head_edge

        return node_out, edge_out

    def _calculate_atomic_predictions(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        requested_output_names: List[str],
        element_indices_nodes: torch.Tensor,
    ) -> Tuple[
        Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]
    ]:
        """
        Apply final linear layers to last layer features to produce
        per-atom predictions. Handles multiple blocks per output and sums
        edge contributions with cutoff weighting.

        The last layer features arrive flat and layer-major, so block ``j`` of
        readout layer ``i`` reads the features at index ``i * heads_per_layer + j``
        -- which is the block's own head under ``head_type="per_block"``, and the
        single head shared by all blocks under ``head_type="per_target"``.

        The readouts also take a per-atom group index, used only when atom-type
        conditioning is active and ignored otherwise. Both the node and the edge
        readouts are conditioned on the central-atom type; for the edge readout the
        type is therefore shared across that atom's neighbors.

        :param node_last_layer_features_dict: Dictionary mapping output names to
            lists of node last layer features.
        :param edge_last_layer_features_dict: Dictionary mapping output names to
            lists of edge last layer features.
        :param padding_mask: Boolean mask indicating real vs padded neighbors
            [n_atoms, max_num_neighbors].
        :param cutoff_factors: Tensor of cutoff factors for edge distances
            [n_atoms, max_num_neighbors].
        :param requested_output_names: Names of the target outputs to compute.
        :param element_indices_nodes: Species index of each central atom [n_atoms].
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
                heads_per_layer = self.heads_per_layer[output_name]
                node_features = node_last_layer_features_dict[output_name]
                for i, node_last_layer in enumerate(node_last_layers):
                    node_atomic_predictions_by_block: List[torch.Tensor] = []
                    j = 0
                    for node_last_layer_by_block in node_last_layer.values():
                        # ``j % heads_per_layer`` is this block's own head under
                        # "per_block", and 0 (the shared head) under "per_target".
                        features = node_features[
                            i * heads_per_layer + j % heads_per_layer
                        ]
                        node_atomic_predictions_by_block.append(
                            node_last_layer_by_block(features, element_indices_nodes)
                        )
                        j += 1
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
                heads_per_layer = self.heads_per_layer[output_name]
                edge_features = edge_last_layer_features_dict[output_name]
                for i, edge_last_layer in enumerate(edge_last_layers):
                    edge_atomic_predictions_by_block: List[torch.Tensor] = []
                    j = 0
                    for edge_last_layer_by_block in edge_last_layer.values():
                        features = edge_features[
                            i * heads_per_layer + j % heads_per_layer
                        ]
                        j += 1
                        edge_atomic_predictions = edge_last_layer_by_block(
                            features, element_indices_nodes
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
