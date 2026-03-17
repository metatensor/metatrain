"""
Utilities for capturing intermediate tensor outputs from PET sub-modules.
"""

from typing import Any, Dict, List, Set

import torch
import torch.nn
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput


FEATURIZER_INPUT_NAMES: Set[str] = {
    "element_indices_nodes",
    "element_indices_neighbors",
    "edge_vectors",
    "edge_distances",
    "reverse_neighbor_index",
    "padding_mask",
    "cutoff_factors",
}

DIAGNOSTIC_PREFIX = "mtt::features::"


def create_diagnostic_feature_tensormap(
    tensor: torch.Tensor,
    centers: torch.Tensor,
    nef_to_edges_neighbor: torch.Tensor,
    sample_labels: Labels,
    pair_sample_labels: Labels,
) -> TensorMap:
    """
    Wrap an intermediate tensor into a :class:`~metatensor.torch.TensorMap` suitable for
    diagnostic output. TensorMaps are a single block with a dummy key. The TensorBlock
    has samples either per-atom or per-pair (depending on the features captured), no
    components, and a single property dimension "_" that indexes the feature vector.

    The dimensionality of ``tensor`` is used to decide which sample labels to attach:

    * Shape: ``(n_atoms, d)`` -> node-like tensor. Samples are per-atom and
      ``sample_labels`` with dimensions ``["system", "atom"]`` are used.
    * Shape: ``(n_atoms, max_neighbors, d)`` -> edge-like tensor in NEF format. The
      tensor is first flattened to ``(n_edges, d)``, and ``pair_sample_labels`` with
      dimensions ``["system", "first_atom", "second_atom", "cell_shift_a",
      "cell_shift_b", "cell_shift_c"]``) is used.

    :param tensor: The intermediate tensor to wrap into a TensorMap.
    :param centers: Flat center-atom indices, shape ``(n_edges,)``.
    :param nef_to_edges_neighbor: NEF-slot indices, shape ``(n_edges,)``.
    :param sample_labels: Per-atom labels with dimensions ``["system", "atom"]``.
    :param pair_sample_labels: Per-pair labels with dimension ``["system", "first_atom",
        "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"]``.
    :return: A single-block TensorMap containing the captured features in ``tensor``.

    :raises ValueError: If ``tensor`` has an unexpected shape.
    """
    device = tensor.device
    outp = tensor.detach().clone()

    if outp.ndim == 2:
        # Node-like: (n_atoms, d)
        labels = sample_labels
    elif outp.ndim == 3:
        if outp.shape[1] == 1:  # special case: node-like with a dummy neighbor axis
            outp = outp.squeeze(1)  # (n_atoms, 1, d) → (n_atoms, d)
            labels = sample_labels
        else:
            # Edge-like in NEF format: (n_atoms, max_neighbors, d) → (n_edges, d)
            outp = outp[centers, nef_to_edges_neighbor]
            labels = pair_sample_labels
    else:
        raise ValueError(
            f"Unexpected tensor shape for diagnostic capture: {outp.shape}. "
            "Expected a 2-D node-like tensor (n_atoms, d) or a 3-D edge-like "
            "tensor in NEF format (n_atoms, max_neighbors, d).  "
            "Raw featurizer inputs (element_indices_nodes, padding_mask, etc.) "
            "must be normalized with normalize_featurizer_input_tensor() before "
            "being passed here."
        )

    return TensorMap(
        Labels(["_"], torch.tensor([[0]], device=device)),
        [
            TensorBlock(
                values=outp,
                samples=labels.to(device=device),
                components=[],
                properties=Labels(
                    ["_"],
                    torch.arange(outp.shape[1], device=device).reshape(-1, 1),
                ).to(device=device),
            )
        ],
    )


def standardize_featurizer_input_tensor(
    name: str, tensor: torch.Tensor
) -> torch.Tensor:
    """
    Standardize special-case tensors for capturing and wrapping into a TensorMap by
    :func:`create_diagnostic_feature_tensormap`. The arrays need modifying are typically
    the raw featurizer inputs from ``systems_to_batch``, which have non-standard shapes
    that don't conform to the standard node-like or edge-like conventions. integer and
    boolean tensors are cast to ``float``.

    Special cases are as follows (name: raw shape -> standardized shape):

        * element_indices_nodes: (n_atoms,) -> (n_atoms, 1)
        * element_indices_neighbors: (n_atoms, max_nbrs) -> (n_atoms, max_nbrs, 1)
        * edge_distances: (n_atoms, max_nbrs) -> (n_atoms, max_nbrs, 1)
        * reverse_neighbor_index: (n_atoms, max_nbrs) -> (n_atoms, max_nbrs, 1)
        * padding_mask: (n_atoms, max_nbrs) -> (n_atoms, max_nbrs, 1)`` float  [edge]
        * cutoff_factors: (n_atoms, max_nbrs) -> (n_atoms, max_nbrs, 1)``  [edge]
        * edge_vectors: already in the correct shape (n_atoms, max_nbrs, 3)

    :param name: the featurizer-input name.
    :param tensor: the raw tensor from ``systems_to_batch``.
    :return: normalized tensor ready for :func:`create_diagnostic_feature_tensormap`.
    :raises ValueError: If ``name`` is not a recognised featurizer-input name.
    """
    if name == "element_indices_nodes":
        return tensor.unsqueeze(-1).float()

    elif name == "element_indices_neighbors":
        return tensor.unsqueeze(-1).float()

    elif name == "edge_vectors":
        return tensor

    elif name == "edge_distances":
        return tensor.unsqueeze(-1)

    elif name == "reverse_neighbor_index":
        return tensor.unsqueeze(-1).float()

    elif name == "padding_mask":
        return tensor.float().unsqueeze(-1)

    elif name == "cutoff_factors":
        return tensor.unsqueeze(-1)

    else:
        raise ValueError(
            f"'{name}' is not a recognised featurizer-input name. "
            f"Known names: {sorted(FEATURIZER_INPUT_NAMES)}.  "
            "Add an explicit branch with shape documentation if this tensor "
            "should be capturable."
        )


def prepare_diagnostic_handles(
    model: torch.nn.Module,
    outputs: Dict[str, ModelOutput],
    return_dict: Dict[str, Any],
    centers: torch.Tensor,
    nef_to_edges_neighbor: torch.Tensor,
    sample_labels: Labels,
    pair_sample_labels: Labels,
) -> List[Any]:
    """
    Register forward hooks that capture the output of arbitrary named sub-modules as
    diagnostic :class:`~metatensor.torch.TensorMap` objects.

    **Discovering valid module paths**

    A user can find capturable module names by printing ``repr(model)``. Every
    sub-module name visible in the repr (using PyTorch's dotted ``named_modules()``
    convention) can be requested as::

        outputs = {
            "mtt::features::<module_path>": ModelOutput(per_atom=True), ...
        }

    For example, after seeing ``node_heads`` → ``energy`` → ``0`` in the repr, the user
    can request ``"mtt::features::node_heads.energy.0"``.

    **Modules that return tuples**

    :class:`~metatrain.pet.modules.transformer.CartesianTransformer` and
    :class:`~metatrain.pet.modules.transformer.TransformerLayer` return a
    ``(node_features, edge_features)`` tuple.  To capture only one element of the tuple,
    append ``_node`` or ``_edge`` to the module path::

        "mtt::features::gnn_layers.0_node"  # node features from layer 0

        "mtt::features::gnn_layers.0_edge"  # edge features from layer 0

    The suffix is tried *after* an exact module-path lookup, so it only applies when no
    module with the literal name ``<path>_node`` / ``<path>_edge`` exists.

    **Raw featurizer inputs**

    The tensors listed in :data:`FEATURIZER_INPUT_NAMES` (``edge_vectors``,
    ``padding_mask``, etc.) are plain tensors, not module outputs. They are handled
    separately in the caller and are skipped here.

    :param model: The :class:`torch.nn.Module` whose sub-modules will be hooked.
        Typically ``self`` inside ``PET.forward``.
    :param outputs: Requested outputs dict (from ``forward``).
    :param return_dict: The dict that will be returned by ``forward``; hooks write their
        results into this dict.
    :param centers: Flat center-atom indices ``(n_edges,)``.
    :param nef_to_edges_neighbor: NEF-slot indices ``(n_edges,)``.
    :param sample_labels: Per-atom labels.
    :param pair_sample_labels: Per-edge labels.
    :return: List of :class:`torch.utils.hooks.RemovableHandle` objects that must be
        removed after the forward pass.
    :raises AttributeError: If a requested module path cannot be resolved.
    """
    diagnostic_handles: List[Any] = []

    # Build a lookup dict once so resolution is O(1) per requested key.
    named_modules_dict = dict(model.named_modules())

    def make_hook(resolved_path: str, suffix: str) -> Any:
        def _hook(
            module: torch.nn.Module,
            inp: Any,
            outp: Any,
        ) -> None:
            if isinstance(outp, tuple):
                if suffix not in ("_node", "_edge"):
                    raise ValueError(
                        f"Module '{resolved_path}' returns a tuple of tensors. "
                        "Append '_node' or '_edge' to the path to select one "
                        "element, e.g. "
                        f"'{DIAGNOSTIC_PREFIX}{resolved_path}_node'."
                    )
                tensor = outp[0] if suffix == "_node" else outp[1]
            else:
                tensor = outp

            return_dict[DIAGNOSTIC_PREFIX + resolved_path + suffix] = (
                create_diagnostic_feature_tensormap(
                    tensor,
                    centers,
                    nef_to_edges_neighbor,
                    sample_labels,
                    pair_sample_labels,
                )
            )

        return _hook

    for output_key in outputs:
        if not output_key.startswith(DIAGNOSTIC_PREFIX):
            continue

        path = output_key[len(DIAGNOSTIC_PREFIX) :]

        # Skip featurizer inputs – handled separately in the forward pass.
        if path in FEATURIZER_INPUT_NAMES:
            continue

        # 1) Exact match
        if path in named_modules_dict:
            module = named_modules_dict[path]
            handle = module.register_forward_hook(make_hook(path, ""))
            diagnostic_handles.append(handle)
            continue

        # 2) Suffix match: special module that returns bothnode and edge features in a
        # tuple
        resolved_path = path
        suffix = ""
        for s in ("_node", "_edge"):
            if path.endswith(s):
                candidate = path[: -len(s)]
                if candidate in named_modules_dict:
                    resolved_path = candidate
                    suffix = s
                    break

        if resolved_path in named_modules_dict:
            module = named_modules_dict[resolved_path]
            handle = module.register_forward_hook(make_hook(resolved_path, suffix))
            diagnostic_handles.append(handle)
            continue

        # 3) No match - raise error
        valid_paths = sorted(
            p
            for p in named_modules_dict
            if p
            and not p.startswith("additive_models")
            and not p.startswith("scaler")
            and not p.startswith("long_range_featurizer")
        )
        raise AttributeError(
            f"Module path '{path}' (from output key '{output_key}') was not "
            "found in the model.  Print repr(model) to see the architecture "
            "and discover valid module paths.  A selection of valid paths: "
            f"{valid_paths[:30]}{'...' if len(valid_paths) > 30 else ''}."
        )

    return diagnostic_handles
