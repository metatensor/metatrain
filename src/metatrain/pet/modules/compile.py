"""Full-graph FX compilation for PET.

Traces the entire PET forward pass (including force/stress computation via
``autograd.grad``) into a single FX graph, then compiles it with
``torch.compile(dynamic=True, fullgraph=True)``.  This gives maximum kernel
fusion, zero compiled/eager boundary crossings, and always uses SDPA
(``scaled_dot_product_attention``) since forces use
``create_graph=False`` (no double backward).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn.utils._named_member_accessor import NamedMemberAccessor

from .utilities import replace_silu_modules


class _PETBatchForward(torch.nn.Module):
    """Thin wrapper whose ``forward()`` delegates to ``pet._forward_from_batch``.

    PET is registered as a submodule so its parameters/buffers are visible
    to ``functional_call`` / ``NamedMemberAccessor``.

    :param pet: The PET model whose ``_forward_from_batch`` is called.
    """

    def __init__(self, pet: torch.nn.Module) -> None:
        super().__init__()
        self.pet = pet

    def forward(
        self,
        element_indices_nodes: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
        padding_mask: torch.Tensor,
        reverse_neighbor_index: torch.Tensor,
        cutoff_factors: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.pet._forward_from_batch(
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
        )


def _make_pet_compiled_forward(
    batch_model: _PETBatchForward,
    param_names: List[str],
    buffer_names: List[str],
    target_names: List[str],
    output_shapes: Dict[str, Dict[str, List[int]]],
    compute_forces: bool,
    compute_stress: bool,
) -> Callable[
    ...,
    Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ],
]:
    """Build the traceable forward function for ``make_fx``.

    The returned function accepts all batch tensors and the model's
    parameters/buffers as positional arguments (required by
    ``make_fx`` with ``functional_call``).  It returns
    ``(per_structure_preds, forces, stress, raw_predictions)``.

    :param batch_model: Wrapper module whose ``forward`` delegates to
        ``pet._forward_from_batch``.
    :param param_names: Ordered parameter names for the batch model.
    :param buffer_names: Ordered buffer names for the batch model.
    :param target_names: Names of the prediction targets.
    :param output_shapes: Mapping of target name to block key to shape.
    :param compute_forces: Whether to include force computation in the graph.
    :param compute_stress: Whether to include stress computation in the graph.
    :return: A callable that can be traced by ``make_fx``.
    """
    n_params = len(param_names)
    accessor = NamedMemberAccessor(batch_model)

    # Identify which target is the energy target (quantity == "energy")
    # For force/stress we need to aggregate per-atom energy to per-structure.
    energy_target_name: Optional[str] = None
    energy_block_key: Optional[str] = None
    pet = batch_model.pet
    for tname in target_names:
        if hasattr(pet, "outputs") and tname in pet.outputs:
            if pet.outputs[tname].quantity == "energy":
                energy_target_name = tname
                # First block key for this target
                energy_block_key = next(iter(output_shapes[tname]))
                break

    if (compute_forces or compute_stress) and energy_target_name is None:
        raise ValueError(
            "Force/stress compilation requested but no energy target found."
        )

    def forward_fn(
        edge_vectors: torch.Tensor,
        element_indices_nodes: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        padding_mask: torch.Tensor,
        reverse_neighbor_index: torch.Tensor,
        cutoff_factors: torch.Tensor,
        system_indices: torch.Tensor,
        neighbor_atom_indices: torch.Tensor,
        n_structures: int,
        *params_and_buffers: torch.Tensor,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        # Swap in the provided params/buffers via NamedMemberAccessor
        params_buffers = {}
        for i, name in enumerate(param_names):
            params_buffers[name] = params_and_buffers[i]
        for i, name in enumerate(buffer_names):
            params_buffers[name] = params_and_buffers[n_params + i]

        orig_values, _ = accessor.swap_tensors_dict(params_buffers, allow_missing=True)

        # Compute edge_distances inside compiled graph (differentiable)
        edge_distances = torch.sqrt((edge_vectors**2).sum(-1) + 1e-15)

        raw_predictions = batch_model(
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
        )

        # Restore original params/buffers
        accessor.swap_tensors_dict(orig_values, allow_missing=True)

        # Aggregate per-atom predictions to per-structure for the energy target
        n_atoms = edge_vectors.shape[0]
        # +1 for padding structure index (scatter needs valid indices)
        n_struct = n_structures + 1

        energy: Optional[torch.Tensor] = None
        forces: Optional[torch.Tensor] = None
        stress: Optional[torch.Tensor] = None

        if energy_target_name is not None and energy_block_key is not None:
            per_atom_energy = raw_predictions[energy_target_name][energy_block_key]
            energy = torch.zeros(
                n_struct, dtype=edge_vectors.dtype, device=edge_vectors.device
            )
            energy.scatter_add_(0, system_indices, per_atom_energy.squeeze(-1))

        if (compute_forces or compute_stress) and energy is not None:
            (dE_dR,) = torch.autograd.grad(
                energy[:n_structures].sum(),
                edge_vectors,
                create_graph=False,
            )
            dE_dR = dE_dR * padding_mask[:, :, None].float()

            if compute_forces:
                # d(E)/d(pos[i]):
                #   as center: -sum_j dE_dR[i, j]
                #   as neighbor: +sum_{(k,j): neighbor_atom=i} dE_dR[k, j]
                grad_as_center = -dE_dR.sum(dim=1)  # [n_atoms, 3]
                flat_dE = dE_dR.reshape(-1, 3)
                flat_idx = neighbor_atom_indices.reshape(-1, 1).expand(-1, 3).long()
                grad_as_neighbor = torch.zeros(
                    n_atoms, 3, dtype=edge_vectors.dtype, device=edge_vectors.device
                )
                grad_as_neighbor.scatter_add_(0, flat_idx, flat_dE)
                forces = grad_as_center + grad_as_neighbor

            if compute_stress:
                # Virial: sigma = (1/V) sum r otimes (dE/dr)
                virial_per_atom = torch.einsum("ema,emb->eab", edge_vectors, dE_dR)
                stress_buf = torch.zeros(
                    n_struct,
                    3,
                    3,
                    dtype=edge_vectors.dtype,
                    device=edge_vectors.device,
                )
                stress_buf.scatter_add_(
                    0,
                    system_indices[:, None, None].expand(-1, 3, 3),
                    virial_per_atom,
                )
                stress = stress_buf[:n_structures]

        if energy is not None:
            energy = energy[:n_structures]

        return energy, forces, stress, raw_predictions

    return forward_fn


def compile_pet_model(
    model: torch.nn.Module,
    train_dataloader: Any,
    compute_forces: bool,
    compute_stress: bool,
) -> Tuple[torch.nn.Module, List[str], List[str]]:
    """Trace and compile the PET model as a single FX graph.

    :param model: The PET model instance.
    :param train_dataloader: A dataloader to get a sample batch for tracing.
    :param compute_forces: Whether force computation is included.
    :param compute_stress: Whether stress computation is included.
    :return: Tuple of (compiled_module, param_names, buffer_names).
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    from metatrain.utils.data import unpack_batch
    from metatrain.utils.transfer import batch_to

    from ..modules.structures import systems_to_batch

    batch_model = _PETBatchForward(model)
    replace_silu_modules(batch_model)

    params = dict(batch_model.named_parameters())
    buffers = dict(batch_model.named_buffers())
    param_names = list(params.keys())
    buffer_names = list(buffers.keys())

    forward_fn = _make_pet_compiled_forward(
        batch_model,
        param_names,
        buffer_names,
        model.target_names,
        model.output_shapes,
        compute_forces,
        compute_stress,
    )

    # Get a sample batch for tracing
    batch = next(iter(train_dataloader))
    systems, _targets, _extra_data = unpack_batch(batch)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    systems, _, _ = batch_to(systems, {}, {}, dtype=dtype, device=device)

    (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        padding_mask,
        reverse_neighbor_index,
        cutoff_factors,
        system_indices,
        neighbor_atom_indices,
        _sample_labels,
    ) = systems_to_batch(
        systems,
        model.requested_nl,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_function,
        model.cutoff_width,
        model.num_neighbors_adaptive,
    )

    n_structures = int(system_indices.max().item()) + 1

    # edge_vectors needs grad for force tracing
    tracing_edge_vectors = edge_vectors.clone().requires_grad_(True)

    logging.info("Tracing PET model with make_fx (symbolic tracing)...")

    old_duck = torch.fx.experimental._config.use_duck_shape
    torch.fx.experimental._config.use_duck_shape = False
    try:
        fx_graph = make_fx(
            forward_fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(
            tracing_edge_vectors,
            element_indices_nodes,
            element_indices_neighbors,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
            system_indices,
            neighbor_atom_indices,
            n_structures,
            *list(params.values()),
            *list(buffers.values()),
        )
    finally:
        torch.fx.experimental._config.use_duck_shape = old_duck

    logging.info("Compiling traced FX graph with torch.compile...")
    compiled = torch.compile(fx_graph, dynamic=True, fullgraph=True)

    return compiled, param_names, buffer_names
