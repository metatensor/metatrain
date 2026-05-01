"""GapPET: a size-intensive HOMO-LUMO gap predictor on top of the PET backbone.

The model wraps :class:`metatrain.pet.PET` (in ``residual`` featurizer mode) and
replaces its summation readout with two heads (HOMO, LUMO) followed by an
extremal log-sum-exp pool. The result is a per-system gap energy that does not
scale with system size, suitable for excited-state surrogates in molecular
dynamics.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

from metatrain.pet.model import PET
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from .documentation import ModelHypers


HOMO_PER_ATOM_OUTPUT_NAME = "mtt::aux::homo_per_atom"
LUMO_PER_ATOM_OUTPUT_NAME = "mtt::aux::lumo_per_atom"


class _ScalarMLP(torch.nn.Module):
    """Two-layer MLP projecting a feature vector to a scalar.

    ``Linear(in_dim, hidden) -> SiLU -> Linear(hidden, 1)``.

    Applied identically to per-atom features ``(N, in_dim)`` or per-edge features
    ``(N, M, in_dim)``; the output drops the trailing dimension of size 1.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _HomoLumoHead(torch.nn.Module):
    """One head (HOMO or LUMO): per-head node MLP and edge MLP, shared across
    all GNN layers and summed.

    For each atom ``i``::

        h_i = sum_l [ MLP_node(g_i^l) + sum_{j in N(i)} cutoff_ij * MLP_edge(f_ij^l) ]

    where the cutoff factor is the same one applied throughout PET, ensuring the
    output is a smooth function of atomic positions even when neighbours enter
    or leave the cutoff sphere.
    """

    def __init__(self, d_node: int, d_edge: int, d_head: int) -> None:
        super().__init__()
        self.node_mlp = _ScalarMLP(d_node, d_head)
        self.edge_mlp = _ScalarMLP(d_edge, d_head)

    def forward(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        cutoff_factors: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param node_features_list: ``L`` tensors of shape ``(N, d_node)``.
        :param edge_features_list: ``L`` tensors of shape ``(N, M, d_edge)``.
        :param cutoff_factors: ``(N, M)`` smooth cutoff factor (already zeroed
            at padded positions).
        :return: per-atom scalar field, shape ``(N,)``.
        """
        h: Optional[torch.Tensor] = None
        for g_l in node_features_list:
            term = self.node_mlp(g_l)  # (N,)
            h = term if h is None else h + term
        for f_l in edge_features_list:
            edge_scalar = self.edge_mlp(f_l)  # (N, M)
            term = (edge_scalar * cutoff_factors).sum(dim=1)  # (N,)
            h = term if h is None else h + term
        assert h is not None  # at least one of the lists is non-empty
        return h


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
    # Defensive: if any system has no atoms (shouldn't happen for atomistic
    # systems), the max stays at -inf and the subtraction below is undefined.
    # Replace those with zero; the corresponding sum_exp will be zero too,
    # leading to log(0) = -inf, which signals a malformed input clearly.
    sys_max = torch.where(
        torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max
    )
    shifted_exp = torch.exp(scaled - sys_max[system_indices])
    sum_exp = torch.zeros(
        num_systems, dtype=values.dtype, device=values.device
    ).scatter_add(0, system_indices, shifted_exp)
    log_sum_exp = sys_max + torch.log(sum_exp)
    return log_sum_exp / alpha


class GapPET(PET):
    """Size-intensive HOMO-LUMO gap predictor.

    See :mod:`metatrain.gap_pet.documentation` for the full description.
    """

    __checkpoint_version__ = 1
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "https://arxiv.org/abs/2305.19302v3",  # PET backbone
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        # GapPET requires the residual featurizer (all GNN layers are read out
        # by the heads). We always override ``featurizer_type`` to ``residual``
        # silently; the user-facing default ``feedforward`` is the right default
        # for PET but never for GapPET.

        # GapPET expects exactly one target (the gap energy).
        if len(dataset_info.targets) != 1:
            raise ValueError(
                f"GapPET expects exactly one target (the gap energy), got "
                f"{list(dataset_info.targets.keys())}."
            )
        self._gap_target_name = next(iter(dataset_info.targets))

        # We pass the same dict to the parent (PET reads only the keys it
        # knows; the extra `head` and `pooling` keys are ignored).
        backbone_hypers = dict(hypers)
        backbone_hypers["featurizer_type"] = "residual"
        super().__init__(backbone_hypers, dataset_info)

        # Restore the gap hypers on self.hypers so that get_checkpoint round-trips.
        self.hypers = hypers

        # Build the heads.
        head_hypers = hypers["head"]
        d_head = head_hypers["d_head"]
        d_head_homo = head_hypers.get("d_head_homo") or d_head
        d_head_lumo = head_hypers.get("d_head_lumo") or d_head
        self.homo_head = _HomoLumoHead(self.d_node, self.d_pet, d_head_homo)
        self.lumo_head = _HomoLumoHead(self.d_node, self.d_pet, d_head_lumo)

        # Pooling alphas: fixed scalar buffers (so they move with .to(device)
        # and round-trip through state_dict, but are not optimised).
        pooling_hypers = hypers["pooling"]
        self.register_buffer(
            "alpha_homo", torch.tensor(float(pooling_hypers["alpha_homo"]))
        )
        self.register_buffer(
            "alpha_lumo", torch.tensor(float(pooling_hypers["alpha_lumo"]))
        )

        # Register the per-atom auxiliary outputs.
        self.outputs[HOMO_PER_ATOM_OUTPUT_NAME] = ModelOutput(
            quantity="energy",
            unit=dataset_info.targets[self._gap_target_name].unit,
            per_atom=True,
            description="Per-atom HOMO contribution h_i^HOMO",
        )
        self.outputs[LUMO_PER_ATOM_OUTPUT_NAME] = ModelOutput(
            quantity="energy",
            unit=dataset_info.targets[self._gap_target_name].unit,
            per_atom=True,
            description="Per-atom LUMO contribution h_i^LUMO",
        )

        # Override the gap target's ModelOutput: the gap is intrinsically per
        # system (intensive), not summed-over-atoms.
        target_info = dataset_info.targets[self._gap_target_name]
        self.outputs[self._gap_target_name] = ModelOutput(
            quantity=target_info.quantity or "energy",
            unit=target_info.unit,
            per_atom=False,
            description=target_info.description
            or "Per-system electronic gap (intensive readout)",
        )

        # Replace the composition model with one that has no targets: the gap
        # is intensive and a per-atomic-type linear baseline is the wrong
        # inductive bias. The trainer iterates self.additive_models, so we
        # keep the list non-empty but with an empty-target model that is a
        # no-op at both training and evaluation.
        empty_dataset_info = DatasetInfo(
            length_unit=dataset_info.length_unit,
            atomic_types=dataset_info.atomic_types,
            targets={},
        )
        self.additive_models = torch.nn.ModuleList(
            [CompositionModel(hypers={}, dataset_info=empty_dataset_info)]
        )
        # ZBL was already added if hypers["zbl"]; for the gap target it is also
        # not meaningful, so remove it. (Default is False, so this is usually
        # a no-op.)
        # (No-op: only the composition model remains in additive_models.)

        # Pre-build sample labels prototypes. The per-system labels are built on
        # the fly in forward() since the number of systems varies.

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
        dtype = systems[0].positions.dtype
        nl_options = self.requested_neighbor_lists()[0]

        if self.single_label.values.device != device:
            self._move_labels_to_device(device)

        # Stage 0: build the batch tensors.
        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
            system_indices,
            sample_labels,
            _species,
        ) = systems_to_batch(
            systems,
            nl_options,
            self.atomic_types,
            self.species_to_species_index,
            self.cutoff_function,
            self.cutoff_width,
            self.num_neighbors_adaptive,
        )

        use_manual_attention = edge_vectors.requires_grad and self.training

        # Stage 1: backbone features (residual mode -> all L layers).
        featurizer_inputs: Dict[str, torch.Tensor] = {
            "element_indices_nodes": element_indices_nodes,
            "element_indices_neighbors": element_indices_neighbors,
            "edge_vectors": edge_vectors,
            "edge_distances": edge_distances,
            "reverse_neighbor_index": reverse_neighbor_index,
            "padding_mask": padding_mask,
            "cutoff_factors": cutoff_factors,
        }
        node_features_list, edge_features_list = self._calculate_features(
            featurizer_inputs, use_manual_attention=use_manual_attention
        )
        if self.long_range:
            long_range_features = self._calculate_long_range_features(
                systems, node_features_list, edge_distances, padding_mask
            )
            for i in range(self.num_readout_layers):
                node_features_list[i] = (
                    node_features_list[i] + long_range_features
                ) * 0.5**0.5

        # Stage 2: HOMO/LUMO heads -> per-atom scalars.
        h_homo = self.homo_head(node_features_list, edge_features_list, cutoff_factors)
        h_lumo = self.lumo_head(node_features_list, edge_features_list, cutoff_factors)
        # h_homo, h_lumo: (N,)

        # Stage 3: extremal pooling -> per-system scalars.
        num_systems = len(systems)
        e_homo = _scatter_logsumexp(h_homo, self.alpha_homo, system_indices, num_systems)
        e_lumo = _scatter_logsumexp(h_lumo, self.alpha_lumo, system_indices, num_systems)
        e_gap = e_lumo - e_homo  # (S,)

        return_dict: Dict[str, TensorMap] = {}

        # Gap output (per-system).
        if self._gap_target_name in outputs:
            gap_block = TensorBlock(
                values=e_gap.reshape(-1, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.arange(num_systems, device=device).reshape(-1, 1),
                    assume_unique=True,
                ),
                components=[],
                properties=Labels(
                    names=["energy"],
                    values=torch.zeros((1, 1), dtype=torch.int64, device=device),
                    assume_unique=True,
                ),
            )
            return_dict[self._gap_target_name] = TensorMap(
                keys=self.single_label, blocks=[gap_block]
            )

        # Per-atom auxiliary outputs.
        for aux_name, aux_values in (
            (HOMO_PER_ATOM_OUTPUT_NAME, h_homo),
            (LUMO_PER_ATOM_OUTPUT_NAME, h_lumo),
        ):
            if aux_name not in outputs:
                continue
            block = TensorBlock(
                values=aux_values.reshape(-1, 1),
                samples=sample_labels,
                components=[],
                properties=Labels(
                    names=["energy"],
                    values=torch.zeros((1, 1), dtype=torch.int64, device=device),
                    assume_unique=True,
                ),
            )
            tmap = TensorMap(keys=self.single_label, blocks=[block])
            if not outputs[aux_name].per_atom:
                # Sum over atoms to get a per-system value (rarely useful, but
                # the framework allows it).
                from metatrain.utils.sum_over_atoms import sum_over_atoms

                tmap = sum_over_atoms(tmap)
            return_dict[aux_name] = tmap

        # Post-processing (eval only): reapply scaler. Composition is empty so
        # has no effect, but iterating over it stays consistent with the PET
        # convention.
        if not self.training:
            return_dict = self.scaler(
                systems, return_dict, selected_atoms=selected_atoms
            )
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                if not outputs_for_additive_model:
                    continue
                additive_contributions = additive_model(
                    systems, outputs_for_additive_model, selected_atoms
                )
                for name in additive_contributions:
                    # Sparse add; gap_pet's composition is empty so this never
                    # triggers in practice. Kept for parity with PET.
                    return_dict[name] = additive_contributions[name].to(
                        device=return_dict[name].device,
                        dtype=return_dict[name].dtype,
                    )
        # (Note: dtype/device unification is handled by the scaler.)
        _ = dtype  # silence linter

        return return_dict

    # --- restart ----------------------------------------------------------

    def restart(self, dataset_info: DatasetInfo) -> "GapPET":
        # GapPET only supports a single target by design; restarting with a
        # different target list is not allowed. Otherwise defer to PET.
        if set(dataset_info.targets) != {self._gap_target_name}:
            raise ValueError(
                "GapPET supports exactly one target (the gap), so restart() "
                "must use the same target as the original training. Got "
                f"{list(dataset_info.targets)} vs "
                f"expected {{'{self._gap_target_name}'}}."
            )
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        if new_atomic_types:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "GapPET does not support adding new atomic types."
            )
        self.has_new_targets = False
        self.dataset_info = merged_info
        self.scaler = self.scaler.restart(self._train_dataset_info(dataset_info))
        return self

    # --- export -----------------------------------------------------------

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        # Same as PET.export, but using GapPET's class attributes.
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for GapPET")
        self.to(dtype)
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)
        interaction_ranges = [self.num_gnn_layers * self.cutoff]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)
        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )
        metadata = merge_metadata(self.metadata, metadata)
        return AtomisticModel(self.eval(), metadata, capabilities)

    # --- checkpoint -------------------------------------------------------

    def get_checkpoint(self) -> Dict[str, Any]:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        return {
            "architecture_name": "gap_pet",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": self.state_dict(),
        }

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "GapPET":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model_data = checkpoint["model_data"]
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        model_state_dict.pop("finetune_config", {})
        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)  # species_to_species_index
        dtype = next(state_dict_iter).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))
        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                "Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current "
                f"GapPET version is {cls.__checkpoint_version__}."
            )
        return checkpoint
