"""GapPET: a size-intensive HOMO-LUMO gap predictor on top of the PET backbone.

The model wraps :class:`metatrain.pet.PET` (in ``residual`` featurizer mode) and
replaces its summation readout with two heads (HOMO, LUMO) followed by an
extremal log-sum-exp pool. The result is a per-system gap energy that does not
scale with system size, suitable for excited-state surrogates in molecular
dynamics.

The HOMO and LUMO heads use PET's *standard* per-target readout machinery
(``node_heads``, ``edge_heads``, ``node_last_layers``, ``edge_last_layers``):
one MLP per GNN layer, separate node and edge MLPs, summed across layers with
edge contributions weighted by the cutoff factor. Everything is identical to a
standard PET energy head except for the final pool, which is replaced by the
extremal smooth max / smooth min instead of summation.
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

from metatrain.pet.model import PET, get_last_layer_features_name
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from .documentation import ModelHypers


HOMO_PER_ATOM_OUTPUT_NAME = "mtt::aux::homo_per_atom"
LUMO_PER_ATOM_OUTPUT_NAME = "mtt::aux::lumo_per_atom"

# Internal pseudo-target names used to register PET-style heads for the HOMO
# and LUMO per-atom scalar fields. They are *not* exposed through
# ``supported_outputs()``: users only see the gap target and the two per-atom
# auxiliary outputs.
_HOMO_INTERNAL_KEY = "__gap_pet_homo_internal__"
_LUMO_INTERNAL_KEY = "__gap_pet_lumo_internal__"


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
        # silently; PET's default ``feedforward`` only ever exposes the last
        # layer, which is wrong for GapPET.

        if len(dataset_info.targets) != 1:
            raise ValueError(
                f"GapPET expects exactly one target (the gap energy), got "
                f"{list(dataset_info.targets.keys())}."
            )
        self._gap_target_name = next(iter(dataset_info.targets))

        # PET reads only the keys it knows; the extra ``pooling`` key is ignored.
        backbone_hypers = dict(hypers)
        backbone_hypers["featurizer_type"] = "residual"
        super().__init__(backbone_hypers, dataset_info)

        # Restore the gap hypers on self.hypers so that get_checkpoint round-trips.
        self.hypers = hypers

        # PET's ``__init__`` registered standard energy heads + last layers for
        # the gap target. We don't use them (the gap is read out by extremal
        # pooling on the internal HOMO/LUMO heads instead), so remove them to
        # avoid wasted compute and parameters.
        self._unregister_pet_target(self._gap_target_name)

        # Register two *internal* pseudo-targets, one for HOMO and one for LUMO.
        # Each gets the standard PET readout: per-layer node + edge MLPs and a
        # per-layer linear projection to a scalar. The user-visible
        # ``self.outputs`` entries that ``_add_output`` would create are
        # immediately removed -- the heads are an implementation detail of the
        # gap readout, not user outputs.
        unit = dataset_info.targets[self._gap_target_name].unit or ""
        synth_target_info = get_energy_target_info(
            "__synth__",
            {"quantity": "energy", "unit": unit},
            add_position_gradients=False,
        )
        self._add_output(_HOMO_INTERNAL_KEY, synth_target_info)
        self._add_output(_LUMO_INTERNAL_KEY, synth_target_info)
        for key in (_HOMO_INTERNAL_KEY, _LUMO_INTERNAL_KEY):
            self.outputs.pop(key, None)
            self.outputs.pop(get_last_layer_features_name(key), None)

        # Pooling alphas: fixed scalar buffers (move with .to(device) and
        # round-trip through state_dict, but are not optimised).
        pooling_hypers = hypers["pooling"]
        self.register_buffer(
            "alpha_homo", torch.tensor(float(pooling_hypers["alpha_homo"]))
        )
        self.register_buffer(
            "alpha_lumo", torch.tensor(float(pooling_hypers["alpha_lumo"]))
        )

        # Per-atom auxiliary outputs (interpretability).
        target_info = dataset_info.targets[self._gap_target_name]
        self.outputs[HOMO_PER_ATOM_OUTPUT_NAME] = ModelOutput(
            quantity="energy",
            unit=target_info.unit,
            per_atom=True,
            description="Per-atom HOMO contribution h_i^HOMO",
        )
        self.outputs[LUMO_PER_ATOM_OUTPUT_NAME] = ModelOutput(
            quantity="energy",
            unit=target_info.unit,
            per_atom=True,
            description="Per-atom LUMO contribution h_i^LUMO",
        )

        # The gap target itself is intrinsically per-system (intensive), so we
        # register it as ``per_atom=False`` -- the trainer must not divide by
        # ``N_atoms`` when computing the loss. (Also requires the user's YAML
        # to list it under ``per_structure_targets``.)
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

    def _unregister_pet_target(self, target_name: str) -> None:
        """Remove every trace of a target previously added via ``_add_output``.

        Used in ``__init__`` to drop the gap target's PET-style energy heads
        (which we replace with the internal HOMO/LUMO heads + pooling).

        ``torch.nn.ModuleDict.pop`` does not accept a default argument, so we
        guard each ``del`` with an ``in`` check.
        """
        for module_dict in (
            self.node_heads,
            self.edge_heads,
            self.node_last_layers,
            self.edge_last_layers,
        ):
            if target_name in module_dict:
                del module_dict[target_name]
        self.output_shapes.pop(target_name, None)
        self.key_labels.pop(target_name, None)
        self.property_labels.pop(target_name, None)
        self.component_labels.pop(target_name, None)
        self.last_layer_parameter_names.pop(target_name, None)
        if target_name in self.target_names:
            self.target_names.remove(target_name)
        self.outputs.pop(target_name, None)
        self.outputs.pop(get_last_layer_features_name(target_name), None)

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
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

        # Stage 2: PET-standard last-layer features per readout layer, for
        # both internal heads. ``_calculate_last_layer_features`` iterates
        # ``self.node_heads`` and ``self.edge_heads`` -- both internal targets
        # are registered there, so this gives us
        #   node_ll_dict[KEY] = [Tensor(N, d_head)] * L
        #   edge_ll_dict[KEY] = [Tensor(N, M, d_head)] * L
        node_ll_dict, edge_ll_dict = self._calculate_last_layer_features(
            node_features_list, edge_features_list
        )

        # Stage 3: PET-standard atomic predictions per readout layer per block.
        # We pass our internal targets in ``internal_outputs`` so
        # ``_calculate_atomic_predictions`` actually computes them. Each
        # internal target has a single block (synthesised with a single scalar
        # property), so the inner block list has length 1.
        internal_outputs = {
            _HOMO_INTERNAL_KEY: ModelOutput(per_atom=True),
            _LUMO_INTERNAL_KEY: ModelOutput(per_atom=True),
        }
        node_apr_dict, edge_apr_dict = self._calculate_atomic_predictions(
            node_ll_dict,
            edge_ll_dict,
            padding_mask,
            cutoff_factors,
            internal_outputs,
        )

        # Sum across GNN layers (and across the single block) to get per-atom
        # scalars -- this is exactly what PET's ``_get_output_atomic_predictions``
        # does for an energy-style scalar target.
        h_homo = self._sum_per_atom_scalar(node_apr_dict, edge_apr_dict, _HOMO_INTERNAL_KEY)
        h_lumo = self._sum_per_atom_scalar(node_apr_dict, edge_apr_dict, _LUMO_INTERNAL_KEY)
        # h_homo, h_lumo: (N,)

        # Stage 4: extremal pooling -> per-system scalars.
        num_systems = len(systems)
        e_homo = _scatter_logsumexp(h_homo, self.alpha_homo, system_indices, num_systems)
        e_lumo = _scatter_logsumexp(h_lumo, self.alpha_lumo, system_indices, num_systems)
        e_gap = e_lumo - e_homo  # (S,)

        return_dict: Dict[str, TensorMap] = {}

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
                from metatrain.utils.sum_over_atoms import sum_over_atoms

                tmap = sum_over_atoms(tmap)
            return_dict[aux_name] = tmap

        # Post-processing (eval only): reapply scaler. Composition is empty, so
        # the additive loop is a no-op for the gap target.
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
                    return_dict[name] = additive_contributions[name].to(
                        device=return_dict[name].device,
                        dtype=return_dict[name].dtype,
                    )

        return return_dict

    @staticmethod
    def _sum_per_atom_scalar(
        node_apr_dict: Dict[str, List[List[torch.Tensor]]],
        edge_apr_dict: Dict[str, List[List[torch.Tensor]]],
        target_name: str,
    ) -> torch.Tensor:
        """Sum the per-layer, per-block atomic predictions for a scalar target
        and return a flat ``(N,)`` per-atom tensor.

        For our internal HOMO/LUMO targets each ``inner`` block list has length
        one (single scalar block, single property), so we just unwrap and sum.
        """
        node_layers = node_apr_dict[target_name]  # List[List[Tensor]]: L x B
        edge_layers = edge_apr_dict[target_name]
        per_atom: Optional[torch.Tensor] = None
        for node_blocks, edge_blocks in zip(node_layers, edge_layers, strict=True):
            # Single-block scalar: sum the (only) block from node and edge.
            layer_contrib = node_blocks[0] + edge_blocks[0]  # (N, 1)
            per_atom = layer_contrib if per_atom is None else per_atom + layer_contrib
        assert per_atom is not None, "GapPET requires at least one GNN layer."
        return per_atom.squeeze(-1)

    # --- restart ----------------------------------------------------------

    def restart(self, dataset_info: DatasetInfo) -> "GapPET":
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
