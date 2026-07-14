"""GapPET: a size-intensive HOMO-LUMO gap predictor on top of the PET backbone.

The model wraps :class:`metatrain.pet.PET` (in ``residual`` featurizer mode) and
replaces its summation readout with two heads (HOMO, LUMO) followed by an
intensive pool. The result is a per-system gap energy that does not scale with
system size, suitable for excited-state surrogates in molecular dynamics.

Two pooling schemes are available (see :mod:`metatrain.gap_pet.documentation`):
``"smoothmax"`` (an extremal log-sum-exp / smooth max-min, the default) and
``"softmax"`` (a strictly-intensive self-weighted softmax pool). The predicted
gap is ``E_gap = E_LUMO - E_HOMO``.

The HOMO and LUMO heads use PET's *standard* per-target readout machinery
(``node_heads``, ``edge_heads``, ``node_last_layers``, ``edge_last_layers``):
one MLP per GNN layer, separate node and edge MLPs, summed across layers with
edge contributions weighted by the cutoff factor. Everything is identical to a
standard PET energy head except for the final pool, which replaces summation.

Multiple gap targets (e.g. several excitation levels computed on the same
geometries) are supported: each target gets its own HOMO/LUMO head pair on top
of the shared backbone. The pooling parameters ``alpha_homo``/``alpha_lumo``
are *shared* across targets, since the pool sharpness reflects how localized
the frontier orbitals are -- a property of the system, not of which excitation
is being read out.
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
from metatrain.utils.data.target_info import TargetInfo, get_energy_target_info
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.sum_over_atoms import sum_over_atoms

from .documentation import ModelHypers


# Prefixes for the internal pseudo-target names carrying the per-target HOMO
# and LUMO heads. They are not user-visible outputs.
HOMO_HEAD_PREFIX = "__gap_pet_homo__"
LUMO_HEAD_PREFIX = "__gap_pet_lumo__"


def homo_per_atom_output_name(target_name: str) -> str:
    """Name of the per-atom HOMO auxiliary output for ``target_name``."""
    return _aux_output_name(target_name, "homo")


def lumo_per_atom_output_name(target_name: str) -> str:
    """Name of the per-atom LUMO auxiliary output for ``target_name``."""
    return _aux_output_name(target_name, "lumo")


def _aux_output_name(target_name: str, which: str) -> str:
    # metatomic requires non-standard outputs to live under the ``mtt::``
    # namespace, so strip a leading ``mtt::`` from the target before nesting it
    # under ``mtt::aux::`` rather than emitting ``mtt::aux::mtt::...``.
    short_name = target_name
    if short_name.startswith("mtt::"):
        short_name = short_name[len("mtt::") :]
    return f"mtt::aux::{short_name}_{which}_per_atom"


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
    sys_max = torch.where(torch.isinf(sys_max), torch.zeros_like(sys_max), sys_max)
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

    __checkpoint_version__ = 2
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "https://arxiv.org/abs/2305.19302v3",  # PET backbone
            ]
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        # GapPET reads out *all* GNN layers via the HOMO/LUMO heads, so it works
        # best with the ``residual`` featurizer; ``feedforward`` exposes only the
        # last layer. GapPET therefore *defaults* ``featurizer_type`` to
        # ``residual`` (see documentation.ModelHypers), but the configured value
        # is honored, so ``featurizer_type: feedforward`` can be requested
        # explicitly (e.g. to ablate the two featurizers).

        if len(dataset_info.targets) == 0:
            raise ValueError("GapPET expects at least one gap target, got none.")

        # PET reads only the keys it knows; the extra ``pooling`` key is ignored.
        # ``featurizer_type`` is passed through as configured; if it is somehow
        # absent we fall back to GapPET's ``residual`` default.
        backbone_hypers = dict(hypers)
        backbone_hypers.setdefault("featurizer_type", "residual")
        super().__init__(backbone_hypers, dataset_info)

        # Restore the gap hypers on self.hypers so that get_checkpoint round-trips,
        # recording the featurizer actually used.
        self.hypers = dict(hypers)
        self.hypers.setdefault("featurizer_type", "residual")

        # Pooling configuration. Shared by every gap target.
        pooling_hypers = hypers["pooling"]
        self._pooling_type: str = str(pooling_hypers.get("type", "smoothmax"))
        if self._pooling_type not in ("smoothmax", "softmax"):
            raise ValueError(
                f"Unknown pooling type {self._pooling_type!r}; "
                "expected 'smoothmax' or 'softmax'."
            )
        # ``alpha_homo``/``alpha_lumo`` parametrize both pools: the sign selects
        # max- vs min-pool and the magnitude the sharpness. ``alpha_homo > 0``
        # (smooth/soft max for E_HOMO) and ``alpha_lumo < 0`` (smooth/soft min
        # for E_LUMO).
        self.register_buffer(
            "alpha_homo", torch.tensor(float(pooling_hypers["alpha_homo"]))
        )
        self.register_buffer(
            "alpha_lumo", torch.tensor(float(pooling_hypers["alpha_lumo"]))
        )

        # Per-target bookkeeping. These are plain ``Dict[str, str]`` attributes
        # rather than f-strings computed on the fly so that ``forward`` stays
        # TorchScript-compatible.
        self._gap_target_names: List[str] = []
        self._homo_head_keys: Dict[str, str] = {}
        self._lumo_head_keys: Dict[str, str] = {}
        self._homo_aux_names: Dict[str, str] = {}
        self._lumo_aux_names: Dict[str, str] = {}

        # PET's ``__init__`` registered standard energy heads + last layers for
        # each gap target. We don't use them (the gap is read out by extremal
        # pooling on the internal HOMO/LUMO heads instead), so remove them to
        # avoid wasted compute and parameters, then register our own readout.
        for target_name in list(dataset_info.targets):
            self._unregister_pet_target(target_name)
            self._register_gap_target(target_name, dataset_info.targets[target_name])

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

    def _register_gap_target(self, target_name: str, target_info: TargetInfo) -> None:
        """Register the GapPET readout for one gap target.

        Creates a pair of *internal* pseudo-targets (HOMO and LUMO), each with
        the standard PET readout: per-layer node + edge MLPs and a per-layer
        linear projection to a scalar. The user-visible ``self.outputs`` entries
        that ``_add_output`` creates for them are immediately removed -- the
        heads are an implementation detail of the gap readout, not user outputs.
        """
        homo_key = HOMO_HEAD_PREFIX + target_name
        lumo_key = LUMO_HEAD_PREFIX + target_name

        synth_target_info = get_energy_target_info(
            "__synth__",
            {"quantity": "energy", "unit": target_info.unit or ""},
            add_position_gradients=False,
        )
        for head_key in (homo_key, lumo_key):
            self._add_output(head_key, synth_target_info)
            self.outputs.pop(head_key, None)
            self.outputs.pop(get_last_layer_features_name(head_key), None)

        homo_aux_name = homo_per_atom_output_name(target_name)
        lumo_aux_name = lumo_per_atom_output_name(target_name)

        self._gap_target_names.append(target_name)
        self._homo_head_keys[target_name] = homo_key
        self._lumo_head_keys[target_name] = lumo_key
        self._homo_aux_names[target_name] = homo_aux_name
        self._lumo_aux_names[target_name] = lumo_aux_name

        # Per-atom auxiliary outputs (interpretability).
        self.outputs[homo_aux_name] = ModelOutput(
            quantity="energy",
            unit=target_info.unit,
            per_atom=True,
            description=f"Per-atom HOMO contribution h_i^HOMO for {target_name}",
        )
        self.outputs[lumo_aux_name] = ModelOutput(
            quantity="energy",
            unit=target_info.unit,
            per_atom=True,
            description=f"Per-atom LUMO contribution h_i^LUMO for {target_name}",
        )

        # The gap target itself is intrinsically per-system (intensive), so we
        # register it as ``per_atom=False`` -- the trainer must not divide by
        # ``N_atoms`` when computing the loss. (Also requires the user's YAML
        # to list it under ``per_structure_targets``.)
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity or "energy",
            unit=target_info.unit,
            per_atom=False,
            description=target_info.description
            or "Per-system electronic gap (intensive readout)",
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
        #
        # Only compute the head pairs whose gap target (or per-atom auxiliary)
        # was actually requested: with several excitation levels registered,
        # evaluating every head on every call would be wasted compute.
        requested_targets: List[str] = []
        internal_outputs: Dict[str, ModelOutput] = {}
        for target_name in self._gap_target_names:
            if (
                target_name in outputs
                or self._homo_aux_names[target_name] in outputs
                or self._lumo_aux_names[target_name] in outputs
            ):
                requested_targets.append(target_name)
                internal_outputs[self._homo_head_keys[target_name]] = ModelOutput(
                    per_atom=True
                )
                internal_outputs[self._lumo_head_keys[target_name]] = ModelOutput(
                    per_atom=True
                )

        return_dict: Dict[str, TensorMap] = {}
        if len(requested_targets) == 0:
            return return_dict

        node_apr_dict, edge_apr_dict = self._calculate_atomic_predictions(
            node_ll_dict,
            edge_ll_dict,
            padding_mask,
            cutoff_factors,
            internal_outputs,
        )

        num_systems = len(systems)
        system_labels = Labels(
            names=["system"],
            values=torch.arange(num_systems, device=device).reshape(-1, 1),
            assume_unique=True,
        )
        energy_labels = Labels(
            names=["energy"],
            values=torch.zeros((1, 1), dtype=torch.int64, device=device),
            assume_unique=True,
        )

        for target_name in requested_targets:
            # Sum across GNN layers (and across the single block) to get per-atom
            # scalars -- this is exactly what PET's
            # ``_get_output_atomic_predictions`` does for an energy-style scalar
            # target. h_homo, h_lumo: (N,)
            h_homo = self._sum_per_atom_scalar(
                node_apr_dict, edge_apr_dict, self._homo_head_keys[target_name]
            )
            h_lumo = self._sum_per_atom_scalar(
                node_apr_dict, edge_apr_dict, self._lumo_head_keys[target_name]
            )

            # Stage 4: pool per-atom contributions -> per-system scalars.
            if self._pooling_type == "softmax":
                # Self-weighted softmax pool: the softmax weights are computed
                # directly from the per-atom values themselves, so atoms with the
                # most extreme contribution dominate. Strictly intensive (weights
                # sum to 1) and recovers a hard max/min as ``|alpha| -> infinity``.
                e_homo = _scatter_softmax_pool(
                    h_homo, self.alpha_homo, system_indices, num_systems
                )
                e_lumo = _scatter_softmax_pool(
                    h_lumo, self.alpha_lumo, system_indices, num_systems
                )
            else:
                e_homo = _scatter_logsumexp(
                    h_homo, self.alpha_homo, system_indices, num_systems
                )
                e_lumo = _scatter_logsumexp(
                    h_lumo, self.alpha_lumo, system_indices, num_systems
                )
            e_gap = e_lumo - e_homo  # (S,)

            if target_name in outputs:
                gap_block = TensorBlock(
                    values=e_gap.reshape(-1, 1),
                    samples=system_labels,
                    components=[],
                    properties=energy_labels,
                )
                return_dict[target_name] = TensorMap(
                    keys=self.single_label, blocks=[gap_block]
                )

            for aux_name, aux_values in (
                (self._homo_aux_names[target_name], h_homo),
                (self._lumo_aux_names[target_name], h_lumo),
            ):
                if aux_name in outputs:
                    block = TensorBlock(
                        values=aux_values.reshape(-1, 1),
                        samples=sample_labels,
                        components=[],
                        properties=energy_labels,
                    )
                    tmap = TensorMap(keys=self.single_label, blocks=[block])
                    if not outputs[aux_name].per_atom:
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
                if outputs_for_additive_model:
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
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        if new_atomic_types:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "GapPET does not support adding new atomic types."
            )

        # Any target not already known gets its own freshly-initialized
        # HOMO/LUMO head pair on top of the shared (pre-trained) backbone. This
        # is what makes multi-head finetuning work: point ``read_from`` at a
        # single-gap checkpoint and list several gap targets in the training
        # set, optionally seeding the new heads via ``inherit_heads``.
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0
        for target_name, target_info in new_targets.items():
            self._register_gap_target(target_name, target_info)

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
        version = checkpoint["model_ckpt_version"]
        if version == cls.__checkpoint_version__:
            return checkpoint

        if version == 1:
            # v1 held a single, target-agnostic head pair
            # (``__gap_pet_homo_internal__`` / ``__gap_pet_lumo_internal__``).
            # v2 keys the heads by target name so several gap targets can
            # coexist, so rename the v1 heads onto the checkpoint's only target.
            target_names = list(checkpoint["model_data"]["dataset_info"].targets)
            if len(target_names) != 1:
                raise RuntimeError(
                    "Unable to upgrade the v1 GapPET checkpoint: expected exactly "
                    f"one target, found {target_names}."
                )
            renames = {
                "__gap_pet_homo_internal__": HOMO_HEAD_PREFIX + target_names[0],
                "__gap_pet_lumo_internal__": LUMO_HEAD_PREFIX + target_names[0],
            }
            for state_dict_key in ("model_state_dict", "best_model_state_dict"):
                state_dict = checkpoint.get(state_dict_key)
                if state_dict is None:
                    continue
                # Rebuild in iteration order: ``load_checkpoint`` reads the dtype
                # off the second entry, so the ordering is load-bearing.
                upgraded = {}
                for name, value in state_dict.items():
                    for old_key, new_key in renames.items():
                        if old_key in name:
                            name = name.replace(old_key, new_key)
                            break
                    upgraded[name] = value
                checkpoint[state_dict_key] = upgraded
            checkpoint["model_ckpt_version"] = 2
            return checkpoint

        raise RuntimeError(
            "Unable to upgrade the checkpoint: the checkpoint is using model "
            f"version {version}, while the current "
            f"GapPET version is {cls.__checkpoint_version__}."
        )
