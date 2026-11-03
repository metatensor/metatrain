import copy
import logging
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
    register_autograd_neighbors,
)

from metatrain.utils.abc import ModelInterface, common_upgrade_checkpoint
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.atom_pair_helpers import check_no_atom_pair_targets
from metatrain.utils.data.match_tmaps import match_layout
from metatrain.utils.hypers import raise_if_hypers_mismatch
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .documentation import ModelHypers


class LLPRUncertaintyModel(ModelInterface[ModelHypers]):
    """A wrapper that adds LLPR uncertainties to a model.

    In order to be compatible with this class, the core of a ``MetatrainModel`` needs
    to be capable of returning the last-layer features of its targets (see
    :ref:`mtt-aux-target-last-layer-features`), optionally per atom to calculate LPRs
    (per-atom uncertainties) with the LLPR method.

    All uncertainties provided by this class are standard deviations (as opposed to
    variances). Prediction rigidities (local and total) can be calculated, according to
    their definition, as the inverse of the square of the standard deviations returned
    by this class.

    :param hypers: The hypers of the model.
    :param dataset_info: The dataset info of the targets to wrap.
    """

    __checkpoint_version__ = 7

    ensemble_gradient_outputs: List[str]
    llpr_outputs: List[str]

    # all torch devices and dtypes are supported, if they are supported by the wrapped
    # the check is performed in the trainer
    __supported_devices__ = ["cuda", "cpu", "mps"]
    __supported_dtypes__ = [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    # more to be added if needed

    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "LLPR (uncertainty method): https://iopscience.iop.org/article/10.1088/2632-2153/ad805f",  # noqa: E501
                "LPR (if using per-atom uncertainty): https://pubs.acs.org/doi/10.1021/acs.jctc.3c00704",  # noqa: E501
            ],
        }
    )

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        check_no_atom_pair_targets(dataset_info.targets, self.__class__.__name__)

        self.hypers = hypers
        self.dataset_info = dataset_info

    def set_wrapped_model(self, model: MetatrainModel) -> None:
        """Wrap ``model``, and set up the LLPR state of each of its targets.

        :param model: the model to wrap.
        """
        self.model = model

        # the export of the core converts its dataset info to capabilities, and
        # gives its dtype
        old_capabilities = self.model.core.export().capabilities()
        dtype = getattr(torch, old_capabilities.dtype)

        # covariance, calibration and uncertainties need inference-mode predictions
        # (with scaler and additive contributions)
        self.model.eval()

        if self.dataset_info.length_unit != old_capabilities.length_unit:
            raise ValueError(
                "The length unit in the dataset info is different from the "
                "length unit of the wrapped model"
            )
        for atomic_type in self.dataset_info.atomic_types:
            if atomic_type not in old_capabilities.atomic_types:
                raise ValueError(
                    f"Atomic type {atomic_type} not supported by the wrapped model"
                )
        for target_name, target in self.dataset_info.targets.items():
            if target_name not in old_capabilities.outputs:
                raise ValueError(
                    f"Target {target_name} not supported by the wrapped model"
                )
            if target.unit != old_capabilities.outputs[target_name].unit:
                raise ValueError(
                    f"Target {target_name} has unit {target.unit}, but the "
                    f"wrapped model has unit "
                    f"{old_capabilities.outputs[target_name].unit}"
                )

        # native outputs only, metatomic re-adds the deprecated aliases at export
        backbone_outputs = self.model.core.supported_outputs()
        num_ensemble_members = self.hypers["num_ensemble_members"]

        names: List[str] = []
        for name in backbone_outputs:
            if name not in self.model.dataset_info.targets:
                continue
            if _get_last_layer_features_name(name) not in backbone_outputs:
                logging.warning(
                    f"The wrapped model exposes no last-layer features for '{name}'; "
                    "LLPR uncertainties for it are unavailable."
                )
                continue
            names.append(name)
        if len(names) == 0:
            raise ValueError(
                "The wrapped model exposes no last-layer features, so LLPR can not "
                "compute any uncertainty"
            )

        self.targets = torch.nn.ModuleDict()
        self.llpr_outputs = []
        self.ensemble_gradient_outputs = []
        outputs = {name: old_capabilities.outputs[name] for name in backbone_outputs}
        for name in names:
            output = outputs[name]
            target = _TargetLLPR(
                name,
                backbone_outputs[_get_last_layer_features_name(name)].layout,
                self.model.core.dataset_info.targets[name].layout,
                self.model.dataset_info.targets[name].layout,
                num_ensemble_members.get(name, 0),
                dtype,
            )
            self.targets[name] = target

            outputs[target.uncertainty_name] = ModelOutput(
                unit=output.unit,
                sample_kind=output.sample_kind,
                description=output.description,
            )
            self.llpr_outputs.append(target.uncertainty_name)
            if target.has_ensemble:
                explicit_gradients = self._ensemble_explicit_gradients(name)
                if len(explicit_gradients) > 0:
                    self.ensemble_gradient_outputs.append(target.ensemble_name)
                outputs[target.ensemble_name] = ModelOutput(
                    unit=output.unit,
                    sample_kind=output.sample_kind,
                    explicit_gradients=explicit_gradients,
                    description=f"ensemble of {name}",
                )
                self.llpr_outputs.append(target.ensemble_name)

        for name in num_ensemble_members:
            if name not in self.targets:
                raise ValueError(
                    f"Output '{name}' in ensembles section is not supported by "
                    "the model"
                )

        self.capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> "LLPRUncertaintyModel":

        if model_hypers is not None:
            raise_if_hypers_mismatch(self.hypers, model_hypers)

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at
            for at in merged_info.atomic_types
            if at not in self.model.core.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }

        if len(new_targets) > 0:
            raise ValueError(
                f"New targets found in the dataset: {new_targets}. "
                "The LLPR ensemble calibration does not support adding new targets."
            )
        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The LLPR ensemble calibration does not support adding new atomic "
                "types."
            )

        self.dataset_info = merged_info

        # invoke restart routine for the wrapped model
        self.model.core.restart(dataset_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        outputs_for_model: Dict[str, ModelOutput] = {}
        for name, output in outputs.items():
            if name not in self.llpr_outputs:
                outputs_for_model[name] = output

        # the targets whose uncertainty or ensemble is requested also need their
        # prediction and last-layer features
        requested: List[str] = []
        for name, target in self.targets.items():
            request: Optional[ModelOutput] = None
            if target.uncertainty_name in outputs:
                request = outputs[target.uncertainty_name]
            elif target.has_ensemble and target.ensemble_name in outputs:
                request = outputs[target.ensemble_name]
            if request is not None:
                requested.append(name)
                if name not in outputs_for_model:
                    outputs_for_model[name] = request
                outputs_for_model[target.features_name] = ModelOutput(
                    sample_kind=request.sample_kind
                )

        if len(requested) == 0:
            return self.model(systems, outputs, selected_atoms)

        # Explicit gradients of an energy ensemble require grad-enabled positions/cell
        # and autograd-registered neighbor lists *before* the wrapped model runs.
        # Engines generally hand over systems without any of this set up, so rebuild
        # them here if needed.
        for name in self.ensemble_gradient_outputs:
            if name in outputs and len(outputs[name].explicit_gradients) > 0:
                if selected_atoms is not None:
                    # the gradient samples built by `_add_energy_ensemble_gradients`
                    # assume one value sample per system, in order, and cover every
                    # atom of every system. A selection can drop systems from the
                    # value block, which leaves the two out of sync.
                    raise ValueError(
                        f"explicit gradients of '{name}' are not supported together "
                        "with 'selected_atoms'"
                    )
                systems = self._systems_with_grad(systems)
                break

        results = self.model(systems, outputs_for_model, selected_atoms)

        for name, target in self.targets.items():
            if name in requested:
                features = results[target.features_name]
                target.move_layouts(features.device)

                if target.uncertainty_name in outputs:
                    uncertainty = self._to_target_units(
                        systems, name, target.uncertainty(features), selected_atoms
                    )
                    results[target.uncertainty_name] = target.apply_multipliers(
                        uncertainty
                    )

                if target.has_ensemble and target.ensemble_name in outputs:
                    deviations = self._to_target_units(
                        systems, name, target.deviations(features), selected_atoms
                    )
                    results[target.ensemble_name] = self._ensemble(
                        target.ensemble_name,
                        results[name],
                        target.apply_multipliers(deviations),
                        outputs[target.ensemble_name],
                        systems,
                    )

        # drop the predictions and features only requested for the uncertainties and
        # ensembles
        for name in list(results.keys()):
            if name not in outputs:
                results.pop(name)

        return results

    def _to_target_units(
        self,
        systems: List[System],
        name: str,
        tensor: TensorMap,
        selected_atoms: Optional[Labels],
    ) -> TensorMap:
        """Bring a quantity that is linear in the ``name`` target of the core model,
        such as its uncertainty, to the layout and units of the target of the wrapped
        model. Unlike the target, it gets the scales but not the additive models.

        :param systems: the systems the quantity was computed for.
        :param name: the name of the target.
        :param tensor: the quantity, in the layout and units of the core model.
        :param selected_atoms: the atoms the quantity was computed for.
        :return: the quantity, in the layout and units of the target.
        """
        scaler = self.model.scaler
        if scaler is not None and name in scaler.dataset_info.targets:
            layout = scaler.dataset_info.targets[name].layout
            tensor = match_layout(
                tensor, layout, target_info_layout=layout, systems=systems
            )
            tensor = scaler.apply_scales(
                systems, {name: tensor}, selected_atoms=selected_atoms
            )[name]
        layout = self.model.dataset_info.targets[name].layout
        return match_layout(tensor, layout, target_info_layout=layout, systems=systems)

    def _ensemble(
        self,
        ensemble_name: str,
        prediction: TensorMap,
        deviations: TensorMap,
        output: ModelOutput,
        systems: List[System],
    ) -> TensorMap:
        """The ensemble members, as the prediction plus their deviations.

        Centering the members on the prediction also accounts for the contributions
        that are not in the last layer (composition, additive models, ...).

        :param ensemble_name: name of the ensemble output.
        :param prediction: the prediction of the target.
        :param deviations: the calibrated deviations of the members from their mean,
            in the layout of the target, with the members as the last component.
        :param output: the requested ensemble output.
        :param systems: the systems the prediction was computed for.
        :return: the ensemble, with the members in the properties.
        """
        want_positions = False
        want_strain = False
        if ensemble_name in self.ensemble_gradient_outputs:
            want_positions = "positions" in output.explicit_gradients
            want_strain = "strain" in output.explicit_gradients
            if (want_positions or want_strain) and output.sample_kind != "system":
                # `_add_energy_ensemble_gradients` assumes one sample per system. For
                # a per-atom energy it would silently return the gradient of the
                # summed energy, labelled as if it were per-sample, so refuse rather
                # than produce wrong numbers.
                raise ValueError(
                    f"explicit gradients of '{ensemble_name}' are only "
                    "supported for a per-system energy, but this output was "
                    f"requested with sample_kind '{output.sample_kind}'"
                )

        blocks: List[TensorBlock] = []
        for key, block in prediction.items():
            values = block.values.unsqueeze(-2) + deviations.block(key).values
            num_ens = values.shape[-2]
            # shape: samples, components, (num_ens * num_properties)
            values = values.reshape(list(block.values.shape[:-1]) + [-1])

            ensemble_properties = _ensemble_properties(
                block.properties, num_ens, ensemble_name == "energy_ensemble"
            )
            ensemble_block = TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=ensemble_properties,
            )
            if want_positions or want_strain:
                self._add_energy_ensemble_gradients(
                    ensemble_block,
                    systems,
                    values,
                    ensemble_properties,
                    num_ens,
                    want_positions,
                    want_strain,
                )
            blocks.append(ensemble_block)

        return TensorMap(keys=prediction.keys, blocks=blocks)

    def _systems_with_grad(self, systems: List[System]) -> List[System]:
        """Rebuild ``systems`` so that positions and cell require grad and neighbor
        lists are registered with autograd, as needed by
        ``_add_energy_ensemble_gradients``.

        Systems whose positions and cell already require grad (e.g. because the
        engine is doing its own autograd on top of this call) are passed through
        untouched, keeping the caller's graph intact. Everything else is replaced by
        a copy built on grad-enabled positions/cell tensors, with neighbor lists
        re-registered against them.

        A tensor that already requires grad is *reused as is* rather than detached,
        even when only one of positions/cell does. Detaching it would silently cut
        the caller out of the graph: an engine that grad-enables positions only (as
        one doing forces but not stress does) would then get "one of the
        differentiated Tensors appears to not have been used in the graph" from its
        own backward pass.

        :param systems: systems to rebuild.
        :return: list with one system per input system, either the original object
            (already grad-enabled) or its grad-enabled copy.
        """
        new_systems: List[System] = []
        for system in systems:
            if system.positions.requires_grad and system.cell.requires_grad:
                new_systems.append(system)
                continue
            positions = system.positions
            if not positions.requires_grad:
                positions = positions.detach().requires_grad_(True)
            cell = system.cell
            if not cell.requires_grad:
                cell = cell.detach().requires_grad_(True)
            new_system = System(system.types, positions, cell, system.pbc)
            for nl_options in system.known_neighbor_lists():
                neighbors = mts.detach_block(system.get_neighbor_list(nl_options))
                register_autograd_neighbors(new_system, neighbors, False)
                new_system.add_neighbor_list(nl_options, neighbors)
            for data_name in system.known_data():
                new_system.add_data(data_name, system.get_data(data_name))
            new_systems.append(new_system)
        return new_systems

    def _add_energy_ensemble_gradients(
        self,
        ensemble_block: TensorBlock,
        systems: List[System],
        ensemble_values: torch.Tensor,
        ens_prop: Labels,
        num_ens: int,
        want_positions: bool,
        want_strain: bool,
    ) -> None:
        """Attach "positions"/"strain" gradients to an "energy_ensemble" block.

        Positions and cell are differentiated directly (no strain-trick deformation
        needed): since the "strain" gradient is evaluated at strain = identity, the
        virial can be recovered from the positions/cell gradients alone as
        ``positions^T @ d(output)/d(positions) + cell^T @ d(output)/d(cell)``, which is
        numerically equivalent to differentiating w.r.t. an explicit strain parameter.
        This avoids needing a dedicated strain tensor.

        :param ensemble_block: the "energy_ensemble" value block gradients are
            attached to, in place, via ``add_gradient``.
        :param systems: the systems this batch was computed for; ``positions`` and
            ``cell`` must already require grad (ensured by ``_systems_with_grad`` in
            ``forward``, or set up by the caller) for any of this to produce
            non-trivial gradients.
        :param ensemble_values: the (already recentered) ensemble values, with shape
            ``(num_samples, num_ens)`` (one row per system, one column per ensemble
            member). This is only valid for the "energy" quantity, which has no
            components and a single property.
        :param ens_prop: properties of ``ensemble_block``, reused verbatim for the
            gradient blocks, as required by metatensor.
        :param num_ens: number of ensemble members.
        :param want_positions: whether to attach a "positions" gradient.
        :param want_strain: whether to attach a "strain" gradient.
        """
        # one column per ensemble member, i.e. a single property and no components:
        # anything else interleaves the properties with the members, and the
        # member-by-member differentiation below would pick the wrong columns
        if len(ensemble_values.shape) != 2 or ensemble_values.shape[1] != num_ens:
            raise ValueError(
                "explicit gradients of an energy ensemble are only supported for a "
                "single-property energy without components"
            )

        n_systems = len(systems)
        all_positions = [system.positions for system in systems]
        all_cells = [system.cell for system in systems]
        n_atoms_per_system = [p.shape[0] for p in all_positions]
        # cells are only differentiated when the strain gradient is actually wanted:
        # asking autograd for them otherwise costs a gradient per system per member
        # that is then discarded
        grad_inputs = all_positions + all_cells if want_strain else all_positions

        dtype = ensemble_values.dtype
        device = ensemble_values.device
        n_atoms_total = sum(n_atoms_per_system)

        positions_grad_values = torch.zeros(
            (n_atoms_total, 3, num_ens), dtype=dtype, device=device
        )
        strain_grad_values = torch.zeros(
            (n_systems, 3, 3, num_ens) if want_strain else (0, 3, 3, num_ens),
            dtype=dtype,
            device=device,
        )

        for member in range(num_ens):
            grads = torch.autograd.grad(
                [ensemble_values[:, member].sum()],
                grad_inputs,
                # never False: the graph may be the caller's (see `_systems_with_grad`,
                # which passes already-grad-enabled systems straight through), and
                # freeing it here would break the caller's own backward pass with
                # "trying to backward through the graph a second time". The graph is
                # released normally once the last reference to it goes away.
                retain_graph=True,
                create_graph=False,
            )

            atom_offset = 0
            for s in range(n_systems):
                pos_grad = grads[s]
                assert pos_grad is not None

                n_atoms_s = n_atoms_per_system[s]
                # store dE/dr directly (not the force -dE/dr)
                positions_grad_values[
                    atom_offset : atom_offset + n_atoms_s, :, member
                ] = pos_grad
                if want_strain:
                    cell_grad = grads[n_systems + s]
                    assert cell_grad is not None
                    strain_grad_values[s, :, :, member] = (
                        all_positions[s].detach().t() @ pos_grad
                        + all_cells[s].detach().t() @ cell_grad
                    )
                atom_offset += n_atoms_s

        xyz = torch.tensor([[0], [1], [2]], device=device)

        if want_positions:
            sample_col = torch.repeat_interleave(
                torch.arange(n_systems, device=device),
                torch.tensor(n_atoms_per_system, device=device),
            )
            atom_col = torch.cat(
                [torch.arange(n, device=device) for n in n_atoms_per_system]
            )
            ensemble_block.add_gradient(
                "positions",
                TensorBlock(
                    values=positions_grad_values,
                    samples=Labels(
                        names=["sample", "system", "atom"],
                        values=torch.stack([sample_col, sample_col, atom_col], dim=1),
                    ),
                    components=[Labels(["xyz"], xyz)],
                    properties=ens_prop,
                ),
            )

        if want_strain:
            # build the "sample" Labels with the plain constructor, not
            # `Labels.range(...)` as the latter breaks TorchScript compatibility
            ensemble_block.add_gradient(
                "strain",
                TensorBlock(
                    values=strain_grad_values,
                    samples=Labels(
                        names=["sample"],
                        values=torch.arange(n_systems, device=device).reshape(-1, 1),
                    ),
                    components=[Labels(["xyz_1"], xyz), Labels(["xyz_2"], xyz)],
                    properties=ens_prop,
                ),
            )

    def compute_cholesky_decomposition(
        self, regularizer: Optional[float] = None
    ) -> None:
        """Compute the Cholesky factor of each covariance.

        :param regularizer: A regularization parameter to ensure the matrix is
            positive-definite. If not provided, the function will try to compute the
            Cholesky decomposition without regularization and increase the
            regularization parameter until the matrix is positive-definite.
        """
        for name, target in self.targets.items():
            for index, covariance in enumerate(target.covariances):
                used_regularizer = covariance.compute_cholesky(regularizer)
                if regularizer is None:
                    logging.info(
                        f"Used regularization parameter of {used_regularizer:.1e} to "
                        "compute the Cholesky decomposition for "
                        f"`{name}` (feature block {index})"
                    )

    def generate_ensemble(self) -> None:
        """Sample the weights of the ensemble members of each target.

        The weights are sampled from a multivariate normal distribution with zero mean
        and covariance given by the inverse covariance matrix.
        """
        for target in self.targets.values():
            target.sample_ensemble()

    def get_checkpoint(
        self, best_model_state_dict: Optional[dict[str, Any]] = None
    ) -> Dict[str, Any]:
        wrapped_model_checkpoint = self.model.get_checkpoint()
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        checkpoint = {
            "architecture_name": "llpr",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": state_dict,
            "best_model_state_dict": best_model_state_dict or state_dict,
            "wrapped_model_checkpoint": wrapped_model_checkpoint,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "LLPRUncertaintyModel":
        model = model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)
        if context == "finetune":
            # In this case, we want to allow fine-tuning of the underlying model by
            # extracting it and returning it directly
            return model  # type: ignore[return-value]

        if context == "restart":
            logging.info(
                "Restart for LLPRUncertaintyModel will attempt continuation of "
                "ensemble calibration"
            )
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        else:
            # the best state dict is None if the ensemble was not trained by
            # backpropagation
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]

        llpr_model = cls(**checkpoint["model_data"])
        llpr_model.set_wrapped_model(model)

        # the covariances are always in float64
        dtype = next(
            value.dtype
            for key, value in model_state_dict.items()
            if isinstance(value, torch.Tensor)
            and torch.is_floating_point(value)
            and not key.endswith(".covariance")
        )
        # the wrapped model is not part of the state dict, as it is loaded from its
        # own checkpoint
        missing_keys, unexpected_keys = llpr_model.to(dtype).load_state_dict(
            model_state_dict, strict=False
        )
        missing_keys = [key for key in missing_keys if not key.startswith("model.")]
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            raise ValueError(
                "Unable to load the LLPR checkpoint: missing keys "
                f"{missing_keys}, unexpected keys {unexpected_keys}."
            )
        return llpr_model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype

        # Make sure the model is all in the same dtype
        self.to(dtype)

        exported = self.model.export()
        metadata = merge_metadata(
            merge_metadata(self.__default_metadata__, metadata), exported.metadata()
        )

        # copy with the wrapped model replaced by its scriptable, exported version
        llpr_model = copy.deepcopy(self, memo={id(self.model): exported.module})

        return AtomisticModel(llpr_model.eval(), metadata, self.capabilities)

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict[str, Any], version: Optional[int] = None
    ) -> dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "model", checkpoints
        )

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        outputs = self.model.supported_outputs()
        for name, target in self.targets.items():
            target_info = self.model.dataset_info.targets[name]
            blocks: List[TensorBlock] = []
            ensemble_blocks: List[TensorBlock] = []
            for block in target_info.layout.blocks():
                # the uncertainty has the layout of the target, without gradients
                blocks.append(
                    TensorBlock(
                        values=torch.empty_like(block.values),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
                if target.has_ensemble:
                    properties = _ensemble_properties(
                        block.properties,
                        target.num_ensemble_members,
                        target.ensemble_name == "energy_ensemble",
                    )
                    ensemble_blocks.append(
                        TensorBlock(
                            values=torch.empty(
                                list(block.values.shape[:-1]) + [len(properties)],
                                dtype=block.values.dtype,
                            ),
                            samples=block.samples,
                            components=block.components,
                            properties=properties,
                        )
                    )
            keys = target_info.layout.keys
            outputs[target.uncertainty_name] = TargetInfo(
                TensorMap(keys, blocks), unit=target_info.unit
            )
            if target.has_ensemble:
                outputs[target.ensemble_name] = TargetInfo(
                    TensorMap(keys, ensemble_blocks), unit=target_info.unit
                )
        return outputs

    def _ensemble_explicit_gradients(self, name: str) -> List[str]:
        """Explicit gradients the ensemble of the ``name`` target is able to produce.

        Only energy targets are currently supported.

        :param name: name of the target the ensemble is built from.
        :return: gradient names for the corresponding ensemble output.
        """
        target = self.dataset_info.targets.get(name)
        if target is None or target.quantity != "energy":
            return []
        return ["positions", "strain"]


class _TargetLLPR(torch.nn.Module):
    """The LLPR state of one target: the covariance of each block of its last-layer
    features, one calibration factor for each property of each block of the target,
    and, for ensembles, one linear layer for each block of the core's target.

    The last-layer features are either a single block without components, shared by
    all the blocks of the target, or have the keys and components of the target in the
    core. Uncertainties and ensembles are computed in the layout of the core, which the
    wrapped model can transform to give the layout of the target.

    :param name: name of the target.
    :param features_layout: layout of the last-layer features of the target.
    :param core_layout: layout of the target in the core.
    :param layout: layout of the target.
    :param num_ensemble_members: number of ensemble members, 0 for no ensemble.
    :param dtype: dtype of the buffers.
    """

    def __init__(
        self,
        name: str,
        features_layout: TensorMap,
        core_layout: TensorMap,
        layout: TensorMap,
        num_ensemble_members: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.uncertainty_name = _get_uncertainty_name(name)
        self.ensemble_name = _get_ensemble_name(name)
        self.features_name = _get_last_layer_features_name(name)
        self.num_ensemble_members = num_ensemble_members
        self.has_ensemble = num_ensemble_members > 0
        self.core_layout = core_layout
        self.layout_keys = layout.keys

        self.shared, feature_sizes = _feature_layout(
            name, features_layout.to(core_layout.device), self.core_layout
        )
        # the covariance of the features read by each block of the core's target
        self.feature_indices: List[int] = [
            0 if self.shared else index for index in range(len(core_layout))
        ]
        self.covariances = torch.nn.ModuleList(
            [_Covariance(size, dtype) for size in feature_sizes]
        )

        # shared features are mapped to all the components of a block at once
        self.ensemble_layers = torch.nn.ModuleList()
        if self.has_ensemble:
            for index, block in enumerate(core_layout.blocks()):
                num_outputs = len(block.properties)
                if self.shared:
                    num_outputs *= _prod([len(c) for c in block.components])
                self.ensemble_layers.append(
                    torch.nn.Linear(
                        feature_sizes[self.feature_indices[index]],
                        num_ensemble_members * num_outputs,
                        bias=False,
                        dtype=dtype,
                    )
                )

        # the multipliers of all the blocks of the target, one after the other
        self.multiplier_offsets: List[int] = [0]
        for block in layout.blocks():
            self.multiplier_offsets.append(
                self.multiplier_offsets[-1] + len(block.properties)
            )
        self.register_buffer(
            "multipliers", torch.ones(self.multiplier_offsets[-1], dtype=dtype)
        )

    def move_layouts(self, device: torch.device) -> None:
        """Move the layouts to ``device``.

        :param device: the device of the features.
        """
        if self.layout_keys.values.device != device:
            self.core_layout = self.core_layout.to(device)
            self.layout_keys = self.layout_keys.to(device)

    def _feature_blocks(self, features: TensorMap) -> List[TensorBlock]:
        """The blocks of ``features`` matching each covariance.

        :param features: the last-layer features of the target.
        :return: the blocks.
        """
        if self.shared:
            return [features.block(0)]
        return [
            features.block(self.core_layout.keys.entry(index))
            for index in range(len(self.core_layout))
        ]

    def uncertainty(self, features: TensorMap) -> TensorMap:
        """The uncalibrated uncertainty, in the layout and units of the core.

        :param features: the last-layer features of the target.
        :return: the uncertainty.
        """
        # computed once for each feature block, and shared by the blocks reading it
        feature_blocks = self._feature_blocks(features)
        one_over_pr: List[torch.Tensor] = []
        for index, covariance in enumerate(self.covariances):
            one_over_pr.append(covariance.one_over_pr(feature_blocks[index].values))

        blocks: List[TensorBlock] = []
        for index, layout_block in enumerate(self.core_layout.blocks()):
            feature_index = self.feature_indices[index]
            values = one_over_pr[feature_index]
            shape = [values.shape[0]] + list(layout_block.values.shape[1:])
            values = values.expand(-1, _prod(shape[1:-1]), shape[-1]).reshape(shape)
            blocks.append(
                TensorBlock(
                    values=torch.sqrt(values),
                    samples=feature_blocks[feature_index].samples,
                    components=layout_block.components,
                    properties=layout_block.properties,
                )
            )
        return TensorMap(keys=self.core_layout.keys, blocks=blocks)

    def deviations(self, features: TensorMap) -> TensorMap:
        """The uncalibrated deviations of the ensemble members from their mean, in the
        layout and units of the core, with the members as the last component.

        :param features: the last-layer features of the target.
        :return: the deviations.
        """
        feature_blocks = self._feature_blocks(features)
        blocks: List[TensorBlock] = []
        for index, layer in enumerate(self.ensemble_layers):
            feature_block = feature_blocks[self.feature_indices[index]]
            layout_block = self.core_layout.block(index)
            values = layer(feature_block.values)

            num_samples = values.shape[0]
            components_shape = list(layout_block.values.shape[1:-1])
            num_properties = layout_block.values.shape[-1]
            if self.shared:
                values = values.reshape(
                    [num_samples, -1] + components_shape + [num_properties]
                ).movedim(1, -2)
            else:
                # the features carry the components, and each member's weights are
                # shared across them
                values = values.reshape(
                    [num_samples] + components_shape + [-1, num_properties]
                )

            members = Labels(
                names=["ensemble_member"],
                values=torch.arange(values.shape[-2], device=values.device).unsqueeze(
                    -1
                ),
            )
            blocks.append(
                TensorBlock(
                    values=values - values.mean(dim=-2, keepdim=True),
                    samples=feature_block.samples,
                    components=layout_block.components + [members],
                    properties=layout_block.properties,
                )
            )
        return TensorMap(keys=self.core_layout.keys, blocks=blocks)

    def apply_multipliers(self, tensor: TensorMap) -> TensorMap:
        """Multiply each block of ``tensor`` by its calibration factors.

        :param tensor: a tensor in the layout of the target.
        :return: the calibrated tensor.
        """
        blocks: List[TensorBlock] = []
        for key, block in tensor.items():
            index = self.layout_keys.position(key)
            assert index is not None
            multipliers = self.multipliers[
                self.multiplier_offsets[index] : self.multiplier_offsets[index + 1]
            ]
            blocks.append(
                TensorBlock(
                    values=block.values * multipliers,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )
        return TensorMap(keys=tensor.keys, blocks=blocks)

    def set_multipliers(self, block_index: int, values: torch.Tensor) -> None:
        """Set the calibration factors of a block of the target.

        :param block_index: index of the block in the layout of the target.
        :param values: one calibration factor for each property of the block.
        """
        start = self.multiplier_offsets[block_index]
        end = self.multiplier_offsets[block_index + 1]
        self.multipliers[start:end] = values.to(self.multipliers)

    def accumulate(self, features: TensorMap, n_atoms: Optional[torch.Tensor]) -> None:
        """Add the contribution of a batch of last-layer features to the covariances.

        :param features: the last-layer features of the target.
        :param n_atoms: the number of atoms of each system, to normalize per-system
            features, or ``None`` for per-atom features.
        """
        self.move_layouts(features.device)
        feature_blocks = self._feature_blocks(features)
        for index, covariance in enumerate(self.covariances):
            values = feature_blocks[index].values.detach()
            # TODO: interface ll_feat calculation with the loss function, paying
            # attention to normalization w.r.t. n_atoms
            if n_atoms is not None:
                values = values / n_atoms.reshape([-1] + [1] * (values.dim() - 1))
            # components count as samples, as the last layer is shared across them
            values = values.reshape(-1, values.shape[-1]).to(torch.float64)
            covariance.covariance += values.T @ values

    def sample_ensemble(self) -> None:
        """Sample the weights of the ensemble members."""
        for index, layer in enumerate(self.ensemble_layers):
            cholesky = self.covariances[self.feature_indices[index]].cholesky
            z = torch.randn(
                (layer.in_features, layer.out_features),
                device=cholesky.device,
                dtype=cholesky.dtype,
            )
            with torch.no_grad():
                layer.weight.copy_(
                    torch.linalg.solve_triangular(cholesky.T, z, upper=True).T
                )


class _Covariance(torch.nn.Module):
    """The covariance of a block of last-layer features, and its Cholesky factor.

    The covariance is always in float64: summed over many batches in float32, it loses
    its smallest eigenvalues and depends on the order of the batches.

    :param size: the number of features.
    :param dtype: dtype of the Cholesky factor.
    """

    def __init__(self, size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer(
            "covariance", torch.zeros((size, size), dtype=torch.float64)
        )
        self.register_buffer("cholesky", torch.zeros((size, size), dtype=dtype))

    def _apply(self, fn: Callable, recurse: bool = True) -> "_Covariance":
        # moves the covariance to other devices, but keeps it in float64
        covariance = self.covariance
        super()._apply(fn, recurse)
        self.register_buffer("covariance", covariance.to(device=self.cholesky.device))
        return self

    def one_over_pr(self, values: torch.Tensor) -> torch.Tensor:
        """``1/PR`` for each sample and component; the same for PR and LPR.

        :param values: the features.
        :return: ``1/PR``, with shape ``(samples, components, 1)``, where
            components is 1 for features without components.
        """
        values = values.reshape(values.shape[0], -1, values.shape[-1])
        v = torch.linalg.solve_triangular(
            self.cholesky, values.reshape(-1, values.shape[-1]).T, upper=False
        )
        return torch.sum(v**2, dim=0).reshape(values.shape[0], values.shape[1], 1)

    def compute_cholesky(self, regularizer: Optional[float]) -> float:
        """Compute the Cholesky factor of the regularized covariance.

        :param regularizer: the regularization, or ``None`` to use the smallest power
            of ten that makes the covariance positive-definite.
        :return: the regularization used.
        """
        covariance = 0.5 * (self.covariance + self.covariance.T)
        identity = torch.eye(
            covariance.shape[0], device=covariance.device, dtype=torch.float64
        )
        if regularizer is not None:
            cholesky = _cholesky(covariance + regularizer * identity)
            if cholesky is None:
                raise ValueError(
                    f"the covariance with a regularization of {regularizer} is not "
                    "positive-definite"
                )
            self.cholesky[:] = cholesky.to(self.cholesky.dtype)
            return regularizer

        r = 1e-20
        while r < 1e16:
            cholesky = _cholesky(covariance + r * identity)
            if cholesky is not None:
                self.cholesky[:] = cholesky.to(self.cholesky.dtype)
                return r
            r *= 10.0
        raise RuntimeError(
            "Could not compute Cholesky decomposition. Something went wrong. Please "
            "contact the metatrain developers"
        )


def _feature_layout(
    name: str, features_layout: TensorMap, core_layout: TensorMap
) -> Tuple[bool, List[int]]:
    """Whether the last-layer features of a target are shared by all its blocks, and
    the number of features of each of their blocks, in the order of the target's
    blocks.

    :param name: name of the target.
    :param features_layout: layout of the last-layer features of the target.
    :param core_layout: layout of the target in the core.
    :return: whether the features are shared, and the size of each feature block.
    """
    if len(features_layout) == 1 and len(features_layout.block(0).components) == 0:
        return True, [len(features_layout.block(0).properties)]

    feature_sizes: List[int] = []
    if features_layout.keys.names == core_layout.keys.names:
        for key, block in core_layout.items():
            position = features_layout.keys.position(key)
            if position is None:
                break
            feature_block = features_layout.block(position)
            if feature_block.components != block.components:
                break
            feature_sizes.append(len(feature_block.properties))
    if len(feature_sizes) != len(core_layout):
        raise ValueError(
            f"the last-layer features of '{name}' must either be a single block "
            "without components, or have the keys and components of the target"
        )
    return False, feature_sizes


def _cholesky(matrix: torch.Tensor) -> Optional[torch.Tensor]:
    """The Cholesky factor of ``matrix``, or ``None`` if it is not positive-definite.

    :param matrix: a symmetric matrix.
    :return: the lower-triangular factor, or ``None``.
    """
    cholesky, info = torch.linalg.cholesky_ex(matrix)
    # on GPUs, an indefinite matrix can give NaNs instead of a non-zero ``info``
    if int(info) != 0 or bool(torch.isnan(cholesky).any()):
        return None
    return cholesky


def _get_uncertainty_name(name: str) -> str:
    if name == "energy":
        return "energy_uncertainty"
    return f"mtt::aux::{name.replace('mtt::', '')}_uncertainty"


def _get_ensemble_name(name: str) -> str:
    if name == "energy":
        return "energy_ensemble"
    return f"mtt::aux::{name.replace('mtt::', '')}_ensemble"


def _ensemble_properties(properties: Labels, num_members: int, energy: bool) -> Labels:
    """The properties of an ensemble: each member, then the properties of the target.

    :param properties: the properties of the target.
    :param num_members: the number of members of the ensemble.
    :param energy: whether this is the standard ``energy_ensemble`` output.
    :return: the properties of the ensemble.
    """
    values = properties.values
    members = torch.arange(num_members, device=values.device, dtype=values.dtype)
    members = members.repeat_interleave(values.shape[0]).unsqueeze(1)
    if energy:
        # "energy_ensemble" quantity requires a single "energy"
        # property column holding the ensemble member index
        return Labels(names=["energy"], values=members)
    return Labels(
        names=["ensemble_member"] + properties.names,
        values=torch.cat([members, values.repeat(num_members, 1)], dim=-1),
    )


def _get_last_layer_features_name(name: str) -> str:
    return f"mtt::aux::{name.replace('mtt::', '')}_last_layer_features"


def _prod(list_of_int: List[int]) -> int:
    # for torchscript compatibility (math.prod is not supported)
    result = 1
    for x in list_of_int:
        result = result * x
    return result
