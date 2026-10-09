from collections.abc import Mapping
from typing import Any, Dict, Generic, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from typing_extensions import TypeVar

# from metatrain.scaler import Scaler
from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.match_tmaps import match_layout
from metatrain.utils.dtype import dtype_to_str


CoreModelClass = TypeVar("CoreModelClass", bound=ModelInterface)
ScalerClass = TypeVar("ScalerClass", default=None, bound=Optional[ModelInterface])


class MetatrainModel(torch.nn.Module, Generic[CoreModelClass, ScalerClass]):
    __checkpoint_version__ = 1

    # Dictionary storing the targets that need to be transformed at evaluation
    _eval_transforms: dict[str, list[str]]

    def __init__(
        self,
        model: CoreModelClass,
        additive_models: list[ModelInterface],
        scaler: ScalerClass,  # Scaler,
        dataset_info: DatasetInfo,
    ):
        super().__init__()
        self.model = model
        self.additive_models = torch.nn.ModuleList(additive_models)
        self.scaler = scaler
        self.dataset_info = dataset_info

        # Compute transformations needed at evaluation time.
        self._eval_transforms = self._get_eval_transforms()

    def _get_eval_transforms(self) -> dict[str, list[str]]:
        """Finds out which targets need to be transformed at evaluation time to
        make sure that the outputs of the model, scaler and additive models
        are compatible.

        :return: A dictionary with the targets that need to be transformed at
            evaluation time. There is a key for each step of the evaluation,
            and the value is a list of targets that need to be transformed.
        """
        eval_transforms: dict[str, list[str]] = {}

        # Get targets of each model.
        model_targets = self.model.dataset_info.targets
        scaler_targets = (
            self.scaler.dataset_info.targets if self.scaler is not None else {}
        )
        additive_targets = [
            additive_model.dataset_info.targets
            for additive_model in self.additive_models
        ]

        # Go through the steps of the evaluation and find out
        # the required transformations.
        current_targets = model_targets.copy()
        # Targets to transform before applying the scaler
        eval_transforms["pre_scaler"] = []
        for name, target in current_targets.items():
            if name in scaler_targets and not mts.equal_metadata(
                target.layout, scaler_targets[name].layout
            ):
                eval_transforms["pre_scaler"].append(name)
                current_targets[name] = scaler_targets[name]

        # Targets to transform before applying each additive model
        for i in range(len(self.additive_models)):
            eval_transforms[f"pre_additive_{i}"] = []
            for name, target in current_targets.items():
                if name in additive_targets[i] and not mts.equal_metadata(
                    target.layout, additive_targets[i][name].layout
                ):
                    eval_transforms[f"pre_additive_{i}"].append(name)
                    current_targets[name] = additive_targets[i][name]

        # Targets to transform before returning the final outputs
        eval_transforms["pre_return"] = []
        for name, target in current_targets.items():
            if name in self.dataset_info.targets and not mts.equal_metadata(
                target.layout, self.dataset_info.targets[name].layout
            ):
                eval_transforms["pre_return"].append(name)

        return eval_transforms

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return_dict = self.model(systems, outputs, selected_atoms=selected_atoms)

        with torch.profiler.record_function("MTT_WRAPPER::post-processing"):
            if not self.training:
                # at evaluation, we also introduce the scaler and additive contributions

                if self.scaler is not None:
                    # We are going to apply the scaler, convert all targets whose layout
                    # is different between the model and the scaler.
                    for k in self._eval_transforms["pre_scaler"]:
                        if k in return_dict:
                            return_dict[k] = match_layout(
                                return_dict[k],
                                self.scaler.dataset_info.targets[k].layout,
                                target_info_layout=self.scaler.dataset_info.targets[
                                    k
                                ].layout,
                                systems=systems,
                            )

                    return_dict = self.scaler.apply_scales(
                        systems,
                        return_dict,
                        selected_atoms=selected_atoms,
                        use_per_target_scales=True,
                        use_per_property_scales=True,
                    )

                for i, additive_model in enumerate(self.additive_models):
                    # Convert all targets whose layout is different at this point
                    # compared to the additive model that we will apply.
                    for k in self._eval_transforms[f"pre_additive_{i}"]:
                        if k in return_dict:
                            return_dict[k] = match_layout(
                                return_dict[k],
                                additive_model.dataset_info.targets[k].layout,
                                target_info_layout=additive_model.dataset_info.targets[
                                    k
                                ].layout,
                                systems=systems,
                            )
                    outputs_for_additive_model: Dict[str, ModelOutput] = {}
                    for name, output in outputs.items():
                        if name in additive_model.outputs:
                            outputs_for_additive_model[name] = output
                    additive_contributions = additive_model(
                        systems,
                        outputs_for_additive_model,
                        selected_atoms,
                    )
                    for name in additive_contributions:
                        # TODO: uncomment this after metatensor.torch.add
                        # is updated to handle sparse sums
                        # return_dict[name] = metatensor.torch.add(
                        #     return_dict[name],
                        #     additive_contributions[name].to(
                        #         device=return_dict[name].device,
                        #         dtype=return_dict[name].dtype
                        #         ),
                        # )
                        # TODO: "manual" sparse sum: update to metatensor.torch.add
                        # after sparse sum is implemented in metatensor.operations
                        output_blocks: List[TensorBlock] = []
                        for k, b in return_dict[name].items():
                            if k in additive_contributions[name].keys:
                                output_blocks.append(
                                    _add_block_block(
                                        b,
                                        additive_contributions[name]
                                        .block(k)
                                        .to(device=b.device, dtype=b.dtype),
                                    )
                                )
                            else:
                                output_blocks.append(
                                    TensorBlock(
                                        values=b.values,
                                        samples=b.samples,
                                        components=b.components,
                                        properties=b.properties,
                                    )
                                )
                        return_dict[name] = TensorMap(
                            return_dict[name].keys, output_blocks
                        )

                # Convert all targets whose layout is different at this point
                # compared to what is declared in the dataset_info of the wrapper model.
                for k in self._eval_transforms["pre_return"]:
                    if k in return_dict:
                        return_dict[k] = match_layout(
                            return_dict[k],
                            self.dataset_info.targets[k].layout,
                            target_info_layout=self.dataset_info.targets[k].layout,
                            systems=systems,
                        )

        return return_dict

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        requested_inputs = {}

        def _add_model_requested_inputs(model: ModelInterface) -> None:
            if not hasattr(model, "requested_inputs"):
                return
            for name, output in model.requested_inputs().items():
                if name not in requested_inputs:
                    requested_inputs[name] = output

        for additive_model in self.additive_models:
            _add_model_requested_inputs(additive_model)

        if self.scaler is not None:
            _add_model_requested_inputs(self.scaler)
        _add_model_requested_inputs(self.model)

        return requested_inputs

    def requested_neighbor_lists(self) -> list[NeighborListOptions]:
        requested_neighbor_lists = []

        def _add_model_requested_neighbor_lists(model: ModelInterface) -> None:
            if not hasattr(model, "requested_neighbor_lists"):
                return
            for nl in model.requested_neighbor_lists():
                if nl not in requested_neighbor_lists:
                    requested_neighbor_lists.append(nl)

        for additive_model in self.additive_models:
            _add_model_requested_neighbor_lists(additive_model)

        if self.scaler is not None:
            _add_model_requested_neighbor_lists(self.scaler)
        _add_model_requested_neighbor_lists(self.model)

        return requested_neighbor_lists

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.model.supported_outputs()

    def export(
        self,
        metadata: Optional[ModelMetadata] = None,
    ) -> AtomisticModel:
        """
        Turn this model into an instance of
        :py:class:`metatomic.torch.MetatensorAtomisticModel`, containing the model
        itself, a definition of the model capabilities and some metadata about the
        model.

        :param metadata: additional metadata to add in the model as specified by the
            user.

        :return: An instance of :py:class:`metatomic.torch.MetatensorAtomisticModel`
        """
        # Export all models, making sure that the additive models and scaler have
        # the same dtype as the main model.
        model = self.model.export(metadata)
        dtype = getattr(torch, model.capabilities().dtype)
        additive_models = [
            model.to(dtype).export(metadata) for model in self.additive_models
        ]
        if self.scaler is not None:
            scaler = self.scaler.to(dtype).export(metadata)
        else:
            scaler = None

        # Get a list of the capabilities of each model
        all_capabilities = [model.capabilities()] + [
            model.capabilities() for model in additive_models
        ]
        if scaler is not None:
            all_capabilities.append(scaler.capabilities())

        # The interaction range of the model is the maximum interaction range
        # of all the models involved.
        all_interaction_ranges = [
            cap.interaction_range
            for cap in all_capabilities
            if cap.interaction_range is not None
        ]
        interaction_range = (
            max(all_interaction_ranges) if all_interaction_ranges else None
        )

        all_supported_devices = [cap.supported_devices for cap in all_capabilities]
        # Get the intersection of all supported devices
        supported_devices = set(all_supported_devices[0])
        for devices in all_supported_devices[1:]:
            supported_devices.intersection_update(devices)

        # Build the wrapper model again with the exported modules.
        to_export = self.__class__(
            model=model.module,
            additive_models=[model.module for model in additive_models],
            scaler=scaler.module if scaler is not None else None,  # type: ignore[arg-type]
            dataset_info=self.dataset_info,
        )
        # Some architectures have attributes that downstream code reads from the
        # exported module (e.g. FlashMD's ``timestep``, read by the ``flashmd``
        # integrators and by LAMMPS' ``fix metatomic``). Expose them on the wrapper
        # too, so that they can still be found at the same place.
        for name in getattr(model.module, "__exported_buffers__", []):
            to_export.register_buffer(name, getattr(model.module, name))

        if metadata is None:
            metadata = model.metadata()

        capabilities = ModelCapabilities(
            outputs=self.supported_outputs(),
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=supported_devices,
            dtype=dtype_to_str(dtype),
        )

        return AtomisticModel(to_export.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict["str", Any]) -> Dict["str", Any]:
        """
        Upgrade the checkpoint to the current version of the model.

        :param checkpoint: Checkpoint's state dictionary.

        :raises RuntimeError: if the checkpoint cannot be upgraded to the current
            version of the model.

        :return: The upgraded checkpoint.
        """
        for k in ["model", "scaler"]:
            subcheckpoint = checkpoint[k]
            if k == "scaler" and subcheckpoint is None:
                continue

            architecture_name = subcheckpoint["architecture_name"]
            model_cls = import_architecture(architecture_name).__model__
            checkpoint[k] = model_cls.upgrade_checkpoint(subcheckpoint)

        for i, additive_model_checkpoint in enumerate(checkpoint["additive_models"]):
            architecture_name = additive_model_checkpoint["architecture_name"]
            model_cls = import_architecture(architecture_name).__model__
            checkpoint["additive_models"][i] = model_cls.upgrade_checkpoint(
                additive_model_checkpoint
            )

        return checkpoint

    def get_checkpoint(
        self, best_model_state_dict: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Get the checkpoint of the model. This should contain all the information
        needed by `load_checkpoint` to recreate the same model instance.

        :param best_model_state_dict: Optional state dictionary of the best model
            (if available). If provided, the submodels will receive their
            corresponding best state dict so that they can include it in their
            checkpoints.

        :return: The model's checkpoint.
        """
        model_best_state_dict: dict[str, Any] | None = None
        additive_models_best_state_dicts: list[dict[str, Any] | None] = [None] * len(
            self.additive_models
        )
        scaler_best_state_dict: dict[str, Any] | None = None

        # Split the best_model_state_dict into the corresponding sub-models' state dicts
        if best_model_state_dict is not None:
            model_best_state_dict = {
                k.removeprefix("model."): v
                for k, v in best_model_state_dict.items()
                if k.startswith("model.")
            }
            additive_models_best_state_dicts = [
                {
                    k.removeprefix(f"additive_models.{i}."): v
                    for k, v in best_model_state_dict.items()
                    if k.startswith(f"additive_models.{i}.")
                }
                for i in range(len(self.additive_models))
            ]
            scaler_best_state_dict = {
                k.removeprefix("scaler."): v
                for k, v in best_model_state_dict.items()
                if k.startswith("scaler.")
            }

        checkpoint = {
            "model_ckpt_version": self.__checkpoint_version__,
            "dataset_info": self.dataset_info,
            "model": self.model.get_checkpoint(model_best_state_dict),
            "additive_models": [
                m.get_checkpoint(best_state)
                for m, best_state in zip(
                    self.additive_models, additive_models_best_state_dicts, strict=True
                )
            ],
            "scaler": self.scaler.get_checkpoint(scaler_best_state_dict)
            if self.scaler is not None
            else None,
        }
        return checkpoint

    @staticmethod
    def get_state_dict_from_checkpoint(
        checkpoint: dict[str, Any], key: str = "state_dict"
    ) -> Optional[dict[str, Any]]:
        """
        Get the model's state dictionary from a checkpoint.

        It loops through all the sub-models and extracts their state dicts,
        then combines them into a single state dict.

        :param checkpoint: The checkpoint to extract the state dict from.
        :param key: The key in the sub-models checkpoints where the state
           dict is stored.
        :return: The model's state dict.
        """
        state_dict = {}
        if (model_state_dict := checkpoint["model"][key]) is not None:
            state_dict.update({f"model.{k}": v for k, v in model_state_dict.items()})
        for i, additive_model_checkpoint in enumerate(checkpoint["additive_models"]):
            if (
                additive_model_state_dict := additive_model_checkpoint[key]
            ) is not None:
                state_dict.update(
                    {
                        f"additive_models.{i}.{k}": v
                        for k, v in additive_model_state_dict.items()
                    }
                )
        if (scaler_checkpoint := checkpoint["scaler"]) is not None:
            if (scaler_state_dict := scaler_checkpoint[key]) is not None:
                state_dict.update(
                    {f"scaler.{k}": v for k, v in scaler_state_dict.items()}
                )

        if len(state_dict) == 0:
            return None

        return state_dict

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_dataset_info: Optional[DatasetInfo] = None,
        additive_models_dataset_info: Optional[list[DatasetInfo]] = None,
        scaler_dataset_info: Optional[DatasetInfo] = None,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Restart the model with new dataset_info and model_hypers.
        This is used when the model is loaded from a checkpoint and the dataset_info
        has changed (e.g. when finetuning on a new dataset).

        :param dataset_info: The new dataset_info to use.
        :param model_dataset_info: The new dataset_info to use for the main model.
            If None, `dataset_info` will be used.
        :param additive_models_dataset_info: The new dataset_info to use for the
            additive models.
            If None, `dataset_info` will be used for all additive models.
        :param scaler_dataset_info: The new dataset_info to use for the scaler.
            If None, `dataset_info` will be used.
        :param model_hypers: The new model_hypers to use. If None, the current
            hypers will be used.
        """
        # Update the dataset info for the wrapper.
        merged_info = self.dataset_info.union(dataset_info)
        self.dataset_info = merged_info

        # Get the dataset infos for each of the components.
        model_dataset_info = model_dataset_info or dataset_info
        additive_models_dataset_info = additive_models_dataset_info or [
            dataset_info
        ] * len(self.additive_models)
        scaler_dataset_info = scaler_dataset_info or dataset_info

        # Update main model
        self.model.restart(model_dataset_info, model_hypers)

        # Update additive models
        for info, additive_model in zip(
            additive_models_dataset_info, self.additive_models, strict=True
        ):
            additive_model.restart(dataset_info=info)

        if self.scaler is not None:
            self.scaler = self.scaler.restart(scaler_dataset_info)

        # Compute transformations needed at evaluation time.
        self._eval_transforms = self._get_eval_transforms()

    def remove_output(self, target_name: str) -> None:
        """
        Remove a previously registered output target.

        Used to drop targets whose heads are no longer meaningful after a
        backbone-altering fine-tuning run (``full``/``lora``).

        :param target_name: Name of the target to remove.
        """
        self.model.remove_output(target_name)

        for additive_model in self.additive_models:
            if target_name in additive_model.supported_outputs():
                additive_model.remove_output(target_name)

        if self.scaler is not None and target_name in self.scaler.supported_outputs():
            self.scaler.remove_output(target_name)

        self.dataset_info.targets.pop(target_name, None)
