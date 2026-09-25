from typing import Any, Dict, List, Literal, Optional, NotRequired, Union, Callable

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import (
    AtomisticModel,
    ModelMetadata,
    ModelOutput,
    ModelCapabilities,
    System,
    NeighborListOptions,
)
from typing_extensions import TypedDict

#from metatrain.scaler import Scaler
from metatrain.utils.abc import ModelInterface
from metatrain.utils.architectures import import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.dtype import dtype_to_str

class MetatrainModel(torch.nn.Module):
    __checkpoint_version__ = 1

    def __init__(
        self,
        model: ModelInterface,
        additive_models: list[ModelInterface],
        scaler: Optional[ModelInterface], #Scaler,
        dataset_info: DatasetInfo,
    ):
        super().__init__()
        self.model = model
        self.additive_models = torch.nn.ModuleList(additive_models)
        self.scaler = scaler
        self.dataset_info = dataset_info

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        return_dict = self.model(
            systems, outputs, selected_atoms=selected_atoms
        )

        with torch.profiler.record_function("MTT_WRAPPER::post-processing"):
            if not self.training:
                if self.scaler is not None:
                    # at evaluation, we also introduce the scaler and additive contributions
                    return_dict = self.scaler.apply_scales(
                        systems,
                        return_dict,
                        selected_atoms=selected_atoms,
                        use_per_target_scales=True,
                        use_per_property_scales=True,
                    )

                # For atomic basis targets, sparsify to create blocks with "atom_type"
                # in the key dimensions, and ensure properties are unpadded. This is
                # done before adding the additive contributions, which are also
                # sparsified (by the additive models themselves, in eval mode).
                # for k in atomic_predictions_dict.keys():
                #     if self.model.dataset_info.targets[k].is_atomic_basis:
                #         return_dict[k] = sparsify_atomic_basis_target(
                #             systems,
                #             return_dict[k],
                #             self.dataset_info.targets[k].layout,
                #             species,
                #         )

                for additive_model in self.additive_models:
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
        return return_dict
    
    def requested_inputs(self) -> Dict[str, ModelOutput]:
        requested_inputs = {}

        def _add_model_requested_inputs(model: ModelInterface):
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

        def _add_model_requested_neighbor_lists(model: ModelInterface):
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
        additive_models = [model.to(dtype).export(metadata) for model in self.additive_models]
        if self.scaler is not None:
            scaler = self.scaler.to(dtype).export(metadata)
        else:
            scaler = None

        # Get a list of the capabilities of each model
        all_capabilities = [model.capabilities()] + [
            model.capabilities() for model in additive_models
        ] + [scaler.capabilities()] if scaler is not None else []

        # The interaction range of the model is the maximum interaction range
        # of all the models involved.
        all_interaction_ranges = [cap.interaction_range for cap in all_capabilities if cap.interaction_range is not None]
        interaction_range = max(all_interaction_ranges) if all_interaction_ranges else None

        all_supported_devices = [cap.supported_devices for cap in all_capabilities]
        # Get the intersection of all supported devices
        supported_devices = set(all_supported_devices[0])
        for devices in all_supported_devices[1:]:
            supported_devices.intersection_update(devices)

        # Build the wrapper model again with the exported modules.
        to_export = self.__class__(
            model=model.module,
            additive_models=[model.module for model in additive_models],
            scaler=scaler.module,
            dataset_info=self.dataset_info,
        )

        if metadata is None:
            metadata = ModelMetadata()

        capabilities = ModelCapabilities(
            outputs=self.supported_outputs(),
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.model.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return AtomisticModel(to_export.eval(), metadata, capabilities)

    @staticmethod
    def _ckpt_from_arch_ckpt(checkpoint: dict) -> dict:
        """
        Convert a checkpoint from an architecture to one that can be loaded
        by MetatrainModel.
         
        This is specially useful for converting checkpoints that were generated
        before the introduction of the MetatrainModel class.
        """
        from metatrain.utils.io import trainer_from_checkpoint
        # Use the trainer to setup a MetatrainModel
        trainer = trainer_from_checkpoint(checkpoint, context="restart", hypers={})
        model_data = checkpoint["model_data"]
        mtt_model = trainer.setup(
            model_hypers=model_data.get("hypers", model_data.get("model_hypers")),
            dataset_info=model_data["dataset_info"]
        )

        # Get the new checkpoint skeleton from the MetatrainModel
        new_ckpt = mtt_model.get_checkpoint()

        # Copy all the trainer keys.
        for k in list(checkpoint):
            if k not in new_ckpt["model"]:
                new_ckpt[k] = checkpoint.pop(k)

        # The checkpoint of the model now goes to the "model" key.
        # (here we replace completely the model in new_ckpt, since that one
        # is simply an untrained model)
        new_ckpt["model"] = checkpoint

        # Fill the state dicts of the scaler and additive models by finding
        # them in the state dict of the original checkpoint.
        scaler_state_dict = {}
        additive_models_state_dict = [{}] * len(new_ckpt["additive_models"])

        state_dict = checkpoint["model_state_dict"]
        for k, v in state_dict.items():
            if k.startswith("scaler."):
                scaler_state_dict[k.replace("scaler.", "")] = v
            for i, additive_model_state_dict in enumerate(additive_models_state_dict):
                if k.startswith(f"additive_models.{i}."):
                    additive_model_state_dict[k.replace(f"additive_models.{i}.", "")] = v

        if new_ckpt["scaler"] is None:
            if len(scaler_state_dict) > 0:
                raise ValueError(
                    "The checkpoint contains a scaler state dict, but the model "
                    "does not have a scaler."
                )
        else:
            new_ckpt["scaler"]["model_state_dict"] = scaler_state_dict
            new_ckpt["scaler"]["best_model_state_dict"] = scaler_state_dict

        for i, additive_model_state_dict in enumerate(additive_models_state_dict):
            new_ckpt["additive_models"][i]["model_state_dict"] = additive_model_state_dict
            new_ckpt["additive_models"][i]["best_model_state_dict"] = additive_model_state_dict

        return new_ckpt

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "ModelInterface":
        from .io import arch_model_from_checkpoint
        if "architecture_name" in checkpoint:
            checkpoint = cls._ckpt_from_arch_ckpt(checkpoint)

        return cls(
            model=arch_model_from_checkpoint(checkpoint["model"], context=context),
            additive_models=[
                arch_model_from_checkpoint(additive_model_checkpoint, context=context)
                for additive_model_checkpoint in checkpoint["additive_models"]
            ],
            scaler=arch_model_from_checkpoint(checkpoint["scaler"], context=context) if checkpoint["scaler"] is not None else None,
            dataset_info=checkpoint["dataset_info"]
        )

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

    def get_checkpoint(self) -> Dict[str, Any]:
        """
        Get the checkpoint of the model. This should contain all the information
        needed by `load_checkpoint` to recreate the same model instance.

        :return: The model's checkpoint.
        """
        checkpoint = {
            "model_ckpt_version": self.__checkpoint_version__,
            "dataset_info": self.dataset_info,
            "model": self.model.get_checkpoint(),
            "additive_models": [m.get_checkpoint() for m in self.additive_models],
            "scaler": self.scaler.get_checkpoint() if self.scaler is not None else None,
        }
        return checkpoint
    
    def restart(self, dataset_info, model_hypers = None):
        """
        Restart the model with new dataset_info and model_hypers.
        This is used when the model is loaded from a checkpoint and the dataset_info
        has changed (e.g. when finetuning on a new dataset).

        :param dataset_info: The new dataset_info to use.
        :param model_hypers: The new model_hypers to use. If None, the current
            hypers will be used.
        """
        self.model.restart(dataset_info, model_hypers)

        if len(self.additive_models) > 0:
            composition_model = self.additive_models[0]
            self.additive_models[0] = composition_model.restart(
                dataset_info=DatasetInfo(
                    length_unit=dataset_info.length_unit,
                    atomic_types=dataset_info.atomic_types,
                    targets={
                        target_name: target_info
                        for target_name, target_info in dataset_info.targets.items()
                        if composition_model.is_valid_target(target_name, target_info)
                    },
                ),
            )

        if self.scaler is not None:
            self.scaler = self.scaler.restart(dataset_info)

        self.dataset_info = dataset_info

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
