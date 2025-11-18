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

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.per_atom import divide_by_num_atoms
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers


class Classifier(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1

    # all torch devices and dtypes are supported, if they are supported by the wrapped
    # model; the check is performed in the trainer
    __supported_devices__ = ["cuda", "cpu", "mps"]
    __supported_dtypes__ = [torch.float32, torch.float64, torch.bfloat16, torch.float16]

    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "Classifier (transfer learning): implemented in metatrain",
            ],
        }
    )

    """A classifier model that trains on top of a pre-trained backbone.

    This model takes a pre-trained checkpoint, freezes its backbone, and trains
    a multi-layer perceptron on top of the features extracted from the backbone.
    The targets should be class labels specified as floats (0.0, 1.0, 2.0, etc.).
    The loss function is a negative log-likelihood (NLL) classification loss.

    :param hypers: Model hyperparameters
    :param dataset_info: Dataset information
    """

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.hypers = hypers
        self.dataset_info = dataset_info
        self.model: Optional[ModelInterface] = None

    def set_wrapped_model(self, model: ModelInterface) -> None:
        """Set and freeze the wrapped pre-trained model.

        :param model: The pre-trained model to wrap
        """
        self.model = model

        # Freeze the backbone model
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the capabilities from the wrapped model
        old_capabilities = self.model.export().capabilities()

        # Check compatibility between dataset_info and model outputs
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

        # Check that the model can output features
        if "features" not in old_capabilities.outputs:
            raise ValueError(
                "The wrapped model does not support 'features' output. "
                "The Classifier model requires a backbone that can output features."
            )

        # Get the feature size from the wrapped model
        # We'll determine this during the first forward pass
        self.feature_size: Optional[int] = None

        # Build the MLP classifier
        # We'll build this after we know the feature size
        self.mlp: Optional[torch.nn.Module] = None

        # Store capabilities
        self.capabilities = ModelCapabilities(
            outputs=self.dataset_info.targets,
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

    def _build_mlp(self, input_size: int, num_classes: int, dtype: torch.dtype) -> None:
        """Build the MLP classifier.

        :param input_size: Size of the input features
        :param num_classes: Number of output classes
        :param dtype: Data type for the MLP parameters
        """
        layers = []
        current_size = input_size

        # Hidden layers (the last one acts as a bottleneck for feature extraction)
        for hidden_size in self.hypers["hidden_sizes"]:
            layers.append(torch.nn.Linear(current_size, hidden_size, dtype=dtype))
            layers.append(torch.nn.ReLU())
            current_size = hidden_size

        # Final classification layer
        layers.append(torch.nn.Linear(current_size, num_classes, dtype=dtype))

        self.mlp = torch.nn.Sequential(*layers)
        self.feature_size = input_size

    def restart(self, dataset_info: DatasetInfo) -> "ModelInterface":
        raise ValueError("Restarting from a Classifier model is not supported.")

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if self.model is None:
            raise ValueError(
                "Wrapped model not set. Call set_wrapped_model() before forward()."
            )

        # Request features from the wrapped model (per-atom features)
        features_output = ModelOutput(
            quantity="",
            unit="",
            per_atom=True,  # Request per-atom features
        )
        features_dict = self.model(
            systems, {"features": features_output}, selected_atoms
        )

        # Sum over atoms first, then average to get system-level features
        system_features = sum_over_atoms(features_dict["features"])
        num_atoms = torch.tensor(
            [len(s) for s in systems],
            device=system_features.block().values.device,
        )
        averaged_features = divide_by_num_atoms(system_features, num_atoms)
        features = averaged_features.block().values

        # Build MLP if not already built
        if self.mlp is None:
            feature_size = features.shape[-1]
            self.feature_size = feature_size
            # MLP not built yet, return empty dict
            # This will happen during training initialization
            return {}

        # Forward through MLP
        logits = self.mlp(features)

        # Create output TensorMap
        return_dict = {}
        for name in outputs:
            # Create TensorMap with logits
            # For classification, we output logits for each class
            output_tmap = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor([[0]], device=logits.device),
                ),
                blocks=[
                    TensorBlock(
                        values=logits,
                        samples=system_features.block().samples,
                        components=[],
                        properties=Labels(
                            names=["class"],
                            values=torch.arange(
                                logits.shape[-1], device=logits.device
                            ).reshape(-1, 1),
                            assume_unique=True,
                        ),
                    )
                ],
            )
            return_dict[name] = output_tmap

        return return_dict

    def get_checkpoint(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Cannot get checkpoint: wrapped model not set")

        wrapped_model_checkpoint = self.model.get_checkpoint()
        state_dict = {
            k: v for k, v in self.state_dict().items() if not k.startswith("model.")
        }
        checkpoint = {
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "architecture_name": "experimental.classifier",
            "model_ckpt_version": self.__checkpoint_version__,
            "wrapped_model_checkpoint": wrapped_model_checkpoint,
            "state_dict": state_dict,
            "feature_size": self.feature_size,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "Classifier":
        model = model_from_checkpoint(checkpoint["wrapped_model_checkpoint"], context)
        if context == "finetune":
            return model
        elif context == "restart":
            raise NotImplementedError(
                "Restarting from the Classifier checkpoint is not supported. "
                "Please consider finetuning the model, or just export it "
                "in the TorchScript format for final usage."
            )
        elif context == "export":
            classifier_model = cls(**checkpoint["model_data"])
            classifier_model.set_wrapped_model(model)
            dtype = next(model.parameters()).dtype
            classifier_model.to(dtype).load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            return classifier_model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        if self.model is None:
            raise ValueError("Cannot export: wrapped model not set")

        dtype = next(self.parameters()).dtype

        # Make sure the model is all in the same dtype
        self.to(dtype)

        metadata = merge_metadata(
            merge_metadata(self.__default_metadata__, metadata),
            self.model.export().metadata(),
        )

        return AtomisticModel(self.eval(), metadata, self.capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.dataset_info.targets
