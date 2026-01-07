from typing import Any, Dict, List, Literal, Optional

import metatensor.torch as mts
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
    The targets should be class probabilities as vectors, supporting both one-hot
    encodings (e.g., [1.0, 0.0, 0.0]) and soft/fractional targets (e.g., [0.7, 0.2,
    0.1]). The loss function is a standard cross-entropy loss for classification.

    :param hypers: Model hyperparameters
    :param dataset_info: Dataset information
    """

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.dataset_info = dataset_info
        self.hidden_sizes = hypers["hidden_sizes"]
        self.feature_layer_index = hypers["feature_layer_index"]

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

        # Store capabilities
        outputs = {name: ModelOutput() for name in self.dataset_info.targets.keys()}
        outputs["features"] = ModelOutput(quantity="", unit="", per_atom=False)
        self.capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=old_capabilities.atomic_types,
            interaction_range=old_capabilities.interaction_range,
            length_unit=old_capabilities.length_unit,
            supported_devices=old_capabilities.supported_devices,
            dtype=old_capabilities.dtype,
        )

    def build_mlp(self, feature_size: int, num_classes: int) -> None:
        """Build the MLP classifier based on the feature size."""
        # Use a ModuleList to allow accessing intermediate layers for feature extraction
        self.mlp = torch.nn.ModuleList()
        current_size = feature_size
        n_layers = len(self.hidden_sizes)

        for i, hidden_size in enumerate(self.hidden_sizes):
            block_layers = []
            # Add LayerNorm if it's not the last layer
            if i != n_layers - 1:
                block_layers.append(torch.nn.LayerNorm(current_size))

            block_layers.append(torch.nn.Linear(current_size, hidden_size))
            block_layers.append(torch.nn.SiLU())

            self.mlp.append(torch.nn.Sequential(*block_layers))
            current_size = hidden_size

        # Final classification layer
        self.linear = torch.nn.Linear(current_size, num_classes, bias=False)

    def restart(self, dataset_info: DatasetInfo) -> "Classifier":
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

        return_dict: Dict[str, TensorMap] = {}

        # Request features from the wrapped model (per-atom features)
        features_output = ModelOutput(per_atom=True)
        features_dict = self.model(
            systems, {"features": features_output}, selected_atoms
        )

        # Average over atoms to get system-level features
        averaged_features = mts.mean_over_samples(
            features_dict["features"], sample_names=["atom"]
        )
        features = averaged_features.block().values

        # Resolve the feature layer index
        feature_layer_index = self.feature_layer_index
        num_layers = len(self.mlp)
        if feature_layer_index < 0:
            feature_layer_index += num_layers

        if not (0 <= feature_layer_index < num_layers):
            raise ValueError(
                f"feature_layer_index {self.feature_layer_index} "
                f"is out of bounds for an MLP with {num_layers} layers."
            )

        # Forward through MLP layers
        current_tensor = features
        features_for_output = current_tensor

        for i, layer in enumerate(self.mlp):
            current_tensor = layer(current_tensor)
            if i == feature_layer_index:
                features_for_output = current_tensor

        # Perform classification on the final output
        logits = self.linear(current_tensor)

        # Store features output if requested
        if "features" in outputs:
            output_tmap = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor([[0]], device=features_for_output.device),
                ),
                blocks=[
                    TensorBlock(
                        values=features_for_output,
                        samples=averaged_features.block().samples,
                        components=[],
                        properties=Labels(
                            names=["feature"],
                            values=torch.arange(
                                features_for_output.shape[-1],
                                device=features_for_output.device,
                            ).reshape(-1, 1),
                            assume_unique=True,
                        ),
                    )
                ],
            )
            return_dict["features"] = output_tmap

        # Handle logits output (for training with CrossEntropyLoss)
        for name in outputs:
            if name == "features":
                continue  # Skip features output
            
            # Check if logits are requested (for training)
            if "logits" in name:
                # Return raw logits for CrossEntropyLoss
                output_tmap = TensorMap(
                    keys=Labels(
                        names=["_"],
                        values=torch.tensor([[0]], device=logits.device),
                    ),
                    blocks=[
                        TensorBlock(
                            values=logits,
                            samples=averaged_features.block().samples,
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
            else:
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                # Return probabilities for prediction
                output_tmap = TensorMap(
                    keys=Labels(
                        names=["_"],
                        values=torch.tensor([[0]], device=probabilities.device),
                    ),
                    blocks=[
                        TensorBlock(
                            values=probabilities,
                            samples=averaged_features.block().samples,
                            components=[],
                            properties=Labels(
                                names=["class"],
                                values=torch.arange(
                                    probabilities.shape[-1], device=probabilities.device
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
            raise NotImplementedError(
                "Finetuning from the Classifier checkpoint is not supported. "
                "Please consider restarting from the backbone model checkpoint, "
                "and then training the Classifier on top of it."
            )
        elif context == "restart":
            raise NotImplementedError(
                "Restarting from the Classifier checkpoint is not supported. "
                "Please consider finetuning the model, or just export it "
                "in the TorchScript format for final usage."
            )
        elif context == "export":
            classifier_model = cls(**checkpoint["model_data"])
            classifier_model.set_wrapped_model(model)

            state_dict = checkpoint["state_dict"]
            input_feat_size = None

            # Check the first layer of the first block (mlp.0.0)
            if "mlp.0.0.weight" in state_dict:
                w = state_dict["mlp.0.0.weight"]
                # If LayerNorm (1D), the size is shape[0]
                # If Linear (2D), the input size is shape[1]
                input_feat_size = w.shape[0] if w.ndim == 1 else w.shape[1]

            if input_feat_size is None:
                raise ValueError(
                    "Could not detect input feature size from checkpoint state_dict."
                )

            num_classes = state_dict["linear.weight"].shape[0]

            classifier_model.build_mlp(input_feat_size, num_classes)

            dtype = next(model.parameters()).dtype
            classifier_model.to(dtype).load_state_dict(state_dict, strict=False)
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
        # Currently at version 1, no upgrades needed yet
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.dataset_info.targets
