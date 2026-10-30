"""A minimal model with the interface LLPR relies on, to test LLPR without the
architectures."""

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_features_info
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.sum_over_atoms import sum_over_atoms


class DummyModel(ModelInterface):
    """Each block of each target is a linear function of the last-layer features:
    radial functions of the distance of the atoms from the center of their system,
    times the spherical harmonics of the block (up to lambda=1). With ``shared``, the
    features are the radial functions only, shared by all the blocks. With ``labelled``
    too, each block reads its own power of the radial functions instead, labelled by
    its key in a single block of features. With ``shared`` and ``per_type``, the last
    layer has different weights for each atomic type.

    :param hypers: ``num_features``, ``shared``, ``labelled``, ``per_type``, and
        whether to expose ``last_layer_features``.
    :param dataset_info: the dataset info.
    """

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers: Dict[str, Any], dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        self.shared = bool(hypers["shared"])
        self.labelled = bool(hypers["labelled"])
        self.per_type = bool(hypers["per_type"])
        self.register_buffer(
            "atomic_types", torch.tensor(dataset_info.atomic_types), persistent=False
        )
        self.num_features = int(hypers["num_features"])
        self.register_buffer("centers", torch.linspace(0.0, 2.0, self.num_features))

        self.outputs: Dict[str, TargetInfo] = {}
        self.layouts: Dict[str, TensorMap] = {}
        self.last_layers = torch.nn.ModuleDict()
        for name, target in dataset_info.targets.items():
            self.outputs[name] = target
            if hypers["last_layer_features"]:
                self.outputs[_features_name(name)] = self._features_info(target)
            self.layouts[name] = target.layout
            # shared features map to all the components of a block at once
            self.last_layers[name] = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.num_features,
                        (len(dataset_info.atomic_types) if self.per_type else 1)
                        * (
                            math.prod(block.values.shape[1:])
                            if self.shared
                            else block.values.shape[-1]
                        ),
                        bias=False,
                    )
                    for block in target.layout.blocks()
                ]
            )

    def _features_info(self, target: TargetInfo) -> TargetInfo:
        """The info of the last-layer features of ``target``.

        :param target: the info of the target.
        :return: the info of its last-layer features.
        """
        if self.shared and self.labelled:
            properties = _labelled_features(target.layout.keys, self.num_features)
            layout = TensorMap(
                Labels.single(),
                [
                    TensorBlock(
                        values=torch.empty(0, len(properties), dtype=torch.float64),
                        samples=Labels.empty(["system", "atom"]),
                        components=[],
                        properties=properties,
                    )
                ],
            )
            return TargetInfo(layout)
        if self.shared:
            return get_features_info(self.num_features)
        layout = TensorMap(
            target.layout.keys,
            [
                TensorBlock(
                    values=torch.empty(
                        list(block.values.shape[:-1]) + [self.num_features],
                        dtype=torch.float64,
                    ),
                    samples=Labels.empty(["system", "atom"]),
                    components=block.components,
                    properties=Labels.range("feature", self.num_features),
                )
                for block in target.layout.blocks()
            ],
        )
        return TargetInfo(layout)

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return self.outputs

    def last_layer_depends_on_atomic_type(self, target_name: str) -> bool:
        return self.per_type

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        self.layouts = {k: v.to(device) for k, v in self.layouts.items()}

        samples_values: List[torch.Tensor] = []
        vectors: List[torch.Tensor] = []
        for index, system in enumerate(systems):
            atoms = torch.arange(len(system), device=device)
            samples_values.append(
                torch.stack([torch.full_like(atoms, index), atoms], 1)
            )
            vectors.append(system.positions - system.positions.mean(dim=0))
        samples = Labels(["system", "atom"], torch.cat(samples_values).to(torch.int32))
        type_indices = torch.searchsorted(
            self.atomic_types, torch.cat([system.types for system in systems])
        )
        vector = torch.cat(vectors)
        distance = torch.linalg.norm(vector, dim=1, keepdim=True)
        radial = torch.exp(-((distance - self.centers) ** 2))
        # the lambda=1 spherical harmonics, in the order of o3_mu
        harmonics = (vector / distance)[:, [1, 2, 0]]
        feature_labels = Labels(
            ["feature"], torch.arange(self.num_features, device=device).reshape(-1, 1)
        )

        results: Dict[str, TensorMap] = {}
        for name, last_layers in self.last_layers.items():
            layout = self.layouts[name]
            targets: List[TensorBlock] = []
            features: List[TensorBlock] = []
            labelled_features: List[torch.Tensor] = []
            for index, last_layer in enumerate(last_layers):
                block = layout.block(index)
                shape = [len(samples)] + [len(c) for c in block.components]
                if self.shared and self.labelled:
                    block_radial = radial ** (index + 1)
                    values = last_layer(block_radial).reshape(shape + [-1])
                    labelled_features.append(block_radial)
                elif self.shared:
                    values = last_layer(radial)
                    if self.per_type:
                        # the weights of the type of each atom
                        values = values.reshape(
                            len(samples), len(self.atomic_types), -1
                        )
                        values = values[torch.arange(len(samples)), type_indices]
                    values = values.reshape(shape + [-1])
                else:
                    angular = (
                        harmonics if shape[1:] == [3] else torch.ones_like(distance)
                    )
                    block_features = angular.unsqueeze(2) * radial.unsqueeze(1)
                    block_features = block_features.reshape(shape + [-1])
                    values = last_layer(block_features)
                    features.append(
                        TensorBlock(
                            block_features, samples, block.components, feature_labels
                        )
                    )
                targets.append(
                    TensorBlock(values, samples, block.components, block.properties)
                )
            if self.shared and self.labelled:
                features = [
                    TensorBlock(
                        torch.cat(labelled_features, dim=1),
                        samples,
                        [],
                        _labelled_features(layout.keys, self.num_features),
                    )
                ]
            elif self.shared:
                features = [TensorBlock(radial, samples, [], feature_labels)]
            if self.shared:
                feature_keys = Labels(
                    ["_"], torch.zeros((1, 1), dtype=torch.int32, device=device)
                )
            else:
                feature_keys = layout.keys

            for output_name, tensor in [
                (name, TensorMap(layout.keys, targets)),
                (_features_name(name), TensorMap(feature_keys, features)),
            ]:
                if output_name in outputs:
                    if outputs[output_name].sample_kind == "system":
                        tensor = sum_over_atoms(tensor)
                    results[output_name] = tensor
        return results

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> "DummyModel":
        return self

    def get_checkpoint(
        self, best_model_state_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "architecture_name": "dummy",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {"hypers": self.hypers, "dataset_info": self.dataset_info},
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": self.state_dict(),
        }

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "DummyModel":
        model = cls(**checkpoint["model_data"])
        dtype = checkpoint["model_state_dict"]["centers"].dtype
        model.to(dtype).load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: Dict[str, Any], version: Optional[int] = None
    ) -> Dict[str, Any]:
        return checkpoint

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        capabilities = ModelCapabilities(
            outputs={
                name: info.to_model_output() for name, info in self.outputs.items()
            },
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=float("inf"),
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(self.centers.dtype),
        )
        return AtomisticModel(self.eval(), ModelMetadata(), capabilities)


def _labelled_features(keys: Labels, num_features: int) -> Labels:
    """The properties of features made of ``num_features`` features for each key.

    :param keys: the keys of the target.
    :param num_features: the number of features of each key.
    :return: the properties, named after the keys followed by ``"feature"``.
    """
    features = torch.arange(num_features, dtype=torch.int32, device=keys.device)
    values: List[torch.Tensor] = []
    for key in keys.values:
        values.append(
            torch.cat([key.expand(num_features, -1), features.unsqueeze(1)], dim=1)
        )
    return Labels(keys.names + ["feature"], torch.cat(values))


def _features_name(target_name: str) -> str:
    return f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
