from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import AtomisticModel, ModelMetadata, ModelOutput, System

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import Dataset, DatasetInfo


class MyTrainer(TrainerInterface):
    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        raise NotImplementedError()

    def save_checkpoint(self, model, path: Union[str, Path]):
        raise NotImplementedError()

    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict) -> Dict:
        raise NotImplementedError()

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: Dict[str, Any],
        context: Literal["restart", "finetune"],
    ) -> "TrainerInterface":
        raise NotImplementedError()


def test_trainer_interface():
    message = (
        "missing '__checkpoint_version__' class attribute for "
        "'utils.test_abc.MyTrainer'"
    )
    with pytest.raises(TypeError, match=message):
        _ = MyTrainer({})

    def init(self, hypers):
        self.hypers = hypers

    MyTrainer.__init__ = init

    message = (
        "you must call `super\\(\\).__init__\\(hypers\\)` before setting new fields"
    )
    with pytest.raises(ValueError, match=message):
        _ = MyTrainer({})


class MyModel(ModelInterface):
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        raise NotImplementedError()

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        raise NotImplementedError()

    def restart(self, dataset_info: DatasetInfo) -> ModelInterface:
        raise NotImplementedError()

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> ModelInterface:
        raise NotImplementedError()

    def export(
        self,
        metadata: Optional[ModelMetadata] = None,
    ) -> AtomisticModel:
        raise NotImplementedError()

    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict["str", Any]) -> Dict["str", Any]:
        raise NotImplementedError()

    def get_checkpoint(self) -> Dict[str, Any]:
        raise NotImplementedError()


def test_model_interface():
    EXPECTED_ATTRS = [
        "__checkpoint_version__",
        "__supported_devices__",
        "__supported_dtypes__",
        "__default_metadata__",
    ]

    for attr in EXPECTED_ATTRS:
        message = f"missing '{attr}' class attribute for 'utils.test_abc.MyModel'"
        with pytest.raises(TypeError, match=message):
            _ = MyModel({}, DatasetInfo("", [], {}))

        setattr(MyModel, attr, None)
