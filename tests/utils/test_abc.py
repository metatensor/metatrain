from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import AtomisticModel, ModelMetadata, ModelOutput, System

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import Dataset, DatasetInfo


class MyTrainer(TrainerInterface):
    def setup(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> Any:  # "MetatrainModel"
        """
        Setup the trainer and return an initialized model ready for training.

        :param model_hypers: The hyper-parameters of the model to be trained.
        :param dataset_info: Information about the dataset to be used for training.

        :return: An initialized MetatrainModel ready for training.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support the `setup` method. "
        )

    def restart(
        self,
        model: Any,  # MetatrainModel
        dataset_info: DatasetInfo,
        model_hypers: dict,
    ) -> Any:  # MetatrainModel
        """Restart a model for training with a new dataset info.

        :param model: The model to be restarted.
        :param dataset_info: Information about the new dataset to be used for training.
        :param model_hypers: The new hyperparameters to set for the model.

        :return: The restarted model, ready for training with the new dataset info.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support the `restart` method. "
        )

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
    def upgrade_checkpoint(checkpoint: Dict, version: int | None = None) -> Dict:
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
        "'tests.utils.test_abc.MyTrainer'"
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

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> ModelInterface:
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
    def upgrade_checkpoint(
        checkpoint: Dict["str", Any], version: int | None = None
    ) -> Dict["str", Any]:
        raise NotImplementedError()

    def get_checkpoint(
        self, best_model_state_dict: dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        raise NotImplementedError()


def test_model_interface():
    EXPECTED_ATTRS = [
        "__checkpoint_version__",
        "__supported_devices__",
        "__supported_dtypes__",
        "__default_metadata__",
    ]

    for attr in EXPECTED_ATTRS:
        message = f"missing '{attr}' class attribute for 'tests.utils.test_abc.MyModel'"
        with pytest.raises(TypeError, match=message):
            _ = MyModel({}, DatasetInfo("", [], {}), ModelMetadata())

        setattr(MyModel, attr, None)


def test_model_interface_default_remove_output():
    """Architectures that do not override ``remove_output`` should raise
    ``NotImplementedError``, since not every architecture supports it."""
    for attr in [
        "__checkpoint_version__",
        "__supported_devices__",
        "__supported_dtypes__",
        "__default_metadata__",
    ]:
        setattr(MyModel, attr, None)

    model = MyModel({}, DatasetInfo("", [], {}), ModelMetadata())

    with pytest.raises(NotImplementedError, match="does not support"):
        model.remove_output("some_target")
