from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelMetadata,
    ModelOutput,
    System,
)

from metatrain.utils.data.dataset import Dataset, DatasetInfo


class ModelInterface(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for a machine learning model in metatrain.

    All architectures in metatrain must be implemented as sub-class of this class, and
    implement the corresponding methods.
    """

    def __init__(self):
        """"""
        super().__init__()

        required_attributes = [
            "__supported_devices__",
            "__supported_dtypes__",
            "__default_metadata__",
        ]
        for attribute in required_attributes:
            if not hasattr(self.__class__, attribute):
                raise TypeError(
                    f"missing '{attribute}' class attribute for "
                    f"{self.__class__.__name__}"
                )

    @abstractmethod
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Execute the model for the given ``systems``, computing the requested
        ``outputs``.

        .. seealso::

            :py:class:`metatensor.torch.atomistic.ModelInterface` for more explanation
            about the different arguments.
        """

    @abstractmethod
    def supported_outputs(self) -> Dict[str, ModelOutput]:
        """
        Get the outputs currently supported by this model.

        This will likely be the same outputs that are set as this model capabilities in
        :py:func:`ModelInterface.export`.
        """

    @abstractmethod
    def restart(self, dataset_info: DatasetInfo) -> "ModelInterface":
        """
        Update a model to restart training, potentially with different dataset and/or
        targets.

        This function is called whenever training restarts, with the same or a different
        dataset. It enables transfer learning (changing the targets), and fine-tuning
        (same targets, different datasets)

        This function should return the updated model, or a new instance of the model
        able to handle the new dataset.
        """

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "ModelInterface":
        """
        Create a model from a checkpoint (i.e. state dictionary).

        :param checkpoint: Checkpoint's state dictionary.
        :param context: Context in which to load the model. Possible values are
            ``"restart"`` when restarting a stopped traininf run, ``"finetune"`` when
            loading a model for further fine-tuning or transfer learning, and
            ``"export"`` when loading a model for final export. When multiple
            checkpoints are stored together, this can be used to pick one of them
            depending on the context.
        """

    @abstractmethod
    def export(
        self,
        metadata: Optional[ModelMetadata] = None,
    ) -> MetatensorAtomisticModel:
        """
        Turn this model into an instance of
        :py:class:`metatensor.torch.atomistic.MetatensorAtomisticModel`, containing the
        model itself, a definition of the model capabilities and some metadata about the
        model.

        :param metadata: additional metadata to add in the model as specified by the
            user.
        """


class TrainerInterface(metaclass=ABCMeta):
    """
    Abstract base class for a model trainer in metatrain.

    All architectures in metatrain must implement such a trainer, which is responsible
    for training the model. The trainer must be a be sub-class of this class, and
    implement the corresponding methods.
    """

    @abstractmethod
    def __init__(self, train_hypers):
        """
        Create a trainer using the hyper-parameters in ``train_hypers``.
        """

    @abstractmethod
    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        """
        Train the ``model`` using the ``train_datasets``. How to train the model is left
        to this class, using the hyper-parameter given in ``__init__``.

        :param model: the model to train
        :param dtype: ``torch.dtype`` used by the data in the datasets
        :param devices: ``torch.device`` to use for training the model. When training
            with more than one device (e.g. multi-GPU training), this can contains
            multiple devices.
        :param train_datasets: datasets to use to train the model
        :param val_datasets: datasets to use for model validation
        :param checkpoint_dir: directory where checkpoints shoudl be saved
        """

    @abstractmethod
    def save_checkpoint(self, model, path: Union[str, Path]):
        """
        Save a checkoint of both the ``model`` and trainer state to the given ``path``
        """

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        train_hypers: Dict[str, Any],
        context: Literal["restart", "finetune"],
    ) -> "TrainerInterface":
        """
        Create a trainer instance from data stored in the ``checkpoint``.

        :param checkpoint: Checkpoint's state dictionary.
        :param train_hypers: Hyper-parameters for the trainer, as specified by the user.
        :param context: Context in which to load the model. Possible values are
            ``"restart"`` when restarting a stopped traininf run, and ``"finetune"``
            when loading a model for further fine-tuning or transfer learning. When
            multiple checkpoints are stored together, this can be used to pick one of
            them depending on the context.
        """
