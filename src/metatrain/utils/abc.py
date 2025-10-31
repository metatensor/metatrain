from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelMetadata,
    ModelOutput,
    System,
)

from metatrain.utils.data.dataset import Dataset, DatasetInfo


HypersType = TypeVar("HypersType")


class ModelInterface(torch.nn.Module, Generic[HypersType], metaclass=ABCMeta):
    """
    Abstract base class for a machine learning model in metatrain.

    All architectures in metatrain must be implemented as sub-class of this class,
    and implement the corresponding methods.

    :param hypers: A dictionary with the model's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    :param metadata: Metadata about the model, e.g. author, description, and
        references.
    """

    def __init__(
        self, hypers: HypersType, dataset_info: DatasetInfo, metadata: ModelMetadata
    ) -> None:
        """"""
        super().__init__()

        required_attributes = [
            "__checkpoint_version__",
            "__supported_devices__",
            "__supported_dtypes__",
            "__default_metadata__",
        ]
        for attribute in required_attributes:
            if not hasattr(self.__class__, attribute):
                raise TypeError(
                    f"missing '{attribute}' class attribute for "
                    f"'{self.__class__.__module__}.{self.__class__.__name__}'"
                )

        self.hypers = hypers
        """The model hyper passed at initialization"""

        self.dataset_info = dataset_info
        """The dataset info passed at initialization"""

        self.metadata = metadata
        """The metadata passed at initialization"""

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

        :param systems: List of systems to evaluate the model on.
        :param outputs: Dictionary of outputs that the model should compute.
        :param selected_atoms: Optional ``Labels`` specifying a subset of atoms to
            compute the outputs for. If ``None``, the outputs are computed for all
            atoms in each system.

        :return: A dictionary mapping each requested output name to the corresponding
            ``TensorMap`` containing the computed values.

        .. seealso::

            :py:class:`metatomic.torch.ModelInterface` for more explanation about the
            different arguments.
        """

    @abstractmethod
    def supported_outputs(self) -> Dict[str, ModelOutput]:
        """
        Get the outputs currently supported by this model.

        This will likely be the same outputs that are set as this model capabilities in
        :py:func:`ModelInterface.export`.

        :return: A dictionary of the supported outputs by this model.
        """

    @abstractmethod
    def restart(self, dataset_info: DatasetInfo) -> "ModelInterface":
        """
        Update a model to restart training, potentially with different dataset and/or
        targets.

        This function is called whenever training restarts, with the same or a different
        dataset. It enables transfer learning (changing the targets), and fine-tuning
        (same targets, different datasets)

        :param dataset_info: Information about the new dataset, including the targets
            that will be used for training.

        :return: The updated model, or a new instance of the model, that is able to
            handle the new dataset.
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

        :return: An instance of the model.
        """

    @abstractmethod
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

    @classmethod
    @abstractmethod
    def upgrade_checkpoint(cls, checkpoint: Dict["str", Any]) -> Dict["str", Any]:
        """
        Upgrade the checkpoint to the current version of the model.

        :param checkpoint: Checkpoint's state dictionary.

        :raises RuntimeError: if the checkpoint cannot be upgraded to the current
            version of the model.

        :return: The upgraded checkpoint.
        """

    @abstractmethod
    def get_checkpoint(self) -> Dict[str, Any]:
        """
        Get the checkpoint of the model. This should contain all the information
        needed by `load_checkpoint` to recreate the same model instance.

        :return: The model's checkpoint.
        """


class TrainerInterface(Generic[HypersType], metaclass=ABCMeta):
    """
    Abstract base class for a model trainer in metatrain.

    All architectures in metatrain must implement such a trainer, which is responsible
    for training the model. The trainer must be a be sub-class of this class, and
    implement the corresponding methods.

    :param hypers: A dictionary with the trainer's hyper-parameters.
    """

    def __init__(self, hypers: HypersType):
        required_attributes = [
            "__checkpoint_version__",
        ]
        for attribute in required_attributes:
            if not hasattr(self.__class__, attribute):
                raise TypeError(
                    f"missing '{attribute}' class attribute for "
                    f"'{self.__class__.__module__}.{self.__class__.__name__}'"
                )

        self.__dict__["__intialized"] = True

        self.hypers = hypers
        """The trainer hypers passed at intialization"""

    def __setattr__(self, name: str, value: Any) -> None:
        if not hasattr(self, "__intialized") or not self.__dict__["__intialized"]:
            raise ValueError(
                "you must call `super().__init__(hypers)` before setting new fields"
            )
        super().__setattr__(name, value)

    @abstractmethod
    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
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
    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        """
        Save a checkoint of both the ``model`` and trainer state to the given ``path``

        :param model: The model to save in the checkpoint.
        :param path: The path where to save the checkpoint.
        """

    @classmethod
    @abstractmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        """
        Upgrade the checkpoint to the current version of the trainer.

        :param checkpoint: Checkpoint's state dictionary.

        :raises RuntimeError: if the checkpoint cannot be upgraded to the current
            version of the trainer.

        :return: The upgraded checkpoint.
        """

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: HypersType,
        context: Literal["restart", "finetune"],
    ) -> "TrainerInterface":
        """
        Create a trainer instance from data stored in the ``checkpoint``.

        :param checkpoint: Checkpoint's state dictionary.
        :param hypers: Hyper-parameters for the trainer, as specified by the user.
        :param context: Context in which to load the model. Possible values are
            ``"restart"`` when restarting a stopped traininf run, and ``"finetune"``
            when loading a model for further fine-tuning or transfer learning. When
            multiple checkpoints are stored together, this can be used to pick one of
            them depending on the context.

        :return: The loaded trainer instance.
        """
