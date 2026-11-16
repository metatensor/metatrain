from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo


HypersType = TypeVar("HypersType")


class ExtenderInterface(torch.nn.Module, Generic[HypersType], metaclass=ABCMeta):
    """
    Abstract base class for an extender in metatrain.

    Extenders are a way to enhance the functionality of architectures in
    metatrain. They implement functions that are modular and therefore
    can be applied to any architecture that supports extenders. Extenders
    can, for example, allow an architecture to produce an output that it
    wouldn't be able to produce natively.

    Extenders communicate with models and other architectures purely by
    exchanging data on which inputs and outputs they support. On forward,
    an extender receives all the inputs that it has requested and should
    produce the outputs it has been asked to compute.

    All extenders in metatrain must be implemented as a sub-class of this class,
    and implement the corresponding methods.

    :param hypers: A dictionary with the extender's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__: int
    """The current version of the extender's checkpoint.

    This is used to upgrade checkpoints produced with earlier versions of the code.
    See :ref:`ckpt_version` for more information."""

    def __init__(
        self,
        hypers: HypersType,
        dataset_info: DatasetInfo,
    ) -> None:
        """"""
        super().__init__()

        required_attributes = [
            "__checkpoint_version__",
        ]
        for attribute in required_attributes:
            if not hasattr(self.__class__, attribute):
                raise TypeError(
                    f"missing '{attribute}' class attribute for "
                    f"'{self.__class__.__module__}.{self.__class__.__name__}'"
                )

        self.hypers = hypers
        """The extender hypers passed at initialization"""

        self.dataset_info = dataset_info
        """The dataset info passed at initialization"""

    @abstractmethod
    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        """
        Execute the extender for the given ``systems``, computing the requested
        ``outputs``, and using the inputs

        :param systems: List of systems to evaluate the extender on.
        :param outputs: Dictionary of outputs that the extender should compute.
        :param inputs: Dictionary of input tensors for the extender.
        :param selected_atoms: Optional ``Labels`` specifying a subset of atoms to
            compute the outputs for. If ``None``, the outputs are computed for all
            atoms in each system.

        :return: A dictionary mapping each requested output name to the corresponding
            ``TensorMap`` containing the computed values.
        """

    @abstractmethod
    def requested_extender_inputs(
        self, outputs: Optional[dict[str, ModelOutput]] = None
    ) -> dict[str, TargetInfo]:
        """
        Returns the inputs that the extender requires to produce some outputs.

        :param outputs: Dictionary of the outputs that will be
           requested to the extender. If ``None``, the extender
           should return the inputs that it needs to produce all
           the outputs.
        :return: A dictionary with all the required inputs.
        """

    @abstractmethod
    def supported_outputs(self) -> dict[str, TargetInfo]:
        """
        Get the outputs currently supported by this extender.

        :return: A dictionary of the supported outputs by this extender.
        """

    @abstractmethod
    def get_checkpoint(
        self, best_state_dict: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Get the checkpoint of the extender.

        :param best_state_dict: Optional dictionary containing the best state
            of the extender. If provided and it makes sense for the extender, it should
            be stored in the checkpoint as ``"best_model_state_dict"``.

        :return: A dictionary containing the state of the extender.
        """
