from abc import ABCMeta, abstractmethod
from typing import Generic, Optional, TypeVar

import torch
from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo


HypersType = TypeVar("HypersType")


class HookInterface(torch.nn.Module, Generic[HypersType], metaclass=ABCMeta):
    """
    Abstract base class for a hook in metatrain.

    All hooks in metatrain must be implemented as sub-class of this class,
    and implement the corresponding methods.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__: int
    """The current version of the model's checkpoint.

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
        """The hook hypers passed at initialization"""

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
        Execute the model for the given ``systems``, computing the requested
        ``outputs``, and using the inputs

        :param systems: List of systems to evaluate the model on.
        :param outputs: Dictionary of outputs that the model should compute.
        :param inputs: Dictionary of input tensors for the model.
        :param selected_atoms: Optional ``Labels`` specifying a subset of atoms to
            compute the outputs for. If ``None``, the outputs are computed for all
            atoms in each system.

        :return: A dictionary mapping each requested output name to the corresponding
            ``TensorMap`` containing the computed values.
        """

    @abstractmethod
    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the inputs that the hook requires, with their layout.

        :return: The requested inputs. Each key is the name of the requested input,
          and the value is a :py:class:`TargetInfo` object describing the layout of
          the requested input.
        """

    @abstractmethod
    def requested_inputs(self, outputs: Optional[dict[str, ModelOutput]] = None) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
          are handled by other hooks or the main model. In that case, the hook should
          ignore those outputs.
        :return: A list of requested input names.
        """


    @abstractmethod
    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Get the outputs currently supported by this hook.

        :return: A dictionary of the supported outputs by this hook.
        """
