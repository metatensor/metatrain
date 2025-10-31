"""Trainer for the example ZeroModel."""

from metatrain.utils.mlip import MLIPTrainer


class ZeroTrainer(MLIPTrainer):
    """
    Trainer for the ZeroModel.

    This trainer uses the base MLIPTrainer without rotational augmentation.

    :param hypers: Training hyperparameters.
    """

    __checkpoint_version__ = 1

    def use_rotational_augmentation(self) -> bool:
        """
        Specify whether the trainer should use rotational augmentation.

        For this example, we do not use rotational augmentation.

        :return: False, indicating no rotational augmentation.
        """
        return False
