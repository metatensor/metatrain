import logging
from typing import Dict, Tuple

import numpy as np
from metatensor.torch.atomistic import ModelCapabilities


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MetricLogger:
    """This class provides a simple interface to log training metrics to a file."""

    def __init__(
        self,
        model_capabilities: ModelCapabilities,
        train_loss_0: float,
        validation_loss_0: float,
        train_info_0: Dict[str, float],
        validation_info_0: Dict[str, float],
    ):
        """
        Initialize the logger with metrics that are supposed to
        decrease during training.

        In this way, the logger can align the output to make it easier to read.

        :param model_capabilities: The capabilities of the model.
        :param train_loss_0: The initial training loss.
        :param validation_loss_0: The initial validation loss.
        :param train_info_0: The initial training metrics.
        :param validation_info_0: The initial validation metrics.
        """

        # Since the quantities are supposed to decrease, we want to store the
        # number of digits at the start of the training, so that we can align
        # the output later:
        self.digits = {}
        self.digits["train_loss"] = _get_digits(train_loss_0)
        self.digits["validation_loss"] = _get_digits(validation_loss_0)
        for name, information_holder in zip(
            ["train", "valid"], [train_info_0, validation_info_0]
        ):
            for key, value in information_holder.items():
                self.digits[f"{name}_{key}"] = _get_digits(value)

        # This will be useful later for printing forces/virials/stresses:
        energy_counter = 0
        for output in model_capabilities.outputs.values():
            if output.quantity == "energy":
                energy_counter += 1
        if energy_counter == 1:
            self.only_one_energy = True
        else:
            self.only_one_energy = False

        # Save the model capabilities for later use:
        self.model_capabilities = model_capabilities

    def log(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        train_info: Dict[str, float],
        validation_info: Dict[str, float],
    ):
        """
        Log the training metrics.

        The training metrics are automatically aligned to make them easier to read,
        based on the order of magnitude of each metric at the start of the training.

        :param epoch: The current epoch.
        :param train_loss: The current training loss.
        :param validation_loss: The current validation loss.
        :param train_info: The current training metrics.
        :param validation_info: The current validation metrics.
        """

        # The epoch is printed with 4 digits, assuming that the training
        # will not last more than 9999 epochs
        logging_string = (
            f"Epoch {epoch:4}, train loss: "
            f"{train_loss:{self.digits['train_loss'][0]}.{self.digits['train_loss'][1]}f}, "  # noqa: E501
            f"validation loss: "
            f"{validation_loss:{self.digits['validation_loss'][0]}.{self.digits['validation_loss'][1]}f}"  # noqa: E501
        )
        for name, information_holder in zip(
            ["train", "valid"], [train_info, validation_info]
        ):
            for key, value in information_holder.items():
                new_key = key
                if key.endswith("_positions_gradients"):
                    # check if this is a force
                    target_name = key[: -len("_positions_gradients")]
                    if (
                        self.model_capabilities.outputs[target_name].quantity
                        == "energy"
                    ):
                        # if this is a force, replace the ugly name with "force"
                        if self.only_one_energy:
                            new_key = "force"
                        else:
                            new_key = f"force[{target_name}]"
                elif key.endswith("_strain_gradients"):
                    # check if this is a virial/stress
                    target_name = key[: -len("_strain_gradients")]
                    if (
                        self.model_capabilities.outputs[target_name].quantity
                        == "energy"
                    ):
                        # if this is a virial/stress,
                        # replace the ugly name with "virial/stress"
                        if self.only_one_energy:
                            new_key = "virial/stress"
                        else:
                            new_key = f"virial/stress[{target_name}]"
                logging_string += (
                    f", {name} {new_key} RMSE: "
                    f"{value:{self.digits[f'{name}_{key}'][0]}.{self.digits[f'{name}_{key}'][1]}f}"  # noqa: E501
                )
        logger.info(logging_string)


def _get_digits(value: float) -> Tuple[int, int]:
    """
    Finds the number of digits to print before and after the decimal point,
    based on the order of magnitude of the value.

    5 "significant" digits are guaranteed to be printed.

    :param value: The value for which the number of digits is calculated.
    """

    # Get order of magnitude of the value:
    order = int(np.floor(np.log10(value)))

    # Get the number of digits before the decimal point:
    if order < 0:
        digits_before = 1
    else:
        digits_before = order + 1

    # Get the number of digits after the decimal point:
    if order < 0:
        digits_after = 4 - order
    else:
        digits_after = max(1, 4 - order)

    total_characters = digits_before + digits_after + 1  # +1 for the point

    return total_characters, digits_after
