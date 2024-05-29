import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..dataset import TargetInfo, TargetInfoDict
from .systems import SYSTEM_READERS
from .targets import ENERGY_READERS, FORCES_READERS, STRESS_READERS, VIRIAL_READERS


logger = logging.getLogger(__name__)


def _base_reader(
    readers: dict,
    filename: str,
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    **reader_kwargs,
):
    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = readers[fileformat]
    except KeyError:
        raise ValueError(f"fileformat {fileformat!r} is not supported")

    return reader(filename, dtype=dtype, **reader_kwargs)


def read_energy(
    filename: str,
    target_value: str = "energy",
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read energy informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the system file. If :py:obj:`None` the format is
        determined from the suffix
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=ENERGY_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
        dtype=dtype,
    )


def read_forces(
    filename: str,
    target_value: str = "forces",
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read force informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file
    :param fileformat: format of the system file. If :py:obj:`None` the format is
        determined from the suffix
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=FORCES_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
        dtype=dtype,
    )


def read_stress(
    filename: str,
    target_value: str = "stress",
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read stress informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the system file. If :py:obj:`None` the format is
        determined from the suffix
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=STRESS_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
        dtype=dtype,
    )


def read_systems(
    filename: str,
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[System]:
    """Read system informations from a file.

    :param filename: name of the file to read
    :param fileformat: format of the system file. If :py:obj:`None` the format is
        determined from the suffix.
    :param dtype: desired data type of returned tensor
    :returns: list of systems
    """
    return _base_reader(
        readers=SYSTEM_READERS,
        filename=filename,
        fileformat=fileformat,
        dtype=dtype,
    )


def read_virial(
    filename: str,
    target_value: str = "virial",
    fileformat: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read virial informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param fileformat: format of the system file. If :py:obj:`None` the format is
        determined from the suffix.
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        readers=VIRIAL_READERS,
        filename=filename,
        fileformat=fileformat,
        key=target_value,
        dtype=dtype,
    )


def read_targets(
    conf: DictConfig,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, List[TensorMap]], TargetInfoDict]:
    """Reading all target information from a fully expanded config.

    To get such a config you can use
    :func:`metatensor.models.utils.omegaconf.expand_dataset_config`.

    This function uses subfunctions like :func:`read_energy` to parse the requested
    target quantity. Currently only `energy` is a supported target property. But, within
    the `energy` section gradients such as `forces`, the `stress` or the `virial` can be
    added. Other gradients are silentlty irgnored.

    :param conf: config containing the keys for what should be read.
    :param dtype: desired data type of returned tensor
    :returns: Dictionary containing one TensorMaps for each target section in the config
        as well as a ``TargetInfoDict`` instance containing the metadata of the targets.

    :raises ValueError: if the target name is not valid. Valid target names are
        those that either start with ``mtm::`` or those that are in the list of
        standard outputs of ``metatensor.torch.atomistic`` (see
        https://docs.metatensor.org/latest/atomistic/outputs.html)
    """
    target_dictionary = {}
    target_info_dictionary = TargetInfoDict()
    standard_outputs_list = ["energy"]

    for target_key, target in conf.items():
        target_info_gradients = []

        if target_key not in standard_outputs_list and not target_key.startswith(
            "mtm::"
        ):
            raise ValueError(
                f"Target names must either be one of {standard_outputs_list} "
                "or start with `mtm::`."
            )
        if target["quantity"] == "energy":
            blocks = read_energy(
                filename=target["read_from"],
                target_value=target["key"],
                fileformat=target["file_format"],
                dtype=dtype,
            )

            if target["forces"]:
                try:
                    position_gradients = read_forces(
                        filename=target["forces"]["read_from"],
                        target_value=target["forces"]["key"],
                        fileformat=target["forces"]["file_format"],
                        dtype=dtype,
                    )
                except KeyError:
                    logger.warning(
                        f"No Forces found in section {target_key!r}. "
                        "Continue without forces!"
                    )
                else:
                    logger.info(
                        f"Forces found in section {target_key!r}. Forces are taken for "
                        "training!"
                    )
                    for block, position_gradient in zip(blocks, position_gradients):
                        block.add_gradient(
                            parameter="positions", gradient=position_gradient
                        )

                    target_info_gradients.append("positions")

            if target["stress"] and target["virial"]:
                raise ValueError("Cannot use stress and virial at the same time!")

            if target["stress"]:
                try:
                    strain_gradients = read_stress(
                        filename=target["stress"]["read_from"],
                        target_value=target["stress"]["key"],
                        fileformat=target["stress"]["file_format"],
                        dtype=dtype,
                    )
                except KeyError:
                    logger.warning(
                        f"No Stress found in section {target_key!r}. "
                        "Continue without stress!"
                    )
                else:
                    logger.info(
                        f"Stress found in section {target_key!r}. Stress is taken for "
                        f"training!"
                    )
                    for block, strain_gradient in zip(blocks, strain_gradients):
                        block.add_gradient(parameter="strain", gradient=strain_gradient)

                    target_info_gradients.append("strain")

            if target["virial"]:
                try:
                    strain_gradients = read_virial(
                        filename=target["virial"]["read_from"],
                        target_value=target["virial"]["key"],
                        fileformat=target["virial"]["file_format"],
                        dtype=dtype,
                    )
                except KeyError:
                    logger.warning(
                        f"No Virial found in section {target_key!r}. "
                        "Continue without virial!"
                    )
                else:
                    logger.info(
                        f"Virial found in section {target_key!r}. Virial is taken for "
                        f"training!"
                    )
                    for block, strain_gradient in zip(blocks, strain_gradients):
                        block.add_gradient(parameter="strain", gradient=strain_gradient)

                    target_info_gradients.append("strain")
        else:
            raise ValueError(
                f"Quantity: {target['quantity']!r} is not supported. Choose 'energy'."
            )

        target_dictionary[target_key] = [
            TensorMap(
                keys=Labels(["_"], torch.tensor([[0]])),
                blocks=[block],
            )
            for block in blocks
        ]

        target_info_dictionary[target_key] = TargetInfo(
            quantity=target["quantity"],
            unit=target["unit"],
            per_atom=False,  # TODO: read this from the config
            gradients=target_info_gradients,
        )

    return target_dictionary, target_info_dictionary
