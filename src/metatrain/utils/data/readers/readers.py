import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..dataset import TargetInfo, TargetInfoDict


logger = logging.getLogger(__name__)

AVAILABLE_READERS = ["ase"]
""":py:class:`list`: list containing all implemented reader libraries"""

DEFAULT_READER = {
    ".xyz": "ase",
    ".extxyz": "ase",
}
""":py:class:`dict`: dictionary mapping file extensions to a default reader"""


def _base_reader(
    target: str,
    filename: str,
    reader: Optional[str] = None,
    **reader_kwargs,
) -> List[Any]:
    if reader is None:
        try:
            filesuffix = Path(filename).suffix
            reader = DEFAULT_READER[filesuffix]
        except KeyError:
            raise ValueError(
                f"File extension {filesuffix!r} is not linked to a default reader "
                "library. You can try reading it by setting a specific 'reader' from "
                f"the known ones: {', '.join(AVAILABLE_READERS)} "
            )

    try:
        reader_mod = importlib.import_module(
            name=f".{reader}", package="metatrain.utils.data.readers"
        )
    except ImportError:
        raise ValueError(
            f"Reader library {reader!r} not supported. Choose from "
            f"{', '.join(AVAILABLE_READERS)}"
        )

    try:
        reader_met = getattr(reader_mod, f"read_{target}_{reader}")
    except AttributeError:
        raise ValueError(f"Reader library {reader!r} can't read {target!r}.")

    data = reader_met(filename, **reader_kwargs)

    # elements in data are `torch.ScriptObject`s and their `dtype` is an integer.
    # A C++ double/torch.float64 is `7` according to
    # https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/c10/core/ScalarType.h#L54-L93
    assert all(d.dtype == 7 for d in data)

    return data


def read_energy(
    filename: str,
    target_value: str = "energy",
    reader: Optional[str] = None,
) -> List[TensorBlock]:
    """Read energy informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        is tried to determined from the file extension.
    :returns: energy stored stored in double precision as a
        :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="energy", filename=filename, reader=reader, key=target_value
    )


def read_forces(
    filename: str,
    target_value: str = "forces",
    reader: Optional[str] = None,
) -> List[TensorBlock]:
    """Read force informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        is tried to determined from the file extension.
    :returns: forces stored in double precision stored as a
        :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="forces", filename=filename, reader=reader, key=target_value
    )


def read_stress(
    filename: str,
    target_value: str = "stress",
    reader: Optional[str] = None,
) -> List[TensorBlock]:
    """Read stress informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        is tried to determined from the file extension.
    :returns: stress stored in double precision as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="stress", filename=filename, reader=reader, key=target_value
    )


def read_systems(
    filename: str,
    reader: Optional[str] = None,
) -> List[System]:
    """Read system informations from a file.

    :param filename: name of the file to read
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        is tried to determined from the file extension.
    :param dtype: desired data type of returned tensor
    :returns: list of systems
        determined from the file extension.
    :returns: list of systems stored in double precision
    """
    return _base_reader(target="systems", filename=filename, reader=reader)


def read_virial(
    filename: str,
    target_value: str = "virial",
    reader: Optional[str] = None,
) -> List[TensorBlock]:
    """Read virial informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        is tried to determined from the file extension.
    :returns: virial stored in double precision as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="virial",
        filename=filename,
        reader=reader,
        key=target_value,
    )


def read_targets(
    conf: DictConfig,
) -> Tuple[Dict[str, List[TensorMap]], TargetInfoDict]:
    """Reading all target information from a fully expanded config.

    To get such a config you can use :func:`expand_dataset_config
    <metatrain.utils.omegaconf.expand_dataset_config>`. All targets are stored in double
    precision.

    This function uses subfunctions like :func:`read_energy` to parse the requested
    target quantity. Currently only `energy` is a supported target property. But, within
    the `energy` section gradients such as `forces`, the `stress` or the `virial` can be
    added. Other gradients are silentlty irgnored.

    :param conf: config containing the keys for what should be read.
    :returns: Dictionary containing a list of TensorMaps for each target section in the
        config as well as a :py:class:`TargetInfoDict
        <metatrain.utils.data.TargetInfoDict>` instance containing the metadata of the
        targets.

    :raises ValueError: if the target name is not valid. Valid target names are
        those that either start with ``mtt::`` or those that are in the list of
        standard outputs of ``metatensor.torch.atomistic`` (see
        https://docs.metatensor.org/latest/atomistic/outputs.html)
    """
    target_dictionary = {}
    target_info_dictionary = TargetInfoDict()
    standard_outputs_list = ["energy"]

    for target_key, target in conf.items():
        target_info_gradients: List[str] = []

        is_standard_target = target_key in standard_outputs_list
        if not is_standard_target and not target_key.startswith("mtt::"):
            raise ValueError(
                f"Target name ({target_key}) must either be one of "
                f"{standard_outputs_list} or start with `mtt::`."
            )

        if target["quantity"] == "energy":
            blocks = read_energy(
                filename=target["read_from"],
                target_value=target["key"],
                reader=target["reader"],
            )

            if target["forces"]:
                try:
                    position_gradients = read_forces(
                        filename=target["forces"]["read_from"],
                        target_value=target["forces"]["key"],
                        reader=target["forces"]["reader"],
                    )
                except Exception:
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
                        reader=target["stress"]["reader"],
                    )
                except Exception:
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
                        reader=target["virial"]["reader"],
                    )
                except Exception:
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
