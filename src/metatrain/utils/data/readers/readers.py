import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..dataset import TargetInfo, TargetInfoDict


logger = logging.getLogger(__name__)

SUPPORTED_READER_LIBRARIES = ["ase"]
""":py:class:`list`: list containing all implemented reader libraries"""

READER_FORMATS = {
    ".xyz": "ase",
    ".extxyz": "ase",
}
""":py:class:`dict`: dictionary mapping file suffixes to default readers"""


def _base_reader(
    target: str,
    filename: str,
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    **reader_kwargs,
):
    if reader is None:
        try:
            filesuffix = Path(filename).suffix
            reader = READER_FORMATS[filesuffix]
        except KeyError:
            raise ValueError(
                f"File suffix {filesuffix!r} is not linked to a default reader "
                "library. You can reading it by setting a specific 'reader' from the "
                f"known ones: {', '.join(SUPPORTED_READER_LIBRARIES)} "
            )

    try:
        reader_mod = importlib.import_module(
            name=f".{reader}", package="metatrain.utils.data.readers"
        )
    except ImportError:
        raise ValueError(
            f"Reader library {reader!r} not supported. Choose from "
            f"{', '.join(SUPPORTED_READER_LIBRARIES)}"
        )

    try:
        reader_met = getattr(reader_mod, f"read_{target}_{reader}")
    except AttributeError:
        raise ValueError(f"Reader library {reader!r} can't read {target!r}.")

    return reader_met(filename, dtype=dtype, **reader_kwargs)


def read_energy(
    filename: str,
    target_value: str = "energy",
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read energy informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        tried to determined from the file suffix.
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="energy",
        filename=filename,
        reader=reader,
        key=target_value,
        dtype=dtype,
    )


def read_forces(
    filename: str,
    target_value: str = "forces",
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read force informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        tried to determined from the file suffix.
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="forces",
        filename=filename,
        reader=reader,
        key=target_value,
        dtype=dtype,
    )


def read_stress(
    filename: str,
    target_value: str = "stress",
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read stress informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        tried to determined from the file suffix.
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="stress",
        filename=filename,
        reader=reader,
        key=target_value,
        dtype=dtype,
    )


def read_systems(
    filename: str,
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[System]:
    """Read system informations from a file.

    :param filename: name of the file to read
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        tried to determined from the file suffix.
    :param dtype: desired data type of returned tensor
    :returns: list of systems
    """
    return _base_reader(
        target="systems",
        filename=filename,
        reader=reader,
        dtype=dtype,
    )


def read_virial(
    filename: str,
    target_value: str = "virial",
    reader: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> List[TensorBlock]:
    """Read virial informations from a file.

    :param filename: name of the file to read
    :param target_value: target value key name to be parsed from the file.
    :param reader: reader library for parsing the file. If :py:obj:`None` the library is
        tried to determined from the file suffix.
    :param dtype: desired data type of returned tensor
    :returns: target value stored stored as a :class:`metatensor.TensorBlock`
    """
    return _base_reader(
        target="virial",
        filename=filename,
        reader=reader,
        key=target_value,
        dtype=dtype,
    )


def read_targets(
    conf: DictConfig,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, List[TensorMap]], TargetInfoDict]:
    """Reading all target information from a fully expanded config.

    To get such a config you can use
    :func:`metatrain.utils.omegaconf.expand_dataset_config`.

    This function uses subfunctions like :func:`read_energy` to parse the requested
    target quantity. Currently only `energy` is a supported target property. But, within
    the `energy` section gradients such as `forces`, the `stress` or the `virial` can be
    added. Other gradients are silentlty irgnored.

    :param conf: config containing the keys for what should be read.
    :param dtype: desired data type of returned tensor
    :returns: Dictionary containing one TensorMaps for each target section in the config
        as well as a ``TargetInfoDict`` instance containing the metadata of the targets.

    :raises ValueError: if the target name is not valid. Valid target names are
        those that either start with ``mtt::`` or those that are in the list of
        standard outputs of ``metatensor.torch.atomistic`` (see
        https://docs.metatensor.org/latest/atomistic/outputs.html)
    """
    target_dictionary = {}
    target_info_dictionary = TargetInfoDict()
    standard_outputs_list = ["energy"]

    for target_key, target in conf.items():
        target_info_gradients = set()

        if target_key not in standard_outputs_list and not target_key.startswith(
            "mtt::"
        ):
            raise ValueError(
                f"Target names must either be one of {standard_outputs_list} "
                "or start with `mtt::`."
            )
        if target["quantity"] == "energy":
            blocks = read_energy(
                filename=target["read_from"],
                target_value=target["key"],
                reader=target["reader"],
                dtype=dtype,
            )

            if target["forces"]:
                try:
                    position_gradients = read_forces(
                        filename=target["forces"]["read_from"],
                        target_value=target["forces"]["key"],
                        reader=target["forces"]["reader"],
                        dtype=dtype,
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

                    target_info_gradients.add("positions")

            if target["stress"] and target["virial"]:
                raise ValueError("Cannot use stress and virial at the same time!")

            if target["stress"]:
                try:
                    strain_gradients = read_stress(
                        filename=target["stress"]["read_from"],
                        target_value=target["stress"]["key"],
                        reader=target["stress"]["reader"],
                        dtype=dtype,
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

                    target_info_gradients.add("strain")

            if target["virial"]:
                try:
                    strain_gradients = read_virial(
                        filename=target["virial"]["read_from"],
                        target_value=target["virial"]["key"],
                        reader=target["virial"]["reader"],
                        dtype=dtype,
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

                    target_info_gradients.add("strain")
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
