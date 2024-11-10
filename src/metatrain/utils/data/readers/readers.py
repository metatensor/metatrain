import importlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..dataset import TargetInfo


logger = logging.getLogger(__name__)

AVAILABLE_READERS = ["ase", "metatensor"]
""":py:class:`list`: list containing all implemented reader libraries"""

DEFAULT_READER = {".xyz": "ase", ".extxyz": "ase", ".mts": "metatensor"}
""":py:class:`dict`: dictionary mapping file extensions to a default reader"""


def _call_base_systems_reader(
    filename: str, reader: Optional[str] = None
) -> List[System]:

    if reader is None:
        try:
            file_suffix = Path(filename).suffix
            reader = DEFAULT_READER[file_suffix]
        except KeyError:
            raise ValueError(
                f"File extension {file_suffix!r} is not linked to a default reader "
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
        reader_met = getattr(reader_mod, f"read_{reader}_systems")
    except AttributeError:
        raise ValueError(f"Reader library {reader!r} can't read systems.")

    systems = reader_met(filename)

    # elements in data are `torch.ScriptObject`s and their `dtype` is an integer.
    # A C++ double/torch.float64 is `7` according to
    # https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/c10/core/ScalarType.h#L54-L93
    assert all(s.dtype == 7 for s in systems)

    return systems


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
    return _call_base_systems_reader(filename=filename, reader=reader)


def _call_base_target_reader(
    target: DictConfig, target_kind: str
) -> Tuple[List[TensorMap], TargetInfo]:
    reader = target["reader"]
    filename = target["read_from"]

    if reader is None:
        try:
            file_suffix = Path(filename).suffix
            reader = DEFAULT_READER[file_suffix]
        except KeyError:
            raise ValueError(
                f"File extension {file_suffix!r} is not linked to a default reader "
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
        reader_met = getattr(reader_mod, f"read_{reader}_{target_kind}")
    except AttributeError:
        raise ValueError(f"Reader library {reader!r} can't read {target!r}.")

    target_data_and_target_info = reader_met(target)

    # elements in data are `torch.ScriptObject`s and their `dtype` is an integer.
    # A C++ double/torch.float64 is `7` according to
    # https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/c10/core/ScalarType.h#L54-L93
    data = target_data_and_target_info[0]
    assert all(d.dtype == 7 for d in data)

    return target_data_and_target_info


def read_targets(
    conf: DictConfig,
) -> Tuple[Dict[str, List[TensorMap]], Dict[str, TargetInfo]]:
    """Reading all target information from a fully expanded config.

    To get such a config you can use :func:`expand_dataset_config
    <metatrain.utils.omegaconf.expand_dataset_config>`. All targets are stored in double
    precision.

    This function uses subfunctions like :func:`read_energy` to parse the requested
    target quantity. Currently only `energy` is a supported target property. But, within
    the `energy` section gradients such as `forces`, the `stress` or the `virial` can be
    added. Other gradients are silently ignored.

    :param conf: config containing the keys for what should be read.
    :returns: Dictionary containing a list of TensorMaps for each target section in the
        config as well as a ``Dict[str, TargetInfo]`` object
        containing the metadata of the targets.

    :raises ValueError: if the target name is not valid. Valid target names are
        those that either start with ``mtt::`` or those that are in the list of
        standard outputs of ``metatensor.torch.atomistic`` (see
        https://docs.metatensor.org/latest/atomistic/outputs.html)
    """
    target_dictionary = {}
    target_info_dictionary = {}
    standard_outputs_list = ["energy"]

    for target_key, target in conf.items():

        is_standard_target = target_key in standard_outputs_list
        if not is_standard_target and not target_key.startswith("mtt::"):
            if target_key.lower() in ["force", "forces", "virial", "stress"]:
                warnings.warn(
                    f"{target_key!r} should not be its own top-level target, "
                    "but rather a sub-section of the 'energy' target",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    f"Target name ({target_key}) must either be one of "
                    f"{standard_outputs_list} or start with `mtt::`."
                )

        is_energy = (
            (target["quantity"] == "energy")
            and (not target["per_atom"])
            and target["num_properties"] == 1
            and target["type"] == "scalar"
        )
        energy_or_generic = "energy" if is_energy else "generic"
        targets_as_list_of_tensor_maps, target_info = _call_base_target_reader(
            target, energy_or_generic
        )
        target_dictionary[target_key] = targets_as_list_of_tensor_maps
        target_info_dictionary[target_key] = target_info

    return target_dictionary, target_info_dictionary


def _empty_tensor_map_like(tensor_map: TensorMap) -> TensorMap:
    new_keys = tensor_map.keys
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        new_block = _empty_tensor_block_like(block)
        new_blocks.append(new_block)
    return TensorMap(keys=new_keys, blocks=new_blocks)


def _empty_tensor_block_like(tensor_block: TensorBlock) -> TensorBlock:
    new_block = TensorBlock(
        values=torch.empty(
            (0,) + tensor_block.values.shape[1:],
            dtype=torch.float64,  # metatensor can't serialize otherwise
            device=tensor_block.values.device,
        ),
        samples=Labels(
            names=tensor_block.samples.names,
            values=torch.empty(
                (0, tensor_block.samples.values.shape[1]),
                dtype=tensor_block.samples.values.dtype,
                device=tensor_block.samples.values.device,
            ),
        ),
        components=tensor_block.components,
        properties=tensor_block.properties,
    )
    for gradient_name, gradient in tensor_block.gradients():
        new_block.add_gradient(gradient_name, _empty_tensor_block_like(gradient))
    return new_block
