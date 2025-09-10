import importlib
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from metatensor.torch import TensorMap
from metatomic.torch import System
from omegaconf import DictConfig

from ..target_info import TargetInfo


AVAILABLE_READERS = ["ase", "metatensor"]
""":py:class:`list`: list containing all implemented reader libraries"""

DEFAULT_READER = {
    ".xyz": "ase",
    ".extxyz": "ase",
    ".mts": "metatensor",
}
""":py:class:`dict`: mapping file extensions to a default reader"""

logger = logging.getLogger(__name__)


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
    :raises ValueError: if no reader is found or data not in double precision
    """
    # Determine reader if not provided
    if reader is None:
        file_suffix = Path(filename).suffix
        try:
            reader = DEFAULT_READER[file_suffix]
        except KeyError:
            raise ValueError(
                f"File extension {file_suffix!r} is not linked to a default reader "
                "library. You can try reading it by setting a specific 'reader' from "
                f"the known ones: {', '.join(AVAILABLE_READERS)} "
            )

    module = _load_reader_module(reader)

    # Fetch and call read_systems
    try:
        reader_fn = module.read_systems
    except AttributeError as e:
        raise ValueError(
            f"Reader library {reader!r} cannot read systems."
            f"You can try with other readers: {AVAILABLE_READERS}"
        ) from e

    systems = reader_fn(filename)

    # elements in data are `torch.ScriptObject`s and their `dtype` is an integer.
    # A C++ double/torch.float64 is `7` according to
    # https://github.com/pytorch/pytorch/blob/207564bab1c4fe42750931765734ee604032fb69/c10/core/ScalarType.h#L54-L93
    if not all(s.dtype == 7 for s in systems):
        raise ValueError("The loaded systems are not in double precision.")

    return systems


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
        config as well as a ``Dict[str, TargetInfo]`` object containing the metadata of
        the targets.

    :raises ValueError: if the target name is not valid. Valid target names are those
        that either start with ``mtt::`` or those that are in the list of standard
        outputs of ``metatomic`` (see
        https://docs.metatensor.org/metatomic/latest/outputs/)
    """
    return _read_conf_section(
        conf,
        decide_reader=_decide_target_reader,
        validate_key=_validate_target,
    )


def read_extra_data(
    conf: DictConfig,
) -> Tuple[Dict[str, List[TensorMap]], Dict[str, TargetInfo]]:
    """Read extra data from a fully expanded config.

    This function is similar to :func:`read_targets`, but it is used to read additional
    data that is not part of the main targets. It can be used to read auxiliary data
    that might be useful for training or evaluation.

    :param conf: config containing the keys for what should be read.
    :returns: Dictionary containing a list of TensorMaps for each extra data section in
        the config as well as a ``Dict[str, TargetInfo]`` object containing the metadata
        of the extra data.
    """
    return _read_conf_section(
        conf,
        decide_reader=_decide_generic_reader,
        validate_key=_no_validate,
    )


def _read_conf_section(
    conf: DictConfig,
    decide_reader: Callable[[str, DictConfig], str],
    validate_key: Callable[[str, DictConfig], None],
) -> Tuple[Dict[str, List[TensorMap]], Dict[str, TargetInfo]]:
    """
    Generic loader for any DictConfig section (targets, extra_data, …).

    :param conf:          mapping of section names to entry configs
    :param decide_reader: callback(key, entry) -> either "energy" or "generic"
    :param validate_key:  callback(key, entry) -> None (may raise or log)
    :returns: (data_dict, info_dict)
    :raises ValueError: on unsupported file types, readers, or dtype mismatch
    """
    data_dict: Dict[str, List[TensorMap]] = {}
    info_dict: Dict[str, TargetInfo] = {}

    for key, entry in conf.items():
        # section-specific key validation
        validate_key(key, entry)

        # decide which reader method to call
        reader_kind = decide_reader(key, entry)

        # resolve reader name (explicit or default via suffix)
        reader = entry.get("reader")
        filename = entry.get("read_from")
        if reader is None:
            suffix = Path(filename).suffix
            try:
                reader = DEFAULT_READER[suffix]
            except KeyError:
                raise ValueError(
                    f"File extension {suffix!r} has no default reader. "
                    f"Set 'reader' explicitly from: {AVAILABLE_READERS}"
                )

        module = _load_reader_module(reader)

        # fetch the appropriate read_* function
        method_name = f"read_{reader_kind}"
        try:
            reader_fn = getattr(module, method_name)
        except AttributeError as e:
            available = [m for m in dir(module) if m.startswith("read_")]
            raise ValueError(
                f"Reader {reader!r} has no method {method_name!r}. "
                f"Available methods: {available}"
            ) from e

        # execute reader and collect outputs
        tensormaps, info = reader_fn(entry)

        # enforce double precision (dtype == 7)
        if not all(t.dtype == 7 for t in tensormaps):
            raise ValueError(f"Data for '{key}' not in double precision (dtype==7).")

        data_dict[key] = tensormaps
        info_dict[key] = info

    return data_dict, info_dict


# Callbacks for targets
_standard_outputs_list = {
    "energy",
    "non_conservative_forces",
    "non_conservative_stress",
    "positions",
    "momenta",
}


def _validate_target(key: str, entry: DictConfig) -> None:
    if key not in _standard_outputs_list and not key.startswith("mtt::"):
        if key.lower() in {"force", "forces", "virial", "stress"}:
            warnings.warn(
                f"{key!r} should not be its own top-level target, "
                "but rather a sub-section of the 'energy' target",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Target name ({key}) must either be one of "
                f"{_standard_outputs_list} or start with `mtt::`."
            )
    if any(name in key.lower() for name in ("force", "virial", "stress")):
        warnings.warn(
            f"the name of {key!r} resembles to a gradient of "
            "energies; it should probably not be its own top-level target, "
            "but rather a gradient sub-section of a target with the "
            "`energy` quantity",
            stacklevel=2,
        )


def _decide_target_reader(key: str, entry: DictConfig) -> str:
    is_energy = (
        entry.get("quantity") == "energy"
        and not entry.get("per_atom", False)
        and entry.get("num_subtargets", 1) == 1
        and entry.get("type") == "scalar"
    )
    return "energy" if is_energy else "generic"


# Callbacks for "extra_data"
def _no_validate(key: str, entry: DictConfig) -> None:
    pass


def _decide_generic_reader(key: str, entry: DictConfig) -> str:
    return "generic"


@lru_cache(maxsize=None)
def _load_reader_module(reader_name: str):
    """
    Load (and cache) a reader module by name.
    Raises ValueError if the module cannot be imported.
    """
    module_path = f"metatrain.utils.data.readers.{reader_name}"
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Reader library {reader_name!r} not supported. Choose from "
            f"{', '.join(AVAILABLE_READERS)}"
        ) from e
