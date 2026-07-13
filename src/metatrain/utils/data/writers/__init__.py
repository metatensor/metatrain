import os
from pathlib import Path
from typing import Dict, Optional, Protocol, Type, Union

from metatomic.torch import ModelCapabilities

from .ase import ASEWriter
from .diskdataset import DiskDatasetWriter
from .memmap import MemmapWriter
from .metatensor import MetatensorWriter
from .writers import (
    Writer,
)
from .writers import (
    _split_tensormaps as _split_tensormaps,
)


class WriterFactory(Protocol):
    def __call__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = None,
    ) -> Writer: ...


def _make_factory(
    cls: Type[Writer],
) -> WriterFactory:
    def factory(
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = None,
    ) -> Writer:
        return cls(filename, capabilities, append)

    return factory


PREDICTIONS_WRITERS: Dict[str, WriterFactory] = {
    ".xyz": _make_factory(ASEWriter),
    ".mts": _make_factory(MetatensorWriter),
    ".zip": _make_factory(DiskDatasetWriter),
}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writer"""

DEFAULT_WRITER: WriterFactory = _make_factory(ASEWriter)

MEMMAP_WRITER: WriterFactory = _make_factory(MemmapWriter)


def get_writer(
    filename: Union[str, Path],
    capabilities: Optional[ModelCapabilities] = None,
    append: Optional[bool] = None,
    fileformat: Optional[str] = None,
) -> Writer:
    """Selects the appropriate writer based on the file extension.

    For certain file suffixes, the systems will also be written (i.e ``xyz``).

    A path ending in a path separator (e.g. ``predictions/``) is treated as a
    memory-mapped ``MemmapDataset`` directory rather than a single file. Since
    nothing exists on disk yet when a writer is selected, the trailing separator
    is the write-side equivalent of that check.

    The capabilities of the model are used to infer the type (physical quantity) of
    the predictions. In this way, for example, position gradients of energies can be
    saved as forces.

    For the moment, strain gradients of the energy are saved as stresses
    (and not as virials).

    :param filename: name of the file to write, or a directory path ending in a path
        separator for a memmap dataset
    :param capabilities: capabilities of the model
    :param append: if :py:obj:`True`, the data will be appended to the file, if it
        exists. If :py:obj:`False`, the file will be overwritten. If :py:obj:`None`,
        the default behavior of the writer is used.
    :param fileformat: format of the target value file. If :py:obj:`None` the format is
        determined from the file extension.
    :return: a :py:class:`Writer` instance.
    :raises ValueError: if ``filename`` both ends in a path separator and has a
        recognized file suffix (e.g. ``"predictions.zip/"``), since it is then
        ambiguous whether a memmap directory or a file of that format was intended.
    """

    filename_str = str(filename)
    if filename_str.endswith(("/", os.sep)):
        stripped_suffix = Path(filename_str.rstrip("/" + os.sep)).suffix
        if stripped_suffix in PREDICTIONS_WRITERS:
            raise ValueError(
                f"'{filename_str}' is ambiguous: it looks like both a "
                f"'{stripped_suffix}' file and a directory path (it ends with a "
                "path separator). Remove the trailing separator to write a "
                f"'{stripped_suffix}' file, or remove the '{stripped_suffix}' suffix "
                "to write a memmap dataset directory."
            )
        return MEMMAP_WRITER(filename, capabilities, append)

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        writer_factory = PREDICTIONS_WRITERS[fileformat]
    except KeyError:
        raise ValueError(
            f"fileformat '{fileformat}' is not supported. Use a trailing path "
            "separator (e.g. 'predictions/') to write a memmap dataset instead."
        )

    return writer_factory(Path(filename).stem + fileformat, capabilities, append)
