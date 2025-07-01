from pathlib import Path
from typing import Dict, Optional, Protocol, Type, Union

from metatomic.torch import ModelCapabilities

from .ase import ASEWriter
from .metatensor import MetatensorWriter
from .writers import DiskDatasetWriter, Writer  # noqa E61


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
DEFAULT_WRITER: WriterFactory = _make_factory(ASEWriter)


def get_writer(
    filename: Union[str, Path],
    capabilities: Optional[ModelCapabilities] = None,
    append: Optional[bool] = None,
    fileformat: Optional[str] = None,
) -> Writer:
    """Select the appropriate writer based on the file extension."""

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        writer_factory = PREDICTIONS_WRITERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return writer_factory(Path(filename).stem + fileformat, capabilities, append)
