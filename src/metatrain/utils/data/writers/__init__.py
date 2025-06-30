from pathlib import Path
from typing import Dict, Optional, Protocol, Type, Union

from metatensor.torch import ModelCapabilities

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


# PREDICTIONS_WRITERS: Dict[str, Writer] = {
PREDICTIONS_WRITERS: Dict[str, WriterFactory] = {
    ".xyz": _make_factory(ASEWriter),
    ".mts": _make_factory(MetatensorWriter),
    ".zip": _make_factory(DiskDatasetWriter),
}
DEFAULT_WRITER: WriterFactory = _make_factory(ASEWriter)
