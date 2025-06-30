from typing import Type

from .ase import ASEWriter
from .metatensor import MetatensorWriter
from .writers import DiskDatasetWriter, Writer


PREDICTIONS_WRITERS = {
    ".xyz": ASEWriter,
    ".mts": MetatensorWriter,
    ".zip": DiskDatasetWriter,
}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writers"""

DEFAULT_WRITER: Type[Writer] = ASEWriter
