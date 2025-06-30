# from pathlib import Path
from typing import Type

# from metatensor.torch import TensorMap
# from metatomic.torch import ModelCapabilities, System
# from .metatensor import write_mts
# from .xyz import write_xyz
from .writer import ASEWriter, DiskDatasetWriter, MetatensorWriter, Writer


PREDICTIONS_WRITERS = {
    ".xyz": ASEWriter,
    ".mts": MetatensorWriter,
    ".zip": DiskDatasetWriter,
}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writers"""

DEFAULT_WRITER: Type[Writer] = ASEWriter

__all__ = [
    "ASEWriter",
    "DiskDatasetWriter",
    "MetatensorWriter",
    "Writer",
    "PREDICTIONS_WRITERS",
    "DEFAULT_WRITER",
]
