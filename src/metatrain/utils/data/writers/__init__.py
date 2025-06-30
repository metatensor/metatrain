from typing import Dict, Type, Union

from .ase import ASEWriter
from .metatensor import MetatensorWriter
from .writers import DiskDatasetWriter
from .writers import Writer as Writer


# make mypy happy
ConcreteWriterClass = Union[
    Type[ASEWriter],
    Type[MetatensorWriter],
    Type[DiskDatasetWriter],
]

PREDICTIONS_WRITERS: Dict[str, ConcreteWriterClass] = {
    ".xyz": ASEWriter,
    ".mts": MetatensorWriter,
    ".zip": DiskDatasetWriter,
}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writers"""

DEFAULT_WRITER: ConcreteWriterClass = ASEWriter
