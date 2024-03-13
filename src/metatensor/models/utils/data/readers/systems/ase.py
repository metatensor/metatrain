from typing import List

import ase.io
import torch
from metatensor.torch.atomistic import System, systems_to_torch


def read_systems_ase(filename: str, dtype: torch.dtype = torch.float64) -> List[System]:
    """Store system informations using ase.

    :param filename: name of the file to read
    :param dtype: desired data type of returned tensor

    :returns:
        A list of systems
    """
    systems = [atoms for atoms in ase.io.read(filename, ":")]

    return [s.to(dtype=dtype) for s in systems_to_torch(systems)]
