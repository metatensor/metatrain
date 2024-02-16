from typing import List

import ase.io
import torch
from rascaline.systems import AseSystem
from rascaline.torch.system import System, systems_to_torch


def read_structures_ase(
    filename: str, dtype: torch.dtype = torch.float64
) -> List[System]:
    """Store structure informations using ase.

    :param filename: name of the file to read
    :param dtype: desired data type of returned tensor

    :returns:
        A list of structures
    """
    systems = [AseSystem(atoms) for atoms in ase.io.read(filename, ":")]

    return [s.to(dtype=dtype) for s in systems_to_torch(systems)]
