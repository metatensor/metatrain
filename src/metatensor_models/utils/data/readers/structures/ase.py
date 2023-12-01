from typing import List

import ase.io
from rascaline.systems import AseSystem
from rascaline.torch.system import System, systems_to_torch


def read_ase(filename: str) -> List[System]:
    systems = [AseSystem(atoms) for atoms in ase.io.read(filename, ":")]

    return systems_to_torch(systems)
