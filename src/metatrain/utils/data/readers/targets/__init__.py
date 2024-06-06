from .ase import read_energy_ase, read_forces_ase, read_stress_ase, read_virial_ase

ENERGY_READERS = {".extxyz": read_energy_ase, ".xyz": read_energy_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a target energy reader"""

FORCES_READERS = {".extxyz": read_forces_ase, ".xyz": read_forces_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a target forces reader"""

STRESS_READERS = {".extxyz": read_stress_ase, ".xyz": read_stress_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a target stress reader"""

VIRIAL_READERS = {".extxyz": read_virial_ase, ".xyz": read_virial_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a target virial reader"""
