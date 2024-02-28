from .ase import read_systems_ase

SYSTEM_READERS = {".extxyz": read_systems_ase, ".xyz": read_systems_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a system reader"""
