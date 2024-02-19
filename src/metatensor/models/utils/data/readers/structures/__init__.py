from .ase import read_structures_ase

STRUCTURE_READERS = {".extxyz": read_structures_ase, ".xyz": read_structures_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a structure reader"""
