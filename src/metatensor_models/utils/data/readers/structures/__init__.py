from .ase import read_ase

STRUCTURE_READERS = {".xyz": read_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a structure reader"""
