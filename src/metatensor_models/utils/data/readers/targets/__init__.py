from .ase import read_ase

TARGET_READERS = {".xyz": read_ase}
""":py:class:`dict`: dictionary mapping file suffixes to a target structure reader"""
