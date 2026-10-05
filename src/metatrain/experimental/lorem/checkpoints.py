"""Checkpoint upgrade helpers for the LOREM architecture.

LOREM has not had a public release yet, so there is no prior checkpoint
format to stay compatible with: version 1 is the first checkpoint format,
and there is nothing to upgrade from. Future versions should add a
``model_update_v{n}_v{n + 1}`` function here, following the pattern used by
metatrain's other architectures.
"""
