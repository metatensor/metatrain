"""GapPET reuses the PET trainer unchanged.

The PET trainer is target-agnostic: it iterates over the targets registered on
the model, computes predictions via :func:`metatrain.utils.evaluate_model`
(which handles autograd through positions for energy-quantity targets), and
accumulates an aggregator loss. GapPET registers its single ``gap_energy``
target as ``ModelOutput(quantity="energy", per_atom=False)``, so the trainer's
existing energy + force machinery applies without modification.

This module re-exports :class:`metatrain.pet.trainer.Trainer` for discovery via
``metatrain.gap_pet.__trainer__``.
"""

from metatrain.pet.trainer import Trainer


__all__ = ["Trainer"]
