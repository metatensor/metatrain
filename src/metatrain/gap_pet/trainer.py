"""GapPET reuses the PET trainer, with one adjustment for ``inherit_heads``.

The PET trainer is target-agnostic: it iterates over the targets registered on
the model, computes predictions via :func:`metatrain.utils.evaluate_model`
(which handles autograd through positions for energy-quantity targets), and
accumulates an aggregator loss. GapPET registers its ``gap_energy`` targets as
``ModelOutput(quantity="energy", per_atom=False)``, so the trainer's existing
energy + force machinery applies without modification.

The one thing that does not carry over is ``inherit_heads``. PET's
:func:`apply_finetuning_strategy` copies head weights by matching the target
name between dots in parameter names (``node_heads.<target>.0.0.weight``), but
GapPET's readout does not live under the target name: each gap target owns a
*pair* of internal heads, ``__gap_pet_homo__<target>`` and
``__gap_pet_lumo__<target>``. This trainer therefore rewrites an
``inherit_heads`` mapping expressed in user-facing target names into one over
those internal head names, so that

.. code-block:: yaml

    inherit_heads:
      mtt::gap_s1: mtt::gap_energy

seeds *both* the HOMO and the LUMO head of ``mtt::gap_s1`` from the
corresponding heads of ``mtt::gap_energy``.
"""

from typing import Any, Dict, List, Union

import torch

from metatrain.pet.trainer import Trainer as PETTrainer
from metatrain.utils.data import Dataset

from .model import HOMO_HEAD_PREFIX, LUMO_HEAD_PREFIX, GapPET


__all__ = ["Trainer"]


class Trainer(PETTrainer):
    def train(
        self,
        model: GapPET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        original_finetune_hypers = self.hypers["finetune"]
        expanded = self._expand_inherit_heads(model, original_finetune_hypers)
        self.hypers["finetune"] = expanded
        try:
            super().train(
                model=model,
                dtype=dtype,
                devices=devices,
                train_datasets=train_datasets,
                val_datasets=val_datasets,
                checkpoint_dir=checkpoint_dir,
            )
        finally:
            # ``finetune_config`` is written into the checkpoint, so restore the
            # user-facing form rather than persisting the internal head names.
            self.hypers["finetune"] = original_finetune_hypers
            if hasattr(model, "finetune_config"):
                model.finetune_config = original_finetune_hypers

    @staticmethod
    def _expand_inherit_heads(
        model: GapPET, finetune_hypers: Dict[str, Any]
    ) -> Dict[str, Any]:
        inherit_heads = finetune_hypers.get("inherit_heads") or {}
        if not inherit_heads:
            return finetune_hypers

        known_targets = set(model._gap_target_names)
        expanded: Dict[str, str] = {}
        for dest, source in inherit_heads.items():
            if dest.startswith(HOMO_HEAD_PREFIX) or dest.startswith(LUMO_HEAD_PREFIX):
                # Already an internal head name; pass through untouched.
                expanded[dest] = source
                continue
            for name, role in ((dest, "destination"), (source, "source")):
                if name not in known_targets:
                    raise ValueError(
                        f"`inherit_heads` {role} target '{name}' is not a gap target "
                        f"of this model. Known gap targets: {sorted(known_targets)}."
                    )
            expanded[HOMO_HEAD_PREFIX + dest] = HOMO_HEAD_PREFIX + source
            expanded[LUMO_HEAD_PREFIX + dest] = LUMO_HEAD_PREFIX + source

        expanded_hypers = dict(finetune_hypers)
        expanded_hypers["inherit_heads"] = expanded
        return expanded_hypers
