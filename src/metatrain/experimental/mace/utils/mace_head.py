import warnings
from typing import Optional

import torch


def get_mace_head_index(
    model: torch.nn.Module, requested_head_name: Optional[str]
) -> int:
    """Resolve which internal head to use. Official foundation models
    are inconsistent: some have no ``heads`` attribute at all (older
    single-head MP-0a/0b, OFF23), some name their single head
    "default"/"Default" or something else entirely (e.g. "omat_pbe").
    Match by exact name, then case-insensitively, then fall back to
    the only head for single-head models.
    """
    head_names = list(getattr(model, "heads", []))
    if not head_names or len(head_names) == 1:
        # The former one is the case for MP-0a / MP-0b / OFF23.
        # Only one head, so we use it regardless of its name. Warn only if the
        # user asked for a specific head that does not match the one available
        # (e.g. requested "default" but MACE-OMAT-0's single head is "omat_pbe").
        if (
            requested_head_name is not None
            and head_names
            and head_names[0] != requested_head_name
        ):
            warnings.warn(
                f"Requested head '{requested_head_name}' but the loaded MACE model "
                f"has only one head '{head_names[0]}'. Using that head.",
                stacklevel=2,
            )
        return 0
    else:
        # Multi-head model, we need to find the right head by name.
        if requested_head_name is None:
            raise ValueError(
                f"Requested head name is None but the loaded MACE model has multiple "
                f"heads: {head_names}. Please specify which head to use."
            )
        matches = [i for i, h in enumerate(head_names) if h == requested_head_name]
        if not matches:
            raise ValueError(
                f"Head '{requested_head_name}' not found in the loaded MACE model. "
                f"Available heads: {head_names}"
            )
        return matches[0]
