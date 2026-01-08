"""Utility functions for handling batches in distributed training."""

from typing import Any, Optional

import torch


def should_skip_batch(
    batch: Optional[Any], is_distributed: bool, device: torch.device
) -> bool:
    """
    Check if a batch should be skipped in distributed training.

    In distributed mode, synchronizes across all processes to ensure
    all processes agree on whether to skip the batch. If any process
    has a None batch, all processes will skip.

    :param batch: The batch to check (None if invalid).
    :param is_distributed: Whether distributed training is enabled.
    :param device: The device to use for distributed communication.
    :return: True if the batch should be skipped, False otherwise.
    """
    if is_distributed:
        # Broadcast whether this batch should be skipped
        # 1 if batch should be kept, 0 if it should be skipped
        batch_valid = torch.tensor([1 if batch is not None else 0], device=device)
        torch.distributed.all_reduce(batch_valid, op=torch.distributed.ReduceOp.MIN)
        return batch_valid.item() == 0
    else:
        return batch is None
