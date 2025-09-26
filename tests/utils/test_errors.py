import pytest
import torch

from metatrain.utils.errors import ArchitectureError, OutOfMemoryError


def test_architecture_error():
    match = "The error above most likely originates from an architecture"
    with pytest.raises(ArchitectureError, match=match):
        try:
            raise ValueError("An example error from the architecture")
        except Exception as e:
            raise ArchitectureError(e) from e


def test_oom_error():
    match = (
        "The error above likely means that the model ran out of memory during training."
    )
    with pytest.raises(OutOfMemoryError, match=match):
        try:
            raise torch.cuda.OutOfMemoryError("An example out of memory error")
        except torch.cuda.OutOfMemoryError as e:
            raise OutOfMemoryError(e) from e
