import pytest
import torch

from metatensor.models.utils.device import string_to_device


def test_string_to_device():
    """Test the string_to_device function."""

    # Test the CPU option.
    assert string_to_device("cpu") == [torch.device("cpu")]

    # Test the CUDA option.
    if torch.cuda.is_available():
        assert string_to_device("cuda") == [torch.device("cuda")]
    else:
        with pytest.raises(ValueError, match="CUDA is not available on this system"):
            string_to_device("cuda")

    # Test the multi-GPU option.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            assert string_to_device("multi-gpu") == [
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            ]
        else:
            with pytest.raises(
                ValueError, match="Only one CUDA-capable GPU was found on this system"
            ):
                string_to_device("multi-gpu")
    else:
        with pytest.raises(
            ValueError, match="No CUDA-capable GPUs were found on this system"
        ):
            string_to_device("multi-gpu")

    # Test an invalid option.
    with pytest.raises(ValueError, match="Unrecognized device string `invalid-option`"):
        string_to_device("invalid-option")
