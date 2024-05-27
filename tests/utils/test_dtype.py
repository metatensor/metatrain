import torch

from metatensor.models.utils.dtype import dtype_to_str


def test_dtype_to_string():
    assert dtype_to_str(torch.float64) == "float64"
    assert dtype_to_str(torch.float32) == "float32"
    assert dtype_to_str(torch.int64) == "int64"
    assert dtype_to_str(torch.int32) == "int32"
    assert dtype_to_str(torch.bool) == "bool"
