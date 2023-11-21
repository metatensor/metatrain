import torch
from metatensor_models.nn.example import Example


def test_example():
    example = Example(equivariant_selection=[(0, 1)], hypers={})
    torch.jit.script(example)
