import torch
from metatensor_models.linear.example import Example


def test_jit_script():
    example = Example(equivariant_selection=[(0, 1)], hypers={})
    torch.jit.script(example)
