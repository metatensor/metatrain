import torch

from metatrain.utils.data.spherical_helpers import (
    couple_tensor_blocks,
    uncouple_tensor_blocks,
)


def test_jit_script_couple():
    torch.jit.script(couple_tensor_blocks)


def test_jit_script_uncouple():
    torch.jit.script(uncouple_tensor_blocks)
