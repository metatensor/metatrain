import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor_models.kernel.example import Example


def test_output():
    tensor_map = TensorMap(
        Labels.empty("_"),
        []
    )
    example = Example(equivariant_selection=[(0, 1)], hypers={})
    output = example(tensor_map)
    assert isinstance(output, torch.ScriptObject)


def test_jit_script():
    example = Example(equivariant_selection=[(0, 1)], hypers={})
    torch.jit.script(example)
