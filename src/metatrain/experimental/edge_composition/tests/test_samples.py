import torch
from metatensor.torch import Labels, TensorBlock

from metatrain.experimental.edge_composition.utils.samples import (
    sample_from_tensorblock,
)


def test_sample_from_tensorblock():
    # Create a sample tensor block
    block = TensorBlock(
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 1, 2),
        samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0], [0, 1]])),
        components=[Labels(names=["component"], values=torch.tensor([[0]]))],
        properties=Labels(names=["property"], values=torch.tensor([[0], [1]])),
    )

    # Create a sample labels object for the requested samples
    samples = Labels(
        names=["system", "atom"], values=torch.tensor([[0, 0], [0, 2], [1, 1]])
    )

    fill_value = 456667.4
    # Call the function
    sampled_block = sample_from_tensorblock(block, samples, missing_value=fill_value)

    # Check the output
    assert sampled_block.values.shape == (3, 1, 2)
    assert sampled_block.samples == samples
    assert sampled_block.components == block.components
    assert sampled_block.properties == block.properties

    assert torch.allclose(sampled_block.values[0], block.values[0])  # Existing sample
    assert torch.allclose(
        sampled_block.values[1:], torch.full((2, 1, 2), fill_value)
    )  # Missing sample

def test_sample_from_tensorblock_with_property():
    # Create a sample tensor block
    block = TensorBlock(
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 1, 2),
        samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0], [0, 1]])),
        components=[Labels(names=["component"], values=torch.tensor([[0]]))],
        properties=Labels(names=["property"], values=torch.tensor([[0], [1]])),
    )

    # Create a sample labels object for the requested samples
    samples = Labels(
        names=["system", "atom"], values=torch.tensor([[0, 0], [0, 2], [1, 1]])
    )
    properties = Labels(names=["property"], values=torch.tensor([[1]]))

    fill_value = 456667.4
    # Call the function
    sampled_block = sample_from_tensorblock(block, samples, properties=properties, missing_value=fill_value)

    # Check the output
    assert sampled_block.values.shape == (3, 1, 1)
    assert sampled_block.samples == samples
    assert sampled_block.components == block.components
    assert sampled_block.properties == properties

    assert torch.allclose(sampled_block.values[0, 0], block.values[0, ..., 1])  # Existing sample and property
    assert torch.allclose(
        sampled_block.values[1:], torch.full((2, 1, 1), fill_value)
    )  # Missing sample

