import metatensor.torch
import torch
from metatensor.torch import Labels
from metatensor.torch import TensorMap, TensorBlock

# use torch split
# torch split does smth like this torch.split(tensor, [split_1_length, split_2_length, ...])
#  torch.unique_consecutive(x, return_counts=True) actually returns exactly this as counts.

# we can assume that the tensormaps are "dense" (ie every atom has a value)
# and they are already sorted

#a = torch.arange(1000).reshape(2, 500)
#b = torch.split(a, torch.tensor([2 for i in range(500)]).tolist())


def split_structurewise(tensormap):
    """
    Split a TensorMap structurewise.
    Assumes dense and sorted TensorMap. 
    """

    #tensormap = metatensor.torch.sort(tensormap, axes="samples")
    
    sample_values = tensormap.block(0).samples.values
    _, counts = torch.unique_consecutive(sample_values[:,0], return_counts=True)

    # get the keys of all blocks
    # is that even possible different sample names?
    sample_names_block = { str(key): block.samples.names for key, block in tensormap.items() }
    components_block_wise = { str(key): block.components for key, block in tensormap.items() }
    properties_block_wise = { str(key): block.properties for key, block in tensormap.items() }

    splitted = {}
    splitted_samples = torch.split(sample_values, counts.tolist())

    for key, block in tensormap.items():
        splitted[str(key)] = torch.split(block.values, counts.tolist())

    tensor_maps = []

    for i, sample in enumerate(splitted_samples):
        
        blocks = []
        
        for key in splitted.keys():
            samples_block = Labels(sample_names_block[key], sample)

            blocks.append(TensorBlock(
                samples=samples_block,
                values=splitted[key][i],
                components=components_block_wise[key],
                properties=properties_block_wise[key]
            ))

        tensor_maps.append(TensorMap(keys=tensormap.keys, blocks=blocks))

    # return a list of tensormaps
    return tensor_maps

