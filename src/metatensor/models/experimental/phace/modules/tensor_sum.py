from typing import List

import metatensor.torch
import torch


class TensorAdd(torch.nn.Module):
    # Kept as a torch.nn.module.
    # Maybe we will want to do linear combinations of these in the future
    # with learnable coefficients.

    def __init__(self):
        super().__init__()
        self.equivariant_tensor_add = EquivariantTensorAdd()

    def forward(
        self,
        list_1: List[metatensor.torch.TensorMap],
        list_2: List[metatensor.torch.TensorMap],
    ) -> List[metatensor.torch.TensorMap]:
        nu_max = max(len(list_1), len(list_2)) - 1
        len_1 = len(list_1)
        len_2 = len(list_2)
        new_list: List[metatensor.torch.TensorMap] = []
        for nu in range(nu_max + 1):
            if nu < len_1 and nu < len_2:
                tensor_1 = list_1[nu]
                tensor_2 = list_2[nu]
                if tensor_1.keys.names == ["dummy"]:
                    if tensor_2.keys.names == ["dummy"]:
                        raise ValueError()
                    else:
                        new_list.append(tensor_2)
                elif tensor_2.keys.names == ["dummy"]:
                    if tensor_1.keys.names == ["dummy"]:
                        raise ValueError()
                    else:
                        new_list.append(tensor_1)
                else:
                    new_list.append(self.equivariant_tensor_add(tensor_1, tensor_2))
            elif nu < len_1:
                tensor_1 = list_1[nu]
                if tensor_1.keys.names == ["dummy"]:
                    raise ValueError()
                else:
                    new_list.append(tensor_1)
            elif nu < len_2:
                tensor_2 = list_2[nu]
                if tensor_2.keys.names == ["dummy"]:
                    raise ValueError()
                else:
                    new_list.append(tensor_2)
            else:
                raise ValueError("This should never happen")

        return new_list


class EquivariantTensorAdd(torch.nn.Module):
    # Kept as a torch.nn.module.
    # Maybe we will want to do linear combinations of these in the future
    # with learnable coefficients.

    def __init__(self):
        super().__init__()

    def forward(
        self, tmap_1: metatensor.torch.TensorMap, tmap_2: metatensor.torch.TensorMap
    ):
        # for now, this assumes that each of the two tensor maps have the same "nu" keys
        # actually, we take the maximum
        nu_1 = int(torch.max(tmap_1.keys.column("nu")))
        nu_2 = int(torch.max(tmap_2.keys.column("nu")))
        nu_max = max(nu_1, nu_2)

        new_blocks: List[metatensor.torch.TensorBlock] = []
        new_keys: List[List[int]] = []

        ls_1 = tmap_1.keys.remove("nu")
        ls_2 = tmap_2.keys.remove("nu")
        key_union = ls_1.union(ls_2)

        for key in key_union.values:
            o3_lambda = int(key[0])
            o3_sigma = int(key[1])
            blocks_1 = tmap_1.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            blocks_2 = tmap_2.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            assert len(blocks_1) <= 1
            assert len(blocks_2) <= 1
            if len(blocks_1) == 1 and len(blocks_2) == 1:
                block_1 = blocks_1[0]
                block_2 = blocks_2[0]
                new_block = metatensor.torch.TensorBlock(
                    values=block_1.values + block_2.values,
                    samples=block_1.samples,
                    components=block_1.components,
                    properties=block_1.properties,
                )
                new_blocks.append(new_block)
                new_keys.append([nu_max, o3_lambda, o3_sigma])
            elif len(blocks_1) == 1:
                new_blocks.append(blocks_1[0])
                new_keys.append([nu_1, o3_lambda, o3_sigma])
            elif len(blocks_2) == 1:
                new_blocks.append(blocks_2[0])
                new_keys.append([nu_2, o3_lambda, o3_sigma])
            else:
                raise ValueError("This should never happen. Metatensor bug?")

        new_map = metatensor.torch.TensorMap(
            keys=metatensor.torch.Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(new_keys, device=tmap_1.keys.values.device),
            ),
            blocks=new_blocks,
        )
        return new_map
