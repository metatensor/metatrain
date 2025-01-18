from typing import List, Tuple

import metatensor.torch
import torch
from metatensor.torch import Labels

from .layers import EquivariantLinear


class EquivariantTensorAdd(torch.nn.Module):

    def __init__(self, common_irreps: List[Tuple[int, int]], k_max_l):
        super().__init__()

        self.linear_contractions = EquivariantLinear(
            common_irreps, k_max_l, double=True
        )

    def forward(
        self, tmap_1: metatensor.torch.TensorMap, tmap_2: metatensor.torch.TensorMap
    ):
        # for now, this assumes that each of the two tensor maps have the same "nu" keys
        # actually, we take the maximum
        nu_1 = int(torch.max(tmap_1.keys.column("nu")))
        nu_2 = int(torch.max(tmap_2.keys.column("nu")))
        nu_max = max(nu_1, nu_2)

        intersection_blocks: List[metatensor.torch.TensorBlock] = []
        intersection_keys: List[List[int]] = []

        ls_1 = tmap_1.keys.remove("nu")
        ls_2 = tmap_2.keys.remove("nu")

        key_intersection = ls_1.intersection(ls_2)
        for key in key_intersection.values:
            o3_lambda = int(key[0])
            o3_sigma = int(key[1])
            blocks_1 = tmap_1.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            blocks_2 = tmap_2.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            assert len(blocks_1) <= 1
            assert len(blocks_2) <= 1
            if len(blocks_1) == 1 and len(blocks_2) == 1:
                block_1 = blocks_1[0]
                block_2 = blocks_2[0]
                concat = torch.concatenate([block_1.values, block_2.values], dim=-1)
                new_block = metatensor.torch.TensorBlock(
                    values=concat,
                    samples=block_1.samples,
                    components=block_1.components,
                    properties=Labels(
                        names=block_1.properties.names,
                        values=torch.arange(
                            concat.shape[-1], device=concat.device
                        ).reshape(-1, 1),
                    ),
                )
                intersection_blocks.append(new_block)
                intersection_keys.append([nu_max, o3_lambda, o3_sigma])
            else:
                raise ValueError("This should never happen. Metatensor bug?")
        intersection_map = metatensor.torch.TensorMap(
            keys=metatensor.torch.Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(
                    intersection_keys, device=tmap_1.keys.values.device
                ),
            ),
            blocks=intersection_blocks,
        )
        intersection_map = self.linear_contractions(intersection_map)

        key_union = ls_1.union(ls_2)
        union_blocks: List[metatensor.torch.TensorBlock] = []
        union_keys: List[List[int]] = []
        for key in key_union.values:
            o3_lambda = int(key[0])
            o3_sigma = int(key[1])
            blocks_1 = tmap_1.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            blocks_2 = tmap_2.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
            assert len(blocks_1) <= 1
            assert len(blocks_2) <= 1
            if len(blocks_1) == 1 and len(blocks_2) == 1:
                continue
            elif len(blocks_1) == 1:
                union_blocks.append(blocks_1[0])
                union_keys.append([nu_1, o3_lambda, o3_sigma])
            elif len(blocks_2) == 1:
                union_blocks.append(blocks_2[0])
                union_keys.append([nu_2, o3_lambda, o3_sigma])
            else:
                raise ValueError("This should never happen. Metatensor bug?")

        if len(union_keys) == 0:
            return intersection_map

        union_map = metatensor.torch.TensorMap(
            keys=metatensor.torch.Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(union_keys, device=tmap_1.keys.values.device),
            ),
            blocks=union_blocks,
        )

        new_map = metatensor.torch.TensorMap(
            keys=metatensor.torch.Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.concatenate(
                    [intersection_map.keys.values, union_map.keys.values]
                ),
            ),
            blocks=intersection_map.blocks() + union_map.blocks(),
        )

        return new_map


# class EquivariantTensorAdd(torch.nn.Module):
#     # Kept as a torch.nn.module.
#     # Maybe we will want to do linear combinations of these in the future
#     # with learnable coefficients.

#     def __init__(self):
#         super().__init__()

#     def forward(
#         self, tmap_1: metatensor.torch.TensorMap, tmap_2: metatensor.torch.TensorMap
#     ):
#         # for now, this assumes that each of the two tensor maps have the same "nu"
#         # keys
#         # actually, we take the maximum
#         nu_1 = int(torch.max(tmap_1.keys.column("nu")))
#         nu_2 = int(torch.max(tmap_2.keys.column("nu")))
#         nu_max = max(nu_1, nu_2)

#         new_blocks: List[metatensor.torch.TensorBlock] = []
#         new_keys: List[List[int]] = []

#         ls_1 = tmap_1.keys.remove("nu")
#         ls_2 = tmap_2.keys.remove("nu")
#         key_union = ls_1.union(ls_2)

#         for key in key_union.values:
#             o3_lambda = int(key[0])
#             o3_sigma = int(key[1])
#             blocks_1 = tmap_1.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
#             blocks_2 = tmap_2.blocks({"o3_lambda": o3_lambda, "o3_sigma": o3_sigma})
#             assert len(blocks_1) <= 1
#             assert len(blocks_2) <= 1
#             if len(blocks_1) == 1 and len(blocks_2) == 1:
#                 block_1 = blocks_1[0]
#                 block_2 = blocks_2[0]
#                 new_block = metatensor.torch.TensorBlock(
#                     values=(block_1.values + block_2.values) * (0.5 ** 0.5),
#                     samples=block_1.samples,
#                     components=block_1.components,
#                     properties=block_1.properties,
#                 )
#                 new_blocks.append(new_block)
#                 new_keys.append([nu_max, o3_lambda, o3_sigma])
#             elif len(blocks_1) == 1:
#                 new_blocks.append(blocks_1[0])
#                 new_keys.append([nu_1, o3_lambda, o3_sigma])
#             elif len(blocks_2) == 1:
#                 new_blocks.append(blocks_2[0])
#                 new_keys.append([nu_2, o3_lambda, o3_sigma])
#             else:
#                 raise ValueError("This should never happen. Metatensor bug?")

#         new_map = metatensor.torch.TensorMap(
#             keys=metatensor.torch.Labels(
#                 names=["nu", "o3_lambda", "o3_sigma"],
#                 values=torch.tensor(new_keys, device=tmap_1.keys.values.device),
#             ),
#             blocks=new_blocks,
#         )
#         return new_map
