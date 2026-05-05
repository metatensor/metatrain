from collections import defaultdict

import numpy as np
import sisl
import torch
from e3nn import o3
from graph2mat import (
    Formats,
    conversions,
)
from graph2mat.core.data.basis import get_change_of_basis
from metatensor.torch import Labels, TensorBlock, TensorMap


Formats.TENSORMAP = "tensormap"


@conversions.converter(Formats.BASISCONFIGURATION, Formats.TENSORMAP)
def basisconfiguration_to_tensormap(config, i: int = 0):
    tensorblocks = defaultdict(list)
    tensorblocks_samples = defaultdict(list)
    block_shapes = {}

    block_dict_matrix = config.matrix.block_dict
    lattice = sisl.Lattice(config.cell, nsc=config.matrix.nsc)

    for k, v in block_dict_matrix.items():
        first_atom, second_atom, cell_index = k

        cell_shift = lattice.sc_off[cell_index]

        center_type = config.point_types[first_atom]
        neighbor_type = config.point_types[second_atom]

        tensorblocks[(center_type, neighbor_type)].append(v)
        tensorblocks_samples[(center_type, neighbor_type)].append(
            [i, first_atom, second_atom, *cell_shift]
        )

        block_shapes[(center_type, neighbor_type)] = v.shape

    tensorblocks = {k: np.array(v) for k, v in tensorblocks.items()}

    keys = list(tensorblocks.keys())

    return TensorMap(
        keys=Labels(["first_atom_type", "second_atom_type"], torch.tensor(keys)),
        blocks=[
            TensorBlock(
                values=torch.tensor(tensorblocks[key])
                .reshape(-1, block_shapes[key][0], block_shapes[key][1], 1)
                .to(torch.float64),
                samples=Labels(
                    names=[
                        "system",
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                    ],
                    values=torch.tensor(tensorblocks_samples[key]),
                ),
                components=[
                    Labels(
                        ["first_atom_basis_function"],
                        torch.arange(block_shapes[key][0]).reshape(-1, 1),
                    ),
                    Labels(
                        ["second_atom_basis_function"],
                        torch.arange(block_shapes[key][1]).reshape(-1, 1),
                    ),
                ],
                properties=Labels(["_"], torch.tensor([[0]])),
            )
            for key in keys
        ],
    )


def get_target_converters(
    basis_table,
    in_format: str,
    out_format: str,
) -> dict[int, torch.Tensor]:
    """Get the converters from spherical harmonics for each basis type in the basis table."""

    cob, _ = get_change_of_basis(in_format, out_format)

    p_change_of_basis = torch.tensor(cob)
    d_change_of_basis = o3.Irrep(2, 1).D_from_matrix(p_change_of_basis)

    converters = {}
    for point_basis in basis_table.basis:
        M = torch.eye(point_basis.basis_size)

        start = 0
        for mul, l, p in point_basis.basis:
            if p != (-1) ** l:
                raise ValueError(
                    "Only spherical basis with definite parity are supported."
                )

            for i in range(mul):
                end = start + (2 * l + 1)
                if l == 1:
                    M[start:end, start:end] = p_change_of_basis
                elif l == 2:
                    M[start:end, start:end] = d_change_of_basis
                else:
                    M[start:end, start:end] = o3.Irrep(l, 1).D_from_matrix(
                        p_change_of_basis
                    )
                start = end

        converters[point_basis.type] = M

    return converters


def transform_tensormap_matrix(
    tmap: TensorMap,
    converters: dict[int, torch.Tensor],
) -> TensorMap:
    transformed_blocks = []

    for key in tmap.keys:
        block = tmap[key]

        block_values = block.values
        first_atom_type, second_atom_type = key

        conv_left = converters[int(first_atom_type)].to(block_values.dtype)
        conv_right = converters[int(second_atom_type)].to(block_values.dtype)

        transformed_values = torch.einsum(
            "ij, bjkp, kl -> bilp", conv_left, block_values, conv_right.T
        )

        transformed_block = TensorBlock(
            values=transformed_values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        transformed_blocks.append(transformed_block)

    return TensorMap(
        keys=tmap.keys,
        blocks=transformed_blocks,
    )


@conversions.converter(Formats.TENSORMAP, Formats.BLOCK_DICT)
def tensormap_to_blockdict(tmap, lattice):
    block_dict = {}

    for key in tmap.keys:
        block = tmap[key]

        block_values = block.values
        samples = block.samples.values

        for i_pair, (
            i_system,
            first_atom,
            second_atom,
            *cell_shifts,
        ) in enumerate(samples):
            cell_index = lattice.isc_off[cell_shifts[0], cell_shifts[1], cell_shifts[2]]

            block_dict[(int(first_atom), int(second_atom), int(cell_index))] = (
                block_values[i_pair].squeeze(-1)
            )

    return block_dict
