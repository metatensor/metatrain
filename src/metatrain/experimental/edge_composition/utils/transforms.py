from copy import deepcopy
from typing import Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System
from torch import Tensor

from elearn.interface.metatensor.couple import couple_tensor_blocks, uncouple_tensor_blocks
from featomic.torch.clebsch_gordan._coefficients import calculate_cg_coefficients

from .samples import match_samples_to_neighborlist

def batch_neighborlist(
    systems: list[System], nl_options: NeighborListOptions
) -> TensorMap:

    first_nl = systems[0].get_neighbor_list(nl_options)

    all_samples = {}
    # Accumulate the vectors for each pair of atom types across all systems.
    all_vs = {}
    for i_system, system in enumerate(systems):
        neighbor_list = system.get_neighbor_list(nl_options)
        system_vs = neighbor_list.values
        first_atom_types = system.types[neighbor_list.samples.values[:, 0]]
        second_atom_types = system.types[neighbor_list.samples.values[:, 1]]

        unique_atom_types = torch.unique(
            torch.cat([first_atom_types, second_atom_types])
        )

        # Get masks for each pair of atom types
        masks = {}
        for type1 in unique_atom_types:
            for type2 in unique_atom_types:
                if not type2 >= type1:
                    continue
                mask = (first_atom_types == type1) & (second_atom_types == type2)
                key = (int(type1.item()), int(type2.item()))
                if str(key) not in masks:
                    masks[str(key)] = mask
                else:
                    masks[str(key)] = masks[str(key)] | mask

        # Accumulate vs
        for key, mask in masks.items():
            edge_type_vs = system_vs[mask]
            samples_values = neighbor_list.samples.values[mask]
            system_samples = torch.full((samples_values.shape[0], 1), i_system)
            samples_values = torch.concatenate([system_samples, samples_values], dim=1)
            if key not in all_vs:
                all_vs[key] = edge_type_vs
                all_samples[key] = samples_values
            else:
                all_vs[key] = torch.concatenate([all_vs[key], edge_type_vs], dim=0)
                all_samples[key] = torch.concatenate(
                    [all_samples[key], samples_values], dim=0
                )

    # Create TensorBlocks for each pair of atom types
    blocks: list[TensorBlock] = []
    keys: list[list[int]] = []
    for key, vs in all_vs.items():
        samples = Labels(
            names=[
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=all_samples[key],
        )
        block = TensorBlock(
            values=vs,
            samples=samples,
            components=first_nl.components,
            properties=first_nl.properties,
        )
        blocks.append(block)
        type1, type2 = key.replace("(", "").replace(")", "").split(", ")
        keys.append([int(type1), int(type2)])

    return TensorMap(
        keys=Labels(
            names=["first_atom_type", "second_atom_type"],
            values=torch.tensor(keys, dtype=torch.int64, device=first_nl.device),
        ),
        blocks=blocks,
    )

def radial_to_spherical_harmonics(
    radial_outputs: dict[str, Tensor],
    all_shs: dict[str, Tensor],
    layout: TensorMap,
    batched_neighborlist: TensorMap,
    dense_cg_coeffs: Optional[TensorMap] = None,
) -> TensorMap:
    
    rank = len(layout.block(0).components)

    blocks: list[TensorBlock] = []
    keys = []
    pointers: dict[str, int] = {}
    for block_key, layout_block in layout.items():

        type1, type2 = block_key["first_atom_type"], block_key["second_atom_type"]

        search_key = str((type1, type2))

        if search_key not in radial_outputs:
            continue

        start_prop = pointers.get(search_key, 0)
        n_props = layout_block.properties.values.shape[0]
        end_prop = start_prop + n_props
        pointers[search_key] = end_prop

        radial_out = radial_outputs[search_key][..., start_prop:end_prop]

        shs = all_shs[search_key]
        if rank == 2:
            l1, l2 = block_key["o3_lambda_1"], block_key["o3_lambda_2"]
            shs1 = shs[:, int(l1**2) : int(l1**2 + (2 * l1 + 1))]
            shs2 = shs[:, int(l2**2) : int(l2**2 + (2 * l2 + 1))]
            sph_out = torch.einsum("sp, sC, sc -> sCcp", radial_out, shs1, shs2)
        else:
            l = block_key["o3_lambda"]
            shs1 = shs[:, int(l**2) : int(l**2 + (2 * l + 1))]
            sph_out = torch.einsum("sp, sC -> sCp", radial_out, shs1)

        samples = batched_neighborlist.block(
            dict(first_atom_type=type1, second_atom_type=type2)
        ).samples
        
        if type1 == type2:
            # Select only the upper triangular part of the block to avoid double counting.
            mask = samples.values[:, 1] < samples.values[:, 2]
            samples = Labels(
                names=samples.names,
                values=samples.values[mask],
            )
            sph_out = sph_out[mask]
        
        block = TensorBlock(
            values=sph_out,
            samples=samples,
            components=layout_block.components,
            properties=layout_block.properties,
        )

        blocks.append(block)
        keys.append(block_key.values)

    tmap = TensorMap(
        keys=Labels(
            names=layout.keys.names,
            values=torch.stack(keys, dim=0),
        ),
        blocks=blocks,
    )

    if rank == 1:
        tmap = uncouple_tensor_blocks(tmap, dense_cg_coeffs.to(device=tmap.device, dtype=tmap.dtype))

    return tmap

def spherical_harmonics_to_radial(sph_tmap, all_shs, layout, batched_neighborlist):

    rank = len(sph_tmap.block(0).components)

    radial_values = {}
    for block_key, layout_block in layout.items():

        try:
            sph_block = sph_tmap.block(block_key)
        except ValueError:
            continue

        *_, type1, type2 = block_key
        shs = all_shs[(type1, type2)]

        if rank == 2:
            l1, l2, o3_sigma_1, o3_sigma_2, type1, type2 = block_key

            shs1 = shs[:, l1**2 : l1**2 + (2 * l1 + 1)]
            shs2 = shs[:, l2**2 : l2**2 + (2 * l2 + 1)]

            radial_vals = torch.einsum("sCcp, sC, sc -> sp", sph_block.values, shs1, shs2)
        else:
            l, sigma, type1, type2 = block_key
            shs1 = shs[:, l**2 : l**2 + (2 * l + 1)]
            radial_vals = torch.einsum("sCp, sC -> sp", sph_block.values, shs1)

        if (type1, type2) not in radial_values:
            radial_values[(type1, type2)] = radial_vals
        else:
            radial_values[(type1, type2)] = torch.concatenate(
                [radial_values[(type1, type2)], radial_vals], dim=-1
            )

    blocks = []
    keys = []
    for (type1, type2), radial in radial_values.items():
        samples = batched_neighborlist.block(
            dict(first_atom_type=type1, second_atom_type=type2)
        ).samples
        block = TensorBlock(
            values=radial,
            samples=samples,
            components=[],
            properties=Labels(
                names=["_"], values=torch.arange(radial.shape[1]).reshape(-1, 1)
            ),
        )

        blocks.append(block)
        keys.append([type1, type2])

    return TensorMap(
        keys=Labels(
            names=["first_atom_type", "second_atom_type"],
            values=torch.tensor(keys, dtype=torch.int64, device=sph_tmap.keys.device),
        ),
        blocks=blocks,
    )


def get_batch_neighborlist_transform(nl_options):
    def transform(systems, targets, extra_data):
        extra_data["batched_neighborlist"] = batch_neighborlist(systems, nl_options)
        for target_name, target_tmap in targets.items():
            targets[target_name] = match_samples_to_neighborlist(
                target_tmap, extra_data["batched_neighborlist"], extra_data
            )
        return systems, targets, extra_data

    return transform


def get_radial_training_transform(nl_options, model):

    layouts = deepcopy(model.layouts)
    spherical_harmonics = model.spherical_harmonics.__class__(model.max_l)

    def transform(systems, targets, extra_data):
        extra_data["batched_neighborlist"] = batch_neighborlist(systems, nl_options)

        # Compute spherical harmonics for all pairs of atom types
        all_shs = {}
        all_ds = {}
        for key, block in extra_data["batched_neighborlist"].items():
            type1, type2 = int(key[0]), int(key[1])
            vs = block.values.reshape(-1, 3)
            shs = spherical_harmonics(vs)
            # Set scalars to 1 for convienience.
            all_shs[(type1, type2)] = shs
            ds = torch.linalg.norm(vs, dim=-1)
            all_ds[(type1, type2)] = ds

        for target_name, target_tmap in targets.items():
            target_tmap = match_samples_to_neighborlist(
                target_tmap, extra_data["batched_neighborlist"], extra_data
            )

            targets[target_name] = spherical_harmonics_to_radial(
                target_tmap,
                all_shs,
                layouts[target_name],
                extra_data["batched_neighborlist"],
            )

        return systems, targets, extra_data

    return transform


def get_coupling_transform(max_l):
    
    # Precompute CG coefficients to avoid deadlocks with multiple workers.
    # For some reason the CG coefficients can't be computed in a forked
    # process if they have already been computed in the main process.
    cg_coeffs = calculate_cg_coefficients(
        max_l,
        cg_backend="python-sparse",
        arrays_backend="torch",
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    
    def transform(systems, targets, extra_data):
        for target_name, target_tmap in targets.items():
            targets[target_name] = couple_tensor_blocks(target_tmap, cg_coeffs)
        return systems, targets, extra_data

    return transform
