from typing import List, Tuple, Optional

import numpy as np
import torch
import vesin

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap


def symmetrize_samples(
    block: TensorBlock, second_atom_type: Optional[int] = None
) -> Tuple[TensorBlock]:
    """
    Symmetrizes the samples dimension of a tensor block.
    """

    # Define the samples
    all_samples = block.samples
    sample_names = all_samples.names

    # Permute the samples to get the negative samples
    permuted_samples = all_samples.permute([0, 2, 1, 3, 4, 5]).values.clone()
    permuted_samples[:, -3:] *= -1
    values = block.values  # .clone() # TODO: is this to be cloned?

    # Find the indices of the samples to symmetrize
    idx_to_symmetrize = all_samples.select(
        Labels(
            names=sample_names,
            values=permuted_samples,
        )
    )

    # print(sample_names, permuted_samples)
    # print(idx_to_symmetrize.shape, values.shape, idx_to_symmetrize)
    # Symmetrize the sample values
    values_plus = values + values[idx_to_symmetrize]
    values_minus = values - values[idx_to_symmetrize]

    reduced_samples_mask = (
        (all_samples.values[:, 1] < all_samples.values[:, 2])
        & torch.isclose(
            torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
        )
    ) | (
        (all_samples.values[:, 1] <= all_samples.values[:, 2])
        & (
            ~torch.isclose(
                torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
            )
        )
    )
    reduced_samples = Labels(
        sample_names,
        all_samples.values[reduced_samples_mask],
    )
    values_plus = values_plus[reduced_samples_mask]
    values_minus = values_minus[reduced_samples_mask]

    if second_atom_type is not None:
        properties = block.properties.insert(
            1,
            "neighbor_2_type",
            torch.tensor(block.properties.values.shape[0] * [second_atom_type]),
        )
    else:
        properties = block.properties

    block_plus = TensorBlock(
        samples=reduced_samples,  # mts.Labels(b.samples.names, np.array(samples)),
        values=values_plus,
        components=block.components,
        properties=properties,
    )
    block_minus = TensorBlock(
        samples=reduced_samples,  # mts.Labels(b.samples.names, np.array(samples)),
        values=values_minus,
        components=block.components,
        properties=properties,
    )

    return block_plus, block_minus


def keys_triu_center_type(
    in_keys_edge: Labels, out_properties_edge: List[Labels]
) -> TensorMap:

    idxs_to_keep = []
    for key_i, key in enumerate(in_keys_edge):
        # Keep blocks where the first atom type is less than the second atom type
        if key["first_atom_type"] <= key["second_atom_type"]:
            idxs_to_keep.append(key_i)

    in_keys_edge_sliced = Labels(
        in_keys_edge.names,
        in_keys_edge.values[idxs_to_keep],
    )
    out_properties_edge_sliced = [
        out_props
        for i, out_props in enumerate(out_properties_edge)
        if i in idxs_to_keep
    ]

    return in_keys_edge_sliced, out_properties_edge_sliced


def get_neighbor_list(
    frames,
    frame_idxs: List[int],
    cutoff: float,
) -> mts.Labels:
    """
    Computes the neighbour list for each frame in ``frames`` and returns a
    :py:class:`metatensor
    """
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True)

    labels_values = []
    for A, frame in zip(frame_idxs, frames):

        # Compute the neighbor list
        if np.any([d == 0 for d in frame.cell]):  # for ase
            # if np.any([d == 0 for d in frame.cell]):  # for chemfiles
            box = np.zeros((3, 3))
            periodic = False
        else:
            box = frame.cell.matrix
            periodic = True

        i_list, j_list, S_list = nl.compute(
            points=frame.positions,
            box=box,
            periodic=periodic,
            quantities="ijS",
        )

        # Now add in the self terms as vesin does not include them
        i_list = np.concatenate([i_list, np.arange(len(frame.positions), dtype=int)])
        j_list = np.concatenate([j_list, np.arange(len(frame.positions), dtype=int)])
        S_list = np.concatenate(
            [S_list, np.array([0, 0, 0] * len(frame.positions)).reshape(-1, 3)],
            dtype=int,
        )

        # Add dimension for system index
        for i, j, S in zip(i_list, j_list, S_list):
            a, b, c = S
            label = [A, i, j, a, b, c]
            if label not in labels_values:
                labels_values.append(label)

    return mts.Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.tensor(labels_values, dtype=torch.int32),
    )


# ===== Metadata from basis sets =====


def get_one_center_metadata(
    basis_set: dict,
) -> Tuple[mts.Labels, List[mts.Labels], mts.Labels, List[mts.Labels]]:
    """
    Parses the basis set definition and returns the metadata for one-center targets.

    Return the keys and out properties of the node features, in a dict
    """
    o3_sigma = 1  # by definition for now

    # Node keys
    keys_values_node = []
    for center_symbol, atom_basis in basis_set.items():
        for o3_lambda, radial_basis in basis_set[center_symbol].items():

            # Node key
            keys_value_node = [
                o3_lambda,
                o3_sigma,
                ATOMIC_SYMBOLS_TO_NUMBERS[center_symbol],
            ]
            if keys_value_node not in keys_values_node:
                keys_values_node.append(keys_value_node)

    in_keys_node = mts.Labels(
        ["o3_lambda", "o3_sigma", "center_type"], torch.tensor(keys_values_node)
    )

    # Node properties
    out_properties_node = []
    for key in in_keys_node:

        o3_lambda, o3_sigma, center_type = key

        radial_basis = basis_set[ATOMIC_NUMBERS_TO_SYMBOLS[center_type]][o3_lambda]
        if isinstance(radial_basis, int):
            radial_basis = list(range(radial_basis))

        out_properties_node.append(
            mts.Labels(
                ["n"], torch.tensor(radial_basis, dtype=torch.int64).reshape(-1, 1)
            )
        )

    return {
        "in_keys_node": in_keys_node,
        "out_properties_node": out_properties_node,
    }


def get_two_center_metadata(
    basis_set: dict,
) -> Tuple[mts.Labels, List[mts.Labels], mts.Labels, List[mts.Labels]]:
    """
    Parses the basis set definition and returns the metadata for two-center targets.

    Return the keys and out properties of the node and edge features, in a dict.
    """

    # Edge keys
    keys_values_edge = []
    out_properties_edge = []
    for center_1_symbol, atom_1_basis in basis_set.items():
        for center_2_symbol, atom_2_basis in basis_set.items():
            for o3_lambda_1, radial_basis_1 in basis_set[center_1_symbol].items():

                if isinstance(radial_basis_1, int):
                    radial_basis_1 = list(range(radial_basis_1))

                for o3_lambda_2, radial_basis_2 in basis_set[center_2_symbol].items():

                    if isinstance(radial_basis_2, int):
                        radial_basis_2 = list(range(radial_basis_2))

                    for o3_lambda in range(
                        abs(o3_lambda_1 - o3_lambda_2),
                        abs(o3_lambda_1 + o3_lambda_2) + 1,
                    ):

                        o3_sigma = (-1) ** (o3_lambda_1 + o3_lambda_2 + o3_lambda)

                        # Create the edge properties for this block. This doesn't depend
                        #  on the block type
                        out_properties_values_edge = []
                        for n_1 in radial_basis_1:
                            for n_2 in radial_basis_2:

                                out_properties_value_edge = [
                                    n_1,
                                    o3_lambda_1,
                                    n_2,
                                    o3_lambda_2,
                                ]
                                if (
                                    out_properties_value_edge
                                    not in out_properties_values_edge
                                ):
                                    out_properties_values_edge.append(
                                        out_properties_value_edge
                                    )

                        out_properties = mts.Labels(
                            ["n_1", "l_1", "n_2", "l_2"],
                            torch.tensor(out_properties_values_edge, dtype=torch.int64),
                        )

                        # Edge keys, taking care of block types
                        if center_1_symbol == center_2_symbol:

                            for block_type in [-1, 0, 1]:

                                # Skip blocks that are zero by symmetry
                                same_orbital = (n_1 == n_2 and o3_lambda_1 == o3_lambda_2)
                                if same_orbital and (
                                    (o3_sigma == -1 and block_type in [0, 1])
                                    or (o3_sigma == 1 and block_type == -1)
                                ):
                                    continue

                                # Create the edge key values
                                keys_value_edge = [
                                    o3_lambda,
                                    o3_sigma,
                                    ATOMIC_SYMBOLS_TO_NUMBERS[center_1_symbol],
                                    ATOMIC_SYMBOLS_TO_NUMBERS[center_2_symbol],
                                    block_type,
                                ]
                                if keys_value_edge not in keys_values_edge:
                                    keys_values_edge.append(keys_value_edge)
                                    out_properties_edge.append(out_properties)
                        else:
                            block_type = 2
                            keys_value_edge = [
                                o3_lambda,
                                o3_sigma,
                                ATOMIC_SYMBOLS_TO_NUMBERS[center_1_symbol],
                                ATOMIC_SYMBOLS_TO_NUMBERS[center_2_symbol],
                                block_type,
                            ]

                            if keys_value_edge not in keys_values_edge:
                                keys_values_edge.append(keys_value_edge)
                                out_properties_edge.append(out_properties)

    in_keys_edge = mts.Labels(
        ["o3_lambda", "o3_sigma", "first_atom_type", "second_atom_type", "block_type"],
        torch.tensor(keys_values_edge),
    )

    # Finally treat the special case of node, where block_type == 0
    in_keys_values_node = []
    out_properties_node = []
    in_keys_values_edge = []
    out_properties_edge_new = []
    for key_i, key in enumerate(in_keys_edge):
        if key["block_type"] == 0:  # this is a node
            assert key["first_atom_type"] == key["second_atom_type"]
            in_keys_values_node.append(key.values[:3])
            out_properties_node.append(out_properties_edge[key_i])
        else:
            in_keys_values_edge.append(key.values)
            out_properties_edge_new.append(out_properties_edge[key_i])
        

    in_keys_node = mts.Labels(
        ["o3_lambda", "o3_sigma", "center_type"], torch.stack(in_keys_values_node)
    )
    in_keys_edge = mts.Labels(
        ["o3_lambda", "o3_sigma", "first_atom_type", "second_atom_type", "block_type"],
        torch.stack(in_keys_values_edge)
    )

    return {
        "in_keys_node": in_keys_node,
        "out_properties_node": out_properties_node,
        "in_keys_edge": in_keys_edge,
        "out_properties_edge": out_properties_edge,
    }


# ===== For converting between atomic numbers and symbols

ATOMIC_SYMBOLS_TO_NUMBERS = {
    "X": 0,
    # Period 1
    "H": 1,
    "He": 2,
    # 2
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    # 3
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    # 4
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    # 5
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    # 6
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    # 7
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}
ATOMIC_NUMBERS_TO_SYMBOLS = {v: k for k, v in ATOMIC_SYMBOLS_TO_NUMBERS.items()}
