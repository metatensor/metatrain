from functools import partial
from typing import List, Tuple, Optional, Dict, Union, NamedTuple

import numpy as np
import torch
import vesin

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatensor.torch.learn.data import IndexedDataset, DataLoader
from metatensor.torch.learn.data._namedtuple import namedtuple

from metatrain.experimental.nanopet.modules.augmentation import (
    RotationalAugmenter,
    get_random_rotation,
    get_random_inversion,
)
from metatrain.utils.data import TargetInfo


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

    assert len(in_keys_edge_sliced) == len(out_properties_edge_sliced)

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


def drop_empty_blocks(tensor: TensorMap) -> TensorMap:
    """
    Drops blocks from a TensorMap that have been sliced to zero samples.
    """
    # Return None if None
    if tensor is None:
        return tensor

    # Find keys to drop
    keys_to_drop = []
    for key, block in tensor.items():
        if any([dim == 0 for dim in block.values.shape]):
            keys_to_drop.append(key)

    if len(keys_to_drop) == 0:
        return tensor

    # Drop blocks
    tensor_dropped = mts.drop_blocks(
        tensor,
        keys=mts.Labels(
            names=keys_to_drop[0].names,
            values=torch.tensor(
                [[i for i in k.values] for k in keys_to_drop], dtype=torch.int64
            ),
        ),
    )

    return tensor_dropped


def get_tensor_invariant_mean(tensor: TensorMap) -> TensorMap:
    """
    Computes the per-property mean for all invariant blocks and returns this in a
    TensorMap.
    """
    mean_key_values = []
    mean_blocks = []
    for key, block in tensor.items():
        if key["o3_lambda"] != 0:
            continue

        mean_key_values.append(key.values)
        mean_values = torch.mean(block.values, dim=0)
        mean_blocks.append(
            TensorBlock(
                samples=Labels.single(),
                components=[],
                properties=block.properties,
                values=mean_values.reshape(1, -1),
            )
        )
    return TensorMap(
        mts.Labels(
            tensor.keys.names,
            torch.stack(mean_key_values).reshape(-1, len(tensor.keys.names)),
        ),
        mean_blocks,
    )


def get_tensor_std(tensor: TensorMap) -> TensorMap:
    """
    For each block, computes the norm over components and then a standard deviation over
    samples. Scales this by the length of the ISC (i.e. 2l + 1). Stores the resulting
    vector of length `len(block.properties)` in a TensorBlock, and returns the total
    result in a TensorMap.
    """
    std_blocks = []
    for key, block in tensor.items():

        # std_values = torch.std(torch.norm(block.values, dim=1), dim=0) * (
        #     (2 * key["o3_lambda"] + 1) ** 0.5
        # )
        # # If nan, set std to 1
        # if torch.any(torch.isnan(std_values)):
        #     print("Warning: std is nan, setting to 1")
        #     std_values = torch.tensor([1.0] * len(block.properties))
        # std_blocks.append(
        #     TensorBlock(
        #         samples=Labels.single(),
        #         components=[],
        #         properties=block.properties,
        #         values=std_values.reshape(1, -1),
        #     )
        # )

        std_values = torch.std(block.values, dim=0) * (2 * key["o3_lambda"] + 1)
        std_blocks.append(
            TensorBlock(
                samples=Labels.single(),
                components=block.components,
                properties=block.properties,
                values=std_values.reshape(1, 2 * key["o3_lambda"] + 1, -1),
            )
        )

    return TensorMap(tensor.keys, std_blocks)


# ===== Metadata from basis sets =====


def get_one_center_metadata(
    basis_set: Dict[str, Dict[int, Union[int, List[int]]]],
) -> Dict[str, Union[mts.Labels, List[mts.Labels]]]:
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
    basis_set: Dict[str, Dict[int, Union[int, List[int]]]],
    triangulate_center_types: bool = False,
    permutation_symmetry: bool = False,
) -> Dict[str, Union[mts.Labels, List[mts.Labels]]]:
    """
    Parses the basis set definition and returns the metadata for two-center targets.

    Return the keys and out properties of the node and edge features, in a dict.
    """

    edge_key_names = ["o3_lambda", "o3_sigma", "first_atom_type", "second_atom_type"]
    if permutation_symmetry:
        edge_key_names.append("block_type")
    node_key_names = ["o3_lambda", "o3_sigma", "center_type"]

    # Edge keys
    keys_values_edge = {}
    for center_1_symbol in basis_set:
        for center_2_symbol in basis_set:
            same_species = center_1_symbol == center_2_symbol
            if permutation_symmetry:
                if same_species:
                    block_types = [-1, 0, 1]
                else:
                    block_types = [2]
            else:
                block_types = [2]
            for block_type in block_types:
                for o3_lambda_1, radial_basis_1 in basis_set[center_1_symbol].items():
                    if isinstance(radial_basis_1, int):
                        radial_basis_1 = list(range(radial_basis_1))
                    for o3_lambda_2, radial_basis_2 in basis_set[
                        center_2_symbol
                    ].items():
                        if isinstance(radial_basis_2, int):
                            radial_basis_2 = list(range(radial_basis_2))
                        for o3_lambda in range(
                            abs(o3_lambda_1 - o3_lambda_2),
                            abs(o3_lambda_1 + o3_lambda_2) + 1,
                        ):
                            o3_sigma = (-1) ** (o3_lambda_1 + o3_lambda_2 + o3_lambda)
                            # Skip blocks that are zero by symmetry
                            zero_by_symmetry = (
                                o3_sigma == -1 and block_type in (0, 1)
                            ) or (o3_sigma == 1 and block_type == -1)

                            for n_1 in radial_basis_1:
                                for n_2 in radial_basis_2:

                                    same_orbital = (n_1 == n_2) and (
                                        o3_lambda_1 == o3_lambda_2
                                    )

                                    if (
                                        same_species
                                        and same_orbital
                                        and zero_by_symmetry
                                    ):
                                        continue

                                    key = (
                                        o3_lambda,
                                        o3_sigma,
                                        ATOMIC_SYMBOLS_TO_NUMBERS[center_1_symbol],
                                        ATOMIC_SYMBOLS_TO_NUMBERS[center_2_symbol],
                                        block_type,
                                        o3_lambda_1,
                                        o3_lambda_2,
                                    )
                                    if key not in keys_values_edge:
                                        keys_values_edge[key] = []
                                    keys_values_edge[key].append([n_1, n_2])

    keys_values_edge_new = {}
    for key, val in keys_values_edge.items():
        if len(val) != 0:
            keys_values_edge_new[key] = val
    keys_values_edge = keys_values_edge_new

    metadata = TensorMap(
        Labels(
            [
                "o3_lambda",
                "o3_sigma",
                "first_atom_type",
                "second_atom_type",
                "block_type",
                "orb_l_1",
                "orb_l_2",
            ],
            torch.tensor(list(keys_values_edge.keys())),
        ),
        [
            TensorBlock(
                samples=Labels.single(),
                components=[],
                properties=Labels(
                    ["orb_n_1", "orb_n_2"], torch.tensor(keys_values_edge[key])
                ),
                values=torch.empty(1, len(keys_values_edge[key])),
            )
            for key in keys_values_edge
        ],
    )
    metadata = mts.permute_dimensions(
        metadata.keys_to_properties(["orb_l_1", "orb_l_2"]), "properties", [2, 0, 3, 1]
    )

    if not permutation_symmetry:
        metadata = mts.remove_dimension(metadata, "keys", "block_type")

    in_keys_edge = metadata.keys
    out_properties_edge = [block.properties for block in metadata]
    assert len(in_keys_edge) == len(out_properties_edge)

    # Finally treat the special case of node, where block_type == 0
    in_keys_values_node = []
    out_properties_node = []
    in_keys_values_edge = []
    out_properties_edge_new = []
    for key_i, key in enumerate(in_keys_edge):
        if permutation_symmetry:
            if key["block_type"] == 0:  # this is a node
                assert key["first_atom_type"] == key["second_atom_type"]
                in_keys_values_node.append(key.values[:3])
                out_properties_node.append(out_properties_edge[key_i])
            else:
                in_keys_values_edge.append(key.values)
                out_properties_edge_new.append(out_properties_edge[key_i])
        else:
            in_keys_values_edge.append(key.values)
            out_properties_edge_new.append(out_properties_edge[key_i])
            if key["first_atom_type"] == key["second_atom_type"]:
                in_keys_values_node.append(key.values[:3])
                out_properties_node.append(out_properties_edge[key_i])

    in_keys_node = mts.Labels(node_key_names, torch.stack(in_keys_values_node))
    in_keys_edge = mts.Labels(
        edge_key_names,
        torch.stack(in_keys_values_edge),
    )

    if triangulate_center_types:
        in_keys_edge, out_properties_edge_new = keys_triu_center_type(
            in_keys_edge, out_properties_edge_new
        )

    assert len(in_keys_edge) == len(out_properties_edge_new)
    assert len(in_keys_node) == len(out_properties_node)

    return {
        "in_keys_node": in_keys_node,
        "out_properties_node": out_properties_node,
        "in_keys_edge": in_keys_edge,
        "out_properties_edge": out_properties_edge_new,
    }


def get_edges(tensor: TensorMap) -> Dict[str, TensorMap]:
    """
    Splits the two-center target ``tensor`` into node and edge tensors.
    """

    # Now edges
    edge_keys = []
    edge_blocks = []
    for key, block in tensor.items():

        edge_keys.append(key.values)

        # Assert non-periodic for now. TODO: periodic!
        assert all(block.samples.column("cell_shift_a") == 0)
        assert all(block.samples.column("cell_shift_b") == 0)
        assert all(block.samples.column("cell_shift_c") == 0)

        # Slice samples to off-site
        samples_mask = torch.where(
            block.samples.values[:, block.samples.names.index("first_atom")]
            != block.samples.values[:, block.samples.names.index("second_atom")]
        )
        edge_blocks.append(
            TensorBlock(
                samples=Labels(
                    block.samples.names,
                    block.samples.values[samples_mask],
                ),
                components=block.components,
                properties=block.properties,
                values=block.values[samples_mask],
            )
        )

    edge_tensor = TensorMap(
        Labels(tensor.keys.names, torch.stack(edge_keys)), edge_blocks
    )
    edge_tensor = drop_empty_blocks(edge_tensor)

    # return node_tensor, edge_tensor
    return edge_tensor


# ===== Training utils ===== #


def group_and_join_nonetypes(
    batch: List[NamedTuple],
    fields_to_join: Optional[List[str]] = None,
    join_kwargs: Optional[dict] = None,
) -> NamedTuple:
    """
    A modified form of :py:meth:`metatensor.torch.learn.data.group_and_join` that
    handles data fields that are NoneType. Any fields that are a list of ``None`` are
    'joined' to a single ``None``. All other functionality is the same, but

    This is useful for passing data straight to the :py:class:`rholearn.loss.RhoLoss`
    class.
    """
    data: List[Union[TensorMap, torch.Tensor]] = []
    names = batch[0]._fields
    if fields_to_join is None:
        fields_to_join = names
    if join_kwargs is None:
        join_kwargs = {}
    for name, field in zip(names, list(zip(*batch))):

        if name == "sample_id":  # special case, keep as is
            data.append(field)
            continue

        if name in fields_to_join:  # Join tensors if requested
            if isinstance(field[0], torch.ScriptObject) and field[0]._has_method(
                "keys_to_properties"
            ):  # inferred metatensor.torch.TensorMap type
                data.append(mts.join(field, axis="samples", **join_kwargs))
            elif isinstance(field[0], torch.Tensor):  # torch.Tensor type
                data.append(torch.vstack(field))
            elif isinstance(field[0], type(None)):  # NoneType
                data.append(None)
            else:
                data.append(field)

        else:  # otherwise just keep as a list
            data.append(field)

    return namedtuple("Batch", names)(*data)


def get_dataset(systems, system_id, target_node, target_edge):
    """Returns a dataset with systems, and target nodes and edges"""
    return IndexedDataset(
        sample_id=system_id,
        systems=[systems[i] for i in system_id],
        targets_node=[
            mts.slice(
                target_node,
                "samples",
                mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)),
            )
            for A in system_id
        ],
        targets_edge=[
            mts.slice(
                target_edge,
                "samples",
                mts.Labels(["system"], torch.tensor([A]).reshape(-1, 1)),
            )
            for A in system_id
        ],
    )


def get_dataloader(dataset, **kwargs):
    """Returns a dataloader"""
    return DataLoader(
        dataset,
        collate_fn=partial(
            group_and_join_nonetypes,
            join_kwargs={
                "remove_tensor_name": True,
                "different_keys": "union",
            },
        ),
        shuffle=True,
        **kwargs,
    )


def get_augmenter(
    target_node: Optional[TensorMap] = None,
    target_edge: Optional[TensorMap] = None,
) -> RotationalAugmenter:
    """
    Returns a RotationalAugmenter for node and/or edge targets.
    """
    target_info_dict = {}
    if target_node is not None:
        target_info_dict.update(
            {
                "mtt::target_node": TargetInfo(
                    quantity="node",
                    unit="Angstrom^-3",
                    layout=mts.slice(
                        target_node,
                        "samples",
                        mts.Labels(["system"], torch.tensor([-1]).reshape(-1, 1)),
                    ),
                )
            }
        )
    if target_edge is not None:
        target_info_dict.update(
            {
                "mtt::target_edge": TargetInfo(
                    quantity="edge",
                    unit="Angstrom^-3",
                    layout=mts.slice(
                        target_edge,
                        "samples",
                        mts.Labels(["system"], torch.tensor([-1]).reshape(-1, 1)),
                    ),
                )
            }
        )
    return RotationalAugmenter(target_info_dict)


def l2loss(input: TensorMap, target: TensorMap) -> torch.Tensor:
    """Computes the squared loss (reduction = sum) between the input and target TensorMaps"""

    loss = 0
    for k in target.keys():
        assert k in input.keys()
        mts.equal_metadata_raise(input[k], target[k])
        for key in target[k].keys:
            loss += torch.sum((input[k][key].values - target[k][key].values) ** 2)

    return loss


def get_system_transformations(systems) -> List[torch.Tensor]:
    """
    Returns a series of random transformations to be applied for each system in
    ``systems``.
    """
    rotations = [get_random_rotation() for _ in range(len(systems))]
    inversions = [get_random_inversion() for _ in range(len(systems))]
    return rotations, inversions


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
