from typing import List

import torch
from metatensor.torch import Labels, TensorMap


def is_spherical_atomic_basis(target: TensorMap) -> bool:
    """
    Check if the target is a spherical atomic basis.

    :param target: The target tensor map to check.
    :returns: True if the target is a spherical atomic basis, False otherwise.
    """
    if not ("o3_lambda" in target.keys.names and "o3_sigma" in target.keys.names):
        return False

    # i.e. electron density on basis
    if (
        target.sample_names == ["system", "atom"]
        and len(target[0].components) == 1
        and target[0].components[0].names == ["o3_mu"]
        and target.property_names == ["center_type", "n"]
    ):
        return True

    # i.e. hamiltonian
    if (
        target.sample_names
        == [
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ]
        and len(target[0].components) == 1
        and target[0].components[0].names == ["o3_mu"]
        and target.property_names
        == ["first_atom_type", "second_atom_type", "l_1", "l_2", "n_1", "n_2"]
    ):
        return True

    return False


def get_block_sample_idxs_per_atom(
    sample_labels_with_types: Labels,
    atomic_basis: Labels,
    o3_lambda: int,
) -> torch.Tensor:
    """
    Based on the atomic basis, gets the sample indices for slicing features for a given
    block key. Assumes the key values correspond to dimensions ["o3_lambda", "o3_sigma"]
    respectively.
    """
    assert sample_labels_with_types.names == ["system", "atom", "center_type"]

    # find the center types that have basis functions with the given o3_lambda
    valid_center_types = torch.sort(
        torch.unique(atomic_basis.values[atomic_basis.values[:, 0] == o3_lambda][:, 1])
    )[0]

    # find the sample indices that have these center types
    slice_indices = sample_labels_with_types.select(
        Labels(
            ["center_type"],
            valid_center_types.reshape(-1, 1),
        )
    )
    return slice_indices


def get_block_sample_idxs_per_pair(
    sample_labels_with_types: Labels,
    atomic_basis: Labels,
    o3_lambda: int,
    o3_sigma: int,
    s2_pi: int,
    node_or_edge: str,
) -> torch.Tensor:
    """
    Based on the atomic basis, gets the sample indices for slicing features for a given
    block key.
    """

    if node_or_edge == "node":
        assert sample_labels_with_types.names == ["system", "atom", "center_type"]
        # Find the pairs of atom types that can be in this block
        atomic_type_pairs = atomic_basis.values[
            atomic_basis.select(
                Labels(
                    ["o3_lambda", "o3_sigma", "s2_pi"],
                    torch.tensor([o3_lambda, o3_sigma, s2_pi]).reshape(1, 3),
                )
            )
        ]
        # reduce to the same central species (i.e. a node) and remove the redundant
        # s2_pi dimension
        center_types = atomic_type_pairs[
            atomic_type_pairs[:, 3] == atomic_type_pairs[:, 4]
        ][:, 3]
        sample_idxs = sample_labels_with_types.select(
            Labels(
                ["center_type"],
                center_types.reshape(-1, 1),
            )
        )

    else:
        assert node_or_edge == "edge"
        assert sample_labels_with_types.names == [
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
            "first_atom_type",
            "second_atom_type",
            "s2_pi",
        ]
        # Find the pairs of atom types that can be in this block
        atomic_type_pairs = atomic_basis.values[
            atomic_basis.select(
                Labels(
                    ["o3_lambda", "o3_sigma", "s2_pi"],
                    torch.tensor([o3_lambda, o3_sigma, s2_pi]).reshape(1, 3),
                )
            )
        ]
        sample_idxs = sample_labels_with_types.select(
            Labels(
                ["first_atom_type", "second_atom_type", "s2_pi"],
                torch.hstack(
                    [
                        atomic_type_pairs[:, 3:],
                        torch.full((atomic_type_pairs.shape[0], 1), s2_pi),
                    ]
                ),
            )
        )

    return sample_idxs


def get_onsite_samples_mask(samples: Labels) -> torch.Tensor:
    """
    Returns a boolean mask of the samples :py:class:`Labels`, where True indicates that
    the sample is a node sample, i.e. the atom indices are equal and the cell shifts are
    zero.
    """
    assert samples.names == [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    # Get the arrays
    atom_i = samples.values[:, 1]
    atom_j = samples.values[:, 2]
    norms = torch.linalg.norm(1.0 * samples.values[:, -3:], axis=1)
    zeros = torch.tensor([0.0], dtype=norms.dtype, device=norms.device)
    is_central_cell = torch.isclose(norms, zeros)
    return (atom_i == atom_j) & is_central_cell


def symmetrize_edge_samples(
    edge_samples: Labels,
) -> torch.Tensor:
    """
    Symmetrizes the samples labels of the raw PET edge features.

    This takes in the samples labels for the raw PET edge features, i.e. off site atom
    pairs. The following procedure is applied to the different kinds of pair sample
    present:

    - atom pairs with different atom types: no symmetrization required. As only unique
      edge features are needed, keep only the features where the first atom type is less
      than the second atom type. The s2_pi value is set to 0.

    - atom pairs with the same atom type: symmetrization required. For each of these
      pairs, the corresponding permuted sample is found by swapping the atom indices and
      inverting the sign of the cell shifts. A "plus" and "minus" combination of the
      features is created by adding or subtracting the feature with its permutation, and
      the s2_pi value set to +1 and -1 respectively.
    """
    sample_names: List[str] = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
        "first_atom_type",
        "second_atom_type",
    ]
    assert edge_samples.names == sample_names

    # 1) first_atom_type < second_atom_type: no symmetrization required. Ensure
    #    triangular in atom type.
    diff_type_mask = edge_samples.values[:, 6] != edge_samples.values[:, 7]
    edge_sample_values_sym_0 = edge_samples.values[diff_type_mask]
    edge_sample_values_sym_0 = torch.hstack(
        [
            # s2_pi = 0
            edge_sample_values_sym_0,
            torch.zeros((len(edge_sample_values_sym_0), 1), dtype=torch.int32),
        ]
    )

    # 2) first_atom_type == second_atom_type: symmetrization required
    same_type_mask = edge_samples.values[:, 6] == edge_samples.values[:, 7]
    edge_sample_values_sym = edge_samples.values[same_type_mask]

    # Create the plus and minus combinations
    edge_sample_values_sym_p1 = torch.hstack(
        [
            # s2_pi = +1
            edge_sample_values_sym,
            torch.ones((len(edge_sample_values_sym), 1), dtype=torch.int32),
        ]
    )
    edge_sample_values_sym_m1 = torch.hstack(
        [
            # s2_pi = -1
            edge_sample_values_sym,
            torch.ones((len(edge_sample_values_sym), 1), dtype=torch.int32) * -1,
        ]
    )

    return torch.vstack(
        [
            edge_sample_values_sym_0,
            edge_sample_values_sym_p1,
            edge_sample_values_sym_m1,
        ]
    )


def symmetrize_edge_features(
    edge_samples: Labels,
    edge_features: torch.Tensor,
) -> torch.Tensor:
    """Symmetrizes the edge PET features for atoms with the same type"""
    assert len(edge_samples) == edge_features.shape[0]

    sample_names: List[str] = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
        "first_atom_type",
        "second_atom_type",
    ]
    assert edge_samples.names == sample_names

    # 1) different atom types, no symmetrization required
    diff_type_mask = edge_samples.values[:, 6] != edge_samples.values[:, 7]
    edge_features_sym_0 = edge_features[diff_type_mask]

    # 2) same atom type, symmetrized required
    same_type_mask = edge_samples.values[:, 6] == edge_samples.values[:, 7]
    edge_sample_values_sym = edge_samples.values[same_type_mask]
    edge_features_sym = edge_features[same_type_mask]

    # Permute the samples values by swapping the atom indices (and atom types) and
    # reversing the sign of the cell shifts
    edge_samples_values_sym_perm = (
        Labels(sample_names, edge_sample_values_sym)
        .permute([0, 2, 1, 3, 4, 5, 7, 6])
        .values.clone()
    )
    edge_samples_values_sym_perm[:, 3:6] *= -1

    # Find the indices that map the unpermuted to the permuted samples
    idx_sym = Labels(sample_names, edge_sample_values_sym).select(
        Labels(sample_names, edge_samples_values_sym_perm)
    )

    # Create the plus and minus combinations
    edge_features_sym_p1 = edge_features_sym + edge_features_sym[idx_sym]
    edge_features_sym_m1 = edge_features_sym - edge_features_sym[idx_sym]

    return torch.vstack(
        [
            edge_features_sym_0,
            edge_features_sym_p1,
            edge_features_sym_m1,
        ]
    )
