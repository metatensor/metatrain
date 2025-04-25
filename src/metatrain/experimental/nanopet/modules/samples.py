from typing import Dict, List

import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import System


def get_samples(
    systems: List[System],
    atomic_types: torch.Tensor,
    output_info: Dict[str, Dict[str, str]],
):
    """
    Builds the various samples labels for the PET node and edge features, for the given
    input ``systems``.

    The samples labels for the node features are always returned. For edge features,
    these are only constructed and returned if there are two-center targets being
    predicted.

    Furthermore, if spherical targets on an atomic basis are being predicted (either
    one- or two-center), the samples labels that slice the last layer features are
    returned too, in a dictionary indexed by strings that describe:

        - the atomic type of the target block (node features)

        - pairs of atomic types of the target block (edge features)

        - pairs of atomic types and the permutational symmetry (edge features,
          symmetrized)
    """
    # node samples as standard
    node_samples: Labels = Labels(
        names=["system", "atom"],
        values=get_node_sample_values(
            systems,
            sample_kind="per_atom",
            include_atom_type=False,
        ),
    )

    # edge samples (symmetrized or not) only if two-center targets are being
    # predicted. First define dummy labels, for torchscript compatibility.
    edge_samples: Labels = Labels(
        ["_"], torch.empty((0, 1), dtype=node_samples.values.dtype)
    )
    edge_samples_sym: Labels = Labels(
        ["_"], torch.empty((0, 1), dtype=node_samples.values.dtype)
    )
    ll_per_pair_samples: Labels = Labels(
        ["_"], torch.empty((0, 1), dtype=node_samples.values.dtype)
    )
    ll_per_pair_samples_sym: Labels = Labels(
        ["_"], torch.empty((0, 1), dtype=node_samples.values.dtype)
    )

    any_per_atom: bool = any(
        [
            info["target_type"] == "spherical_atomic_basis"
            and info["sample_kind"] == "per_atom"
            for info in output_info.values()
        ]
    )
    any_per_pair: bool = any(
        [
            info["target_type"] == "spherical_atomic_basis"
            and info["sample_kind"] == "per_pair"
            for info in output_info.values()
        ]
    )
    any_per_pair_sym: bool = any(
        [
            info["target_type"] == "spherical_atomic_basis"
            and info["sample_kind"] == "per_pair"
            and info["symmetrized"] == "true"
            for info in output_info.values()
        ]
    )
    if any_per_pair:  # store the samples labels for the raw PET edge features
        edge_samples = Labels(
            [
                "first_atom_type",
                "second_atom_type",
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            get_edge_sample_values(systems, include_atom_types=True),
        )
        # store the samples labels for the last layer PET features for per_pair targets,
        # i.e. the stacked node and edge samples
        ll_per_pair_samples = Labels(
            edge_samples.names,
            torch.vstack(
                [
                    get_node_sample_values(
                        systems,
                        sample_kind="per_pair",
                        include_atom_type=True,
                    ),
                    get_edge_sample_values(systems, include_atom_types=True),
                ]
            ),
        )
    if any_per_pair_sym:
        # store the samples labels for the symmetrized PET edge features
        edge_samples_sym = Labels(
            [
                "s2_pi",
                "first_atom_type",
                "second_atom_type",
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            symmetrize_edge_samples(
                get_edge_sample_values(systems, include_atom_types=True)
            ),
        )
        # store the samples labels for the last layer PET features for per_pair
        # (symmetrized) targets, i.e. the stacked node and edge samples
        ll_per_pair_samples_sym = Labels(
            edge_samples_sym.names,
            torch.vstack(
                [
                    get_node_sample_values(
                        systems,
                        sample_kind="per_pair_sym",
                        include_atom_type=True,
                    ),
                    edge_samples_sym.values,
                ]
            ),
        )

    # if there are spherical targets on an atomic basis, we need to store the sample
    # labels for each basis block, used to slice the last layer PET node/edge features
    # just before application of the output layer
    atomic_basis_samples: Dict[str, Dict[str, Labels]] = {
        "per_atom": {
            "": Labels(["_"], torch.empty((0, 1), dtype=node_samples.values.dtype))
        },
        "per_pair": {
            "": Labels(["_"], torch.empty((0, 1), dtype=node_samples.values.dtype))
        },
        "per_pair_sym": {
            "": Labels(["_"], torch.empty((0, 1), dtype=node_samples.values.dtype))
        },
    }
    if any_per_atom:
        atomic_basis_samples["per_atom"] = samples_for_atomic_basis_per_atom(
            systems, node_samples, atomic_types
        )
    if any_per_pair:
        atomic_basis_samples["per_pair"] = samples_for_atomic_basis_per_pair(
            ll_per_pair_samples, atomic_types, sample_kind="per_pair"
        )
    if any_per_pair_sym:
        atomic_basis_samples["per_pair_sym"] = samples_for_atomic_basis_per_pair(
            ll_per_pair_samples_sym, atomic_types, sample_kind="per_pair_sym"
        )

    return (
        node_samples,
        edge_samples,
        edge_samples_sym,
        ll_per_pair_samples,
        ll_per_pair_samples_sym,
        atomic_basis_samples,
    )


# ===== Samples labels for the PET node/edge features =====


def get_node_sample_values(
    systems: List[System],
    sample_kind: str,
    include_atom_type: bool,
) -> torch.Tensor:
    """
    Returns a torch tensor of the sample values that correspond to the internal PET node
    features.

    If ``sample_kind="per_atom"``, the samples values are returned with 2 dimensions
    corresponding to "system" and "atom".

    If ``sample_kind="per_pair"``, the dimensions returned correspond to "system",
    "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c". As
    these are nodes, the atom indices are equal and the cell shifts are zero.

    If ``include_atom_type=True``, the atom types are prepended dimensions, either
    corresponding to "center_type" if ``sample_kind="per_atom`` or ["first_atom_type",
    "second_atom_type"] if ``sample_kind="per_pair`` or ``sample_kind="per_pair_sym``.
    """
    assert sample_kind in ["per_atom", "per_pair", "per_pair_sym"]
    node_sample_values = torch.stack(
        [
            torch.concatenate(
                [
                    torch.full(
                        (len(system),),
                        system_i,
                        device=system.device,
                    )
                    for system_i, system in enumerate(systems)
                ],
            ),
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=system.device,
                    )
                    for system_i, system in enumerate(systems)
                ],
            ),
        ],
        dim=1,
    ).to(dtype=torch.int32)

    if sample_kind.startswith("per_pair"):
        node_sample_values = torch.hstack(
            [
                node_sample_values,
                node_sample_values[:, 1].reshape(-1, 1),  # second_atom == first_atom
                torch.zeros(node_sample_values.shape[0], 3),  # cell shifts
            ],
        ).to(dtype=torch.int32)

    if not include_atom_type:
        return node_sample_values

    # For the systems, get the atomic types
    atomic_types: Dict[int, torch.Tensor] = {}
    for system_i, system in enumerate(systems):
        atomic_types[system_i] = system.types

    first_atom_type = torch.tensor(
        [
            atomic_types[system_i.item()][atom_i]
            for system_i, atom_i in zip(
                node_sample_values[:, 0],
                node_sample_values[:, 1],
            )
        ],
        dtype=torch.int32,
    ).reshape(-1, 1)

    if sample_kind == "per_atom":
        node_sample_values = torch.hstack(
            [
                first_atom_type,
                node_sample_values,
            ]
        )
    elif sample_kind == "per_pair":
        node_sample_values = torch.hstack(
            [
                first_atom_type,
                first_atom_type,  # first_atom_type == second_atom_type
                node_sample_values,
            ]
        )
    elif sample_kind == "per_pair_sym":
        node_sample_values = torch.hstack(
            [
                # s2_pi = 0
                torch.zeros((len(first_atom_type), 1), dtype=torch.int32),
                first_atom_type,
                first_atom_type,  # first_atom_type == second_atom_type
                node_sample_values,
            ]
        )

    return node_sample_values


def get_edge_sample_values(
    systems: List[System],
    include_atom_types: bool,
) -> torch.Tensor:
    """
    Returns a torch tensor of the sample values that correspond to the internal PET edge
    features. Note that as usual for a neighbor list these do not include on-site
    samples, are not permutationally symmetrized, and are not triangularized.

    If ``include_atom_types=False``, the edge samples values with dimensions
    corresponding to ["system", "first_atom", "second_atom", "cell_shift_a",
    "cell_shift_b", "cell_shift_c"] are returned.

    If ``include_atom_types=True``, the edge samples values with the dimensions as above
    but with prepended dimensions ["first_atom_type", "second_atom_type"] are returned.
    """
    # Use each system's neighbor list to get the indices
    edge_sample_values_: List[torch.Tensor] = []
    for system_id, system in enumerate(systems):
        sample_values_edge_system = system.get_neighbor_list(
            system.known_neighbor_lists()[0]
        ).samples.values
        system_id_ = (
            torch.ones(sample_values_edge_system.shape[0], device=system.device)
            * system_id
        )
        edge_sample_values_.append(
            torch.cat(
                (system_id_.unsqueeze(1), sample_values_edge_system),
                dim=1,
            )
        )
    edge_sample_values = torch.vstack(edge_sample_values_).to(dtype=torch.int32)

    if not include_atom_types:
        return edge_sample_values

    # For the systems, get the atomic types
    atomic_types: Dict[int, torch.Tensor] = {}
    for system_i, system in enumerate(systems):
        atomic_types[system_i] = system.types

    # Insert the atom type dimensions
    edge_first_atom_type = torch.tensor(
        [
            atomic_types[system_i.item()][atom_i]
            for system_i, atom_i in zip(
                edge_sample_values[:, 0],
                edge_sample_values[:, 1],
            )
        ],
        dtype=torch.int32,
    ).reshape(-1, 1)
    edge_second_atom_type = torch.tensor(
        [
            atomic_types[system_i.item()][atom_i]
            for system_i, atom_i in zip(
                edge_sample_values[:, 0],
                edge_sample_values[:, 2],
            )
        ],
        dtype=torch.int32,
    ).reshape(-1, 1)

    return torch.hstack(
        [
            edge_first_atom_type,
            edge_second_atom_type,
            edge_sample_values,
        ]
    )


def symmetrize_edge_samples(
    edge_sample_values: torch.Tensor,
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
      the s2_pi value set to +1 and -1 respectively. Once symmetrized, and to only
      contain unique features, the features are triangularized in atom index by keeping
      only the features where the first atom index is less than the second atom index.
    """

    # 1) first_atom_type < second_atom_type: no symmetrization required. Ensure
    #    triangular in atom type.
    diff_atom_type = edge_sample_values[:, 0] < edge_sample_values[:, 1]
    edge_sample_values_diff_types = edge_sample_values[diff_atom_type]
    edge_sample_values_diff_types = torch.hstack(
        [
            # s2_pi = 0
            torch.zeros((len(edge_sample_values_diff_types), 1), dtype=torch.int32),
            edge_sample_values_diff_types,
        ]
    )

    # 2) first_atom_type == second_atom_type: symmetrization required, then enforce
    #    triangular in atom index.
    same_atom_type = edge_sample_values[:, 0] == edge_sample_values[:, 1]
    edge_sample_values_same_types = edge_sample_values[same_atom_type]

    # Create the plus and minus combinations
    edge_sample_values_same_types_plus = torch.hstack(
        [
            # s2_pi = +1
            torch.ones((len(edge_sample_values_same_types), 1), dtype=torch.int32),
            edge_sample_values_same_types,
        ]
    )
    edge_sample_values_same_types_minus = torch.hstack(
        [
            # s2_pi = -1
            torch.ones((len(edge_sample_values_same_types), 1), dtype=torch.int32) * -1,
            edge_sample_values_same_types,
        ]
    )

    # Triangularize in atom index
    triangle_mask = (
        edge_sample_values_same_types[:, 3] < edge_sample_values_same_types[:, 4]
    )
    edge_sample_values_same_types_plus = edge_sample_values_same_types_plus[
        triangle_mask
    ]
    edge_sample_values_same_types_minus = edge_sample_values_same_types_minus[
        triangle_mask
    ]

    return torch.vstack(
        [
            edge_sample_values_diff_types,
            edge_sample_values_same_types_plus,
            edge_sample_values_same_types_minus,
        ]
    )


def symmetrize_edge_features(
    edge_samples: Labels,
    edge_features: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetrizes the raw PET edge features.

    This takes in the raw PET edge features, i.e. features for off site atom pairs. The
    following procedure is applied to the different kinds of pair sample present:

    - atom pairs with different atom types: no symmetrization required. As only unique
      edge features are needed, keep only the features where the first atom type is less
      than the second atom type. These correspond to the samples with s2_pi = 0.

    - atom pairs with the same atom type: symmetrization required. For each of these
      pairs, the corresponding permuted sample is found by swapping the atom indices and
      inverting the sign of the cell shifts. A "plus" and "minus" combination of the
      features is created by adding or subtracting the feature with its permutation, and
      the s2_pi value set to +1 and -1 respectively. Once symmetrized, and to only
      contain unique features, the features are triangularized in atom index by keeping
      only the features where the first atom index is less than the second atom index.

    This is analogous to the procedure used to symmetrize the samples labels in
    :py:func:`symmetrize_edge_samples`, except the actual features are symmetrized too
    here. Only the torch tensor of the features are returned, which are
    """
    assert len(edge_samples) == edge_features.shape[0]

    sample_names: List[str] = [
        "first_atom_type",
        "second_atom_type",
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    # 1) different atom types, no symmetrization required
    diff_atom_type = edge_samples.values[:, 0] < edge_samples.values[:, 1]
    edge_features_diff_types = edge_features[diff_atom_type]

    # 2) same atom type, symmetrized required
    same_atom_type = edge_samples.values[:, 0] == edge_samples.values[:, 1]
    edge_sample_values_sym = edge_samples.values[same_atom_type]
    edge_features_sym = edge_features[same_atom_type]

    # Permute the samples values by swapping the atom indices and reversing the sign of
    # the cell shifts
    edge_samples_values_sym_perm = (
        Labels(sample_names, edge_sample_values_sym)
        .permute([0, 1, 2, 4, 3, 5, 6, 7])
        .values.clone()
    )
    edge_samples_values_sym_perm[:, -3:] *= -1

    # Find the indices that map the unpermuted to the permuted samples
    idx_sym = Labels(sample_names, edge_sample_values_sym).select(
        Labels(sample_names, edge_samples_values_sym_perm)
    )

    # Create the plus and minus combinations
    edge_features_sym_plus = edge_features_sym + edge_features_sym[idx_sym]
    edge_features_sym_minus = edge_features_sym - edge_features_sym[idx_sym]

    # Triangularize in atom index
    triangle_mask = edge_sample_values_sym[:, 3] < edge_sample_values_sym[:, 4]
    edge_features_sym_plus = edge_features_sym_plus[triangle_mask]
    edge_features_sym_minus = edge_features_sym_minus[triangle_mask]

    return torch.vstack(
        [
            edge_features_diff_types,
            edge_features_sym_plus,
            edge_features_sym_minus,
        ]
    )


# ===== Slicing PET node/edge features for an atomic basis =====


def samples_for_atomic_basis_per_atom(
    systems: List[System],
    node_samples: Labels,
    atomic_types: torch.Tensor,
) -> Dict[str, Labels]:
    """
    For spherical targets on an atomic basis, the PET node features need to be sliced
    and passed to different heads depending on atom types. This function returns the
    Labels objects for each atomic basis block.
    """
    # dict key contains info on the center type of the node feature
    return {
        f"{atomic_type}": Labels(
            node_samples.names,
            node_samples.values[
                torch.concatenate([system.types == atomic_type for system in systems])
            ],
        )
        for atomic_type in atomic_types
    }


def samples_for_atomic_basis_per_pair(
    edge_samples: Labels,
    atomic_types: torch.Tensor,
    sample_kind: str,
) -> Dict[str, Labels]:
    """
    For spherical targets on an atomic basis, the PET edge features need to be sliced
    and passed to different heads depending on atom types. This function returns the
    Labels objects for each atomic basis block, depending on the permutational symmetry
    and atomic types.
    """
    edge_sample_labels_sym_atomic_basis: Dict[str, Labels] = {}

    for first_atom_type in atomic_types:
        first_atom_type_mask = edge_samples.column("first_atom_type") == first_atom_type

        for second_atom_type in atomic_types:
            if (
                first_atom_type > second_atom_type
            ):  # edge keys are triangular in atomic type
                continue

            second_atom_type_mask = (
                edge_samples.column("second_atom_type") == second_atom_type
            )

            # dict key contains info on permutational symmetry and pair of atomic types
            if sample_kind == "per_pair_sym":
                for s2_pi in [0, 1, -1]:
                    s2_pi_mask = edge_samples.column("s2_pi") == s2_pi
                    block_mask = (
                        (first_atom_type_mask) & (second_atom_type_mask) & (s2_pi_mask)
                    )
                    edge_sample_labels_sym_atomic_basis[
                        f"{s2_pi}_{first_atom_type}_{second_atom_type}"
                    ] = Labels(
                        edge_samples.names,
                        edge_samples.values[block_mask],
                    )

            # dict key contains info only on pair of atomic types
            else:
                assert sample_kind == "per_pair", (
                    "``sample_kind`` must be either 'per_pair' or 'per_pair_sym'"
                )

                block_mask = (first_atom_type_mask) & (second_atom_type_mask)
                edge_sample_labels_sym_atomic_basis[
                    f"{first_atom_type}_{second_atom_type}"
                ] = Labels(
                    edge_samples.names,
                    edge_samples.values[block_mask],
                )

    return edge_sample_labels_sym_atomic_basis
