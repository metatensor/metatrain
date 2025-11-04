from typing import Dict, List, Optional, Tuple

import torch
from featomic.torch.clebsch_gordan import calculate_cg_coefficients
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


@torch.jit.script
def transpose_tensormap(tensor: TensorMap) -> TensorMap:
    """
    Transposes the input TensorMap by swapping the sample (atom indices, cell shifts)
    and property (angular basis indices, radial basis indices) axes in each block,
    assuming they are Hamiltonian-like.

    :param tensor: The input TensorMap to transpose.
    :return: The transposed TensorMap.
    """

    blocks_T: List[TensorBlock] = []
    for key, block in tensor.items():
        vals_T = block.values.clone()

        # Only permute samples if two-center
        if key["n_centers"] == 2:
            samples_vals = block.samples.permute((0, 2, 1, 3, 4, 5)).values
            samples_vals[:, 3:6] *= -1
            samples_perm = Labels(block.samples.names, samples_vals)
            sample_idxs = samples_perm.select(block.samples)

        else:
            sample_idxs = torch.arange(len(block.samples))

        vals_T = vals_T[sample_idxs]

        # Permute properties
        properties_vals = block.properties.permute((1, 0, 3, 2)).values
        properties_perm = Labels(block.properties.names, properties_vals)
        property_idxs = properties_perm.select(block.properties)
        vals_T = vals_T[:, :, property_idxs]
        blocks_T.append(
            TensorBlock(
                samples=block.samples,
                components=block.components,
                properties=block.properties,
                values=vals_T,
            )
        )
    return TensorMap(tensor.keys, blocks_T)


@torch.jit.script
def build_orbital_mask(
    systems: List[System],
    basis_set: Dict[str, int],
    tensor: TensorMap,
) -> TensorMap:
    """
    Builds a boolean mask TensorMap from the input ``tensor``, where the mask is implied
    from the basis set definition. The returned mask has the same metadata structure as
    the input tensor, but with float values as either 1.0 or 0.0 indicating whether each
    entry is valid (not NaN) or not.

    :param systems: List of metatomic.torch.System objects the matrix ``tensor``
        corresponds to.
    :param tensor: The TensorMap of the matrix quantity (either predicted or target).
    :param basis_set: The basis set definition used to determine valid entries.
    :return: A TensorMap containing the boolean mask.
    """
    mask_blocks: List[TensorBlock] = []
    for k, b in tensor.items():
        o3_lambda = int(k["o3_lambda"])
        o3_sigma = int(k["o3_sigma"])

        # Initialize as zeros
        mask_values = torch.zeros_like(b.values)

        # loop over samples
        for i_s, s in enumerate(b.samples.values):
            A, i_1, i_2 = int(s[0]), int(s[1]), int(s[2])

            # get atomic numbers
            Z_1 = systems[A].types[i_1].item()
            Z_2 = systems[A].types[i_2].item()

            # loop over properties
            for i_p, p in enumerate(b.properties.values):
                l_1, l_2, n_1, n_2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])

                # skip if either (l, Z) pair is not in basis set
                if f"{l_1}_{Z_1}" not in basis_set or f"{l_2}_{Z_2}" not in basis_set:
                    continue

                if o3_sigma != (-1) ** (l_1 + l_2 + o3_lambda):
                    continue

                if k["n_centers"] == 1 and o3_sigma == -1:
                    continue

                # get allowed radial quantum numbers
                allowed_n_1 = list(range(basis_set[f"{l_1}_{Z_1}"]))
                allowed_n_2 = list(range(basis_set[f"{l_2}_{Z_2}"]))

                # skip if either n_1 or n_2 is not allowed
                if n_1 not in allowed_n_1 or n_2 not in allowed_n_2:
                    continue

                # mark this (sample, property) pair as valid
                mask_values[i_s, ..., i_p] = 1.0

        mask_blocks.append(
            TensorBlock(
                samples=b.samples,
                components=b.components,
                properties=b.properties,
                values=mask_values,
            )
        )

    return TensorMap(tensor.keys, mask_blocks)


@torch.jit.script
def make_orbital_map(
    systems: List[System], basis_set: Dict[str, int]
) -> Tuple[List[Dict[str, int]], List[int]]:
    """
    Build a per-system map:
      (atom_idx, o3_lambda, radial, mu) -> global_orbital_index

    systems:      list of metatomic.torch.System
    basis_set:     dict mapping ("o3_lambda_Z") -> n_radial

    Returns:
      orbital_map: list (over systems) of dicts
      n_orbitals:  list of int, total orbitals per system
    """
    orbital_map: List[Dict[str, int]] = []
    n_orbitals: List[int] = []

    for system in systems:
        omap: Dict[str, int] = {}
        idx = 0

        for atom_idx, Z in enumerate(system.types):
            Z = int(Z)
            # gather all lambdas available for this Z, sorted
            lambdas: List[int] = []
            for o3_lambda_Z in basis_set:
                o3_lambda, Z0 = o3_lambda_Z.split("_")
                o3_lambda, Z0 = int(o3_lambda), int(Z0)
                if Z0 == Z:
                    lambdas.append(o3_lambda)
            lambdas = sorted(lambdas)
            for o3_lambda in lambdas:
                n_radial = basis_set[f"{o3_lambda}_{Z}"]
                # radial index runs 0 .. n_radial-1
                for radial in range(n_radial):
                    # mu runs from -o3_lambda .. +o3_lambda
                    for mu in range(-o3_lambda, o3_lambda + 1):
                        omap[f"{atom_idx}_{o3_lambda}_{radial}_{mu}"] = idx
                        idx += 1

        assert len(omap) > 0

        orbital_map.append(omap)
        n_orbitals.append(idx)

    return orbital_map, n_orbitals


def get_coupled_basis_set(
    basis_set: dict, is_hermitian: bool
) -> Dict[Tuple[int], List[List[int]]]:
    """
    Takes an atomic basis set and returns the per-pair coupled version of it.
    """
    coupled_basis_set: Dict[Tuple[int], List[List[int]]] = {}
    for o3_lambda_1, first_atom_type in basis_set.keys():
        for o3_lambda_2, second_atom_type in basis_set.keys():
            lams = range(abs(o3_lambda_1 - o3_lambda_2), o3_lambda_1 + o3_lambda_2 + 1)

            for o3_lambda in lams:
                o3_sigma = (-1) ** (o3_lambda_1 + o3_lambda_2 + o3_lambda)

                for n_centers in [1, 2]:
                    if (
                        n_centers == 1 and first_atom_type != second_atom_type
                    ):  # cannot be on-site
                        continue

                    if is_hermitian:
                        if n_centers == 1 and o3_sigma == -1:
                            continue

                    key = (o3_lambda, o3_sigma, n_centers)
                    pair = [first_atom_type, second_atom_type]
                    if key in coupled_basis_set:
                        if pair not in coupled_basis_set[key]:
                            coupled_basis_set[key].append(pair)
                    else:
                        coupled_basis_set[key] = [pair]

    return coupled_basis_set


class Blocks2Matrix(torch.nn.Module):
    def __init__(
        self,
        basis_set: Dict[Tuple[int, int], int],
        o3_lambda_max: int,
        *,
        dtype,
        device,
    ):
        super().__init__()

        self.basis_set = {f"{k[0]}_{k[1]}": v for k, v in basis_set.items()}
        self.coupled_basis_set = {
            f"{k[0]}_{k[1]}_{k[2]}": v
            for k, v in get_coupled_basis_set(basis_set, True).items()
        }
        self.cg_coeffs = calculate_cg_coefficients(
            o3_lambda_max * 2,
            cg_backend="python-dense",
            arrays_backend="torch",
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        systems: List[System],
        tensor: TensorMap,
        mask: Optional[TensorMap] = None,
    ) -> TensorMap:
        """
        Takes the coupled block representation of a per-pair target property on an
        atom-centered basis and uncouples the blocks.
        """

        device = tensor[0].values.device
        dtype = tensor[0].values.dtype

        if mask is None:
            mask = build_orbital_mask(systems, self.basis_set, tensor)

        # Check key names
        assert tensor.keys.names == ["o3_lambda", "o3_sigma", "n_centers"]

        orbital_map, n_orbs = make_orbital_map(systems, self.basis_set)

        per_system_indices = {}
        per_system_values = {}
        unique_system_idx = []

        is_to_skip = False

        for k, b in tensor.items():
            o3_lambda = int(k["o3_lambda"])
            o3_sigma = int(k["o3_sigma"])
            n_centers = int(k["n_centers"])
            samples = b.samples.values.tolist()

            mask_block = None
            if mask is not None:
                mask_block = mask.block(k).values

            for ip, (l1, l2, n1, n2) in enumerate(b.properties.values.tolist()):
                if (
                    len(
                        self.cg_coeffs.blocks({"l1": l1, "l2": l2, "lambda": o3_lambda})
                    )
                    == 0
                ):
                    continue
                C = (
                    o3_sigma
                    * self.cg_coeffs.block(
                        {"l1": l1, "l2": l2, "lambda": o3_lambda}
                    ).values
                )
                # TODO: fix for the case of sparse cg coeffs
                C = C.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * o3_lambda + 1)

                uncoupled_block = torch.einsum(
                    "mnM,SM->Smn",
                    C,
                    b.values[..., ip],
                )

                for isample, s in enumerate(samples):
                    if mask_block is not None:
                        is_to_skip = torch.allclose(
                            mask_block[isample, ..., ip],
                            torch.zeros_like(mask_block[isample, ..., ip]),
                        )
                        if is_to_skip:
                            continue

                    A, i, j = s[:3]
                    A, i, j = int(A), int(i), int(j)
                    Zi = int(systems[A].types[i])
                    Zj = int(systems[A].types[j])

                    # skip if this pair of atoms does not have this coupled basis
                    if [Zi, Zj] not in self.coupled_basis_set[
                        f"{o3_lambda}_{o3_sigma}_{n_centers}"
                    ]:
                        continue

                    if A not in per_system_indices:
                        per_system_indices[A] = {}
                        per_system_values[A] = {}

                    unique_system_idx.append(A)
                    cell_shift = tuple(x for x in s[3:6])

                    if cell_shift not in per_system_indices[A]:
                        per_system_indices[A][cell_shift] = []
                        per_system_values[A][cell_shift] = []

                    for im1, m1 in enumerate(range(-l1, l1 + 1)):
                        for im2, m2 in enumerate(range(-l2, l2 + 1)):
                            idx_1 = f"{i}_{l1}_{n1}_{m1}"
                            idx_2 = f"{j}_{l2}_{n2}_{m2}"

                            idx1_does_not_exist = idx_1 not in orbital_map[A]
                            idx2_does_not_exist = idx_2 not in orbital_map[A]
                            if idx1_does_not_exist or idx2_does_not_exist:
                                continue

                            idx_i = orbital_map[A][idx_1]
                            idx_j = orbital_map[A][idx_2]

                            per_system_indices[A][cell_shift].append((idx_i, idx_j))
                            per_system_values[A][cell_shift].append(
                                uncoupled_block[isample, im1, im2]
                            )
        unique_system_idx = torch.unique(torch.tensor(unique_system_idx)).tolist()

        dense_list = []
        for sys_id in unique_system_idx:
            if len(per_system_indices[sys_id]) == 0:
                # zero matrix if no data
                # TODO: check
                H_dense = {
                    (0, 0, 0): torch.zeros(
                        (n_orbs[sys_id], n_orbs[sys_id]), device=device, dtype=dtype
                    )
                }
            else:
                indices = {
                    cell_shift: torch.tensor(
                        per_system_indices[sys_id][cell_shift],
                        dtype=torch.long,
                        device=device,
                    ).T
                    for cell_shift in per_system_indices[sys_id]
                }
                values = {
                    cell_shift: torch.stack(per_system_values[sys_id][cell_shift])
                    for cell_shift in per_system_indices[sys_id]
                }

                H_sparse = {
                    cell_shift: torch.sparse_coo_tensor(
                        indices[cell_shift],
                        values[cell_shift],
                        size=(n_orbs[sys_id], n_orbs[sys_id]),
                        device=device,
                        requires_grad=True,
                    ).coalesce()
                    for cell_shift in indices
                }

                H_dense = {
                    cell_shift: H.to_dense() for cell_shift, H in H_sparse.items()
                }

            dense_list.append(H_dense)

        return dense_list
