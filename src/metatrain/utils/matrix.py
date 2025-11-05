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
    """
    Transform Hamiltonian coupled blocks into dense matrices.


    Precompute CG blocks and normalize input maps.
    :param basis_set: Basis set for the matrices
    :param o3_lambda_max: Maximum o3_lambda in the basis set
    :param dtype: Data type for internal computations
    :param device: Device for internal computations
    """

    basis_set: Dict[str, int]
    _cached_system_types: Dict[int, torch.Tensor]

    def __init__(
        self,
        basis_set: Dict[Tuple[int, int], int],
        o3_lambda_max: int,
        *,
        dtype,
        device,
    ):
        super().__init__()

        # keep a string-keyed local version of basis_set for forward to use
        self.basis_set: Dict[str, int] = {
            f"{k[0]}_{k[1]}": v for k, v in basis_set.items()
        }

        # Build coupled_basis_set
        raw_coupled = get_coupled_basis_set(basis_set, True)
        cb: Dict[str, List[List[int]]] = {}
        for k, v in raw_coupled.items():
            # k presumably (lambda, sigma, n_centers), v is list of (Zi, Zj)
            key = f"{k[0]}_{k[1]}_{k[2]}"
            # convert pairs to lists of ints for easier TorchScript compatibility later
            cb[key] = [[int(p[0]), int(p[1])] for p in v]
        self.coupled_basis_set: Dict[str, List[List[int]]] = cb

        # Precompute CG coefficient blocks for all lambda up to 2*o3_lambda_max
        cg = calculate_cg_coefficients(
            o3_lambda_max * 2,
            cg_backend="python-dense",
            arrays_backend="torch",
            dtype=dtype,
            device=device,
        )

        # only store the values for combinations that exist in cg.blocks
        cg_cache: Dict[str, torch.Tensor] = {}
        max_l = o3_lambda_max * 2
        for l1 in range(max_l + 1):
            for l2 in range(max_l + 1):
                for lam in range(max_l + 1):
                    blocks = cg.blocks({"l1": l1, "l2": l2, "lambda": lam})
                    if len(blocks) == 0:
                        continue
                    # fetch block values and store reshaped tensor
                    key = f"{l1}_{l2}_{lam}"
                    try:
                        t = cg.block({"l1": l1, "l2": l2, "lambda": lam}).values
                        # reshape to (2*l1+1, 2*l2+1, 2*lam+1)
                        t = t.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * lam + 1)
                        cg_cache[key] = t
                    except Exception:
                        # if fetching fails, skip
                        continue

        # Store cache
        self._cg_cache: Dict[str, torch.Tensor] = cg_cache

        # Precompute basis_table (l,Z) -> max_n
        max_l = max(k[0] for k in basis_set.keys())
        max_Z = max(k[1] for k in basis_set.keys())
        self._basis_table = torch.full(
            (max_l + 1, max_Z + 1), -1, dtype=torch.long, device=device
        )
        for key, val in self.basis_set.items():
            ell, Z = map(int, key.split("_"))
            self._basis_table[ell, Z] = val

        # Cache atomic numbers per system
        self._cached_system_types: Dict[int, torch.Tensor] = {}

        # dtype/device for constructing zero matrices if needed
        self._dtype = dtype
        self._device = device

    @torch.jit.export
    def build_orbital_mask(
        self,
        systems: List[System],
        tensor: TensorMap,
    ) -> TensorMap:
        mask_blocks: List[TensorBlock] = []

        basis_table = self._basis_table

        for k, b in tensor.items():
            o3_lambda = int(k["o3_lambda"])
            o3_sigma = int(k["o3_sigma"])

            mask_values = torch.zeros_like(b.values)

            samples = b.samples.values
            props = b.properties.values
            S = samples.shape[0]
            P = props.shape[0]

            if S == 0 or P == 0:
                mask_blocks.append(
                    TensorBlock(
                        samples=b.samples,
                        components=b.components,
                        properties=b.properties,
                        values=mask_values,
                    )
                )
                continue

            # Precompute sample info
            samples_A = samples[:, 0].to(torch.long)
            samples_i1 = samples[:, 1].to(torch.long)
            samples_i2 = samples[:, 2].to(torch.long)

            Z_1 = torch.empty((S,), dtype=torch.long, device=samples.device)
            Z_2 = torch.empty((S,), dtype=torch.long, device=samples.device)

            # Fill Z_1/Z_2 per system
            sys_seen: List[int] = []
            for idx in range(S):
                Ai = int(samples_A[idx].item())
                if Ai not in sys_seen:
                    sys_seen.append(Ai)
            for Ai in sys_seen:
                pos_mask = samples_A == Ai
                idxs = torch.nonzero(pos_mask).squeeze(1)
                if idxs.numel() == 0:
                    continue
                # cache system types if not already
                if Ai not in self._cached_system_types:
                    self._cached_system_types[Ai] = systems[Ai].types.to(torch.long)
                types_tensor = self._cached_system_types[Ai]
                Z_1[idxs] = types_tensor[samples_i1[idxs]]
                Z_2[idxs] = types_tensor[samples_i2[idxs]]

            # Extract properties
            l1s = torch.tensor(
                [int(p[0].item()) for p in props],
                dtype=torch.long,
                device=samples.device,
            )
            l2s = torch.tensor(
                [int(p[1].item()) for p in props],
                dtype=torch.long,
                device=samples.device,
            )
            n1s = torch.tensor(
                [int(p[2].item()) for p in props],
                dtype=torch.long,
                device=samples.device,
            )
            n2s = torch.tensor(
                [int(p[3].item()) for p in props],
                dtype=torch.long,
                device=samples.device,
            )

            # Parity and n_centers
            parity_ok = (-1) ** (l1s + l2s + o3_lambda) == o3_sigma
            skip_prop = torch.zeros_like(parity_ok, dtype=torch.bool)
            if k["n_centers"] == 1 and o3_sigma == -1:
                skip_prop[:] = True
            valid_prop_mask = parity_ok & (~skip_prop)
            if not valid_prop_mask.any().item():
                mask_blocks.append(
                    TensorBlock(
                        samples=b.samples,
                        components=b.components,
                        properties=b.properties,
                        values=mask_values,
                    )
                )
                continue

            # Expand for vectorized checks
            l1_expand = l1s.unsqueeze(0).expand(S, P)
            l2_expand = l2s.unsqueeze(0).expand(S, P)
            n1_expand = n1s.unsqueeze(0).expand(S, P)
            n2_expand = n2s.unsqueeze(0).expand(S, P)
            Z1_expand = Z_1.unsqueeze(1).expand(S, P)
            Z2_expand = Z_2.unsqueeze(1).expand(S, P)

            # Safe indexing
            in_l1 = (l1_expand >= 0) & (l1_expand < basis_table.shape[0])
            in_l2 = (l2_expand >= 0) & (l2_expand < basis_table.shape[0])
            in_Z1 = (Z1_expand >= 0) & (Z1_expand < basis_table.shape[1])
            in_Z2 = (Z2_expand >= 0) & (Z2_expand < basis_table.shape[1])

            allowed_n1 = torch.full((S, P), -1, dtype=torch.long, device=samples.device)
            allowed_n2 = torch.full((S, P), -1, dtype=torch.long, device=samples.device)

            ok1 = in_l1 & in_Z1
            ok2 = in_l2 & in_Z2

            if ok1.any().item():
                idxs1 = torch.nonzero(ok1).t()
                allowed_n1[idxs1[0], idxs1[1]] = basis_table[
                    l1_expand[idxs1[0], idxs1[1]], Z1_expand[idxs1[0], idxs1[1]]
                ]
            if ok2.any().item():
                idxs2 = torch.nonzero(ok2).t()
                allowed_n2[idxs2[0], idxs2[1]] = basis_table[
                    l2_expand[idxs2[0], idxs2[1]], Z2_expand[idxs2[0], idxs2[1]]
                ]

            # Final mask
            valid_mask = (allowed_n1 > n1_expand) & (allowed_n2 > n2_expand)
            valid_mask = valid_mask & valid_prop_mask.unsqueeze(0).expand(S, P)

            if valid_mask.any().item():
                if mask_values.dim() == 2:
                    mask_values[valid_mask] = 1.0
                else:
                    # reshape middle dims
                    S_dim = mask_values.shape[0]
                    P_dim = mask_values.shape[-1]
                    M = 1
                    for d in mask_values.shape[1:-1]:
                        M *= d
                    mid = mask_values.view(S_dim, M, P_dim)
                    nonzero = torch.nonzero(valid_mask)
                    S_idx, P_idx = nonzero[:, 0], nonzero[:, 1]
                    mid[S_idx, :, P_idx] = 1.0
                    mask_values = mid.view(mask_values.shape)

            mask_blocks.append(
                TensorBlock(
                    samples=b.samples,
                    components=b.components,
                    properties=b.properties,
                    values=mask_values,
                )
            )

        return TensorMap(tensor.keys, mask_blocks)

    def forward(
        self,
        systems: List[System],
        tensor: TensorMap,
        mask: Optional[TensorMap] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply the block-to-matrix transformation.

        :param systems: List of metatomic.torch.System objects the TensorMap matrix
            corresponds to.
        :param tensor: The TensorMap with coupled blocks to be transformed.
        :param mask: Optional TensorMap mask indicating valid entries.
        :return: A list (over systems) of dictionaries mapping cell shifts to dense
            matrix tensors.
        """

        device = tensor[0].values.device
        dtype = tensor[0].values.dtype

        if mask is None:
            mask = self.build_orbital_mask(systems, tensor)

        assert tensor.keys.names == ["o3_lambda", "o3_sigma", "n_centers"]

        # create orbital_map and n_orbs
        orbital_map, n_orbs = make_orbital_map(systems, self.basis_set)

        per_system_indices: Dict[int, Dict[str, List[List[int]]]] = {}
        per_system_values: Dict[int, Dict[str, List[torch.Tensor]]] = {}
        unique_system_idx_py: List[int] = []

        cg_cache = self._cg_cache
        coupled_bs = self.coupled_basis_set

        # iterate blocks
        for k, b in tensor.items():
            o3_lambda = int(k["o3_lambda"])
            o3_sigma = int(k["o3_sigma"])
            n_centers = int(k["n_centers"])

            samples = b.samples.values

            mask_block = None
            if mask is not None:
                mask_block = mask.block(k).values

            props = b.properties.values
            n_props = props.shape[0]

            coupled_key = f"{o3_lambda}_{o3_sigma}_{n_centers}"
            if coupled_key not in coupled_bs:
                continue
            pair_list = coupled_bs[coupled_key]

            for ip in range(n_props):
                l1 = int(props[ip, 0].item())
                l2 = int(props[ip, 1].item())
                n1 = int(props[ip, 2].item())
                n2 = int(props[ip, 3].item())

                cg_key = f"{l1}_{l2}_{o3_lambda}"
                if cg_key not in cg_cache:
                    continue
                C = cg_cache[cg_key]

                # Multiply by sigma for decoupling
                C_sig = o3_sigma * C

                vals_ip = b.values[..., ip]

                # flatten C_sig: (m1, m2, M) -> (M, m1*m2)
                C_flat = C_sig.permute(2, 0, 1).reshape(C_sig.shape[2], -1)

                # uncouple block
                uncoupled_block = (vals_ip @ C_flat).view(
                    vals_ip.shape[0], C_sig.shape[0], C_sig.shape[1]
                )

                if mask_block is not None:
                    mb = mask_block[..., ip]
                    if mb.dim() > 1:
                        # reduce all but sample dim
                        allowed_mask = ~torch.all(mb == 0, dim=1)
                    else:
                        allowed_mask = mb != 0
                else:
                    allowed_mask = torch.ones(
                        vals_ip.shape[0], dtype=torch.bool, device=device
                    )

                allowed_idx = torch.nonzero(allowed_mask).squeeze(1)
                if allowed_idx.numel() == 0:
                    continue  # skip if no allowed samples

                # select allowed samples at once
                uncoupled_block_allowed = uncoupled_block[allowed_idx]
                samples_allowed = samples[allowed_idx]

                # now loop over allowed samples only
                for s_idx, s in enumerate(samples_allowed):
                    A = int(s[0].item())
                    i = int(s[1].item())
                    j = int(s[2].item())

                    Zi = int(systems[A].types[i].item())
                    Zj = int(systems[A].types[j].item())

                    # membership test in coupled basis
                    pair_ok = any(p[0] == Zi and p[1] == Zj for p in pair_list)
                    if not pair_ok:
                        continue

                    empty_dict: Dict[str, List[List[int]]] = {}
                    empty_values_dict: Dict[str, List[torch.Tensor]] = {}
                    if A not in per_system_indices:
                        per_system_indices[A] = empty_dict
                        per_system_values[A] = empty_values_dict
                    unique_system_idx_py.append(A)

                    cs0 = int(s[3].item())
                    cs1 = int(s[4].item())
                    cs2 = int(s[5].item())
                    cell_shift = f"{cs0}_{cs1}_{cs2}"

                    empty_list: List[List[int]] = []
                    empty_values_list: List[torch.Tensor] = []
                    if cell_shift not in per_system_indices[A]:
                        per_system_indices[A][cell_shift] = empty_list
                        per_system_values[A][cell_shift] = empty_values_list

                    omap = orbital_map[A]

                    for m1 in range(-l1, l1 + 1):
                        lbl1 = f"{i}_{l1}_{n1}_{m1}"
                        if lbl1 not in omap:
                            continue
                        ii = int(omap[lbl1])
                        for m2 in range(-l2, l2 + 1):
                            lbl2 = f"{j}_{l2}_{n2}_{m2}"
                            if lbl2 not in omap:
                                continue
                            jj = int(omap[lbl2])
                            per_system_indices[A][cell_shift].append([ii, jj])
                            per_system_values[A][cell_shift].append(
                                uncoupled_block_allowed[s_idx, m1 + l1, m2 + l2]
                            )

        # assemble dense matrices
        if len(unique_system_idx_py) == 0:
            unique_system_idx = torch.tensor([], dtype=torch.long)
        else:
            unique_system_idx = torch.unique(
                torch.tensor(unique_system_idx_py, dtype=torch.long)
            )

        dense_list: List[Dict[str, torch.Tensor]] = []
        for sys_id in unique_system_idx:
            sys_id = int(sys_id.item())
            sys_map: Dict[str, List[List[int]]] = {}
            if sys_id in per_system_indices:
                sys_map = per_system_indices[sys_id]
            if len(sys_map) == 0:
                H_dense = {
                    "0_0_0": torch.zeros(
                        (n_orbs[sys_id], n_orbs[sys_id]), device=device, dtype=dtype
                    )
                }
            else:
                H_dense: Dict[str, torch.Tensor] = {}
                for cell_shift, pairs in sys_map.items():
                    if len(pairs) == 0:
                        continue
                    inds = torch.tensor(pairs, dtype=torch.long, device=device).t()
                    vals = torch.stack(per_system_values[sys_id][cell_shift])
                    size = (n_orbs[sys_id], n_orbs[sys_id])
                    Hs = torch.sparse_coo_tensor(
                        inds, vals, size=size, device=device
                    ).coalesce()
                    Hs.requires_grad_(True)
                    H_dense[cell_shift] = Hs.to_dense()
            dense_list.append(H_dense)

        return dense_list
