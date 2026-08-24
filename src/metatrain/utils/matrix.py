import torch

from metatensor.torch import TensorMap
from metatomic.torch import System

def reconstruct_full(b2m, systems, nodes_tm, edges_tm):
    node_list = b2m.forward(systems, nodes_tm)
    out = {k: {"0_0_0": node_list[k]["0_0_0"].clone()} for k in range(len(systems))}

    edge_out = b2m.forward_uncoupled_edges(systems, edges_tm)
    for A, cells in edge_out.items():
        for cell_str, H in cells.items():
            if cell_str in out[A]:
                out[A][cell_str] = out[A][cell_str] + H
            else:
                out[A][cell_str] = H
    return out

def make_orbital_map(
    systems: list[System], basis_set: dict[str, int]
) -> tuple[list[dict[str, int]], list[int]]:
    """
    Build a per-system map:
      (atom_idx, o3_lambda, radial, mu) -> global_orbital_index

    systems:      list of metatomic.torch.System
    basis_set:     dict mapping ("o3_lambda_Z") -> n_radial

    Returns:
      orbital_map: list (over systems) of dicts
      n_orbitals:  list of int, total orbitals per system
    """
    orbital_map: list[dict[str, int]] = []
    n_orbitals: list[int] = []

    for system in systems:
        omap: dict[str, int] = {}
        idx = 0

        for atom_idx, Z in enumerate(system.types):
            Z = int(Z)
            # gather all lambdas available for this Z, sorted
            lambdas: list[int] = []
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

class Blocks2Matrix(torch.nn.Module):
    """
    Transform Hamiltonian coupled or uncoupled nodes blocks into dense matrices.

    Precompute CG blocks and normalize input maps.
    :param basis_set: Basis set for the matrices, format: {l_Z: n_max}
    :param dtype: Data type for internal computations
    :param device: Device for internal computations
    """

    basis_set: dict[str, int]

    def __init__(
        self,
        basis_set: dict[str, int],
        *,
        dtype,
        device,
        precompute_cg: bool = False,
    ):
        super().__init__()

        self.basis_set: dict[str, int] = basis_set

        # Precompute CG coefficient blocks only if we'll need them (coupled path).
        # The uncoupled path never touches self._cg_cache.
        cg_cache: dict[str, torch.Tensor] = {}
        if precompute_cg:
            from featomic.torch.clebsch_gordan import calculate_cg_coefficients

            o3_lambda_max = max(int(key.split("_")[0]) for key in basis_set)
            cg = calculate_cg_coefficients(
                o3_lambda_max * 2,
                cg_backend="python-dense",
                arrays_backend="torch",
                dtype=dtype,
                device=device,
            )
            max_l = o3_lambda_max * 2
            for l1 in range(max_l + 1):
                for l2 in range(max_l + 1):
                    for lam in range(max_l + 1):
                        if len(cg.blocks({"l1": l1, "l2": l2, "lambda": lam})) == 0:
                            continue
                        try:
                            t = cg.block({"l1": l1, "l2": l2, "lambda": lam}).values
                            t = t.reshape(2 * l1 + 1, 2 * l2 + 1, 2 * lam + 1)
                            cg_cache[f"{l1}_{l2}_{lam}"] = t
                        except Exception:
                            continue

        self._cg_cache: dict[str, torch.Tensor] = cg_cache

        # Precompute basis_table (l, Z) -> max_n  (unchanged)
        max_l = max(int(k.split("_")[0]) for k in basis_set.keys())
        max_Z = max(int(k.split("_")[1]) for k in basis_set.keys())
        self._basis_table = torch.full(
            (max_l + 1, max_Z + 1), -1, dtype=torch.long, device=device
        )
        for key, val in self.basis_set.items():
            ell, Z = map(int, key.split("_"))
            self._basis_table[ell, Z] = val

        self._dtype = dtype
        self._device = device

    def _assemble_dense(
        self,
        per_system_indices: dict[int, list[list[int]]],
        per_system_values: dict[int, list[torch.Tensor]],
        unique_system_idx_py: list[int],
        n_orbs: dict[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[dict[str, torch.Tensor]]:
        """Assemble dense matrices from accumulated indices and values."""
        if len(unique_system_idx_py) == 0:
            return []

        unique_system_idx = torch.unique(
            torch.tensor(unique_system_idx_py, dtype=torch.long)
        )

        dense_list: list[dict[str, torch.Tensor]] = []
        for sys_id_t in unique_system_idx:
            sys_id = int(sys_id_t.item())

            if (
                sys_id not in per_system_indices
                or len(per_system_indices[sys_id]) == 0
            ):
                dense_list.append({
                    "0_0_0": torch.zeros(
                        (n_orbs[sys_id], n_orbs[sys_id]),
                        device=device,
                        dtype=dtype,
                    )
                })
                continue

            inds = torch.tensor(
                per_system_indices[sys_id], dtype=torch.long, device=device
            ).t()
            vals = torch.stack(per_system_values[sys_id])
            size = (n_orbs[sys_id], n_orbs[sys_id])

            H_sparse = torch.sparse_coo_tensor(
                inds, vals, size=size, device=device
            ).coalesce()
            dense_list.append({"0_0_0": H_sparse.to_dense()})

        return dense_list

    def forward_coupled(
        self,
        systems: list[System],
        tensor: TensorMap,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Reconstruct dense on-site Hamiltonian matrices from a coupled nodes TensorMap.

        Input keys:       (o3_lambda, o3_sigma, atom_type)
        Input samples:    (system, atom)
        Input components: (o3_mu,)
        Input properties: (l_1, l_2, n_1, n_2)

        :param systems: list of metatomic.torch.System objects.
        :param tensor: Coupled nodes TensorMap.
        :return: list (over systems) of dicts mapping cell shift to dense matrix.
                 On-site only, so the only key is '0_0_0'.
        """
        device = tensor[0].values.device
        dtype  = tensor[0].values.dtype

        orbital_map, n_orbs = make_orbital_map(systems, self.basis_set)
        cg_cache = self._cg_cache

        per_system_indices: dict[int, list[list[int]]] = {}
        per_system_values:  dict[int, list[torch.Tensor]] = {}
        unique_system_idx_py: list[int] = []

        for key, block in tensor.items():
            lam_out  = int(key["o3_lambda"])
            o3_sigma = int(key["o3_sigma"])
            at       = int(key["atom_type"])

            samples = block.samples.values    # (n_samples, 2): (system, atom)
            props   = block.properties.values  # (n_props, 4): (l_1, l_2, n_1, n_2)
            # block.values: (n_samples, 2*lam_out+1, n_props)

            n_samples = samples.shape[0]
            n_props   = props.shape[0]

            for s_idx in range(n_samples):
                A = int(samples[s_idx, 0].item())
                i = int(samples[s_idx, 1].item())

                Zi = int(systems[A].types[i].item())
                if Zi != at:
                    continue

                if A not in per_system_indices:
                    per_system_indices[A] = []
                    per_system_values[A]  = []
                unique_system_idx_py.append(A)

                omap = orbital_map[A]

                for ip in range(n_props):
                    l1 = int(props[ip, 0].item())
                    l2 = int(props[ip, 1].item())
                    n1 = int(props[ip, 2].item())
                    n2 = int(props[ip, 3].item())

                    cg_key = f"{l1}_{l2}_{lam_out}"
                    if cg_key not in cg_cache:
                        continue
                    # C shape: (2*l1+1, 2*l2+1, 2*lam_out+1)
                    C = cg_cache[cg_key]

                    # coupled values: (2*lam_out+1,)
                    coupled_vals = block.values[s_idx, :, ip]

                    # inverse CG: H[mu_1, mu_2] = sigma * sum_M C[mu_1, mu_2, M] * H_coupled[M]
                    # result shape: (2*l1+1, 2*l2+1)
                    uncoupled = o3_sigma * torch.einsum("mnM,M->mn", C, coupled_vals)

                    for mu1_idx, mu1 in enumerate(range(-l1, l1 + 1)):
                        lbl1 = f"{i}_{l1}_{n1}_{mu1}"
                        if lbl1 not in omap:
                            continue
                        ii = int(omap[lbl1])

                        for mu2_idx, mu2 in enumerate(range(-l2, l2 + 1)):
                            lbl2 = f"{i}_{l2}_{n2}_{mu2}"
                            if lbl2 not in omap:
                                continue
                            jj = int(omap[lbl2])

                            per_system_indices[A].append([ii, jj])
                            per_system_values[A].append(
                                uncoupled[mu1_idx, mu2_idx]
                            )

        return self._assemble_dense(
            per_system_indices,
            per_system_values,
            unique_system_idx_py,
            n_orbs,
            device,
            dtype,
        )

    def forward_uncoupled(
        self,
        systems: list[System],
        tensor: TensorMap,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Reconstruct dense on-site Hamiltonian matrices from an uncoupled nodes TensorMap.

        Input keys:       (o3_lambda_1, o3_lambda_2, o3_sigma_1, o3_sigma_2, atom_type)
        Input samples:    (system, atom)
        Input components: (o3_mu_1, o3_mu_2)
        Input properties: (n_1, n_2)

        :param systems: list of metatomic.torch.System objects.
        :param tensor: Uncoupled nodes TensorMap.
        :return: list (over systems) of dicts mapping cell shift to dense matrix.
                 On-site only, so the only key is '0_0_0'.
        """
        device = tensor[0].values.device
        dtype  = tensor[0].values.dtype

        orbital_map, n_orbs = make_orbital_map(systems, self.basis_set)

        per_system_indices: dict[int, list[list[int]]] = {}
        per_system_values:  dict[int, list[torch.Tensor]] = {}
        unique_system_idx_py: list[int] = []

        for key, block in tensor.items():
            lam_1 = int(key["o3_lambda_1"])
            lam_2 = int(key["o3_lambda_2"])
            at    = int(key["atom_type"])

            samples = block.samples.values    # (n_samples, 2): (system, atom)
            props   = block.properties.values  # (n_props, 2): (n_1, n_2)
            # block.values: (n_samples, 2*lam_1+1, 2*lam_2+1, n_props)

            n_samples = samples.shape[0]
            n_props   = props.shape[0]

            for s_idx in range(n_samples):
                A = int(samples[s_idx, 0].item())
                i = int(samples[s_idx, 1].item())

                Zi = int(systems[A].types[i].item())
                if Zi != at:
                    continue

                if A not in per_system_indices:
                    per_system_indices[A] = []
                    per_system_values[A]  = []
                unique_system_idx_py.append(A)

                omap = orbital_map[A]

                for ip in range(n_props):
                    n1 = int(props[ip, 0].item())
                    n2 = int(props[ip, 1].item())

                    for mu1_idx, mu1 in enumerate(range(-lam_1, lam_1 + 1)):
                        lbl1 = f"{i}_{lam_1}_{n1}_{mu1}"
                        if lbl1 not in omap:
                            continue
                        ii = int(omap[lbl1])

                        for mu2_idx, mu2 in enumerate(range(-lam_2, lam_2 + 1)):
                            lbl2 = f"{i}_{lam_2}_{n2}_{mu2}"
                            if lbl2 not in omap:
                                continue
                            jj = int(omap[lbl2])

                            per_system_indices[A].append([ii, jj])
                            per_system_values[A].append(
                                block.values[s_idx, mu1_idx, mu2_idx, ip]
                            )

        return self._assemble_dense(
            per_system_indices,
            per_system_values,
            unique_system_idx_py,
            n_orbs,
            device,
            dtype,
        )

    def forward(
        self,
        systems: list[System],
        tensor: TensorMap,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Reconstruct dense on-site Hamiltonian matrices from either a coupled or
        uncoupled nodes TensorMap. Dispatches to forward_coupled or
        forward_uncoupled based on the key names.

        :param systems: list of metatomic.torch.System objects.
        :param tensor: Coupled or uncoupled nodes TensorMap.
        :return: list (over systems) of dicts mapping cell shift to dense matrix.
        """
        key_names = tensor.keys.names
        if "o3_lambda" in key_names:
            return self.forward_coupled(systems, tensor)
        elif "o3_lambda_1" in key_names:
            return self.forward_uncoupled(systems, tensor)
        else:
            raise ValueError(
                f"Unrecognised key names {key_names}. Expected either "
                "'o3_lambda' (coupled) or 'o3_lambda_1' (uncoupled)."
            )
        
    def forward_uncoupled_edges(
        self,
        systems: list[System],
        tensor: TensorMap,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Reconstruct off-site (edge) blocks from an uncoupled-nodes-style edge TensorMap.

        Input keys:    (o3_lambda_1, o3_lambda_2, o3_sigma_1, o3_sigma_2,
                        first_atom_type, second_atom_type)
        Input samples: (system, first_atom, second_atom[, cell_shift_a/b/c])
        Components:    (o3_mu_1, o3_mu_2)
        Properties:    (n_1, n_2)

        Returns: {system_id: {cell_shift_str: dense matrix}} where cell_shift_str is
        "0_0_0" for on-site-shaped (sa=sb=sc=0) edges and "a_b_c" otherwise.
        Row orbital = first atom, column orbital = second atom.
        """
        device = tensor[0].values.device
        dtype  = tensor[0].values.dtype

        def _local_of(global_A: int) -> int:
            return int(global_A)

        orbital_map, n_orbs = make_orbital_map(systems, self.basis_set)

        # per (system, cell) accumulation
        indices_acc: dict[tuple, list[list[int]]] = {}
        values_acc:  dict[tuple, list[torch.Tensor]] = {}

        for key, block in tensor.items():
            lam_1 = int(key["o3_lambda_1"])
            lam_2 = int(key["o3_lambda_2"])
            Z1    = int(key["first_atom_type"])
            Z2    = int(key["second_atom_type"])

            names = list(block.samples.names)
            cols = _resolve_edge_sample_cols(names)
            samples = block.samples.values
            props   = block.properties.values  # (n_props, 2): (n_1, n_2)

            has_cell = cols["cell_a"] is not None

            for s_idx in range(samples.shape[0]):
                A = int(samples[s_idx, cols["system"]].item())
                i = int(samples[s_idx, cols["first_atom"]].item())
                j = int(samples[s_idx, cols["second_atom"]].item())

                # filter to the type pair this block describes
                if int(systems[_local_of(A)].types[i].item()) != Z1:
                    continue
                if int(systems[_local_of(A)].types[j].item()) != Z2:
                    continue

                if has_cell:
                    sa = int(samples[s_idx, cols["cell_a"]].item())
                    sb = int(samples[s_idx, cols["cell_b"]].item())
                    sc = int(samples[s_idx, cols["cell_c"]].item())
                else:
                    sa = sb = sc = 0
                cell_str = f"{sa}_{sb}_{sc}"

                omap = orbital_map[_local_of(A)]
                acc_key = (A, cell_str)
                if acc_key not in indices_acc:
                    indices_acc[acc_key] = []
                    values_acc[acc_key] = []

                for ip in range(props.shape[0]):
                    n1 = int(props[ip, 0].item())
                    n2 = int(props[ip, 1].item())

                    for mu1_idx, mu1 in enumerate(range(-lam_1, lam_1 + 1)):
                        lbl1 = f"{i}_{lam_1}_{n1}_{mu1}"
                        if lbl1 not in omap:
                            continue
                        ii = int(omap[lbl1])
                        for mu2_idx, mu2 in enumerate(range(-lam_2, lam_2 + 1)):
                            lbl2 = f"{j}_{lam_2}_{n2}_{mu2}"
                            if lbl2 not in omap:
                                continue
                            jj = int(omap[lbl2])
                            indices_acc[acc_key].append([ii, jj])
                            values_acc[acc_key].append(
                                block.values[s_idx, mu1_idx, mu2_idx, ip]
                            )

        # assemble per system
        result: dict[int, dict[str, torch.Tensor]] = {}
        for (A, cell_str), idx_list in indices_acc.items():
            if len(idx_list) == 0:
                continue
            loc = _local_of(A)
            size = (n_orbs[loc], n_orbs[loc])
            inds = torch.tensor(idx_list, dtype=torch.long, device=device).t()
            vals = torch.stack(values_acc[(A, cell_str)])
            H = torch.sparse_coo_tensor(inds, vals, size=size, device=device).coalesce().to_dense()
            result.setdefault(A, {})[cell_str] = H
        return result
    
def _resolve_edge_sample_cols(sample_names: list[str]) -> dict[str, int]:

    name_to_col = {n: i for i, n in enumerate(sample_names)}

    def pick(*candidates):
        for c in candidates:
            if c in name_to_col:
                return name_to_col[c]
        return None

    cols = {
        "system":      pick("system", "structure"),
        "first_atom":  pick("first_atom", "atom_1", "first_atom_index", "i"),
        "second_atom": pick("second_atom", "atom_2", "second_atom_index", "j"),
        "cell_a":      pick("cell_shift_a", "cell_shift_1"),
        "cell_b":      pick("cell_shift_b", "cell_shift_2"),
        "cell_c":      pick("cell_shift_c", "cell_shift_3"),
    }
    missing = [k for k in ("system", "first_atom", "second_atom") if cols[k] is None]
    if missing:
        raise ValueError(
            f"Could not locate edge sample columns {missing} in {sample_names}. "
            "Inspect block.samples.names and update _resolve_edge_sample_cols."
        )
    return cols