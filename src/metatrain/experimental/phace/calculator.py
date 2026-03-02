"""JAX-based ASE calculator for the PhACE model.

Uses the Equinox port in ``eqx.py`` with JAX automatic differentiation
for force computation.
"""

import math
from typing import Optional

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .eqx import EqxPhACE, _get_adaptive_cutoffs, load_from_checkpoint

# Optimization-level presets: name → geometric bucket ratio.
# Higher optimization = finer padding grid = more JIT compilations = better
# throughput per call.  Lower optimization = coarser grid = fewer compilations
# = less memory used for cached compiled functions.
#
#   high   (1.1): ≤10 % padding waste, O(log_{1.1} n) compilations.
#                 Best for homogeneous datasets.
#   medium (1.5): ≤50 % padding waste, O(log_{1.5} n) compilations.  [default]
#   low    (2.0): ≤100 % padding waste, O(log_2 n) compilations.
#                 Best for heterogeneous datasets (fewest recompilations, least
#                 memory pressure from cached XLA executables).
_OPTIMIZATION_PRESETS: dict[str, float] = {
    "high": 1.1,
    "medium": 1.5,
    "low": 2.0,
}

# Optional: nvalchemi GPU neighbor list (NVIDIA's fast CUDA NL).
# Used when available and JAX is running on GPU.
try:
    import torch as _torch
    from nvalchemiops.neighborlist import neighbor_list as _nvalchemi_nl

    _HAS_NVALCHEMI = _torch.cuda.is_available() and jax.default_backend() == "gpu"
except ImportError:
    _HAS_NVALCHEMI = False


def _next_bucket(n: int, ratio: float = 1.5) -> int:
    """Return the smallest integer >= n of the form ceil(ratio**k), k >= 0.

    Using ratio=1.5 limits JIT recompilations to O(log_{1.5}(n_max)) distinct
    shapes while keeping padding overhead at most 50 % above the true count.
    """
    if n <= 1:
        return max(1, n)
    k = math.ceil(math.log(n) / math.log(ratio))
    bucket = int(math.ceil(ratio**k))
    if bucket < n:  # guard against floating-point rounding
        bucket = int(math.ceil(ratio ** (k + 1)))
    return bucket


def _make_prefilter_fn(model: EqxPhACE, n_atoms_pad: int):
    """Return a JIT-compiled function that computes adaptive cutoffs and keep mask.

    Computes pair distances internally from positions and cell shifts, so that
    no CPU-side distance array is needed.  Runs entirely on the GPU.

    Returns ``keep [n_pairs_full_pad]``: boolean mask, True for pairs that
    survive the adaptive cutoff filter (padded pairs always False).

    ``n_atoms_pad`` is baked in as a static constant for ``segment_sum``.
    """
    r_cut = model.r_cut
    cutoff_width = model.cutoff_width
    num_neighbors_adaptive = model.num_neighbors_adaptive

    @jax.jit
    def _fn(
        positions,
        cells,
        cell_shifts_full,
        center_indices_full,
        neighbor_indices_full,
        structure_pairs_full,
        pair_mask_full,
    ):
        cart_vecs = (
            positions[neighbor_indices_full]
            - positions[center_indices_full]
            + jnp.einsum(
                "ab,abc->ac",
                cell_shifts_full.astype(positions.dtype),
                cells[structure_pairs_full],
            )
        )
        r_sq = jnp.sum(cart_vecs**2, axis=-1)
        r = jnp.sqrt(jnp.where(r_sq > 0, r_sq, 1.0))

        if num_neighbors_adaptive is not None:
            atomic_cutoffs = _get_adaptive_cutoffs(
                center_indices_full,
                r,
                num_neighbors_adaptive,
                n_atoms_pad,
                r_cut,
                cutoff_width,
                pair_mask=pair_mask_full,
            )
            pair_cutoffs = (
                atomic_cutoffs[center_indices_full]
                + atomic_cutoffs[neighbor_indices_full]
            ) / 2.0
            keep = (r <= pair_cutoffs) & pair_mask_full
        else:
            keep = pair_mask_full

        return keep

    return _fn


def _make_energy_forces_stress_fn(model: EqxPhACE, n_atoms: int):
    """Return a JIT-compiled function that returns (energy, forces, stress).

    Stress is computed by differentiating E w.r.t. a 3×3 strain tensor ε,
    where a strain ε deforms positions and cell homogeneously:
        positions → positions @ (I + ε)
        cells     → cells     @ (I + ε)

    Both gradients (forces from ∂E/∂pos and stress from ∂E/∂ε) are produced
    by a single backward pass via ``jax.value_and_grad(..., argnums=(0, 1))``.

    ``n_atoms`` is the *padded* atom count, baked in as a static constant so
    JAX can infer array shapes for scatter operations.
    """

    @jax.jit
    def _fn(
        positions,
        cells,
        cell_shifts_full,
        center_indices_full,
        neighbor_indices_full,
        structure_pairs_full,
        pair_mask_full,
        cell_shifts_filt,
        center_indices_filt,
        neighbor_indices_filt,
        structure_pairs_filt,
        pair_mask_filt,
        structure_centers,
        species,
        atom_mask,
    ):
        def _e(pos, strain):
            F = jnp.eye(3, dtype=pos.dtype) + strain
            return model.energy(
                pos @ F,
                cells @ F,
                cell_shifts_full,
                center_indices_full,
                neighbor_indices_full,
                structure_pairs_full,
                cell_shifts_filt,
                center_indices_filt,
                neighbor_indices_filt,
                structure_pairs_filt,
                species,
                structure_centers,
                n_atoms,
                pair_mask_full=pair_mask_full,
                pair_mask_filt=pair_mask_filt,
                atom_mask=atom_mask,
            )

        strain0 = jnp.zeros((3, 3), dtype=positions.dtype)
        e, (grad_pos, grad_strain) = jax.value_and_grad(_e, argnums=(0, 1))(
            positions, strain0
        )
        forces = -grad_pos
        volume = jnp.abs(jnp.linalg.det(cells[0]))
        # Symmetrise to remove any numerical antisymmetric noise
        stress = (grad_strain + grad_strain.T) / (2.0 * volume)
        return e, forces, stress

    return _fn


def _torch_to_jax(t: "_torch.Tensor") -> jnp.ndarray:
    """torch CUDA tensor → JAX GPU array.

    ``edge_index[0/1]`` are rows of a [2, n_pairs] tensor.  When n_pairs is
    not divisible by 4, ``edge_index[1]``'s start address is 8-byte aligned
    but not 16-byte aligned (JAX's minimum requirement).  ``.clone()`` creates
    a fresh, 512-byte-aligned CUDA allocation, so ``jax.dlpack.from_dlpack``
    never sees a misaligned buffer.
    """
    return jax.dlpack.from_dlpack(t.clone().contiguous())


class PhACEJAXCalculator(Calculator):
    """ASE calculator using the JAX/Equinox PhACE model.

    Example usage::

        calc = PhACEJAXCalculator("model.ckpt")
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

        # For heterogeneous datasets, lower the optimization level to reduce
        # the number of JIT recompilations and avoid OOM:
        calc = PhACEJAXCalculator("model.ckpt", optimization_level="low")
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        checkpoint_path: str,
        *,
        optimization_level: "str | float" = "medium",
        restart=None,
        label=None,
        atoms=None,
        **kwargs,
    ):
        """
        :param checkpoint_path: Path to the metatrain ``model.ckpt`` file.
        :param optimization_level: Controls the geometric bucket ratio used for
            array padding.  Accepts a preset name or a custom float ratio.

            * ``"high"`` (ratio 1.1) — fine padding grid, most JIT
              compilations, best throughput per call.  Suited to homogeneous
              datasets.
            * ``"medium"`` (ratio 1.5, **default**) — good middle ground.
            * ``"low"`` (ratio 2.0) — coarse padding grid, fewest
              compilations, least memory from cached XLA executables.  Best
              choice for heterogeneous datasets to avoid OOM.
            * A custom ``float > 1`` may be passed for fine-grained control.
        """
        super().__init__(restart=restart, label=label, atoms=atoms, **kwargs)
        if isinstance(optimization_level, str):
            if optimization_level not in _OPTIMIZATION_PRESETS:
                raise ValueError(
                    f"Unknown optimization_level {optimization_level!r}. "
                    f"Choose one of {list(_OPTIMIZATION_PRESETS)} or pass a float ratio."
                )
            self._bucket_ratio: float = _OPTIMIZATION_PRESETS[optimization_level]
        else:
            if float(optimization_level) <= 1.0:
                raise ValueError("optimization_level ratio must be > 1.0")
            self._bucket_ratio = float(optimization_level)
        self.model: EqxPhACE = load_from_checkpoint(checkpoint_path)
        self._cached_n_atoms: Optional[int] = None
        self._prefilter_fn = None
        self._energy_and_forces_fn = None

    def _get_fns(self, n_atoms_pad: int):
        """Return (possibly cached) JIT-compiled prefilter + energy functions."""
        if self._cached_n_atoms != n_atoms_pad:
            self._prefilter_fn = _make_prefilter_fn(self.model, n_atoms_pad)
            self._energy_and_forces_fn = _make_energy_forces_stress_fn(
                self.model, n_atoms_pad
            )
            self._cached_n_atoms = n_atoms_pad
        return self._prefilter_fn, self._energy_and_forces_fn

    def calculate(
        self,
        atoms: Optional[ase.Atoms] = None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms

        positions_np = atoms.positions.astype(np.float32)
        species_np = np.array(atoms.numbers, dtype=np.int32)
        cell_np = atoms.cell.array.astype(np.float32)
        n_atoms = len(atoms)
        n_atoms_pad = _next_bucket(n_atoms, self._bucket_ratio)
        pad_a = n_atoms_pad - n_atoms
        r_cut = self.model.r_cut

        prefilter_fn, energy_fn = self._get_fns(n_atoms_pad)

        # ------------------------------------------------------------------
        # 1.  Upload small atom arrays to GPU (positions, cell, species)
        # ------------------------------------------------------------------
        positions_jax = jnp.array(positions_np)  # [n_atoms, 3]
        cell_jax = jnp.array(cell_np)  # [3, 3]
        positions_pad_jax = jnp.pad(positions_jax, ((0, pad_a), (0, 0)))
        cells_jax = cell_jax[None, :, :]  # [1, 3, 3]
        species_pad_jax = jnp.array(
            np.pad(species_np, (0, pad_a), constant_values=int(species_np[0]))
        )
        sc_jax = jnp.zeros(n_atoms_pad, dtype=jnp.int32)
        atom_mask_jax = jnp.concatenate(
            [
                jnp.ones(n_atoms, dtype=bool),
                jnp.zeros(pad_a, dtype=bool),
            ]
        )

        # ------------------------------------------------------------------
        # 2.  Build full neighbor list on GPU
        # ------------------------------------------------------------------
        if _HAS_NVALCHEMI:
            # Share positions/cell zero-copy with nvalchemi via DLPack
            positions_torch = _torch.from_dlpack(positions_jax)
            cell_torch = _torch.from_dlpack(cell_jax)
            pbc_torch = _torch.tensor(atoms.pbc, device="cuda")

            result = _nvalchemi_nl(
                positions_torch,
                float(r_cut),
                cell=cell_torch,
                pbc=pbc_torch,
                return_neighbor_list=True,
            )
            edge_index = result[0]  # [2, n_pairs] int32 CUDA
            S = (
                result[2]
                if len(result) == 3
                else _torch.zeros(
                    edge_index.shape[1], 3, dtype=_torch.int32, device="cuda"
                )
            )

            # torch CUDA → JAX GPU (GPU-to-GPU copy for alignment)
            i_full_jax = _torch_to_jax(edge_index[0])
            j_full_jax = _torch_to_jax(edge_index[1])
            S_full_jax = _torch_to_jax(S).astype(jnp.float32)
            n_pairs_full = i_full_jax.shape[0]
        else:
            from vesin import NeighborList

            _nl = NeighborList(cutoff=float(r_cut), full_list=True)
            i_np, j_np, S_np, _ = _nl.compute(
                points=atoms.positions,
                box=atoms.cell.array,
                periodic=atoms.pbc,
                quantities="ijSd",
            )
            i_full_jax = jnp.array(i_np.astype(np.int32))
            j_full_jax = jnp.array(j_np.astype(np.int32))
            S_full_jax = jnp.array(S_np.astype(np.float32))
            n_pairs_full = len(i_np)

        # ------------------------------------------------------------------
        # 3.  Pad full NL on GPU
        # ------------------------------------------------------------------
        n_pairs_full_pad = _next_bucket(n_pairs_full, self._bucket_ratio)
        pad_full = n_pairs_full_pad - n_pairs_full

        i_full_pad_jax = jnp.pad(i_full_jax, (0, pad_full))
        j_full_pad_jax = jnp.pad(j_full_jax, (0, pad_full))
        S_full_pad_jax = jnp.pad(S_full_jax, ((0, pad_full), (0, 0)))
        sp_full_pad_jax = jnp.zeros(n_pairs_full_pad, dtype=jnp.int32)
        pair_mask_full_jax = jnp.concatenate(
            [
                jnp.ones(n_pairs_full, dtype=bool),
                jnp.zeros(pad_full, dtype=bool),
            ]
        )

        # ------------------------------------------------------------------
        # 4.  Adaptive cutoffs + keep mask on GPU
        # ------------------------------------------------------------------
        keep_jax = prefilter_fn(
            positions_pad_jax,
            cells_jax,
            S_full_pad_jax,
            i_full_pad_jax,
            j_full_pad_jax,
            sp_full_pad_jax,
            pair_mask_full_jax,
        )

        # ------------------------------------------------------------------
        # 5.  Stream compaction on GPU
        #
        # Transfer one scalar to CPU to determine the bucket size, then
        # jnp.nonzero packs the surviving pair indices — all on GPU.
        # ------------------------------------------------------------------
        n_pairs_filt = int(keep_jax.sum())
        n_pairs_filt_pad = _next_bucket(n_pairs_filt, self._bucket_ratio)

        filt_idx = jnp.nonzero(keep_jax, size=n_pairs_filt_pad, fill_value=0)[0]
        pair_mask_filt_jax = jnp.arange(n_pairs_filt_pad) < n_pairs_filt

        i_filt_jax = i_full_pad_jax[filt_idx]
        j_filt_jax = j_full_pad_jax[filt_idx]
        S_filt_jax = S_full_pad_jax[filt_idx]
        sp_filt_jax = jnp.zeros(n_pairs_filt_pad, dtype=jnp.int32)

        # ------------------------------------------------------------------
        # 6.  Energy + forces + stress
        # ------------------------------------------------------------------
        energy_jax, forces_jax, stress_jax = energy_fn(
            positions_pad_jax,
            cells_jax,
            S_full_pad_jax,
            i_full_pad_jax,
            j_full_pad_jax,
            sp_full_pad_jax,
            pair_mask_full_jax,
            S_filt_jax,
            i_filt_jax,
            j_filt_jax,
            sp_filt_jax,
            pair_mask_filt_jax,
            sc_jax,
            species_pad_jax,
            atom_mask_jax,
        )

        s = np.array(stress_jax)  # [3, 3]
        self.results["energy"] = float(energy_jax)
        self.results["forces"] = np.array(forces_jax)[:n_atoms]
        # Voigt notation: [xx, yy, zz, yz, xz, xy] (eV/Å³)
        self.results["stress"] = np.array(
            [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
        )
