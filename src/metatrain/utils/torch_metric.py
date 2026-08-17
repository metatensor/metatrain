"""Two-centre metric matrices of Gaussian auxiliary bases, in pure torch.

This module rebuilds the matrices of
:py:func:`~metatrain.utils.pyscf_loss.compute_metric_matrix` — overlap ``S`` and
Coulomb ``J`` — as batched torch operations, so a density loss can assemble them
on the training device in milliseconds instead of shipping PySCF matrices out
of the dataloader workers (seconds per system single-threaded, and hundreds of
megabytes per system over the worker pipe and PCIe).

It is generic over auxiliary bases because every standard one — named sets like
``def2-universal-jfit`` as well as even-tempered ``etb:...`` expansions — is a
contracted real solid-harmonic Gaussian basis, and two ingredients cover that
whole family:

- **Representation.** Each basis function is a homogeneous degree-``l``
  polynomial times a contraction of radial Gaussians, i.e. a linear combination
  of (Cartesian monomial × primitive Gaussian) terms. The combination
  coefficients are recovered *numerically*, per element and shell, by least
  squares against PySCF-evaluated function values on a radial×angular grid,
  and residual-checked. Nothing about PySCF's normalisation or component
  ordering is hardcoded, so a convention mismatch surfaces as a loud failure
  at construction instead of a silently wrong metric (the same approach as
  qpet's closed-form ESP evaluator).
- **Closed forms.** Two-centre integrals between monomial Gaussians follow
  McMurchie–Davidson: the overlap from Hermite expansion coefficients, the
  Coulomb matrix from Hermite derivatives of the Boys function, which is an
  incomplete gamma (``torch.special.gammainc``) and therefore batchable on any
  device. The recursions hold for arbitrary angular momentum, and the
  long-range kernel only rescales the Boys argument.

Everything geometric is evaluated in float64 — the recursions lose digits in
float32 — and the caller casts the finished matrix to the model dtype, exactly
as the PySCF path did via ``batch_to``. Gradients never flow through the
metric (it is data, built from the unaugmented geometry), so the assembly runs
under ``torch.no_grad()``.

On top of the numeric fit, every element is validated once at first use: a
random diatomic's ``S`` and ``J`` are
compared against PySCF itself, which pins down row ordering and every
convention end to end.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .pyscf_loss import (
    METRICS,
    _import_pyscf,
    _load_auxiliary_basis,
    build_auxiliary_molecule,
    compute_coulomb_matrix,
    compute_overlap_matrix,
)


#: Relative residual above which a fitted shell is rejected rather than trusted.
FIT_TOLERANCE = 1e-9

#: Relative tolerance of the per-element diatomic validation against PySCF.
VALIDATION_TOLERANCE = 1e-8

#: Overlap pairs with ``mu * R^2`` beyond this contribute below 1e-20 of their
#: prefactor and are dropped. The Coulomb kernel has no such decay and is never
#: screened.
OVERLAP_SCREEN = 46.0


def _monomials(degree: int) -> torch.Tensor:
    """Exponent triples of every monomial of exactly ``degree``.

    :param degree: The homogeneous degree.
    :return: Integer tensor of shape ``(n_monomials, 3)``.
    """
    triples = [
        (i, j, degree - i - j)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
    ]
    return torch.tensor(triples, dtype=torch.int64)


def _hermite_indices(order: int) -> torch.Tensor:
    """All Hermite index triples ``(t, u, v)`` with ``t + u + v <= order``.

    :param order: Maximum total order.
    :return: Integer tensor of shape ``(n_indices, 3)``.
    """
    triples = [
        (t, u, v)
        for t in range(order + 1)
        for u in range(order + 1 - t)
        for v in range(order + 1 - t - u)
    ]
    return torch.tensor(triples, dtype=torch.int64)


def _fibonacci_directions(count: int) -> torch.Tensor:
    """Roughly uniform unit vectors, enough to resolve the angular content.

    :param count: Number of directions.
    :return: Tensor of shape ``(count, 3)``, float64.
    """
    indices = torch.arange(count, dtype=torch.float64) + 0.5
    phi = math.pi * (3.0 - math.sqrt(5.0)) * indices
    z = 1.0 - 2.0 * indices / count
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    return torch.stack([radius * torch.cos(phi), radius * torch.sin(phi), z], dim=1)


class _PrimitiveGroup:
    """All primitives of one angular momentum for one element, stacked.

    One "primitive unit" is a single Gaussian exponent of a single
    (shell, contraction) pair, carrying its share of the fitted weights. The
    metric assembly works at this granularity so contraction happens for free
    through scatter-accumulation.

    :param alphas: Exponents, shape ``(n_units,)``, Bohr^-2.
    :param weights: Fitted weights mapping Cartesian monomials to the shell's
        spherical components, shape ``(n_units, 2l + 1, n_monomials)``.
    :param offsets: Row offset of each unit's first spherical component within
        the element's block of the AO vector, shape ``(n_units,)``.
    """

    def __init__(
        self, alphas: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor
    ) -> None:
        self.alphas = alphas
        self.weights = weights
        self.offsets = offsets


class _ElementBasis:
    """The fitted auxiliary basis of one element.

    :param groups: Primitive groups keyed by angular momentum.
    :param naux: Number of auxiliary functions the element carries.
    """

    def __init__(self, groups: Dict[int, _PrimitiveGroup], naux: int) -> None:
        self.groups = groups
        self.naux = naux


def _fit_element(aux_basis: str, atomic_number: int) -> _ElementBasis:
    """Fit one element's auxiliary functions as monomial-Gaussian combinations.

    For each (shell, contraction) pair the ``2l + 1`` spherical functions are
    expressed exactly as ``sum_{c,k} w[m, c, k] monomial_c(r) exp(-alpha_k r^2)``
    (atomic units), with ``w`` recovered by least squares from PySCF's own
    ``eval_gto`` values on a log-radial × Fibonacci-angular grid. The functions
    lie exactly in the span, so the fit is exact up to conditioning, and the
    residual is checked against :py:data:`FIT_TOLERANCE`.

    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param atomic_number: Element to fit.
    :return: The element's stacked primitive data.
    """
    import copy

    gto, elements = _import_pyscf()

    mol = gto.Mole()
    mol.atom = f"{elements.ELEMENTS[atomic_number]} 0.0 0.0 0.0"
    mol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, (atomic_number,)))
    mol.unit = "Bohr"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()

    ao_loc = mol.ao_loc_nr()
    directions = _fibonacci_directions(50).numpy()

    per_l: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for shell in range(mol.nbas):
        angular = int(mol.bas_angular(shell))
        n_sph = 2 * angular + 1
        exponents = np.asarray(mol.bas_exp(shell), dtype=np.float64)
        n_contractions = int(mol.bas_nctr(shell))

        radii = np.geomspace(
            0.03 / math.sqrt(float(exponents.max())),
            4.0 / math.sqrt(float(exponents.min())),
            48,
        )
        points = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)
        values = mol.eval_gto("GTOval_sph", points)

        powers = _monomials(angular).numpy()
        monomial_values = np.prod(points[:, None, :] ** powers[None, :, :], axis=2)
        radial_values = np.exp(-exponents[None, :] * np.sum(points**2, axis=1)[:, None])
        # (n_points, n_monomials * n_primitives), monomial-major.
        design = (monomial_values[:, :, None] * radial_values[:, None, :]).reshape(
            len(points), -1
        )

        for i_contraction in range(n_contractions):
            start = ao_loc[shell] + i_contraction * n_sph
            block = values[:, start : start + n_sph]
            solution, *_ = np.linalg.lstsq(design, block, rcond=None)
            residual = np.linalg.norm(design @ solution - block)
            scale = np.linalg.norm(block)
            if residual > FIT_TOLERANCE * max(scale, 1.0):
                raise RuntimeError(
                    f"failed to represent shell {shell} (l={angular}) of element "
                    f"{atomic_number} in basis '{aux_basis}' as monomial "
                    f"Gaussians: relative residual {residual / max(scale, 1.0):.3e}."
                )
            # (2l + 1, n_monomials, n_primitives)
            fitted = solution.reshape(len(powers), len(exponents), n_sph)
            fitted = np.ascontiguousarray(np.moveaxis(fitted, 2, 0))
            per_l.setdefault(angular, []).append(
                (
                    exponents,
                    fitted,
                    np.full(len(exponents), start, dtype=np.int64),
                )
            )

    groups: Dict[int, _PrimitiveGroup] = {}
    for angular, entries in per_l.items():
        alphas = np.concatenate([exponents for exponents, _, _ in entries])
        weights = np.concatenate(
            [np.moveaxis(fitted, 2, 0) for _, fitted, _ in entries]
        )
        offsets = np.concatenate([offset for _, _, offset in entries])
        groups[angular] = _PrimitiveGroup(
            torch.from_numpy(alphas),
            torch.from_numpy(np.ascontiguousarray(weights)),
            torch.from_numpy(offsets),
        )
    return _ElementBasis(groups, int(mol.nao))


def _boys(order: int, argument: torch.Tensor) -> torch.Tensor:
    """Boys functions ``F_0 .. F_order`` for a batch of arguments.

    Only the top order is evaluated from a special function — ``F_0`` from
    ``erf``, higher orders from the regularised lower incomplete gamma,
    ``F_n(T) = Gamma(n + 1/2) P(n + 1/2, T) / (2 T^(n + 1/2))`` — and the rest
    follow from the downward recursion ``F_{n-1} = (2 T F_n + e^{-T}) /
    (2n - 1)``, which is stable in that direction. Near ``T = 0`` the closed
    forms are 0/0 and a two-term Taylor series (error ``~T^2``) replaces them.

    :param order: Highest Boys order needed.
    :param argument: Arguments ``T``, shape ``(n,)``.
    :return: Tensor of shape ``(n, order + 1)``.
    """
    cutoff = 1e-13 if argument.dtype == torch.float64 else 3e-4
    t_safe = argument.clamp(min=cutoff)
    if order == 0:
        closed = 0.5 * torch.sqrt(torch.pi / t_safe) * torch.erf(torch.sqrt(t_safe))
    else:
        a = torch.tensor(order + 0.5, dtype=argument.dtype, device=argument.device)
        closed = (
            0.5
            * torch.exp(torch.lgamma(a))
            * torch.special.gammainc(a, t_safe)
            / t_safe**a
        )
    series = 1.0 / (2.0 * order + 1.0) - argument / (2.0 * order + 3.0)
    top = torch.where(argument < cutoff, series, closed)

    if order == 0:
        return top[:, None]
    columns = [top]
    decay = torch.exp(-argument)
    for n in range(order, 0, -1):
        columns.append((2.0 * argument * columns[-1] + decay) / (2.0 * n - 1.0))
    columns.reverse()
    return torch.stack(columns, dim=-1)


def _e0_axis(
    l1: int,
    l2: int,
    pa: torch.Tensor,
    pb: torch.Tensor,
    inv_2p: torch.Tensor,
) -> torch.Tensor:
    """Hermite ``t = 0`` expansion coefficients ``E_0^{i j}`` along one axis.

    Standard McMurchie–Davidson recursion, batched over the pair dimension. The
    ``exp(-mu X^2)`` prefactor is *not* included here; the caller folds the
    full ``exp(-mu R^2)`` once.

    :param l1: Maximum monomial power on the first centre.
    :param l2: The same for the second centre.
    :param pa: ``P - A`` along the axis, shape ``(n,)``.
    :param pb: ``P - B`` along the axis, shape ``(n,)``.
    :param inv_2p: ``1 / (2 (alpha + beta))``, shape ``(n,)``.
    :return: Tensor of shape ``(n, l1 + 1, l2 + 1)``.
    """
    table: Dict[Tuple[int, int, int], torch.Tensor] = {(0, 0, 0): torch.ones_like(pa)}

    def term(t: int, i: int, j: int) -> Optional[torch.Tensor]:
        if t < 0 or t > i + j:
            return None
        return table[(t, i, j)]

    for i in range(1, l1 + 1):
        for t in range(i + 1):
            pieces = []
            below = term(t - 1, i - 1, 0)
            if below is not None:
                pieces.append(inv_2p * below)
            same = term(t, i - 1, 0)
            if same is not None:
                pieces.append(pa * same)
            above = term(t + 1, i - 1, 0)
            if above is not None:
                pieces.append((t + 1.0) * above)
            table[(t, i, 0)] = sum(pieces[1:], pieces[0])
    for j in range(1, l2 + 1):
        for i in range(l1 + 1):
            for t in range(i + j + 1):
                pieces = []
                below = term(t - 1, i, j - 1)
                if below is not None:
                    pieces.append(inv_2p * below)
                same = term(t, i, j - 1)
                if same is not None:
                    pieces.append(pb * same)
                above = term(t + 1, i, j - 1)
                if above is not None:
                    pieces.append((t + 1.0) * above)
                table[(t, i, j)] = sum(pieces[1:], pieces[0])

    zero = torch.zeros_like(pa)
    return torch.stack(
        [
            torch.stack([table.get((0, i, j), zero) for j in range(l2 + 1)], dim=-1)
            for i in range(l1 + 1)
        ],
        dim=-2,
    )


def _d_table(angular: int, alphas: torch.Tensor) -> torch.Tensor:
    """Single-centre Hermite coefficients ``d_t^i`` for one exponent batch.

    ``x^i exp(-alpha x^2) = sum_t d_t^i Lambda_t(x)`` with Hermite Gaussians
    ``Lambda_t``; the recursion is the two-centre one at zero separation.
    Entries with ``t > i`` or ``i - t`` odd vanish and stay exactly zero.

    :param angular: Maximum monomial power.
    :param alphas: Exponents, shape ``(n,)``.
    :return: Tensor of shape ``(n, angular + 1, angular + 1)``, indexed
        ``[., i, t]``.
    """
    inv_2a = 0.5 / alphas
    zero = torch.zeros_like(alphas)
    rows: List[List[torch.Tensor]] = [[torch.ones_like(alphas)] + [zero] * angular]
    for i in range(1, angular + 1):
        previous = rows[i - 1]
        row = []
        for t in range(angular + 1):
            value = zero
            if t - 1 >= 0:
                value = value + inv_2a * previous[t - 1]
            if t + 1 <= i - 1:
                value = value + (t + 1.0) * previous[t + 1]
            row.append(value)
        rows.append(row)
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _hermite_coulomb(
    order: int,
    separation: torch.Tensor,
    theta: torch.Tensor,
    boys: torch.Tensor,
) -> torch.Tensor:
    """Hermite Coulomb integrals ``R^0_{tuv}`` up to total ``order``.

    Seeded by ``R^n_{000} = (-2 theta)^n F_n`` and raised with the standard
    downward-in-``n`` recursion. Passing pre-scaled Boys values makes the same
    recursion serve the long-range kernel.

    :param order: Maximum ``t + u + v``.
    :param separation: ``A - B``, shape ``(n, 3)``.
    :param theta: Reduced exponents, shape ``(n,)``.
    :param boys: Boys values (possibly long-range-scaled), ``(n, order + 1)``.
    :return: Tensor of shape ``(n, n_indices)``, columns ordered as
        :py:func:`_hermite_indices`.
    """
    minus_2t = -2.0 * theta
    levels: Dict[int, Dict[Tuple[int, int, int], torch.Tensor]] = {
        n: {(0, 0, 0): minus_2t**n * boys[:, n]} for n in range(order + 1)
    }
    for total in range(1, order + 1):
        for n in range(order - total + 1):
            upper = levels[n + 1]
            for t in range(total + 1):
                for u in range(total + 1 - t):
                    v = total - t - u
                    if t > 0:
                        axis, step = 0, (t - 1, u, v)
                        lower = (t - 2, u, v)
                        factor = t - 1.0
                    elif u > 0:
                        axis, step = 1, (t, u - 1, v)
                        lower = (t, u - 2, v)
                        factor = u - 1.0
                    else:
                        axis, step = 2, (t, u, v - 1)
                        lower = (t, u, v - 2)
                        factor = v - 1.0
                    value = separation[:, axis] * upper[step]
                    if factor > 0.0:
                        value = value + factor * upper[lower]
                    levels[n][(t, u, v)] = value

    columns = _hermite_indices(order).tolist()
    return torch.stack([levels[0][(t, u, v)] for t, u, v in columns], dim=-1)


def _exclusive_cumsum(values: List[int]) -> List[int]:
    """Exclusive prefix sums of a list of Python integers.

    :param values: The values.
    :return: Same length; entry ``i`` is ``sum(values[:i])``.
    """
    sums, running = [], 0
    for value in values:
        sums.append(running)
        running += value
    return sums


def _pair_lists(
    counts_1: List[int], counts_2: List[int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Within-system all-pairs indices for units concatenated across systems.

    Counts stay Python integers so the index construction never reads a device
    tensor back — a hidden synchronisation per pair group otherwise.

    :param counts_1: Units per system on the first side.
    :param counts_2: The same for the second side.
    :param device: Device to build the index tensors on.
    :return: ``(system, first, second)`` index tensors, one entry per pair.
    """
    pairs_per_system = [c1 * c2 for c1, c2 in zip(counts_1, counts_2, strict=True)]
    total = sum(pairs_per_system)
    system = torch.repeat_interleave(
        torch.arange(len(counts_1), device=device),
        torch.tensor(pairs_per_system, device=device),
    )
    rank = torch.arange(total, device=device)
    rank = (
        rank - torch.tensor(_exclusive_cumsum(pairs_per_system), device=device)[system]
    )
    starts_1 = torch.tensor(_exclusive_cumsum(counts_1), device=device)
    starts_2 = torch.tensor(_exclusive_cumsum(counts_2), device=device)
    width = torch.tensor(counts_2, device=device)[system]
    return (
        system,
        starts_1[system] + rank.div(width, rounding_mode="floor"),
        starts_2[system] + rank.remainder(width),
    )


class TorchMetricBuilder:
    """Assemble two-centre metric matrices on any torch device.

    Supports the ``overlap`` and ``coulomb`` base metrics.

    Each element's shell representation is fitted once, on first encounter, and
    validated against PySCF on a random diatomic — covering the fit, the
    integral recursions and the AO row ordering in one check.

    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    """

    def __init__(self, aux_basis: str, metric: str) -> None:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}.")
        self.aux_basis = aux_basis
        self.metric = metric
        self._elements: Dict[int, _ElementBasis] = {}
        self._device_cache: Dict[
            Tuple[int, torch.device, torch.dtype], _ElementBasis
        ] = {}
        self._tables_cache: Dict[
            Tuple[str, int, int, torch.dtype, torch.device],
            Dict[str, torch.Tensor],
        ] = {}
        self._primitive_cache: Dict[
            Tuple[torch.device, torch.dtype, int],
            Tuple[
                Dict[int, Dict[str, torch.Tensor]],
                Dict[Tuple[int, int], Tuple[int, np.ndarray]],
            ],
        ] = {}

    # ── element tables ────────────────────────────────────────────────────

    def _host_element(self, atomic_number: int) -> _ElementBasis:
        if atomic_number not in self._elements:
            fitted = _fit_element(self.aux_basis, atomic_number)
            self._elements[atomic_number] = fitted
            self._validate_element(atomic_number)
        return self._elements[atomic_number]

    def _element(
        self, atomic_number: int, device: torch.device, dtype: torch.dtype
    ) -> _ElementBasis:
        host = self._host_element(atomic_number)
        key = (atomic_number, device, dtype)
        if key not in self._device_cache:
            self._device_cache[key] = _ElementBasis(
                {
                    angular: _PrimitiveGroup(
                        group.alphas.to(device=device, dtype=dtype),
                        group.weights.to(device=device, dtype=dtype),
                        group.offsets.to(device),
                    )
                    for angular, group in host.groups.items()
                },
                host.naux,
            )
        return self._device_cache[key]

    def _validate_element(self, atomic_number: int) -> None:
        """Compare against PySCF on a random diatomic of this element.

        :param atomic_number: Element to validate.
        """
        from metatomic.torch import System

        generator = torch.Generator().manual_seed(atomic_number)
        positions = torch.zeros((2, 3), dtype=torch.float64)
        positions[1] = 1.5 + torch.rand(3, generator=generator, dtype=torch.float64)
        types = torch.full((2,), atomic_number, dtype=torch.int32)
        system = System(
            types=types,
            positions=positions,
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.zeros(3, dtype=torch.bool),
        )

        checks = [("overlap", compute_overlap_matrix(system, self.aux_basis))]
        if self.metric == "coulomb":
            checks.append(("coulomb", compute_coulomb_matrix(system, self.aux_basis)))
        for kind, reference in checks:
            ours = self._assemble(
                [(types, positions)], torch.device("cpu"), torch.float64, kind, 0.0
            )[0]
            error = torch.linalg.norm(ours - reference) / torch.linalg.norm(reference)
            if float(error) > VALIDATION_TOLERANCE:
                raise RuntimeError(
                    f"torch {kind} metric disagrees with PySCF for element "
                    f"{atomic_number} in basis '{self.aux_basis}' (relative "
                    f"error {float(error):.3e}); this indicates an AO ordering "
                    "or normalisation convention this backend does not handle."
                )

    # ── public API ────────────────────────────────────────────────────────

    def naux(self, types: torch.Tensor) -> int:
        """Auxiliary-basis size of one system.

        :param types: Atomic numbers, shape ``(n_atoms,)``.
        :return: Number of auxiliary functions.
        """
        return sum(
            self._host_element(int(atomic_number)).naux for atomic_number in types
        )

    def compute_batch(
        self,
        geometries: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> List[torch.Tensor]:
        """Metric matrices of a batch of systems, assembled on ``device``.

        :param geometries: Per system, ``(types, positions)`` with positions in
            Angstrom — the convention of the PySCF path.
        :param device: Device to assemble on.
        :param dtype: Working dtype of the assembly. ``torch.float64`` matches
            the PySCF matrices to ~1e-13; ``torch.float32`` is accurate to
            ~1e-6 — the same as casting a float64 matrix down, which is what
            training in float32 consumed anyway — and is an order of magnitude
            faster on consumer GPUs, whose float64 throughput is fractional.
        :return: One dense ``(naux, naux)`` matrix per system, in ``dtype``.
        """
        with torch.no_grad():
            matrices = self._assemble(geometries, device, dtype, self.metric, 0.0)
        return matrices

    # ── assembly ──────────────────────────────────────────────────────────

    def _units(
        self,
        geometries: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, List[int]], List[int]]:
        """Concatenate all systems' primitive units, grouped by ``l``.

        :param geometries: Per system, ``(types, positions)`` in Angstrom.
        :param device: Device to build on.
        :param dtype: Working dtype.
        :return: Per angular momentum a dict with keys ``centers`` (Bohr),
            ``alphas``, ``weights``, ``rows`` (system-local AO row of the first
            component); per angular momentum the units-per-system counts, as
            Python integers so pair-list construction never synchronises; and
            the per-system ``naux`` list.
        """
        from pyscf.lib.parameters import BOHR

        elements = sorted(
            {
                int(atomic_number)
                for types, _ in geometries
                for atomic_number in types.tolist()
            }
        )
        for atomic_number in elements:
            self._host_element(atomic_number)
        tables, slices = self._primitive_tables(device, dtype)
        angulars = sorted(
            {
                angular
                for atomic_number in elements
                for angular in self._elements[atomic_number].groups
            }
        )

        # All index arithmetic happens in numpy: device-tensor index building
        # would launch hundreds of tiny kernels and pageable host-to-device
        # copies, each a stream synchronisation. Only three index arrays per
        # angular momentum cross to the device.
        unit_atom: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        unit_prim: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        unit_row: Dict[int, List[np.ndarray]] = {a: [] for a in angulars}
        counts: Dict[int, List[int]] = {a: [] for a in angulars}
        sizes: List[int] = []
        positions_parts: List[torch.Tensor] = []
        atom_base = 0

        for types, positions in geometries:
            positions_parts.append(positions.to(device=device, dtype=dtype))
            type_array = np.asarray(types.detach().cpu(), dtype=np.int64)
            naux_per_atom = np.array([self._elements[int(z)].naux for z in type_array])
            atom_rows = np.concatenate([[0], np.cumsum(naux_per_atom)[:-1]])
            sizes.append(int(naux_per_atom.sum()))

            for angular in angulars:
                total = 0
                for z in elements:
                    entry = slices.get((z, angular))
                    if entry is None:
                        continue
                    start, local_rows = entry
                    atoms = np.nonzero(type_array == z)[0]
                    if len(atoms) == 0:
                        continue
                    n_atoms, n_prims = len(atoms), len(local_rows)
                    unit_atom[angular].append(np.repeat(atoms + atom_base, n_prims))
                    unit_prim[angular].append(
                        np.tile(np.arange(start, start + n_prims), n_atoms)
                    )
                    unit_row[angular].append(
                        np.repeat(atom_rows[atoms], n_prims)
                        + np.tile(local_rows, n_atoms)
                    )
                    total += n_atoms * n_prims
                counts[angular].append(total)
            atom_base += len(type_array)

        positions_bohr = torch.cat(positions_parts) / BOHR
        units: Dict[int, Dict[str, torch.Tensor]] = {}
        for angular in angulars:
            indices = torch.from_numpy(
                np.stack(
                    [
                        np.concatenate(unit_atom[angular]),
                        np.concatenate(unit_prim[angular]),
                        np.concatenate(unit_row[angular]),
                    ]
                )
            ).to(device)
            units[angular] = {
                "centers": positions_bohr[indices[0]],
                "alphas": tables[angular]["alphas"][indices[1]],
                "weights": tables[angular]["weights"][indices[1]],
                "rows": indices[2],
            }
        return units, counts, sizes

    def _primitive_tables(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[
        Dict[int, Dict[str, torch.Tensor]],
        Dict[Tuple[int, int], Tuple[int, np.ndarray]],
    ]:
        """Per-angular-momentum primitive tables of every fitted element.

        Concatenates the fitted elements' exponents and weights per ``l`` on
        the device, once, so per-batch unit construction is a single gather.
        Rebuilt when a new element is fitted.

        :param device: Device the tables live on.
        :param dtype: Working dtype.
        :return: ``(tables, slices)``: per ``l`` the stacked ``alphas`` and
            ``weights``; per ``(element, l)`` the element's start in that stack
            and its primitives' AO row offsets (host side).
        """
        key = (device, dtype, len(self._elements))
        cached = self._primitive_cache.get(key)
        if cached is not None:
            return cached

        tables: Dict[int, Dict[str, torch.Tensor]] = {}
        slices: Dict[Tuple[int, int], Tuple[int, np.ndarray]] = {}
        per_l: Dict[int, Dict[str, list]] = {}
        for atomic_number in sorted(self._elements):
            for angular, group in self._elements[atomic_number].groups.items():
                entry = per_l.setdefault(angular, {"alphas": [], "weights": []})
                start = sum(len(a) for a in entry["alphas"])
                slices[(atomic_number, angular)] = (
                    start,
                    np.asarray(group.offsets),
                )
                entry["alphas"].append(group.alphas)
                entry["weights"].append(group.weights)
        for angular, entry in per_l.items():
            tables[angular] = {
                "alphas": torch.cat(entry["alphas"]).to(device=device, dtype=dtype),
                "weights": torch.cat(entry["weights"]).to(device=device, dtype=dtype),
            }
        self._primitive_cache[key] = (tables, slices)
        return tables, slices

    def _assemble(
        self,
        geometries: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        base: str,
        omega: float,
    ) -> List[torch.Tensor]:
        """One base metric for every system of the batch.

        All systems' primitive pairs of a given ``(l1, l2)`` are processed in
        one batched sweep; results scatter-accumulate into a flat buffer that
        concatenates the per-system matrices, so the number of kernel launches
        is independent of both batch size and system size. Both metrics are
        symmetric, so only ``l1 <= l2`` groups are computed and off-diagonal
        blocks are scattered twice, once transposed.

        :param geometries: Per system, ``(types, positions)`` in Angstrom.
        :param device: Device to assemble on.
        :param dtype: Working dtype.
        :param base: ``"overlap"`` or ``"coulomb"``.
        :param omega: Range-separation parameter; ``0`` is the plain kernel.
        :return: One matrix per system, in ``dtype``.
        """
        units, counts, sizes = self._units(geometries, device, dtype)
        bases = _exclusive_cumsum([size * size for size in sizes])
        sizes_tensor = torch.tensor(sizes, device=device)
        bases_tensor = torch.tensor(bases, device=device)
        buffer = torch.zeros(
            sum(size * size for size in sizes), dtype=dtype, device=device
        )

        budget = 1 << 24 if dtype == torch.float32 else 1 << 23
        for l1, group_1 in units.items():
            for l2, group_2 in units.items():
                if l2 < l1:
                    continue
                system, first, second = _pair_lists(counts[l1], counts[l2], device)
                tables = self._group_tables(base, l1, l2, dtype, device)
                n_hermite = (
                    len(_hermite_indices(l1)) * len(_hermite_indices(l2))
                    if base == "coulomb"
                    else (l1 + 1) * (l2 + 1)
                )
                chunk = max(4096, budget // max(1, n_hermite))
                for start in range(0, len(system), chunk):
                    self._pair_block(
                        base,
                        omega,
                        l1,
                        l2,
                        group_1,
                        group_2,
                        system[start : start + chunk],
                        first[start : start + chunk],
                        second[start : start + chunk],
                        sizes_tensor,
                        bases_tensor,
                        buffer,
                        tables,
                    )

        matrices = []
        for start, size in zip(bases, sizes, strict=True):
            matrices.append(buffer[start : start + size * size].view(size, size))
        return matrices

    def _group_tables(
        self,
        base: str,
        l1: int,
        l2: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Constant index tensors of one ``(l1, l2)`` pair group, cached.

        Hoisted out of the chunk loop: rebuilding them per chunk costs a
        host-to-device transfer and a handful of kernel launches each time.

        :param base: ``"overlap"`` or ``"coulomb"``.
        :param l1: Angular momentum of the first side.
        :param l2: The same for the second side.
        :param dtype: Working dtype (the parity signs are float).
        :param device: Device the tensors live on.
        :return: The tables, keyed by name.
        """
        key = (base, l1, l2, dtype, device)
        cached = self._tables_cache.get(key)
        if cached is not None:
            return cached

        tables = {
            "powers_1": _monomials(l1).to(device),
            "powers_2": _monomials(l2).to(device),
        }
        if base == "coulomb":
            order = l1 + l2
            indices = _hermite_indices(order).to(device)
            index_of = torch.full(
                (order + 1,) * 3, -1, dtype=torch.int64, device=device
            )
            index_of[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.arange(
                len(indices), device=device
            )
            hermite_1 = _hermite_indices(l1).to(device)
            hermite_2 = _hermite_indices(l2).to(device)
            tables["hermite_1"] = hermite_1
            tables["hermite_2"] = hermite_2
            tables["combined"] = index_of[
                (hermite_1[:, None, :] + hermite_2[None, :, :]).unbind(-1)
            ]
            tables["parity"] = (-1.0) ** hermite_2.sum(-1).to(dtype)
        self._tables_cache[key] = tables
        return tables

    def _pair_block(
        self,
        base: str,
        omega: float,
        l1: int,
        l2: int,
        group_1: Dict[str, torch.Tensor],
        group_2: Dict[str, torch.Tensor],
        system: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        sizes: torch.Tensor,
        bases: torch.Tensor,
        buffer: torch.Tensor,
        tables: Dict[str, torch.Tensor],
    ) -> None:
        """Compute one chunk of primitive pairs and scatter it into the buffer.

        :param base: ``"overlap"`` or ``"coulomb"``.
        :param omega: Range-separation parameter; ``0`` is the plain kernel.
        :param l1: Angular momentum of the first side.
        :param l2: The same for the second side.
        :param group_1: First side's unit tensors.
        :param group_2: The same for the second side.
        :param system: System index of each pair.
        :param first: Unit index of each pair on the first side.
        :param second: The same for the second side.
        :param sizes: Per-system matrix dimension.
        :param bases: Per-system offset into ``buffer``.
        :param buffer: Flat accumulation buffer of all matrices.
        :param tables: The group's constant index tensors
            (:py:meth:`_group_tables`).
        """
        centers_1 = group_1["centers"][first]
        centers_2 = group_2["centers"][second]
        alphas_1 = group_1["alphas"][first]
        alphas_2 = group_2["alphas"][second]
        separation = centers_1 - centers_2
        distance_sq = (separation**2).sum(-1)

        if base == "overlap":
            keep = (
                alphas_1 * alphas_2 / (alphas_1 + alphas_2) * distance_sq
                < OVERLAP_SCREEN
            )
            if not bool(keep.all()):
                (indices,) = torch.where(keep)
                system, first, second = system[indices], first[indices], second[indices]
                centers_1, centers_2 = centers_1[indices], centers_2[indices]
                alphas_1, alphas_2 = alphas_1[indices], alphas_2[indices]
                separation, distance_sq = separation[indices], distance_sq[indices]
        if len(system) == 0:
            return

        weights_1 = group_1["weights"][first]
        weights_2 = group_2["weights"][second]

        if base == "overlap":
            total = alphas_1 + alphas_2
            reduced = alphas_1 * alphas_2 / total
            centre = (
                alphas_1[:, None] * centers_1 + alphas_2[:, None] * centers_2
            ) / total[:, None]
            inv_2p = 0.5 / total
            e_axes = [
                _e0_axis(
                    l1,
                    l2,
                    (centre - centers_1)[:, axis],
                    (centre - centers_2)[:, axis],
                    inv_2p,
                )
                for axis in range(3)
            ]
            powers_1 = tables["powers_1"]
            powers_2 = tables["powers_2"]
            cartesian = (
                e_axes[0][:, powers_1[:, 0]][:, :, powers_2[:, 0]]
                * e_axes[1][:, powers_1[:, 1]][:, :, powers_2[:, 1]]
                * e_axes[2][:, powers_1[:, 2]][:, :, powers_2[:, 2]]
            )
            prefactor = torch.exp(-reduced * distance_sq) * (torch.pi / total) ** 1.5
            block = (
                torch.einsum("pmc,pcd,pnd->pmn", weights_1, cartesian, weights_2)
                * prefactor[:, None, None]
            )
        else:
            theta = alphas_1 * alphas_2 / (alphas_1 + alphas_2)
            order = l1 + l2
            argument = theta * distance_sq
            if omega > 0.0:
                sigma_sq = omega**2 / (omega**2 + theta)
                exponents = torch.arange(
                    order + 1, dtype=theta.dtype, device=system.device
                )
                boys = _boys(order, sigma_sq * argument) * sigma_sq[:, None] ** (
                    exponents + 0.5
                )
            else:
                boys = _boys(order, argument)
            prefactor = (
                2.0
                * torch.pi**2.5
                / (alphas_1 * alphas_2 * torch.sqrt(alphas_1 + alphas_2))
            )
            if order == 0:
                # s-s pairs: the Hermite machinery collapses to F_0, and this
                # group is by far the largest, so the einsum path would spend
                # its time on 1x1 matrix products.
                block = (
                    weights_1[:, 0, 0] * weights_2[:, 0, 0] * prefactor * boys[:, 0]
                )[:, None, None]
                self._scatter(
                    l1,
                    l2,
                    group_1,
                    group_2,
                    system,
                    first,
                    second,
                    sizes,
                    bases,
                    buffer,
                    block,
                )
                return
            hermite = _hermite_coulomb(order, separation, theta, boys)
            gathered = hermite[:, tables["combined"]]

            side_1 = self._hermite_side(
                l1, alphas_1, weights_1, tables["powers_1"], tables["hermite_1"]
            )
            side_2 = self._hermite_side(
                l2, alphas_2, weights_2, tables["powers_2"], tables["hermite_2"]
            )
            block = (
                torch.einsum(
                    "pmh,phk,pnk->pmn", side_1, gathered, side_2 * tables["parity"]
                )
                * prefactor[:, None, None]
            )

        self._scatter(
            l1,
            l2,
            group_1,
            group_2,
            system,
            first,
            second,
            sizes,
            bases,
            buffer,
            block,
        )

    @staticmethod
    def _scatter(
        l1: int,
        l2: int,
        group_1: Dict[str, torch.Tensor],
        group_2: Dict[str, torch.Tensor],
        system: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        sizes: torch.Tensor,
        bases: torch.Tensor,
        buffer: torch.Tensor,
        block: torch.Tensor,
    ) -> None:
        """Accumulate pair blocks into the flat buffer of matrices.

        :param l1: Angular momentum of the first side.
        :param l2: The same for the second side.
        :param group_1: First side's unit tensors.
        :param group_2: The same for the second side.
        :param system: System index of each pair.
        :param first: Unit index of each pair on the first side.
        :param second: The same for the second side.
        :param sizes: Per-system matrix dimension.
        :param bases: Per-system offset into ``buffer``.
        :param buffer: Flat accumulation buffer of all matrices.
        :param block: Pair blocks, shape ``(n, 2 l1 + 1, 2 l2 + 1)``.
        """
        rows_1 = group_1["rows"][first]
        rows_2 = group_2["rows"][second]
        components_1 = torch.arange(2 * l1 + 1, device=system.device)
        components_2 = torch.arange(2 * l2 + 1, device=system.device)
        stride = sizes[system]
        flat = (
            (bases[system] + rows_1 * stride + rows_2)[:, None, None]
            + components_1[None, :, None] * stride[:, None, None]
            + components_2[None, None, :]
        )
        buffer.scatter_add_(0, flat.reshape(-1), block.reshape(-1))
        if l1 != l2:
            # The metric is symmetric and only l1 <= l2 groups are enumerated;
            # place the transposed block on the other side of the diagonal.
            flat_t = (
                (bases[system] + rows_2 * stride + rows_1)[:, None, None]
                + components_2[None, :, None] * stride[:, None, None]
                + components_1[None, None, :]
            )
            buffer.scatter_add_(
                0, flat_t.reshape(-1), block.transpose(1, 2).reshape(-1)
            )

    @staticmethod
    def _hermite_side(
        angular: int,
        alphas: torch.Tensor,
        weights: torch.Tensor,
        powers: torch.Tensor,
        hermite: torch.Tensor,
    ) -> torch.Tensor:
        """One side's spherical-to-Hermite transformation.

        :param angular: The side's angular momentum.
        :param alphas: Exponents, shape ``(n,)``.
        :param weights: Fitted weights, shape ``(n, 2l + 1, n_monomials)``.
        :param powers: Monomial exponents, shape ``(n_monomials, 3)``.
        :param hermite: Hermite indices, shape ``(n_hermite, 3)``.
        :return: Tensor of shape ``(n, 2l + 1, n_hermite)``.
        """
        table = _d_table(angular, alphas)
        transform = (
            table[:, powers[:, 0][:, None], hermite[:, 0][None, :]]
            * table[:, powers[:, 1][:, None], hermite[:, 1][None, :]]
            * table[:, powers[:, 2][:, None], hermite[:, 2][None, :]]
        )
        return torch.einsum("pmc,pch->pmh", weights, transform)


__all__ = ["TorchMetricBuilder"]


# Re-exported for tests that validate against the PySCF reference path.
_pyscf_reference = build_auxiliary_molecule
