import math
from typing import Callable, Dict, List, Literal

import torch


CalibrationMethod = Literal["squared_residuals", "absolute_residuals", "crps"]


class RatioCalibrator:
    """
    Accumulates sufficient statistics for:
      - 'squared_residuals' (Gaussian NLL-style; RMSE-like scaling)
      - 'absolute_residuals' (MAE-like scaling with Gaussian correction sqrt(pi/2))

    This mirrors the existing implementation:
      - accumulate sum over samples of ratios (dim=0)
      - accumulate sample count
      - all-reduce once in finalize() if distributed
      - compute alpha per uncertainty head

    :param method: Calibration method, either 'squared_residuals' or
        'absolute_residuals'.
    """

    def __init__(
        self, method: Literal["squared_residuals", "absolute_residuals"]
    ) -> None:
        self.method = method
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, torch.Tensor] = {}

    def update(
        self,
        *,
        uncertainty_name: str,
        residuals: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> None:
        residuals64 = residuals.to(torch.float64)
        uncertainties64 = uncertainties.to(torch.float64)

        if self.method == "absolute_residuals":
            ratios = torch.abs(residuals64) / uncertainties64
        else:
            ratios = residuals64**2 / uncertainties64**2

        ratios_sum64 = torch.sum(ratios, dim=0)
        count = torch.tensor(ratios.shape[0], dtype=torch.long, device=ratios.device)

        if uncertainty_name not in self.sums:
            self.sums[uncertainty_name] = ratios_sum64
            self.counts[uncertainty_name] = count
        else:
            self.sums[uncertainty_name] += ratios_sum64
            self.counts[uncertainty_name] += count

    def finalize(self) -> Dict[str, torch.Tensor]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for k in self.sums:
                torch.distributed.all_reduce(
                    self.sums[k], op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    self.counts[k], op=torch.distributed.ReduceOp.SUM
                )

        multipliers: Dict[str, torch.Tensor] = {}
        for k in self.sums:
            denom = self.counts[k].to(torch.float64)

            if self.method == "absolute_residuals":
                # MAE-style -> convert to Gaussian sigma units
                alpha = (self.sums[k] / denom) * math.sqrt(math.pi / 2.0)
            else:
                # RMSE-style
                alpha = torch.sqrt(self.sums[k] / denom)

            multipliers[k] = alpha

        return multipliers


class GaussianCRPSCalibrator:
    """
    Gaussian CRPS calibration of a global multiplicative factor alpha per channel (last
    axis).

    Stores per-uncertainty head:
      - local residuals reduced to shape (N, M)
      - local uncertainties reduced/broadcast to shape (N, M)

    Finalize solves alpha per channel M using a distributed-safe root finder.

    :param eps: Small positive constant used for numerical stability.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self._store: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    def update(
        self,
        *,
        uncertainty_name: str,
        residuals: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> None:
        # Accumulate as (N, M) per batch, preserving last dim as the property axis
        if uncertainty_name not in self._store:
            self._store[uncertainty_name] = {"residuals": [], "uncertainties": []}
        _accumulate_local_crps_inputs(
            residuals, uncertainties, self._store[uncertainty_name], eps=self.eps
        )

    def finalize(self) -> Dict[str, torch.Tensor]:
        multipliers: Dict[str, torch.Tensor] = {}
        for uncertainty_name, st in self._store.items():
            local_residuals = torch.cat(st["residuals"], dim=0)  # (Ntot, C, M)
            local_uncertainties = torch.cat(st["uncertainties"], dim=0)  # (Ntot, C, M)
            alpha = _solve_alpha_crps(local_residuals, local_uncertainties)
            multipliers[uncertainty_name] = alpha
        return multipliers


def _accumulate_local_crps_inputs(
    residuals: torch.Tensor,
    uncertainties: torch.Tensor,
    storage: Dict[str, List[torch.Tensor]],
    eps: float = 1e-12,
) -> None:
    """
    Prepare and store per-sample inputs for Gaussian CRPS calibration.

    The CRPS calibration is performed per last-dimension channel (M).
    Uncertainties are clamped from below by ``eps`` to avoid division by zero.

    :param residuals: Residuals between predicted mean and targets.
    :param uncertainties: Non-calibrated predictive standard deviations.
    :param storage: Dict with keys ``'residuals'`` and ``'uncertainties'`` storing lists
        of (N, M) tensors to be concatenated later.
    :param eps: Small positive constant used for numerical stability.
    :return: None
    """
    res = residuals.detach().to(torch.float64)
    unc = uncertainties.detach().to(torch.float64).clamp_min(eps)

    # Ensure last axis exists (M)
    if res.ndim == 1:
        res = res[:, None]
    if unc.ndim == 1:
        unc = unc[:, None]

    storage["residuals"].append(res)
    storage["uncertainties"].append(unc)


def _crps_derivative_channel(
    alpha: float, res_ch: torch.Tensor, unc_ch: torch.Tensor
) -> float:
    """
    Evaluate the CRPS optimality equation (derivative w.r.t. alpha) for one channel.

    We assume a Gaussian predictive distribution with calibrated scale
        s_i(alpha) = alpha * sigma_i,
    where sigma_i is the uncalibrated per-sample standard deviation. Define the
    normalized residual
        u_i = r_i / (alpha * sigma_i),
    with r_i = mu_i - y_i.

    Using the analytical expression of the Gaussian CRPS, the first-order condition
    for minimizing sum_i CRPS(mu_i, alpha*sigma_i; y_i) can be written as
        lhs(alpha) = sum_i sigma_i [ F(u_i) - u_i (1 - 2 Phi(u_i)) ] = 0,
    where
        F(u) = 1/sqrt(pi) - 2 phi(u) - u (2 Phi(u) - 1),
    and phi, Phi are the standard normal PDF and CDF.

    In distributed mode, this function computes the local lhs(alpha) and then
    performs an all-reduce SUM to obtain the global lhs(alpha).

    :param alpha: Positive scale multiplier for uncertainties.
    :param res_ch: Residuals for one channel, shape (N,).
    :param unc_ch: Uncertainties for one channel, shape (N,).
    :return: The global value of lhs(alpha) (zero at the optimum).
    """
    alpha = max(float(alpha), 1e-20)
    u = res_ch / (alpha * unc_ch)

    phi = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * u * u)
    Phi = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))

    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    F_u = inv_sqrt_pi - 2.0 * phi - u * (2.0 * Phi - 1.0)

    lhs_local = torch.sum(unc_ch * (F_u - u * (1.0 - 2.0 * Phi)))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(lhs_local, op=torch.distributed.ReduceOp.SUM)

    return float(lhs_local.item())


def _bracket_root(
    f: Callable[[float], float],
    a_lo: float,
    a_hi: float,
    *,
    ftol: float,
    max_nudge: int,
    max_expand: int,
) -> tuple[float, float]:
    # Get reasonable initial bracket for root-finding with Brent's method.
    f_lo, f_hi = f(a_lo), f(a_hi)

    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        raise RuntimeError(
            "CRPS bracketing failed: non-finite derivative values. "
            f"a_lo={a_lo}, f_lo={f_lo}, a_hi={a_hi}, f_hi={f_hi}"
        )

    if abs(f_lo) <= ftol:
        a = a_lo
        for _ in range(max_nudge):
            a *= 10.0
            f_a = f(a)
            if not math.isfinite(f_a):
                raise RuntimeError(
                    "CRPS bracketing failed: non-finite derivative during nudging. "
                    f"a={a}, f={f_a}"
                )
            if abs(f_a) > ftol:
                a_lo, f_lo = a, f_a
                break
        else:
            raise RuntimeError(
                "CRPS bracketing failed: derivative ~0 at lower bound after nudging. "
                f"Initial a_lo={a_lo}, final a_lo={a}, f_lo={f_lo}"
            )

    if f_lo * f_hi <= 0.0:
        return a_lo, a_hi

    a = a_hi
    for _ in range(max_expand):
        a *= 10.0
        f_a = f(a)
        if not math.isfinite(f_a):
            raise RuntimeError(
                "CRPS bracketing failed: non-finite derivative while expanding high. "
                f"a={a}, f={f_a}"
            )
        if f_lo * f_a <= 0.0:
            return a_lo, a

    a = a_lo
    for _ in range(max_expand):
        a /= 10.0
        f_a = f(a)
        if not math.isfinite(f_a):
            raise RuntimeError(
                "CRPS bracketing failed: non-finite derivative while shrinking low. "
                f"a={a}, f={f_a}"
            )
        if f_a * f_hi <= 0.0:
            return a, a_hi

    raise RuntimeError(
        "CRPS bracketing failed: could not find sign change. "
        f"a_lo={a_lo}, f_lo={f_lo}, a_hi={a_hi}, f_hi={f_hi}"
    )


def _solve_alpha_crps(
    local_residuals: torch.Tensor, local_uncertainties: torch.Tensor
) -> torch.Tensor:
    """
    Solve for the CRPS-optimal alpha per channel.

    This solves lhs_m(alpha_m) = 0 independently for each channel m, where lhs_m
    is computed by ``_crps_derivative_channel`` using a distributed all-reduce.

    Distributed requirement: since lhs(alpha) uses collectives, all ranks must call
    the objective in the same sequence. Therefore, this solver must run on all ranks.

    :param local_residuals: Local residuals reduced to shape (N, M).
    :param local_uncertainties: Local uncertainties reduced to shape (N, M).
    :return: Vector of optimal alpha values with shape (M).
    """
    from scipy.optimize import root_scalar

    _, M = local_residuals.shape
    out = torch.empty((M,), dtype=torch.float64, device=local_residuals.device)

    a_lo0, a_hi0 = 1e-10, 50.0
    ftol = 1e-12
    max_nudge = 8
    max_expand = 12

    for m in range(M):
        res_ch = local_residuals[:, m]
        unc_ch = local_uncertainties[:, m]

        def f(
            a: float, res_ch: torch.Tensor = res_ch, unc_ch: torch.Tensor = unc_ch
        ) -> float:
            return _crps_derivative_channel(a, res_ch, unc_ch)

        a_lo, a_hi = _bracket_root(
            f,
            a_lo0,
            a_hi0,
            ftol=ftol,
            max_nudge=max_nudge,
            max_expand=max_expand,
        )

        sol = root_scalar(f, bracket=[a_lo, a_hi], method="brentq")
        out[m] = float(sol.root)

    return out
