import math
from typing import Dict, List, Literal

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
        squared_residuals = residuals**2
        if squared_residuals.ndim > 2:
            # squared residuals need to be summed over component dimensions,
            # i.e., all but the first and last dimensions
            squared_residuals = torch.sum(
                squared_residuals,
                dim=tuple(range(1, squared_residuals.ndim - 1)),
            )

        if self.method == "absolute_residuals":
            ratios = torch.sqrt(squared_residuals) / uncertainties
        else:
            ratios = squared_residuals / uncertainties**2

        ratios_sum64 = torch.sum(ratios.to(torch.float64), dim=0)
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
        squared_residuals = residuals**2
        if squared_residuals.ndim > 2:
            # squared residuals need to be summed over component dimensions,
            # i.e., all but the first and last dimensions
            squared_residuals = torch.sum(
                squared_residuals,
                dim=tuple(range(1, squared_residuals.ndim - 1)),
            )
        abs_residuals = torch.sqrt(squared_residuals)

        # Accumulate as (N, M) per batch, preserving last dim as the "channel/property"
        # axis.
        if uncertainty_name not in self._store:
            self._store[uncertainty_name] = {"residuals": [], "uncertainties": []}
        _accumulate_local_crps_inputs(
            abs_residuals, uncertainties, self._store[uncertainty_name], eps=self.eps
        )

    def finalize(self) -> Dict[str, torch.Tensor]:
        multipliers: Dict[str, torch.Tensor] = {}
        for uncertainty_name, st in self._store.items():
            local_residuals = torch.cat(st["residuals"], dim=0)  # (Ntot, M)
            local_uncertainties = torch.cat(st["uncertainties"], dim=0)  # (Ntot, M)
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

    The CRPS calibration is performed per last-dimension channel (M). Residuals can
    be vector/tensor valued with component dimensions between the sample axis (N)
    and the channel axis (M). In that case, residuals are reduced to a scalar per
    sample and channel using an L2 norm over component dimensions:
        r̃_{i,m} = ||r_{i,*,m}||_2.

    Uncertainties are clamped from below by ``eps`` to avoid division by zero.

    :param residuals: Residuals between predicted mean and targets.
    :param uncertainties: Non-calibrated predictive standard deviations.
    :param storage: Dict with keys ``'residuals'`` and ``'uncertainties'`` storing lists
        of (N, M) tensors to be concatenated later.
    :param eps: Small positive constant used for numerical stability.
    :return: None
    """
    storage["residuals"].append(residuals)
    storage["uncertainties"].append(uncertainties.clamp_min(eps))


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
    :return: Vector of optimal alpha values with shape (M,).
    """
    from scipy.optimize import root_scalar

    if local_residuals.ndim != 2 or local_uncertainties.ndim != 2:
        raise ValueError(
            "CRPS solver expects (N, M) residuals and uncertainties tensors."
        )

    _, M = local_residuals.shape
    out = torch.empty((M,), dtype=torch.float64, device=local_residuals.device)

    for m in range(M):
        res_ch = local_residuals[:, m]
        unc_ch = local_uncertainties[:, m]

        def f(
            a: float, res_ch: torch.Tensor = res_ch, unc_ch: torch.Tensor = unc_ch
        ) -> float:
            return _crps_derivative_channel(a, res_ch, unc_ch)

        a_lo, a_hi = 1e-10, 50.0
        f_lo, f_hi = f(a_lo), f(a_hi)

        # Brent requires a sign change; expand deterministically if needed.
        if f_lo * f_hi > 0.0:
            a_hi2 = a_hi
            for _ in range(12):
                a_hi2 *= 10.0
                f_hi2 = f(a_hi2)
                if f_lo * f_hi2 <= 0.0:
                    a_hi, f_hi = a_hi2, f_hi2
                    break
            else:
                a_lo2 = a_lo
                for _ in range(12):
                    a_lo2 /= 10.0
                    f_lo2 = f(a_lo2)
                    if f_lo2 * f_hi <= 0.0:
                        a_lo, f_lo = a_lo2, f_lo2
                        break

        sol = root_scalar(f, bracket=[a_lo, a_hi], method="brentq")
        out[m] = float(sol.root)

    return out
