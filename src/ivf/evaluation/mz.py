"""Mincer-Zarnowitz regressions (ADR 0001 4.8, amended M5).

Regress the outcome on the forecast, ``y = alpha + beta * yhat + e``, by OLS. A forecast that is the
conditional mean has ``alpha = 0`` and ``beta = 1``; the joint Wald statistic
``W = (theta - theta0)' V^-1 (theta - theta0)``, ``theta0 = (0, 1)``, is chi-squared with 2 degrees of
freedom under that null. ``beta < 1`` means the forecasts move too much (big forecasts are too big).

The covariance ``V`` must allow for overlapping errors: an h-step error is MA(h-1), so the scores
``g_t = x_t e_t`` are autocorrelated up to lag h-1. ``V = Q^-1 S Q^-1 / n`` with ``Q = X'X/n`` and
``S = Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')``, ``Gamma_j = (1/n) sum_{t>j} g_t g_{t-j}'``.

==================  =============================================================
``cov_method``      weights
==================  =============================================================
``hansen_hodrick``  ``w_j = 1`` for ``j <= h-1`` (rectangular). Not guaranteed positive
                    definite: if it is not, Bartlett at the same lag, stored as
                    ``bartlett_fallback``
``newey_west``      Bartlett ``w_j = 1 - j/h`` for ``j <= h-1`` (the ADR's M0 rule)
``andrews``         Bartlett with Andrews' (1991) AR(1) plug-in bandwidth, computed on
                    the slope score ``g_t[1]``
==================  =============================================================

The primary method (``PRIMARY_METHOD``) settles ADR 0001 O5 by the size simulation of milestone 5
(``simulate_mz``; docs/milestones/05-evaluation.md 7.2). These are the same weights as
``statsmodels``' HAC with ``kernel="uniform"`` / ``"bartlett"`` and ``use_correction=False``
(tested to agree).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from ivf.evaluation import dm

COV_METHODS = ("hansen_hodrick", "newey_west", "andrews")
#: Amended M5, O5 (default taken 2026-10-29, owner can override): the smallest worst-case size error
#: of the simulation at the dev sample size; see the walkthrough, section 7.2.
PRIMARY_METHOD = "hansen_hodrick"


def _window(method: str, h: int, scores: np.ndarray) -> tuple[str, float, int]:
    if method == "hansen_hodrick":
        return "rectangular", float(h - 1), h - 1
    if method == "newey_west":
        return "bartlett", float(h), h - 1
    if method == "andrews":
        s = dm.andrews_bandwidth(scores[:, -1])
        return "bartlett", s, max(int(math.ceil(s)) - 1, 0)
    raise ValueError(f"unknown method {method!r}")


def long_run_cov(scores: np.ndarray, kernel: str, bandwidth: float, max_lag: int) -> np.ndarray:
    """``Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')`` with divisor n (scores are not re-centred)."""
    g = np.asarray(scores, dtype=float)
    n = g.shape[0]
    s = g.T @ g / n
    w = dm.weights(kernel, bandwidth, max_lag)
    for j in range(1, min(max_lag, n - 1) + 1):
        gamma = g[j:].T @ g[:-j] / n
        s = s + w[j - 1] * (gamma + gamma.T)
    return s


@dataclass(frozen=True)
class MzResult:
    n: int
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r2: float
    wald_stat: float
    wald_p: float
    cov_method: str
    bandwidth: float
    max_lag: int


def mz_regression(y, yhat, h: int, method: str = PRIMARY_METHOD) -> MzResult:
    """OLS of ``y`` on ``(1, yhat)`` with an HAC covariance, and the Wald test of (0, 1)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(yhat, dtype=float)
    if y.shape != x.shape or y.ndim != 1 or y.size < 3 or not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise ValueError("y and yhat must be finite 1-d arrays of the same length (>= 3)")
    n = y.size
    design = np.column_stack([np.ones(n), x])
    xtx = design.T @ design
    theta = np.linalg.solve(xtx, design.T @ y)
    resid = y - design @ theta
    scores = design * resid[:, None]
    kernel, bandwidth, max_lag = _window(method, h, scores)
    name = method
    q_inv = np.linalg.inv(xtx / n)

    def cov(k, b):
        return q_inv @ long_run_cov(scores, k, b, max_lag) @ q_inv / n

    v = cov(kernel, bandwidth)
    if method == "hansen_hodrick" and np.linalg.eigvalsh((v + v.T) / 2).min() <= 0:
        kernel, bandwidth, name = "bartlett", float(h), "bartlett_fallback"
        v = cov(kernel, bandwidth)
    diff = theta - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.solve(v, diff))
        p = float(stats.chi2.sf(wald, 2))
    except np.linalg.LinAlgError:
        wald, p = math.nan, math.nan
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / tss if tss > 0 else math.nan
    return MzResult(n, float(theta[0]), float(theta[1]), math.sqrt(max(v[0, 0], 0.0)), math.sqrt(max(v[1, 1], 0.0)),
                    r2, wald, p, name, bandwidth, max_lag)


def simulate_mz(n: int, h: int, reps: int, rng: np.random.Generator, methods=COV_METHODS, phi: float = 0.98,
                mean: float = 20.0, shock_sd: float = 1.2, alpha: float = 0.05) -> dict[str, float]:
    """Size of the joint Wald test when the forecast IS the conditional mean (O5).

    ``y`` is an AR(1) in levels (``phi``, ``mean``, shock sd ``shock_sd``: a VIX-like level, stationary
    sd about 6); the forecast at origin t is ``mean + phi^h (y_t - mean)``, so ``alpha = 0`` and
    ``beta = 1`` hold exactly and the h-step errors overlap (MA(h-1)). Returns each method's
    rejection rate at ``alpha``.
    """
    rejected = dict.fromkeys(methods, 0)
    m = n + h
    x0 = rng.standard_normal(reps) * shock_sd / math.sqrt(1 - phi**2)
    u = rng.standard_normal((reps, m - 1)) * shock_sd
    # x_t = phi x_{t-1} + u_t row by row (an AR(1) filter), started from the stationary draw x0.
    x = np.column_stack([x0, signal.lfilter([1.0], [1.0, -phi], u, axis=1, zi=(phi * x0)[:, None])[0]])
    for row in x:
        forecast = mean + phi**h * row[:n]
        outcome = mean + row[h : h + n]
        for method in methods:
            r = mz_regression(outcome, forecast, h, method)
            rejected[method] += bool(r.wald_p < alpha)
    return {method: count / reps for method, count in rejected.items()}
