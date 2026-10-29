"""Diebold-Mariano tests of equal forecast accuracy (ADR 0001 4.8, amended M5).

For two forecasts of the same target on the same origins, ``d_t = L_a,t - L_b,t`` is the loss
differential. The null is ``E[d] = 0``; the statistic is ``mean(d) / sqrt(LRV / n)``, where LRV is
the long-run variance of ``d``: ``gamma_0 + 2 * sum_j w_j * gamma_j``, with autocovariances
``gamma_j = (1/n) * sum_{t>j} (d_t - mean)(d_{t-j} - mean)`` (divisor n, mean removed).

Four estimators of the LRV, because the right one is what milestone 5 had to measure (O6):

==================  ======================================================  ==================
``lrv_method``      weights and bandwidth                                   critical values
==================  ======================================================  ==================
``rectangular``     ``w_j = 1`` for ``j <= h-1`` (Diebold-Mariano 1995),     HLN factor, t(n-1)
                    Bartlett at the same lag if the estimate is <= 0
                    (stored as ``bartlett_fallback``)
``bartlett``        Newey-West: ``w_j = 1 - j/h`` for ``j <= h-1``           HLN factor, t(n-1)
``andrews``         Bartlett ``w_j = 1 - j/S`` for ``j < S``, S from          normal
                    Andrews' (1991) AR(1) plug-in rule (``andrews_bandwidth``)
``fixed_b``         Bartlett ``w_j = 1 - j/M`` for ``j < M``, ``M = ceil(b n)``  simulated fixed-b
                    with ``b = FIXED_B`` (Kiefer-Vogelsang); no HLN factor  null (``fixed_b_null``)
==================  ======================================================  ==================

Which one is primary is decided in ``PRIMARY_METHOD`` from the size simulation of milestone 5
(``simulate_differentials``; docs/milestones/05-evaluation.md section 7.1). The others are kept in
the ``dm_test`` table as the bandwidth sensitivity the ADR asks for.

The Harvey-Leybourne-Newbold (1997) factor ``sqrt((n + 1 - 2h + h(h-1)/n) / n)`` was derived for the
equal-weight estimator (ADR 0001 4.8); it is applied with t(n-1) p-values to the two lag-(h-1)
estimators only. The automatic and fixed-b statistics have their own reference distributions.

Do not use ``statsmodels.tsa.stattools.diebold_mariano_test`` for any of these: it uses Bartlett
weights with ``max(h-1, ceil(n**(1/3)))`` lags by default (ADR 0001 O6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal, stats

METHODS = ("rectangular", "bartlett", "andrews", "fixed_b")
#: The primary DM method (amended M5, O6; default taken 2026-10-29, owner can override). Chosen by the
#: pre-registered rule of docs/milestones/05-evaluation.md 7.1: the smallest worst-case size error
#: across the three nulls at the dev sample size.
PRIMARY_METHOD = "fixed_b"
#: The fixed-b bandwidth fraction: M = ceil(FIXED_B * n) lags. Fixed before the size simulation ran.
FIXED_B = 0.1
#: Andrews (1991) constant for the Bartlett kernel's AR(1) plug-in bandwidth (eq. 6.2 as commonly
#: quoted; to verify against the paper).
ANDREWS_BARTLETT = 1.1447
#: The fixed-b null is simulated once per b on i.i.d. normal series of this length.
FIXED_B_SIM_N = 1000
FIXED_B_SIM_REPS = 20000
LOSSES = ("sq_error", "qlike")


def loss_series(y, yhat, loss: str) -> np.ndarray:
    """Per-origin losses: ``sq_error`` (level, vol points squared) or ``qlike`` (variance space).

    Their means are ``losses.rmse(...)**2`` and ``losses.qlike(...)`` exactly.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if loss == "sq_error":
        return (y - yhat) ** 2
    if loss == "qlike":
        if (y <= 0).any() or (yhat <= 0).any():
            raise ValueError("QLIKE needs y > 0 and yhat > 0")
        ratio = (y / 100) ** 2 / (yhat / 100) ** 2
        return ratio - np.log(ratio) - 1
    raise ValueError(f"unknown loss {loss!r}")


def autocovariances(d, max_lag: int) -> np.ndarray:
    """``gamma_0 .. gamma_max_lag`` of ``d`` (divisor n, mean removed).

    A 1-d series uses exact dot products; a 2-d array (one series per row, the size simulation) uses
    the FFT, which agrees to rounding (tested).
    """
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        n = d.size
        x = d - d.mean()
        return np.array([x[j:] @ x[: n - j] / n for j in range(min(max_lag, n - 1) + 1)])
    n = d.shape[1]
    x = d - d.mean(axis=1, keepdims=True)
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(x, size, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, : min(max_lag, n - 1) + 1]
    return acov / n


def weights(kernel: str, bandwidth: float, max_lag: int) -> np.ndarray:
    """``w_1 .. w_max_lag``. ``rectangular``: 1 for every lag. ``bartlett``: ``1 - j/bandwidth``, floored at 0."""
    j = np.arange(1, max_lag + 1, dtype=float)
    if kernel == "rectangular":
        return np.ones_like(j)
    if kernel == "bartlett":
        return np.clip(1 - j / bandwidth, 0, None)
    raise ValueError(f"unknown kernel {kernel!r}")


def lrv_from_acov(acov: np.ndarray, kernel: str, bandwidth: float) -> np.ndarray | float:
    """``gamma_0 + 2 * sum_j w_j gamma_j``; ``acov`` is 1-d or one row per series."""
    acov = np.asarray(acov, dtype=float)
    w = weights(kernel, bandwidth, acov.shape[-1] - 1)
    return acov[..., 0] + 2 * (acov[..., 1:] @ w)


def lag_window(method: str, h: int, n: int, d=None) -> tuple[str, float, int]:
    """``(kernel, bandwidth, max_lag)`` of a method. Bartlett with bandwidth B uses lags ``j < B``."""
    if method == "rectangular":
        return "rectangular", float(h - 1), h - 1
    if method == "bartlett":
        return "bartlett", float(h), h - 1  # Newey-West at lag h-1: w_j = 1 - j/h
    if method == "andrews":
        s = andrews_bandwidth(d)
        return "bartlett", s, max(int(math.ceil(s)) - 1, 0)
    if method == "fixed_b":
        m = fixed_b_lags(n)
        return "bartlett", float(m), m - 1
    raise ValueError(f"unknown method {method!r}")


def hln_factor(n: int, h: int) -> float:
    """Harvey-Leybourne-Newbold (1997) eq. 9: ``sqrt((n + 1 - 2h + h(h-1)/n) / n)``."""
    return math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)


def ar1_coefficient(d) -> float:
    """OLS slope of ``d_t - mean`` on ``d_{t-1} - mean``: the plug-in for Andrews' rule."""
    x = np.asarray(d, dtype=float)
    x = x - x.mean()
    den = x[:-1] @ x[:-1]
    return float(x[1:] @ x[:-1] / den) if den > 0 else 0.0


def andrews_bandwidth_from_rho(rho: float, n: int) -> float:
    """Andrews (1991) Bartlett bandwidth for an AR(1) plug-in ``rho``.

    ``alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)`` and ``S = 1.1447 * (alpha(1) * n)^(1/3)``,
    capped at n - 1 lags.
    """
    rho = min(max(rho, -0.99), 0.99)  # the rule explodes as |rho| -> 1; capped, as is usual
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return float(min(ANDREWS_BARTLETT * (alpha * n) ** (1 / 3), n - 1))


def andrews_bandwidth(d) -> float:
    d = np.asarray(d, dtype=float)
    return andrews_bandwidth_from_rho(ar1_coefficient(d), d.size)


def fixed_b_lags(n: int, b: float = FIXED_B) -> int:
    """``M = ceil(b n)``: the Bartlett bandwidth of the fixed-b statistic (lags ``j < M``)."""
    return max(int(math.ceil(b * n)), 1)


@lru_cache(maxsize=8)
def fixed_b_null(b: float = FIXED_B, n: int = FIXED_B_SIM_N, reps: int = FIXED_B_SIM_REPS, seed: int = 20261028) -> np.ndarray:
    """Sorted ``|t|`` of the Bartlett fixed-b statistic on i.i.d. N(0, 1) series of length n.

    Under fixed-b asymptotics (Kiefer-Vogelsang) the statistic's limit depends on b and the kernel
    only, not on the dependence of ``d``; i.i.d. normal series of length 1,000 stand in for that
    limit. Deterministic: a fixed seed. Measured (amended M5, review A): the shipped null's 95%
    quantile of ``|t|`` is 2.2458; over 20 seeds (20261028..20261047) its mean is 2.2290 and its sd
    0.0195, so the shipped critical value sits about 0.9 sd above the seed mean. The comparison with
    Kiefer-Vogelsang's published values is **not verified** (the first M5 version asserted agreement
    with a cubic quoted from memory; review A refuted it).
    """
    rng = np.random.default_rng(seed)
    m = fixed_b_lags(n, b)
    out = []
    for start in range(0, reps, 2000):
        x = rng.standard_normal((min(2000, reps - start), n))
        lrv = lrv_from_acov(autocovariances(x, m - 1), "bartlett", m)
        out.append(np.abs(x.mean(axis=1)) / np.sqrt(lrv / n))
    return np.sort(np.concatenate(out))


def fixed_b_p_value(t: float, b: float = FIXED_B) -> float:
    """Two-sided p-value of ``t`` against the simulated fixed-b null: ``(1 + #{|T| >= |t|}) / (1 + R)``.

    NaN for a NaN statistic (amended M5, review A: it returned ``1/(1 + R)``, the smallest p-value).
    """
    if math.isnan(t):
        return math.nan
    null = fixed_b_null(b)
    exceed = null.size - np.searchsorted(null, abs(t), side="left")
    return float((1 + exceed) / (1 + null.size))


@dataclass(frozen=True)
class DmResult:
    n: int
    mean_d: float
    dm_stat: float
    p_value: float
    lrv: float
    lrv_method: str  # the method, or bartlett_fallback
    bandwidth: float
    max_lag: int


def dm_test(d, h: int, method: str = "rectangular") -> DmResult:
    """The Diebold-Mariano test of ``E[d] = 0`` with the chosen long-run variance (module docstring).

    ``rectangular`` and ``bartlett`` report the HLN-corrected statistic with a t(n-1) p-value;
    ``andrews`` the plain statistic with a normal p-value; ``fixed_b`` the plain statistic with the
    simulated fixed-b p-value. If the LRV is <= 0 the rectangular estimate falls back to Bartlett at
    the same lag (ADR 0001 4.8); any other method with LRV <= 0 returns NaN statistics.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 3 or not np.isfinite(d).all():
        raise ValueError("d must be a finite 1-d series with at least 3 values")
    n = d.size
    kernel, bandwidth, max_lag = lag_window(method, h, n, d)
    acov = autocovariances(d, max_lag)
    lrv = float(lrv_from_acov(acov, kernel, bandwidth))
    name = method
    if method == "rectangular" and lrv <= 0:
        kernel, bandwidth, name = "bartlett", float(h), "bartlett_fallback"
        lrv = float(lrv_from_acov(acov, kernel, bandwidth))
    mean_d = float(d.mean())
    if lrv <= 0:
        return DmResult(n, mean_d, math.nan, math.nan, lrv, name, bandwidth, max_lag)
    t = mean_d / math.sqrt(lrv / n)
    if method in ("rectangular", "bartlett"):
        t *= hln_factor(n, h)
        p = float(2 * stats.t.sf(abs(t), n - 1))
    elif method == "andrews":
        p = float(2 * stats.norm.sf(abs(t)))
    else:
        p = fixed_b_p_value(t)
    return DmResult(n, mean_d, t, p, lrv, name, bandwidth, max_lag)


def holm(p_values) -> np.ndarray:
    """Holm (1979; not cited in this repo before M5, to verify) step-down adjusted p-values.

    Sort ascending; the k-th smallest (k = 1..m) gets ``(m - k + 1) * p``, made non-decreasing in
    that order and capped at 1. Rejecting where the adjusted p < alpha controls the family-wise error
    rate at alpha under any dependence between the tests. NaN p-values are left out of the family.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        out[i] = min(running, 1.0)
    return out


# --------------------------------------------------------------------------
# The size simulation (O6): nulls with E[d] = 0 and overlapping h-step errors
# --------------------------------------------------------------------------


def _overlap(eps: np.ndarray, h: int) -> np.ndarray:
    """Row-wise sums of h consecutive shocks: an MA(h-1) h-step forecast error."""
    c = np.cumsum(eps, axis=1)
    out = c[:, h - 1 :].copy()
    out[:, 1:] -= c[:, : -h]
    return out


def simulate_differentials(design: str, n: int, h: int, reps: int, rng: np.random.Generator, rho: float = 0.8,
                           phi: float = 0.98, state_sd: float = 0.5) -> np.ndarray:
    """``reps`` loss-differential series of length n under a null ``E[d] = 0`` (one per row).

    * ``ma``: ``d`` is itself an overlapping h-step error (a sum of h i.i.d. N(0, 1) shocks): linear,
      MA(h-1), the case DM's truncation at h-1 is exact for.
    * ``sq_error``: two equally good forecasts. Their h-step errors are overlapping sums of shocks
      with correlation ``rho`` between the models; ``d = e_a^2 - e_b^2``. Also MA(h-1).
    * ``state``: as ``sq_error``, but model A's shocks are scaled by ``exp(s_t/2 - state_sd^2/4)``,
      with ``s`` a stationary AR(1) (``phi``, stationary sd ``state_sd``), so ``E[e_a^2] = E[e_b^2]``
      while A's relative accuracy depends on a persistent state: autocorrelation beyond lag h-1
      (ADR 0001 4.8, "When the assumption fails").
    """
    m = n + h - 1
    if design == "ma":
        return _overlap(rng.standard_normal((reps, m)), h)
    za = rng.standard_normal((reps, m))
    zb = rho * za + math.sqrt(1 - rho**2) * rng.standard_normal((reps, m))
    if design == "state":
        s0 = rng.standard_normal(reps) * state_sd
        eta = rng.standard_normal((reps, m - 1)) * state_sd * math.sqrt(1 - phi**2)
        # s_t = phi s_{t-1} + eta_t row by row, started from the stationary draw s0
        s = np.column_stack([s0, signal.lfilter([1.0], [1.0, -phi], eta, axis=1, zi=(phi * s0)[:, None])[0]])
        za = za * np.exp(s / 2 - state_sd**2 / 4)
    elif design != "sq_error":
        raise ValueError(f"unknown design {design!r}")
    return _overlap(za, h) ** 2 - _overlap(zb, h) ** 2


def rejection_rates(d: np.ndarray, h: int, methods=METHODS, alpha: float = 0.05) -> dict[str, float]:
    """The share of rows of ``d`` on which each method rejects ``E[d] = 0`` at ``alpha`` (two-sided).

    The same estimators as ``dm_test``, vectorised over rows (tested to agree with ``dm_test``).
    """
    reps, n = d.shape
    out = {}
    means = d.mean(axis=1)
    needed = {}
    for method in methods:
        if method == "andrews":
            rhos = np.array([ar1_coefficient(row) for row in d])
            bands = np.array([andrews_bandwidth_from_rho(r, n) for r in rhos])
            needed[method] = int(math.ceil(bands.max()))
        else:
            needed[method] = lag_window(method, h, n)[2]
    acov = autocovariances(d, max(needed.values()))
    for method in methods:
        if method == "andrews":
            lrv = np.array([lrv_from_acov(acov[i, : int(math.ceil(s))], "bartlett", s) for i, s in enumerate(bands)])
        else:
            kernel, bandwidth, max_lag = lag_window(method, h, n)
            lrv = lrv_from_acov(acov[:, : max_lag + 1], kernel, bandwidth)
            if method == "rectangular":
                fallback = lrv_from_acov(acov[:, : max_lag + 1], "bartlett", float(h))
                lrv = np.where(lrv > 0, lrv, fallback)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = means / np.sqrt(lrv / n)
        if method in ("rectangular", "bartlett"):
            t = t * hln_factor(n, h)
            crit = stats.t.ppf(1 - alpha / 2, n - 1)
        elif method == "andrews":
            crit = stats.norm.ppf(1 - alpha / 2)
        else:
            null = fixed_b_null(FIXED_B)
            crit = float(np.quantile(null, 1 - alpha))
        out[method] = float(np.mean(np.abs(t) > crit))
    return out
