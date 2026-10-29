"""The GARCH family of milestone 3: GARCH, GJR-GARCH and EGARCH on S&P 500 returns, and ARIMA-GARCH on ln VIX.

**Where each model is fitted (ADR 0001 4.4).** ``garch``, ``gjr`` and ``egarch`` are fitted on daily
``^GSPC`` log returns and forecast the ``rv21`` target (``ivf.models.rv21``). ``arima_garch_log`` puts
an AR mean on ln VIX (d = 0; amended M3, review A) and GARCH on the AR's residuals, the innovations,
and forecasts the VIX level (M2's ``vix_level``). GARCH is never fitted on the VIX level itself, whose
persistence it would mistake for clustering: the category error of concept note 10, demonstrated by
``garch_on_the_level_demo`` below.

**Units.** ``arch`` is given returns in percent, ``e = 100 r`` (its optimiser is tuned for numbers of
order 1; ``rescale=False`` so it never rescales on its own). Its variances are then in percent², and
a decimal daily variance is ``sigma2 / 100**2``.

**Variance aggregation (ADR 0001 4.2 and 4.4, amended M3).** ``rv21`` at date d is
``100 sqrt((252/21) sum_{j=0}^{20} r_{d-j}**2)`` with r decimal and **no mean removed**, so its
variance-space value at ``t + 21`` is ``v = (252/21) sum_{k=1}^{21} r_{t+k}**2``. With a **zero-mean**
return model ``E_t[r_{t+k}**2] = sigma2_{t+k|t} / 100**2``, so::

    G_t = (252/21) * sum_{k=1}^{21} sigma2_{t+k|t} / 100**2      (annualised decimal variance: v-hat)
    yhat = 100 * sqrt(G_t)                                       (vol points, the RMS vol: var_method native)

``G_t`` is ``E_t[v]`` exactly (no Jensen term), which is what QLIKE on variance rewards. Worked
number: a constant ``sigma2 = 1`` (1% a day) gives ``G = 12 * 21 * 1 / 10**4 = 0.0252`` and
``yhat = 15.8745``, the value ``build.rv_trailing`` gives for 21 returns of 0.01.

**The multi-step path** ``sigma2_{t+k|t}``, k = 1..21, from the one-step value ``s1 = sigma2_{t+1|t}``
(known at t, from the recursion through returns up to t):

* GARCH(1,1): ``sigma2_{t+k} = omega + (alpha + beta) sigma2_{t+k-1}``.
* GJR(1,1,1): ``sigma2_{t+k} = omega + (alpha + gamma/2 + beta) sigma2_{t+k-1}``; the 1/2 is
  ``E[z**2 1(z < 0)]`` for a symmetric z (normal or Student-t), as in ``arch``'s analytic forecast.
* EGARCH(1,1,1): ``ln sigma2_{t+1} = omega + g(z_t) + beta ln sigma2_t`` with
  ``g(z) = alpha (|z| - sqrt(2/pi)) + gamma z`` (``arch``'s form: the constant is sqrt(2/pi) whatever
  the distribution). Iterating, ``E_t[sigma2_{t+k}] = exp(omega sum_{i=0}^{k-2} beta**i +
  beta**(k-1) ln s1) * prod_{i=0}^{k-2} M(beta**i)`` with ``M(c) = E[exp(c g(z))]``. Under
  Student-t innovations M is infinite in theory (``exp(c alpha |z|)`` against polynomial tails), but
  the divergence comes only from |z| of a few hundred (review A: for t(6) and c = 0.1 the partial
  integral is 1.08 up to |z| = 200 and 211.5 up to 400), so a simulated mean is a stable, truncated
  value, not a meaningless one; it still carries Monte Carlo noise and depends on the draws. M is
  taken instead as the exact mean over the **training standardised residuals, rescaled to unit
  RMS** (``E[z**2] = 1`` must hold for the model's shocks; amended M3, review A): deterministic,
  finite, and fixed at the cutoff with the parameters. It is exact for that empirical distribution,
  not for the fitted one. Decision D4 of the M3 walkthrough.

**Innovations (pre-registered rule, decision D3).** At every cutoff each model is fitted with normal
and with Student-t innovations, and the fit with the **lower AIC** is kept (a tie keeps normal). A fit
counts only if ``arch`` reports convergence, the AIC is finite, the model is covariance-stationary
(``alpha + beta < 1``; ``alpha + gamma/2 + beta < 1``; ``|beta| < 1`` for EGARCH), and this module's
recursion reproduces ``arch``'s conditional variances on the training sample (``FILTER_RTOL``). If no
candidate counts, ``fit`` raises and the runner keeps the previous fold's fit (ADR 0001 4.5).

**Forecasting inside a fold.** Parameters are frozen at the cutoff (and so is the backcast, the
recursion's start value); the recursion runs through returns up to each origin t, in this module, not
in ``arch``: ``arch``'s variance bounds use the whole sample's variance and maximum, so its filtered
values could depend on data after t. ``filter_variance`` reproduces ``arch``'s recursion (tested to
1e-9) without the bounds.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ivf.features import build
from ivf.models.baselines import Fit, Forecast, lognormal_forecast

#: Returns are given to arch in percent.
SCALE = 100.0
KINDS = ("garch", "gjr", "egarch")
DISTS = ("normal", "t")
#: E|z| for a standard normal; arch's EGARCH uses it for every distribution.
SQRT2_OV_PI = math.sqrt(2 / math.pi)
#: arch's backcast: an exponentially weighted mean of the first 75 squared residuals, decay 0.94.
BACKCAST_TAU = 75
BACKCAST_DECAY = 0.94
#: arch's cap on ln sigma2 in the EGARCH recursion.
LNSIGMA_MAX = float(np.log(np.finfo(np.double).max) - 0.1)
#: How closely filter_variance must reproduce arch's conditional variances on the training sample.
FILTER_RTOL = 1e-7
#: The rv21 window (ADR 0001 4.2) and the annualisation factor 252/21.
WINDOW = build.RV_WINDOW
ANNUALISE = build.SESSIONS_PER_YEAR / build.RV_WINDOW


def backcast(e: np.ndarray) -> float:
    """arch's start value for the recursion: ``sum(w_i e_i**2)``, ``w_i ∝ 0.94**i`` over the first min(75, n).

    It depends on the first 75 values only, so it is the same at every cutoff of a run whose
    training sample starts at the same session. Worked number: e = (1, 3) gives weights
    (1, 0.94)/1.94 and ``(1 + 0.94*9)/1.94 = 4.876289``.
    """
    tau = min(BACKCAST_TAU, len(e))
    w = BACKCAST_DECAY ** np.arange(tau)
    return float(np.sum(e[:tau] ** 2 * (w / w.sum())))


def persistence(kind: str, p: dict) -> float:
    """The factor of the multi-step recursion: alpha+beta, alpha+gamma/2+beta, or |beta| (EGARCH, in logs)."""
    if kind == "garch":
        return p["alpha"] + p["beta"]
    if kind == "gjr":
        return p["alpha"] + p["gamma"] / 2 + p["beta"]
    return abs(p["beta"])


def filter_variance(kind: str, p: dict, e: np.ndarray, bc: float) -> np.ndarray:
    """``sigma2[i]`` = the variance of ``e[i]`` given ``e[:i]``, for i = 0..n (n + 1 values; percent²).

    The last value, ``sigma2[n]``, is the one-step forecast past the sample: at an origin t whose
    return is ``e[n-1]``, it is ``sigma2_{t+1|t}``. Element i uses ``e[:i]`` only (rule L1).

    arch's recursion exactly, without its variance bounds. At i = 0 the lagged terms are the
    backcast ``bc``: ``alpha*bc + gamma*bc/2 + beta*bc`` for GARCH/GJR; for EGARCH only
    ``beta*ln(bc)`` (arch leaves out the shock terms at i = 0).
    """
    n = len(e)
    out = np.empty(n + 1)
    omega, alpha, beta = p["omega"], p["alpha"], p["beta"]
    gamma = p.get("gamma", 0.0)
    if kind in ("garch", "gjr"):
        s2_prev, e2_prev, neg_prev = bc, bc, 0.5 * bc
        for i in range(n + 1):
            s2 = omega + alpha * e2_prev + gamma * neg_prev + beta * s2_prev
            out[i] = s2
            if i < n:
                e2 = e[i] * e[i]
                s2_prev, e2_prev, neg_prev = s2, e2, (e2 if e[i] < 0 else 0.0)
        return out
    if kind != "egarch":
        raise ValueError(f"unknown kind {kind!r}")
    ln_prev, shock = math.log(bc), 0.0
    for i in range(n + 1):
        ln = min(omega + shock + beta * ln_prev, LNSIGMA_MAX)
        s2 = math.exp(ln)
        out[i] = s2
        if i < n:
            z = e[i] / math.sqrt(s2)
            shock = alpha * (abs(z) - SQRT2_OV_PI) + gamma * z
            ln_prev = ln
    return out


def egarch_mgf(p: dict, z: np.ndarray, horizon: int = WINDOW) -> list[float]:
    """``M(beta**i) = mean(exp(beta**i * g(z)))`` for i = 0..horizon-2, over the standardised residuals z.

    The expectation is exact under the empirical distribution of z (module docstring). Worked
    number: z = (-1, 1), alpha = 0.1, gamma = -0.05, beta = 0.5 gives at i = 0
    ``exp(0.1*(1 - 0.797885)) * cosh(0.05) = 1.020417 * 1.001250 = 1.021693``.
    """
    g = p["alpha"] * (np.abs(z) - SQRT2_OV_PI) + p.get("gamma", 0.0) * z
    return [float(np.mean(np.exp(p["beta"] ** i * g))) for i in range(horizon - 1)]


def variance_path(kind: str, p: dict, s1: np.ndarray, horizon: int = WINDOW, mgf: list[float] | None = None) -> np.ndarray:
    """``E_t[sigma2_{t+k}]`` for k = 1..horizon, one row per value of ``s1 = sigma2_{t+1|t}`` (percent²)."""
    s1 = np.asarray(s1, dtype=float)
    path = np.empty((len(s1), horizon))
    path[:, 0] = s1
    if kind in ("garch", "gjr"):
        phi = persistence(kind, p)
        for k in range(1, horizon):
            path[:, k] = p["omega"] + phi * path[:, k - 1]
        return path
    if mgf is None or len(mgf) < horizon - 1:
        raise ValueError("EGARCH needs M(beta**i) for i = 0..horizon-2")
    omega, beta = p["omega"], p["beta"]
    ln1 = np.log(s1)
    for k in range(2, horizon + 1):
        drift = omega * sum(beta**i for i in range(k - 1))
        path[:, k - 1] = np.exp(drift + beta ** (k - 1) * ln1) * math.prod(mgf[: k - 1])
    return path


def aggregate(path: np.ndarray) -> np.ndarray:
    """``G = (252/21) * sum_k sigma2_{t+k|t} / 100**2``: the annualised decimal variance of the next 21 sessions."""
    if path.shape[1] != WINDOW:
        raise ValueError(f"the rv21 window is {WINDOW} sessions, not {path.shape[1]}")
    return ANNUALISE * path.sum(axis=1) / SCALE**2


# --- fitting with arch ---------------------------------------------------------------------------


def _vol_process(kind: str):
    from arch.univariate import EGARCH, GARCH

    if kind == "garch":
        return GARCH(p=1, o=0, q=1)
    if kind == "gjr":
        return GARCH(p=1, o=1, q=1)
    if kind == "egarch":
        return EGARCH(p=1, o=1, q=1)
    raise ValueError(f"unknown kind {kind!r}")


def _distribution(dist: str):
    from arch.univariate import Normal, StudentsT

    return {"normal": Normal, "t": StudentsT}[dist]()


VOL_NAMES = {"garch": ("omega", "alpha", "beta"), "gjr": ("omega", "alpha", "gamma", "beta"), "egarch": ("omega", "alpha", "gamma", "beta")}


@dataclass
class Candidate:
    """One (mean order, distribution) fit at one cutoff, JSON-ready through ``record``."""

    p: int
    dist: str
    converged: bool
    aic: float | None
    loglik: float | None
    values: list[float]
    names: list[str]
    vol: dict[str, float]
    ok: bool
    why: str
    filter_max_rel_diff: float | None

    def record(self) -> dict[str, Any]:
        return {
            "p": self.p, "dist": self.dist, "converged": self.converged, "aic": self.aic, "loglik": self.loglik,
            "values": self.values, "names": self.names, "ok": self.ok, "why": self.why, "filter_max_rel_diff": self.filter_max_rel_diff,
        }


def _finite_or_none(x) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


def ar_stationary(phi) -> bool:
    """True if the AR polynomial ``1 - phi_1 z - ... - phi_p z^p`` has every root outside the unit circle.

    Checked on the companion matrix: every eigenvalue inside the unit circle. Worked numbers:
    (0.5,) yes; (1.0,) no; (1.2, -0.3): eigenvalues 0.845 and 0.355, yes; (0.5, 0.6): 1.064, no.
    """
    phi = list(phi)
    if not phi:
        return True
    companion = np.zeros((len(phi), len(phi)))
    companion[0] = phi
    companion[1:, :-1] = np.eye(len(phi) - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 1)


def ols_residuals(data: np.ndarray, p: int, constant: bool, hold_back: int) -> np.ndarray:
    """Residuals of the mean model by OLS on the held-back sample: arch's own starting point, used for the backcast."""
    y = data[hold_back:]
    cols = ([np.ones(len(y))] if constant else []) + [data[hold_back - i : len(data) - i] for i in range(1, p + 1)]
    if not cols:
        return y.copy()
    z = np.column_stack(cols)
    return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]


def fit_candidate(kind: str, data: np.ndarray, dist: str, bc: float, *, p: int = 0, constant: bool = False, hold_back: int = 0,
                  start=None) -> tuple[Candidate, np.ndarray | None]:
    """Fit one model with arch; return the candidate and the residuals the variance model sees.

    ``data`` is in percent. ``p = 0`` and no constant: zero mean, the residuals are the data.
    Otherwise an AR(p) mean, with a constant if ``constant`` (``arima_garch_log``). ``hold_back``
    drops the first values for every p, so candidates with different p are scored on the same
    sample and their AICs are comparable. A mean with a constant must also be a stationary AR.
    """
    from arch.univariate import ARX, ConstantMean, ZeroMean

    if p:
        mean = ARX(data, lags=p, constant=constant, hold_back=hold_back, rescale=False)
    elif constant:
        mean = ConstantMean(data, hold_back=hold_back or None, rescale=False)
    else:
        mean = ZeroMean(data, hold_back=hold_back or None, rescale=False)
    mean.volatility = _vol_process(kind)
    mean.distribution = _distribution(dist)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mean.fit(disp="off", backcast=bc, starting_values=start)
    except Exception as exc:  # a failed candidate is recorded and skipped, never fatal
        return Candidate(p, dist, False, None, None, [], [], {}, False, type(exc).__name__, None), None
    names = list(res.params.index)
    values = [float(v) for v in res.params]
    k = int(constant)
    const, phi = (values[0] if constant else 0.0), values[k : k + p]
    vol = dict(zip(VOL_NAMES[kind], values[k + p : k + p + len(VOL_NAMES[kind])]))
    resid = data[hold_back:] - const
    for i in range(1, p + 1):
        resid = resid - phi[i - 1] * data[hold_back - i : len(data) - i]
    converged = res.convergence_flag == 0
    aic = _finite_or_none(res.aic)
    why, diff = "", None
    if not converged:
        why = "not converged"
    elif aic is None:
        why = "AIC not finite"
    elif persistence(kind, vol) >= 1:
        why = "not stationary"
    elif constant and not ar_stationary(phi):
        why = "AR not stationary"
    else:
        ours = filter_variance(kind, vol, resid, bc)[:-1]
        theirs = np.asarray(res.conditional_volatility, dtype=float)[hold_back:] ** 2
        diff = float(np.max(np.abs(ours / theirs - 1)))
        if not diff <= FILTER_RTOL:
            why = "recursion does not reproduce arch"
    cand = Candidate(p, dist, converged, aic, _finite_or_none(res.loglikelihood), values, names, vol, why == "", why, diff)
    return cand, resid


def choose(candidates: list[Candidate]) -> Candidate | None:
    """The usable candidate with the smallest AIC; a tie keeps the earlier one (normal before t, small p first)."""
    best = None
    for c in candidates:
        if c.ok and (best is None or c.aic < best.aic):
            best = c
    return best


def first_finite(x: np.ndarray) -> int:
    """The index of the first finite value; the value at the end (the cutoff) must be finite too."""
    finite = np.flatnonzero(np.isfinite(x))
    if not finite.size or not np.isfinite(x[-1]):
        raise ValueError("the value at the cutoff is missing")
    return int(finite[0])


def finite_sequence(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(the finite values in order, count)`` where ``count[i]`` is the number of finite values at or before i.

    **Missing values (gap policy; decision D7 of the M3 walkthrough).** ``arch`` cannot take NaN, so a
    missing return (or a change of ln IV touching a missing close) is **skipped**: the recursion steps
    from the last observed shock to the next one, as the historical means skip nulls. Nothing is
    filled. Amended M3, first real run: the first rule, "the gap-free run ending at the cutoff", left
    ``arima_garch_log`` too few changes to fit at the first real cutoff (real VIX has isolated missing
    closes in burn-in, allowed by G3) and the run stopped; synthetic data has no gaps, so no test had
    caught it.
    """
    keep = np.isfinite(values)
    return values[keep], np.cumsum(keep)


def _warm(previous: Fit | None) -> dict:
    if previous is None or not previous.params.get("candidates"):
        return {}
    return {(c["p"], c["dist"]): c["values"] for c in previous.params["candidates"] if c["values"]}


# --- the rv21 models: GARCH, GJR, EGARCH on S&P 500 returns -------------------------------------


class GarchRv:
    """One of GARCH(1,1), GJR(1,1,1), EGARCH(1,1,1) with zero mean on 100 x the ``^GSPC`` log returns.

    ``fit`` sees returns up to the cutoff only, from the first one; a missing return is skipped (``finite_sequence``).
    ``forecast`` returns ``v_hat = G_t`` and ``yhat = 100 sqrt(G_t)`` at each origin.
    """

    fit_space = "returns"
    var_method = "native"

    def __init__(self, kind: str, dists: tuple[str, ...] = DISTS):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind, self.model_id, self.dists = kind, kind, tuple(dists)

    def fit(self, inputs, previous: Fit | None) -> Fit:
        started = time.perf_counter()
        r = inputs.r
        first = first_finite(r)
        e, _ = finite_sequence(SCALE * r[first:])
        bc = backcast(e)
        warm = _warm(previous)
        cands, resids = [], {}
        for dist in self.dists:
            cand, resid = fit_candidate(self.kind, e, dist, bc, start=warm.get((0, dist)))
            cands.append(cand)
            resids[dist] = resid
        best = choose(cands)
        if best is None:
            raise ValueError(f"{self.kind}: no candidate converged ({[c.why for c in cands]})")
        params: dict[str, Any] = {"dist": best.dist, **best.vol, "backcast": bc, "persistence": persistence(self.kind, best.vol),
                                  "candidates": [c.record() for c in cands]}
        if best.dist == "t":
            params["nu"] = best.values[-1]
        if self.kind == "egarch":
            z = resids[best.dist] / np.sqrt(filter_variance(self.kind, best.vol, resids[best.dist], bc)[:-1])
            rms = float(np.sqrt(np.mean(z**2)))
            params["z_rms"] = rms  # before rescaling; 1 for a well-specified model
            params["mgf"] = egarch_mgf(best.vol, z / rms)  # unit RMS: E[z**2] = 1 (amended M3, review A)
        aics = {c.dist: c.aic for c in cands}
        return Fit(
            params=params,
            hyperparams={"kind": self.kind, "mean": "zero", "scale": SCALE, "dists": list(self.dists), "dist_rule": "aic",
                         "aic_by_dist": aics, "multistep": "empirical_mgf" if self.kind == "egarch" else "analytic",
                         "window": WINDOW, "annualise": ANNUALISE},
            n_train=len(e), train_first=first, train_last_target=len(r) - 1,
            converged=True, loglik=best.loglik, aic=best.aic, fit_seconds=time.perf_counter() - started,
        )

    def one_step(self, fit: Fit, r: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """``sigma2_{t+1|t}`` (percent²) at each origin, from the frozen parameters and returns up to t."""
        first = fit.train_first
        e, count = finite_sequence(SCALE * r[first : int(origins.max()) + 1])
        if not np.isfinite(r[origins]).all():
            raise ValueError(f"{self.model_id}: the return at an origin is missing")
        s2 = filter_variance(self.kind, fit.params, e, fit.params["backcast"])
        return s2[count[origins - first]]  # count = the returns up to t, so this is sigma2 of the next one

    def forecast(self, fit: Fit, inputs, origins: np.ndarray) -> Forecast:
        path = variance_path(self.kind, fit.params, self.one_step(fit, inputs.r, origins), WINDOW, fit.params.get("mgf"))
        g = aggregate(path)
        y = 100 * np.sqrt(g)
        return Forecast(yhat=y, yhat_naive=y.copy(), v_hat=g)


# --- ARIMA-GARCH on ln VIX (a vix_level model) ------------------------------------------------


def psi_weights(phi: list[float], h: int) -> np.ndarray:
    """The MA(inf) weights ``psi_0..psi_{h-1}`` of an AR(p): ``psi_0 = 1``, ``psi_j = sum_i phi_i psi_{j-i}``.

    For d = 0, ``x_{t+h} - E_t x_{t+h} = sum_{j=1}^{h} psi_{h-j} eps_{t+j}``. Worked number: phi = (0.5,)
    gives psi = 1, 0.5, 0.25.
    """
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        psi[j] = sum(phi[i] * psi[j - 1 - i] for i in range(min(len(phi), j)))
    return psi


def psi_cumulative(phi: list[float], h: int) -> np.ndarray:
    """``Psi_m = sum_{i=0}^{m} psi_i``: the weights for d = 1, where the AR is on the changes.

    Worked number: phi = (0.5,) gives Psi = 1, 1.5, 1.75.
    """
    return np.cumsum(psi_weights(phi, h))


class ArimaGarch:
    """An AR(p) on ``x = 100 ln VIX`` with a constant and GARCH(1,1) errors (d = 0; M2's interface).

    ``x_t = c + sum_i phi_i x_{t-i} + eps_t``, ``eps_t`` GARCH(1,1): an ARMA(p, 0)-GARCH on the level of
    ln IV, which reverts to its mean. **Amended M3, review A:** the first version differenced
    (d = 1, no constant, the AR on ``100 Δ ln VIX``); on the synthetic market (ln VIX an AR(1) with
    phi = 0.98) its h = 21 RMSE was 2.8% *worse* than the random walk while d = 0 was 9% better, the
    same lesson as M2's ``arima_log`` (review A there). d = 1 stays an option (``garch.d`` in
    ``configs/models.yaml``). p in 0..max_p and the distribution are chosen together by AIC on a
    common sample (the first max_p values held back); a candidate whose AR is not stationary is
    refused. ``arch`` has AR means, not MA, hence q = 0.

    The forecast of ``ln VIX_{t+h}``: the mean by iterating the AR from the values up to t (d = 1:
    cumulating the forecast changes onto ``ln VIX_t``), and the variance
    ``s2 = sum_{j=1}^{h} w_{h-j}**2 sigma2_{t+j|t} / 100**2`` with ``w = psi`` (d = 0) or the cumulative
    ``Psi`` (d = 1): exact for the conditional variance because the eps are martingale differences.
    ``yhat = exp(mu + s2/2)`` treats ``ln VIX_{t+h}`` as normal, which a sum of GARCH shocks is only
    approximately; ``exp(mu)`` is stored as the median.
    """

    model_id = "arima_garch_log"
    fit_space = "log_level"
    var_method = "plugin"
    D_VALUES = (0, 1)

    def __init__(self, max_p: int = 2, dists: tuple[str, ...] = DISTS, d: int = 0):
        if d not in self.D_VALUES:
            raise ValueError(f"d must be one of {self.D_VALUES}")
        self.max_p, self.dists, self.d = max_p, tuple(dists), d

    def _series(self, x: np.ndarray) -> np.ndarray:
        return SCALE * (x if self.d == 0 else np.diff(x))

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        started = time.perf_counter()
        x = np.log(y)
        first = first_finite(x)
        z, _ = finite_sequence(self._series(x[first:]))
        hb, constant = self.max_p, self.d == 0
        if len(z) <= hb + BACKCAST_TAU:
            raise ValueError("too few values of ln IV to fit ARIMA-GARCH")
        warm = _warm(previous)
        cands, bcs = [], {}
        for p in range(self.max_p + 1):
            bcs[p] = backcast(ols_residuals(z, p, constant, hb))
            for dist in self.dists:
                cand, _ = fit_candidate("garch", z, dist, bcs[p], p=p, constant=constant, hold_back=hb, start=warm.get((p, dist)))
                cands.append(cand)
        best = choose(cands)
        if best is None:
            raise ValueError("arima_garch_log: no candidate converged")
        k = int(constant)
        params: dict[str, Any] = {"p": best.p, "const": best.values[0] if constant else 0.0, "phi": best.values[k : k + best.p],
                                  "dist": best.dist, **best.vol, "backcast": bcs[best.p],
                                  "persistence": persistence("garch", best.vol), "candidates": [c.record() for c in cands]}
        if best.dist == "t":
            params["nu"] = best.values[-1]
        return Fit(
            params=params,
            hyperparams={"max_p": self.max_p, "d": self.d, "constant": constant, "vol": "GARCH(1,1)", "scale": SCALE,
                         "dists": list(self.dists), "criterion": "aic", "hold_back": hb},
            n_train=len(z) - hb, train_first=first, train_last_target=len(y) - 1,
            converged=True, loglik=best.loglik, aic=best.aic, fit_seconds=time.perf_counter() - started,
        )

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        p, first, hb, d = fit.params["p"], fit.train_first, fit.hyperparams["hold_back"], fit.hyperparams["d"]
        const, phi = fit.params.get("const", 0.0), list(fit.params["phi"])
        x = np.log(y[first : int(origins.max()) + 1])
        if not np.isfinite(x[origins - first]).all():
            raise ValueError("arima_garch_log: the close at an origin is missing")
        series = SCALE * (x if d == 0 else np.diff(x))  # d = 1: the change dated first + k + 1 is index k
        z, count = finite_sequence(series)
        eps = z[hb:] - const
        for i in range(1, p + 1):
            eps = eps - phi[i - 1] * z[hb - i : len(z) - i]
        s2 = filter_variance("garch", fit.params, eps, fit.params["backcast"])
        h_max = max(horizons)
        w = psi_weights(phi, h_max) if d == 0 else psi_cumulative(phi, h_max)
        mu = {h: np.empty(len(origins)) for h in horizons}
        var = {h: np.empty(len(origins)) for h in horizons}
        for n, t in enumerate(origins):
            j = int(t) - first  # x[j] is ln y_t
            m = int(count[j] if d == 0 else count[j - 1])  # the finite series values dated <= t; z[m - 1] is the last
            path = variance_path("garch", fit.params, s2[m - hb : m - hb + 1], h_max)[0]
            hist = list(z[max(0, m - p) : m])
            means = []
            for _ in range(h_max):
                nxt = const + sum(phi[i] * hist[-1 - i] for i in range(p))
                means.append(nxt)
                hist.append(nxt)
            level = np.asarray(means) / SCALE if d == 0 else x[j] + np.cumsum(means) / SCALE
            for h in horizons:
                mu[h][n] = level[h - 1]
                var[h][n] = float(np.sum(w[h - 1 :: -1][:h] ** 2 * path[:h])) / SCALE**2
        return {h: lognormal_forecast(mu[h], var[h]) for h in horizons}


# --- concept note 10: GARCH on the VIX level, demonstrated --------------------------------------


def garch_on_the_level_demo(vix: np.ndarray) -> dict[str, float]:
    """Fit a constant-mean GARCH(1,1) (normal) to the VIX **level**, and the right model to its log changes.

    Returns the numbers concept note 10 and notebook 03 discuss. On the level, the "shocks"
    ``V_t - mu`` are the level's own deviations from its mean, so (i) their lag-1 autocorrelation is
    the level's persistence, not zero; (ii) the fitted alpha + beta is that persistence showing up
    as "volatility clustering"; (iii) the one-step mean forecast is the constant mu, the
    historical-mean baseline, whatever the level is today. ``vix`` must be gap-free.
    """
    from arch import arch_model

    from ivf.eda import acf

    vix = np.asarray(vix, dtype=float)
    if not np.isfinite(vix).all():
        raise ValueError("the demo needs a gap-free series")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        level = arch_model(vix, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        dlog = arch_model(SCALE * np.diff(np.log(vix)), mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
    z_level = np.asarray(level.std_resid, dtype=float)
    z_dlog = np.asarray(dlog.std_resid, dtype=float)
    mu = float(level.params["mu"])
    rmse_level_mean = float(np.sqrt(np.mean((vix[1:] - mu) ** 2)))
    rmse_rw = float(np.sqrt(np.mean((vix[1:] - vix[:-1]) ** 2)))
    return {
        "level_alpha_plus_beta": float(level.params["alpha[1]"] + level.params["beta[1]"]),
        "level_std_resid_acf1": float(acf(z_level, 1)[1]),
        "level_one_step_rmse": rmse_level_mean,
        "rw_one_step_rmse": rmse_rw,
        "dlog_alpha": float(dlog.params["alpha[1]"]),
        "dlog_std_resid_acf1": float(acf(z_dlog, 1)[1]),
    }
