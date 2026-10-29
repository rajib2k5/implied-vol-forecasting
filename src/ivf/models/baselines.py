"""The milestone 2 baselines: random walk, historical mean, ARIMA on log IV and HAR on log IV.

Every model has the same two steps, and the walk-forward runner (``ivf.evaluation.walkforward``)
is the only caller:

* ``fit(y, horizons, previous)``: ``y`` is the level series (vol points, NaN where there is no
  close) on the master calendar **up to and including the cutoff**, nothing later. The runner
  slices it, so a fit cannot see a value dated after its cutoff, and every training pair
  ``(x_tau, y_{tau+h})`` has ``tau + h <= cutoff`` (rule L2) because the array ends there.
  That holds because every value sits on the row of the date it is observed. A window target
  stored on its origin's row (M1's ``forward_rv21``) would break it: re-check L2 for ``rv21`` in
  M3 (M2 review A).
* ``forecast(fit, y, origins, horizons)``: ``y`` runs to the fold's last origin, and the
  forecast at origin ``t`` may use ``y[:t+1]`` only (rule L1). Parameters stay frozen; only
  the inputs (``y_t``, the HAR averages, the ARIMA filter state) move with t. The truncation
  test (``tests/test_walkforward.py``) checks that nothing after t is used.

**Spaces and the back-transform (ADR 0001 4.3).** ``rw`` and the historical means work in
levels: their forecast is already a forecast of the mean of y. ``arima_log`` and ``har_log``
forecast ``x = ln y``: a mean ``mu`` and a forecast-error variance ``s2`` for ``x_{t+h}``. If
``x_{t+h}`` is normal given the information at t, ``y_{t+h}`` is lognormal and

* ``exp(mu)`` is its **median** (``yhat_naive``), which is below its mean;
* ``exp(mu + s2/2)`` is its **mean** (``yhat``, the lognormal correction).

``yhat`` is the forecast that is evaluated, stored with ``var_method = plugin`` and
``v_hat = (yhat/100)**2``. ``yhat_naive`` is stored beside it so the losses can show what the
correction does (the median is what MAE rewards).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ivf.eda import longest_gap_free_segment
from ivf.features import build

#: The 90% quantile of the standard normal: an 80% central interval is mu -/+ Z80*s.
Z80 = 1.2815515655446004


@dataclass
class Fit:
    """One fitted model at one cutoff. ``params`` and ``hyperparams`` are JSON-ready."""

    params: dict[str, Any]
    hyperparams: dict[str, Any]
    n_train: int
    train_first: int | None  # index in y of the first training value used (None: no training data)
    train_last_target: int | None  # index of the last value used as a target or a fit input
    converged: bool = True
    fallback: str = "none"  # none | previous_fold | rw_log
    loglik: float | None = None
    aic: float | None = None
    fit_seconds: float = 0.0
    # Milestone 3, rule L2 for a window target: the last training ORIGIN of a supervised pair whose
    # target is a window ending ``horizon`` sessions later (``har_rv``). The runner checks
    # ``train_last_origin + horizon <= cutoff``. None for models without such pairs.
    train_last_origin: int | None = None
    # Milestone 4: one entry per ensemble member (``lstm_log``'s seeds: seed, epochs_run, seconds). The runner
    # writes one ``fit`` row per member (``member`` 1..K); None for a single deterministic fit (``member`` 0).
    members: list[dict[str, Any]] | None = None


@dataclass
class Forecast:
    """Forecasts at the origins passed to ``forecast``, one entry per origin."""

    yhat: np.ndarray  # the evaluated forecast: the conditional mean of y (vol points)
    yhat_naive: np.ndarray  # exp(mu) for log models (the median); equal to yhat otherwise
    mu_log: np.ndarray | None = None
    s2_log: np.ndarray | None = None
    pi80_lo: np.ndarray | None = None
    pi80_hi: np.ndarray | None = None
    # Milestone 3. ``v_hat``: a ``native`` model's own variance forecast E_t[v] (yhat = 100*sqrt(v_hat));
    # None means plugin, v_hat = (yhat/100)**2. ``floored``: True where a forecast was raised to a floor.
    v_hat: np.ndarray | None = None
    floored: np.ndarray | None = None
    # Milestone 4: each member's own yhat, shape (K, n), for the ``forecast_member`` table; None for one model.
    members: np.ndarray | None = None


def lognormal_forecast(mu: np.ndarray, s2: np.ndarray) -> Forecast:
    """The back-transform of a normal forecast of ``ln y``: mean, median and the 80% interval.

    Worked number: mu = ln 20 = 2.995732, s2 = 0.04 gives the median 20 and the mean
    20*exp(0.02) = 20.404027; the interval is 20*exp(-/+1.281552*0.2) = 15.477 .. 25.845.
    """
    mu = np.asarray(mu, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    s = np.sqrt(s2)
    return Forecast(
        yhat=np.exp(mu + s2 / 2),
        yhat_naive=np.exp(mu),
        mu_log=mu,
        s2_log=s2,
        pi80_lo=np.exp(mu - Z80 * s),
        pi80_hi=np.exp(mu + Z80 * s),
    )


def _level_forecast(values: np.ndarray) -> Forecast:
    values = np.asarray(values, dtype=float)
    return Forecast(yhat=values, yhat_naive=values.copy())


def _last_finite(y: np.ndarray) -> int | None:
    idx = np.flatnonzero(~np.isnan(y))
    return int(idx[-1]) if idx.size else None


def _first_finite(y: np.ndarray) -> int | None:
    idx = np.flatnonzero(~np.isnan(y))
    return int(idx[0]) if idx.size else None


class RandomWalk:
    """``yhat = y_t`` at every horizon: no parameters, so nothing to fit."""

    model_id = "rw"
    fit_space = "level"

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        return Fit(params={}, hyperparams={}, n_train=0, train_first=None, train_last_target=None)

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        return {h: _level_forecast(y[origins]) for h in horizons}


class HistoricalMean:
    """The mean of the levels dated <= cutoff, frozen for the fold (data-model.md, ``hist_mean``).

    ``window=None`` is the expanding mean from ``train_start``; ``window=W`` uses the last W
    sessions of the calendar up to the cutoff. Missing closes in the window are left out (gap
    policy G3: the fitters drop null training rows), and ``n_train`` counts what was used.
    """

    fit_space = "level"

    def __init__(self, model_id: str, window: int | None):
        self.model_id = model_id
        self.window = window

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        start = 0 if self.window is None else max(0, len(y) - self.window)
        sample = y[start:]
        present = ~np.isnan(sample)
        if not present.any():
            raise ValueError("no training value in the window")
        first = start + int(np.flatnonzero(present)[0])
        return Fit(
            params={"mean": float(sample[present].mean())},
            hyperparams={"window": self.window},
            n_train=int(present.sum()),
            train_first=first,
            train_last_target=_last_finite(y),
        )

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        return {h: _level_forecast(np.full(len(origins), fit.params["mean"])) for h in horizons}


def har_design(x: np.ndarray) -> np.ndarray:
    """Rows ``[1, x_t, mean(x_{t-4..t}), mean(x_{t-21..t})]`` (Corsi 2009 with ADR 0001's windows).

    The windows are ``build.HAR_WEEK`` (5) and ``build.HAR_MONTH`` (22), the ones the feature
    table's ``har_w`` and ``har_m`` use. Trailing only (rule L1); a window with a gap is NaN (G2).
    """
    return np.column_stack(
        [
            np.ones(len(x)),
            x,
            build.trailing(x, build.HAR_WEEK, "mean"),
            build.trailing(x, build.HAR_MONTH, "mean"),
        ]
    )


class Har:
    """HAR on ln y, one OLS per horizon (direct): ``x_{t+h} = b0 + b_d x_t + b_w xw_t + b_m xm_t + e``.

    ``s2`` is the residual variance ``RSS/(n - 4)``: the variance of the h-step error, used for
    the lognormal correction. It is unconditional (the same at every origin), and the h > 1
    residuals overlap (MA(h-1)), which leaves ``s2`` consistent but its standard error larger.
    """

    model_id = "har_log"
    fit_space = "log_level"
    COEFS = ("b0", "b_d", "b_w", "b_m")

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        x = np.log(y)
        design = har_design(x)
        params: dict[str, Any] = {}
        n_min, first_used, last_target = None, None, None
        for h in horizons:
            if len(x) <= h:
                raise ValueError(f"no training pair for h={h}")
            rows_x, target = design[: len(x) - h], x[h:]  # pair (tau, tau + h), tau + h <= cutoff
            keep = np.isfinite(rows_x).all(axis=1) & np.isfinite(target)
            n = int(keep.sum())
            if n <= len(self.COEFS):
                raise ValueError(f"only {n} complete HAR pairs for h={h}")
            beta, _, rank, _ = np.linalg.lstsq(rows_x[keep], target[keep], rcond=None)
            if rank < len(self.COEFS):
                raise ValueError(f"HAR design is rank {rank} for h={h}")
            resid = target[keep] - rows_x[keep] @ beta
            s2 = float(resid @ resid / (n - len(self.COEFS)))
            params[f"h{h}"] = {**dict(zip(self.COEFS, map(float, beta))), "s2": s2, "n": n}
            used = np.flatnonzero(keep)
            first = int(used[0]) - (build.HAR_MONTH - 1)  # the monthly window reaches 21 sessions back
            first_used = first if first_used is None else min(first_used, first)
            last = int(used[-1]) + h
            last_target = last if last_target is None else max(last_target, last)
            n_min = n if n_min is None else min(n_min, n)
        return Fit(
            params=params,
            hyperparams={"week": build.HAR_WEEK, "month": build.HAR_MONTH, "estimator": "OLS"},
            n_train=int(n_min),
            train_first=first_used,
            train_last_target=last_target,
        )

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        design = har_design(np.log(y))[origins]
        out = {}
        for h in horizons:
            p = fit.params[f"h{h}"]
            beta = np.array([p[c] for c in self.COEFS])
            out[h] = lognormal_forecast(design @ beta, np.full(len(origins), p["s2"]))
        return out


def kpss_p_value(x: np.ndarray) -> tuple[float, int]:
    """KPSS (level stationarity, ``nlags="auto"``) on the longest gap-free run of ``x`` (gap policy G2).

    Returns the p-value and the run length. statsmodels interpolates p inside [0.01, 0.10] and
    clips outside it (with a warning that is silenced here: the clipped value still decides
    which side of 5% the statistic is on).
    """
    from statsmodels.tsa.stattools import kpss

    start, stop = longest_gap_free_segment(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p, _, _ = kpss(x[start:stop], regression="c", nlags="auto")
    return float(p), stop - start


@dataclass
class _ArimaState:
    order: tuple[int, int, int]
    trend: str
    values: list[float] = field(default_factory=list)


class Arima:
    """ARIMA(p, d, q) on ln y, fitted by exact Gaussian MLE (statsmodels ``SARIMAX``, Kalman filter).

    Order selection at every cutoff, on the training data only (ADR 0001 4.6, "in-sample criteria
    are part of fitting"):

    1. ``d`` (``d_rule``). **Default ``"d0"``: d = 0**, an ARMA(p, q) with a constant (the mean
       of ln y) on the level of ln y, so the forecast reverts to the mean. The unit-root case is
       the random walk's (``rw``). ``"kpss"`` (an option, not the default) is the
       Hyndman-Khandakar rule: d = 1 if KPSS rejects level stationarity at ``kpss_alpha`` on the
       training sample's longest gap-free run (G2), else 0. Amended M2, review A: KPSS was the
       default, and it chose d = 1 on stationary AR(1) series with phi = 0.98 on 17 to 52% of
       seeds at fold sample sizes (n = 2,570 to 8,700), which made ``arima_log`` a random walk
       with no mean reversion (+17% RMSE at h = 21 on the synthetic law). AIC cannot choose d:
       the likelihoods of x and of its differences are of different data.
    2. ``(p, q)``: the smallest AIC over ``0..max_p`` x ``0..max_q`` with that d, among fits that
       converged with a finite AIC; a tie goes to the earlier (smaller) order. A constant when
       d = 0; none when d = 1 (a drift would trend log vol for ever).

    Each candidate starts from its own previous-fold estimate when there is one (a warm start; the
    optimum is the same, the search is shorter). **Fallback** (ADR 0001 4.5): if no candidate
    converges, the previous fold's model is kept (``fallback = previous_fold``); at the first fold
    it is ARIMA(0,1,0), whose one parameter has a closed form (``fallback = rw_log``).

    NaN in the training data (core gaps in burn-in, G3; gaps in the other indices, G5) are
    handled by the Kalman filter as missing observations: nothing is filled or bridged.
    Forecasts are iterated: ``mu`` and ``s2`` for ``x_{t+h}`` are the filter's h-step prediction
    from the state at t, with the parameters frozen (``s2`` ignores parameter uncertainty).
    """

    model_id = "arima_log"
    fit_space = "log_level"

    D_RULES = ("d0", "kpss")

    def __init__(self, max_p: int = 2, max_q: int = 2, d_rule: str = "d0", kpss_alpha: float = 0.05):
        if d_rule not in self.D_RULES:
            raise ValueError(f"d_rule must be one of {self.D_RULES}, not {d_rule!r}")
        self.max_p, self.max_q, self.d_rule, self.kpss_alpha = max_p, max_q, d_rule, kpss_alpha

    @staticmethod
    def _trend(d: int) -> str:
        return "c" if d == 0 else "n"

    @staticmethod
    def _model(x: np.ndarray, order, trend):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(x, order=order, trend=trend)

    def _fit_one(self, x: np.ndarray, order, start_params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(x, order, self._trend(order[1])).fit(disp=False, start_params=start_params)
        converged = bool(res.mle_retvals.get("converged", False)) if res.mle_retvals else False
        return res, converged

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        started = time.perf_counter()
        first = _first_finite(y)
        if first is None:
            raise ValueError("no training value")
        x = np.log(y[first:])
        if self.d_rule == "kpss":
            kpss_p, kpss_n = kpss_p_value(x)
            d = 1 if kpss_p < self.kpss_alpha else 0
        else:
            kpss_p, kpss_n, d = None, None, 0
        warm = {}
        if previous is not None and previous.params.get("arima_candidates"):
            warm = {tuple(c["order"]): c["values"] for c in previous.params["arima_candidates"] if c["values"]}
        candidates, best = [], None
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                order = (p, d, q)
                try:
                    res, converged = self._fit_one(x, order, warm.get(order))
                    aic, values = float(res.aic), [float(v) for v in res.params]
                    names = list(res.model.param_names)
                    loglik = float(res.llf)
                except Exception as exc:  # a failed candidate is recorded and skipped, never fatal
                    candidates.append({"order": list(order), "aic": None, "converged": False, "error": type(exc).__name__, "values": []})
                    continue
                ok = converged and math.isfinite(aic)
                candidates.append({"order": list(order), "aic": aic if math.isfinite(aic) else None, "converged": converged, "values": values})
                if ok and (best is None or aic < best[0]):
                    best = (aic, order, names, values, loglik)
        hyper = {
            "max_p": self.max_p, "max_q": self.max_q, "d_rule": self.d_rule, "kpss_alpha": self.kpss_alpha,
            "kpss_p": kpss_p, "kpss_n": kpss_n, "d": d, "criterion": "aic", "trend_if_d0": "c", "trend_if_d1": "n",
        }
        last = _last_finite(y)
        n_train = int(np.isfinite(x).sum())
        if best is not None:
            aic, order, names, values, loglik = best
            return Fit(
                params={"order": list(order), "trend": self._trend(order[1]), "names": names, "values": values, "arima_candidates": candidates},
                hyperparams=hyper, n_train=n_train, train_first=first, train_last_target=last,
                converged=True, loglik=loglik, aic=aic, fit_seconds=time.perf_counter() - started,
            )
        if previous is not None:
            kept = dict(previous.params, arima_candidates=candidates)
            return Fit(
                params=kept, hyperparams=hyper, n_train=n_train, train_first=first, train_last_target=last,
                converged=False, fallback="previous_fold", fit_seconds=time.perf_counter() - started,
            )
        dx = np.diff(x)
        sigma2 = float(np.nanmean(dx**2))  # the MLE of a driftless random walk's step variance
        return Fit(
            params={"order": [0, 1, 0], "trend": "n", "names": ["sigma2"], "values": [sigma2], "arima_candidates": candidates},
            hyperparams=hyper, n_train=n_train, train_first=first, train_last_target=last,
            converged=False, fallback="rw_log", fit_seconds=time.perf_counter() - started,
        )

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        first = fit.train_first
        h_max = max(horizons)
        # The series from the first training value to the fold's last origin, then h_max missing
        # values, so every origin's h-step prediction falls inside the sample. A missing value
        # leaves the filter state unchanged, so the padding cannot move a prediction.
        x = np.concatenate([np.log(y[first:]), np.full(h_max, np.nan)])
        order = tuple(fit.params["order"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(x, order, fit.params["trend"]).filter(np.asarray(fit.params["values"]))
            mu = {h: np.empty(len(origins)) for h in horizons}
            s2 = {h: np.empty(len(origins)) for h in horizons}
            for i, t in enumerate(origins):
                # dynamic=True: the prediction for t+1 uses data <= t, and later steps use the
                # model's own predictions, never the data (iterated multi-step, ADR 0001 4.5).
                k = int(t) - first
                pred = res.get_prediction(start=k + 1, end=k + h_max, dynamic=True)
                mean, var = np.asarray(pred.predicted_mean), np.asarray(pred.var_pred_mean)
                for h in horizons:
                    mu[h][i], s2[h][i] = mean[h - 1], var[h - 1]
        return {h: lognormal_forecast(mu[h], s2[h]) for h in horizons}


def build_models(settings, model_ids=None) -> list:
    """The baseline objects named in ``configs/models.yaml``, in registry order."""
    window = settings.hist_window
    every = {
        m.model_id: m
        for m in (
            RandomWalk(),
            HistoricalMean("hist_mean", None),
            HistoricalMean(f"hist_mean_{window}", window),
            Arima(settings.max_p, settings.max_q, settings.d_rule, settings.kpss_alpha),
            Har(),
        )
    }
    wanted = list(every) if model_ids is None else list(model_ids)
    unknown = [m for m in wanted if m not in every]
    if unknown:
        raise ValueError(f"unknown model id(s) {unknown}; known: {sorted(every)}")
    return [every[m] for m in wanted]
