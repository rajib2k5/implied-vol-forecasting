"""The walk-forward runner shared by every model (ADR 0001 4.5), dev period only in milestone 2.

**Folds** are a pure function of the calendar: cutoffs ``c_k = idx(holdout_start) + fold_len*k``
with ``c_k >= idx(dev_start)`` (``calendar.first_cutoff``), origins ``[c_k, c_{k+1})``. They are
computed once from 𝒯 and passed in, so a run on truncated data uses the same cutoffs (4.7).
``fold_id`` counts folds from the first dev cutoff (0, 1, ...); holdout folds continue the count.

**What a run may read.** Milestone 2 scores the dev period only, and the holdout is untouched:
``pre_holdout_inputs`` is M1's ``eda_inputs`` (sessions and interim rows strictly before
``holdout_start``), and ``run_symbol`` refuses a date on or after it (``HoldoutRowRefused``).
The calendar dates past the last dev session are used only to *date* targets (``target_date``),
never to read a value. A forecast whose target falls on or after ``holdout_start`` is produced
and stored, with no target value, and is not scored: from M2's side it is unobserved, exactly as
a target past ``data_end`` is in a full run (ADR 0001 4.5, "Evaluable origin").

**Per fold and model:** ``fit`` on ``y[:cutoff+1]`` (so rule L2 holds by construction:
``train_last_target <= cutoff``, as long as each value sits on the row of the date it is observed;
a window target stored on its origin's row would not, so M3's ``rv21`` needs its own L2 test), then ``forecast`` at the fold's origins with ``y`` up to the
fold's last origin. A fit that raises keeps the previous fold's fit (``fallback = previous_fold``,
4.5); with no previous fold it is an error.

**Same rows (4.5).** Every model forecasts at exactly the same origins. An origin is forecastable
for a symbol when its close and the 22 closes ``t-21..t`` are present (the HAR monthly window;
gap policy G2), and a fold is fitted for a symbol only when at least ``min_train_sessions``
closes are dated <= its cutoff. Both rules depend on the data, never on the model, and both are
counted in the run summary. For VIX in dev, G3 guarantees every origin is forecastable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ivf.data import calendar as calendar_mod
from ivf.data.process import HoldoutRowRefused, eda_inputs
from ivf.features import build
from ivf.models.baselines import Fit


@dataclass(frozen=True)
class Fold:
    fold_id: int
    cutoff_idx: int  # index in the master calendar
    last_origin_idx: int
    period: str  # dev | holdout


def fold_schedule(dates: list[date], dev_start: date, holdout_start: date, fold_len: int, data_end: date) -> list[Fold]:
    """Every fold of 𝒯 (``dates``, the master calendar), dev then holdout. Dates only, no data.

    The last dev fold ends on the session before ``holdout_start`` because the cutoffs are
    anchored on it; the last holdout fold is partial and ends on ``data_end``.
    """
    index = {d: i for i, d in enumerate(dates)}
    first = index[calendar_mod.first_cutoff(dates, dev_start, holdout_start, fold_len)]
    i_holdout, i_end = index[holdout_start], index[data_end]
    folds = []
    for k, cutoff in enumerate(range(first, i_end + 1, fold_len)):
        period = "dev" if cutoff < i_holdout else "holdout"
        last = min(cutoff + fold_len - 1, i_end)
        if period == "dev" and last >= i_holdout:  # cannot happen when anchored on holdout_start
            raise AssertionError("a dev fold straddles holdout_start")
        folds.append(Fold(k, cutoff, last, period))
    return folds


def fold_table(folds: list[Fold], dates: list[date]) -> pd.DataFrame:
    """The ``fold`` table (data-model.md section 4). The first origin equals the cutoff."""
    return pd.DataFrame(
        {
            "fold_id": pd.array([f.fold_id for f in folds], dtype="int32"),
            "cutoff_date": [dates[f.cutoff_idx] for f in folds],
            "last_origin": [dates[f.last_origin_idx] for f in folds],
            "period": [f.period for f in folds],
            "n_sessions": pd.array([f.last_origin_idx - f.cutoff_idx + 1 for f in folds], dtype="int16"),
        }
    )


def calendar_with_folds(cal: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """M1's calendar with ``fold_id`` filled for every origin session (null in burn-in and past data_end)."""
    fold_id = pd.array([None] * len(cal), dtype="Int32")
    for f in folds:
        fold_id[f.cutoff_idx : f.last_origin_idx + 1] = f.fold_id
    return cal.assign(fold_id=fold_id)


def pre_holdout_inputs(cal: pd.DataFrame, iv: pd.DataFrame, price: pd.DataFrame, holdout_start: date):
    """``(dates, periods, iv, price)`` strictly before ``holdout_start``: M1's ``eda_inputs``, reused."""
    return eda_inputs(cal, iv, price, holdout_start)


def forecastable(y: np.ndarray) -> np.ndarray:
    """True where the close at t and the 22 closes t-21..t are present (the HAR month; G2)."""
    return np.isfinite(build.trailing(np.log(y), build.HAR_MONTH, "mean"))


@dataclass
class SymbolRun:
    forecasts: pd.DataFrame
    fits: pd.DataFrame
    folds_fitted: int
    folds_skipped_min_train: int
    origins_not_forecastable: int
    seconds: dict


def run_symbol(
    symbol: str,
    target_id: str,
    y: np.ndarray,
    dates: list[date],
    calendar_dates: list[date],
    folds: list[Fold],
    models: list,
    horizons: tuple[int, ...],
    holdout_start: date,
    min_train_sessions: int,
) -> SymbolRun:
    """Fit and forecast every model on one symbol's closes ``y`` (aligned on ``dates``).

    ``dates`` are the sessions the run may read (all before ``holdout_start``), a prefix of
    ``calendar_dates`` (𝒯). ``y`` may be shorter than ``dates`` (a truncated run): origins past
    its end are not forecast. ``folds`` are the dev folds from ``fold_schedule``.
    """
    if any(d >= holdout_start for d in dates):
        raise HoldoutRowRefused(f"the baselines read nothing on or after holdout_start {holdout_start}")
    if list(calendar_dates[: len(dates)]) != list(dates):
        raise ValueError("dates must be a prefix of the master calendar")
    if len(y) > len(dates):
        raise ValueError("y is longer than the sessions it may read")
    y = np.asarray(y, dtype=float)
    n = len(y)
    ok = forecastable(y)
    rows, fit_rows = [], []
    previous: dict[str, Fit | None] = {m.model_id: None for m in models}
    seconds = {m.model_id: 0.0 for m in models}
    fitted = skipped = not_forecastable = 0
    for fold in folds:
        if fold.period != "dev":
            raise HoldoutRowRefused("milestone 2 runs dev folds only")
        c = fold.cutoff_idx
        if c >= n:
            break
        if int(np.isfinite(y[: c + 1]).sum()) < min_train_sessions:
            skipped += 1
            continue
        last = min(fold.last_origin_idx, n - 1)
        span = np.arange(c, last + 1)
        origins = span[ok[span]]
        not_forecastable += len(span) - len(origins)
        fitted += 1
        for model in models:
            started = time.perf_counter()
            try:
                fit = model.fit(y[: c + 1], horizons, previous[model.model_id])
            except Exception as exc:
                if previous[model.model_id] is None:
                    raise RuntimeError(f"{model.model_id} on {symbol}: the first fit failed ({exc}) and there is no previous fold") from exc
                kept = previous[model.model_id]
                fit = Fit(**{**kept.__dict__, "fallback": "previous_fold", "converged": False})
            fit.fit_seconds = time.perf_counter() - started
            if fit.train_last_target is not None and fit.train_last_target > c:
                raise AssertionError(f"{model.model_id}: a training value is dated after the cutoff (rule L2)")
            previous[model.model_id] = fit
            fit_rows.append(_fit_row(model.model_id, target_id, fold, fit, dates))
            if len(origins):
                out = model.forecast(fit, y[: last + 1], origins, horizons)
                for h in horizons:
                    rows.append(_forecast_frame(model, target_id, symbol, fold, fit, out[h], origins, h, y, dates, calendar_dates))
            seconds[model.model_id] += time.perf_counter() - started
    forecasts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(forecasts):
        check_same_rows(forecasts, [m.model_id for m in models])
    return SymbolRun(forecasts, pd.DataFrame(fit_rows), fitted, skipped, not_forecastable, seconds)


def _fit_row(model_id: str, target_id: str, fold: Fold, fit: Fit, dates: list[date]) -> dict:
    import json

    def day(i):
        return None if i is None else dates[i]

    return {
        "model_id": model_id,
        "target_id": target_id,
        "fold_id": fold.fold_id,
        "member": 0,
        "cutoff_date": dates[fold.cutoff_idx],
        "n_train": fit.n_train,
        "train_first_date": day(fit.train_first),
        "train_last_target_date": day(fit.train_last_target),
        "converged": fit.converged,
        "fallback": fit.fallback,
        "params_json": json.dumps(fit.params, sort_keys=True, allow_nan=False),
        "hyperparams_json": json.dumps(fit.hyperparams, sort_keys=True, allow_nan=False),
        "seed": None,
        "device": "cpu",
        "fit_seconds": fit.fit_seconds,
        "loglik": fit.loglik,
        "aic": fit.aic,
        "epochs_run": None,
    }


def _forecast_frame(model, target_id, symbol, fold, fit, fc, origins, h, y, dates, calendar_dates) -> pd.DataFrame:
    yhat = np.asarray(fc.yhat, dtype=float)
    if not (np.isfinite(yhat).all() and (yhat > 0).all()):
        raise ValueError(f"{model.model_id} h={h}: a forecast is missing or not positive (a missing row is an error, ADR 0001 4.5)")
    target_idx = origins + h
    observed = target_idx < len(y)
    y_target = np.full(len(origins), np.nan)
    y_target[observed] = y[target_idx[observed]]

    def col(values):
        return np.full(len(origins), np.nan) if values is None else np.asarray(values, dtype=float)

    return pd.DataFrame(
        {
            "model_id": model.model_id,
            "target_id": target_id,
            "symbol": symbol,
            "origin_date": [dates[i] for i in origins],
            "horizon": np.int16(h),
            "target_date": [calendar_dates[i] for i in target_idx],
            "fold_id": np.int32(fold.fold_id),
            "cutoff_date": dates[fold.cutoff_idx],
            "yhat": yhat,
            "v_hat": (yhat / 100) ** 2,
            "var_method": "plugin",
            "yhat_naive": np.asarray(fc.yhat_naive, dtype=float),
            "mu_log": col(fc.mu_log),
            "s2_log": col(fc.s2_log),
            "pi80_lo": col(fc.pi80_lo),
            "pi80_hi": col(fc.pi80_hi),
            "n_members": np.int16(1),
            "fallback": fit.fallback,
            "y": y_target,  # the target if observed before holdout_start; not written to the store
        }
    )


def check_same_rows(forecasts: pd.DataFrame, expected_models=None) -> None:
    """Every expected model has the same origins per (target, horizon), and nothing twice (ADR 0001 4.5).

    ``expected_models`` defaults to every model in the frame. Amended M2, review A: the first
    version compared only the models present in each group, so a model missing every row of one
    horizon passed.
    """
    key = ["model_id", "target_id", "origin_date", "horizon"]
    if forecasts.duplicated(key).any():
        raise AssertionError("a (model, target, origin, horizon) appears twice")
    expected = set(forecasts["model_id"]) if expected_models is None else set(expected_models)
    for (target_id, h), group in forecasts.groupby(["target_id", "horizon"]):
        sets = {m: frozenset(g["origin_date"]) for m, g in group.groupby("model_id")}
        missing = sorted(expected - set(sets))
        if missing:
            raise AssertionError(f"{target_id} h={h}: models {missing} have no row (same-rows rule)")
        reference = next(iter(sets.values()))
        different = [m for m, s in sets.items() if s != reference]
        if different:
            raise AssertionError(f"{target_id} h={h}: models {different} forecast at other origins (same-rows rule)")


def scored(forecasts: pd.DataFrame, holdout_start: date) -> pd.DataFrame:
    """The rows M2 scores: dev origins whose target is observed and dated before holdout_start."""
    keep = forecasts["y"].notna() & (forecasts["target_date"] < holdout_start) & (forecasts["origin_date"] < holdout_start)
    return forecasts[keep]


def metric_table(forecasts: pd.DataFrame, holdout_start: date, min_cell_n: int, log_models: set[str]) -> pd.DataFrame:
    """The ``metric`` table for dev, scheme ``all``. ``forecast = mean`` is the evaluated ``yhat``;
    ``forecast = median`` scores ``yhat_naive`` (``exp(mu)``) for the log-space models, to show
    which back-transform each loss rewards. Regime cells arrive in M5."""
    from ivf.evaluation import losses

    rows = []
    for (model_id, target_id, h), g in scored(forecasts, holdout_start).groupby(["model_id", "target_id", "horizon"]):
        variants = {"mean": g["yhat"]}
        if model_id in log_models:
            variants["median"] = g["yhat_naive"]
        for name, yhat in variants.items():
            rows.append(
                {
                    "model_id": model_id, "target_id": target_id, "horizon": int(h), "period": "dev",
                    "scheme": "all", "regime": "all", "forecast": name,
                    **losses.loss_row(g["y"].to_numpy(), yhat.to_numpy(), min_cell_n),
                }
            )
    return pd.DataFrame(rows)
