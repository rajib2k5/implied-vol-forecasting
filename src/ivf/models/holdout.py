"""Holdout-fold forecasts for the one look (milestone 6; ADR 0001 4.5-4.6, amended M6).

Called ONLY by ``python -m ivf.backtest run --period holdout`` after its preflight passed and its look
record exists. It never runs on its own.

WHAT IS FROZEN AND WHAT IS NOT (ADR 0001 4.6, "In-sample criteria are part of fitting"). Frozen: every
model's specification and hyper-parameters (``configs/models.yaml``; which models exist on each track,
read from the dev forecast store). Re-estimated at every cutoff, holdout folds included, from data
dated <= that cutoff: AIC order choice, MLE parameters, scaler moments, the LSTM's weights on its own
``refit_every`` schedule, and (in M5's report) the ``*_cal`` multipliers. The refits continue the dev
walk-forward: same fold schedule, same runners (``walkforward.run_symbol`` / ``run_rv21`` with
``allow_holdout=True``), same L2 assertions.

WARM-UP (amended M6, review B; default taken 2026-10-29, owner can override). A fit may depend on the
previous fold's fit (the ``previous_fold`` fallback, ARIMA and GARCH warm starts, the LSTM's frozen
network between retrains). So the runner re-runs the dev chain before the holdout and discards its
output after checking it:

* every model except the LSTM: from the FIRST dev fold, so every warm start is the dev chain's own
  (review B measured the first version, which restarted at the last dev fold: ARIMA holdout forecasts
  moved by up to 0.03 vol points). It costs about 200 s on the full synthetic model set;
* ``lstm_log``: from the dev fold whose cutoff is the last dev ``trained_cutoff`` in the stored ``fit``
  table (``lstm_warmup``), so its retrain schedule continues exactly where dev left it; a full LSTM
  chain would cost hours (ADR 0001 O2).

**Reproduction check** (``check_reproduces_dev``): every recomputed dev-fold forecast must equal the
stored dev forecast for the same (model, target, horizon, origin) within ``REPRO_ATOL`` = 1e-12 vol
points, with the same set of rows, before any holdout forecast is used. A difference means the code,
the data or a library changed since the dev run, and the holdout forecasts would not be the frozen
models' forecasts.

**Fallbacks** (``check_no_fallback``): a holdout fit with ``fallback`` other than ``none`` (a failed fit
replaced by the previous fold's, e.g. an LSTM worker that died) fails the stage loudly; the dev fits
had none. The stage can be resumed (``python -m ivf.backtest run --period holdout --resume``).

OUTPUT. ``processed/run_id=<id>/period=holdout/{forecast, fit, forecast_member}.parquet``: the dev
store's tables and columns, holdout folds only. The dev tables are not touched.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ivf.evaluation import walkforward
from ivf.features import build
from ivf.models import baselines, registry

FOLDER = "period=holdout"
#: Largest allowed |yhat| difference, in vol points, between a recomputed dev forecast and the stored one.
REPRO_ATOL = 1e-12


class HoldoutStageError(RuntimeError):
    """The holdout-forecast stage cannot produce the frozen models' forecasts (reproduction or fallback)."""


def build_track_models(target_id: str, model_ids: list[str]) -> list:
    """The non-LSTM model objects of one track, in ``model_ids`` order; raises ValueError for an id with no
    implementation (the preflight calls this before the look record is written)."""
    from ivf.models import garch as garch_mod
    from ivf.models import rv21

    defs, settings = registry.load_models()
    gsettings = registry.load_garch_settings()
    wanted = [m for m in model_ids if m != "lstm_log"]
    if target_id == rv21.TARGET_ID:
        every = {m.model_id: m for m in rv21.build_rv21_models(None, gsettings.dists)}
    elif target_id == "vix_level":
        every = {m.model_id: m for m in baselines.build_models(settings)}
        every["arima_garch_log"] = garch_mod.ArimaGarch(gsettings.max_p, gsettings.dists, gsettings.d)
    else:
        raise ValueError(f"no holdout runner for target {target_id}")
    missing = [m for m in wanted if m not in every]
    if missing:
        raise ValueError(f"{target_id}: no implementation for {missing}")
    if "lstm_log" in model_ids:
        registry.load_lstm_settings()  # raises if the frozen LSTM settings are unreadable
    return [every[m] for m in wanted]


def needed_models(dev_forecast: pd.DataFrame, targets=("vix_level", "rv21")) -> dict[str, list[str]]:
    """Every model the dev store holds on each target: the look forecasts all of them (same-rows rule)."""
    return {t: sorted(set(dev_forecast.loc[dev_forecast.target_id == t, "model_id"])) for t in targets
            if (dev_forecast.target_id == t).any()}


def lstm_start_cutoff(dev_fit: pd.DataFrame, target_id: str) -> int:
    """The index of the last dev full retrain of ``lstm_log`` on ``target_id`` (its ``trained_cutoff``)."""
    rows = dev_fit[(dev_fit.model_id == "lstm_log") & (dev_fit.target_id == target_id)]
    if rows.empty:
        raise ValueError(f"lstm_log on {target_id}: no dev fit rows to continue from")
    last = rows[rows.cutoff_date == rows.cutoff_date.max()].iloc[0]
    return int(json.loads(last.params_json)["trained_cutoff"])


def lstm_warmup(dev_folds: list, start: int) -> list:
    """The dev folds from the one whose cutoff is the LSTM's last dev retrain (``start``) onwards."""
    warm = [f for f in dev_folds if f.cutoff_idx >= start]
    if not warm or warm[0].cutoff_idx != start:
        raise HoldoutStageError(f"no dev fold has the LSTM's last trained cutoff {start}")
    return warm


def check_reproduces_dev(recomputed: pd.DataFrame, stored: pd.DataFrame, atol: float = REPRO_ATOL) -> int:
    """Raise unless every recomputed dev-fold row matches the stored dev forecast (rows and ``yhat``); returns the rows compared."""
    key = ["model_id", "target_id", "horizon", "origin_date"]
    pairs = set(zip(recomputed.model_id, recomputed.target_id))
    mine = stored[[pair in pairs for pair in zip(stored.model_id, stored.target_id)]]
    mine = mine[mine.fold_id.isin(set(recomputed.fold_id))]
    joined = recomputed[key + ["yhat"]].merge(mine[key + ["yhat"]], on=key, how="outer", suffixes=("_new", "_dev"), indicator=True)
    unmatched = int((joined["_merge"] != "both").sum())
    worst = float((joined.yhat_new - joined.yhat_dev).abs().max()) if len(joined) else 0.0
    if unmatched or not worst <= atol:
        raise HoldoutStageError(f"the recomputed dev folds do not reproduce the stored dev forecasts: {unmatched} rows "
                                f"unmatched, largest |yhat difference| {worst:.3g} vol points (allowed {atol})")
    return len(joined)


def check_no_fallback(fit: pd.DataFrame) -> None:
    bad = fit[fit["fallback"] != "none"]
    if len(bad):
        kinds = bad.groupby(["model_id", "target_id", "fallback"]).size()
        raise HoldoutStageError("a holdout fit fell back instead of fitting: "
                                + ", ".join(f"{m}/{t} {k} x{n}" for (m, t, k), n in kinds.items()))


def run_holdout_forecasts(folder, proto, iv: pd.DataFrame, price: pd.DataFrame, needed: dict[str, list[str]],
                          dev_fit: pd.DataFrame | None, workers: int = 5, out=None,
                          dev_forecast: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Refit every needed model on the warm-up + holdout folds, check the warm-up reproduces the stored
    dev forecasts and that no holdout fit fell back, then write and return the holdout tables."""
    from ivf.data import calendar
    from ivf.models import cli as models_cli
    from ivf.models import lstm, rv21

    out = out or sys.stdout
    clock = time.perf_counter()
    data_end = proto.require_data_end()
    _, settings = registry.load_models()
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    dates = [d for d in calendar_dates if d <= data_end]
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, data_end)
    dev_folds = [f for f in folds if f.period == "dev"]
    hold_folds = [f for f in folds if f.period == "holdout"]
    hold_ids = {f.fold_id for f in hold_folds}
    vix = build.align(dates, iv, "symbol", "VIX")
    inputs = rv21.Rv21Inputs(build.log_returns(build.align(dates, price, "ticker", rv21.SYMBOL)), vix)
    forecasts, fits = [], []
    stored = dev_forecast if dev_forecast is not None else pd.read_parquet(folder / "forecast.parquet")
    compared = 0

    def run(target_id, models, run_folds, allow=True):
        if target_id == "vix_level":
            return walkforward.run_symbol("VIX", "vix_level", vix, dates, calendar_dates, run_folds, models, proto.horizons,
                                          proto.holdout_start, settings.min_train_sessions, allow_holdout=allow)
        return walkforward.run_rv21(inputs, dates, calendar_dates, run_folds, models, proto.holdout_start,
                                    settings.min_train_sessions, allow_holdout=allow)

    for target_id, ids in needed.items():
        models = build_track_models(target_id, ids)
        if models:
            r = run(target_id, models, [*dev_folds, *hold_folds])
            compared += check_reproduces_dev(r.forecasts[~r.forecasts.fold_id.isin(hold_ids)], stored)
            forecasts.append(r.forecasts.drop(columns=[c for c in ("floored",) if c in r.forecasts]))
            fits.append(r.fits)
            print(f"  {target_id}: {len(models)} models from the first dev fold, {len(hold_folds)} holdout folds, "
                  f"{int(r.forecasts.fold_id.isin(hold_ids).sum())} holdout forecast rows", file=out)
        if "lstm_log" in ids:
            warm = lstm_warmup(dev_folds, lstm_start_cutoff(dev_fit, target_id))
            lsettings = registry.load_lstm_settings()
            model = (lstm.LstmLog if target_id == "vix_level" else lstm.LstmRv)(lsettings, proto.document["seed"], dates, workers)
            try:
                r = run(target_id, [model], [*warm, *hold_folds])
            finally:
                model.close()
            compared += check_reproduces_dev(r.forecasts[~r.forecasts.fold_id.isin(hold_ids)], stored)
            forecasts.append(r.forecasts.drop(columns=[c for c in ("floored",) if c in r.forecasts]))
            fits.append(r.fits)
            print(f"  {target_id}: lstm_log, {len(warm)} warm-up folds from the last dev retrain, {len(hold_folds)} holdout folds", file=out)
    forecast = pd.concat(forecasts, ignore_index=True)
    fit = pd.concat(fits, ignore_index=True)
    forecast, fit = forecast[forecast.fold_id.isin(hold_ids)].reset_index(drop=True), fit[fit.fold_id.isin(hold_ids)]
    check_no_fallback(fit)
    print(f"dev reproduction: {compared} recomputed dev-fold forecasts equal the stored ones (|diff| <= {REPRO_ATOL} vol points); "
          "0 holdout fits fell back", file=out)
    for target_id, group in forecast.groupby("target_id"):
        walkforward.check_same_rows(group, needed[target_id])
    tables = {"forecast": forecast[models_cli.FORECAST_COLUMNS], "fit": fit, "forecast_member": models_cli.member_table(forecast)}
    run_id = folder.name.split("=", 1)[1]
    for name, frame in tables.items():
        framed = frame.assign(run_id=run_id)[["run_id", *[c for c in frame.columns if c != "run_id"]]]
        models_cli._write(framed, folder / FOLDER / f"{name}.parquet")
    print(f"holdout forecasts: {len(forecast)} rows on {len(hold_folds)} holdout folds, origins "
          f"{forecast.origin_date.min()} .. {forecast.origin_date.max()}; {time.perf_counter() - clock:.1f} s", file=out)
    return tables


__all__ = ["FOLDER", "REPRO_ATOL", "HoldoutStageError", "build_track_models", "needed_models", "lstm_start_cutoff",
           "lstm_warmup", "check_reproduces_dev", "check_no_fallback", "run_holdout_forecasts"]
