"""Command line for the models (milestone 2: the baselines; milestone 3: the GARCH family; milestone 4: the LSTM).

    uv run python -m ivf.models baselines [--symbols VIX ...] [--models rw har_log ...] [--print-metrics]
    uv run python -m ivf.models garch [--models garch rw ...] [--print-metrics]
    uv run python -m ivf.models lstm [--workers 5] [--tracks vix_level rv21] [--print-metrics]
    uv run python -m ivf.models --synthetic baselines|garch|lstm

``garch`` (milestone 3) runs two tracks on the same folds: **rv21** (h = 21: ``rw``, ``har_rv``,
``vix_implied``, ``garch``, ``gjr``, ``egarch``) and **vix_level** on VIX (h = 1, 5, 21: M2's five
baselines and ``arima_garch_log``, so every VIX model is scored on the same rows in one run). Each
track's expected model list comes from ``configs/models.yaml`` (``registry.models_for``).

``lstm`` (milestone 4) runs ``lstm_log`` on both tracks with the same runner, then checks the same-rows
rule against the run's **stored** forecast table with the full expected model list (milestone <= 4), so
it must follow ``garch`` in the same code state (same run_id); it refuses to write otherwise. It adds
``forecast_member`` (one row per seed, data-model.md) and one ``fit`` row per seed.

Reads the interim tables (``ivf.data build-interim``), re-runs the data check against the frozen
``data_end`` and its freeze record exactly as ``eda`` does, and refuses to run if it fails. Then,
for each symbol of ``configs/models.yaml`` with data: every baseline, every dev fold, horizons
1, 5 and 21. **No row on or after holdout_start is read** (``walkforward.pre_holdout_inputs``).

Writes under ``ROOT/processed/run_id=<id>/`` (ADR 0001 D3; ``ROOT/synthetic`` with
``--synthetic``): ``run_input``, ``calendar`` (with ``fold_id``), ``fold``, ``fit``, ``forecast``
and ``metric``, plus this run's row in ``processed/runs.parquet``. Both commands share the run_id
(same code, protocol and inputs), so ``fit``, ``forecast`` and ``metric`` are **merged**: a command
replaces the rows of the (model, target) pairs it ran and keeps the others (amended M3; M2 replaced
the whole table). Every file goes through
``repo_guard.write_file`` (outside the repo). Nothing is committed: on real data every number
here is outside the publication set P (ADR 0001 D2). The loss values are printed only with
``--print-metrics``; the default output is counts and timings.

Exit codes: 0 success; 1 the data check failed; 2 bad command line; 3 missing or malformed
input, or the licensing guard refused the data root.
"""

from __future__ import annotations

import argparse
import io
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ivf
from ivf import config, contract, repo_guard
from ivf import protocol as protocol_mod
from ivf.data import build_interim, calendar, check, process
from ivf.evaluation import walkforward
from ivf.features import build
from ivf.models import baselines, registry

EXIT_OK, EXIT_FAILED, EXIT_INPUT = 0, 1, 3

#: Columns of the forecast table on disk (data-model.md section 4, amended M2), run_id first.
FORECAST_COLUMNS = [
    "model_id", "target_id", "symbol", "origin_date", "horizon", "target_date", "fold_id", "cutoff_date",
    "yhat", "v_hat", "var_method", "yhat_naive", "mu_log", "s2_log", "pi80_lo", "pi80_hi", "n_members", "fallback",
]


def _write(frame: pd.DataFrame, path) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    repo_guard.write_file(path, buffer.getvalue(), outside_repo=True)


MERGED_TABLES = ("fit", "forecast", "metric", "forecast_member")  # amended M4: + forecast_member


def _merge_write(frame: pd.DataFrame, path) -> None:
    """Replace the rows of the (model_id, target_id) pairs in ``frame``, keep every other row of the table."""
    if path.exists():
        old = pd.read_parquet(path)
        mine = set(zip(frame["model_id"], frame["target_id"]))
        keep = old[[pair not in mine for pair in zip(old["model_id"], old["target_id"])]]
        frame = pd.concat([keep, frame], ignore_index=True) if len(keep) else frame
    _write(frame, path)


def _write_tables(folder, run_id: str, tables: dict) -> None:
    for name, frame in tables.items():
        framed = frame.assign(run_id=run_id)[["run_id", *[c for c in frame.columns if c != "run_id"]]]
        (_merge_write if name in MERGED_TABLES else _write)(framed, folder / f"{name}.parquet")


def run_baselines(layout, *, synthetic: bool, symbols=None, model_ids=None, print_metrics: bool = False, out=None) -> int:
    out = out or sys.stdout
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    repo_guard.require_data_root_outside_repo(layout.root)
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    data_end = proto.require_data_end()
    defs, settings = registry.load_models()
    models = baselines.build_models(settings, model_ids)
    registered = {d.model_id for d in defs}
    missing = [m.model_id for m in models if m.model_id not in registered]
    if missing:
        print(f"error: {missing} not in configs/models.yaml", file=sys.stderr)
        return EXIT_INPUT
    log_models = {d.model_id for d in defs if d.fit_space == "log_level"}
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    result = check.frozen_check(proto, iv, price, report)  # the same refusal as eda (M1, review B)
    if not result.ok:
        for violation in result.policy_violations:
            print(f"error: {violation}", file=sys.stderr)
        return EXIT_FAILED
    git = contract.git_state()
    data_mode = "synthetic" if synthetic else "real"
    run_id = process.run_id_for(proto.protocol_hash, git, data_mode, [e["raw_sha256"] for e in report["files"]])
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, data_end)
    dev_folds = [f for f in folds if f.period == "dev"]
    # The holdout is untouched: sessions and interim rows strictly before holdout_start only.
    dates, _, iv_pre, _ = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    wanted = list(symbols) if symbols else list(settings.symbols)
    forecasts, fits = [], []
    print(f"baselines: {len(dev_folds)} dev folds, {len(dates)} sessions before holdout_start {proto.holdout_start}; "
          "no row on or after it is read", file=out)
    for symbol in wanted:
        y = build.align(dates, iv_pre, "symbol", symbol)
        if not np.isfinite(y).any():
            print(f"  {symbol}: no closes before holdout_start, skipped", file=out)
            continue
        target_id = registry.target_id_for(symbol)
        run = walkforward.run_symbol(
            symbol, target_id, y, dates, calendar_dates, dev_folds, models, proto.horizons,
            proto.holdout_start, settings.min_train_sessions,
        )
        forecasts.append(run.forecasts)
        fits.append(run.fits)
        n_origins = run.forecasts["origin_date"].nunique() if len(run.forecasts) else 0
        print(
            f"  {symbol} ({target_id}): {run.folds_fitted} folds fitted, {run.folds_skipped_min_train} skipped "
            f"(< {settings.min_train_sessions} closes), {n_origins} origins, {run.origins_not_forecastable} not forecastable "
            f"(gap in t-21..t), {len(run.forecasts)} forecast rows",
            file=out,
        )
        if len(run.fits):
            fallbacks = run.fits[run.fits["fallback"] != "none"].groupby(["model_id", "fallback"]).size()
            for (model_id, kind), count in fallbacks.items():
                print(f"    fallback {model_id}: {kind} x{count}", file=out)
        print("    seconds: " + ", ".join(f"{m} {s:.1f}" for m, s in run.seconds.items()), file=out)
    if not forecasts:
        print("error: no symbol had data", file=sys.stderr)
        return EXIT_INPUT
    forecast = pd.concat(forecasts, ignore_index=True)
    fit = pd.concat(fits, ignore_index=True)
    metric = walkforward.metric_table(forecast, proto.holdout_start, proto.document["min_cell_n"], log_models)
    n_scored = len(walkforward.scored(forecast, proto.holdout_start))
    tables = {
        "run_input": pd.DataFrame([{"raw_sha256": e["raw_sha256"], "source": e["source"], "symbol": e["symbol"]} for e in report["files"]]),
        "calendar": walkforward.calendar_with_folds(cal, folds),
        "fold": walkforward.fold_table(folds, calendar_dates),
        "fit": fit,
        "forecast": forecast[FORECAST_COLUMNS],
        "metric": metric,
    }
    folder = layout.processed / f"run_id={run_id}"
    _write_tables(folder, run_id, tables)
    process.upsert_run_row(
        layout,
        {
            "run_id": run_id, "protocol_hash": proto.protocol_hash, "data_mode": data_mode,
            "source_commit": git.commit, "source_dirty": git.dirty, "started_at": started,
            "finished_at": datetime.now(timezone.utc), "ivf_version": ivf.__version__,
            "python_version": sys.version.split()[0], "platform": platform.platform(),
        },
    )
    print(f"run_id {run_id} ({data_mode}); {len(forecast)} forecast rows, {n_scored} scored (dev, target before "
          f"holdout_start), {len(fit)} fit rows, {len(metric)} metric rows; {time.perf_counter() - clock:.1f} s", file=out)
    print(f"tables in {folder} (local only, never commit real numbers: ADR 0001 D2)", file=out)
    if print_metrics:
        shown = metric[["target_id", "horizon", "model_id", "forecast", "n", "rmse", "mae", "mape", "qlike"]]
        print(shown.sort_values(["target_id", "horizon", "forecast", "model_id"]).to_string(index=False), file=out)
    return EXIT_OK


def _run_row(run_id, proto, data_mode, git, started) -> dict:
    return {
        "run_id": run_id, "protocol_hash": proto.protocol_hash, "data_mode": data_mode,
        "source_commit": git.commit, "source_dirty": git.dirty, "started_at": started,
        "finished_at": datetime.now(timezone.utc), "ivf_version": ivf.__version__,
        "python_version": sys.version.split()[0], "platform": platform.platform(),
    }


def _fit_summary(fits: pd.DataFrame, out) -> None:
    """Counts only: fallbacks, the distribution each GARCH-family fit chose, and unusable candidates."""
    import json

    fallbacks = fits[fits["fallback"] != "none"].groupby("model_id").size()
    for model_id, count in fallbacks.items():
        print(f"    fallback {model_id}: previous_fold x{count}", file=out)
    for model_id, g in fits.groupby("model_id", sort=False):
        params = [json.loads(p) for p in g["params_json"]]
        if not params or "candidates" not in params[0]:
            continue
        dists = pd.Series([p["dist"] for p in params]).value_counts().to_dict()
        bad = sum(not c["ok"] for p in params for c in p["candidates"])
        total = sum(len(p["candidates"]) for p in params)
        extra = ""
        if model_id == "arima_garch_log":
            extra = ", AR order " + str(pd.Series([p["p"] for p in params]).value_counts().sort_index().to_dict())
        print(f"    {model_id}: {len(g)} fits, converged {int(g['converged'].sum())}, distribution {dists}{extra}; "
              f"{bad} of {total} candidates unusable", file=out)


def run_garch(layout, *, synthetic: bool, model_ids=None, print_metrics: bool = False, out=None) -> int:
    """Milestone 3: the rv21 track and the vix_level track on VIX, dev folds only (module docstring)."""
    from ivf.models import garch as garch_mod
    from ivf.models import rv21

    out = out or sys.stdout
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    repo_guard.require_data_root_outside_repo(layout.root)
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    data_end = proto.require_data_end()
    defs, settings = registry.load_models()
    gsettings = registry.load_garch_settings()
    rv_expected = registry.models_for(defs, rv21.TARGET_ID, 3)
    vix_expected = registry.models_for(defs, "vix_level", 3)
    if model_ids:
        unknown = sorted(set(model_ids) - set(rv_expected) - set(vix_expected))
        if unknown:
            print(f"error: {unknown} are not rv21 or vix_level models in configs/models.yaml", file=sys.stderr)
            return EXIT_INPUT
        rv_expected = [m for m in rv_expected if m in model_ids]
        vix_expected = [m for m in vix_expected if m in model_ids]
    rv_models = rv21.build_rv21_models(rv_expected, gsettings.dists)
    vix_every = {m.model_id: m for m in baselines.build_models(settings)}
    vix_every["arima_garch_log"] = garch_mod.ArimaGarch(gsettings.max_p, gsettings.dists, gsettings.d)
    vix_models = [vix_every[m] for m in vix_expected]
    if [m.model_id for m in rv_models] != rv_expected or [m.model_id for m in vix_models] != vix_expected:
        print("error: a registered rv21 or vix_level model has no implementation", file=sys.stderr)
        return EXIT_INPUT
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    result = check.frozen_check(proto, iv, price, report)
    if not result.ok:
        for violation in result.policy_violations:
            print(f"error: {violation}", file=sys.stderr)
        return EXIT_FAILED
    git = contract.git_state()
    data_mode = "synthetic" if synthetic else "real"
    run_id = process.run_id_for(proto.protocol_hash, git, data_mode, [e["raw_sha256"] for e in report["files"]])
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, data_end)
    dev_folds = [f for f in folds if f.period == "dev"]
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    print(f"garch: {len(dev_folds)} dev folds, {len(dates)} sessions before holdout_start {proto.holdout_start}; "
          "no row on or after it is read", file=out)
    forecasts, fits, log_models = [], [], {d.model_id for d in defs if d.fit_space == "log_level"}
    vix = build.align(dates, iv_pre, "symbol", "VIX")
    if rv_models:
        inputs = rv21.Rv21Inputs(build.log_returns(build.align(dates, price_pre, "ticker", rv21.SYMBOL)), vix)
        run = walkforward.run_rv21(inputs, dates, calendar_dates, dev_folds, rv_models, proto.holdout_start, settings.min_train_sessions)
        n_origins = run.forecasts["origin_date"].nunique() if len(run.forecasts) else 0
        floored = int(run.forecasts["floored"].sum()) if len(run.forecasts) else 0
        print(f"  rv21 (h = 21): {run.folds_fitted} folds fitted, {run.folds_skipped_min_train} skipped, {n_origins} origins, "
              f"{run.origins_not_forecastable} not forecastable, {len(run.forecasts)} forecast rows; har_rv floored {floored}", file=out)
        _fit_summary(run.fits, out)
        print("    seconds: " + ", ".join(f"{m} {s:.1f}" for m, s in run.seconds.items()), file=out)
        forecasts.append(run.forecasts.drop(columns="floored"))
        fits.append(run.fits)
    if vix_models:
        run = walkforward.run_symbol("VIX", "vix_level", vix, dates, calendar_dates, dev_folds, vix_models, proto.horizons,
                                     proto.holdout_start, settings.min_train_sessions)
        n_origins = run.forecasts["origin_date"].nunique() if len(run.forecasts) else 0
        print(f"  vix_level (h = {', '.join(map(str, proto.horizons))}): {run.folds_fitted} folds fitted, {n_origins} origins, "
              f"{run.origins_not_forecastable} not forecastable, {len(run.forecasts)} forecast rows", file=out)
        _fit_summary(run.fits, out)
        print("    seconds: " + ", ".join(f"{m} {s:.1f}" for m, s in run.seconds.items()), file=out)
        forecasts.append(run.forecasts)
        fits.append(run.fits)
    if not forecasts:
        print("error: no model to run", file=sys.stderr)
        return EXIT_INPUT
    forecast = pd.concat(forecasts, ignore_index=True)
    fit = pd.concat(fits, ignore_index=True)
    metric = walkforward.metric_table(forecast, proto.holdout_start, proto.document["min_cell_n"], log_models)
    n_scored = len(walkforward.scored(forecast, proto.holdout_start))
    late = int((fit["train_last_target_date"].dropna() > fit.loc[fit["train_last_target_date"].notna(), "cutoff_date"]).sum())
    tables = {
        "run_input": pd.DataFrame([{"raw_sha256": e["raw_sha256"], "source": e["source"], "symbol": e["symbol"]} for e in report["files"]]),
        "calendar": walkforward.calendar_with_folds(cal, folds),
        "fold": walkforward.fold_table(folds, calendar_dates),
        "fit": fit,
        "forecast": forecast[FORECAST_COLUMNS],
        "metric": metric,
    }
    folder = layout.processed / f"run_id={run_id}"
    _write_tables(folder, run_id, tables)
    process.upsert_run_row(layout, _run_row(run_id, proto, data_mode, git, started))
    print(f"run_id {run_id} ({data_mode}); {len(forecast)} forecast rows, {n_scored} scored (dev, target before "
          f"holdout_start), {len(fit)} fit rows ({late} with train_last_target_date after the cutoff), {len(metric)} metric rows; "
          f"{time.perf_counter() - clock:.1f} s", file=out)
    print(f"tables merged into {folder} (local only, never commit real numbers: ADR 0001 D2)", file=out)
    if print_metrics:
        shown = metric[["target_id", "horizon", "model_id", "forecast", "n", "rmse", "mae", "mape", "qlike", "bias"]]
        print(shown.sort_values(["target_id", "horizon", "forecast", "model_id"]).to_string(index=False), file=out)
    return EXIT_OK


def member_table(forecast: pd.DataFrame) -> pd.DataFrame:
    """``forecast_member`` (data-model.md, M4): one row per (forecast row, member), member 1..K, plug-in ``v_hat``."""
    rows = forecast[forecast["member_yhat"].notna()] if "member_yhat" in forecast else forecast.iloc[:0]
    key = ["model_id", "target_id", "origin_date", "horizon", "fold_id"]
    if not len(rows):
        return pd.DataFrame(columns=[*key, "member", "yhat", "v_hat"])
    k = len(rows["member_yhat"].iloc[0])
    out = rows.loc[rows.index.repeat(k), key].reset_index(drop=True)
    out["member"] = np.tile(np.arange(1, k + 1, dtype=np.int16), len(rows))
    out["yhat"] = np.concatenate(rows["member_yhat"].to_numpy())
    out["v_hat"] = (out["yhat"] / 100) ** 2
    return out


def _lstm_summary(name: str, run, fits: pd.DataFrame, model, out) -> None:
    """Counts and timings only (D2): refits, epochs, the seed spread as a relative size, no loss."""
    import json

    n_origins = run.forecasts["origin_date"].nunique() if len(run.forecasts) else 0
    params = fits.drop_duplicates("fold_id")["params_json"].map(json.loads)
    refits = int(params.map(lambda p: p["refit"]).sum())
    trained = fits[[json.loads(p)["refit"] for p in fits["params_json"]]]
    best = [m["best_epoch"] for p in params if p["refit"] for m in p["members"]]
    fc = run.forecasts
    spread = np.stack(fc["member_yhat"].to_numpy()).std(axis=1) / fc["yhat"].to_numpy() if len(fc) else np.array([np.nan])
    print(f"  {name}: {run.folds_fitted} folds fitted ({refits} trained, {run.folds_fitted - refits} frozen), {n_origins} origins, "
          f"{run.origins_not_forecastable} not forecastable, {len(fc)} forecast rows, {fc['n_members'].iloc[0] if len(fc) else 0} members", file=out)
    print(f"    epochs_run min/median/max {trained['epochs_run'].min()}/{trained['epochs_run'].median():g}/{trained['epochs_run'].max()}; "
          f"best epoch min/median/max {min(best)}/{float(np.median(best)):g}/{max(best)}; "
          f"seed spread sd/yhat median {np.median(spread):.4f}, max {spread.max():.4f}; "
          f"training {model.train_seconds:.1f} s (all members), {run.seconds['lstm_log']:.1f} s fit+forecast", file=out)


def run_lstm(layout, *, synthetic: bool, tracks=("vix_level", "rv21"), workers: int = 5, print_metrics: bool = False, out=None) -> int:
    """Milestone 4: ``lstm_log`` on the vix_level and rv21 tracks, dev folds only (module docstring)."""
    from ivf.models import lstm, rv21

    out = out or sys.stdout
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    repo_guard.require_data_root_outside_repo(layout.root)
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    data_end = proto.require_data_end()
    defs, settings = registry.load_models()
    lsettings = registry.load_lstm_settings()
    expected = {t: registry.models_for(defs, t, 4) for t in ("vix_level", rv21.TARGET_ID)}
    if any("lstm_log" not in m for m in expected.values()):
        print("error: lstm_log is not registered for vix_level and rv21 in configs/models.yaml", file=sys.stderr)
        return EXIT_INPUT
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    result = check.frozen_check(proto, iv, price, report)
    if not result.ok:
        for violation in result.policy_violations:
            print(f"error: {violation}", file=sys.stderr)
        return EXIT_FAILED
    git = contract.git_state()
    data_mode = "synthetic" if synthetic else "real"
    run_id = process.run_id_for(proto.protocol_hash, git, data_mode, [e["raw_sha256"] for e in report["files"]])
    folder = layout.processed / f"run_id={run_id}"
    stored = folder / "forecast.parquet"
    if not stored.exists():
        print(f"error: no forecast table for run_id {run_id}; run `garch` first in this code state (same-rows rule)", file=sys.stderr)
        return EXIT_INPUT
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, data_end)
    dev_folds = [f for f in folds if f.period == "dev"]
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    print(f"lstm: {len(dev_folds)} dev folds, {len(dates)} sessions before holdout_start {proto.holdout_start}; "
          f"no row on or after it is read; {lsettings.members} members (seeds from protocol.seed, ADR 0001 D5), hidden {lsettings.hidden}, refit_every "
          f"{lsettings.refit_every} sessions, {lstm.THREADS} thread per member, {workers} workers, torch {lstm.torch.__version__}", file=out)
    vix = build.align(dates, iv_pre, "symbol", "VIX")
    forecasts, fits = [], []
    for track in tracks:
        model = (lstm.LstmLog if track == "vix_level" else lstm.LstmRv)(lsettings, proto.document["seed"], dates, workers)
        try:
            if track == "vix_level":
                run = walkforward.run_symbol("VIX", "vix_level", vix, dates, calendar_dates, dev_folds, [model], proto.horizons,
                                             proto.holdout_start, settings.min_train_sessions)
            else:
                inputs = rv21.Rv21Inputs(build.log_returns(build.align(dates, price_pre, "ticker", rv21.SYMBOL)), vix)
                run = walkforward.run_rv21(inputs, dates, calendar_dates, dev_folds, [model], proto.holdout_start, settings.min_train_sessions)
        finally:
            model.close()
        _lstm_summary(f"{track} (h = {', '.join(map(str, sorted(run.forecasts['horizon'].unique())))})", run, run.fits, model, out)
        forecasts.append(run.forecasts)
        fits.append(run.fits)
    forecast = pd.concat(forecasts, ignore_index=True)
    fit = pd.concat(fits, ignore_index=True)
    # The same-rows rule against what is stored: every expected model, the LSTM included, on the same origins.
    mine = set(zip(forecast["model_id"], forecast["target_id"]))
    old = pd.read_parquet(stored, columns=["model_id", "target_id", "origin_date", "horizon"])
    old = old[[pair not in mine for pair in zip(old["model_id"], old["target_id"])]]
    combined = pd.concat([old, forecast[old.columns]], ignore_index=True)
    try:
        for target_id in sorted(set(forecast["target_id"])):
            walkforward.check_same_rows(combined[combined.target_id == target_id], expected[target_id])
    except AssertionError as exc:
        print(f"error: {exc}; run `garch` first in this code state", file=sys.stderr)
        return EXIT_FAILED
    log_models = {d.model_id for d in defs if d.fit_space == "log_level"}
    metric = walkforward.metric_table(forecast, proto.holdout_start, proto.document["min_cell_n"], log_models)
    n_scored = len(walkforward.scored(forecast, proto.holdout_start))
    late = int((fit["train_last_target_date"] > fit["cutoff_date"]).sum())
    members = member_table(forecast)
    _write_tables(folder, run_id, {"fit": fit, "forecast": forecast[FORECAST_COLUMNS], "metric": metric, "forecast_member": members})
    process.upsert_run_row(layout, _run_row(run_id, proto, data_mode, git, started))
    print(f"run_id {run_id} ({data_mode}); {len(forecast)} forecast rows, {n_scored} scored, {len(members)} member rows, "
          f"{len(fit)} fit rows ({late} with train_last_target_date after the cutoff), same rows as {expected}; "
          f"{time.perf_counter() - clock:.1f} s", file=out)
    print(f"tables merged into {folder} (local only, never commit real numbers: ADR 0001 D2)", file=out)
    if print_metrics:
        stored_metric = pd.read_parquet(folder / "metric.parquet")
        shown = stored_metric[stored_metric.target_id.isin(set(forecast["target_id"]))]
        shown = shown[["target_id", "horizon", "model_id", "forecast", "n", "rmse", "mae", "mape", "qlike", "bias"]]
        print(shown.sort_values(["target_id", "horizon", "forecast", "rmse"]).to_string(index=False), file=out)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ivf.models", description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", action="store_true", help="use ROOT/synthetic and the synthetic protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("baselines", help="walk-forward baselines on the dev period (milestone 2)")
    p.add_argument("--symbols", nargs="+", help="only these symbols (default: configs/models.yaml)")
    p.add_argument("--models", nargs="+", help="only these model ids (default: every baseline)")
    p.add_argument("--print-metrics", action="store_true", help="print the dev losses (local only)")
    g = commands.add_parser("garch", help="the GARCH family: rv21 track and ARIMA-GARCH on VIX, dev period (milestone 3)")
    g.add_argument("--models", nargs="+", help="only these model ids (default: every rv21 and vix_level model)")
    g.add_argument("--print-metrics", action="store_true", help="print the dev losses (local only)")
    m = commands.add_parser("lstm", help="the LSTM on the vix_level and rv21 tracks, dev period (milestone 4); after garch")
    m.add_argument("--tracks", nargs="+", choices=["vix_level", "rv21"], default=["vix_level", "rv21"])
    m.add_argument("--workers", type=int, default=5, help="processes training the seeds in parallel (results do not depend on it)")
    m.add_argument("--print-metrics", action="store_true", help="print the dev losses of every model on these targets (local only)")
    args = parser.parse_args(argv)
    layout = config.data_layout(synthetic=args.synthetic)
    try:
        repo_guard.require_data_root_outside_repo(layout.root)
        if args.command == "lstm":
            return run_lstm(layout, synthetic=args.synthetic, tracks=tuple(args.tracks), workers=args.workers, print_metrics=args.print_metrics)
        if args.command == "garch":
            return run_garch(layout, synthetic=args.synthetic, model_ids=args.models, print_metrics=args.print_metrics)
        return run_baselines(layout, synthetic=args.synthetic, symbols=args.symbols, model_ids=args.models, print_metrics=args.print_metrics)
    except (protocol_mod.ProtocolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except repo_guard.RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return repo_guard.EXIT_REFUSED
