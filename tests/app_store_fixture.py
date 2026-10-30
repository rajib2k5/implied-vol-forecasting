"""A tiny, hand-made model store for the app-export tests (``ivf.app_results.Store``).

Every number is chosen so the expected payload values can be worked out by hand in the tests:
the losses are round numbers with simple ratios, the actuals are k/8 steps (exact in binary),
and the sessions are 30 weekdays ending the Friday before ``holdout_start``. Nothing here
computes what the code under test computes; it only lays out the input tables.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from ivf.app_results import Store
from ivf.backtest.strategy import StrategyDef
from ivf.models.registry import ModelDef

HOLDOUT = date(2024, 7, 1)
SESSIONS = [d.date() for d in pd.bdate_range(end="2024-06-28", periods=30)]  # 2024-05-20 .. 2024-06-28
RACE = 3


def model(model_id, targets, milestone=2, retired=False, family="naive"):
    return ModelDef(model_id=model_id, label=model_id.upper(), family=family, fit_space="level", multistep="direct",
                    targets=tuple(targets), milestone=milestone, retired=retired)


MODELS = (
    model("rw", ["vix_level", "rv21"]),
    model("har_log", ["vix_level"], family="har"),
    model("garch", ["rv21"], milestone=3, family="garch"),
    model("old_model", ["vix_level"], retired=True),  # retired: never drawn
    model("future_model", ["vix_level"], milestone=5),  # after milestone 4: never drawn
)


def actual(k: int) -> float:
    """y at session k: 20 + k/8 vol points (exact in binary, so rounding to 3 decimals changes nothing)."""
    return 20 + k / 8


def scored_rows() -> pd.DataFrame:
    rows = []
    for target_id, h, models in (("vix_level", 1, ("rw", "har_log", "future_model")), ("vix_level", 5, ("rw", "har_log")),
                                 ("vix_level", 21, ("rw", "har_log")), ("rv21", 21, ("rw", "garch"))):
        for k in range(len(SESSIONS) - h):
            for model_id in models:
                offset = {"rw": 0.0, "har_log": 0.5, "garch": -0.25, "future_model": 9.0}[model_id]
                rows.append({"target_id": target_id, "horizon": h, "model_id": model_id, "origin_date": SESSIONS[k],
                             "target_date": SESSIONS[k + h], "yhat": actual(k) + offset, "y": actual(k + h)})
    return pd.DataFrame(rows)


def metric_rows() -> pd.DataFrame:
    rows = []
    for target_id, h in (("vix_level", 1), ("vix_level", 5), ("vix_level", 21), ("rv21", 21)):
        for model_id, rmse, qlike in (("rw", 4.0, 0.2), ("har_log", 2.0, 0.05), ("garch", 3.0, 0.1)):
            if (target_id, model_id) in (("rv21", "har_log"), ("vix_level", "garch")):
                continue  # har_log is a vix_level model, garch an rv21 one
            base = {"model_id": model_id, "target_id": target_id, "horizon": h, "period": "dev", "n": 100, "suppressed": False,
                    "bias": 0.123456, "rmse": rmse, "mae": rmse / 2, "mape": 0.0512345, "qlike": qlike}
            rows.append({**base, "scheme": "all", "regime": "all", "forecast": "mean"})
            rows.append({**base, "scheme": "vix_level", "regime": "high", "forecast": "mean", "rmse": 99.0})  # a regime cell: not shown
            rows.append({**base, "scheme": "all", "regime": "all", "forecast": "median", "rmse": 77.0})  # the median variant: not shown
    return pd.DataFrame(rows)


def dm_rows() -> pd.DataFrame:
    rows = []
    for h in (1, 5, 21):
        for loss in ("sq_error", "qlike"):
            base = {"target_id": "vix_level", "horizon": h, "period": "dev", "scheme": "all", "regime": "all", "loss": loss,
                    "model_a": "har_log", "model_b": "rw", "family": "pairs", "n": 100, "dm_stat": -3.0, "p_value": 0.003,
                    "p_holm": 0.06 if (h, loss) == (21, "qlike") else 0.009, "lower_loss": "har_log", "method": "fixed_b",
                    "primary": True}  # one pair whose raw p (0.003) passes but whose Holm p (0.06) does not
            rows.append(base)
            rows.append({**base, "method": "rectangular", "primary": False, "p_holm": 0.5})  # not the primary method
            rows.append({**base, "family": "calibration", "model_b": "har_log+cal", "p_holm": 0.6})  # calibration family
            rows.append({**base, "scheme": "vix_level", "regime": "high", "p_holm": 0.7})  # a regime cell
    return pd.DataFrame(rows)


def mz_rows() -> pd.DataFrame:
    rows = []
    for target_id, h, model_id in (("vix_level", 1, "rw"), ("vix_level", 1, "har_log"), ("rv21", 21, "garch")):
        base = {"model_id": model_id, "target_id": target_id, "horizon": h, "period": "dev", "space": "level", "forecast": "mean",
                "n": 100, "alpha": 0.123456, "alpha_se": 0.05, "beta": 0.987654, "beta_se": 0.01, "r2": 0.9, "wald_p": 0.42,
                "method": "hansen_hodrick", "primary": True}
        rows.append(base)
        rows.append({**base, "method": "newey_west", "primary": False, "wald_p": 0.01})
        rows.append({**base, "space": "variance", "wald_p": 0.02})
        rows.append({**base, "forecast": "mean_cal", "wald_p": 0.03})
    return pd.DataFrame(rows)


STRATEGIES = (
    StrategyDef("fvix_rv21_b2", "forecast", "rv21 band 2 (PRIMARY)", False, track="rv21", band=2.0),
    StrategyDef("fvix_rv21_b0", "forecast", "rv21 band 0", False, track="rv21", band=0.0),
    StrategyDef("bench_long_vxx", "benchmark", "always long VXX", False, signal=1),
    StrategyDef("bench_long_svxy", "benchmark", "always long SVXY", False, signal=-1),
    StrategyDef("bench_cash", "benchmark", "cash", False, signal=0),
)
BACKTEST_DATES = SESSIONS[-3:]
EQUITY = {"fvix_rv21_b2": [1.0, 1.1, 0.99], "fvix_rv21_b0": [1.0, 0.5, 0.25], "bench_long_vxx": [1.0, 0.8, 0.4],
          "bench_long_svxy": [1.0, 1.2, 1.5], "bench_cash": [1.0, 1.0, 1.0]}


def daily_rows() -> pd.DataFrame:
    rows = []
    for sid, equity in EQUITY.items():
        peak = 0.0
        for d, e in zip(BACKTEST_DATES, equity):
            peak = max(peak, e)
            for cost in (0.0, 5.0):
                rows.append({"strategy_id": sid, "period": "dev", "date": d, "cost_bps": cost,
                             "equity": e if cost == 5.0 else e + 0.01, "drawdown": e / peak - 1})
    return pd.DataFrame(rows)


def summary_rows() -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(EQUITY):
        for cost in (0.0, 5.0):
            rows.append({"strategy_id": sid, "period": "dev", "cost_bps": cost, "n_days": 3, "ann_return": 0.1 * i - cost / 100,
                         "ann_vol": 0.2, "sharpe_ann": 0.5 * i - cost / 10, "psr0": 0.6, "dsr": 0.25, "n_trials": 2,
                         "max_drawdown": -0.1, "turnover_ann": 2.0})
    return pd.DataFrame(rows)


EDA = {
    "sample": {"first": "1990-01-02", "last": "2024-06-28", "holdout_start": "2024-07-01"},
    "clustering": {
        "r": {"n": 10000, "acf": [0.01, -0.02], "ljung_box_p": [0.3, 0.45]},
        "abs_r": {"n": 10000, "acf": [0.2, 0.15], "ljung_box_p": [0.0, 0.0]},
        "r2": {"n": 10000, "acf": [0.25, 0.123456], "ljung_box_p": [0.0, 1e-9]},
    },
    "persistence": {"log_vix_acf": [1.0, 0.98, 0.96], "n": 400},
    "half_life": {
        "pre_holdout": {"phi": 0.98, "phi_ci95": [0.97, 0.99], "half_life_sessions": 34.30961, "half_life_ci95": [22.7566, 68.9676], "n": 400},
        "dev": {"phi": 0.999, "phi_ci95": [0.99, 1.01], "half_life_sessions": 692.8, "half_life_ci95": [68.97, None], "n": 300},
    },
}


def tiny_store(**changes) -> Store:
    fields = dict(
        run_id="abcdef012345",
        seed=7,
        holdout_start=HOLDOUT,
        race_sessions=RACE,
        scored=scored_rows(),
        metric=metric_rows(),
        dm=dm_rows(),
        mz=mz_rows(),
        calibration=pd.DataFrame({"helps": [False, True, False]}),
        report_meta={"dm_primary": "fixed_b", "fixed_b": 0.1, "mz_primary": "hansen_hodrick"},
        daily=daily_rows(),
        summary=summary_rows(),
        backtest_meta={"cost_bps_default": 5.0, "models": {"rv21": "garch", "vix_level": "har_log"}},
        backtest_config=SimpleNamespace(primary="fvix_rv21_b2", strategies=STRATEGIES,
                                        tracks={"rv21": {"target_id": "rv21", "horizon": 21}, "vix_level": {"target_id": "vix_level", "horizon": 21}}),
        models=MODELS,
        eda=EDA,
    )
    fields.update(changes)
    return Store(**fields)


# --------------------------------------------------------------------------
# The same store on disk (polish review): for the command-line tests, which run the exporter in a
# subprocess and so cannot be handed the in-memory store. The interim tables come from the real
# synthetic generator (``python -m ivf.data --synthetic fetch`` and ``build-interim``, about 3 s);
# the model, evaluation and backtest tables are the hand-made ones above, on real sessions.
# --------------------------------------------------------------------------

DISK_RUN_ID = "0123456789ab"


def write_store(vol_dir) -> str:
    """Write a complete synthetic store under ``vol_dir`` (used as ``$VOL_DATA_DIR``); return its run id."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from ivf import protocol as protocol_mod
    from ivf.data import calendar

    vol_dir = Path(vol_dir)
    env = dict(os.environ, VOL_DATA_DIR=str(vol_dir))
    for step in ("fetch", "build-interim"):
        done = subprocess.run([sys.executable, "-m", "ivf.data", "--synthetic", step], env=env, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"ivf.data --synthetic {step} failed: {done.stderr}")
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=True))
    sessions = [d for d in calendar.master_calendar(proto)["date"] if d < proto.holdout_start][-len(SESSIONS):]
    by_bday = dict(zip(SESSIONS, sessions))  # the hand-made rows, moved onto real sessions
    forecast = scored_rows().drop(columns=["y"])
    forecast["origin_date"] = forecast["origin_date"].map(by_bday)
    forecast["target_date"] = forecast["target_date"].map(by_bday)
    processed = vol_dir / "implied-vol-forecasting" / "synthetic" / "processed"
    folder = processed / f"run_id={DISK_RUN_ID}"
    evaluation, backtest = folder / "evaluation" / "period=dev", folder / "backtest" / "period=dev"
    for path in (evaluation, backtest):
        path.mkdir(parents=True)
    forecast.to_parquet(folder / "forecast.parquet")
    metric_rows().to_parquet(evaluation / "metric.parquet")
    dm_rows().to_parquet(evaluation / "dm_test.parquet")
    mz_rows().to_parquet(evaluation / "mz_regression.parquet")
    pd.DataFrame({"helps": [False, True, False]}).to_parquet(evaluation / "calibration_summary.parquet")
    (evaluation / "report_meta.json").write_text(json.dumps(
        {"source_run_id": DISK_RUN_ID, "period": "dev", "dm_primary": "fixed_b", "fixed_b": 0.1, "mz_primary": "hansen_hodrick"}))
    daily_rows().to_parquet(backtest / "backtest_daily.parquet")
    summary_rows().to_parquet(backtest / "strategy_summary.parquet")
    (backtest / "backtest_meta.json").write_text(json.dumps(
        {"source_run_id": DISK_RUN_ID, "period": "dev", "cost_bps_default": 5.0, "models": {"rv21": "garch", "vix_level": "har_log"}}))
    pd.DataFrame({"run_id": [DISK_RUN_ID], "started_at": pd.to_datetime(["2026-10-29"], utc=True)}).to_parquet(processed / "runs.parquet")
    return DISK_RUN_ID
