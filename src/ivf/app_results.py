"""The result views' JSON payloads, reshaped from a model store (ADR 0001 D1: no new number here).

``tools/export_app_data.py --mode synthetic`` calls this to write four files next to
``manifest.json`` in ``app/public/data/synthetic/``:

* ``model_race.json``: forecast vs actual for the milestone 2-4 models, one panel per
  (target, horizon), over the last ``race_context_sessions`` dev origins (protocol.synthetic.yaml);
* ``evaluation.json``: M5's dev losses (RMSE, QLIKE, ...), the primary Diebold-Mariano pairs
  (fixed-b, Holm, ``lower_loss``), the primary Mincer-Zarnowitz rows and the calibration count;
* ``equity_curves.json``: M6's dev backtest, the primary rule against always-long VXX, always-long
  SVXY and cash, with PSR/DSR and the cost-sensitivity table;
* ``eda.json``: M1's volatility-clustering ACFs and the AR(1) half-life of ln VIX.

Where the numbers come from. Every value is read from the tables the pipeline already wrote under
``ROOT/processed/run_id=<id>/`` (``forecast``, ``evaluation/period=dev/*``,
``backtest/period=dev/*``), or recomputed by the very functions the pipeline uses (the actuals by
``ivf.evaluation.report.attach_targets``; the EDA statistics by ``ivf.data.process.eda_core``, the
part of the M1 EDA report that the view draws). This module only selects, reshapes and rounds.

The dev period only. Nothing dated on or after ``holdout_start`` is read into a payload: the race
keeps rows whose origin AND target are before it (the dev purge of ADR 0001 4.6), the evaluation and
backtest tables are their ``period=dev`` folders, and the EDA inputs are cut by ``eda_inputs``. No
holdout look is needed or taken (``holdout_looks`` stays 0).

Which store. ``choose_store_run`` picks the newest run (``runs.parquet``, ``started_at``) that has
both M5's and M6's dev outputs made from that run's own forecasts. Without one, the exporter stops
with exit 4 and names the steps to run (``StoreUnavailable``).

The real-data hook (data policy, ADR 0001 D2). The builders take any ``DataLayout``. Under the
current policy the real versions of these files are series or regime-free aggregates of vendor
data and may never enter the repository; ``--mode real`` still writes only ``manifest.json`` and
``term_structure.json``, outside the repo. Wiring these builders into ``generate_real`` (they would
go to ``ROOT/app-data/real/`` like the others) is the owner's change once the policy is decided.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The (target, horizon) panels of the model race, in display order (data-model.md 5.3).
PANELS: tuple[tuple[str, int], ...] = (("vix_level", 1), ("vix_level", 5), ("vix_level", 21), ("rv21", 21))
#: Models drawn in the race: the milestone 2-4 registry (``models_for(..., MAX_MILESTONE)``).
MAX_MILESTONE = 4
#: Benchmarks drawn beside the primary rule in the backtest view (configs/backtest.yaml ids).
BENCHMARKS = ("bench_long_vxx", "bench_long_svxy", "bench_cash")
#: Rounding at export (data-model.md section 5). Loss ratios to the random walk keep 6 decimals
#: (amended in the polish review): at 4, ARIMA and ARIMA-GARCH tie at 0.995 (VIX h = 1) and 0.9762
#: (h = 5), and a table could not tell which is lower.
DEC_LEVEL, DEC_STAT, DEC_MAPE, DEC_QLIKE, DEC_EQUITY, DEC_RATIO = 3, 4, 5, 6, 6, 6


class StoreUnavailable(Exception):
    """The store has no run with the dev evaluation and the dev backtest; the message names the steps."""


STEPS = (
    "`uv run python tools/build_synthetic_store.py` (one command: python -m ivf.data --synthetic fetch, build-interim, "
    "check, eda; python -m ivf.models --synthetic baselines, garch, lstm; python -m ivf.evaluation --synthetic report; "
    "python -m ivf.backtest --synthetic prepare, run)"
)


def r(value: Any, decimals: int) -> float | None:
    """Round a finite number; NaN, None and +-inf become None (JSON null)."""
    if value is None:
        return None
    x = float(value)
    return round(x, decimals) if math.isfinite(x) else None


def _rounded(values, decimals: int) -> list[float | None]:
    return [r(v, decimals) for v in values]


# --------------------------------------------------------------------------
# Choosing and reading the store
# --------------------------------------------------------------------------


def _dev_outputs(folder: Path) -> tuple[Path, Path]:
    return folder / "evaluation" / "period=dev" / "report_meta.json", folder / "backtest" / "period=dev" / "backtest_meta.json"


def choose_store_run(processed: Path) -> str:
    """The newest run whose dev evaluation AND dev backtest were both made from its own forecasts."""
    runs_path = processed / "runs.parquet"
    runs = pd.read_parquet(runs_path).sort_values("started_at", ascending=False) if runs_path.exists() else pd.DataFrame(columns=["run_id"])
    for run_id in runs["run_id"]:
        report_meta, backtest_meta = _dev_outputs(processed / f"run_id={run_id}")
        if report_meta.is_file() and backtest_meta.is_file():
            metas = [json.loads(p.read_text(encoding="utf-8")) for p in (report_meta, backtest_meta)]
            if all(m.get("source_run_id") == run_id and m.get("period") == "dev" for m in metas):
                return str(run_id)
    raise StoreUnavailable(f"no run under {processed} has both the dev evaluation and the dev backtest; run: {STEPS}")


@dataclass(frozen=True)
class Store:
    """Everything the four payloads read, already cut to the dev period."""

    run_id: str
    seed: int  # protocol.synthetic.yaml's root seed (ADR 0001 D5), for the notice
    holdout_start: Any  # datetime.date
    race_sessions: int
    scored: pd.DataFrame  # forecast rows with y attached, dev-scored rows only
    metric: pd.DataFrame
    dm: pd.DataFrame
    mz: pd.DataFrame
    calibration: pd.DataFrame
    report_meta: dict[str, Any]
    daily: pd.DataFrame
    summary: pd.DataFrame
    backtest_meta: dict[str, Any]
    backtest_config: Any  # ivf.backtest.strategy.BacktestConfig
    models: tuple  # registry ModelDefs
    eda: dict[str, Any]


def load_store(layout, protocol) -> Store:
    """Read the chosen run's dev tables and recompute the actuals and the EDA statistics."""
    from ivf.data import build_interim, calendar, process
    from ivf.evaluation import report, walkforward
    from ivf.features import build
    from ivf.models import registry

    run_id = choose_store_run(layout.processed)
    folder = layout.processed / f"run_id={run_id}"
    holdout_start = protocol.holdout_start
    iv, price, _ = build_interim.read_interim(layout.root)
    cal = calendar.master_calendar(protocol)
    dates, _, iv_read, price_read = walkforward.pre_holdout_inputs(cal, iv, price, holdout_start)
    forecasts = pd.read_parquet(folder / "forecast.parquet")
    scored = report.scored_rows(report.attach_targets(forecasts, build.build_target_values(dates, iv_read, price_read)), holdout_start, "dev")
    eda_dates, eda_periods, eda_iv, eda_price = process.eda_inputs(cal, iv, price, holdout_start)
    eda = process.eda_core(eda_dates, eda_periods, build.build_features(eda_dates, eda_iv, eda_price), eda_iv, holdout_start)
    evaluation, backtest = folder / "evaluation" / "period=dev", folder / "backtest" / "period=dev"
    report_meta, backtest_meta = (json.loads(p.read_text(encoding="utf-8")) for p in _dev_outputs(folder))
    from ivf.backtest import strategy as strategy_mod

    defs, _ = registry.load_models()
    return Store(
        run_id=run_id,
        seed=int(protocol.document["seed"]),
        holdout_start=holdout_start,
        race_sessions=int(protocol.document["race_context_sessions"]),
        scored=scored,
        metric=pd.read_parquet(evaluation / "metric.parquet"),
        dm=pd.read_parquet(evaluation / "dm_test.parquet"),
        mz=pd.read_parquet(evaluation / "mz_regression.parquet"),
        calibration=pd.read_parquet(evaluation / "calibration_summary.parquet"),
        report_meta=report_meta,
        daily=pd.read_parquet(backtest / "backtest_daily.parquet"),
        summary=pd.read_parquet(backtest / "strategy_summary.parquet"),
        backtest_meta=backtest_meta,
        backtest_config=strategy_mod.load_config(),
        models=tuple(defs),
        eda=eda,
    )


def race_models(models, target_id: str) -> list[str]:
    """Registry order, milestone <= 4, not retired: the order the app assigns colour slots in."""
    return [d.model_id for d in models if target_id in d.targets and not d.retired and d.milestone <= MAX_MILESTONE]


# --------------------------------------------------------------------------
# model_race.json (data-model.md 5.3)
# --------------------------------------------------------------------------


def _all_dev(frame: pd.DataFrame) -> pd.DataFrame:
    """The whole-period rows of an M5 table: period dev, scheme and regime ``all``."""
    keep = frame["period"].eq("dev")
    for column in ("scheme", "regime"):
        if column in frame:
            keep &= frame[column].eq("all")
    return frame[keep]


def model_race_payload(store: Store) -> dict[str, Any]:
    """The last ``race_sessions`` dev origins of each panel, every milestone 2-4 model, and M5's MZ rows.

    ``calendar`` runs from the earliest origin shown to ``holdout_start`` inclusive, so
    ``holdout_start_idx`` is its last index and no row of any panel reaches it.
    """
    s = store.scored
    late = (s["origin_date"] >= store.holdout_start) | (s["target_date"] >= store.holdout_start)
    if late.any():  # the dev purge of ADR 0001 4.6: the race shows no holdout row, even if handed one
        raise ValueError(f"model_race: {int(late.sum())} row(s) dated on or after holdout_start {store.holdout_start}")
    origins_by_panel = {}
    for target_id, h in PANELS:
        rows = s[(s.target_id == target_id) & (s.horizon == h)]
        origins = sorted(set(rows["origin_date"]))[-store.race_sessions:]
        if not origins:
            raise StoreUnavailable(f"no dev-scored rows for {target_id} h={h} in run {store.run_id}")
        origins_by_panel[(target_id, h)] = origins
    first = min(o[0] for o in origins_by_panel.values())
    sessions = sorted(set(s["origin_date"]) | set(s["target_date"]))
    calendar = [d for d in sessions if d >= first] + [store.holdout_start]
    index = {d: i for i, d in enumerate(calendar)}
    mz = _all_dev(store.mz)
    mz = mz[mz.primary & mz.space.eq("level") & mz.forecast.eq("mean")]
    panels = []
    for (target_id, h), origins in origins_by_panel.items():
        rows = s[(s.target_id == target_id) & (s.horizon == h) & s.origin_date.isin(origins)]
        i0 = index[origins[0]]
        if [index[o] for o in origins] != list(range(i0, i0 + len(origins))):
            raise ValueError(f"model_race: the origins of {target_id} h={h} are not consecutive sessions")
        if any(index[t] != index[o] + h for o, t in zip(rows["origin_date"], rows["target_date"])):
            raise ValueError(f"model_race: a target of {target_id} h={h} is not {h} sessions after its origin")
        actual = rows.drop_duplicates("origin_date").set_index("origin_date")["y"].reindex(origins)
        series = []
        for model_id in race_models(store.models, target_id):
            yhat = rows[rows.model_id == model_id].set_index("origin_date")["yhat"].reindex(origins)
            if yhat.notna().any():
                series.append({"model_id": model_id, "yhat": _rounded(yhat, DEC_LEVEL)})
        shown = {x["model_id"] for x in series}
        panel_mz = mz[(mz.target_id == target_id) & (mz.horizon == h) & mz.model_id.isin(shown)]
        panels.append({
            "target_id": target_id,
            "horizon": int(h),
            "origin_idx0": i0,
            "n": len(origins),
            "actual": _rounded(actual, DEC_LEVEL),
            "series": series,
            "mz": [
                {"model_id": m.model_id, "period": "dev", "n": int(m.n), "alpha": r(m.alpha, DEC_STAT), "beta": r(m.beta, DEC_STAT), "wald_p": r(m.wald_p, DEC_STAT)}
                for m in panel_mz.sort_values("model_id").itertuples()
            ],
        })
    return {"calendar": [d.isoformat() for d in calendar], "holdout_start_idx": len(calendar) - 1, "panels": panels}


def check_model_race(payload: dict[str, Any]) -> None:
    """data-model.md 5.3's invariants that the schema cannot express. Raises ``ValueError``."""
    cal = payload["calendar"]
    if any(a >= b for a, b in zip(cal, cal[1:])):
        raise ValueError("model_race: calendar is not strictly increasing")
    for p in payload["panels"]:
        if p["origin_idx0"] + p["n"] - 1 + p["horizon"] > len(cal) - 1:
            raise ValueError(f"model_race: panel {p['target_id']} h={p['horizon']} runs past the calendar")
        if p["origin_idx0"] + p["n"] - 1 + p["horizon"] >= payload["holdout_start_idx"]:
            raise ValueError(f"model_race: panel {p['target_id']} h={p['horizon']} reaches holdout_start")
        lengths = {len(p["actual"]), *(len(x["yhat"]) for x in p["series"])}
        if lengths != {p["n"]}:
            raise ValueError(f"model_race: panel {p['target_id']} h={p['horizon']} has arrays of lengths {sorted(lengths)}, not n={p['n']}")


# --------------------------------------------------------------------------
# evaluation.json
# --------------------------------------------------------------------------


def evaluation_payload(store: Store) -> dict[str, Any]:
    """M5's dev tables, whole period: losses with ratios to ``rw``, primary DM pairs, primary MZ rows."""
    metric = _all_dev(store.metric)
    metric = metric[metric.forecast.eq("mean")]
    losses = []
    for (target_id, h), group in metric.groupby(["target_id", "horizon"], sort=False):
        rw = group[group.model_id == "rw"]
        rw_rmse = float(rw.rmse.iloc[0]) if len(rw) else math.nan
        rw_qlike = float(rw.qlike.iloc[0]) if len(rw) else math.nan
        for model_id in race_models(store.models, target_id):
            row = group[group.model_id == model_id]
            if row.empty:
                continue
            m = row.iloc[0]
            losses.append({
                "target_id": target_id, "horizon": int(h), "model_id": model_id, "n": int(m.n),
                "rmse": r(m.rmse, DEC_STAT), "mae": r(m.mae, DEC_STAT), "bias": r(m.bias, DEC_STAT),
                "mape": r(m.mape, DEC_MAPE), "qlike": r(m.qlike, DEC_QLIKE),
                "rmse_ratio_rw": r(m.rmse / rw_rmse, DEC_RATIO), "qlike_ratio_rw": r(m.qlike / rw_qlike, DEC_RATIO),
            })
    order = {key: i for i, key in enumerate(PANELS)}
    losses.sort(key=lambda x: order.get((x["target_id"], x["horizon"]), len(order)))
    dm = _all_dev(store.dm)
    dm = dm[dm.primary & dm.family.eq("pairs")].sort_values(["target_id", "horizon", "loss", "model_a", "model_b"])
    mz = _all_dev(store.mz)
    mz = mz[mz.primary & mz.space.eq("level") & mz.forecast.eq("mean")].sort_values(["target_id", "horizon", "model_id"])
    cal = store.calibration
    return {
        "period": "dev",
        "source_run_id": store.run_id,
        "dm_method": str(store.report_meta["dm_primary"]),
        "dm_fixed_b": float(store.report_meta["fixed_b"]),
        "mz_method": str(store.report_meta["mz_primary"]),
        "losses": losses,
        "dm": [
            {"target_id": d.target_id, "horizon": int(d.horizon), "loss": d.loss, "model_a": d.model_a, "model_b": d.model_b,
             "n": int(d.n), "dm_stat": r(d.dm_stat, DEC_STAT), "p_value": r(d.p_value, DEC_STAT), "p_holm": r(d.p_holm, DEC_STAT),
             "lower_loss": d.lower_loss}
            for d in dm.itertuples()
        ],
        "mz": [
            {"target_id": m.target_id, "horizon": int(m.horizon), "model_id": m.model_id, "n": int(m.n),
             "alpha": r(m.alpha, DEC_STAT), "alpha_se": r(m.alpha_se, DEC_STAT), "beta": r(m.beta, DEC_STAT),
             "beta_se": r(m.beta_se, DEC_STAT), "r2": r(m.r2, DEC_STAT), "wald_p": r(m.wald_p, DEC_STAT)}
            for m in mz.itertuples()
        ],
        "calibration": {"cells": int(len(cal)), "helps": int(cal["helps"].astype(bool).sum())},
    }


def check_evaluation(payload: dict[str, Any]) -> None:
    """Each DM pair names two models with a loss row on the same (target, horizon), and lower_loss is one of them."""
    have = {(x["target_id"], x["horizon"], x["model_id"]) for x in payload["losses"]}
    for d in payload["dm"]:
        for model in (d["model_a"], d["model_b"]):
            if (d["target_id"], d["horizon"], model) not in have:
                raise ValueError(f"evaluation: DM pair names {model}, which has no loss row at {d['target_id']} h={d['horizon']}")
        if d["lower_loss"] not in (d["model_a"], d["model_b"]):
            raise ValueError(f"evaluation: lower_loss {d['lower_loss']!r} is neither model of its pair")


# --------------------------------------------------------------------------
# equity_curves.json (data-model.md 5.5)
# --------------------------------------------------------------------------


def _strategy_defs(store: Store) -> dict[str, Any]:
    return {s.strategy_id: s for s in store.backtest_config.strategies}


def equity_curves_payload(store: Store) -> dict[str, Any]:
    """The primary rule and the three benchmarks at the default cost, plus every cost level's summary."""
    primary = str(store.backtest_config.primary)
    shown = (primary, *BENCHMARKS)
    defs = _strategy_defs(store)
    cost = float(store.backtest_meta["cost_bps_default"])
    daily = store.daily[store.daily.period.eq("dev") & store.daily.strategy_id.isin(shown)]
    at_cost = daily[np.isclose(daily.cost_bps, cost)]
    dates = sorted(set(at_cost["date"]))
    strategies = []
    for sid in shown:
        rows = at_cost[at_cost.strategy_id == sid].set_index("date").reindex(dates)
        if rows["equity"].isna().any():
            raise ValueError(f"equity_curves: {sid} has no row on some return dates at {cost} bps")
        d = defs[sid]
        track = store.backtest_config.tracks[d.track] if d.track else None
        strategies.append({
            "strategy_id": sid,
            "label": d.label,
            "kind": d.kind,
            "model_id": store.backtest_meta["models"][d.track] if track else None,
            "horizon": int(track["horizon"]) if track else None,
            "equity": _rounded(rows["equity"], DEC_EQUITY),
            "drawdown": [min(0.0, max(-1.0, v)) for v in _rounded(rows["drawdown"], DEC_EQUITY)],
        })
    summary = store.summary[store.summary.period.eq("dev") & store.summary.strategy_id.isin(shown)]
    order = {sid: i for i, sid in enumerate(shown)}
    summary = summary.assign(_o=summary.strategy_id.map(order)).sort_values(["_o", "cost_bps"])

    def row(x) -> dict[str, Any]:
        return {"strategy_id": x.strategy_id, "period": "dev", "n_days": int(x.n_days), "ann_return": r(x.ann_return, DEC_STAT),
                "ann_vol": r(x.ann_vol, DEC_STAT), "sharpe_ann": r(x.sharpe_ann, DEC_STAT), "psr0": r(x.psr0, DEC_STAT),
                "dsr": r(x.dsr, DEC_STAT), "n_trials": int(x.n_trials), "max_drawdown": r(x.max_drawdown, DEC_STAT),
                "turnover_ann": r(x.turnover_ann, DEC_STAT)}

    return {
        "dates": [d.isoformat() for d in dates],
        "holdout_start_idx": None,
        "cost_bps": cost,
        "strategies": strategies,
        "regime_segments": [],
        "summary": [row(x) for x in summary.itertuples() if np.isclose(x.cost_bps, cost)],
        "cost_sensitivity": [{**row(x), "cost_bps": float(x.cost_bps)} for x in summary.itertuples()],
        "caveats": [
            "Synthetic market: the VXX and SVXY paths are generated, not traded prices.",
            f"Dev period only ({dates[0].isoformat()} to {dates[-1].isoformat()}); the holdout has not been looked at.",
            f"Costs: {cost:g} bps per unit of turnover by default (half-spread plus commission, configs/backtest.yaml).",
            f"DSR deflates for {int(summary.n_trials.max())} forecast strategies tried on this period (the trial log).",
            f"The primary rule ({primary}) was fixed in configs/backtest.yaml before the run.",
        ],
    }


def check_equity_curves(payload: dict[str, Any]) -> None:
    """data-model.md 5.5's length invariant and a start at 1 (growth of 1.0)."""
    n = len(payload["dates"])
    for s in payload["strategies"]:
        if len(s["equity"]) != n or len(s["drawdown"]) != n:
            raise ValueError(f"equity_curves: {s['strategy_id']} does not have one value per date")


# --------------------------------------------------------------------------
# eda.json
# --------------------------------------------------------------------------

ACF_SERIES = (
    ("r", "S&P 500 log return r"),
    ("abs_r", "absolute return |r|"),
    ("r2", "squared return r²"),
    ("log_vix", "ln VIX"),
)


def eda_payload(store: Store) -> dict[str, Any]:
    """The clustering ACFs (r, |r|, r², ln VIX; lags 1..21) with a 95% white-noise band, and the half-life."""
    e = store.eda
    series = []
    for series_id, label in ACF_SERIES:
        if series_id == "log_vix":
            acf, n, p = e["persistence"]["log_vix_acf"][1:], e["persistence"]["n"], None
        else:
            box = e["clustering"][series_id]
            acf, n, p = box["acf"], box["n"], box["ljung_box_p"][-1]
        series.append({
            "series_id": series_id, "label": label, "n": int(n), "acf": _rounded(acf, DEC_STAT),
            "band95": r(1.959964 / math.sqrt(n), DEC_STAT),  # +-z_0.975/sqrt(n): Bartlett's white-noise band
            "ljung_box_p_last": r(p, DEC_STAT) if p is not None else None,
        })
    half_life = [
        {"sample": name, "phi": r(hl["phi"], DEC_STAT), "phi_ci95": _rounded(hl["phi_ci95"], DEC_STAT),
         "half_life_sessions": r(hl["half_life_sessions"], 2), "half_life_ci95": _rounded(hl["half_life_ci95"], 2), "n": int(hl["n"])}
        for name, hl in e["half_life"].items()
    ]
    sample = e["sample"]
    return {
        "period": "pre_holdout",
        "source_run_id": store.run_id,
        "first": sample["first"],
        "last": sample["last"],
        "holdout_start": sample["holdout_start"],
        "nlags": len(series[0]["acf"]),
        "acf": series,
        "half_life": half_life,
    }
