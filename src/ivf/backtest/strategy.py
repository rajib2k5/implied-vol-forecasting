"""Pre-registered strategies, the model choice, signals, and one period's backtest (milestone 6).

THE RULE (``configs/backtest.yaml``; PR/FAQ question 3, default taken 2026-10-29, owner can override).
At the close of origin t, with ``f_t`` a frozen model's forecast at h = 21 and ``V_t`` the VIX close,
both in vol points: ``d_t = f_t - V_t``; long VXX if ``d_t > band``, long SVXY if ``d_t < -band``,
else cash. On the ``rv21`` track ``d_t`` is minus the forecast variance risk premium (implied above
expected realised: sell volatility); on the ``vix_level`` track it is the forecast 21-day VIX change.

THE MODEL (``select_model``). Chosen by a rule written before any backtest ran: the lowest dev loss in
M5's ``metric`` table (scheme ``all``, forecast ``mean``) among the track's models minus the excluded
ids, ties to the first id alphabetically, the same ranking as M5's ``holdout_check``. Only dev rows
choose; nothing in the backtest feeds back into the choice.

TRIALS (``count_trials``). Every non-benchmark strategy run on a period is one line per cost level in
the append-only trial log under the data root, keyed by ``(strategy_id, params)`` where ``params``
include the chosen model id. ``n_trials`` is the number of distinct keys ever logged for the period,
so re-running the same pre-registered rule adds nothing and any changed rule (a new band, a new
model, a retired row) adds one. Benchmarks are not trials: nothing is selected among them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ivf import config
from ivf.backtest import engine, metrics, segments as seg
from ivf.protocol import ProtocolError, _exact_keys, _read_yaml

CONFIG_FILE = "backtest.yaml"
_TOP = {"registered_on", "lag_sessions", "cash_rate_annual", "costs", "tracks", "primary", "strategies", "trial_log",
        "splits", "exposure_tolerance", "holdout_rule", "holdout_looks_before_look", "holdout_caveat"}


@dataclass(frozen=True)
class StrategyDef:
    strategy_id: str
    kind: str
    label: str
    retired: bool
    track: str | None = None
    band: float | None = None
    signal: int | None = None


@dataclass(frozen=True)
class BacktestConfig:
    document: dict
    lag: int
    cash_rate: float
    cost_bps: float
    sensitivity_bps: tuple[float, ...]
    tracks: dict
    primary: str
    strategies: tuple[StrategyDef, ...]
    trial_log: str
    splits: list = field(default_factory=list)
    tolerance: float = 0.15

    def cost_levels(self) -> list[float]:
        return sorted({self.cost_bps, *self.sensitivity_bps})

    def active(self) -> list[StrategyDef]:
        return [s for s in self.strategies if not s.retired]


def load_config(path: Path | None = None) -> BacktestConfig:
    path = path or config.configs_dir() / CONFIG_FILE
    doc = _exact_keys(str(path), _read_yaml(path), _TOP)
    lag = int(doc["lag_sessions"])
    if lag < 2:
        raise ProtocolError(f"{path}: lag_sessions must be >= 2 (rule L7: fill at the close AFTER the signal)")
    costs = _exact_keys(f"{path}: costs", doc["costs"], {"half_spread_bps", "commission_bps", "sensitivity_bps"})
    defs = []
    for i, entry in enumerate(doc["strategies"]):
        kind = entry.get("kind")
        keys = {"strategy_id", "kind", "label", "retired"} | ({"track", "band"} if kind == "forecast" else {"signal"})
        e = _exact_keys(f"{path}: strategies[{i}]", entry, keys)
        if kind == "forecast":
            if e["track"] not in doc["tracks"] or float(e["band"]) < 0:
                raise ProtocolError(f"{path}: strategies[{i}]: unknown track or negative band")
            defs.append(StrategyDef(e["strategy_id"], kind, e["label"], bool(e["retired"]), e["track"], float(e["band"])))
        elif kind == "benchmark":
            if e["signal"] not in (-1, 0, 1):
                raise ProtocolError(f"{path}: strategies[{i}]: signal must be -1, 0 or 1")
            defs.append(StrategyDef(e["strategy_id"], kind, e["label"], bool(e["retired"]), signal=int(e["signal"])))
        else:
            raise ProtocolError(f"{path}: strategies[{i}]: kind must be forecast or benchmark")
    ids = [d.strategy_id for d in defs]
    if len(set(ids)) != len(ids) or doc["primary"] not in ids:
        raise ProtocolError(f"{path}: strategy ids must be unique and include the primary")
    return BacktestConfig(doc, lag, float(doc["cash_rate_annual"]), float(costs["half_spread_bps"]) + float(costs["commission_bps"]),
                          tuple(float(c) for c in costs["sensitivity_bps"]), doc["tracks"], doc["primary"], tuple(defs),
                          doc["trial_log"], list(doc["splits"]), float(doc["exposure_tolerance"]))


def select_model(dev_metric: pd.DataFrame, target_id: str, horizon: int, loss: str, exclude) -> str:
    """The dev-best model of one track (module docstring). Raises if no candidate has a finite loss."""
    m = dev_metric[(dev_metric.scheme == "all") & (dev_metric.forecast == "mean") & (dev_metric.target_id == target_id)
                   & (dev_metric.horizon == horizon) & (dev_metric.period == "dev") & ~dev_metric.model_id.isin(list(exclude))]
    ranked = m.set_index("model_id")[loss].dropna().sort_index().sort_values(kind="stable")
    if ranked.empty:
        raise ValueError(f"no dev {loss} for {target_id} h={horizon} outside {list(exclude)}")
    return str(ranked.index[0])


def select_models(cfg: BacktestConfig, dev_metric: pd.DataFrame) -> dict[str, str]:
    return {name: select_model(dev_metric, t["target_id"], int(t["horizon"]), t["select_by"], t["exclude"])
            for name, t in cfg.tracks.items()}


def band_signal(forecast: pd.Series, vix: pd.Series, band: float) -> pd.Series:
    """+1 where ``forecast - vix > band``, -1 where ``< -band``, 0 otherwise (a value AT the band is cash);
    NaN where either input is missing (the engine then refuses to trade on it)."""
    d = forecast - vix.reindex(forecast.index)
    out = pd.Series(np.where(d > band, 1.0, np.where(d < -band, -1.0, 0.0)), index=forecast.index)
    return out.where(d.notna())


def signals(cfg: BacktestConfig, chosen: dict[str, str], forecasts: pd.DataFrame, vix: pd.Series,
            sessions: list[date]) -> dict[str, pd.Series]:
    """``strategy_id -> signal indexed by origin date`` for every active strategy."""
    out = {}
    for s in cfg.active():
        if s.kind == "benchmark":
            out[s.strategy_id] = pd.Series(float(s.signal), index=sessions)
            continue
        t = cfg.tracks[s.track]
        f = forecasts[(forecasts.model_id == chosen[s.track]) & (forecasts.target_id == t["target_id"])
                      & (forecasts.horizon == int(t["horizon"]))]
        if f.origin_date.duplicated().any():
            raise ValueError(f"{chosen[s.track]}: two forecasts at one origin")
        out[s.strategy_id] = band_signal(f.set_index("origin_date")["yhat"].astype(float), vix, s.band)
    return out


def window(segments: tuple, sessions: list[date], period: str, holdout_start: date, data_end: date) -> list[date]:
    """The return dates of ``period`` inside usable segments of both ETPs; they must be consecutive sessions
    (one stable segment), else the backtest would silently bridge a break."""
    usable = seg.usable_return_dates(segments, sessions)
    if period == "dev":
        dates = [d for d in usable if d < holdout_start]
    elif period == "holdout":
        dates = [d for d in usable if holdout_start <= d <= data_end]
    else:
        raise ValueError(f"unknown period {period}")
    if not dates:
        raise ValueError(f"no usable return dates in period {period}")
    index = {d: i for i, d in enumerate(sessions)}
    if index[dates[-1]] - index[dates[0]] + 1 != len(dates):
        raise ValueError(f"period {period}: the usable return dates are not one unbroken segment")
    return dates


def run_period(cfg: BacktestConfig, segments: tuple, sessions: list[date], prices: pd.DataFrame,
               signal_map: dict[str, pd.Series], period: str, holdout_start: date, data_end: date) -> pd.DataFrame:
    """``backtest_daily`` for every strategy and cost level of ``period``."""
    dates = window(segments, sessions, period, holdout_start, data_end)
    returns = pd.DataFrame({t: seg.returns_on_sessions(prices, t, sessions) for t in seg.TRADABLES})
    frames = []
    for sid, sig in signal_map.items():
        pos = engine.positions(sig, sessions, dates, cfg.lag)
        for c in cfg.cost_levels():
            d = engine.daily(pos, returns, c, cfg.cash_rate)
            d.insert(0, "strategy_id", sid)
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "period", period)
    return out


def params_of(s: StrategyDef, chosen: dict[str, str]) -> str:
    p = {"kind": s.kind, "track": s.track, "band": s.band, "model_id": chosen.get(s.track) if s.track else None}
    return json.dumps(p, sort_keys=True)


def trial_rows(cfg: BacktestConfig, chosen: dict[str, str], summary: pd.DataFrame, meta: dict) -> list[dict]:
    """The trial-log lines of one run: every non-benchmark strategy at every cost level."""
    kinds = {s.strategy_id: s for s in cfg.strategies}
    rows = []
    for r in summary.itertuples(index=False):
        s = kinds[r.strategy_id]
        if s.kind != "forecast":
            continue
        rows.append({**meta, "period": r.period, "strategy_id": r.strategy_id, "params": params_of(s, chosen),
                     "cost_bps": float(r.cost_bps), "sr_daily": None if not math.isfinite(r.sr_daily) else float(r.sr_daily),
                     "n_days": int(r.n_days)})
    return rows


def count_trials(log: list[dict], period: str) -> int:
    return len({(r["strategy_id"], r["params"]) for r in log if r["period"] == period})


def trial_srs(log: list[dict], period: str, cost_bps: float) -> list[float]:
    """The latest per-day SR of each distinct trial on ``period`` at ``cost_bps``."""
    latest: dict = {}
    for r in log:
        if r["period"] == period and float(r["cost_bps"]) == float(cost_bps):
            latest[(r["strategy_id"], r["params"])] = r["sr_daily"]
    return [math.nan if v is None else float(v) for v in latest.values()]


def summarise(daily: pd.DataFrame, cfg: BacktestConfig, log: list[dict], trials_period: str = "dev") -> pd.DataFrame:
    """``strategy_summary``: metrics per (strategy, period, cost) with PSR, and DSR with N = the trials of
    ``trials_period`` in ``log``. On that period V[SR] comes from the logged trials; on another period (the
    holdout, where nothing is selected) from this run's forecast strategies on that period."""
    rows = [{"strategy_id": sid, "period": period, "cost_bps": float(cost), **metrics.summary(g)}
            for (sid, period, cost), g in daily.groupby(["strategy_id", "period", "cost_bps"], sort=True)]
    table = pd.DataFrame(rows)
    forecast_ids = {s.strategy_id for s in cfg.strategies if s.kind == "forecast"}
    n_trials = max(count_trials(log, trials_period), 1)
    extra = []
    for r in table.itertuples(index=False):
        if r.period == trials_period:
            srs = trial_srs(log, r.period, r.cost_bps)
        else:
            same = table[(table.period == r.period) & (table.cost_bps == r.cost_bps) & table.strategy_id.isin(forecast_ids)]
            srs = list(same.sr_daily)
        extra.append(metrics.deflate(r.sr_daily, r.n_days, r.skew, r.kurt_pearson, srs, n_trials))
    return pd.concat([table, pd.DataFrame(extra)], axis=1)


def holdout_verdict(summary: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """The pre-registered confirmatory check on the holdout (``holdout_rule`` in the config)."""
    s = summary[(summary.period == "holdout") & (summary.cost_bps == cfg.cost_bps)].set_index("strategy_id")["sharpe_ann"]
    primary, bench = float(s[cfg.primary]), max(float(s["bench_long_vxx"]), float(s["bench_long_svxy"]))
    return {"primary": cfg.primary, "primary_sharpe_ann": primary, "best_benchmark_sharpe_ann": bench,
            "held_up": bool(primary > 0 and primary > bench)}


__all__ = ["StrategyDef", "BacktestConfig", "load_config", "select_model", "select_models", "band_signal", "signals",
           "window", "run_period", "trial_rows", "count_trials", "trial_srs", "summarise", "holdout_verdict"]
