"""The backtest engine: from a signal per origin to a daily net return (milestone 6).

TIMING (ADR 0001 rule L7; ``lag_sessions`` = 2 in ``configs/backtest.yaml``)

    close of t        the forecast and VIX_t are known; the signal s_t is computed
    close of t+1      the order is filled at this close (never at the close of t: the signal was
                      computed from it, and nobody can trade at a price already struck)
    close t+1 -> t+2  the first return the position earns: the row dated d = t+2

So the weight on return date d comes from the signal at origin ``sessions[i(d) - 2]``.

POSITIONS. ``s = +1``: 100% VXX (long volatility); ``s = -1``: 100% SVXY (short volatility, -0.5x
VIX futures since 2018-02-28); ``s = 0``: cash, earning ``cash_rate`` (0 by default). No shorting
and no leverage, so the weights are 0 or 1 and a one-asset book never drifts.

COSTS. ``turnover_d = |w_vxx,d - w_vxx,d-1| + |w_svxy,d - w_svxy,d-1|`` with the weights before
the first day taken as 0 (so the first entry is charged). The fill is at the close of d-1, so the
cost ``c_d = turnover_d * cost_bps / 10,000`` is paid out of the equity of that close, and the
day's simple net return is exact, not the first-order ``gross - cost``:

    net_d = (1 - c_d) * (1 + gross_d) - 1,   gross_d = w_vxx,d r_vxx,d + w_svxy,d r_svxy,d + w_cash,d r_cash

A switch VXX -> SVXY has turnover 2 (sell one side, buy the other). No exit is charged after the
last day: the book is marked, not liquidated.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

WEIGHTS = {1: (1.0, 0.0), -1: (0.0, 1.0), 0: (0.0, 0.0)}


def positions(signal: pd.Series, sessions: list[date], dates: list[date], lag: int) -> pd.DataFrame:
    """One row per return date in ``dates``: ``date``, ``origin_date`` (``lag`` sessions earlier on
    ``sessions``), ``signal`` and the weights. A missing signal at a needed origin is an error, never
    a silent cash day."""
    index = {d: i for i, d in enumerate(sessions)}
    rows = []
    for d in dates:
        i = index[d] - lag
        if i < 0:
            raise ValueError(f"return date {d}: no session {lag} before it on the calendar")
        origin = sessions[i]
        value = signal.get(origin, np.nan)
        if not np.isfinite(value):
            raise ValueError(f"return date {d}: no signal at origin {origin}")
        s = int(value)
        w_vxx, w_svxy = WEIGHTS[s]
        rows.append({"date": d, "origin_date": origin, "signal": s, "w_vxx": w_vxx, "w_svxy": w_svxy})
    return pd.DataFrame(rows)


def daily(pos: pd.DataFrame, returns: pd.DataFrame, cost_bps: float, cash_rate_annual: float = 0.0) -> pd.DataFrame:
    """``backtest_daily`` rows (data-model.md) from ``positions`` and a frame of simple returns
    indexed by date with columns ``VXX`` and ``SVXY``. A missing return while holding is an error."""
    r = returns.reindex(pos["date"])
    w_vxx, w_svxy = pos["w_vxx"].to_numpy(), pos["w_svxy"].to_numpy()
    r_vxx, r_svxy = r["VXX"].to_numpy(dtype=float), r["SVXY"].to_numpy(dtype=float)
    held = ((w_vxx != 0) & ~np.isfinite(r_vxx)) | ((w_svxy != 0) & ~np.isfinite(r_svxy))
    if held.any():
        raise ValueError(f"no return on {list(pos['date'][held])[:3]} for a position held that day")
    r_cash = (1.0 + cash_rate_annual) ** (1.0 / 252) - 1.0
    w_cash = 1.0 - w_vxx - w_svxy
    gross = np.where(w_vxx != 0, w_vxx * np.nan_to_num(r_vxx), 0.0) + np.where(w_svxy != 0, w_svxy * np.nan_to_num(r_svxy), 0.0) + w_cash * r_cash
    turnover = np.abs(np.diff(w_vxx, prepend=0.0)) + np.abs(np.diff(w_svxy, prepend=0.0))
    cost = turnover * cost_bps / 1e4
    net = (1.0 - cost) * (1.0 + gross) - 1.0
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    out = pos.copy()
    out["gross_ret"], out["turnover"], out["cost"], out["net_ret"] = gross, turnover, cost, net
    out["equity"], out["drawdown"] = equity, equity / peak - 1.0
    out["cost_bps"] = float(cost_bps)
    return out


__all__ = ["positions", "daily", "WEIGHTS"]
