"""Performance metrics of one backtest (``strategy_summary``, data-model.md; milestone 6).

All from the daily simple net returns ``r_1..r_T`` of ``engine.daily`` (cash earns 0, so these are
also excess returns over cash):

* ``ann_return`` = mean(r) * 252 (arithmetic, as data-model.md defines it); ``cagr`` = E_T^(252/T) - 1
* ``ann_vol`` = sd(r, ddof 1) * sqrt(252); ``sr_daily`` = mean / sd; ``sharpe_ann`` = sr_daily * sqrt(252)
* ``sortino_ann`` = mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(252) (downside deviation about 0, over all T days)
* ``max_drawdown`` = min over d of E_d / max(1, max_{s<=d} E_s) - 1; ``calmar`` = cagr / |max_drawdown|
* ``turnover_ann`` = sum(turnover) * 252 / T; ``n_trades`` = days with a fill (turnover > 0)
* ``hit_rate`` = share of days in a position (signal != 0) with r > 0
* ``psr0`` = PSR against SR* = 0; ``dsr`` = PSR against E[max SR] of the trials (``deflate``)

PSR and DSR take the PER-DAY Sharpe ratio, skewness and Pearson kurtosis (``ivf.backtest.sharpe``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ivf.backtest import sharpe

DAYS = sharpe.TRADING_DAYS


def summary(day: pd.DataFrame) -> dict:
    r = day["net_ret"].to_numpy(dtype=float)
    n = r.size
    equity = day["equity"].to_numpy(dtype=float)
    sr = sharpe.sharpe_ratio(r)
    skew, kurt = sharpe.sample_moments(r) if n > 2 and np.std(r) > 0 else (math.nan, math.nan)
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if n else math.nan
    mdd = float(day["drawdown"].min()) if n else math.nan
    cagr = float(equity[-1] ** (DAYS / n) - 1.0) if n else math.nan
    invested = day["signal"].to_numpy() != 0
    return {
        "n_days": int(n),
        "ann_return": float(r.mean() * DAYS) if n else math.nan,
        "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * math.sqrt(DAYS)) if n > 1 else math.nan,
        "sr_daily": sr,
        "sharpe_ann": sr * math.sqrt(DAYS),
        "sortino_ann": float(r.mean() / downside * math.sqrt(DAYS)) if downside > 0 else math.nan,
        "calmar": cagr / abs(mdd) if mdd < 0 else math.nan,
        "max_drawdown": mdd,
        "skew": skew,
        "kurt_pearson": kurt,
        "psr0": sharpe.probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        "turnover_ann": float(day["turnover"].sum() * DAYS / n) if n else math.nan,
        "n_trades": int((day["turnover"] > 0).sum()),
        "hit_rate": float((r[invested] > 0).mean()) if invested.any() else math.nan,
        "days_vxx": int((day["signal"] == 1).sum()),
        "days_svxy": int((day["signal"] == -1).sum()),
        "days_cash": int((day["signal"] == 0).sum()),
    }


def deflate(sr_daily: float, n_days: int, skew: float, kurt: float, trial_srs: list[float], n_trials: int) -> dict:
    """E[max SR] and DSR for one strategy (Bailey & López de Prado 2014, eq. 2; see ``sharpe``).

    ``trial_srs`` are the per-day SRs of every trial on the same period and cost (the strategy's own
    included). The null is E[SR] = 0. ``var_sr`` = max(cross-trial variance (ddof 1), 1/T): the
    larger is the conservative choice; ``dsr_paper_var`` uses the cross-trial variance alone."""
    finite = [s for s in trial_srs if math.isfinite(s)]
    var_cross = float(np.var(finite, ddof=1)) if len(finite) >= 2 else 0.0
    var_sr = max(var_cross, sharpe.null_sharpe_variance(n_days))
    e_max = sharpe.expected_max_sharpe(n_trials, 0.0, var_sr)
    e_max_paper = sharpe.expected_max_sharpe(n_trials, 0.0, var_cross)
    return {
        "n_trials": int(n_trials), "var_sr_cross": var_cross, "var_sr": var_sr, "e_max_sr": e_max,
        "dsr": sharpe.deflated_sharpe_ratio(sr_daily, n_days, skew, kurt, e_max),
        "dsr_paper_var": sharpe.deflated_sharpe_ratio(sr_daily, n_days, skew, kurt, e_max_paper),
    }


__all__ = ["summary", "deflate", "DAYS"]
