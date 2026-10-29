"""Walk-forward multiplicative bias calibration of a model's forecast (O-M4-7; milestone 5).

**Pre-registered rule** (written before any calibrated number was seen, default taken 2026-10-29,
owner can override):

1. At every fold cutoff c of the model's own forecasts, take the pairs ``(yhat_s, y_s)`` of that
   model, target and horizon whose **target is dated <= c** (label availability, rule L2) and observed.
   Expanding window, like every model (ADR 0001 4.5).
2. ``k_c = sum(y * yhat) / sum(yhat^2)``: the least-squares multiplier through the origin, the one
   that minimises the squared error of ``k * yhat`` on those pairs (level space, the RMSE's space).
3. With fewer than ``MIN_PAIRS`` (252) pairs, ``k_c = 1``: no calibration (``warm_up``).
4. Every origin of the fold that starts at c gets ``yhat_cal = k_c * yhat`` and
   ``v_hat_cal = k_c^2 * v_hat`` (so ``yhat_cal = 100 * sqrt(v_hat_cal)`` still holds).
5. The calibrated forecast is a **separate variant** (``forecast = mean_cal`` in ``metric``,
   ``<model>+cal`` in ``dm_test``). It never replaces the model's own forecast.
6. **Verdict.** Calibration "helps" a model at a (target, horizon) when the variant has a lower RMSE
   **and** a lower QLIKE than the raw forecast on the same scored rows, **and** the primary DM test
   of raw against calibrated on squared error rejects at 5% after Holm across the models of that
   (target, horizon). Anything else is "not shown".

The multiplier at cutoff c uses only errors whose outcome is known at c, so the calibrated forecast
at an origin t >= c is a function of data dated <= t, like the model's own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAIRS = 252


def multipliers(group: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """``k`` per fold cutoff for one (model, target, horizon): columns cutoff_date, k, n_pairs, warm_up.

    ``group`` needs ``cutoff_date``, ``target_date``, ``yhat`` and ``y`` (NaN where unobserved).
    """
    pairs = group[group["y"].notna()].sort_values("target_date")
    dates = pairs["target_date"].to_numpy()
    cross = np.cumsum(pairs["y"].to_numpy() * pairs["yhat"].to_numpy())
    square = np.cumsum(pairs["yhat"].to_numpy() ** 2)
    rows = []
    for cutoff in sorted(set(group["cutoff_date"])):
        n = int(np.searchsorted(dates, cutoff, side="right"))  # pairs with target_date <= cutoff
        warm = n < min_pairs
        rows.append({"cutoff_date": cutoff, "k": 1.0 if warm else float(cross[n - 1] / square[n - 1]),
                     "n_pairs": n, "warm_up": warm})
    return pd.DataFrame(rows)


def calibrate(forecasts: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(forecasts with yhat_cal and v_hat_cal, the multiplier table)``, per (model, target, horizon)."""
    out, table = [], []
    for (model_id, target_id, h), g in forecasts.groupby(["model_id", "target_id", "horizon"], sort=True):
        k = multipliers(g, min_pairs)
        k = k.assign(model_id=model_id, target_id=target_id, horizon=int(h))
        table.append(k)
        factor = g["cutoff_date"].map(k.set_index("cutoff_date")["k"])
        out.append(g.assign(yhat_cal=g["yhat"] * factor, v_hat_cal=g["v_hat"] * factor**2))
    cols = ["model_id", "target_id", "horizon", "cutoff_date", "k", "n_pairs", "warm_up"]
    return pd.concat(out).sort_index(), pd.concat(table, ignore_index=True)[cols]
