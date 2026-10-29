"""Features, targets and regime labels on the master calendar 𝒯 (ADR 0001 4.1, 4.2, 4.8; rule L1).

Everything here is a pure function of arrays aligned on 𝒯: index i is the
session ``calendar[i]``, and a missing value is NaN (written as null on disk).

Rule L1: a feature at session t uses data dated <= t only. Every window here
is *trailing* (``rolling(window, min_periods=window)``, which looks back), and
a window with any missing value gives NaN (gap policy G2): nothing is filled,
centred or normalised over the sample. The one forward-looking series is the
**target**: ``forward(rv21, 21)`` at origin t is the realised volatility over
the 21 sessions after t. It is the thing being forecast, never an input.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

#: ADR 0001 4.2: rv21 sums 21 squared daily log returns and annualises with 252/21.
RV_WINDOW = 21
SESSIONS_PER_YEAR = 252
HAR_WEEK = 5  # t-4..t
HAR_MONTH = 22  # t-21..t
VOV_WINDOW = 21

#: Term-structure ratios, shorter tenor over longer. Above 1 means that segment
#: of the curve slopes down (inverted, "backwardation"). vix_vix3m is the data
#: model's own column; the other three are added in M1 (see the walkthrough).
RATIO_PAIRS = (("vix9d_vix", "VIX9D", "VIX"), ("vix_vix3m", "VIX", "VIX3M"), ("vix3m_vix6m", "VIX3M", "VIX6M"), ("vix6m_vix1y", "VIX6M", "VIX1Y"))


def align(dates: list[date], frame: pd.DataFrame, key: str, symbol: str, column: str = "close") -> np.ndarray:
    """``column`` of ``symbol`` on each session of ``dates``; NaN where there is no row.

    Rows on dates that are not in ``dates`` (non-sessions, gap policy G4) are not read.
    """
    rows = frame.loc[frame[key] == symbol, ["date", column]]
    series = pd.Series(rows[column].to_numpy(dtype=float), index=rows["date"].to_numpy())
    return series.reindex(pd.Index(dates, dtype=object)).to_numpy(dtype=float)


def log_returns(close: np.ndarray) -> np.ndarray:
    """r[i] = ln(close[i] / close[i-1]) on consecutive sessions; r[0] is NaN, and NaN next to a gap."""
    out = np.full(close.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.log(close[1:] / close[:-1])
    return out


def trailing(values: np.ndarray, window: int, how: str) -> np.ndarray:
    """A trailing-window statistic: at i it uses values[i-window+1 .. i]. NaN if any is missing."""
    rolling = pd.Series(values).rolling(window, min_periods=window)
    if how == "sum":
        result = rolling.sum()
    elif how == "mean":
        result = rolling.mean()
    elif how == "std":
        result = rolling.std(ddof=1)
    else:
        raise ValueError(how)
    return result.to_numpy(dtype=float)


def rv_trailing(r: np.ndarray, window: int = RV_WINDOW) -> np.ndarray:
    """rv(d) = 100*sqrt((252/window) * sum_{j=0}^{window-1} r_{d-j}**2), in vol points (ADR 0001 4.2).

    No mean is removed. Worked number: 21 returns of 0.01 each give
    100*sqrt(12 * 21 * 0.0001) = 100*sqrt(0.0252) = 15.8745.
    """
    return 100 * np.sqrt((SESSIONS_PER_YEAR / window) * trailing(r**2, window, "sum"))


def forward(values: np.ndarray, h: int) -> np.ndarray:
    """out[i] = values[i + h], NaN past the end. For TARGETS and ex-post EDA only, never for features."""
    out = np.full(values.shape, np.nan)
    if h < len(values):
        out[: len(values) - h] = values[h:]
    return out


def build_features(dates: list[date], iv: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """The ``feature`` table (data-model.md section 4) on ``dates`` (𝒯 from train_start to data_end).

    The first rows of each trailing window are null, not dropped (data model:
    "incomplete windows at the start of the training history are left empty").
    """
    vix = align(dates, iv, "symbol", "VIX")
    spx = align(dates, price, "ticker", "^GSPC")
    log_vix = np.log(vix)
    r = log_returns(spx)
    rv21 = rv_trailing(r)
    features = {
        "date": dates,
        "log_vix": log_vix,
        "har_w": trailing(log_vix, HAR_WEEK, "mean"),
        "har_m": trailing(log_vix, HAR_MONTH, "mean"),
        "r_spx": r,
        "rv21_back": rv21,
        "vov21": trailing(log_returns(vix), VOV_WINDOW, "std"),
        # Trailing spread in variance units: implied variance now minus the
        # realised variance of the last 21 sessions. Both are known at t.
        "iv_rv_spread": (vix / 100) ** 2 - (rv21 / 100) ** 2,
    }
    for name, short, long in RATIO_PAIRS:
        features[name] = align(dates, iv, "symbol", short) / align(dates, iv, "symbol", long)
    return pd.DataFrame(features)


def build_target_values(dates: list[date], iv: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """The ``target_value`` table: y in vol points and v = (y/100)**2 per (target_id, date).

    ``rv21`` at date d is the trailing window ending at d, the same number as
    ``rv21_back(d)`` (data model: "one series, two roles"). A forecast from
    origin t at h = 21 is scored against the value at ``𝒯[t + 21]``, whose
    window is exactly the 21 sessions after t. Dates with no value have no row.
    """
    vix = align(dates, iv, "symbol", "VIX")
    rv21 = rv_trailing(log_returns(align(dates, price, "ticker", "^GSPC")))
    frames = []
    for target_id, y in (("vix_level", vix), ("rv21", rv21)):
        keep = ~np.isnan(y)
        frames.append(
            pd.DataFrame(
                {"target_id": target_id, "date": np.asarray(dates, dtype=object)[keep], "y": y[keep], "v": (y[keep] / 100) ** 2}
            )
        )
    return pd.concat(frames, ignore_index=True)


def is_backwardation(ratio: float) -> bool:
    """The one rule for the term-structure regime (ADR 0001 4.8): VIX/VIX3M > 1. Equal is contango.

    ``build_regime_labels`` and ``term_structure_panel`` (the app's colouring flag)
    both call this, so the regime table and the exported flag cannot disagree.
    """
    return ratio > 1


def build_regime_labels(dates: list[date], iv: pd.DataFrame, edges: tuple[float, float]) -> pd.DataFrame:
    """The ``regime_label`` table (ADR 0001 4.8). Fixed thresholds, labelled from data at the date only (L5).

    ``vix_level``: low V < edges[0] <= mid < edges[1] <= high.
    ``term_structure``: backwardation if VIX > VIX3M, else contango; no row
    where either is missing.
    """
    low, high = edges
    vix = align(dates, iv, "symbol", "VIX")
    ratio = vix / align(dates, iv, "symbol", "VIX3M")
    rows = []
    for d, v, q in zip(dates, vix, ratio):
        if not math.isnan(v):
            rows.append({"date": d, "scheme": "vix_level", "regime": "low" if v < low else "mid" if v < high else "high", "driver": v})
        if not math.isnan(q):
            rows.append({"date": d, "scheme": "term_structure", "regime": "backwardation" if is_backwardation(q) else "contango", "driver": q})
    return pd.DataFrame(rows, columns=["date", "scheme", "regime", "driver"])


def term_structure_panel(dates: list[date], iv: pd.DataFrame, symbols: list[str]) -> dict[str, np.ndarray]:
    """Levels of each tenor on ``dates``, plus VIX/VIX3M and the backwardation flag (data-model.md 5.2).

    The flag is ``VIX/VIX3M > 1``, computed here from the unrounded ratio, so
    the app only colours it (ADR 0001 D1). It is NaN where the ratio is.
    """
    panel = {symbol: align(dates, iv, "symbol", symbol) for symbol in symbols}
    ratio = align(dates, iv, "symbol", "VIX") / align(dates, iv, "symbol", "VIX3M")
    flag = np.array([np.nan if np.isnan(q) else float(is_backwardation(q)) for q in ratio])
    return {"levels": panel, "vix_vix3m": ratio, "backwardation": flag}
