"""Tradable segments: where a VXX/SVXY return may be used (data-model.md ``tradable_segment``, amended M6).

A volatility ETP's price history is not one instrument. SVXY's daily target changed from -1x to
-0.5x at the close of 2018-02-27; Yahoo's VXX is the Series B ETN only from 2018-01-25; both had
reverse splits. A backtest that runs across such a break mixes two exposures and reports a number
that belongs to neither. So ``configs/instruments.yaml`` lists segments, each a span of closes with
one stable exposure, and this module enforces three things:

* **Membership.** A return dated d (close of the previous session -> close of d) belongs to a
  segment when BOTH closes are inside ``[valid_from, valid_to]``. Two segments may share their
  boundary close; every return then belongs to exactly one of them (``segment_of_return``).
* **The window.** ``usable_return_dates`` is the set of sessions whose return is inside a
  ``usable`` segment for every ticker the backtest may hold. The engine runs only there.
* **The data support the dates** (``check_segments``): no missing close inside a usable segment;
  each listed split shows no one-day move of the split's size (so the closes are split-adjusted);
  and the OLS slope of each ticker's daily return on the reference ticker's (VXX) is within a
  tolerance of the ratio of their stated leverages. A mismatch is a violation on a USABLE segment
  only; on an unusable one the slope is measured and reported, nothing more (amended M6, after the
  first real run, whose check failed on the unusable -1x SVXY segment: its short overlap with VXX
  includes the February 2018 collapse, when SVXY's close-to-close moves stopped tracking -1x VXX's;
  the measured value stays local). The issuer's statement, not that slope, is the evidence for the
  -1x. The usable -0.5x segment passing the check is the data's
  evidence that the cut took effect by its first return; it cannot pin the exact day by itself.

* **No stale prints** (amended M6, review A; added after the first dev run, default taken 2026-10-29,
  owner can override): a session whose open, high, low and close are equal, or whose volume is 0, is
  a stale print (no trade, or a carried-forward quote). Inside a usable segment, on dev sessions
  (``<= stale_until``), one is a violation; the rule that keeps the segments clean is in
  ``configs/instruments.yaml`` (VXX's usable segment starts at the first close after its last stale
  print). Later stale prints (the holdout) are counted and reported, never moved around.

Each check reads only rows up to ``read_until`` (the day before ``holdout_start`` on the dev path),
so verifying the instruments never reads a holdout price (ADR 0001 4.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ivf import config
from ivf.protocol import ProtocolError, _exact_keys, _read_yaml, parse_date

REFERENCE_TICKER = "VXX"
TRADABLES = ("VXX", "SVXY")
_SEGMENT_KEYS = {"ticker", "valid_from", "valid_to", "leverage", "usable", "note"}
#: The exposure slope needs at least this many overlapping returns to be checked.
MIN_OVERLAP = 10


@dataclass(frozen=True)
class Segment:
    ticker: str
    valid_from: date
    valid_to: date | None
    leverage: float
    usable: bool
    note: str

    def contains_close(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)


def load_segments(path: Path | None = None) -> tuple[Segment, ...]:
    path = path or config.configs_dir() / "instruments.yaml"
    document = _read_yaml(path)
    out = []
    for i, entry in enumerate(document.get("tradable_segments") or []):
        where = f"{path}: tradable_segments[{i}]"
        entry = _exact_keys(where, entry, _SEGMENT_KEYS)
        valid_to = None if entry["valid_to"] is None else parse_date(f"{where}.valid_to", entry["valid_to"])
        out.append(Segment(entry["ticker"], parse_date(f"{where}.valid_from", entry["valid_from"]), valid_to,
                           float(entry["leverage"]), bool(entry["usable"]), str(entry["note"])))
    by_ticker: dict[str, list[Segment]] = {}
    for s in out:
        if s.valid_to is not None and s.valid_to <= s.valid_from:
            raise ProtocolError(f"{path}: segment {s.ticker} {s.valid_from} ends before it starts")
        by_ticker.setdefault(s.ticker, []).append(s)
    for ticker, segs in by_ticker.items():
        segs.sort(key=lambda s: s.valid_from)
        for a, b in zip(segs, segs[1:]):  # may share a boundary close, never overlap further
            if a.valid_to is None or b.valid_from < a.valid_to:
                raise ProtocolError(f"{path}: {ticker} segments starting {a.valid_from} and {b.valid_from} overlap")
    return tuple(out)


def segment_of_return(segments: tuple[Segment, ...], ticker: str, previous: date, day: date) -> Segment | None:
    """The segment holding the return from the close of ``previous`` to the close of ``day``, if any."""
    for s in segments:
        if s.ticker == ticker and s.contains_close(previous) and s.contains_close(day):
            return s
    return None


def usable_return_dates(segments: tuple[Segment, ...], sessions: list[date], tickers=TRADABLES) -> list[date]:
    """Sessions d (never the first) whose return is inside a usable segment of EVERY ticker."""
    out = []
    for previous, day in zip(sessions, sessions[1:]):
        found = [segment_of_return(segments, t, previous, day) for t in tickers]
        if all(s is not None and s.usable for s in found):
            out.append(day)
    return out


def returns_on_sessions(prices: pd.DataFrame, ticker: str, sessions: list[date]) -> pd.Series:
    """Simple returns ``adj_close[d] / adj_close[previous session] - 1`` on ``sessions`` (NaN where a close is missing)."""
    rows = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    level = rows.reindex(sessions).astype(float)
    return level / level.shift(1) - 1.0


def stale_prints(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Boolean by date: open = high = low = close, or volume 0 (module docstring)."""
    x = prices[prices["ticker"] == ticker].set_index("date")
    ohlc = (x["open"] == x["high"]) & (x["high"] == x["low"]) & (x["low"] == x["close"])
    vol0 = pd.to_numeric(x["volume"], errors="coerce") == 0
    return (ohlc | vol0).fillna(False).astype(bool)


def check_segments(prices: pd.DataFrame, segments: tuple[Segment, ...], sessions: list[date], read_until: date,
                   splits: list[dict], tolerance: float, stale_until: date | None = None,
                   values: bool = True) -> tuple[list[str], list[dict]]:
    """``(violations, rows)``: what the fetched data say about the segments, on sessions <= ``read_until``.

    ``rows`` are the measurements (local output only: real-data values, ADR 0001 D2). With
    ``values=False`` (the look's preflight, review B) a violation names the problem, the ticker and the
    date, never a measured number."""
    prices = prices[prices["date"] <= read_until]
    sessions = [d for d in sessions if d <= read_until]
    violations: list[str] = []
    rows: list[dict] = []
    returns = {t: returns_on_sessions(prices, t, sessions) for t in sorted({s.ticker for s in segments})}
    for s in segments:
        have = set(prices.loc[prices["ticker"] == s.ticker, "date"])
        span = [d for d in sessions if s.contains_close(d)]
        first = min(have) if have else None
        missing = [d for d in span if d not in have]
        if s.usable and (first is None or first > s.valid_from):
            violations.append(f"{s.ticker} segment from {s.valid_from}: the data start {first}, after valid_from")
        if s.usable and missing:
            violations.append(f"{s.ticker} segment from {s.valid_from}: {len(missing)} missing closes, first {missing[0]}")
        stale = stale_prints(prices, s.ticker).reindex(span).fillna(False).astype(bool)
        stale_dev = [d for d in span if stale.get(d, False) and (stale_until is None or d <= stale_until)]
        if s.usable and stale_dev:
            violations.append(f"{s.ticker} segment from {s.valid_from}: {len(stale_dev)} stale prints, first {stale_dev[0]}, "
                              f"last {stale_dev[-1]} (O=H=L=C or volume 0)")
        row = {"check": "segment", "ticker": s.ticker, "valid_from": s.valid_from, "valid_to": s.valid_to,
               "usable": s.usable, "sessions": len(span), "missing": len(missing), "slope": math.nan,
               "expected_slope": math.nan, "n_overlap": 0, "stale": int(stale.sum())}
        if s.ticker != REFERENCE_TICKER:
            ref = [r for r in segments if r.ticker == REFERENCE_TICKER and r.usable]
            pairs = []
            for previous, day in zip(sessions, sessions[1:]):
                if segment_of_return(segments, s.ticker, previous, day) is s:
                    r_ref = next((r for r in ref if r.contains_close(previous) and r.contains_close(day)), None)
                    if r_ref is not None:
                        pairs.append((day, r_ref.leverage))
            if pairs:
                days = [d for d, _ in pairs]
                x, y = returns[REFERENCE_TICKER].reindex(days), returns[s.ticker].reindex(days)
                ok = x.notna() & y.notna()
                row["n_overlap"] = int(ok.sum())
                if ok.sum() >= MIN_OVERLAP:
                    slope = float(np.polyfit(x[ok].to_numpy(), y[ok].to_numpy(), 1)[0])
                    expected = s.leverage / pairs[0][1]
                    row.update(slope=slope, expected_slope=expected)
                    if s.usable and abs(slope - expected) > tolerance:
                        violations.append(f"{s.ticker} segment from {s.valid_from}: return slope on {REFERENCE_TICKER} "
                                          + (f"is {slope:.3f}, expected {expected:.3f} +- {tolerance}" if values else
                                             "outside the tolerance of the stated leverage") + " (exposure)")
        rows.append(row)
    for split in splits:
        day = parse_date("splits.date", split["date"])
        if day > read_until or day not in sessions:
            continue
        r = returns[split["ticker"]].get(day, math.nan) if split["ticker"] in returns else math.nan
        size = abs(math.log(float(split["ratio"])))
        jump = abs(math.log1p(r)) if math.isfinite(r) else math.nan
        unadjusted = not math.isfinite(jump) or jump >= 0.5 * size
        rows.append({"check": "split", "ticker": split["ticker"], "valid_from": day, "valid_to": day, "usable": True,
                     "sessions": 1, "missing": int(not math.isfinite(r)), "slope": jump, "expected_slope": size,
                     "n_overlap": 0, "stale": 0})
        if unadjusted:
            violations.append(f"{split['ticker']} split {day} (ratio {split['ratio']}): " + (
                f"the adjusted close moves {jump:.3f} in log terms that day, at least half the split's {size:.3f}: "
                if values else "") + "not adjusted")
    return violations, rows


__all__ = ["Segment", "load_segments", "segment_of_return", "usable_return_dates", "returns_on_sessions", "check_segments", "stale_prints",
           "REFERENCE_TICKER", "TRADABLES"]
