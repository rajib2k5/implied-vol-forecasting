"""The tradable price table, and the synthetic VXX/SVXY (milestone 6).

``build_tradable_table`` reads the latest raw snapshot of each ``role: tradable`` instrument (fetched by
``python -m ivf.data fetch`` exactly like ^GSPC, with provenance) and writes
``interim/tradable_daily.parquet`` in the ``price_daily`` schema. It is a separate table, not rows of
``price_daily``, so adding the ETPs moves no M1-M5 table and no run_id (amended M6).

``synthetic_tradables`` makes a VXX and an SVXY for the synthetic market, with known structure, so the
real code path runs in CI. It is a toy, not a model of the products:

    VXX log return_d = 0.5 * (x_d - x_{d-1}) - (ln VIX3M_{d-1} - ln VIX_{d-1}) / 21
    SVXY simple return_d = L_d * VXX simple return_d,  L_d = -1 up to 2018-02-27, -0.5 after

with ``x = ln VIX``: VXX follows half the VIX's moves and pays the roll (it bleeds in contango, VIX3M >
VIX, and gains in backwardation); SVXY is a daily-rebalanced inverse with the real leverage cut, so
the segment logic has a break to find. VXX starts on 2018-01-25 and SVXY on 2011-10-04, as on Yahoo.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import ivf
from ivf import repo_guard
from ivf.data import build_interim, raw, synthetic

TABLE = "tradable_daily"
FIRST = {"VXX": date(2018, 1, 25), "SVXY": date(2011, 10, 4)}
CUT = date(2018, 2, 27)  # the last close at -1x


def build_tradable_table(root: Path, tickers: list[str]) -> list[dict]:
    """Write ``interim/tradable_daily.parquet`` from the latest snapshot of each ticker; returns one report row each."""
    repo_guard.require_data_root_outside_repo(root)
    chosen = build_interim.select_snapshots(raw.read_provenance(root), None)
    rows, report = [], []
    for ticker in tickers:
        record = build_interim._symbol_source(chosen, ticker)
        if record is None:
            raise build_interim.InterimError(f"{ticker}: no snapshot in the raw store; run `python -m ivf.data fetch --symbols {ticker}`")
        content = (root / record.raw_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise build_interim.InterimError(f"{record.raw_path}: content does not match its hash")
        parsed = build_interim.parse_yfinance_csv(content, record.raw_path)
        rows += [{"ticker": ticker, **row, "raw_sha256": record.sha256} for row in parsed.rows]
        report.append({"ticker": ticker, "source": record.source, "rows": len(parsed.rows), "first": parsed.rows[0]["date"],
                       "last": parsed.rows[-1]["date"], "fetched_at": record.fetched_at, "raw_sha256": record.sha256})
    build_interim._write_parquet(pa.Table.from_pylist(rows, schema=build_interim.PRICE_SCHEMA), root / "interim" / f"{TABLE}.parquet")
    return report


def read_tradable_table(root: Path) -> pd.DataFrame:
    path = root / "interim" / f"{TABLE}.parquet"
    if not path.exists():
        raise build_interim.InterimError(f"missing {path}; run `python -m ivf.backtest prepare`")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def synthetic_tradables(iv: pd.DataFrame) -> pd.DataFrame:
    """VXX and SVXY price rows (``price_daily`` columns) from the synthetic VIX and VIX3M (module docstring)."""
    wide = iv.pivot(index="date", columns="symbol", values="close").sort_index()
    x = np.log(wide["VIX"])
    roll = -(np.log(wide["VIX3M"]) - x).shift(1) / 21
    vxx_log = 0.5 * x.diff() + roll
    out = []
    for ticker in ("VXX", "SVXY"):
        days = wide.index[wide.index >= FIRST[ticker]]
        r = np.expm1(vxx_log.reindex(days).fillna(0.0).to_numpy())
        r[0] = 0.0
        if ticker == "SVXY":
            lev = np.array([-1.0 if d <= CUT else -0.5 for d in days])
            r = np.maximum(lev * r, -0.95)
        close = 100.0 * np.cumprod(1.0 + r)
        previous = np.concatenate([[100.0], close[:-1]])
        out.append(pd.DataFrame({"ticker": ticker, "date": list(days), "open": previous, "high": np.maximum(previous, close),
                                 "low": np.minimum(previous, close), "close": close, "adj_close": close,
                                 "volume": pd.array([None] * len(days), dtype="Int64")}))
    return pd.concat(out, ignore_index=True)


def write_synthetic_tradables(root: Path, protocol) -> list[raw.FetchRecord]:
    """Store the synthetic VXX and SVXY in ``root/raw`` exactly as ``fetch`` stores a download (source ``synthetic``)."""
    repo_guard.require_data_root_outside_repo(root)
    raw.require_appendable_log(root)
    iv, _ = synthetic.synthetic_market(protocol)
    prices = synthetic_tradables(iv)
    end = protocol.require_data_end()
    fetched_at = datetime(end.year, end.month, end.day, 12, tzinfo=timezone.utc) + pd.Timedelta(days=1)
    records = []
    for ticker in ("VXX", "SVXY"):
        content = synthetic.yfinance_shaped_csv_bytes(prices[prices.ticker == ticker].reset_index(drop=True))
        sha256, raw_path, changed = raw.store_snapshot(root, "synthetic", ticker, content)
        record = raw.FetchRecord(fetch_id=uuid.uuid4().hex, source="synthetic", symbol=ticker,
                                 url=f"synthetic://{ticker}?seed=vix-derived", final_url=None,
                                 fetched_at=raw.format_utc(fetched_at), http_status=None, bytes=len(content), sha256=sha256,
                                 raw_path=raw_path, changed=changed, tool_version=f"ivf {ivf.__version__}; ivf.backtest.data")
        raw.append_provenance(root, record)
        records.append(record)
    return records


__all__ = ["TABLE", "build_tradable_table", "read_tradable_table", "synthetic_tradables", "write_synthetic_tradables"]
