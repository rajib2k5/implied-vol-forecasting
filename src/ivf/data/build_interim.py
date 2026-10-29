"""Raw snapshots to typed, de-duplicated interim parquet (ADR 0001 D3; data-model.md section 3).

    uv run python -m ivf.data build-interim
    uv run python -m ivf.data build-interim --snapshot cboe/VIX=<sha256>

Two tables, one per source kind, not joined:

* ``interim/iv_index_daily.parquet``: every index in ``configs/iv_indices.yaml``,
  parsed from Cboe-shaped CSVs;
* ``interim/price_daily.parquet``: every ticker in ``configs/instruments.yaml``,
  parsed from the CSV that ``fetch.yfinance_csv_bytes`` defines.

Each table uses the latest snapshot per ``(source, symbol)`` unless
``--snapshot`` pins one, and every row keeps the ``raw_sha256`` it came from.

Parsing is strict. Cboe's files are not uniform (some have OPEN/HIGH/LOW/CLOSE,
some only a close column named after the index; header case varies), so the
parser accepts a short list of layouts it has seen and refuses anything else,
naming the file and the line. Guessing would turn a format change at Cboe into
silently wrong numbers.

Exact duplicate rows (same date, same values) are kept once and counted in
``interim/build_report.json``. A date that appears twice with *different*
values is an error: there is no rule for which one is right.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ivf import repo_guard
from ivf.data import raw

IV_TABLE = "iv_index_daily"
PRICE_TABLE = "price_daily"
REPORT = "build_report.json"

IV_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("raw_sha256", pa.string(), nullable=False),
    ]
)
PRICE_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("adj_close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64()),
        pa.field("raw_sha256", pa.string(), nullable=False),
    ]
)

_US_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_INTEGER = re.compile(r"-?\d+")


class InterimError(ValueError):
    """A raw file does not parse, or the snapshot selection is impossible."""


@dataclass
class Parsed:
    """Rows of one file, plus what de-duplication removed."""

    rows: list[dict[str, Any]]
    duplicates_dropped: int = 0
    duplicate_dates: list[str] = field(default_factory=list)


def _decode(content: bytes, where: str) -> list[list[str]]:
    try:
        text = content.decode("utf-8-sig")  # tolerate a byte-order mark, nothing else
    except UnicodeDecodeError as exc:
        raise InterimError(f"{where}: not UTF-8 text: {exc}") from exc
    rows = [row for row in csv.reader(io.StringIO(text, newline=""))]
    while rows and rows[-1] == []:  # trailing blank lines only
        rows.pop()
    if not rows:
        raise InterimError(f"{where}: empty file")
    for number, row in enumerate(rows, start=1):
        if row == []:
            raise InterimError(f"{where}:{number}: blank line inside the file")
    return rows


def _parse_dates(cells: list[str], where: str) -> list[date]:
    """One date format per file, either M/D/YYYY (Cboe) or YYYY-MM-DD. Mixed formats are an error."""
    first = cells[0].strip()
    if _US_DATE.fullmatch(first):
        pattern, order = _US_DATE, ("m", "d", "y")
    elif _ISO_DATE.fullmatch(first):
        pattern, order = _ISO_DATE, ("y", "m", "d")
    else:
        raise InterimError(f"{where}:2: unrecognised date {first!r}; expected MM/DD/YYYY or YYYY-MM-DD")
    out = []
    for number, cell in enumerate(cells, start=2):
        match = pattern.fullmatch(cell.strip())
        if not match:
            raise InterimError(f"{where}:{number}: date {cell!r} does not match the file's format ({pattern.pattern})")
        parts = dict(zip(order, (int(g) for g in match.groups())))
        try:
            out.append(date(parts["y"], parts["m"], parts["d"]))
        except ValueError as exc:
            raise InterimError(f"{where}:{number}: {cell!r} is not a calendar date ({exc})") from exc
    return out


def _float(cell: str, where: str, *, required: bool) -> float | None:
    text = cell.strip()
    if text == "":
        if required:
            raise InterimError(f"{where}: missing value")
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise InterimError(f"{where}: {cell!r} is not a number") from exc
    if not math.isfinite(value):  # float() accepts "nan" and "inf"; data does not
        raise InterimError(f"{where}: {cell!r} is not a finite number")
    return value


def _deduplicate(rows: list[dict[str, Any]], where: str) -> Parsed:
    by_date: dict[date, dict[str, Any]] = {}
    parsed = Parsed(rows=[])
    for row in rows:
        seen = by_date.get(row["date"])
        if seen is None:
            by_date[row["date"]] = row
        elif seen == row:
            parsed.duplicates_dropped += 1
            parsed.duplicate_dates.append(row["date"].isoformat())
        else:
            raise InterimError(f"{where}: {row['date']} appears twice with different values ({seen} vs {row})")
    parsed.rows = [by_date[d] for d in sorted(by_date)]
    return parsed


def parse_cboe_csv(content: bytes, symbol: str, where: str = "cboe") -> Parsed:
    """Parse a Cboe daily history CSV.

    Accepted layouts (header compared case-insensitively, spaces stripped):

    * ``DATE,OPEN,HIGH,LOW,CLOSE``
    * ``DATE,CLOSE`` or ``DATE,<SYMBOL>`` (close only, e.g. VVIX's ``DATE,VVIX``)

    Rules: a close is required and must be > 0; an open, high or low may be
    empty (null) but, if present, must be > 0. A zero there would be a
    placeholder, not a level, and the parser does not guess which.
    """
    rows = _decode(content, where)
    header = [cell.strip().upper() for cell in rows[0]]
    if header == ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
        columns = ["open", "high", "low", "close"]
    elif header in (["DATE", "CLOSE"], ["DATE", symbol.upper()]):
        columns = ["close"]
    else:
        raise InterimError(
            f"{where}:1: unexpected header {rows[0]}; expected DATE,OPEN,HIGH,LOW,CLOSE or DATE,CLOSE or DATE,{symbol}"
        )
    body = rows[1:]
    if not body:
        raise InterimError(f"{where}: a header and no rows")
    for number, row in enumerate(body, start=2):
        if len(row) != len(header):
            raise InterimError(f"{where}:{number}: {len(row)} fields, the header has {len(header)}")
    dates = _parse_dates([row[0] for row in body], where)
    out = []
    for number, (day, row) in enumerate(zip(dates, body), start=2):
        values = {"open": None, "high": None, "low": None}
        for name, cell in zip(columns, row[1:]):
            values[name] = _float(cell, f"{where}:{number}: {name}", required=name == "close")
        for name, value in values.items():
            if value is not None and value <= 0:
                raise InterimError(f"{where}:{number}: {name} is {value}; index levels must be > 0")
        out.append({"date": day, **values})
    return _deduplicate(out, where)


def parse_yfinance_csv(content: bytes, where: str = "yfinance") -> Parsed:
    """Parse the CSV that ``fetch.yfinance_csv_bytes`` writes (and the synthetic generator imitates)."""
    from ivf.data.fetch import YF_HEADER

    rows = _decode(content, where)
    if ",".join(cell.strip() for cell in rows[0]) != YF_HEADER:
        raise InterimError(f"{where}:1: unexpected header {rows[0]}; expected {YF_HEADER}")
    body = rows[1:]
    if not body:
        raise InterimError(f"{where}: a header and no rows")
    for number, row in enumerate(body, start=2):
        if len(row) != 7:
            raise InterimError(f"{where}:{number}: {len(row)} fields, expected 7")
    dates = _parse_dates([row[0] for row in body], where)
    out = []
    for number, (day, row) in enumerate(zip(dates, body), start=2):
        values = {
            name: _float(cell, f"{where}:{number}: {name}", required=True)
            for name, cell in zip(("open", "high", "low", "close", "adj_close"), row[1:6])
        }
        for name, value in values.items():
            if value <= 0:
                raise InterimError(f"{where}:{number}: {name} is {value}; prices must be > 0")
        volume_text = row[6].strip()
        if volume_text and not _INTEGER.fullmatch(volume_text):
            raise InterimError(f"{where}:{number}: volume {row[6]!r} is not an integer")
        volume = int(volume_text) if volume_text else None
        if volume is not None and volume < 0:
            raise InterimError(f"{where}:{number}: negative volume {volume}")
        out.append({"date": day, **values, "volume": volume})
    return _deduplicate(out, where)


def parse_pins(texts: list[str]) -> dict[tuple[str, str], str]:
    """``--snapshot SOURCE/SYMBOL=SHA256`` arguments as ``{(source, symbol): sha256}``."""
    pins: dict[tuple[str, str], str] = {}
    for text in texts:
        match = re.fullmatch(r"([a-z]+)/([^=]+)=([0-9a-f]{64})", text)
        if not match:
            raise InterimError(f"--snapshot {text!r}: expected SOURCE/SYMBOL=<64 hex sha256>")
        key = (match.group(1), match.group(2))
        if key in pins:
            raise InterimError(f"--snapshot pins {key[0]}/{key[1]} twice")
        pins[key] = match.group(3)
    return pins


def select_snapshots(
    records: list[raw.FetchRecord], pins: dict[tuple[str, str], str] | None = None
) -> dict[tuple[str, str], raw.FetchRecord]:
    """The snapshot to use per ``(source, symbol)``: the latest fetch, or the pinned hash."""
    pins = pins or {}
    chosen = raw.latest_fetches(records)
    for key, sha256 in pins.items():
        matching = [r for r in records if (r.source, r.symbol) == key and r.sha256 == sha256]
        if not matching:
            raise InterimError(f"--snapshot {key[0]}/{key[1]}={sha256}: no fetch of that symbol stored that hash")
        chosen[key] = raw.latest_fetches(matching)[key]  # the latest fetch that saw this content
    return chosen


def _write_parquet(table: pa.Table, path: Path) -> None:
    """Serialise in memory, then ``repo_guard.write_file``: checked at write time, renamed over the target.

    A reader never sees half a file, and the folder is checked against the repository when the
    file is written, not only when the command starts (amended M1, review B).
    """
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    repo_guard.write_file(path, buffer.getvalue().to_pybytes(), outside_repo=True)


@dataclass
class BuildReport:
    files: list[dict[str, Any]]

    def to_json(self) -> bytes:
        return (json.dumps({"files": self.files}, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _symbol_source(chosen: dict[tuple[str, str], raw.FetchRecord], symbol: str) -> raw.FetchRecord | None:
    matches = [record for (source, sym), record in chosen.items() if sym == symbol]
    if len(matches) > 1:
        sources = sorted(record.source for record in matches)
        raise InterimError(f"{symbol} has snapshots from {sources}; milestone 1 has no rule to merge sources")
    return matches[0] if matches else None


def build_interim(
    root: Path,
    iv_symbols: list[str],
    tickers: list[str],
    pins: dict[tuple[str, str], str] | None = None,
) -> BuildReport:
    """Build both interim tables under ``root/interim`` from the raw store under ``root/raw``.

    A symbol in the registry with no snapshot is an error: a table silently
    missing VIX is worse than no table. A ``root`` inside the repository is
    refused before anything is read (``ivf.repo_guard``; amended M1, review A).
    """
    repo_guard.require_data_root_outside_repo(root)
    records = raw.read_provenance(root)
    if not records:
        raise InterimError(f"{root / 'raw' / raw.PROVENANCE}: no fetches recorded; run `python -m ivf.data fetch` first")
    chosen = select_snapshots(records, pins)
    report: list[dict[str, Any]] = []
    iv_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    for kind, symbols in (("iv", iv_symbols), ("price", tickers)):
        for symbol in symbols:
            record = _symbol_source(chosen, symbol)
            if record is None:
                raise InterimError(f"{symbol}: no snapshot in the raw store; run `python -m ivf.data fetch`")
            path = root / record.raw_path
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != record.sha256:
                raise InterimError(f"{record.raw_path}: content does not match its hash; run `python -m ivf.data verify`")
            where = record.raw_path
            parsed = parse_cboe_csv(content, symbol, where) if kind == "iv" else parse_yfinance_csv(content, where)
            if kind == "iv":
                iv_rows += [{"symbol": symbol, **row, "source": record.source, "raw_sha256": record.sha256} for row in parsed.rows]
            else:
                price_rows += [{"ticker": symbol, **row, "raw_sha256": record.sha256} for row in parsed.rows]
            report.append(
                {
                    "table": IV_TABLE if kind == "iv" else PRICE_TABLE,
                    "source": record.source,
                    "symbol": symbol,
                    "raw_sha256": record.sha256,
                    "fetched_at": record.fetched_at,
                    "rows": len(parsed.rows),
                    "first_date": parsed.rows[0]["date"].isoformat(),
                    "last_date": parsed.rows[-1]["date"].isoformat(),
                    "duplicates_dropped": parsed.duplicates_dropped,
                    "duplicate_dates": parsed.duplicate_dates,
                }
            )
    _write_parquet(pa.Table.from_pylist(iv_rows, schema=IV_SCHEMA), root / "interim" / f"{IV_TABLE}.parquet")
    _write_parquet(pa.Table.from_pylist(price_rows, schema=PRICE_SCHEMA), root / "interim" / f"{PRICE_TABLE}.parquet")
    result = BuildReport(files=report)
    repo_guard.write_file(root / "interim" / REPORT, result.to_json(), outside_repo=True)
    return result


def read_interim(root: Path):
    """``(iv_index_daily, price_daily, build_report)`` as pandas DataFrames and a dict."""
    import pandas as pd

    folder = root / "interim"
    paths = [folder / f"{IV_TABLE}.parquet", folder / f"{PRICE_TABLE}.parquet", folder / REPORT]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise InterimError(f"missing {', '.join(map(str, missing))}; run `python -m ivf.data build-interim`")
    iv = pq.read_table(paths[0]).to_pandas()
    price = pq.read_table(paths[1]).to_pandas()
    for frame in (iv, price):
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return iv, price, json.loads(paths[2].read_text())


__all__ = [
    "InterimError",
    "parse_cboe_csv",
    "parse_yfinance_csv",
    "select_snapshots",
    "build_interim",
    "read_interim",
    "parse_pins",
]
