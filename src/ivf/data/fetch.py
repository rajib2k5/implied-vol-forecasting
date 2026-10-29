"""Fetch Cboe index histories and Yahoo Finance prices into the raw store (ADR 0001 D3).

    uv run python -m ivf.data fetch              # every symbol older than --max-age (default 1d)
    uv run python -m ivf.data fetch --force      # every symbol, now

Each fetch:

1. asks ``network_get`` for the data. It is the ONLY function in ``ivf`` that
   touches the network, so tests replace it and never go online;
2. turns the answer into bytes (a yfinance DataFrame goes through
   ``yfinance_csv_bytes``, which fixes the exact bytes stored for it);
3. stores the bytes as ``raw/{source}/{symbol}/{sha256}.csv`` if that file is
   absent (``raw.store_snapshot``);
4. appends one line to ``raw/provenance.jsonl``, also when the bytes were
   already stored (``changed`` is then false).

A symbol fetched less than ``--max-age`` ago is skipped with no network call
and no log line: nothing was fetched, so there is nothing to record.

A failed download (HTTP status other than 200, an empty body, an empty
DataFrame) raises ``FetchError``. Nothing is stored and nothing is logged,
because the log describes snapshots, and there is none.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import ivf
from ivf import repo_guard
from ivf.data import raw

#: Columns yfinance returns with ``auto_adjust=False, actions=False``, and the
#: names they are stored under.
YF_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
YF_HEADER = "date,open,high,low,close,adj_close,volume"
#: The exchange time zone yfinance stamps ^GSPC daily bars with.
YF_TIMEZONE = "America/New_York"
#: The request, spelled as a URL so the provenance line says what was asked for.
YF_URL = "yfinance://{symbol}?period=max&interval=1d&auto_adjust=False&actions=False"


class FetchError(RuntimeError):
    """A download failed or returned something that is not data."""


@dataclass(frozen=True)
class Request:
    source: str  # "cboe" or "yfinance"
    symbol: str
    url: str


@dataclass(frozen=True)
class Response:
    """What the network gave back. Exactly one of ``body`` and ``frame`` is set."""

    body: bytes | None
    frame: pd.DataFrame | None
    final_url: str | None
    http_status: int | None
    tool_version: str


def network_get(request: Request) -> Response:
    """The one function that touches the network. Tests replace it.

    Cboe answers the published ``cdn.cboe.com`` URL with a 307 redirect to
    ``cdn-api.cboe.com``; ``requests`` follows it, and ``final_url`` records
    where the bytes actually came from.
    """
    if request.source == "cboe":
        import requests

        answer = requests.get(
            request.url, timeout=60, allow_redirects=True, headers={"User-Agent": f"ivf/{ivf.__version__}"}
        )
        return Response(
            body=answer.content,
            frame=None,
            final_url=answer.url,
            http_status=answer.status_code,
            tool_version=f"requests {requests.__version__}",
        )
    if request.source == "yfinance":
        import warnings

        import yfinance

        # Raise on failure instead of logging it and returning an empty frame. yfinance 1.7.0
        # deprecates history(raise_errors=True) in favour of this switch (DeprecationWarning in
        # yfinance/scrapers/history.py; amended M1, review A). It is global, so it is restored.
        debug = yfinance.config.debug
        hide_exceptions = debug.hide_exceptions
        debug.hide_exceptions = False
        try:
            with warnings.catch_warnings():
                # yfinance 1.7.0 calls pandas methods with inplace=True, which pandas
                # 3.1 deprecates (Pandas4Warning). Harmless here, and not ours to fix.
                warnings.filterwarnings("ignore", message=".*inplace keyword.*")
                frame = yfinance.Ticker(request.symbol).history(period="max", interval="1d", auto_adjust=False, actions=False)
        finally:
            debug.hide_exceptions = hide_exceptions
        return Response(body=None, frame=frame, final_url=None, http_status=None, tool_version=f"yfinance {yfinance.__version__}")
    raise FetchError(f"no network access is defined for source {request.source!r}")


def _number(value: float) -> str:
    """Six decimals, like Cboe's own files; a missing value is an empty field."""
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.6f}"


def yfinance_csv_bytes(frame: pd.DataFrame) -> bytes:
    """The exact bytes stored for a yfinance daily DataFrame.

    yfinance returns a DataFrame, not a file, so "the bytes as downloaded"
    has to be defined. Here it is, so that the same data always hashes the same:

    * header ``date,open,high,low,close,adj_close,volume``, then one row per bar;
    * ``date`` is the session date in New York (``YYYY-MM-DD``), taken from the
      index, which must be tz-aware ``America/New_York`` and at midnight;
    * prices with six decimals (``%.6f``), volume as an integer, a missing
      value as an empty field;
    * rows sorted by date (stable, so duplicates keep their order and are
      left for ``build_interim`` to judge), ``\\n`` line ends, ASCII.

    Six decimals round the float64 that yfinance holds: re-reading gives a
    value within 5e-7 of it. For index levels of 300 and up, that changes a
    log return by under 2e-9.

    Anything else (another column, a naive or non-New-York index, a bar not
    at midnight) raises ``FetchError`` instead of being stored.
    """
    if frame.empty:
        raise FetchError("yfinance returned no rows")
    if set(frame.columns) != set(YF_COLUMNS):
        raise FetchError(f"yfinance columns {list(frame.columns)}, expected exactly {list(YF_COLUMNS)}")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != YF_TIMEZONE:
        raise FetchError(f"yfinance index must be tz-aware {YF_TIMEZONE}, got {getattr(index, 'tz', None)!r}")
    if not (index == index.normalize()).all():
        raise FetchError("a yfinance daily bar is not stamped at midnight New York time")
    ordered = frame.assign(_date=index.strftime("%Y-%m-%d")).sort_values("_date", kind="stable")
    lines = [YF_HEADER]
    for row in ordered.itertuples(index=False):
        values = dict(zip(ordered.columns, row))
        volume = values["Volume"]
        volume_text = "" if pd.isna(volume) else str(int(volume))
        lines.append(
            ",".join(
                [values["_date"]]
                + [_number(values[name]) for name in ("Open", "High", "Low", "Close", "Adj Close")]
                + [volume_text]
            )
        )
    return ("\n".join(lines) + "\n").encode("ascii")


@dataclass(frozen=True)
class Outcome:
    source: str
    symbol: str
    action: str  # "fetched" or "skipped"
    record: raw.FetchRecord | None
    message: str


def fetch_one(
    root: Path,
    request: Request,
    *,
    max_age: timedelta,
    force: bool,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    network: Callable[[Request], Response] | None = None,
) -> Outcome:
    """Fetch one symbol into the raw store under ``root``, unless it is fresh enough.

    A ``root`` inside the repository is refused first (``ivf.repo_guard``; amended M1, review A).
    So is a provenance log with a torn last line, before the download and before anything is
    stored: the first version stored the snapshot and only then found it could not log it, which
    left an orphan (amended M1, review B).
    """
    repo_guard.require_data_root_outside_repo(root)
    network = network or network_get
    raw.check_source_symbol(request.source, request.symbol)
    raw.require_appendable_log(root)
    if not force:
        last = raw.latest_fetches(raw.read_provenance(root)).get((request.source, request.symbol))
        if last is not None:
            age = clock() - last.fetched_at_utc
            if age < max_age:
                return Outcome(request.source, request.symbol, "skipped", None, f"fetched {age} ago (< --max-age {max_age})")
    response = network(request)
    fetched_at = clock()  # when the answer arrived: the vintage of the data
    if request.source == "cboe":
        if response.http_status != 200:
            raise FetchError(f"{request.url}: HTTP {response.http_status}")
        if not response.body:
            raise FetchError(f"{request.url}: empty body")
        content = response.body
    else:
        if response.frame is None:
            raise FetchError(f"{request.url}: no DataFrame")
        content = yfinance_csv_bytes(response.frame)
    sha256, raw_path, changed = raw.store_snapshot(root, request.source, request.symbol, content)
    record = raw.FetchRecord(
        fetch_id=uuid.uuid4().hex,
        source=request.source,
        symbol=request.symbol,
        url=request.url,
        final_url=response.final_url,
        fetched_at=raw.format_utc(fetched_at),
        http_status=response.http_status,
        bytes=len(content),
        sha256=sha256,
        raw_path=raw_path,
        changed=changed,
        tool_version=f"ivf {ivf.__version__}; {response.tool_version}",
    )
    raw.append_provenance(root, record)
    state = "new content" if changed else "unchanged"
    return Outcome(request.source, request.symbol, "fetched", record, f"{len(content)} bytes, {state}, sha256 {sha256[:12]}")


def requests_for(indices, instruments) -> list[Request]:
    """One request per Cboe index (its ``cboe_url``) and per instrument (yfinance)."""
    requests_ = [Request("cboe", index.symbol, index.cboe_url) for index in indices]
    requests_ += [Request("yfinance", item.ticker, YF_URL.format(symbol=item.ticker)) for item in instruments]
    return requests_


def parse_max_age(text: str) -> timedelta:
    """``--max-age``: a number and a unit, ``s``, ``m``, ``h`` or ``d`` (``0`` alone also works)."""
    text = text.strip()
    if text == "0":
        return timedelta(0)
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    if len(text) < 2 or text[-1] not in units:
        raise ValueError(f"--max-age {text!r}: use a number and a unit, e.g. 1d, 12h, 30m")
    try:
        amount = float(text[:-1])
    except ValueError as exc:
        raise ValueError(f"--max-age {text!r}: {text[:-1]!r} is not a number") from exc
    if amount < 0 or math.isnan(amount) or math.isinf(amount):
        raise ValueError(f"--max-age {text!r}: must be a finite number >= 0")
    return timedelta(**{units[text[-1]]: amount})
