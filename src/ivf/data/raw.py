"""The raw store: content-addressed snapshots plus an append-only log (ADR 0001 D3).

Layout under a data root (``ROOT`` or ``ROOT/synthetic``)::

    raw/{source}/{symbol}/{sha256}.csv   the bytes as downloaded, named by their hash
    raw/provenance.jsonl                 one JSON object per fetch, never rewritten

Why content addressing. A file named by the hash of its bytes cannot be
overwritten by different bytes, so every result can be rebuilt from exactly the
snapshot it used, and identical downloads are stored once.

Why JSONL for the log. Appending a line cannot corrupt the lines already
written. Rewriting a parquet file could, if the process died half way.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ivf import repo_guard

PROVENANCE = "provenance.jsonl"
#: The columns of raw_fetch (data-model.md section 2), in this order.
FIELDS = (
    "fetch_id",
    "source",
    "symbol",
    "url",
    "final_url",
    "fetched_at",
    "http_status",
    "bytes",
    "sha256",
    "raw_path",
    "changed",
    "tool_version",
)
SOURCES = ("cboe", "fred", "yfinance", "synthetic")
# A symbol becomes a folder name, so it must not be able to climb out of raw/.
_SYMBOL = re.compile(r"\^?[A-Z0-9][A-Z0-9.=_-]{0,31}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TMP_PREFIX = ".tmp-"


class RawStoreError(RuntimeError):
    """The raw store or its log is not in the state the code relies on."""


@dataclass(frozen=True)
class FetchRecord:
    """One line of provenance.jsonl: one fetch (raw_fetch in the data model)."""

    fetch_id: str
    source: str
    symbol: str
    url: str
    final_url: str | None
    fetched_at: str  # RFC 3339 UTC with microseconds, e.g. 2026-10-29T12:00:00.000000Z
    http_status: int | None
    bytes: int
    sha256: str
    raw_path: str  # relative to the data root
    changed: bool
    tool_version: str

    @property
    def fetched_at_utc(self) -> datetime:
        return parse_utc(self.fetched_at)


def format_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("a naive datetime names no instant; pass one with tzinfo=timezone.utc")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def check_source_symbol(source: str, symbol: str) -> None:
    if source not in SOURCES:
        raise RawStoreError(f"unknown source {source!r}; expected one of {SOURCES}")
    if not _SYMBOL.fullmatch(symbol):
        raise RawStoreError(f"symbol {symbol!r} is not a safe folder name")


def raw_path_for(source: str, symbol: str, sha256: str) -> str:
    """Where a snapshot lives, relative to the data root."""
    check_source_symbol(source, symbol)
    return f"raw/{source}/{symbol}/{sha256}.csv"


def store_snapshot(root: Path, source: str, symbol: str, content: bytes) -> tuple[str, str, bool]:
    """Store ``content`` as ``raw/{source}/{symbol}/{sha256}.csv`` if it is new.

    Returns ``(sha256, raw_path, changed)``. ``changed`` is False when a file
    with this hash was already there, and the store is then left untouched.

    ``repo_guard.write_file(..., how="new")`` does the writing (amended M1, review B): it checks,
    at write time, that the folder is not inside the repository, writes a temporary file in it,
    flushes it, and *links* it to its final name. ``os.link`` refuses an existing name, so "create
    only if absent" is one atomic step: two concurrent fetches of the same bytes cannot clobber each
    other, and a crash leaves at most a ``.{name}.*.tmp`` file, never a half-written
    ``{sha256}.csv``. The folder is then flushed (``fsync``), so the new name survives a crash.

    A crash after the link but before ``append_provenance`` leaves a snapshot with no log line (an
    orphan); ``verify`` reports it, and ``fetch_one`` checks the log can take a line *before* it
    stores anything, so a torn log no longer makes one.
    """
    check_source_symbol(source, symbol)
    sha256 = hashlib.sha256(content).hexdigest()
    relative = raw_path_for(source, symbol, sha256)
    changed = repo_guard.write_file(root / relative, content, outside_repo=True, how="new")
    return sha256, relative, changed


def require_appendable_log(root: Path) -> None:
    """Raise ``RawStoreError`` if ``raw/provenance.jsonl`` ends in a torn line (no final newline).

    A crash mid-write leaves one; a new line would be glued onto it and both would be lost to the
    parser. ``fetch_one`` calls this before storing a snapshot, so a torn log stops the fetch before
    it can leave a snapshot that no line will ever describe (amended M1, review B).
    """
    path = root / "raw" / PROVENANCE
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise RawStoreError(f"{path}: the last line has no newline (a crash mid-write?); fix it by hand first")


def append_provenance(root: Path, record: FetchRecord) -> None:
    """Append one line to ``raw/provenance.jsonl`` and flush it to disk.

    Refuses to append after a torn last line (one without its newline, left by
    a crash mid-write): the new line would be glued onto it and both would be
    lost to the parser.
    """
    require_appendable_log(root)
    line = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    repo_guard.write_file(root / "raw" / PROVENANCE, line.encode("utf-8"), outside_repo=True, how="append")


def read_provenance(root: Path) -> list[FetchRecord]:
    """Every fetch record, in the order written. An unreadable line is an error, never skipped."""
    path = root / "raw" / PROVENANCE
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or set(data) != set(FIELDS):
                raise ValueError(f"expected exactly the keys {FIELDS}")
            record = FetchRecord(**data)
            check_source_symbol(record.source, record.symbol)
            if not _SHA256.fullmatch(record.sha256) or record.raw_path != raw_path_for(
                record.source, record.symbol, record.sha256
            ):
                raise ValueError("sha256 or raw_path is malformed")
            record.fetched_at_utc  # parses, or raises ValueError
        except (ValueError, TypeError, RawStoreError) as exc:
            raise RawStoreError(f"{path}:{number}: unreadable provenance line: {exc}") from exc
        records.append(record)
    return records


def latest_fetches(records: list[FetchRecord]) -> dict[tuple[str, str], FetchRecord]:
    """The most recent fetch per ``(source, symbol)``.

    "Most recent" is the latest ``fetched_at``; on a tie, the later line in the
    log. A fetch that found unchanged content still counts: its snapshot is the
    same file, and its time is when that content was last confirmed current.
    """
    latest: dict[tuple[str, str], tuple[datetime, int, FetchRecord]] = {}
    for order, record in enumerate(records):
        key = (record.source, record.symbol)
        candidate = (record.fetched_at_utc, order, record)
        if key not in latest or candidate[:2] > latest[key][:2]:
            latest[key] = candidate
    return {key: value[2] for key, value in latest.items()}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> tuple[list[str], list[str]]:
    """Re-hash every raw file (``ivf data verify``, ADR 0001 D3 step 5).

    Returns ``(problems, warnings)``. Problems make the command fail:

    * a ``{sha256}.csv`` whose bytes no longer hash to its name;
    * a provenance line whose file is missing, or whose ``bytes`` disagrees;
    * a ``.csv`` that no provenance line accounts for (an orphan: a crash
      between storing the snapshot and appending its line, or something wrote
      around ``fetch``), or a file in ``raw/`` that is neither a snapshot nor
      the log;
    * an unreadable provenance log.

    A leftover temporary file (``.{name}.*.tmp``, or ``.tmp-*`` from before
    review B; a crash before the link) is only a warning: it is not a snapshot
    and nothing reads it.
    """
    problems: list[str] = []
    warnings: list[str] = []
    raw = root / "raw"
    try:
        records = read_provenance(root)
    except RawStoreError as exc:
        return [str(exc)], warnings
    referenced = {record.raw_path for record in records}
    seen_paths: set[str] = set()
    if raw.is_dir():
        for path in sorted(p for p in raw.rglob("*") if p.is_file() or p.is_symlink()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                problems.append(f"{relative}: is a symlink; the raw store holds real files only")
            elif path.name == PROVENANCE and path.parent == raw:
                continue
            elif path.name.startswith(_TMP_PREFIX) or (path.name.startswith(".") and path.name.endswith(".tmp")):
                warnings.append(f"{relative}: leftover temporary file from an interrupted fetch (safe to delete)")
            elif path.suffix == ".csv" and _SHA256.fullmatch(path.stem):
                seen_paths.add(relative)
                actual = sha256_of_file(path)
                if actual != path.stem:
                    problems.append(f"{relative}: content hashes to {actual}, not to its name")
                if relative not in referenced:
                    problems.append(
                        f"{relative}: no provenance line refers to this file (an orphan: a fetch that stopped between "
                        "storing the snapshot and logging it, or a file written around fetch); fetch it again with "
                        "--force to log it, or delete it"
                    )
            else:
                problems.append(f"{relative}: unexpected file in the raw store")
    for record in records:
        path = root / record.raw_path
        if record.raw_path not in seen_paths:
            problems.append(f"{record.raw_path}: fetch {record.fetch_id} refers to it, but it is missing")
        elif path.stat().st_size != record.bytes:
            problems.append(f"{record.raw_path}: fetch {record.fetch_id} recorded {record.bytes} bytes, file has {path.stat().st_size}")
    return problems, warnings
