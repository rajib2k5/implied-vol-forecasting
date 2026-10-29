"""The evaluation protocol and the symbol registries (ADR 0001 D3, D4.6).

``configs/protocol.yaml`` (real data) and ``configs/protocol.synthetic.yaml``
hold the protocol. ``configs/iv_indices.yaml`` and ``configs/instruments.yaml``
hold the registries of what is fetched. This module reads them strictly: an
unknown key, a missing key or a value of the wrong type is an error, because a
typo in a protocol file would otherwise change the evaluation silently.

``protocol_hash`` (ADR 0001 D3) is the SHA-256 of the canonical JSON of the
parsed protocol file: keys sorted, compact separators, JSON types only
(``contract.canonical_json_sha256``). That is why the YAML quotes its dates.
Unquoted, YAML turns ``1990-01-02`` into a ``datetime.date``, which has no JSON
spelling, and the hash would be undefined.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ivf import config, contract

PROTOCOL_FILE = "protocol.yaml"
SYNTHETIC_PROTOCOL_FILE = "protocol.synthetic.yaml"
IV_INDICES_FILE = "iv_indices.yaml"
INSTRUMENTS_FILE = "instruments.yaml"

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_PROTOCOL_KEYS = {
    "train_start",
    "dev_start",
    "holdout_start",
    "data_end",
    "horizons",
    "fold_len",
    "min_cell_n",
    "vix_regime_edges",
    "holdout_looks",
    "seed",
    "race_context_sessions",
    "targets",
}
_TARGET_KEYS = {"target_id", "label", "units", "horizons", "definition"}
_INDEX_KEYS = {"symbol", "name", "tenor_days", "underlying", "group", "cboe_url", "fred_series"}
_INSTRUMENT_KEYS = {"ticker", "role", "name"}


class ProtocolError(ValueError):
    """A protocol or registry file is malformed, or a freeze would move a frozen value."""


@dataclass(frozen=True)
class Protocol:
    """The parsed protocol. ``document`` is the file's content as JSON types."""

    path: Path
    document: dict[str, Any]
    train_start: date
    dev_start: date
    holdout_start: date
    data_end: date | None
    horizons: tuple[int, ...]
    fold_len: int
    vix_regime_edges: tuple[float, float]

    @property
    def protocol_hash(self) -> str:
        return contract.canonical_json_sha256(self.document)

    @property
    def h_max(self) -> int:
        """The longest horizon: how far past ``data_end`` the master calendar must reach."""
        return max(self.horizons)

    def require_data_end(self) -> date:
        if self.data_end is None:
            raise ProtocolError(
                f"{self.path}: data_end is not frozen yet. Run: uv run python -m ivf.data check --freeze-data-end"
            )
        return self.data_end


@dataclass(frozen=True)
class IvIndex:
    symbol: str
    name: str
    tenor_days: int | None
    underlying: str | None
    group: str
    cboe_url: str
    fred_series: str | None


@dataclass(frozen=True)
class Instrument:
    ticker: str
    role: str
    name: str


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProtocolError(f"{path}: file not found") from exc
    except yaml.YAMLError as exc:
        raise ProtocolError(f"{path}: not valid YAML: {exc}") from exc


def _exact_keys(where: str, mapping: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ProtocolError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    missing, unknown = keys - set(mapping), set(mapping) - keys
    if missing or unknown:
        raise ProtocolError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return mapping


def parse_date(where: str, value: Any) -> date:
    """A quoted ``YYYY-MM-DD`` string as a ``date``. Anything else is an error."""
    if isinstance(value, date):
        raise ProtocolError(f"{where}: write the date in quotes (\"{value}\"); YAML turned it into a date object")
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ProtocolError(f"{where}: expected a quoted YYYY-MM-DD date, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ProtocolError(f"{where}: {value!r} is not a calendar date") from exc


def _int(where: str, value: Any, minimum: int) -> int:
    # bool is an int in Python; in a config it is a mistake.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{where}: expected an integer >= {minimum}, got {value!r}")
    return value


def load_protocol(path: Path) -> Protocol:
    """Read and validate a protocol file."""
    document = _exact_keys(str(path), _read_yaml(path), _PROTOCOL_KEYS)
    train_start = parse_date(f"{path}: train_start", document["train_start"])
    dev_start = parse_date(f"{path}: dev_start", document["dev_start"])
    holdout_start = parse_date(f"{path}: holdout_start", document["holdout_start"])
    data_end = None if document["data_end"] is None else parse_date(f"{path}: data_end", document["data_end"])
    try:
        contract.canonical_json_sha256(document)  # JSON types only, so protocol_hash is defined
    except contract.ContractError as exc:
        raise ProtocolError(f"{path}: {exc}") from exc
    if not train_start <= dev_start < holdout_start:
        raise ProtocolError(f"{path}: need train_start <= dev_start < holdout_start")
    if data_end is not None and data_end < holdout_start:
        raise ProtocolError(f"{path}: data_end {data_end} is before holdout_start {holdout_start}")
    horizons = document["horizons"]
    if not isinstance(horizons, list) or not horizons:
        raise ProtocolError(f"{path}: horizons must be a non-empty list")
    horizons = tuple(_int(f"{path}: horizons", h, 1) for h in horizons)
    if len(set(horizons)) != len(horizons):
        raise ProtocolError(f"{path}: horizons repeat a value")
    fold_len = _int(f"{path}: fold_len", document["fold_len"], 1)
    for key in ("min_cell_n", "race_context_sessions"):
        _int(f"{path}: {key}", document[key], 1)
    _int(f"{path}: holdout_looks", document["holdout_looks"], 0)
    _int(f"{path}: seed", document["seed"], 0)
    edges = document["vix_regime_edges"]
    if (
        not isinstance(edges, list)
        or len(edges) != 2
        or not all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in edges)
        or not 0 < edges[0] < edges[1]
    ):
        raise ProtocolError(f"{path}: vix_regime_edges must be two increasing positive numbers")
    targets = document["targets"]
    if not isinstance(targets, list) or not targets:
        raise ProtocolError(f"{path}: targets must be a non-empty list")
    for i, target in enumerate(targets):
        _exact_keys(f"{path}: targets[{i}]", target, _TARGET_KEYS)
    return Protocol(
        path=path,
        document=document,
        train_start=train_start,
        dev_start=dev_start,
        holdout_start=holdout_start,
        data_end=data_end,
        horizons=horizons,
        fold_len=fold_len,
        vix_regime_edges=(float(edges[0]), float(edges[1])),
    )


def protocol_path(*, synthetic: bool = False) -> Path:
    return config.configs_dir() / (SYNTHETIC_PROTOCOL_FILE if synthetic else PROTOCOL_FILE)


#: The freeze record of a protocol file sits next to it: protocol.yaml -> protocol.freeze.json.
FREEZE_RECORD_SUFFIX = ".freeze.json"
_FREEZE_RECORD_KEYS = {"protocol_file", "data_end", "protocol_hash"}


def freeze_record_path(path: Path) -> Path:
    """Where the freeze record of the protocol file ``path`` lives (``<stem>.freeze.json`` beside it)."""
    return path.with_name(path.stem + FREEZE_RECORD_SUFFIX)


def write_freeze_record(proto: Protocol) -> Path:
    """Record ``proto``'s frozen ``data_end`` and its ``protocol_hash`` beside the protocol file (amended M1, review B).

    Review B found that a hand edit moving ``data_end`` back to an earlier settled session passed
    ``check`` and ``eda`` (rc 0): every rule ``check`` applies to a frozen ``data_end`` holds for
    any settled session, so nothing tied the value to the one the freeze wrote. The record does:
    ``freeze_record_violations`` fails ``check``, ``eda`` and the notebook when the protocol no
    longer matches it. It is a tripwire for an accidental or careless edit, not a lock: a person who
    edits both files on purpose has made a new protocol, which ADR 0001 rule R1 allows by hand.
    """
    if proto.data_end is None:
        raise ProtocolError(f"{proto.path}: data_end is null; there is no freeze to record")
    from ivf import repo_guard

    record = {"protocol_file": proto.path.name, "data_end": proto.data_end.isoformat(), "protocol_hash": proto.protocol_hash}
    target = freeze_record_path(proto.path)
    repo_guard.write_file(target, (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8"), outside_repo=False)
    return target


def read_freeze_record(path: Path) -> dict[str, str] | None:
    """The freeze record of the protocol file ``path``, or None if there is none. A malformed one raises ``ProtocolError``."""
    target = freeze_record_path(path)
    if not target.exists():
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"{target}: unreadable freeze record: {exc}") from exc
    _exact_keys(str(target), record, _FREEZE_RECORD_KEYS)
    parse_date(f"{target}: data_end", record["data_end"])
    return record


def freeze_record_violations(proto: Protocol) -> list[str]:
    """Why ``proto`` does not match its freeze record (empty if it does, or if neither has a ``data_end``).

    * ``data_end`` is set but there is no record: nothing shows it came from ``check --freeze-data-end``;
    * the record's ``data_end`` differs: the frozen value was moved (rule R1);
    * the record's ``protocol_hash`` differs: the protocol changed after the freeze;
    * ``data_end`` is null but a record exists: a frozen value was removed;
    * the record is unreadable or names another file.
    """
    target = freeze_record_path(proto.path)
    name = target.name
    try:
        record = read_freeze_record(proto.path)
    except ProtocolError as exc:
        return [str(exc)]
    if record is None:
        if proto.data_end is None:
            return []
        return [
            f"{proto.path.name}: data_end {proto.data_end} has no freeze record ({name}), so nothing shows it was "
            "frozen by `check --freeze-data-end` rather than typed in. Run `uv run python -m ivf.data check "
            "--freeze-data-end` (with --synthetic for the synthetic protocol): it records the value if the data support it"
        ]
    problems = []
    if record["protocol_file"] != proto.path.name:
        problems.append(f"{name} records a freeze of {record['protocol_file']}, not of {proto.path.name}")
    frozen = record["data_end"]
    if proto.data_end is None or proto.data_end.isoformat() != frozen:
        problems.append(
            f"{proto.path.name}: data_end {proto.data_end or 'null'} is not the frozen value {frozen} ({name}). "
            "A frozen data_end never moves (ADR 0001 4.6, rule R1): put it back, or, if you mean to start a new "
            f"protocol, change {name} by hand too"
        )
    elif record["protocol_hash"] != proto.protocol_hash:
        problems.append(
            f"{proto.path.name} changed after data_end was frozen: protocol_hash {proto.protocol_hash} now, "
            f"{record['protocol_hash']} at the freeze ({name}). A changed protocol is a new protocol (ADR 0001 rule R1): "
            f"undo the change, or, if you mean it, put the new protocol_hash into {name} by hand"
        )
    return problems


def freeze_data_end(path: Path, value: date) -> bool:
    """Write ``data_end: "<value>"`` into ``path`` if it is still ``null``, and record the freeze.

    Returns True if the file changed, False if it already held ``value``.
    Raises ``ProtocolError`` if it holds a different date: a frozen data_end is
    moved only by a person editing the file on purpose, which creates a new
    protocol_hash and needs a new release under ADR 0001 rule R1. Also raises
    if ``data_end`` is null but a freeze record exists (a frozen value was
    removed; re-freezing would move it).

    Only the one ``data_end:`` line is rewritten, so the comments stay. Both
    files are written by ``repo_guard.write_file`` (atomic rename, flushed;
    amended M1, review B). After writing ``data_end`` it writes the freeze
    record (``write_freeze_record``); called on a value already frozen with
    no record yet, it writes just the record.
    """
    from ivf import repo_guard

    current = load_protocol(path)
    if current.data_end == value:
        if read_freeze_record(path) is None:
            write_freeze_record(current)
        return False
    if current.data_end is not None:
        raise ProtocolError(
            f"{path}: data_end is already frozen at {current.data_end}; refusing to move it to {value}. "
            "Changing it is a new protocol (ADR 0001 4.6, rule R1): edit the file by hand if you mean it"
        )
    if read_freeze_record(path) is not None:
        raise ProtocolError(
            f"{path}: data_end is null but {freeze_record_path(path).name} records a freeze; refusing to freeze again "
            "(that would move a frozen data_end, ADR 0001 rule R1). Put the recorded value back"
        )
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^data_end:[ \t]*null[ \t]*(#.*)?$", re.MULTILINE)
    if len(pattern.findall(text)) != 1:
        raise ProtocolError(f"{path}: expected exactly one 'data_end: null' line to replace")
    new_text = pattern.sub(lambda m: f'data_end: "{value.isoformat()}"' + (f"  {m.group(1)}" if m.group(1) else ""), text)
    repo_guard.write_file(path, new_text.encode("utf-8"), outside_repo=False)
    frozen = load_protocol(path)
    if frozen.data_end != value:  # read back: the edit must mean what we wrote
        repo_guard.write_file(path, text.encode("utf-8"), outside_repo=False)
        raise ProtocolError(f"{path}: freezing data_end did not round-trip; the file was restored")
    write_freeze_record(frozen)
    return True


def load_iv_indices(path: Path | None = None) -> tuple[IvIndex, ...]:
    path = path or config.configs_dir() / IV_INDICES_FILE
    document = _exact_keys(str(path), _read_yaml(path), {"indices"})
    out = []
    for i, entry in enumerate(document["indices"]):
        entry = _exact_keys(f"{path}: indices[{i}]", entry, _INDEX_KEYS)
        if entry["tenor_days"] is not None:
            _int(f"{path}: indices[{i}].tenor_days", entry["tenor_days"], 1)
        out.append(IvIndex(**entry))
    symbols = [index.symbol for index in out]
    if len(set(symbols)) != len(symbols):
        raise ProtocolError(f"{path}: a symbol appears twice")
    return tuple(out)


def load_instruments(path: Path | None = None) -> tuple[Instrument, ...]:
    path = path or config.configs_dir() / INSTRUMENTS_FILE
    document = _exact_keys(str(path), _read_yaml(path), {"instruments", "tradable_segments"})
    return tuple(
        Instrument(**_exact_keys(f"{path}: instruments[{i}]", entry, _INSTRUMENT_KEYS))
        for i, entry in enumerate(document["instruments"])
    )


def term_structure(indices: tuple[IvIndex, ...]) -> tuple[IvIndex, ...]:
    """The term-structure members, shortest tenor first."""
    members = [index for index in indices if index.group == "term_structure"]
    if any(index.tenor_days is None for index in members):
        raise ProtocolError("every term_structure index needs tenor_days")
    return tuple(sorted(members, key=lambda index: index.tenor_days))
