"""``data_check``: is there a VIX close and a ^GSPC close on every session of 𝒯? (ADR 0001 D4.1)

    uv run python -m ivf.data check                    # report; exit 1 if the gap policy is broken
    uv run python -m ivf.data check --freeze-data-end  # also write data_end into configs/protocol.yaml once

The gap policy (ADR 0001 D4.1, amended M1; the walkthrough explains it):

* G1. 𝒯 comes from dates alone. A session of 𝒯 in ``[train_start, data_end]``
  with no VIX close or no ^GSPC close is a **core gap**. It stays a gap: never
  filled, interpolated or removed from 𝒯. It is null in every table built on 𝒯.
* G2. Anything computed over a window that contains a gap is null: a return
  needs two consecutive sessions, a 21-session sum needs all 21. Pairs (as in
  an AR(1) fit) are formed on 𝒯 and dropped if either side is null. A gap is
  never bridged by pairing a value with "the previous row that exists".
* G3. Core gaps are allowed only in burn-in, and at least ``SPILL_SESSIONS``
  sessions before the first fold cutoff, so that no trailing feature window at
  any forecast origin and no target of any scored origin can touch one. A core
  gap anywhere later fails ``check`` (exit 1): ADR 0001 4.5's "same rows" rule
  would need a missing-origin rule, and that is a decision, not a default.
* G4. A row on a date that is not an XNYS session is not on 𝒯, so nothing
  downstream reads it. It stays in interim (the raw truth) and is counted here
  as ``not_a_session``.
* G5. The other indices (VIX9D, VIX3M, VIX6M, VIX1Y, VVIX) are not core. Their
  gaps inside their own history are null and counted (``missing_in_cboe``),
  and never fail ``check``.

``data_end`` is the last session on which both core closes exist and are
**settled**: the fetch that produced them ran on a later New York calendar day
than the session. yfinance can return a bar for a session still in progress (M1
did not observe one: its fetch ran before the open), so a same-day fetch could
otherwise freeze an intraday price as a close. The rule costs at most one
session and needs no knowledge of exchange hours.

A frozen ``data_end`` is **re-validated on every run** (amended M1, review A):
it must be an XNYS session with both core closes present and settled in the
interim tables now being read. Review A found the rule applied only when
freezing: a hand-edited value, or interim tables rebuilt from a same-day
fetch, passed ``check`` and ``eda`` silently. Data that now reach *past* a
frozen ``data_end`` are fine (the value stays frozen, rule R1); data that no
longer support it are a policy violation (exit 1), and ``eda`` refuses to run.

A frozen ``data_end`` must also **match its freeze record** (amended M1,
review B): ``check --freeze-data-end`` writes ``configs/<protocol>.freeze.json``
holding the value and the ``protocol_hash`` it froze, and ``check``, ``eda`` and
the notebook fail when the protocol no longer matches it
(``protocol.freeze_record_violations``). Review B found that moving ``data_end``
back by hand to an earlier settled session passed both (rc 0): such a session
meets every rule above, so the rules alone cannot tell it from the frozen one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ivf import protocol as protocol_mod
from ivf import repo_guard
from ivf.data import build_interim, calendar

NEW_YORK = ZoneInfo("America/New_York")
CORE_INDEX = "VIX"
CORE_TICKER = "^GSPC"
#: The longest trailing window in the feature table is har_m, the mean of
#: ln V over t-21..t: 22 sessions. A gap this far before the first cutoff can
#: reach no feature at a forecast origin. (Targets reach forward from an
#: origin, and every origin is at or after the first cutoff, so they cannot.)
SPILL_SESSIONS = 22
#: How many dates a detail string lists before it says "and N more".
DETAIL_DATES = 50

CHECK_COLUMNS = ["check_id", "symbol", "n_affected", "detail"]


def settled_before(fetched_at: datetime) -> date:
    """The New York calendar date of a fetch. Sessions strictly before it are settled."""
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    return fetched_at.astimezone(NEW_YORK).date()


def _detail(days: list[date]) -> str:
    shown = ", ".join(d.isoformat() for d in days[:DETAIL_DATES])
    more = len(days) - DETAIL_DATES
    return shown + (f", and {more} more" if more > 0 else "")


@dataclass
class CheckResult:
    table: pd.DataFrame  # data_check rows (data-model.md section 4), without run_id
    coverage: pd.DataFrame  # per symbol: first, last, sessions in span, missing
    core_gaps: list[date]
    policy_violations: list[str]
    candidate_data_end: date | None
    first_cutoff: date

    @property
    def ok(self) -> bool:
        return not self.policy_violations


def data_check(
    protocol: protocol_mod.Protocol,
    iv: pd.DataFrame,
    price: pd.DataFrame,
    build_report: dict,
    end: date,
    *,
    frozen: bool = False,
) -> CheckResult:
    """Run every check on sessions of 𝒯 from ``train_start`` to ``end``.

    ``end`` is the frozen ``data_end`` or, before the freeze, the last session
    up to which the candidate is searched. ``build_report`` is what
    ``build_interim`` wrote; it supplies the duplicate counts and fetch times.
    ``frozen=True`` says ``end`` is a frozen ``data_end``: it is then checked
    against the data (``frozen_data_end_violations``).
    """
    sessions = calendar.xnys_sessions(protocol.train_start, end)
    session_set = set(sessions)
    rows: list[dict] = []
    coverage: list[dict] = []
    fetched = {entry["symbol"]: datetime.fromisoformat(entry["fetched_at"].replace("Z", "+00:00")) for entry in build_report["files"]}

    def dates_of(frame: pd.DataFrame, key: str, symbol: str) -> set[date]:
        return set(frame.loc[frame[key] == symbol, "date"])

    symbols = [(s, "iv") for s in sorted(iv["symbol"].unique())] + [(t, "price") for t in sorted(price["ticker"].unique())]
    present: dict[str, set[date]] = {}
    for symbol, kind in symbols:
        have = dates_of(iv, "symbol", symbol) if kind == "iv" else dates_of(price, "ticker", symbol)
        present[symbol] = have
        in_range = sorted(d for d in have if protocol.train_start <= d <= end)
        first = in_range[0] if in_range else None
        span = [s for s in sessions if first is not None and s >= first]
        missing = [s for s in span if s not in have]
        extra = [d for d in in_range if d not in session_set]
        check_missing = "missing_in_cboe" if kind == "iv" else "missing_in_yf"
        rows.append({"check_id": check_missing, "symbol": symbol, "n_affected": len(missing), "detail": _detail(missing)})
        rows.append({"check_id": "not_a_session", "symbol": symbol, "n_affected": len(extra), "detail": _detail(extra)})
        settled = settled_before(fetched[symbol]) if symbol in fetched else None
        unsettled = sorted(d for d in have if settled is not None and d >= settled)
        rows.append({"check_id": "not_settled_at_fetch", "symbol": symbol, "n_affected": len(unsettled), "detail": _detail(unsettled)})
        coverage.append(
            {
                "symbol": symbol,
                "first": first,
                "last": in_range[-1] if in_range else None,
                "sessions_in_span": len(span),
                "missing": len(missing),
                "not_a_session": len(extra),
            }
        )
    for entry in build_report["files"]:
        rows.append(
            {
                "check_id": "duplicate_date",
                "symbol": entry["symbol"],
                "n_affected": int(entry["duplicates_dropped"]),
                "detail": _detail([date.fromisoformat(d) for d in entry["duplicate_dates"]]),
            }
        )

    for needed in (CORE_INDEX, CORE_TICKER):
        if needed not in present:
            raise build_interim.InterimError(f"{needed} is not in the interim tables; it is core data")
    vix, spx = present[CORE_INDEX], present[CORE_TICKER]
    violations = []
    # ADR 0001 4.6: train_start is "the first VIX date". Checked on the file, not assumed.
    first_vix = min(vix)
    rows.append(
        {
            "check_id": "train_start_is_first_vix",
            "symbol": CORE_INDEX,
            "n_affected": int(first_vix != protocol.train_start),
            "detail": f"first VIX date {first_vix}, train_start {protocol.train_start}",
        }
    )
    if first_vix != protocol.train_start:
        violations.append(f"train_start {protocol.train_start} is not the first VIX date ({first_vix})")
    core_gaps = [s for s in sessions if s not in vix or s not in spx]
    rows.append({"check_id": "core_gap", "symbol": f"{CORE_INDEX}+{CORE_TICKER}", "n_affected": len(core_gaps), "detail": _detail(core_gaps)})

    full = calendar.xnys_sessions(protocol.train_start, max(end, protocol.holdout_start))
    cutoff = calendar.first_cutoff(full, protocol.dev_start, protocol.holdout_start, protocol.fold_len)
    index = {d: i for i, d in enumerate(full)}
    latest_allowed = index[cutoff] - SPILL_SESSIONS
    late = [g for g in core_gaps if index[g] > latest_allowed]
    if late:
        violations.append(
            f"core gaps on {len(late)} session(s) later than {SPILL_SESSIONS} sessions before the first fold cutoff "
            f"{cutoff} (gap policy G3): {_detail(late)}"
        )

    settle_limit = min(settled_before(fetched[s]) for s in (CORE_INDEX, CORE_TICKER) if s in fetched) if fetched else None
    candidates = [s for s in sessions if s in vix and s in spx and (settle_limit is None or s < settle_limit)]
    if frozen:
        violations += frozen_data_end_violations(end, sessions, vix, spx, fetched)
    return CheckResult(
        table=pd.DataFrame(rows, columns=CHECK_COLUMNS),
        coverage=pd.DataFrame(coverage),
        core_gaps=core_gaps,
        policy_violations=violations,
        candidate_data_end=candidates[-1] if candidates else None,
        first_cutoff=cutoff,
    )


def frozen_check(protocol: protocol_mod.Protocol, iv: pd.DataFrame, price: pd.DataFrame, build_report: dict) -> CheckResult:
    """``data_check`` with the frozen ``data_end`` re-validated, plus the freeze record (amended M1, review B).

    What ``eda`` and the notebook run before computing anything. ``ok`` is False if the data no
    longer support the frozen ``data_end`` or the protocol no longer matches its freeze record.
    """
    result = data_check(protocol, iv, price, build_report, protocol.require_data_end(), frozen=True)
    result.policy_violations += protocol_mod.freeze_record_violations(protocol)
    return result


def frozen_data_end_violations(
    data_end: date, sessions: list[date], vix: set[date], spx: set[date], fetched: dict[str, datetime]
) -> list[str]:
    """Why a frozen ``data_end`` is not supported by the data being read (empty if it is).

    The same rule that chose it (module docstring): an XNYS session, with a VIX
    close and a ^GSPC close, each from a fetch made on a later New York day.
    """
    problems = []
    where = f"frozen data_end {data_end}"
    if not sessions or sessions[-1] != data_end:
        return [f"{where} is not an XNYS session"]
    for symbol, have in ((CORE_INDEX, vix), (CORE_TICKER, spx)):
        if data_end not in have:
            problems.append(f"{where} has no {symbol} close in the interim tables")
        elif symbol not in fetched:
            problems.append(f"{where}: no fetch time for {symbol}, so its close cannot be shown to be settled")
        elif data_end >= settled_before(fetched[symbol]):
            problems.append(
                f"{where} is not settled for {symbol}: its snapshot was fetched on {settled_before(fetched[symbol])} "
                "New York time, and a close counts only if fetched on a later New York day"
            )
    return problems


def print_summary(result: CheckResult, proto: protocol_mod.Protocol, end: date, out=None) -> None:
    """The gap and coverage summary. It goes to the terminal only: data_check counts are local (ADR 0001 D2).

    ``out`` defaults to the ``sys.stdout`` of the moment of the call. A default of
    ``out=sys.stdout`` would bind the stream at import time, and a notebook or
    pytest that swaps ``sys.stdout`` later would never see the summary.
    """
    out = out if out is not None else sys.stdout
    print(f"master calendar: XNYS sessions {proto.train_start}..{end}; first fold cutoff {result.first_cutoff}", file=out)
    print("coverage (sessions of the calendar from each symbol's first date):", file=out)
    for row in result.coverage.itertuples(index=False):
        print(
            f"  {row.symbol:6} {row.first}..{row.last}  sessions {row.sessions_in_span:5}  missing {row.missing:3}  "
            f"rows on non-sessions {row.not_a_session:3}",
            file=out,
        )
    first_vix = result.table.loc[result.table.check_id == "train_start_is_first_vix", "detail"]
    if len(first_vix):
        print(f"train_start check: {first_vix.iloc[0]}", file=out)
    print(f"core gaps (no VIX close or no ^GSPC close on a session): {len(result.core_gaps)}", file=out)
    for gap in result.core_gaps:
        print(f"  {gap}", file=out)
    other = result.table[(result.table.n_affected > 0) & ~result.table.check_id.isin(["core_gap"])]
    for row in other.itertuples(index=False):
        print(f"  {row.check_id:21} {row.symbol:6} {row.n_affected:4}: {row.detail[:120]}", file=out)
    print(f"gap policy: {'OK' if result.ok else 'VIOLATED'}", file=out)
    for violation in result.policy_violations:
        print(f"  {violation}", file=out)
    print(f"candidate data_end (last session with both core closes, settled): {result.candidate_data_end}", file=out)


def run_check(layout, *, synthetic: bool, freeze: bool, now: datetime) -> int:
    repo_guard.require_data_root_outside_repo(layout.root)
    path = protocol_mod.protocol_path(synthetic=synthetic)
    proto = protocol_mod.load_protocol(path)
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    end = proto.data_end
    if end is None:
        # Search up to the last session before today in New York; nothing later can be settled.
        sessions = calendar.xnys_sessions(proto.train_start, settled_before(now))
        end = sessions[-1] if sessions[-1] < settled_before(now) else sessions[-2]
    result = data_check(proto, iv, price, report, end, frozen=proto.data_end is not None)
    if freeze and result.ok and proto.data_end is not None and not protocol_mod.freeze_record_path(path).exists():
        # Frozen before freeze records existed (review B), and the data still support the value: record it.
        record = protocol_mod.write_freeze_record(proto)
        print(f"recorded the frozen data_end {proto.data_end} (protocol_hash {proto.protocol_hash}) in {record.name}")
    result.policy_violations += protocol_mod.freeze_record_violations(proto)
    print_summary(result, proto, end)
    print(f"protocol: {path.name}, data_end {proto.data_end or 'not frozen'}, protocol_hash {proto.protocol_hash}")
    if not result.ok:
        return 1
    if freeze:
        if result.candidate_data_end is None:
            print("error: no session has both core closes; nothing to freeze", file=sys.stderr)
            return 1
        if proto.data_end is not None and proto.data_end != result.candidate_data_end:
            print(
                f"data_end stays frozen at {proto.data_end} (the data now reach {result.candidate_data_end}); "
                "moving it is a new protocol (ADR 0001 4.6, rule R1)"
            )
            return 0
        changed = protocol_mod.freeze_data_end(path, result.candidate_data_end)
        frozen = protocol_mod.load_protocol(path)
        print(
            f"{'froze' if changed else 'already frozen:'} data_end = {frozen.data_end} in {path}; "
            f"protocol_hash {frozen.protocol_hash}"
        )
    return 0
