"""The master calendar 𝒯 (ADR 0001 D4.1, D4.5).

𝒯 is the NYSE (``XNYS``) session calendar from ``exchange_calendars``, from
``train_start`` to ``data_end``, extended by the ``h_max`` scheduled sessions
after ``data_end`` that date the longest-horizon targets. ``t+h`` means the
h-th session after ``t`` in 𝒯.

It is computed from dates alone. No function here sees any market data, so the
calendar, the fold cutoffs and the target dates cannot move when the data are
truncated or refreshed. That is what lets the truncation test (ADR 0001 4.7)
compare like with like.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import exchange_calendars
import pandas as pd

from ivf.protocol import Protocol

EXCHANGE = "XNYS"
PERIODS = ("burn_in", "dev", "holdout")


def xnys_sessions(start: date, end: date) -> list[date]:
    """XNYS sessions with ``start <= session <= end``, oldest first."""
    if end < start:
        return []
    # exchange_calendars only answers inside the bounds a calendar was built
    # with, so build one that covers the range with room to spare.
    with warnings.catch_warnings():
        # exchange_calendars 4.13.2 calls Series.values on tz-aware data, which
        # pandas 3.1 deprecates (Pandas4Warning). Harmless here, and not ours to fix.
        warnings.filterwarnings("ignore", message=".*Series.values returning an ndarray.*")
        calendar = exchange_calendars.get_calendar(EXCHANGE, start=start - timedelta(days=7), end=end + timedelta(days=7))
        return [ts.date() for ts in calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]


def sessions_after(day: date, count: int) -> list[date]:
    """The ``count`` scheduled sessions strictly after ``day``."""
    if count <= 0:
        return []
    # 21 sessions span at most 38 calendar days (ADR 0001 4.4); 3 days per
    # session plus a margin covers any horizon with long holiday runs.
    out = xnys_sessions(day + timedelta(days=1), day + timedelta(days=3 * count + 14))
    if len(out) < count:
        raise RuntimeError(f"exchange_calendars returned {len(out)} sessions after {day}, needed {count}")
    return out[:count]


def first_cutoff(sessions: list[date], dev_start: date, holdout_start: date, fold_len: int) -> date:
    """The first fold cutoff: the earliest ``c_k = idx(holdout_start) + fold_len*k`` with ``c_k >= idx(dev_start)``.

    Folds are anchored on ``holdout_start`` (ADR 0001 4.5), so no fold straddles
    dev and holdout. With 2000-01-03 and 2024-07-01 on XNYS this is 2000-03-15.
    """
    index = {day: i for i, day in enumerate(sessions)}
    for name, day in (("dev_start", dev_start), ("holdout_start", holdout_start)):
        if day not in index:
            raise ValueError(f"{name} {day} is not an XNYS session")
    i_dev, i_holdout = index[dev_start], index[holdout_start]
    k = (i_holdout - i_dev) // fold_len
    return sessions[i_holdout - fold_len * k]


def master_calendar(protocol: Protocol, data_end: date | None = None) -> pd.DataFrame:
    """The calendar table (data-model.md section 4) for ``protocol``.

    ``data_end`` defaults to the protocol's frozen value. ``ivf.data check``
    passes a provisional one before the freeze.

    Columns: ``date``, ``t_idx`` (0-based), ``period`` and ``fold_id``.
    ``period`` is ``burn_in`` before the first fold cutoff (training data that
    is never a forecast origin), ``dev`` from there to ``holdout_start``, and
    ``holdout`` on and after it. ``fold_id`` stays null in milestone 1: folds
    are numbered when milestone 2 builds the ``fold`` table. The last ``h_max``
    rows are the sessions after ``data_end``: they exist only to date targets
    (``date > data_end`` tells them apart).
    """
    end = data_end if data_end is not None else protocol.require_data_end()
    if protocol.train_start not in set(xnys_sessions(protocol.train_start, protocol.train_start)):
        raise ValueError(f"train_start {protocol.train_start} is not an XNYS session")
    core = xnys_sessions(protocol.train_start, end)
    if not core or core[-1] != end:
        raise ValueError(f"data_end {end} is not an XNYS session")
    extension = sessions_after(end, protocol.h_max)
    sessions = core + extension
    cutoff = first_cutoff(sessions, protocol.dev_start, protocol.holdout_start, protocol.fold_len)
    period = ["burn_in" if d < cutoff else "dev" if d < protocol.holdout_start else "holdout" for d in sessions]
    return pd.DataFrame(
        {
            "date": sessions,
            "t_idx": pd.array(range(len(sessions)), dtype="int32"),
            "period": period,
            "fold_id": pd.array([None] * len(sessions), dtype="Int32"),
        }
    )
