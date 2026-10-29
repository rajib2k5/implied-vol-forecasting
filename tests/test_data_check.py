"""Tests for ivf.data.check: coverage of 𝒯, the gap policy G1-G5 and the settled data_end."""

from __future__ import annotations

import functools
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ivf import protocol
from ivf.data import calendar, check, synthetic

pytestmark = pytest.mark.usefixtures("no_network")

END = date(2026, 10, 26)
# The day after END at noon UTC (08:00 in New York): every session up to END is settled.
FETCHED = "2026-10-27T12:00:00.000000Z"


@functools.cache
def market():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    return proto, *synthetic.synthetic_market(proto)


def report(iv, price, fetched=FETCHED, duplicates=None):
    duplicates = duplicates or {}
    symbols = sorted(iv["symbol"].unique()) + sorted(price["ticker"].unique())
    return {
        "files": [
            {
                "symbol": s,
                "source": "synthetic",
                "raw_sha256": "0" * 64,
                "fetched_at": fetched if isinstance(fetched, str) else fetched[s],
                "duplicates_dropped": len(duplicates.get(s, [])),
                "duplicate_dates": duplicates.get(s, []),
            }
            for s in symbols
        ]
    }


def drop(frame, key, symbol, day):
    return frame[~((frame[key] == symbol) & (frame["date"] == day))].reset_index(drop=True)


def rows(result, check_id, symbol):
    table = result.table
    return table[(table.check_id == check_id) & (table.symbol == symbol)].iloc[0]


def test_complete_data_passes_and_the_candidate_is_the_last_session():
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert result.ok and result.core_gaps == []
    assert result.candidate_data_end == END
    assert result.first_cutoff == date(2000, 3, 15)
    assert rows(result, "train_start_is_first_vix", "VIX").n_affected == 0


def test_a_burn_in_gap_is_reported_with_its_date_and_allowed():
    proto, iv, price = market()
    iv = drop(iv, "symbol", "VIX", date(1995, 6, 1))
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert result.ok
    assert result.core_gaps == [date(1995, 6, 1)]
    missing = rows(result, "missing_in_cboe", "VIX")
    assert (missing.n_affected, missing.detail) == (1, "1995-06-01")


def test_gap_policy_g3_boundary_is_22_sessions_before_the_first_cutoff():
    proto, iv, price = market()
    sessions = calendar.xnys_sessions(date(1990, 1, 2), date(2000, 3, 15))
    assert sessions[-1] == date(2000, 3, 15)  # the first cutoff, index 2578
    allowed, too_late = sessions[-1 - 22], sessions[-1 - 21]  # indices 2556 and 2557
    ok = check.data_check(proto, *_without_spx(iv, price, allowed), END)
    late = check.data_check(proto, *_without_spx(iv, price, too_late), END)
    assert ok.ok
    assert not late.ok and "G3" in late.policy_violations[0] and too_late.isoformat() in late.policy_violations[0]


def _without_spx(iv, price, day):
    price = drop(price, "ticker", "^GSPC", day)
    return iv, price, report(iv, price)


def test_a_holdout_gap_breaks_the_policy():
    proto, iv, price = market()
    result = check.data_check(proto, *_without_spx(iv, price, date(2025, 4, 8)), END)
    assert not result.ok
    assert rows(result, "missing_in_yf", "^GSPC").n_affected == 1


def test_a_row_on_a_non_session_is_counted_not_a_gap():
    proto, iv, price = market()
    holiday = pd.DataFrame([{"symbol": "VIX", "date": date(2024, 7, 4), "open": 13.0, "high": 13.0, "low": 13.0, "close": 13.0}])
    iv = pd.concat([iv, holiday], ignore_index=True)
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert result.ok and result.core_gaps == []
    extra = rows(result, "not_a_session", "VIX")
    assert (extra.n_affected, extra.detail) == (1, "2024-07-04")


def test_non_core_gaps_are_counted_and_never_fail():
    proto, iv, price = market()
    iv = drop(iv, "symbol", "VIX1Y", date(2008, 10, 13))
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert result.ok
    assert rows(result, "missing_in_cboe", "VIX1Y").n_affected == 1
    assert result.core_gaps == []


@pytest.mark.parametrize(
    "fetched, candidate",
    [
        # 2026-10-26 21:00 UTC is 17:00 in New York (EDT, UTC-4): the same day, so 10-26 is not settled.
        ("2026-10-26T21:00:00.000000Z", date(2026, 10, 23)),
        # 2026-10-27 03:59 UTC is still 23:59 on 10-26 in New York.
        ("2026-10-27T03:59:00.000000Z", date(2026, 10, 23)),
        # 2026-10-27 04:00 UTC is 00:00 on 10-27 in New York: 10-26 is settled.
        ("2026-10-27T04:00:00.000000Z", date(2026, 10, 26)),
    ],
)
def test_data_end_is_the_last_session_settled_before_the_fetch_day_in_new_york(fetched, candidate):
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price, fetched=fetched), END)
    assert result.candidate_data_end == candidate
    assert rows(result, "not_settled_at_fetch", "^GSPC").n_affected == (0 if candidate == END else 1)


def test_the_earlier_of_the_two_core_fetches_decides():
    proto, iv, price = market()
    times = {s: FETCHED for s in list(iv["symbol"].unique()) + ["^GSPC"]}
    times["^GSPC"] = "2026-10-26T21:00:00.000000Z"  # yfinance fetched during the evening of 10-26
    result = check.data_check(proto, iv, price, report(iv, price, fetched=times), END)
    assert result.candidate_data_end == date(2026, 10, 23)


def test_the_candidate_skips_a_last_session_missing_a_core_close():
    proto, iv, price = market()
    iv = drop(iv, "symbol", "VIX", END)
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert result.candidate_data_end == date(2026, 10, 23)


def test_train_start_must_be_the_first_vix_date():
    proto, iv, price = market()
    iv = drop(iv, "symbol", "VIX", date(1990, 1, 2))
    result = check.data_check(proto, iv, price, report(iv, price), END)
    assert not result.ok
    assert any("not the first VIX date (1990-01-03)" in v for v in result.policy_violations)


def test_duplicates_from_the_build_report_appear_as_rows():
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price, duplicates={"VIX": ["2001-02-02"]}), END)
    assert (rows(result, "duplicate_date", "VIX").n_affected, rows(result, "duplicate_date", "VIX").detail) == (1, "2001-02-02")


def test_settled_before_needs_an_aware_time():
    assert check.settled_before(datetime(2026, 10, 27, 3, 59, tzinfo=timezone.utc)) == date(2026, 10, 26)
    with pytest.raises(ValueError):
        check.settled_before(datetime(2026, 10, 27, 3, 59))


# --- a frozen data_end is re-validated on every run (amended M1, review A) ----------------------
# Review A: a frozen data_end fetched on the same New York day passed check and eda (violations: []).


def test_a_frozen_data_end_that_the_data_support_passes():
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price), END, frozen=True)
    assert result.ok


def test_a_frozen_data_end_fetched_on_the_same_new_york_day_fails():
    # 2026-10-26T15:00Z is 11:00 in New York on data_end itself: the bar may be intraday.
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price, fetched="2026-10-26T15:00:00.000000Z"), END, frozen=True)
    assert not result.ok
    assert any("frozen data_end 2026-10-26 is not settled for VIX" in v for v in result.policy_violations)
    assert any("not settled for ^GSPC" in v for v in result.policy_violations)
    # Without frozen=True (the search before a freeze) the same data only move the candidate back.
    assert check.data_check(proto, iv, price, report(iv, price, fetched="2026-10-26T15:00:00.000000Z"), END).ok


def test_only_one_core_fetch_on_the_same_day_is_enough_to_fail():
    proto, iv, price = market()
    times = {entry["symbol"]: FETCHED for entry in report(iv, price)["files"]}
    times["^GSPC"] = "2026-10-26T21:00:00.000000Z"  # 17:00 New York, still data_end's own day
    result = check.data_check(proto, iv, price, report(iv, price, fetched=times), END, frozen=True)
    assert [v for v in result.policy_violations if "settled" in v] == [
        "frozen data_end 2026-10-26 is not settled for ^GSPC: its snapshot was fetched on 2026-10-26 New York time, "
        "and a close counts only if fetched on a later New York day"
    ]


def test_a_frozen_data_end_without_a_vix_close_fails():
    proto, iv, price = market()
    iv = drop(iv, "symbol", "VIX", END)
    result = check.data_check(proto, iv, price, report(iv, price), END, frozen=True)
    assert "frozen data_end 2026-10-26 has no VIX close in the interim tables" in result.policy_violations


def test_a_hand_edited_data_end_that_is_not_a_session_fails():
    proto, iv, price = market()
    result = check.data_check(proto, iv, price, report(iv, price), date(2026, 10, 25), frozen=True)  # a Sunday
    assert "frozen data_end 2026-10-25 is not an XNYS session" in result.policy_violations


def test_data_reaching_past_a_frozen_data_end_are_fine():
    # A later fetch adds sessions after data_end; data_end stays frozen (rule R1) and still holds.
    proto, iv, price = market()
    later = date(2026, 10, 27)
    iv = pd.concat([iv, pd.DataFrame([{"symbol": "VIX", "date": later, "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0}])], ignore_index=True)
    price = pd.concat([price, price.iloc[[-1]].assign(date=later)], ignore_index=True)
    result = check.data_check(proto, iv, price, report(iv, price, fetched="2026-10-28T12:00:00.000000Z"), END, frozen=True)
    assert result.ok


# --- the freeze record through check and eda (amended M1, review B) ------------------------------


def test_moving_the_frozen_data_end_back_to_a_settled_session_fails_check_and_eda(tmp_path, monkeypatch, capsys):
    # Review B: 2026-10-23 is a settled session with both closes, so every data rule holds for it,
    # and check and eda both returned 0 after the hand edit. The freeze record catches it.
    import shutil

    from ivf import config
    from ivf.data import build_interim, process

    configs = tmp_path / "configs"
    shutil.copytree(config.configs_dir(), configs)
    monkeypatch.setattr(config, "configs_dir", lambda: configs)
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "data"))
    layout = config.data_layout(synthetic=True)
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    synthetic.write_synthetic_raw(layout.root, proto)
    build_interim.build_interim(layout.root, [i.symbol for i in protocol.load_iv_indices()], ["^GSPC"])
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
    assert check.run_check(layout, synthetic=True, freeze=False, now=now) == 0
    path = configs / "protocol.synthetic.yaml"
    path.write_text(path.read_text().replace('data_end: "2026-10-26"', 'data_end: "2026-10-23"'))
    capsys.readouterr()
    assert check.run_check(layout, synthetic=True, freeze=False, now=now) == 1
    out = capsys.readouterr().out
    assert "gap policy: VIOLATED" in out and "data_end 2026-10-23 is not the frozen value 2026-10-26" in out
    assert process.run_eda(layout, synthetic=True) == 1
    assert "is not the frozen value" in capsys.readouterr().err
    assert not layout.eda.exists()  # eda stopped before writing anything
