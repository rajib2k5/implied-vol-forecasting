"""Tests for ivf.protocol (the YAML configs, protocol_hash, the data_end freeze) and ivf.data.calendar (𝒯).

Calendar facts are checked against figures ADR 0001 measured in milestone 0
with exchange_calendars, not recomputed by the code under test.
"""

from __future__ import annotations

from datetime import date

import pytest

from ivf import protocol
from ivf.data import calendar

SMALL = """\
# a comment that must survive the freeze
train_start: "1990-01-02"
dev_start: "2000-01-03"
holdout_start: "2024-07-01"
data_end: null   # frozen by check
horizons: [1, 5, 21]
fold_len: 63
min_cell_n: 20
vix_regime_edges: [15, 25]
holdout_looks: 0
seed: 1
race_context_sessions: 252
targets:
  - {target_id: vix_level, label: L, units: vol_points, horizons: [1], definition: d}
"""


@pytest.fixture
def small(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(SMALL)
    return path


# --- protocol ------------------------------------------------------------------------


def test_protocol_hash_is_sha256_of_the_canonical_json_computed_outside_python(small):
    # printf '%s' '{"data_end":null,"dev_start":"2000-01-03","fold_len":63,"holdout_looks":0,
    #   "holdout_start":"2024-07-01","horizons":[1,5,21],"min_cell_n":20,"race_context_sessions":252,
    #   "seed":1,"targets":[{"definition":"d","horizons":[1],"label":"L","target_id":"vix_level",
    #   "units":"vol_points"}],"train_start":"1990-01-02","vix_regime_edges":[15,25]}' | sha256sum
    assert protocol.load_protocol(small).protocol_hash == "e48f7517a7d69238cfe20c091974905c8df1dc8100b1ac69e6692e6bf4e8abc7"


def test_freezing_data_end_writes_one_line_keeps_comments_and_changes_the_hash(small):
    assert protocol.freeze_data_end(small, date(2026, 10, 28)) is True
    text = small.read_text()
    assert 'data_end: "2026-10-28"  # frozen by check' in text
    assert text.startswith("# a comment that must survive the freeze\n")
    loaded = protocol.load_protocol(small)
    assert loaded.data_end == date(2026, 10, 28)
    # The same JSON as above with "data_end":"2026-10-28", through sha256sum.
    assert loaded.protocol_hash == "67fdc0f0f9d474b50a70a235d9d9d2d0e8434b64810c9b355299047871747ca1"


def test_freezing_again_to_the_same_date_is_a_no_op(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    before = small.read_bytes()
    assert protocol.freeze_data_end(small, date(2026, 10, 28)) is False
    assert small.read_bytes() == before


# --- the freeze record (amended M1, review B) ------------------------------------------------
# Review B: moving a frozen data_end back by hand to an earlier settled session passed check and eda
# (rc 0), because that session meets every rule check applies. The freeze now leaves a record.


def test_freezing_writes_a_record_of_the_value_and_the_hash(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    frozen = protocol.load_protocol(small)
    record = protocol.read_freeze_record(small)
    assert protocol.freeze_record_path(small) == small.with_name("protocol.freeze.json")
    assert record == {"protocol_file": "protocol.yaml", "data_end": "2026-10-28", "protocol_hash": frozen.protocol_hash}
    assert protocol.freeze_record_violations(frozen) == []


def test_moving_a_frozen_data_end_back_by_hand_is_a_violation(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    small.write_text(small.read_text().replace('data_end: "2026-10-28"', 'data_end: "2026-10-23"'))
    (problem,) = protocol.freeze_record_violations(protocol.load_protocol(small))
    assert "data_end 2026-10-23 is not the frozen value 2026-10-28 (protocol.freeze.json)" in problem


def test_another_edit_after_the_freeze_is_a_violation_too(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    before = protocol.load_protocol(small).protocol_hash
    small.write_text(small.read_text().replace("holdout_looks: 0", "holdout_looks: 1"))
    after = protocol.load_protocol(small).protocol_hash
    (problem,) = protocol.freeze_record_violations(protocol.load_protocol(small))
    assert f"changed after data_end was frozen: protocol_hash {after} now, {before} at the freeze" in problem


def test_a_data_end_typed_in_without_a_freeze_is_a_violation(small):
    small.write_text(small.read_text().replace("data_end: null", 'data_end: "2026-10-28"'))
    (problem,) = protocol.freeze_record_violations(protocol.load_protocol(small))
    assert "data_end 2026-10-28 has no freeze record (protocol.freeze.json)" in problem


def test_removing_a_frozen_data_end_is_a_violation_and_blocks_a_new_freeze(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    small.write_text(small.read_text().replace('data_end: "2026-10-28"', "data_end: null"))
    (problem,) = protocol.freeze_record_violations(protocol.load_protocol(small))
    assert "data_end null is not the frozen value 2026-10-28" in problem
    with pytest.raises(protocol.ProtocolError, match="records a freeze; refusing to freeze again"):
        protocol.freeze_data_end(small, date(2026, 10, 29))


def test_an_older_freeze_without_a_record_gets_one_when_frozen_again_at_the_same_value(small):
    # data_end frozen before records existed (M1 before review B): freezing the same value records it.
    small.write_text(small.read_text().replace("data_end: null", 'data_end: "2026-10-28"'))
    assert protocol.freeze_data_end(small, date(2026, 10, 28)) is False  # the protocol file is unchanged
    assert protocol.freeze_record_violations(protocol.load_protocol(small)) == []


def test_the_committed_protocols_match_their_freeze_records():
    for synthetic in (False, True):
        assert protocol.freeze_record_violations(protocol.load_protocol(protocol.protocol_path(synthetic=synthetic))) == []


def test_a_frozen_data_end_is_never_moved(small):
    protocol.freeze_data_end(small, date(2026, 10, 28))
    before = small.read_bytes()
    with pytest.raises(protocol.ProtocolError, match="already frozen at 2026-10-28"):
        protocol.freeze_data_end(small, date(2026, 10, 29))
    assert small.read_bytes() == before


@pytest.mark.parametrize(
    "old, new, complaint",
    [
        ('train_start: "1990-01-02"', "train_start: 1990-01-02", "in quotes"),
        ('train_start: "1990-01-02"', 'train_start: "1990-02-30"', "not a calendar date"),
        ('train_start: "1990-01-02"', 'train_start: "2001-01-02"', "train_start <= dev_start"),
        ("seed: 1", "seed: 1\nextra: 2", "unknown keys \\['extra'\\]"),
        ("seed: 1\n", "", "missing keys \\['seed'\\]"),
        ("fold_len: 63", "fold_len: true", "fold_len"),
        ("horizons: [1, 5, 21]", "horizons: [1, 1]", "repeat"),
        ("vix_regime_edges: [15, 25]", "vix_regime_edges: [25, 15]", "increasing"),
        ("data_end: null   # frozen by check", 'data_end: "2020-01-02"', "before holdout_start"),
    ],
)
def test_a_malformed_protocol_is_refused(small, old, new, complaint):
    small.write_text(SMALL.replace(old, new))
    with pytest.raises(protocol.ProtocolError, match=complaint):
        protocol.load_protocol(small)


def test_the_committed_protocols_load_and_share_everything_but_data_end():
    real = protocol.load_protocol(protocol.protocol_path())
    synthetic = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    assert real.train_start == date(1990, 1, 2) and real.holdout_start == date(2024, 7, 1)
    assert real.data_end is not None, "M1 freezes data_end in configs/protocol.yaml"
    assert synthetic.data_end == date(2026, 10, 26)
    without_end = lambda p: {k: v for k, v in p.document.items() if k != "data_end"}  # noqa: E731
    assert without_end(real) == without_end(synthetic)


def test_the_registries_list_five_tenors_in_order_plus_vvix_and_the_sp500():
    indices = protocol.load_iv_indices()
    assert [(i.symbol, i.tenor_days) for i in protocol.term_structure(indices)] == [
        ("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 184), ("VIX1Y", 366)
    ]
    assert {i.symbol: i.group for i in indices}["VVIX"] == "vol_of_vol"
    assert all(i.cboe_url == f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{i.symbol}_History.csv" for i in indices)
    assert [i.ticker for i in protocol.load_instruments()] == ["^GSPC"]


# --- the master calendar -----------------------------------------------------------------


def synthetic_protocol():
    return protocol.load_protocol(protocol.protocol_path(synthetic=True))


def test_calendar_matches_the_figures_adr_0001_measured_in_milestone_0():
    sessions = calendar.xnys_sessions(date(1990, 1, 2), date(2026, 10, 26))
    index = {d: i for i, d in enumerate(sessions)}
    # ADR 0001 4.5: "The first fit already has 2,579 sessions (1990-01-02..2000-03-15, measured)".
    assert index[date(2000, 3, 15)] + 1 == 2579
    # ADR 0001 4.5: first cutoff 2000-03-15, 97 dev folds, 50 burn-in sessions 2000-01-03..2000-03-14.
    cutoff = calendar.first_cutoff(sessions, date(2000, 1, 3), date(2024, 7, 1), 63)
    assert cutoff == date(2000, 3, 15)
    assert (index[date(2024, 7, 1)] - index[cutoff]) / 63 == 97
    assert index[cutoff] - index[date(2000, 1, 3)] == 50
    # ADR 0001 4.5: the holdout ran 2024-07-01..2026-10-26, 583 sessions.
    assert index[date(2026, 10, 26)] - index[date(2024, 7, 1)] + 1 == 583


def test_calendar_matches_the_session_figures_of_adr_0001_4_4():
    # ADR 0001 4.4, measured on XNYS 2000-2025 (amended M1, review A: reproduced, now pinned).
    # Over the 6,539 sessions t of 2000-2025 (6,539/26 years = 251.50 sessions a year):
    #   * t to t+21 spans 29 to 38 calendar days, 199,447 days in all: mean 199,447/6,539 = 30.50;
    #   * the 30 calendar days after t, (t, t+30], hold 15 to 22 sessions, 134,936 in all: mean 20.64.
    sessions = calendar.xnys_sessions(date(1999, 12, 1), date(2026, 3, 1))
    index = {d: i for i, d in enumerate(sessions)}
    years = [d for d in sessions if date(2000, 1, 1) <= d <= date(2025, 12, 31)]
    assert len(years) == 6539 and len(years) / 26 == 251.5
    spans = [(sessions[index[t] + 21] - t).days for t in years]
    assert (min(spans), max(spans), sum(spans)) == (29, 38, 199_447)
    assert round(sum(spans) / len(spans), 2) == 30.50
    within = [sum(1 for s in sessions[index[t] + 1 : index[t] + 31] if (s - t).days <= 30) for t in years]
    assert (min(within), max(within), sum(within)) == (15, 22, 134_936)
    assert round(sum(within) / len(within), 2) == 20.64


@pytest.mark.parametrize(
    "target, horizon, origin",
    [
        # ADR 0001 4.6: the 5-Aug-2024 spike is a holdout target at every horizon...
        (date(2024, 8, 5), 21, date(2024, 7, 5)),
        (date(2024, 8, 5), 5, date(2024, 7, 29)),
        (date(2024, 8, 5), 1, date(2024, 8, 2)),
        # ...and so is early April 2025: origin 2025-03-10 targets 2025-04-08 at h = 21.
        (date(2025, 4, 8), 21, date(2025, 3, 10)),
    ],
)
def test_the_holdout_origins_adr_0001_4_6_names(target, horizon, origin):
    # t+h is the h-th session after t on the master calendar (ADR 0001 4.1).
    dates = list(calendar.master_calendar(synthetic_protocol())["date"])
    i = dates.index(target)
    assert dates[i - horizon] == origin
    assert origin >= date(2024, 7, 1)  # the origin itself is in the holdout


@pytest.mark.parametrize(
    "closed",
    [date(2001, 9, 11), date(2001, 9, 14), date(2012, 10, 29), date(2012, 10, 30), date(2025, 1, 9), date(1994, 4, 27), date(2018, 12, 5)],
)
def test_unscheduled_closures_are_not_sessions(closed):
    # 9/11, Hurricane Sandy, and the national days of mourning (Carter, Nixon, G.H.W. Bush).
    assert calendar.xnys_sessions(closed, closed) == []


def test_master_calendar_reaches_h_max_sessions_past_data_end_with_dense_indices():
    cal = calendar.master_calendar(synthetic_protocol())
    dates = list(cal["date"])
    assert dates[0] == date(1990, 1, 2)
    assert sum(d > date(2026, 10, 26) for d in dates) == 21
    # 21 sessions after 2026-10-26: Thanksgiving (2026-11-26) is skipped, so the last is 2026-11-24.
    assert dates[-1] == date(2026, 11, 24)
    assert list(cal["t_idx"]) == list(range(len(cal)))
    assert cal["fold_id"].isna().all()


def test_periods_are_burn_in_before_the_first_cutoff_dev_until_holdout_then_holdout():
    cal = calendar.master_calendar(synthetic_protocol()).set_index("date")["period"]
    assert cal[date(1990, 1, 2)] == "burn_in"
    assert cal[date(2000, 3, 14)] == "burn_in"
    assert cal[date(2000, 3, 15)] == "dev"
    assert cal[date(2024, 6, 28)] == "dev"
    assert cal[date(2024, 7, 1)] == "holdout"
    assert (cal == "burn_in").sum() == 2578  # 2,579 sessions to 2000-03-15 inclusive, less the cutoff itself


def test_a_truncated_calendar_is_a_prefix_of_the_full_one():
    # 𝒯 comes from dates alone: cutting data_end earlier changes no date, index or period before it.
    full = calendar.master_calendar(synthetic_protocol())
    short = calendar.master_calendar(synthetic_protocol(), data_end=date(2025, 3, 10))
    n = len(short)
    assert short.equals(full.iloc[:n].reset_index(drop=True))


def test_a_data_end_that_is_not_a_session_is_refused():
    with pytest.raises(ValueError, match="not an XNYS session"):
        calendar.master_calendar(synthetic_protocol(), data_end=date(2026, 10, 25))  # a Sunday
