"""Tests for ivf.features.build: returns, rv21 (ADR 0001 4.2), trailing windows (rule L1), gaps (G2), regimes.

Expected values are worked by hand in the comments.
"""

from __future__ import annotations

import functools
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ivf import protocol
from ivf.data import calendar, synthetic
from ivf.features import build


def approx(x, rel=1e-12, abs=1e-12):
    return pytest.approx(x, rel=rel, abs=abs, nan_ok=True)


def test_log_returns_on_consecutive_sessions():
    # ln(110/100) = 0.0953101798, ln(99/110) = ln(0.9) = -0.1053605157
    r = build.log_returns(np.array([100.0, 110.0, 99.0]))
    assert math.isnan(r[0])
    assert r[1:] == approx([0.0953101798043249, -0.1053605156578263])


def test_a_missing_close_makes_two_returns_missing_and_is_never_bridged():
    # Gap policy G2: the return into and out of the gap need it. 121/100 across the gap is not used.
    r = build.log_returns(np.array([100.0, np.nan, 121.0, 133.1]))
    assert np.isnan(r[:3]).all()
    assert r[3] == approx(math.log(1.1))


def test_rv21_of_21_returns_of_one_percent_is_15_8745():
    # 100*sqrt((252/21) * 21 * 0.01**2) = 100*sqrt(12 * 0.0021) = 100*sqrt(0.0252) = 15.874508
    r = np.full(25, 0.01)
    rv = build.rv_trailing(r)
    assert np.isnan(rv[:20]).all()
    assert rv[20:] == approx([15.874507866387544] * 5)


def test_rv21_removes_no_mean():
    # A constant return of 0.01 has zero variance about its mean; rv21 is still 15.87.
    assert build.rv_trailing(np.full(21, 0.01))[-1] == approx(15.874507866387544)


def test_rv21_is_missing_for_every_window_that_touches_a_gap():
    r = np.full(60, 0.01)
    r[30] = np.nan
    rv = build.rv_trailing(r)
    assert not np.isnan(rv[29])
    assert np.isnan(rv[30:51]).all()  # windows ending 30..50 all contain index 30
    assert rv[51] == approx(15.874507866387544)


def test_forward_shifts_values_back_by_h_and_pads_with_nan():
    assert build.forward(np.array([1.0, 2.0, 3.0, 4.0]), 2) == approx([3.0, 4.0, np.nan, np.nan])
    assert np.isnan(build.forward(np.array([1.0, 2.0]), 5)).all()


def test_the_rv21_target_at_origin_t_covers_exactly_the_21_sessions_after_t():
    # Returns are 0 except r[t+1..t+21] = 0.02 for t = 30. The forward target at t is then
    # 100*sqrt(12 * 21 * 0.0004) = 100*sqrt(0.1008) = 31.749016, while rv21 at t itself
    # (the trailing window, known at t) is 0, and at t+1 the forward window has lost one 0.02.
    t = 30
    r = np.zeros(80)
    r[t + 1 : t + 22] = 0.02
    rv = build.rv_trailing(r)
    target = build.forward(rv, 21)
    assert target[t] == approx(31.74901573277509)
    assert rv[t] == 0.0
    # At t+1 the forward window is r[t+2..t+22]: 20 of 0.02 and one 0: 100*sqrt(12*20*0.0004) = 30.983867
    assert target[t + 1] == approx(30.983866769659336)


@functools.cache
def market():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, price = synthetic.synthetic_market(proto)
    dates = [d for d in calendar.master_calendar(proto)["date"] if d <= proto.data_end]
    return proto, dates, iv, price


def test_feature_columns_and_first_empty_rows_match_the_data_model():
    _, dates, iv, price = market()
    f = build.build_features(dates, iv, price)
    assert list(f.columns) == [
        "date", "log_vix", "har_w", "har_m", "r_spx", "rv21_back", "vov21", "iv_rv_spread",
        "vix9d_vix", "vix_vix3m", "vix3m_vix6m", "vix6m_vix1y",
    ]
    first_valid = {c: int(f[c].first_valid_index()) for c in ("log_vix", "har_w", "har_m", "r_spx", "rv21_back", "vov21", "iv_rv_spread")}
    # data-model.md 4: har_w empty for the first 4 sessions, har_m and rv21_back and vov21 and
    # iv_rv_spread for the first 21, r_spx on the first date.
    assert first_valid == {"log_vix": 0, "har_w": 4, "har_m": 21, "r_spx": 1, "rv21_back": 21, "vov21": 21, "iv_rv_spread": 21}
    assert f["vix_vix3m"].first_valid_index() == dates.index(date(2009, 9, 18))  # null before VIX3M


def test_features_obey_rule_l1_truncating_the_data_changes_no_earlier_row():
    # The truncation test of ADR 0001 4.7, on features: build on data that ends at session k
    # and compare with the full build's rows up to k. Any look-ahead would show up as a difference.
    _, dates, iv, price = market()
    full = build.build_features(dates, iv, price)
    for k in (25, 2600, len(dates) - 30):
        cut = dates[k]
        short = build.build_features(dates[: k + 1], iv[iv["date"] <= cut], price[price["date"] <= cut])
        pd.testing.assert_frame_equal(short, full.iloc[: k + 1].reset_index(drop=True))


def test_har_windows_are_trailing_means_of_log_vix():
    # ln V = 1, 2, ..., 30: har_w at index 4 is mean(1..5) = 3; har_m at index 21 is mean(1..22) = 11.5.
    dates = [date(2020, 1, 1 + i) if i < 31 else None for i in range(30)]
    iv = pd.DataFrame({"symbol": "VIX", "date": dates, "close": np.exp(np.arange(1.0, 31.0))})
    price = pd.DataFrame({"ticker": "^GSPC", "date": dates, "close": np.full(30, 100.0)})
    f = build.build_features(dates, iv, price)
    assert f["har_w"][4] == approx(3.0)
    assert f["har_m"][21] == approx(11.5)
    assert f["har_m"][29] == approx(np.mean(np.arange(9.0, 31.0)))  # mean(9..30) = 19.5


def test_vov21_is_the_sample_sd_of_21_log_changes():
    # d ln V alternates +0.1, -0.1, ... 11 of +0.1 and 10 of -0.1 in the window ending at index 21.
    # mean m = 0.1/21; deviations 2.0/21 (11 times) and -2.2/21 (10 times);
    # sum of squares = (11*4.0 + 10*4.84)/441 = 92.4/441 = 0.2095238; /20 = 0.01047619; sqrt = 0.1023533.
    steps = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(21)])
    log_v = np.concatenate([[3.0], 3.0 + np.cumsum(steps)])
    dates = [date(2020, 1, 1 + i) for i in range(22)]
    iv = pd.DataFrame({"symbol": "VIX", "date": dates, "close": np.exp(log_v)})
    price = pd.DataFrame({"ticker": "^GSPC", "date": dates, "close": np.full(22, 100.0)})
    assert build.build_features(dates, iv, price)["vov21"][21] == approx(0.10235326, rel=1e-7)


def test_ratios_put_the_shorter_tenor_on_top_and_iv_rv_spread_is_in_variance_units():
    dates = [date(2020, 1, 1 + i) for i in range(23)]
    iv = pd.concat(
        [pd.DataFrame({"symbol": s, "date": dates, "close": np.full(23, v)}) for s, v in (("VIX9D", 18.0), ("VIX", 20.0), ("VIX3M", 22.0), ("VIX6M", 24.0), ("VIX1Y", 25.0))]
    )
    closes = 100 * np.exp(np.cumsum(np.full(23, 0.01)))  # r = 0.01 every session, so rv21 = 15.8745
    price = pd.DataFrame({"ticker": "^GSPC", "date": dates, "close": closes})
    f = build.build_features(dates, iv, price).iloc[22]
    assert (f.vix9d_vix, f.vix_vix3m, f.vix3m_vix6m, f.vix6m_vix1y) == approx((0.9, 20 / 22, 22 / 24, 0.96))
    # (20/100)**2 - (15.874508/100)**2 = 0.04 - 0.0252 = 0.0148
    assert f.iv_rv_spread == approx(0.0148, rel=1e-9)


def test_target_values_hold_y_and_v_and_skip_dates_without_a_value():
    dates = [date(2020, 1, 1 + i) for i in range(23)]
    iv = pd.DataFrame({"symbol": "VIX", "date": dates, "close": np.where(np.arange(23) == 5, np.nan, 20.0)})
    iv = iv.dropna()
    price = pd.DataFrame({"ticker": "^GSPC", "date": dates, "close": 100 * np.exp(np.cumsum(np.full(23, 0.01)))})
    t = build.build_target_values(dates, iv, price)
    vix = t[t.target_id == "vix_level"]
    assert len(vix) == 22 and date(2020, 1, 6) not in set(vix.date)
    assert vix.v.to_numpy() == approx([0.04] * 22)  # (20/100)**2
    rv = t[t.target_id == "rv21"]
    assert list(rv.date) == [date(2020, 1, 22), date(2020, 1, 23)]  # indices 21 and 22
    assert rv.y.to_numpy() == approx([15.874507866387544] * 2)
    assert rv.v.to_numpy() == approx([0.0252] * 2, rel=1e-12)


def test_regime_labels_use_fixed_edges_and_strict_backwardation():
    # Just below the low edge is low; the edge itself is mid (inclusive above); just below the high
    # edge is mid; the high edge is high. VIX/VIX3M = 1 is contango.
    dates = [date(2020, 1, 1 + i) for i in range(4)]
    iv = pd.concat(
        [
            pd.DataFrame({"symbol": "VIX", "date": dates, "close": [14.99, 15.0, 24.99, 25.0]}),
            pd.DataFrame({"symbol": "VIX3M", "date": dates[:3], "close": [14.99, 14.99, 25.0]}),
        ]
    )
    labels = build.build_regime_labels(dates, iv, (15, 25))
    level = labels[labels.scheme == "vix_level"]
    assert list(level.regime) == ["low", "mid", "mid", "high"]
    term = labels[labels.scheme == "term_structure"]
    assert list(term.regime) == ["contango", "backwardation", "contango"]  # no row where VIX3M is missing
    assert list(term.driver) == approx([1.0, 15.0 / 14.99, 24.99 / 25.0])


def test_term_structure_panel_flags_backwardation_only_above_one():
    dates = [date(2020, 1, 1 + i) for i in range(3)]
    iv = pd.concat(
        [
            pd.DataFrame({"symbol": "VIX", "date": dates, "close": [20.0, 20.0, 20.0]}),
            pd.DataFrame({"symbol": "VIX3M", "date": dates[:2], "close": [20.0, 19.0]}),
        ]
    )
    panel = build.term_structure_panel(dates, iv, ["VIX", "VIX3M"])
    assert panel["backwardation"] == approx([0.0, 1.0, np.nan])
    assert panel["vix_vix3m"] == approx([1.0, 20 / 19, np.nan])
