"""Tests for ivf.data.process: what the M1 EDA may see, and when it refuses to run (amended M1, review A).

Review A found four things nothing tested: the EDA printed holdout statistics (a look at the
holdout before any model is frozen, ADR 0001 4.6), it would have read sessions past data_end
(mutant V2 passed all 578 tests), the forward window of the implied-minus-realised spread could be
trailing or one session late (V3, V4: same mean on stationary synthetic data), and a frozen
data_end was never re-checked.
"""

from __future__ import annotations

import math
import shutil
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ivf import config, eda, protocol
from ivf.data import build_interim, calendar, check, process, synthetic
from ivf.features import build

PROTO = protocol.load_protocol(protocol.protocol_path(synthetic=True))
HOLDOUT = PROTO.holdout_start  # 2024-07-01
DATA_END = PROTO.data_end  # 2026-10-26
IV_SYMBOLS = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX"]


# --- the forward window of the spread, origin by origin -------------------------------------------


def one_spike():
    """80 sessions of zero returns except r[40] = 0.3/sqrt(12) = 0.0866025, and rv21_back from them.

    rv21_back[d] = 100*sqrt(12 * sum r^2 over d-20..d) = 100*sqrt(12 * 0.09/12) = 100*0.3 = 30 when
    the window holds session 40 (d = 40..60), 0 for other d >= 20, NaN for d < 20 (21 returns needed).
    """
    r = np.zeros(80)
    r[40] = 0.3 / math.sqrt(12)
    return pd.DataFrame({"rv21_back": build.rv_trailing(r)})


def test_forward_rv21_at_origin_t_covers_exactly_the_21_sessions_after_t():
    # Origin t's window is t+1..t+21; it holds session 40 iff 19 <= t <= 39. rv21_back[t+21] needs
    # t+21 <= 79, so t >= 59 has no value. A trailing window (t-20..t) would put the 30s on 40..60;
    # one session late (t+2..t+22) would put them on 18..38.
    fwd = process.forward_rv21(one_spike())
    assert fwd[:19] == pytest.approx([0.0] * 19, abs=1e-12)
    assert fwd[19:40] == pytest.approx([30.0] * 21, rel=1e-12)
    assert fwd[40:59] == pytest.approx([0.0] * 19, abs=1e-12)
    assert np.isnan(fwd[59:]).all()


def test_the_spread_pairs_each_vix_with_the_rv_of_its_own_next_21_sessions():
    # V = 31 on origins 19..39 (whose next 21 sessions hold the spike, RV 30) and 20 elsewhere (RV 0).
    # Origins 0..58 have a forward RV: n = 59. Gaps: +1 on 21 origins, +20 on 38. Every gap is
    # positive (share 1.0); mean = (21*1 + 38*20)/59 = 781/59 = 13.237288 vol points; in variance
    # (21*(0.0961 - 0.09) + 38*0.04)/59 = (0.1281 + 1.52)/59 = 0.02793390. With trailing RV the
    # origins 40..60 would show 20 - 30 = -10, and the share would fall below 1.
    vix = np.full(80, 20.0)
    vix[19:40] = 31.0
    spread = eda.iv_rv_spread(vix, process.forward_rv21(one_spike()))
    assert spread.n == 59 and spread.share_positive == 1.0
    assert spread.mean_vol_points == pytest.approx(781 / 59, rel=1e-12)
    assert spread.mean_variance == pytest.approx(1.6481 / 59, rel=1e-9)


# --- the EDA never sees the holdout --------------------------------------------------------------


def test_eda_report_refuses_a_holdout_session_or_row():
    features = pd.DataFrame({"log_vix": [3.0], "r_spx": [0.0], "rv21_back": [np.nan]})
    iv = pd.DataFrame({"symbol": ["VIX"], "date": [HOLDOUT], "close": [20.0]})
    with pytest.raises(process.HoldoutRowRefused, match="holdout_start 2024-07-01"):
        process.eda_report([HOLDOUT], ["holdout"], features, iv, HOLDOUT)
    before = date(2024, 6, 28)
    with pytest.raises(process.HoldoutRowRefused):  # the session is fine, but a holdout row came along
        process.eda_report([before], ["dev"], features, iv, HOLDOUT)


def test_eda_inputs_stop_the_day_before_holdout_start():
    cal = calendar.master_calendar(PROTO)
    iv, price = synthetic.synthetic_market(PROTO)
    dates, periods, eda_iv, eda_price = process.eda_inputs(cal, iv, price, HOLDOUT)
    assert dates[-1] == date(2024, 6, 28) and set(periods) == {"burn_in", "dev"}
    assert eda_iv["date"].max() < HOLDOUT and eda_price["date"].max() < HOLDOUT
    assert len(dates) == (cal["date"] < HOLDOUT).sum()


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    """A synthetic raw store and interim tables under <tmp>/implied-vol-forecasting/synthetic."""
    data_dir = tmp_path_factory.mktemp("vol")
    root = data_dir / config.PROJECT_DIRNAME / "synthetic"
    synthetic.write_synthetic_raw(root, PROTO)
    build_interim.build_interim(root, IV_SYMBOLS, ["^GSPC"])
    return data_dir


@pytest.fixture
def layout(built_store, tmp_path, monkeypatch):
    """A private copy of the built store, as the layout run_eda reads."""
    shutil.copytree(built_store, tmp_path / "vol")
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "vol"))
    return config.data_layout(synthetic=True)


def test_run_eda_hands_the_eda_no_holdout_row(layout, monkeypatch, capsys):
    seen = {}
    real = process.eda_report

    def spy(dates, periods, features, iv, holdout_start, *args, **kwargs):
        seen.update(dates=list(dates), periods=list(periods), n_features=len(features), iv_last=iv["date"].max())
        return real(dates, periods, features, iv, holdout_start, *args, **kwargs)

    monkeypatch.setattr(process, "eda_report", spy)
    assert process.run_eda(layout, synthetic=True) == 0
    assert max(seen["dates"]) < HOLDOUT and seen["iv_last"] < HOLDOUT
    assert "holdout" not in seen["periods"] and seen["n_features"] == len(seen["dates"])
    out = capsys.readouterr().out
    assert "before holdout_start 2024-07-01; no holdout row is read" in out
    assert "half-life of ln VIX, burn_in" in out and "half-life of ln VIX, dev" in out
    assert "half-life of ln VIX, holdout" not in out  # the first version printed this


def tables(layout):
    folder = next(layout.processed.glob("run_id=*"))
    return {p.stem: pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}, next(layout.eda.glob("*/eda_report.json")).read_text()


def append_rows_after_data_end(root):
    """A later fetch, in effect: interim rows on the 5 sessions after data_end for VIX and ^GSPC."""
    later = calendar.sessions_after(DATA_END, 5)
    folder = root / "interim"
    iv = pq.read_table(folder / f"{build_interim.IV_TABLE}.parquet")
    price = pq.read_table(folder / f"{build_interim.PRICE_TABLE}.parquet")
    extra_iv = pa.Table.from_pylist(
        [{"symbol": "VIX", "date": d, "open": 90.0, "high": 90.0, "low": 90.0, "close": 90.0, "source": "synthetic", "raw_sha256": "f" * 64} for d in later],
        schema=build_interim.IV_SCHEMA,
    )
    extra_price = pa.Table.from_pylist(
        [{"ticker": "^GSPC", "date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "adj_close": 1.0, "volume": None, "raw_sha256": "f" * 64} for d in later],
        schema=build_interim.PRICE_SCHEMA,
    )
    pq.write_table(pa.concat_tables([iv, extra_iv]), folder / f"{build_interim.IV_TABLE}.parquet")
    pq.write_table(pa.concat_tables([price, extra_price]), folder / f"{build_interim.PRICE_TABLE}.parquet")


def test_rows_after_data_end_change_nothing_the_eda_writes(layout):
    assert process.run_eda(layout, synthetic=True) == 0
    before, report_before = tables(layout)
    assert max(before["feature"]["date"]) == DATA_END and max(before["target_value"]["date"]) == DATA_END
    append_rows_after_data_end(layout.root)
    assert process.run_eda(layout, synthetic=True) == 0
    after, report_after = tables(layout)
    assert report_after == report_before
    for name in ("feature", "target_value", "regime_label", "eda_stat", "calendar"):
        pd.testing.assert_frame_equal(after[name], before[name], check_exact=True)


def test_check_and_eda_fail_when_the_frozen_data_end_is_not_settled(tmp_path, monkeypatch):
    # The synthetic files fetched at 11:00 New York time on data_end itself (2026-10-26T15:00Z).
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    layout = config.data_layout(synthetic=True)
    synthetic.write_synthetic_raw(layout.root, PROTO, fetched_at=datetime(2026, 10, 26, 15, tzinfo=timezone.utc))
    build_interim.build_interim(layout.root, IV_SYMBOLS, ["^GSPC"])
    assert check.run_check(layout, synthetic=True, freeze=False, now=datetime(2026, 10, 29, tzinfo=timezone.utc)) == 1
    assert process.run_eda(layout, synthetic=True) == 1
    assert not layout.processed.exists() and not layout.eda.exists()
