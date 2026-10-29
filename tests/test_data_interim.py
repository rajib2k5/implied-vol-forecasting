"""Tests for ivf.data.build_interim: strict parsing and the interim tables (ADR 0001 D3; data-model.md 3)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest

from ivf.data import build_interim, fetch, raw
from ivf.data.build_interim import InterimError, parse_cboe_csv, parse_yfinance_csv

pytestmark = pytest.mark.usefixtures("no_network")

# Every value in this file is made up (amended M1, review A: the first version held real Cboe and
# Yahoo closes; tools/licensing/scan_real_rows.py now checks for that).
OHLC = b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,20.000000,20.050000,19.950000,20.000000\n01/03/1990,21.000000,21.000000,21.000000,21.000000\n"


# --- Cboe layouts -------------------------------------------------------------------


def test_ohlc_layout_parses_to_typed_rows():
    parsed = parse_cboe_csv(OHLC, "VIX")
    assert parsed.rows == [
        {"date": date(1990, 1, 2), "open": 20.00, "high": 20.05, "low": 19.95, "close": 20.00},
        {"date": date(1990, 1, 3), "open": 21.00, "high": 21.00, "low": 21.00, "close": 21.00},
    ]
    assert parsed.duplicates_dropped == 0


def test_close_only_layout_named_after_the_symbol_like_vvix():
    parsed = parse_cboe_csv(b"DATE,VVIX\n03/06/2006,80.000000\n", "VVIX")
    assert parsed.rows == [{"date": date(2006, 3, 6), "open": None, "high": None, "low": None, "close": 80.00}]


def test_close_only_layout_named_close():
    assert parse_cboe_csv(b"Date,Close\n2020-01-02,14.25\n", "VIX").rows[0]["close"] == 14.25


def test_header_case_spaces_bom_crlf_and_iso_dates_are_accepted():
    content = "﻿date, Open ,HIGH,low,Close\r\n2020-01-02,14.00,14.50,13.50,14.25\r\n".encode("utf-8")
    assert parse_cboe_csv(content, "VIX").rows[0] == {"date": date(2020, 1, 2), "open": 14.00, "high": 14.50, "low": 13.50, "close": 14.25}


def test_unpadded_us_dates_are_accepted():
    assert parse_cboe_csv(b"DATE,CLOSE\n1/2/1990,20.00\n", "VIX").rows[0]["date"] == date(1990, 1, 2)


def test_an_empty_ohlc_field_is_null_but_an_empty_close_is_an_error():
    assert parse_cboe_csv(b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,,,,20.00\n", "VIX").rows[0]["open"] is None
    with pytest.raises(InterimError, match=":2: close: missing value"):
        parse_cboe_csv(b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,1,1,1,\n", "VIX")


@pytest.mark.parametrize(
    "content, complaint",
    [
        (b"DATE,PRICE\n01/02/1990,20.00\n", "unexpected header"),  # an unknown column
        (b"DATE,VIX9D\n01/02/1990,20.00\n", "unexpected header"),  # close-only named after another index
        (b"DATE,OPEN,HIGH,LOW,CLOSE,VOLUME\n01/02/1990,1,1,1,1,0\n", "unexpected header"),
        (b"DATE,CLOSE,HIGH,LOW,OPEN\n01/02/1990,1,1,1,1\n", "unexpected header"),  # same columns, other order
        (b"DATE,CLOSE\n01/02/1990,20.00\n1990-01-03,21.00\n", ":3: date"),  # mixed date formats
        (b"DATE,CLOSE\n13/01/1990,20.00\n", "not a calendar date"),  # day-first: month 13
        (b"DATE,CLOSE\n02.01.1990,20.00\n", "unrecognised date"),
        (b"DATE,CLOSE\n01/02/1990,20.00,1\n", "3 fields"),
        (b"DATE,CLOSE\n01/02/1990,nan\n", "not a finite number"),
        (b"DATE,CLOSE\n01/02/1990,inf\n", "not a finite number"),
        (b"DATE,CLOSE\n01/02/1990,abc\n", "not a number"),
        (b"DATE,CLOSE\n01/02/1990,0\n", "must be > 0"),
        (b"DATE,CLOSE\n01/02/1990,-1.5\n", "must be > 0"),
        (b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,0,1,1,1\n", "open is 0.0"),
        (b"DATE,CLOSE\n", "a header and no rows"),
        (b"", "empty file"),
        (b"DATE,CLOSE\n01/02/1990,1\n\n01/03/1990,2\n", "blank line"),
        (b"DATE,CLOSE\n01/02/1990,\xff\n", "not UTF-8"),
    ],
)
def test_anything_unexpected_fails_loudly_with_a_location(content, complaint):
    with pytest.raises(InterimError, match=complaint):
        parse_cboe_csv(content, "VIX", "raw/cboe/VIX/x.csv")


def test_an_exact_duplicate_row_is_kept_once_and_counted():
    content = b"DATE,CLOSE\n01/02/1990,20.00\n01/03/1990,21.00\n01/02/1990,20.00\n"
    parsed = parse_cboe_csv(content, "VIX")
    assert [row["date"] for row in parsed.rows] == [date(1990, 1, 2), date(1990, 1, 3)]
    assert parsed.duplicates_dropped == 1
    assert parsed.duplicate_dates == ["1990-01-02"]


def test_a_date_with_two_different_values_is_an_error():
    with pytest.raises(InterimError, match="appears twice with different values"):
        parse_cboe_csv(b"DATE,CLOSE\n01/02/1990,20.00\n01/02/1990,20.05\n", "VIX")


def test_rows_come_out_sorted_by_date():
    parsed = parse_cboe_csv(b"DATE,CLOSE\n01/03/1990,21.00\n01/02/1990,20.00\n", "VIX")
    assert [row["date"] for row in parsed.rows] == [date(1990, 1, 2), date(1990, 1, 3)]


# --- the yfinance CSV ------------------------------------------------------------------


def test_yfinance_csv_parses_including_a_missing_volume():
    content = b"date,open,high,low,close,adj_close,volume\n1990-01-02,400.0,404.00,398.00,404.00,404.00,\n"
    assert parse_yfinance_csv(content).rows == [
        {"date": date(1990, 1, 2), "open": 400.0, "high": 404.00, "low": 398.00, "close": 404.00, "adj_close": 404.00, "volume": None}
    ]


@pytest.mark.parametrize(
    "content, complaint",
    [
        (b"Date,Open,High,Low,Close,Adj Close,Volume\n", "unexpected header"),
        (b"date,open,high,low,close,adj_close,volume\n1990-01-02,1,1,1,,1,5\n", "close: missing value"),
        (b"date,open,high,low,close,adj_close,volume\n1990-01-02,1,1,1,1,1,5.5\n", "not an integer"),
        (b"date,open,high,low,close,adj_close,volume\n1990-01-02,0,1,1,1,1,5\n", "prices must be > 0"),
        (b"date,open,high,low,close,adj_close,volume\n1990-01-02,1,1,1,1,1,-5\n", "negative volume"),
    ],
)
def test_yfinance_csv_refuses_anything_unexpected(content, complaint):
    with pytest.raises(InterimError, match=complaint):
        parse_yfinance_csv(content)


# --- snapshot selection and the tables ------------------------------------------------

T0 = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
YF = b"date,open,high,low,close,adj_close,volume\n1990-01-02,400.0,404.00,398.00,404.00,404.00,150000000\n"


def put(root, source, symbol, content, when):
    """Store a snapshot and its provenance line the way fetch does."""
    sha, path, changed = raw.store_snapshot(root, source, symbol, content)
    raw.append_provenance(
        root,
        raw.FetchRecord("f" * 32, source, symbol, "u", None, raw.format_utc(when), None, len(content), sha, path, changed, "test"),
    )
    return sha


def test_tables_have_the_data_model_columns_types_and_raw_sha256(tmp_path):
    vix_sha = put(tmp_path, "cboe", "VIX", OHLC, T0)
    spx_sha = put(tmp_path, "yfinance", "^GSPC", YF, T0)
    build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"])
    iv = pq.read_table(tmp_path / "interim" / "iv_index_daily.parquet")
    price = pq.read_table(tmp_path / "interim" / "price_daily.parquet")
    assert iv.schema.names == ["symbol", "date", "open", "high", "low", "close", "source", "raw_sha256"]
    assert price.schema.names == ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "raw_sha256"]
    assert str(iv.schema.field("date").type) == "date32[day]"
    assert str(price.schema.field("volume").type) == "int64"
    assert iv.column("raw_sha256").to_pylist() == [vix_sha, vix_sha]
    assert iv.column("source").to_pylist() == ["cboe", "cboe"]
    assert price.column("raw_sha256").to_pylist() == [spx_sha]


def test_the_latest_snapshot_is_used_unless_one_is_pinned(tmp_path):
    old = put(tmp_path, "cboe", "VIX", b"DATE,CLOSE\n01/02/1990,20.00\n", T0)
    new = put(tmp_path, "cboe", "VIX", OHLC, T0 + timedelta(days=1))
    put(tmp_path, "yfinance", "^GSPC", YF, T0)
    build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"])
    assert set(pq.read_table(tmp_path / "interim" / "iv_index_daily.parquet").column("raw_sha256").to_pylist()) == {new}
    build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"], pins={("cboe", "VIX"): old})
    table = pq.read_table(tmp_path / "interim" / "iv_index_daily.parquet")
    assert table.column("raw_sha256").to_pylist() == [old]
    assert table.column("close").to_pylist() == [20.00]


def test_a_pin_to_a_hash_that_symbol_never_had_is_an_error(tmp_path):
    put(tmp_path, "cboe", "VIX", OHLC, T0)
    with pytest.raises(InterimError, match="no fetch of that symbol stored that hash"):
        build_interim.select_snapshots(raw.read_provenance(tmp_path), {("cboe", "VIX"): "0" * 64})


@pytest.mark.parametrize("text", ["cboe/VIX", "cboe/VIX=abc", "VIX=" + "0" * 64])
def test_snapshot_arguments_must_be_source_symbol_equals_sha256(text):
    with pytest.raises(InterimError):
        build_interim.parse_pins([text])


def test_a_registry_symbol_without_a_snapshot_is_an_error(tmp_path):
    put(tmp_path, "cboe", "VIX", OHLC, T0)
    put(tmp_path, "yfinance", "^GSPC", YF, T0)
    with pytest.raises(InterimError, match="VIX3M: no snapshot"):
        build_interim.build_interim(tmp_path, ["VIX", "VIX3M"], ["^GSPC"])


def test_a_snapshot_whose_bytes_changed_is_refused(tmp_path):
    sha = put(tmp_path, "cboe", "VIX", OHLC, T0)
    put(tmp_path, "yfinance", "^GSPC", YF, T0)
    (tmp_path / "raw" / "cboe" / "VIX" / f"{sha}.csv").write_bytes(OHLC.replace(b"20.00", b"99.99"))
    with pytest.raises(InterimError, match="does not match its hash"):
        build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"])


def test_two_sources_for_one_symbol_are_an_error_in_milestone_1(tmp_path):
    put(tmp_path, "cboe", "VIX", OHLC, T0)
    put(tmp_path, "fred", "VIX", OHLC, T0)
    put(tmp_path, "yfinance", "^GSPC", YF, T0)
    with pytest.raises(InterimError, match="no rule to merge sources"):
        build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"])


def test_the_build_report_counts_dropped_duplicates(tmp_path):
    put(tmp_path, "cboe", "VIX", OHLC + b"01/03/1990,21.000000,21.000000,21.000000,21.000000\n", T0)
    put(tmp_path, "yfinance", "^GSPC", YF, T0)
    report = build_interim.build_interim(tmp_path, ["VIX"], ["^GSPC"])
    vix = next(entry for entry in report.files if entry["symbol"] == "VIX")
    assert vix["duplicates_dropped"] == 1 and vix["duplicate_dates"] == ["1990-01-03"]
    assert vix["rows"] == 2


def test_the_stored_yfinance_bytes_parse_back_to_the_frame():
    import numpy as np
    import pandas as pd

    index = pd.DatetimeIndex(["1990-01-02", "1990-01-03"]).tz_localize("America/New_York")
    frame = pd.DataFrame(
        {"Open": [400.0, 404.00], "High": [404.00, 406.00], "Low": [398.00, 401.00], "Close": [404.00, 402.50], "Adj Close": [404.00, 402.50], "Volume": [np.nan, 170000000.0]},
        index=index,
    )
    rows = parse_yfinance_csv(fetch.yfinance_csv_bytes(frame)).rows
    assert [row["close"] for row in rows] == [404.00, 402.50]
    assert [row["volume"] for row in rows] == [None, 170000000]
