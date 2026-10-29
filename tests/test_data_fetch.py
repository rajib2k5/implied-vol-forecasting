"""Tests for the raw store and fetch (ADR 0001 D3): ivf.data.raw, ivf.data.fetch, `ivf data verify`.

``network_get`` is replaced by a fake, and the session-wide guard in
``tests/conftest.py`` makes any real connection fail (its routes are tested in
``tests/test_network_guard.py``; a subprocess is not covered). Expected hashes
were computed outside Python with ``sha256sum`` (the command is in each comment).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ivf.data import cli, fetch, raw

pytestmark = pytest.mark.usefixtures("no_network")

# printf 'DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,20.000000,20.000000,20.000000,20.000000\n' | sha256sum
ONE_ROW = b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,20.000000,20.000000,20.000000,20.000000\n"
ONE_ROW_SHA = "61b8c587db2303d14487c6fe689c0be85e45b7ad35dbf5016aeeeb7589aebb88"
# The same file with a second row appended (a day later at Cboe).
# printf 'DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,20.000000,20.000000,20.000000,20.000000\n01/03/1990,21.000000,21.000000,21.000000,21.000000\n' | sha256sum
# Every value in these fixtures is made up (amended M1, review A: the first version held two real
# Cboe rows, byte for byte; tools/licensing/scan_real_rows.py now checks for that).
TWO_ROWS = ONE_ROW + b"01/03/1990,21.000000,21.000000,21.000000,21.000000\n"
TWO_ROWS_SHA = "68031993c6e11efa8806f4dfc485fb2bf9e1214a807b3bcd95d211192af7d3e6"
VIX = fetch.Request("cboe", "VIX", "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
FINAL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
T0 = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)


class FakeNetwork:
    """Serves queued answers and counts calls."""

    def __init__(self, *bodies, status=200):
        self.bodies = list(bodies)
        self.status = status
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return fetch.Response(
            body=self.bodies.pop(0), frame=None, final_url=FINAL, http_status=self.status, tool_version="requests fake"
        )


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def fetch_vix(root, network, clock, *, force=False, max_age=timedelta(days=1)):
    return fetch.fetch_one(root, VIX, max_age=max_age, force=force, clock=clock, network=network)


# --- storing snapshots ---------------------------------------------------------


def test_snapshot_is_named_by_the_sha256_that_sha256sum_computes(tmp_path):
    sha, relative, changed = raw.store_snapshot(tmp_path, "cboe", "VIX", ONE_ROW)
    assert sha == ONE_ROW_SHA
    assert relative == f"raw/cboe/VIX/{ONE_ROW_SHA}.csv"
    assert changed is True
    assert (tmp_path / relative).read_bytes() == ONE_ROW


def test_identical_bytes_are_stored_once_but_every_fetch_is_logged(tmp_path):
    network, clock = FakeNetwork(ONE_ROW, ONE_ROW), Clock(T0)
    first = fetch_vix(tmp_path, network, clock)
    clock.now = T0 + timedelta(days=2)
    second = fetch_vix(tmp_path, network, clock)
    assert [p.name for p in (tmp_path / "raw" / "cboe" / "VIX").iterdir()] == [f"{ONE_ROW_SHA}.csv"]
    records = raw.read_provenance(tmp_path)
    assert [r.changed for r in records] == [True, False]
    assert [r.sha256 for r in records] == [ONE_ROW_SHA, ONE_ROW_SHA]
    assert first.record.fetch_id != second.record.fetch_id


def test_new_content_adds_a_snapshot_and_keeps_the_old_one(tmp_path):
    network, clock = FakeNetwork(ONE_ROW, TWO_ROWS), Clock(T0)
    fetch_vix(tmp_path, network, clock)
    fetch_vix(tmp_path, network, clock, force=True)
    names = sorted(p.name for p in (tmp_path / "raw" / "cboe" / "VIX").iterdir())
    assert names == sorted([f"{ONE_ROW_SHA}.csv", f"{TWO_ROWS_SHA}.csv"])
    assert (tmp_path / "raw" / "cboe" / "VIX" / f"{ONE_ROW_SHA}.csv").read_bytes() == ONE_ROW


def test_an_existing_snapshot_is_never_overwritten(tmp_path):
    # Plant different bytes under the name ONE_ROW would get: the store must not replace them
    # (os.link refuses an existing name), and verify must then flag the file.
    folder = tmp_path / "raw" / "cboe" / "VIX"
    folder.mkdir(parents=True)
    (folder / f"{ONE_ROW_SHA}.csv").write_bytes(b"planted\n")
    _, _, changed = raw.store_snapshot(tmp_path, "cboe", "VIX", ONE_ROW)
    assert changed is False
    assert (folder / f"{ONE_ROW_SHA}.csv").read_bytes() == b"planted\n"


def test_no_temporary_file_is_left_behind(tmp_path):
    raw.store_snapshot(tmp_path, "cboe", "VIX", ONE_ROW)
    raw.store_snapshot(tmp_path, "cboe", "VIX", ONE_ROW)
    assert [p.name for p in (tmp_path / "raw" / "cboe" / "VIX").iterdir()] == [f"{ONE_ROW_SHA}.csv"]


@pytest.mark.parametrize("symbol", ["../VIX", "VIX/../../x", "", "vix lower", "A" * 40])
def test_a_symbol_cannot_become_a_path_outside_the_store(tmp_path, symbol):
    with pytest.raises(raw.RawStoreError):
        raw.store_snapshot(tmp_path, "cboe", symbol, ONE_ROW)


# --- the provenance line -----------------------------------------------------------


def test_provenance_line_holds_exactly_the_data_model_fields(tmp_path):
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0))
    lines = (tmp_path / "raw" / "provenance.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert sorted(record) == sorted(
        ["fetch_id", "source", "symbol", "url", "final_url", "fetched_at", "http_status", "bytes", "sha256", "raw_path", "changed", "tool_version"]
    )
    assert record["source"] == "cboe" and record["symbol"] == "VIX"
    assert record["url"] == VIX.url
    assert record["final_url"] == FINAL  # where the 307 redirect led
    assert record["fetched_at"] == "2026-10-29T12:00:00.000000Z"
    assert record["http_status"] == 200
    assert record["bytes"] == 76  # printf '...' | wc -c
    assert record["sha256"] == ONE_ROW_SHA
    assert record["raw_path"] == f"raw/cboe/VIX/{ONE_ROW_SHA}.csv"
    assert record["changed"] is True
    assert record["tool_version"].startswith("ivf ") and "requests fake" in record["tool_version"]
    assert len(record["fetch_id"]) == 32


def test_a_torn_last_line_blocks_the_next_append(tmp_path):
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0))
    log = tmp_path / "raw" / "provenance.jsonl"
    log.write_bytes(log.read_bytes()[:-10])  # a crash mid-write: no newline at the end
    with pytest.raises(raw.RawStoreError, match="no newline"):
        fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0), force=True)


def test_a_torn_log_stops_a_forced_fetch_before_the_download_and_before_anything_is_stored(tmp_path):
    # Review B: fetch --force with a torn log stored the new snapshot, then failed to log it, which
    # left an orphan. The log is now checked first (amended M1, review B).
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0))
    log = tmp_path / "raw" / "provenance.jsonl"
    log.write_bytes(log.read_bytes()[:-10])
    network = FakeNetwork(TWO_ROWS)
    with pytest.raises(raw.RawStoreError, match="no newline"):
        fetch_vix(tmp_path, network, Clock(T0), force=True)
    assert network.calls == 0
    assert not (tmp_path / "raw" / "cboe" / "VIX" / f"{TWO_ROWS_SHA}.csv").exists()


def test_verify_names_the_orphan_a_crash_between_storing_and_logging_leaves(tmp_path, monkeypatch):
    # A crash after the snapshot's link and before its log line (simulated: the append raises).
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0))

    def crash(root, record):
        raise KeyboardInterrupt("power cut")

    with monkeypatch.context() as patched:
        patched.setattr(raw, "append_provenance", crash)
        with pytest.raises(KeyboardInterrupt):
            fetch_vix(tmp_path, FakeNetwork(TWO_ROWS), Clock(T0), force=True)
    assert (tmp_path / "raw" / "cboe" / "VIX" / f"{TWO_ROWS_SHA}.csv").exists()  # stored, never logged
    problems, _ = raw.verify(tmp_path)
    assert problems == [
        f"raw/cboe/VIX/{TWO_ROWS_SHA}.csv: no provenance line refers to this file (an orphan: a fetch that stopped "
        "between storing the snapshot and logging it, or a file written around fetch); fetch it again with --force "
        "to log it, or delete it"
    ]
    # The remedy the message names works: a forced re-fetch of the same bytes logs the existing file.
    fetch_vix(tmp_path, FakeNetwork(TWO_ROWS), Clock(T0), force=True)
    assert raw.verify(tmp_path) == ([], [])


def test_an_unreadable_provenance_line_is_an_error_not_skipped(tmp_path):
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), Clock(T0))
    log = tmp_path / "raw" / "provenance.jsonl"
    log.write_text(log.read_text() + '{"not": "a record"}\n')
    with pytest.raises(raw.RawStoreError, match=":2:"):
        raw.read_provenance(tmp_path)


def test_latest_fetch_is_the_latest_time_and_on_a_tie_the_later_line(tmp_path):
    clock = Clock(T0)
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), clock)
    fetch_vix(tmp_path, FakeNetwork(TWO_ROWS), clock, force=True)  # same instant, later line
    assert raw.latest_fetches(raw.read_provenance(tmp_path))[("cboe", "VIX")].sha256 == TWO_ROWS_SHA
    clock.now = T0 - timedelta(hours=1)  # an older timestamp written later does not win
    fetch_vix(tmp_path, FakeNetwork(ONE_ROW), clock, force=True)
    assert raw.latest_fetches(raw.read_provenance(tmp_path))[("cboe", "VIX")].sha256 == TWO_ROWS_SHA


# --- max-age, force, failures ---------------------------------------------------


def test_a_fresh_symbol_is_skipped_with_no_network_call_and_no_log_line(tmp_path):
    network, clock = FakeNetwork(ONE_ROW), Clock(T0)
    fetch_vix(tmp_path, network, clock)
    clock.now = T0 + timedelta(hours=23, minutes=59)
    outcome = fetch_vix(tmp_path, network, clock)
    assert outcome.action == "skipped"
    assert network.calls == 1
    assert len(raw.read_provenance(tmp_path)) == 1


def test_a_stale_symbol_is_fetched_again(tmp_path):
    network, clock = FakeNetwork(ONE_ROW, ONE_ROW), Clock(T0)
    fetch_vix(tmp_path, network, clock)
    clock.now = T0 + timedelta(days=1)  # age == max_age is no longer "younger than" it
    assert fetch_vix(tmp_path, network, clock).action == "fetched"
    assert network.calls == 2


def test_force_fetches_a_fresh_symbol(tmp_path):
    network, clock = FakeNetwork(ONE_ROW, ONE_ROW), Clock(T0)
    fetch_vix(tmp_path, network, clock)
    assert fetch_vix(tmp_path, network, clock, force=True).action == "fetched"
    assert network.calls == 2


@pytest.mark.parametrize("status, body", [(404, b"not found"), (200, b""), (500, ONE_ROW)])
def test_a_failed_download_stores_nothing_and_logs_nothing(tmp_path, status, body):
    with pytest.raises(fetch.FetchError):
        fetch_vix(tmp_path, FakeNetwork(body, status=status), Clock(T0))
    assert not (tmp_path / "raw" / "provenance.jsonl").exists()
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize(
    "text, expected",
    [("1d", timedelta(days=1)), ("12h", timedelta(hours=12)), ("30m", timedelta(minutes=30)), ("0", timedelta(0)), ("1.5d", timedelta(hours=36))],
)
def test_max_age_parses_a_number_and_a_unit(text, expected):
    assert fetch.parse_max_age(text) == expected


@pytest.mark.parametrize("text", ["1", "d", "-1d", "1w", "nand", "abc"])
def test_max_age_rejects_anything_else(text):
    with pytest.raises(ValueError):
        fetch.parse_max_age(text)


# --- the bytes stored for a yfinance DataFrame -----------------------------------


def yf_frame(dates, rows, tz="America/New_York"):
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    index = index.tz_localize(tz) if tz else index
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"], index=index)


def test_yfinance_frame_becomes_exactly_these_bytes():
    # Rows arrive out of order and one volume is missing: sorted by date, six
    # decimals, an empty field for the missing volume, "\n" line ends.
    frame = yf_frame(
        ["1990-01-03", "1990-01-02"],
        [[404.00, 406.00, 401.00, 402.50, 402.50, 170000000.0], [400.00, 404.00, 398.00, 404.00, 404.00, np.nan]],
    )
    assert fetch.yfinance_csv_bytes(frame) == (
        b"date,open,high,low,close,adj_close,volume\n"
        b"1990-01-02,400.000000,404.000000,398.000000,404.000000,404.000000,\n"
        b"1990-01-03,404.000000,406.000000,401.000000,402.500000,402.500000,170000000\n"
    )


def test_yfinance_bytes_do_not_depend_on_column_order():
    frame = yf_frame(["1990-01-02"], [[1.0, 2.0, 0.5, 1.5, 1.5, 10.0]])
    assert fetch.yfinance_csv_bytes(frame) == fetch.yfinance_csv_bytes(frame[frame.columns[::-1]])


@pytest.mark.parametrize(
    "make, complaint",
    [
        (lambda: yf_frame(["1990-01-02"], [[1, 2, 0.5, 1.5, 1.5, 10]], tz=None), "tz-aware"),
        (lambda: yf_frame(["1990-01-02"], [[1, 2, 0.5, 1.5, 1.5, 10]], tz="UTC"), "tz-aware"),
        (lambda: yf_frame(["1990-01-02 15:59"], [[1, 2, 0.5, 1.5, 1.5, 10]]), "midnight"),
        (lambda: yf_frame(["1990-01-02"], [[1, 2, 0.5, 1.5, 1.5, 10]]).assign(Dividends=0.0), "columns"),
        (lambda: yf_frame([], []), "no rows"),
    ],
)
def test_an_unexpected_yfinance_frame_is_refused(make, complaint):
    with pytest.raises(fetch.FetchError, match=complaint):
        fetch.yfinance_csv_bytes(make())


def test_a_yfinance_fetch_logs_no_status_and_stores_the_defined_bytes(tmp_path):
    frame = yf_frame(["1990-01-02"], [[1.0, 2.0, 0.5, 1.5, 1.5, 10.0]])
    request = fetch.Request("yfinance", "^GSPC", fetch.YF_URL.format(symbol="^GSPC"))

    def network(req):
        return fetch.Response(body=None, frame=frame, final_url=None, http_status=None, tool_version="yfinance fake")

    outcome = fetch.fetch_one(tmp_path, request, max_age=timedelta(days=1), force=False, clock=Clock(T0), network=network)
    assert outcome.record.http_status is None and outcome.record.final_url is None
    assert (tmp_path / outcome.record.raw_path).read_bytes() == fetch.yfinance_csv_bytes(frame)
    assert outcome.record.raw_path.startswith("raw/yfinance/^GSPC/")


# --- verify ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    network, clock = FakeNetwork(ONE_ROW, TWO_ROWS), Clock(T0)
    fetch_vix(tmp_path, network, clock)
    fetch_vix(tmp_path, network, clock, force=True)
    return tmp_path


def test_verify_passes_on_an_untouched_store(store):
    assert raw.verify(store) == ([], [])


def test_verify_flags_a_symlink_even_when_its_target_has_the_right_bytes(store, tmp_path_factory):
    # Review A (mutant V11): with the symlink check removed, a {sha256}.csv that is a link to a file
    # with exactly the right bytes, elsewhere, passed verify. The raw store holds real files only:
    # a link's target can change under it, and it is a second name for something outside raw/.
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "copy.csv"
    elsewhere.write_bytes(ONE_ROW)
    path = store / "raw" / "cboe" / "VIX" / f"{ONE_ROW_SHA}.csv"
    path.unlink()
    path.symlink_to(elsewhere)
    problems, _ = raw.verify(store)
    assert f"raw/cboe/VIX/{ONE_ROW_SHA}.csv: is a symlink; the raw store holds real files only" in problems


def test_verify_catches_a_changed_byte(store):
    path = store / "raw" / "cboe" / "VIX" / f"{ONE_ROW_SHA}.csv"
    path.write_bytes(ONE_ROW.replace(b"20.00", b"20.05"))
    problems, _ = raw.verify(store)
    assert len(problems) == 1 and "not to its name" in problems[0]


def test_verify_catches_a_missing_snapshot(store):
    (store / "raw" / "cboe" / "VIX" / f"{TWO_ROWS_SHA}.csv").unlink()
    problems, _ = raw.verify(store)
    assert len(problems) == 1 and "missing" in problems[0]


def test_verify_catches_a_file_no_fetch_accounts_for(store):
    stray = b"x\n"  # printf 'x\n' | sha256sum
    sha = "73cb3858a687a8494ca3323053016282f3dad39d42cf62ca4e79dda2aac7d9ac"
    (store / "raw" / "cboe" / "VIX" / f"{sha}.csv").write_bytes(stray)
    (store / "raw" / "cboe" / "VIX" / "notes.txt").write_text("hi")
    problems, _ = raw.verify(store)
    assert len(problems) == 2
    assert any("no provenance line" in p for p in problems) and any("unexpected file" in p for p in problems)


def test_verify_only_warns_about_a_leftover_temporary_file(store):
    # Both names: ".tmp-*" from before review B, and ".<name>.<hex>.tmp" from repo_guard.write_file.
    (store / "raw" / "cboe" / "VIX" / ".tmp-abc").write_bytes(b"half")
    (store / "raw" / "cboe" / "VIX" / f".{ONE_ROW_SHA}.csv.0123abcd.tmp").write_bytes(b"half")
    problems, warnings = raw.verify(store)
    assert problems == [] and len(warnings) == 2


def run_module(*args, env_root):
    import os

    env = dict(os.environ, VOL_DATA_DIR=str(env_root))
    return subprocess.run([sys.executable, "-m", "ivf.data", *args], capture_output=True, text=True, env=env, check=False)


def test_verify_command_exits_0_then_1_after_tampering(tmp_path):
    root = tmp_path / "implied-vol-forecasting"
    fetch_vix(root, FakeNetwork(ONE_ROW), Clock(T0))
    ok = run_module("verify", env_root=tmp_path)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.startswith("ok: 1 raw file(s)")
    (root / "raw" / "cboe" / "VIX" / f"{ONE_ROW_SHA}.csv").write_bytes(b"tampered\n")
    bad = run_module("verify", env_root=tmp_path)
    assert bad.returncode == 1
    assert "MISMATCH" in bad.stderr


# --- the fetch command with a fake network --------------------------------------------


def test_fetch_command_fetches_every_registry_symbol_then_skips_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    calls = []

    def network(request):
        calls.append((request.source, request.symbol))
        if request.source == "yfinance":
            return fetch.Response(None, yf_frame(["1990-01-02"], [[1.0, 2.0, 0.5, 1.5, 1.5, 10.0]]), None, None, "yfinance fake")
        return fetch.Response(ONE_ROW, None, request.url, 200, "requests fake")

    monkeypatch.setattr(fetch, "network_get", network)
    assert cli.main(["fetch"]) == 0
    # amended M6: the tradables VXX and SVXY are fetched like ^GSPC
    assert sorted(symbol for _, symbol in calls) == sorted(["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "^GSPC", "VXX", "SVXY"])
    assert cli.main(["fetch"]) == 0  # all fresh now
    assert len(calls) == 9
    assert capsys.readouterr().out.count("skipped") == 9


def test_one_failed_symbol_does_not_stop_the_others_but_sets_exit_5(tmp_path, monkeypatch):
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))

    def network(request):
        if request.symbol == "VIX3M":
            return fetch.Response(b"", None, request.url, 503, "requests fake")
        if request.source == "yfinance":
            return fetch.Response(None, yf_frame(["1990-01-02"], [[1.0, 2.0, 0.5, 1.5, 1.5, 10.0]]), None, None, "fake")
        return fetch.Response(ONE_ROW, None, request.url, 200, "fake")

    monkeypatch.setattr(fetch, "network_get", network)
    assert cli.main(["fetch"]) == 5
    fetched = {r.symbol for r in raw.read_provenance(tmp_path / "implied-vol-forecasting")}
    assert fetched == {"VIX9D", "VIX", "VIX6M", "VIX1Y", "VVIX", "^GSPC", "VXX", "SVXY"}  # amended M6: + the tradables


def test_the_network_guard_itself_works():
    # If this failed, "no test touches the network" would be an unchecked claim.
    import socket

    from conftest import NetworkBlocked, expect_blocked

    with expect_blocked(), pytest.raises(NetworkBlocked):  # expected: not charged to this test (amended M1, review B)
        socket.create_connection(("cdn.cboe.com", 443), timeout=1)


def test_yfinance_is_asked_to_raise_on_failure_without_the_deprecated_keyword(monkeypatch):
    # yfinance 1.7.0 deprecates history(raise_errors=True) (amended M1, review A): network_get sets
    # yfinance.config.debug.hide_exceptions = False for the call instead, and restores it after.
    # Ticker is replaced, so nothing goes online.
    import yfinance

    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def history(self, **kwargs):
            seen.update(kwargs=kwargs, hide_exceptions=yfinance.config.debug.hide_exceptions)
            return pd.DataFrame()

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    before = yfinance.config.debug.hide_exceptions
    response = fetch.network_get(fetch.Request("yfinance", "^GSPC", fetch.YF_URL.format(symbol="^GSPC")))
    assert seen["symbol"] == "^GSPC" and response.frame is not None
    assert seen["kwargs"] == {"period": "max", "interval": "1d", "auto_adjust": False, "actions": False}
    assert seen["hide_exceptions"] is False
    assert yfinance.config.debug.hide_exceptions == before
