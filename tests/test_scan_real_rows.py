"""Tests for tools/licensing/scan_real_rows.py on a made-up raw store and a scratch git repository.

The scanner looks for real Cboe/Yahoo values copied into the repo. CI has no real data, so these
tests write a small made-up store under ``tmp_path`` and plant its values in the spellings the
scanner must recognise. Every planted case has a control that must stay unflagged: the same
number next to another session's date, or one unit off in its last decimal. Review B (M1) planted
values in 16 shapes and found 9 of them missed entirely; each of those shapes is a case here.

The store's dates are in 2031 and later, where no real data can exist, so this file cannot
itself trip the scan of the real tree; its values are made up and spaced so that no raw value of
the store is within 0.1 of another, which keeps the one-cent controls from matching by chance.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ivf.data import raw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "licensing"))

import scan_real_rows as scan  # noqa: E402


def _sessions(start: date, count: int) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


SESSIONS = _sessions(date(2031, 3, 3), 30)
# VIX-like closes about 1.5 apart; open, high and low within half a point of the close.
CLOSES = [13 + 1.5 * i + 0.020417 + 0.012 * (i * 37 % 30) for i in range(30)]
OHLC = [(c - 0.217, c + 0.341, c - 0.452, c) for c in CLOSES]
# S&P-like closes (thousands separators), one of them on an exact rounding tie, and volumes.
SPX = [4100.235] + [4100 + 13.7 * i + 0.230417 + 0.031 * (i * 11 % 30) for i in range(1, 30)]
VOLUMES = [2_000_000_000 + 7_654_321 * i + 123 for i in range(30)]
I = 10  # the planted session: 2031-03-17, a day above 12, so day-first dates are unambiguous
OTHER = 20  # a session whose date the controls borrow


def _epoch(day: date, unit: int) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * unit


def _store(root: Path, symbol: str, text: str) -> Path:
    content = text.encode()
    sha, path, _ = raw.store_snapshot(root, "synthetic", symbol, content)
    raw.append_provenance(
        root,
        raw.FetchRecord(
            fetch_id=uuid.uuid4().hex,
            source="synthetic",
            symbol=symbol,
            url=f"synthetic://{symbol}",
            final_url=None,
            fetched_at=raw.format_utc(datetime(2031, 5, 1, tzinfo=timezone.utc)),
            http_status=None,
            bytes=len(content),
            sha256=sha,
            raw_path=path,
            changed=True,
            tool_version="test",
        ),
    )
    return root / path


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("data-root")
    vix = "DATE,OPEN,HIGH,LOW,CLOSE\n" + "".join(
        f"{d:%m/%d/%Y},{o:.6f},{h:.6f},{low:.6f},{c:.6f}\n" for d, (o, h, low, c) in zip(SESSIONS, OHLC)
    )
    spx = "date,open,high,low,close,adj_close,volume\n" + "".join(
        f"{d.isoformat()},{c - 3.17:.6f},{c + 5.41:.6f},{c - 6.29:.6f},{c:.6f},{c:.6f},{v}\n"
        for d, c, v in zip(SESSIONS, SPX, VOLUMES)
    )
    # A long series far from the others in value and time, so rule 4 has a size of 6+ digits.
    long = "DATE,VVIX\n" + "".join(f"{d:%m/%d/%Y},{150 + i % 97 * 0.731 + 0.004321:.6f}\n" for i, d in enumerate(_sessions(date(2040, 1, 2), 6000)))
    paths = {"VIX": _store(root, "VIX", vix), "^GSPC": _store(root, "^GSPC", spx), "VVIX": _store(root, "VVIX", long)}
    return {"root": root, "paths": paths}


@pytest.fixture(scope="module")
def index(store) -> scan.RawIndex:
    return scan.RawIndex(store["root"])


def flagged(index: scan.RawIndex, *lines: str) -> list[int]:
    return index.scan_text("\n".join(lines) + "\n")


# --- Rule 3: a dated value, in every date spelling -------------------------------------------

DATE_SPELLINGS = {
    "iso": lambda d, v: f"{d.isoformat()},{v}",
    "slash_iso": lambda d, v: f"{d:%Y/%m/%d},{v}",
    "dotted_iso": lambda d, v: f"{d:%Y.%m.%d} {v}",
    "us": lambda d, v: f"{d:%m/%d/%Y},{v}",
    "us_unpadded": lambda d, v: f"{d.month}/{d.day}/{d.year} {v}",
    "us_dashes": lambda d, v: f"{d:%m-%d-%Y} {v}",
    "day_first": lambda d, v: f"{d:%d/%m/%Y}; {v}",
    "compact": lambda d, v: f"{d:%Y%m%d} {v}",
    "date_call": lambda d, v: f"    (date({d.year}, {d.month}, {d.day}), {v}),",
    "datetime_call": lambda d, v: f"    datetime({d.year}, {d.month}, {d.day}, 16, 15): {v},",
    "timestamp_keywords": lambda d, v: f"pd.Timestamp(year={d.year}, month={d.month}, day={d.day}) -> {v}",
    "js_date_utc": lambda d, v: f"  [Date.UTC({d.year}, {d.month - 1}, {d.day}), {v}],",
    "text_month_first": lambda d, v: f"On {d:%B} {d.day}, {d.year} it closed at {v}.",
    "text_abbreviated": lambda d, v: f"({d:%b}. {d.day} {d.year}) {v}",
    "text_day_first": lambda d, v: f"- {d.day} {d:%B} {d.year}: {v}",
    "text_dashes": lambda d, v: f"{d.day:02d}-{d:%b}-{d.year}  {v}",
    "epoch_seconds": lambda d, v: f'  "{_epoch(d, 1)}": {v},',
    "epoch_milliseconds": lambda d, v: f'  "{_epoch(d, 1000)}": {v},',
    "epoch_nanoseconds": lambda d, v: f"{_epoch(d, 10**9)} {v}",
}


@pytest.mark.parametrize("spelling", DATE_SPELLINGS)
def test_a_close_next_to_its_date_is_flagged_in_every_date_spelling(index, spelling):
    render = DATE_SPELLINGS[spelling]
    close = f"{CLOSES[I]:.2f}"
    assert flagged(index, render(SESSIONS[I], close)) == [1]
    # Controls: the same close next to another session's date, and the right date one cent off.
    assert flagged(index, render(SESSIONS[OTHER], close)) == []
    assert flagged(index, render(SESSIONS[I], f"{round(CLOSES[I], 2) + 0.01:.2f}")) == []


# --- Rule 3: every number spelling ------------------------------------------------------------

# name -> (planted, control one unit off in its last decimal), next to an ISO date of session I.
NUMBER_SPELLINGS = {
    "two_decimals": (f"{CLOSES[I]:.2f}", f"{round(CLOSES[I], 2) + 0.01:.2f}"),
    "three_decimals": (f"{CLOSES[I]:.3f}", f"{round(CLOSES[I], 3) + 0.001:.3f}"),
    "full_precision": (f"{CLOSES[I]:.6f}", f"{round(CLOSES[I], 6) + 0.000001:.6f}"),
    "one_decimal": (f"{CLOSES[I]:.1f}", f"{round(CLOSES[I], 1) + 0.1:.1f}"),
    "per_100": (f"{CLOSES[I] / 100:.4f}", f"{(round(CLOSES[I], 2) + 0.01) / 100:.4f}"),
    "per_100_leading_point": (f"{CLOSES[I] / 100:.4f}"[1:], f"{(round(CLOSES[I], 2) + 0.01) / 100:.4f}"[1:]),
    "percent": (f"{CLOSES[I]:.2f}%", f"{round(CLOSES[I], 2) + 0.01:.2f}%"),
    "thousands_separator": (f"{SPX[I]:,.2f}", f"{round(SPX[I], 2) + 0.01:,.2f}"),
    "volume": (str(VOLUMES[I]), str(VOLUMES[I] + 1)),
}


@pytest.mark.parametrize("spelling", NUMBER_SPELLINGS)
def test_a_dated_value_is_flagged_in_every_number_spelling(index, spelling):
    planted, control = NUMBER_SPELLINGS[spelling]
    assert flagged(index, f"{SESSIONS[I].isoformat()} value {planted} (made up)") == [1]
    assert flagged(index, f"{SESSIONS[OTHER].isoformat()} value {planted} (made up)") == []
    assert flagged(index, f"{SESSIONS[I].isoformat()} value {control} (made up)") == []


def test_a_rounding_tie_matches_both_ways_of_rounding(index):
    # SPX[0] is 4100.235000: half up gives 4100.24, half to even (and a float's repr) 4100.23.
    day = SESSIONS[0].isoformat()
    assert flagged(index, f"{day} 4,100.23") == [1]
    assert flagged(index, f"{day} 4,100.24") == [1]
    assert flagged(index, f"{day} 4,100.22") == []


# --- Rule 3: how far a date reaches -----------------------------------------------------------


def test_a_pretty_json_record_puts_the_value_lines_away_from_its_date(index):
    def record(day):
        fields = {"date": day.isoformat(), **{f"field_{k}": "made-up" for k in range(8)}}
        return json.dumps([{**fields, "value": round(CLOSES[I], 2)}], indent=1)

    text = record(SESSIONS[I])
    value_line = text.split("\n").index(f'  "value": {round(CLOSES[I], 2)}') + 1
    assert value_line - 3 == 9  # the date is on line 3: beyond NEAR_LINES, within FAR_LINES
    assert index.scan_text(text) == [value_line]
    assert index.scan_text(record(SESSIONS[OTHER])) == []


def test_every_date_within_three_lines_counts_not_only_the_nearest(index):
    lines = [
        f'"date": "{SESSIONS[I].isoformat()}",',
        f'"previous": "{SESSIONS[I - 1].isoformat()}",',
        f'"close": {CLOSES[I]:.2f},',
    ]
    assert flagged(index, *lines) == [3]


def test_a_date_out_of_reach_is_not_compared(index):
    close = f"{CLOSES[I]:.2f}"
    # Its own date 13 lines up, and nothing nearer: beyond FAR_LINES.
    assert flagged(index, SESSIONS[I].isoformat(), *["text"] * 12, close) == []
    # Its own date 5 lines up, with another dated line nearer: beyond NEAR_LINES and not the nearest.
    assert flagged(index, SESSIONS[I].isoformat(), "text", SESSIONS[OTHER].isoformat(), "text", "text", close) == []
    # A long line (minified JSON) lends its dates to no other line.
    assert flagged(index, SESSIONS[I].isoformat() + " " + "x" * 400, close) == []


def test_a_one_decimal_value_needs_its_date_on_the_same_line(index):
    assert flagged(index, f"{SESSIONS[I].isoformat()},{CLOSES[I]:.1f}") == [1]
    assert flagged(index, SESSIONS[I].isoformat(), f"{CLOSES[I]:.1f}") == []


# --- Rule 6: runs of consecutive sessions, no dates needed -----------------------------------


def closes(indices, fmt=".2f"):
    return [f"{CLOSES[i]:{fmt}}" for i in indices]


def test_three_consecutive_closes_without_dates_are_flagged(index):
    assert flagged(index, "CLOSES = [", *(f"    {c}," for c in closes(range(I, I + 3))), "]") == [2, 3, 4]
    assert flagged(index, f"CLOSES = [{', '.join(closes(range(I, I + 3)))}]") == [1]
    # Newest first, as a web table shows them.
    assert flagged(index, *closes(range(I + 2, I - 1, -1))) == [1, 2, 3]


def test_two_sessions_or_sessions_with_gaps_are_not_a_run(index):
    assert flagged(index, *closes(range(I, I + 2))) == []
    assert flagged(index, *closes(range(I, I + 6, 2))) == []
    assert flagged(index, *closes([I, I + 1, I + 3])) == []


def test_open_high_low_close_rows_without_dates_are_flagged(index):
    rows = [f"    [{o:.2f}, {h:.2f}, {low:.2f}, {c:.2f}]," for o, h, low, c in OHLC[I : I + 3]]
    assert flagged(index, *rows) == [1, 2, 3]
    # A computed column that equals no raw value (a return) does not break the run.
    returns = [f"{CLOSES[i] / CLOSES[i - 1] - 1:.4f}" for i in range(I, I + 3)]
    assert flagged(index, *(r[:-2] + f", {ret}]," for r, ret in zip(rows, returns))) == [1, 2, 3]


def test_numbers_between_two_sessions_must_belong_to_them(index):
    # A run's numbers 2 apart: the number between must equal the same file at one of the two
    # sessions (an OHLC row), or nothing at all in a run read at 2 or more decimals.
    a, b, c = closes(range(I, I + 3))
    x, y = closes([0, 1])  # raw values, but of other sessions
    assert flagged(index, a, x, b, y, c) == []
    assert flagged(index, a, "777.77", b, "777.78", c) == [1, 3, 5]


def test_one_decimal_runs_need_five_sessions(index):
    assert flagged(index, *closes(range(I, I + 5), ".1f")) == [1, 2, 3, 4, 5]
    assert flagged(index, *closes(range(I, I + 4), ".1f")) == []
    # At 1 decimal, a number between two sessions must belong to them: one that equals nothing
    # does not do (made-up coordinates formed such runs on the real tree).
    interleaved = [x for close in closes(range(I, I + 5), ".1f") for x in (close, "777.7")]
    assert flagged(index, *interleaved) == []
    assert flagged(index, *interleaved[::2]) == [1, 2, 3, 4, 5]


# --- Rules 1, 2, 4 and 5, unchanged -------------------------------------------------------------


def test_rule_1_a_line_found_verbatim_in_a_raw_file(index):
    o, h, low, _ = OHLC[I]
    assert flagged(index, f"{o:.6f},{h:.6f},{low:.6f}") == [1]
    assert flagged(index, f"{o:.6f},{h:.6f},{low + 0.000001:.6f}") == []


def test_rule_2_a_whole_raw_row_inside_a_bytes_literal(index):
    o, h, low, c = OHLC[I]
    row = f"{SESSIONS[I]:%m/%d/%Y},{o:.6f},{h:.6f},{low:.6f},{c:.6f}"
    line = f'CONTENT = b"DATE,OPEN,HIGH,LOW,CLOSE\\n{row}\\n"'
    assert index.rule_lines(line)["1245"] == {0}
    assert index.rule_lines(line.replace(f"{c:.6f}", f"{c + 0.000001:.6f}"))["1245"] == set()


def test_rule_4_a_raw_file_size_or_total_with_separators(index, store):
    total = sum(path.stat().st_size for path in store["paths"].values())
    biggest = max(path.stat().st_size for path in store["paths"].values())
    assert total >= 100_000 and biggest >= 100_000
    assert flagged(index, f"the raw files hold {total:,} bytes") == [1]
    assert flagged(index, f"one file is {biggest:_} bytes") == [1]
    assert flagged(index, f"the raw files hold {total + 1:,} bytes") == []


def test_rule_5_a_raw_snapshot_hash(index, store):
    sha = store["paths"]["VIX"].stem
    assert flagged(index, f"sha256 = {sha}") == [1]
    assert flagged(index, f"sha256 = {sha[:-1]}{'0' if sha[-1] != '0' else '1'}") == []


# --- The command: what it reads and what it prints ----------------------------------------------


def test_the_command_prints_only_file_and_line_and_never_a_value(store, tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    d, c = SESSIONS[I], CLOSES[I]
    files = {
        "notes.md": f"Made up.\n\nOn {d:%B} {d.day}, {d.year} the index was {c:.1f}.\n",
        "fixture.py": f"ROWS = [\n    (date({d.year}, {d.month}, {d.day}), {c:.2f}),\n]\n",
        "closes.txt": "\n".join(closes(range(I, I + 3))) + "\n",
        "clean.txt": "nothing here but 1.5 and 2031-03-17\n",
        "ignored.txt": f"{d.isoformat()},{c:.2f}\n",
        ".gitignore": "ignored.txt\n",
    }
    for name, text in files.items():
        (repo / name).write_text(text)
    code = scan.main(["--root", str(store["root"]), "--repo", str(repo)])
    out, err = capsys.readouterr()
    assert code == 1
    assert out.splitlines() == [
        "closes.txt:1",
        "closes.txt:2",
        "closes.txt:3",
        "fixture.py:2",
        "notes.md:3",
        "5 line(s) flagged in 5 text file(s) against 3 raw snapshot(s)",
    ]
    planted = {f"{c:.1f}", f"{c:.2f}", *closes(range(I, I + 3)), f"{c:.6f}"[:-1]}
    assert not [value for value in planted if value in out or value in err]


def test_the_command_exits_2_without_raw_data(tmp_path, capsys):
    assert scan.main(["--root", str(tmp_path), "--repo", str(REPO)]) == 2
    assert "no raw snapshots" in capsys.readouterr().err
