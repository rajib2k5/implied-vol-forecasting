"""Look for real Cboe/Yahoo data copied into the repository (ADR 0001 D2). Local only.

    uv run python tools/licensing/scan_real_rows.py            # ROOT from $VOL_DATA_DIR
    uv run python tools/licensing/scan_real_rows.py --root DIR [--repo DIR]

It reads every raw snapshot under ``ROOT/raw/{source}/{symbol}/{sha256}.csv`` (the real data; the
synthetic store under ``ROOT/synthetic/`` is not licensed and is not read) and every text file git
would commit (tracked, plus untracked and not ignored: ``git ls-files -co --exclude-standard``).
A repository line is flagged when:

1. it is at least 12 characters long (after stripping) and appears verbatim in a raw file;
2. it contains a whole raw data row (a raw line with a digit, at least 12 characters), for
   example inside a Python bytes literal, where ``\\n`` escapes keep rule 1 from matching;
3. it holds a number equal to a raw value of a date named near it. "Equal" means equal to the raw
   value rounded to the number's own decimals (half up and half to even both count), or, for a
   number with 3 or more decimals, to the raw value divided by 100 (a vol written as a decimal:
   0.1694 for 16.94, read at 2 decimals). The number needs 3 or more significant digits, and then:
   - read at 2 or more decimals, the date is on the same line within ``NEAR_CHARS`` characters,
     on any line up to ``NEAR_LINES`` lines away, or on the nearest dated line above or below up
     to ``FAR_LINES`` lines away (one pretty-printed JSON record);
   - read at 1 decimal, the date is on the same line within ``NEAR_CHARS`` characters;
   - an integer of 6 or more digits (a volume) is treated like 2 decimals.
   A date is any of: ``1990-05-25``, ``1990/05/25`` or ``1990.05.25`` (one-digit month or day
   allowed); ``05/25/1990``, ``5-25-1990`` and day first ``25.05.1990`` (both readings when
   ambiguous); compact ``19900525``; ``(1990, 5, 25)`` as in ``date(...)``, ``datetime(...)`` or
   ``Timestamp(...)`` (``Date(1990, 4, 25)`` and ``Date.UTC(...)`` count months from 0, as
   JavaScript does); ``May 25, 1990``, ``25 May 1990`` or ``25-May-1990`` (full or 3-letter month
   names, any case); epoch seconds, milliseconds, microseconds or nanoseconds (the UTC day, as
   pandas ``to_json`` writes index keys). Written years 1900 to 2099 only;
4. it holds an integer of 6 or more digits (thousands separators ``,``, ``_`` or a space allowed)
   equal to the size of a raw file, the total of all raw files, or the total of the latest
   snapshot per symbol;
5. it holds a 64-hex string equal to a raw snapshot's SHA-256;
6. it holds one of ``RUN_MIN`` or more numbers that, in file order and with no date needed, equal
   consecutive sessions of one raw series (one column of one snapshot), oldest or newest first: a
   list of closes, one per line or all on one line. The numbers are rule 3's (same readings). Up
   to ``RUN_MAX_STRIDE`` - 1 other numbers may sit between two sessions, the same count each time,
   if each equals a value of that raw file at one of the two sessions (the other columns of an
   OHLC row) or, in a run read at 2 or more decimals, equals no raw value (a computed column).
   A run with any number read at 1 decimal must be ``RUN_MIN_ONE_DECIMAL`` long.

A number is digits with a decimal part, or an integer of 6 or more digits; thousands commas
(``2,345.67``, also read as ``345.67``), a decimal comma (``16,94``, not with exactly 3 digits
after it, which is read as thousands; M1 final check) and a leading point (``.1694``) are allowed,
and a full stop after it ends it. Digits inside a written date are the date, not a number.

Rules 2 to 5 go beyond rule 1 because review A's two findings would pass rule 1 alone: the 1990
VIX rows sat inside ``b"...\\n..."`` literals, and the byte count was a sum with thousands commas.
Rule 3's date spellings, its 1-decimal and /100 readings, ``FAR_LINES`` and rule 6 come from
review B, which planted synthetic stand-ins for real values in 16 shapes and found 9 missed
entirely (``date(...)`` objects, epoch-ms keys, lists of closes without dates, pretty JSON with
the value more than 3 lines from its date, compact, slash-ISO and textual dates, 1-decimal
values, /100 rescaling): the scanner then knew only ``MM/DD/YYYY`` and ``YYYY-MM-DD``, 2 or more
decimals, within 3 lines. The window, the run lengths and the in-between condition were chosen
by counting the lines each variant flags in the repo against the real store (only counts and
``file:line`` printed); the counts are next to each constant and in ``RawIndex.runs``.

Rules 3 and 6 can flag a coincidence: a made-up number that happens to equal a real value of a
nearby date, or made-up numbers that happen to follow real consecutive sessions. The fix is the
same: pick other made-up numbers. Still not caught: values transformed beyond rounding and /100
(returns, logs, differences, ratios, sums, truncation); a lone value with no date near it and no
run; a 1-decimal value whose date is on another line; fewer than 3 consecutive sessions (5 at
1 decimal); every k-th session (weekly or monthly closes, k > 1); Excel serial dates
(``37214``); 2-digit years and date spellings not listed; numbers in scientific notation or with
fewer than 3 significant digits; numbers spelled out in words; and anything in a file git
ignores or that is binary or not UTF-8 (images, PDFs, Parquet). The M1 final check found the
k-th-session and Excel-serial gaps, and added the decimal comma (tools/licensing/README.md).

It prints only ``file:line`` for each flagged line and a count, never the text that matched, so its
output can be pasted anywhere. Exit 0 clean, 1 something flagged, 2 no raw data to compare with.
It is not a test: CI has no raw data. Run it before every commit (the walkthrough's gates).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

from ivf import config
from ivf.data import raw

MIN_LINE = 12
NEAR_CHARS = 120  # rule 3, same line: a one-line JSON file names many dates far from any value
# Rule 3, other lines. Lines of the repo flagged against the real store (review B fix): every date
# within 3 lines plus the nearest dated line within 12, as set: 1 (a coincidence); every date
# within 10 lines: 3; within 15: 6; the nearest within 20: 2. Letting 1-decimal numbers use these
# lines too: 2 against the real store, and 1 against the synthetic one (0 as set).
NEAR_LINES = 3  # every date on a line up to 3 lines away: a dict or a table row over a few lines
FAR_LINES = 12  # the nearest dated line above and below: a pretty-printed JSON record (6 lines in review B's)
LONG_LINE = 300  # a longer line (minified JSON) lends its dates to no other line
MIN_SIGNIFICANT = 3
MIN_INT_DIGITS = 6
RUN_MIN = 3  # rule 6, read at 2 or more decimals (or integers); 2 flags 135 lines of the repo
RUN_MIN_ONE_DECIMAL = 5  # rule 6, any number read at 1 decimal; 4 flags 8 lines of the repo
RUN_MAX_STRIDE = 8  # rule 6: up to 7 numbers between two sessions (OHLC, adj close, volume)
SCALE = 6  # raw values are compared in millionths, the raw files' own precision
_ROW_BITS = 24  # rule 6 packs (series, row) into one int: series << _ROW_BITS | row ...
_COLUMN_BITS = 6  # ... with series = file << _COLUMN_BITS | column
_ROW_MASK = (1 << _ROW_BITS) - 1
_FILE_SHIFT = _ROW_BITS + _COLUMN_BITS

# Rule 2 and the raw index: the two spellings the raw files use, at the start of a row.
_US_DATE = re.compile(r"(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_BIG_INT = re.compile(r"(?<![\w.,])\d{1,3}(?:[,_ ]\d{3})+(?![\w.,]?\d)|(?<![\w.,])\d{6,}(?![\w.])")
_SHA = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_RAW_ALPHABET = re.compile(r"[0-9A-Za-z/.,:^\- ]+")
_RAW_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

# Rule 3: every date spelling (module docstring). Years 1900 to 2099 only.
_YEAR = r"((?:19|20)\d{2})"
_MONTH_NAME = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_YMD = re.compile(r"(?<![\d.])" + _YEAR + r"([-/.])(\d{1,2})\2(\d{1,2})(?!\d)")
_MDY = re.compile(r"(?<![\d.])(\d{1,2})([-/.])(\d{1,2})\2" + _YEAR + r"(?!\d)")
_COMPACT = re.compile(r"(?<![\d.])" + _YEAR + r"(\d{2})(\d{2})(?!\d|\.\d)")
_TUPLE = re.compile(
    r"[(\[]\s*(?:year\s*=\s*)?" + _YEAR + r"\s*,\s*(?:month\s*=\s*)?(\d{1,2})\s*,\s*(?:day\s*=\s*)?(\d{1,2})(?!\d)"
)
_JS_DATE = re.compile(r"\bDate(?:\.UTC)?\s*$")
_TEXT_MDY = re.compile(r"\b" + _MONTH_NAME + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+" + _YEAR + r"(?!\d)", re.I)
_TEXT_DMY = re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?[\s-]+" + _MONTH_NAME + r"\.?,?[\s-]+" + _YEAR + r"(?!\d)", re.I)
_EPOCH = re.compile(r"(?<![\d.])(\d{9,19})(?!\d)")  # the lengths in _EPOCH_UNIT only
_EPOCH_UNIT = {9: 1, 10: 1, 12: 10**3, 13: 10**3, 15: 10**6, 16: 10**6, 18: 10**9, 19: 10**9}
_EPOCH_DAY0 = date(1970, 1, 1)

# Rules 3 and 6: numbers. Thousands-grouped ones are read both ways (``2,345.67`` and ``345.67``).
_PLAIN = re.compile(r"(?<![\w.])(?:\d+\.\d+|\.\d+|\d{6,})(?!\w|\.\d)")  # shorter integers never qualify
_GROUPED = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\w|[.,]\d)")
# A decimal comma (``16,94``, as a ``;``-separated European CSV writes 16.94; M1 final check). Not 3
# digits after the comma, which is a thousands group, and nothing glued on either side.
_DECIMAL_COMMA = re.compile(r"(?<![\w.,])\d+,(?:\d{1,2}|\d{4,6})(?![\w,]|\.\d)")


def _dates_in(text: str) -> list[tuple[int, int, date]]:
    """``(start, end, date)`` for every M/D/Y or ISO date token in ``text`` that is a real calendar date.

    Only the raw files' own spellings: the raw index and rule 2 use it. Rule 3 uses ``_date_spans``.
    """
    found = []
    for pattern, order in ((_US_DATE, (2, 0, 1)), (_ISO_DATE, (0, 1, 2))):
        for match in pattern.finditer(text):
            parts = [int(g) for g in match.groups()]
            try:
                found.append((match.start(), match.end(), date(parts[order[0]], parts[order[1]], parts[order[2]])))
            except ValueError:
                continue
    return found


def _date_spans(text: str) -> list[tuple[int, int, date]]:
    """``(start, end, date)`` for every date in ``text``, in any spelling rule 3 knows."""
    found: list[tuple[int, int, date]] = []

    def add(match: re.Match, ymd: set[tuple[int, int, int]]) -> None:
        for year, month, day in ymd:
            try:
                found.append((match.start(), match.end(), date(year, month, day)))
            except ValueError:
                continue

    for m in _YMD.finditer(text):
        add(m, {(int(m[1]), int(m[3]), int(m[4]))})
    for m in _MDY.finditer(text):
        a, b, year = int(m[1]), int(m[3]), int(m[4])
        add(m, {(year, a, b), (year, b, a)})
    for m in _COMPACT.finditer(text):
        add(m, {(int(m[1]), int(m[2]), int(m[3]))})
    for m in _TUPLE.finditer(text):
        js = _JS_DATE.search(text, max(0, m.start() - 12), m.start()) is not None
        add(m, {(int(m[1]), int(m[2]) + js, int(m[3]))})
    for m in _TEXT_MDY.finditer(text):
        add(m, {(int(m[3]), _MONTHS.index(m[1][:3].lower()) + 1, int(m[2]))})
    for m in _TEXT_DMY.finditer(text):
        add(m, {(int(m[3]), _MONTHS.index(m[2][:3].lower()) + 1, int(m[1]))})
    for m in _EPOCH.finditer(text):
        digits = m[1]
        if len(digits) not in _EPOCH_UNIT:
            continue
        days = int(digits) // (86400 * _EPOCH_UNIT[len(digits)])
        found.append((m.start(), m.end(), _EPOCH_DAY0 + timedelta(days=days)))
    found.sort(key=lambda span: span[0])
    return found


def _numbers(text: str) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """``(start, end, readings)`` for every number rules 3 and 6 consider, in file order.

    A reading is ``(decimals, digits)``: the number's value is ``digits / 10**decimals``. A number
    with 3 or more decimals is also read as points: ``0.1694`` gives ``(4, 1694)`` and ``(2, 1694)``.
    No written date yields a number: ``_PLAIN`` needs a decimal part not followed by another
    ``.digit``, or 6 or more digits. A compact or epoch date is also a number (it may be a volume).
    """
    grouped = [(m.start(), m.end()) for m in _GROUPED.finditer(text)]
    group_starts = [start for start, _ in grouped]
    tokens = {start: (end, [text[start:end].replace(",", "")]) for start, end in grouped}
    for m in _PLAIN.finditer(text):
        k = bisect_right(group_starts, m.start()) - 1
        if k >= 0 and m.start() < grouped[k][1]:
            # Inside a grouped number (the "345.67" of "2,345.67"): another reading of that number.
            tokens[grouped[k][0]][1].append(m.group())
        else:
            tokens[m.start()] = (m.end(), [m.group()])
    taken = sorted((start, end) for start, (end, _) in tokens.items())
    for m in _DECIMAL_COMMA.finditer(text):
        k = bisect_right(taken, (m.start(), len(text))) - 1
        if k >= 0 and m.start() < taken[k][1]:
            continue  # inside a number already read
        tokens[m.start()] = (m.end(), [m.group().replace(",", ".")])
    out = []
    for start, (end, spellings) in sorted(tokens.items()):
        readings: list[tuple[int, int]] = []
        for spelling in spellings:
            whole, _, fraction = spelling.partition(".")
            digits = int(whole + fraction or "0")
            if not fraction:
                if len(whole) >= MIN_INT_DIGITS and digits:
                    readings.append((0, digits))
            elif len(str(digits)) >= MIN_SIGNIFICANT and digits:
                readings.append((len(fraction), digits))
                if len(fraction) >= 3:
                    readings.append((len(fraction) - 2, digits))
        if readings:
            out.append((start, end, readings))
    return out


def _interval(digits: int, decimals: int) -> tuple[int, int]:
    """The raw values (in millionths) that round to ``digits / 10**decimals``, ends included."""
    if decimals <= SCALE:
        unit = 10 ** (SCALE - decimals)
        return digits * unit - unit // 2, digits * unit + unit // 2
    unit = 10 ** (decimals - SCALE)  # more decimals than the raw files: equal at their precision
    return -((unit - 2 * digits) // (2 * unit)), (2 * digits + unit) // (2 * unit)


def _millionths(cell: str) -> int | None:
    """A raw cell in millionths (``"16.936306"`` -> 16936306), or None if it is not a plain number."""
    if not _RAW_NUMBER.fullmatch(cell):
        return None
    whole, _, fraction = cell.partition(".")
    if len(fraction) > SCALE:  # more decimals than the raw files use: round, half to even
        return int((Decimal(cell) * 10**SCALE).to_integral_value(ROUND_HALF_EVEN))
    magnitude = int(whole.lstrip("-")) * 10**SCALE + int(fraction.ljust(SCALE, "0"))
    return -magnitude if whole.startswith("-") else magnitude


class RawIndex:
    """What rules 1 to 6 compare against, built from the raw snapshots."""

    def __init__(self, root: Path):
        self.files = sorted((root / "raw").glob("*/*/*.csv"))
        self.texts: list[str] = []
        self.rows_by_prefix: dict[str, list[str]] = defaultdict(list)
        self.by_date: dict[date, list[int]] = defaultdict(list)  # every raw value of a date, in millionths
        self.sizes: set[int] = set()
        self.shas: set[str] = set()
        cells: list[tuple[int, int]] = []  # (millionths, series << _ROW_BITS | row)
        for number, path in enumerate(self.files):
            blob = path.read_bytes()
            text = blob.decode("utf-8", errors="replace")
            self.texts.append(text)
            self.sizes.add(len(blob))
            self.shas.add(path.stem)
            row = 0
            for line in text.splitlines():
                line = line.strip()
                dates = _dates_in(line)
                if not dates or dates[0][0] != 0:
                    continue  # a header, or not a data row
                day = dates[0][2]
                if len(line) >= MIN_LINE:
                    self.rows_by_prefix[line[: dates[0][1]]].append(line)
                for column, cell in enumerate(line[dates[0][1] :].split(",")[1:]):
                    value = _millionths(cell.strip())
                    if value is not None and column < 1 << _COLUMN_BITS and row <= _ROW_MASK:
                        self.by_date[day].append(value)
                        cells.append((value, (number << _COLUMN_BITS | column) << _ROW_BITS | row))
                row += 1
        for values in self.by_date.values():
            values.sort()
        cells.sort()
        self.values = [value for value, _ in cells]
        self.codes = [code for _, code in cells]
        self.sizes.add(sum(path.stat().st_size for path in self.files))
        latest = raw.latest_fetches(raw.read_provenance(root)).values()
        self.sizes.add(sum((root / record.raw_path).stat().st_size for record in latest if (root / record.raw_path).is_file()))
        self.joined = "\n".join(self.texts)

    def hits(self, line: str) -> bool:
        """Rules 1, 2, 4 and 5 for one line (rules 3 and 6 need the whole file: ``scan_text``)."""
        stripped = line.strip()
        # Rule 1: the whole line is in a raw file. Raw files hold only this alphabet, so anything
        # else cannot match and the expensive search is skipped.
        if len(stripped) >= MIN_LINE and _RAW_ALPHABET.fullmatch(stripped) and stripped in self.joined:
            return True
        dates = _dates_in(line)
        for start, end, day in dates:
            # Rule 2: a whole raw row starts at this date.
            for row in self.rows_by_prefix.get(line[start:end], ()):
                if line.startswith(row, start):
                    return True
        # Rule 4: a raw file size, or a total of them.
        for match in _BIG_INT.finditer(line):
            if int(re.sub(r"[,_ ]", "", match.group())) in self.sizes:
                return True
        # Rule 5: a raw snapshot's hash.
        return any(token in self.shas for token in _SHA.findall(line))

    def _on(self, day: date, low: int, high: int) -> bool:
        values = self.by_date.get(day)
        if not values:
            return False
        k = bisect_left(values, low)
        return k < len(values) and values[k] <= high

    def scan_text(self, text: str) -> list[int]:
        """The 1-based numbers of the lines of ``text`` that rules 1 to 6 flag."""
        return sorted({i + 1 for lines in self.rule_lines(text).values() for i in lines})

    def rule_lines(self, text: str) -> dict[str, set[int]]:
        """The 0-based lines of ``text`` each rule flags: ``"1245"`` (rules 1, 2, 4, 5), ``"3"``, ``"6"``."""
        lines = text.split("\n")
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line) + 1)
        found: dict[str, set[int]] = {"1245": {i for i, line in enumerate(lines) if self.hits(line)}, "3": set(), "6": set()}

        spans = _date_spans(text)
        same_line: dict[int, list[tuple[int, int, date]]] = defaultdict(list)
        lends: dict[int, list[date]] = defaultdict(list)  # line -> its dates, if short enough to lend
        for start, end, day in spans:
            i = bisect_right(starts, start) - 1
            same_line[i].append((start, end, day))
            if len(lines[i]) <= LONG_LINE:
                lends[i].append(day)
        lending = sorted(lends)

        def other_lines(i: int) -> list[date]:
            """Dates on every line up to NEAR_LINES away, and on the nearest dated line above and below up to FAR_LINES."""
            near = lending[bisect_left(lending, i - NEAR_LINES) : bisect_right(lending, i + NEAR_LINES)]
            above, below = bisect_left(lending, i) - 1, bisect_right(lending, i)
            if above >= 0 and i - lending[above] <= FAR_LINES:
                near.append(lending[above])
            if below < len(lending) and lending[below] - i <= FAR_LINES:
                near.append(lending[below])
            return [day for j in set(near) if j != i for day in lends[j]]

        strict: list[set[int]] = []  # rule 6: the raw sessions each number equals, read at 0 or 2+ decimals
        loose: list[set[int]] = []  # ... or at 1 decimal
        where: list[int] = []
        for start, end, readings in _numbers(text):
            i = bisect_right(starts, start) - 1
            where.append(i)
            codes_strict: set[int] = set()
            codes_loose: set[int] = set()
            for decimals, digits in readings:
                low, high = _interval(digits, decimals)
                # Rule 3: a raw value of a date on this line, close by, or (unless read at 1 decimal)
                # of a date on another line near it.
                if i not in found["3"] and (
                    any(
                        min(abs(start - e), abs(s - end)) <= NEAR_CHARS and self._on(day, low, high)
                        for s, e, day in same_line.get(i, ())
                    )
                    or (decimals != 1 and any(self._on(day, low, high) for day in other_lines(i)))
                ):
                    found["3"].add(i)
                # Rule 6: every raw session this reading equals.
                a, b = bisect_left(self.values, low), bisect_right(self.values, high)
                if a < b:
                    (codes_loose if decimals == 1 else codes_strict).update(self.codes[a:b])
            strict.append(codes_strict)
            loose.append(codes_strict | codes_loose)
        found["6"] = {where[k] for k in self.runs(strict, RUN_MIN, True) | self.runs(loose, RUN_MIN_ONE_DECIMAL, False)}
        return found

    @staticmethod
    def runs(tokens: list[set[int]], minimum: int, unmatched_between: bool) -> set[int]:
        """Rule 6: the positions of ``minimum`` or more numbers that equal consecutive sessions of one series.

        ``tokens`` holds, per number in file order, the raw sessions it equals
        (``series << _ROW_BITS | row``). A run steps through the numbers at a constant stride of 1 to
        ``RUN_MAX_STRIDE`` and through the rows one at a time, up or down. With a stride above 1,
        every number in between must equal a value of the same raw file at one of the two sessions
        it sits between (another column of an OHLC row), or, if ``unmatched_between``, equal no raw
        value at all (a computed column, such as a return, next to the raw ones).

        Why these conditions, measured on the repo against the real store (review B fix): with any
        number allowed in between, made-up numbers formed chance runs at some stride (a synthetic
        control at stride 4; 76 lines of SVG coordinates on the real tree, all 1-decimal runs at a
        stride above 1). With the same-session condition, and numbers that equal nothing allowed
        only in runs read at 2 or more decimals, the real tree has 0. ``RUN_MIN`` 2 instead of 3
        flags 135 lines there; ``RUN_MIN_ONE_DECIMAL`` 4 instead of 5 flags 8.
        """
        file_rows = [{(code >> _FILE_SHIFT) << _ROW_BITS | (code & _ROW_MASK) for code in codes} for codes in tokens]

        def between_ok(x: int, code: int, step: int) -> bool:
            if not tokens[x]:
                return unmatched_between
            here = (code >> _FILE_SHIFT) << _ROW_BITS | (code & _ROW_MASK)
            return here in file_rows[x] or here + step in file_rows[x]

        hits: set[int] = set()
        n = len(tokens)
        for k, codes in enumerate(tokens):
            if not codes:
                continue
            for step in (1, -1):
                shifted = {code + step for code in codes}
                for stride in range(1, RUN_MAX_STRIDE + 1):
                    if k + stride >= n:
                        break
                    following = shifted & tokens[k + stride]
                    if not following:
                        continue
                    if stride > 1:  # cheap bound first: the run's length without the in-between condition
                        bound, j, length = following, k + stride, 2
                        while length < minimum and j + stride < n and bound:
                            bound = {code + step for code in bound} & tokens[j + stride]
                            j, length = j + stride, length + bool(bound)
                        if length < minimum:
                            continue
                    j, chain = k, [k]
                    while following:
                        if stride > 1:
                            following = {c for c in following if all(between_ok(x, c - step, step) for x in range(j + 1, j + stride))}
                            if not following:
                                break
                        j += stride
                        chain.append(j)
                        following = {c + step for c in following} & tokens[j + stride] if j + stride < n else set()
                    if len(chain) >= minimum:
                        hits.update(chain)
        return hits


def repo_text_files(repo: Path) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo, capture_output=True, check=True
    ).stdout.split(b"\0")
    return [repo / name.decode() for name in listed if name and (repo / name.decode()).is_file()]


def scan_repo(repo: Path, index: RawIndex) -> tuple[list[str], int]:
    """``file:line`` of every flagged line in the text files git would commit, and how many files were read."""
    flagged: list[str] = []
    scanned = 0
    for path in repo_text_files(repo):
        blob = path.read_bytes()
        if b"\0" in blob[:8192]:
            continue  # binary
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        flagged.extend(f"{path.relative_to(repo)}:{n}" for n in index.scan_text(text))
    return flagged, scanned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, help="the data root (default: ROOT from VOL_DATA_DIR)")
    parser.add_argument("--repo", type=Path, help="the checkout to scan (default: this one)")
    args = parser.parse_args(argv)
    root = (args.root or config.data_layout().root).expanduser()
    index = RawIndex(root)
    if not index.files:
        print(f"error: no raw snapshots under {root / 'raw'}; nothing to compare with", file=sys.stderr)
        return 2
    repo = (args.repo or config.repo_root()).expanduser()
    flagged, scanned = scan_repo(repo, index)
    for where in flagged:
        print(where)
    if flagged:
        print(f"{len(flagged)} line(s) flagged in {scanned} text file(s) against {len(index.files)} raw snapshot(s)")
        return 1
    print(f"clean: 0 lines flagged in {scanned} text file(s) against {len(index.files)} raw snapshot(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
