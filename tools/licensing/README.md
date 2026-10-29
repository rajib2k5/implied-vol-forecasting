# Licensing probes (ADR 0001 D2)

These scripts test what the publication set P (ADR 0001 D2) can reveal. **They use simulated data
only**: every series is drawn by `p_set.sim()` from a seeded generator, no file is read, and no real
market value is embedded. They are committed so that R1's "rerun the P probes" is reproducible.

| Script | What it shows | Run from the repo root |
|---|---|---|
| `p_set.py` | P's inputs exclude every known attack; P is invariant to rescaling VIX; P is sensitive to single days | `uv run python tools/licensing/p_set.py` |
| `p_sens.py` | Which P entries flip when one close moves by ±0.25 | `uv run python tools/licensing/p_sens.py` |
| `p_falsify.py` | The rescale test does detect level-unit statistics (falsifier) | `uv run python tools/licensing/p_falsify.py` |
| `p_tie.py`, `p_tie_seeds.py` | Review E: scale-tied rows plus public SPX reveal level statistics; `*_cal` stays invariant; LSTM inputs matter | `uv run python tools/licensing/p_tie.py` |

## `scan_real_rows.py`: still not caught (M1 final check)

The final M1 check planted synthetic stand-ins in three shapes the scanner did not know. One is now
caught: a decimal-comma CSV (`DD.MM.YYYY;16,94`); the day-first dotted date was already read, the
decimal comma is now a number too (not with exactly 3 digits after the comma, which is read as a
thousands group). The scanner still prints no value, only `file:line` and counts.

Still not caught, stated rather than closed:

- **Excel serial dates** (`37214,19.16`: day 37214 counted from 1899-12-30). Any 5-digit integer
  would have to be read as a date, and repos are full of them.
- **Every k-th session** (weekly or monthly closes, k > 1). Rule 6 needs consecutive sessions;
  a stride over the raw rows was not added.
- Everything the scanner's docstring lists after "Still not caught": values transformed beyond
  rounding and /100, a lone value with no date near it and no run, a 1-decimal value whose date is
  on another line, short runs, 2-digit years and unlisted spellings, scientific notation, numbers
  in words, and ignored, binary or non-UTF-8 files.
