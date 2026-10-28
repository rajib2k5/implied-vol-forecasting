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
