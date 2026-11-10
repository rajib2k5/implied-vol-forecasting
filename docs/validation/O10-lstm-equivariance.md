# O10: is `lstm_log` scale-equivariant in the VIX level?

ADR 0001 O10 asks whether `lstm_log` rows can join the publication set P. A row can join P only if it
does not move when the vendor's VIX level V is replaced by c·V, with the S&P 500 returns held fixed.
If the row does not move, the committed number cannot reveal the vendor's scale. The toy network in
`tools/licensing/p_tie.py` (c4) was the M0 stand-in. This note tests the real M4 code.

**Route:** router row 4 (ML code with no test oracle): a Channel B probe built on a metamorphic
relation (V → cV) with falsifiers that must fire, then re-measurement by the coordinator. Channel A is not
run because the claim names no library semantics. Adversarial reading of the maths is in row 4's "do not run" column.

## Probe

`uv run python tools/licensing/p_tie_lstm.py` (simulated data only: `p_set.sim(20261028)`, no file
read). It calls `lstm.LstmLog` (VIX track, h = 1, 5, 21) and `lstm.LstmRv` (rv21 track, h = 21, ln VIX
as a static input) through `.fit` and `.forecast`, as `walkforward` does. It compares four entries per
target and horizon: RMSE ratio vs rw, QLIKE ratio vs rw, MAPE ratio vs rw and absolute MAPE, plus the raw
forecasts (divided by c on the VIX track).

Config (cheap, stated; not the production one): 3 cutoffs 21 sessions apart, each a full refit,
2 members, hidden 8, `max_epochs` 4, window 22, `val_sessions` 100, 1 thread.

## Results

| Run | Measured by | c = 0.8: entries changed at 6 s.f. | c = 1.25 | Largest entry difference |
|---|---|---|---|---|
| cheap, float32 | coordinator (re-run) and probe author | 0 of 16 | 0 of 16 | 4.4e-16 |
| float64 (`--float64`, a monkeypatch) | probe author | 0 of 16 | 0 of 16 | about 4e-16 |
| heavier float32 (`--epochs 30 --hidden 16 --folds 4`) | probe author | 0 of 16 | 0 of 16 | 1.7e-15 |

Raw forecasts (coordinator's re-run, cheap float32): rv21 track identical (difference 0); VIX track
yhat(cV)/c within 2.1e-14 absolute, 9.9e-16 relative.

**Falsifiers** (the coordinator's re-run, cheap float32; the probe must see these):

| Mutant | Entries changed at 6 s.f. | at 3 s.f. (published precision) | Largest entry difference |
|---|---|---|---|
| `raw_level`: extra input V/100, not z-scored | 16 of 16 at each c | 1 to 2 of 16 | 1.1e-3 |
| `flag20`: extra input 1[V > 20] | 16 of 16 at each c | most entries (for example 15 of 16 at c = 0.8 in the author's run) | 5.6e-2 |

Why it holds: the network sees ln V. The scaler is fitted on the training rows only, so it removes the
ln c shift. The target is a log change, and the forecast is the anchor plus the prediction, so yhat scales
by exactly c, and every ratio against rw is unchanged.

## Verdict

`lstm_log` is V-equivariant on both tracks to about 1e-15. Under D2's row rule (its "Which rows" paragraph) its unitless rows qualify for
P, with the same rounding as the other models.

Not covered:
- The production config (hidden 32, 5 seeds, 50 epochs, `val_sessions` 252) was not run. The argument
  above does not depend on the config.
- The frozen-fit path (`refit_every: 252`) was not run.
- DM and MZ entries were not run. Both are built from the same scale-free losses and forecasts.
- Only one simulated seed was run.
