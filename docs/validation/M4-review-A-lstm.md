# M4 review A: the LSTM

Adversarial review (Opus) of the eight numbered claims of the first M4 report, 2026-10-29. The reviewer
worked in a scratch copy of the repo (`<session scratchpad>/m4-review-a/copy`) with its own probes
(`p1.py`: P1, P2, P3, P6; `p4_s2.py`: P4; `p5.py`: P5) and mutants (`mutate.py`: M1 frozen-fold look-ahead,
M2 frozen fold refits the scaler, M3 sequence scaler on training + validation), all under
`<session scratchpad>/m4-review-a/`. Synthetic data only; every number below is publishable. The
fixes are in [the M4 walkthrough](../milestones/04-lstm.md#what-review-a-found).

## The reviewer's pre-registered ledger (verbatim)

```
# Pre-registered ledger (mode: single-context sequential fallback). All material, all start REJECTED.
C1 forecast at t bit-exact fn of data<=t, incl frozen folds. Probe P1: run_vix/run_rv21 3 dev folds refit_every=126 (fold1 frozen), truncate at t in fold1 (c1, c1+10); assert_frame_equal atol=0. Falsifier: monkeypatch frozen-only leak (anchor[t+1] when refit False) -> must fail. Also check: does the repo truncation test ever hit a frozen fold? (print refit flags under its settings) + run it on the frozen-only leak mutant.
C2 targets <=c, gap H, poison-val -> identical weights. Probe P2: split arithmetic at c=8626 (max train target idx < min val target idx); poison val with max_epochs=1 vs 6 -> hash equal? Falsifier: gap H-1 in split.
C3 scaler training-only. Probe P3: recompute seq/static/target mean+sd from train idx manually, compare exactly (all columns). Falsifier: mutant seq stats on train+val -> which repo test fails.
C4 yhat=exp(mu+s2/2), s2 per-horizon log val-MSE of seed mean. Probe P4: stored synthetic run 85a849917197: rv21 lstm rows: mean log residual, test log-MSE vs s2, k* = sum(y m)/sum(m^2) vs exp(s2/2), skew. Pass for "s2 inflated" iff mean s2 > test residual variance.
C5 determinism in-proc == pool at hidden 32 batch 64; seed per ADR D5. Probe P5: fit in-proc vs workers=2, compare params dict ==. Falsifier: THREADS=4 in-proc only.
C6 frozen uses training-cutoff scaler/s2; same rows. Probe P6: fold1 forecasts == manual apply of fold0 net/scaler/s2. Mutant M2: frozen fold refits scaler at current cutoff -> does any repo test fail?
C7 no holdout row read. Probe: repo test passes; runner guard raises on holdout date.
C8 O2 grounded; 2.8h derived. Check arithmetic from table; re-time one member at first dev cutoff (vix) once.
Doc honesty: seed-spread vs model gap: compute per-member RMSE spread on stored synthetic member table.
```

## Verdicts, as relayed by the coordinator (who re-verified C1)

| Claim | Verdict | What was measured | Fix |
|---|---|---|---|
| C1 bit-exact truncation, incl. frozen folds | **Test refuted (high)** | `tests/test_lstm.py` truncation tests used `refit_every` 63 with 63-session folds: no frozen fold was ever truncated; mutant M1 (frozen-fold anchor `x_{t+1}`) left 21 passed | `refit_every` 126 in both truncation tests, the frozen flag asserted; M1 and an rv21 twin killed |
| C6 frozen fold uses the training cutoff's network, scaler, `s2` | **Tests refuted (medium)** | mutant M2 left 21 passed: the tests checked metadata and counts only | P6 added as `test_a_frozen_fold_applies_the_trained_network_scaler_and_s2_exactly`; M2 and "frozen fold halves s2" killed |
| C4 (the walkthrough's diagnosis of rv21 mean vs median) | **Refuted (medium)** | `s2` understates the realised log MSE (median `s2`/MSE 0.865 per training cutoff, 15 of 25 below 1); skew +0.178; `mu` biased high (mean log error −0.0206), worst on high forecasts; RMSE-optimal multiplier k* = 1.0046 | Section 7.3 rewritten; mean forecast kept (L6); bias calibration named for M5 (O-M4-7) |
| (unasked) "0.9% RMSE gap inside one seed's wobble" | **Refuted** | per-member RMSE rv21 4.0747–4.1493, VIX h = 21 4.5408–4.5926, all above HAR (4.0238, 4.5133) | Sentence removed (walkthrough and notebook 04); per-member RMSE printed in the notebook |
| C2 validation-only poison → identical weights | Scope | holds at max_epochs = 1; with early stopping the validation values choose the kept epoch | Stated in the walkthrough (section 6) and concept note 11 |
| C8 O2 grounded in measured times | Partly | the first-cutoff VIX row (9 epochs, best 4) was from a pre-D5-seed run; the committed seed gives 7, best 2 | Row marked; re-timed by the builder: 7 epochs, best 2, 4.2 s |
| C3 scaler fitted on the training slice only | SURVIVES | P3 compared all 6 scaler fields exactly with training-slice statistics: `True`; mutant M3 (sequence statistics from train plus validation) killed by 2 tests | — |
| C5 determinism and seeds | SURVIVES | `p5.py 1` (hidden 32, batch 64): in-process vs spawn pool `weights/params/forecasts identical True`; falsifier `p5.py 4` (4 threads) `False`, so the 1-thread pin is load-bearing (torch's `use_deterministic_algorithms` lists CUDA operations only); code matches ADR D5; a fresh process reproduced stored member 1 (seed 686519100: 7 epochs, best 2); both tracks share seeds, disclosed | — |
| C7 holdout untouched | SURVIVES | the repo's ×100 holdout poison test passed in the reviewer's run; `run_lstm` reads only `pre_holdout_inputs`; the builder's mutant was not re-run | — |
| Suite | — | the reviewer's run: `874 passed in 235.41s` | — |

The rows for C3, C5 and C7 were added by the coordinator from the reviewer's final report, which the
builder did not have.

The builder re-ran `p4_s2.py` and confirmed the C4 and per-member numbers, and measured the bias by
`yhat` tercile itself (mean log error +0.020, −0.035, −0.047).

The builder also corrected a label: the refit-cadence question the brief called "O10" is ADR 0001
**O2**; ADR 0001's **O10** (is `lstm_log` in P?) is still open.
