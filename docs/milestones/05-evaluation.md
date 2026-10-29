# Milestone 5: evaluation

**Status:** milestone 5 of 6: every stored model scored on the **dev** period at h = 1, 5, 21 on
`vix_level` and h = 21 on `rv21` (RMSE, MAE, MAPE, QLIKE, bias); Diebold-Mariano tests between every
pair of models, with the long-run variance chosen by a measured size simulation (fixed-b, ADR 0001 O6)
and Holm's correction; Mincer-Zarnowitz regressions with a Hansen-Hodrick covariance (O5), `rv21` in
level and variance space; the losses by regime at the origin; a walk-forward multiplicative bias
calibration of every model as a separate variant (O-M4-7); the `python -m ivf.evaluation report` step;
notebook 05; concept notes 12 and 13. **The holdout is not read** (section 3, D1).

**Purpose.** To explain how forecasts are compared without fooling yourself: why the variance of a
loss differential is the hard part, how a size simulation picks the estimator, why many comparisons
need a correction, what Mincer-Zarnowitz adds to a ranking, and how to test a bias fix without
fitting it on the scored data. Read in order: answer [Predict before you look](#1-predict-before-you-look);
read sections 2 to 7; compare with [the answers](#8-answers-to-the-predict-first-questions); then
[Explain it back](#9-explain-it-back).

**Licensing (ADR 0001 D2).** Numbers below are hand-worked, simulated, or measured on **synthetic**
data. For real data this walkthrough records only the exit code, row counts, test-decision counts and
timings.

**Line numbers are a snapshot**, taken with `grep -n` when this was written.

**Companions.** [ADR 0001](../adr/0001-python-core-versioned-json-app.md) (4.6, 4.8, O5 and O6 carry
the M5 amendments), [data model](../architecture/data-model.md) (`metric`, `dm_test`, `mz_regression`,
`calibration`, amended M5), concept notes [12 Diebold-Mariano](../concepts/12-diebold-mariano-and-forecast-comparison.md)
and [13 Mincer-Zarnowitz and calibration](../concepts/13-mincer-zarnowitz-and-calibration.md),
[milestone 4](04-lstm.md).

## What review A found

An adversarial review (Opus) attacked the first M5 report; the ledger is
[docs/validation/M5-review-A-evaluation.md](../validation/M5-review-A-evaluation.md). Every statement
changed because of it says "(amended M5, review A)".

| Claim or gap | Verdict | Fix |
|---|---|---|
| The primary (fixed-b) DM statistic | **untested (high)**: applying the HLN factor to it passed all 23 tests (orchestrator re-measured: 23 passed) | a hand-worked fixed-b test (n = 11, M = 2); the mutant is killed |
| The fixed-b critical value "within Monte Carlo error" of Kiefer-Vogelsang's cubic (2.2606) | **refuted**: the reviewer's 20 seeds give 2.2368 (sd 0.0108), about 10 SE below; the cubic was quoted from memory | assertion and claim removed; a seeded reproducibility test; "KV comparison not verified" (7.1) |
| "The dev report reads no interim row on or after `holdout_start`" | **refuted as worded**: the frozen `data_end` check reads the full interim tables; the printed line was a fixed string | "no value dated on or after `holdout_start` enters any output", shown by a ×100 poison test; the line prints the last session actually used |
| Rule 6 (calibration verdict) | **untested**: dropping the QLIKE condition, or using the raw p, survived | a test on a constructed frame |
| Regime labels from post-holdout data | **untested**: the spy watched the targets only | the spy covers the labels too |
| `fixed_b_p_value(nan)` | returned the smallest p-value | NaN |
| The DM sign | stated in one docstring only | `dm_test.lower_loss`; "positive `mean_d` means `model_a` is worse" everywhere; 7.4 re-checked |
| Notebook 05's 63-pair calibration warm-up | not the pre-registered 252 | labelled and printed as a synthetic illustration |
| Size at the holdout's n ≈ 583 | DM 0.10, MZ 0.17 under the nulls of 7.1-7.2 | a pre-registered holdout rule (D9): DM and MZ descriptive only, a directional confirmatory check, implemented |

## 1. Predict before you look

1. Two forecasts of the 21-session-ahead VIX are equally good. A DM test with Newey-West weights at
   lag 20 is run on 6,090 origins. About how often does it reject at 5%?
2. One model is better in calm markets and worse in stressed ones, equal on average. Does using the
   "exact" rectangular weights up to lag 20 fix the size?
3. Seven models, 21 pairs, every pair tested at 5%. If all seven are equally good, how many
   "significant" pairs do you expect?
4. A random walk forecast of VIX 21 sessions ahead: will its Mincer-Zarnowitz slope be above or
   below 1?
5. You fit a multiplier `k` on all dev forecasts of a model and score `k·yhat` on the same dev rows.
   What is wrong, and how would you fit `k` instead?

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| DM | `src/ivf/evaluation/dm.py` | per-origin losses; autocovariances (direct and FFT); four long-run variance estimators; the HLN factor; Andrews' bandwidth; the simulated fixed-b null; `dm_test`; Holm; the size simulation (`simulate_differentials`, `rejection_rates`) |
| MZ | `src/ivf/evaluation/mz.py` | OLS of y on (1, yhat), HAC covariances (Hansen-Hodrick, Newey-West, Andrews) with the positive-definite fallback, the joint Wald test, the size simulation (`simulate_mz`) |
| Regimes | `src/ivf/evaluation/regimes.py` | the M1 labels attached at the **origin**; the `metric` rows per (scheme, regime, forecast), with suppression |
| Calibration | `src/ivf/evaluation/calibration.py` | the pre-registered walk-forward multiplier per fold cutoff, the `mean_cal` variant |
| Report step | `src/ivf/evaluation/report.py`, `__main__.py` | `python -m ivf.evaluation [--synthetic] report [--run-id] [--period dev\|holdout] [--print-tables]`: chooses the run, re-checks the data, scores, writes the tables under the data root |
| Size probe | `tools/probes/dm_mz_size.py` | the measurement behind O5 and O6 (section 7.1, 7.2) |
| Notebook | `notebooks/05_evaluation.ipynb` (spec in `tools/build_notebooks.py`) | synthetic only: one DM statistic by hand, a small size simulation, the tables, the holdout filter |
| Tests | `tests/test_evaluation_m5.py` (26; amended M5, review A: was 23), `tests/test_notebooks.py` (+1, and 05 in the byte-equality check) | hand-worked DM, HLN, fallback, Andrews, Holm, MZ (exact linear case, lag 1 by hand, statsmodels agreement, PD fallback); the DM and MZ size at h = 21; calibration by hand and poisoned; regimes; the holdout refusal; the command end to end, with a ×100 poison of every interim value on or after `holdout_start`; *(review A)* fixed-b by hand, rule 6, the holdout check |

**App.** No export is added. The schema has a model-race view (`model_race.json`, data-model 5.3),
but its panels span 252 dev sessions **plus the holdout**, the model runners refuse holdout folds by
design, and no app page reads it yet. It is noted for the polish step, after the M6 holdout look.
The error-surface view (5.4) needs regime DM p-values, which M5 does not compute (D6).

## 3. Decisions

Each default below was taken on 2026-10-29; the owner can override any of them.

| # | Decision | Rejected alternative | Would win if |
|---|---|---|---|
| D1 | **No look at the holdout in M5.** `report` scores `dev` only, builds targets and regimes from the sessions before `holdout_start` (`pre_holdout_inputs`), refuses `--period holdout` on real data before reading anything (exit 2), and runs the holdout path on synthetic data only. The one look (ADR 4.6) covers forecasts and the M6 backtest together. `holdout_looks` stays 0. | Score the holdout forecasts now | M6's rules were already frozen (one look could then cover both) |
| D2 | **DM primary = fixed-b** (Bartlett, M = ceil(0.1·n), simulated fixed-b null), by the pre-registered rule "smallest worst-case size error across the three nulls at the dev size" (O6, section 7.1). The rectangular (+HLN), Newey-West (+HLN) and Andrews versions are stored beside it. | Rectangular + HLN (ADR M0) | Loss differentials were known to be correlated only through the overlap (then rectangular has more power at the same size) |
| D3 | **Benchmarks:** `rw` on each track, HAR (`har_log`, `har_rv`) as the strong baseline. Every pair is tested, so both benchmark views are rows of one matrix. | Benchmark pairs only | — |
| D4 | **Holm** within each (target, horizon, loss, estimator) table of pairs, and separately within the calibration family. Valid under any dependence between the tests; needs only p-values. | Hansen-Lunde-Nason's Model Confidence Set | A block bootstrap of `d_t` had measured size under the state null (it inherits the bandwidth problem) |
| D5 | **MZ primary = Hansen-Hodrick** at lag h−1 with a Newey-West fallback when not positive definite (O5, section 7.2). `rv21` in level and variance space. | Newey-West at lag h−1 (ADR M0); Andrews | MZ were run at the holdout's size, where all three over-reject (then a fixed-b Wald) |
| D6 | **No regime DM tests.** Regime cells carry losses and counts. A regime's origins are not consecutive, so the overlap structure inside a regime is not the one the estimators assume. | DM on the regime's own origins | A masked-series estimator were derived and its size measured |
| D7 | **Calibration rule** (`calibration.py`): `k_c = Σ y·yhat / Σ yhat²` on the model's own pairs with **target ≤ c**, expanding, 252-pair warm-up (k = 1), applied to the fold that starts at c; reported as `mean_cal` / `<model>+cal`; "helps" only if RMSE and QLIKE both fall and the calibration DM rejects after Holm. | One `k` on the whole dev period (look-ahead); separate multipliers per loss | A model's bias were stable enough that a regression `a + b·yhat` beat the one-parameter fix out of sample |
| D9 | **The holdout rule, pre-registered now** (amended M5, review A; O-M5-5). At n ≈ 583 neither test is size-controlled (DM fixed-b 0.1025 under the state null, MZ Hansen-Hodrick 0.166), so on the holdout every DM and MZ p-value is **descriptive only** (`descriptive_only = true`). The **confirmatory** check is directional (`report.holdout_check`): for each (target, h), the dev period's best model by RMSE (and by QLIKE on `rv21`), among the models' own `mean` forecasts, must also have a lower holdout loss than `rw` for the result to count as "held up" (`held_up`; None when `rw` is itself the dev best). No model choice may use holdout results. M6's single look applies it mechanically; tested on synthetic rows. | Holdout DM p-values as confirmatory | A test with measured size near 5% at n ≈ 583 under the state null |
| D8 | **Where the tables go:** `processed/run_id=<id>/evaluation/period=dev/`, beside the forecasts they score, with `report_meta.json` naming the evaluation commit. The newest run with `rw` forecasts is chosen unless `--run-id` is given. | A new run_id for the evaluation | The evaluation changed the forecasts (it does not) |

## 4. Code walkthrough by `file:line`

### 4.1 `src/ivf/evaluation/dm.py`

- `:44-56` the four `METHODS`, `PRIMARY_METHOD = "fixed_b"`, `FIXED_B = 0.1`, Andrews' constant, the
  fixed-b simulation size.
- `:60` `loss_series`: per-origin squared error and QLIKE; their means are `rmse²` and `qlike` exactly.
- `:77` `autocovariances`: divisor n, mean removed; dot products for one series, the FFT for the
  simulation's matrix (tested equal).
- `:96-125` `weights`, `lrv_from_acov`, `lag_window`: rectangular = 1 for j ≤ h−1; Bartlett
  `1 − j/B` for j < B; Newey-West at h−1 is B = h.
- `:128` `hln_factor`; `:133-154` the AR(1) slope and Andrews' bandwidth `1.1447·(α(1)·n)^(1/3)`.
- `:157-195` fixed-b: `M = ceil(b·n)`; `fixed_b_null` simulates `|t|` on 20,000 i.i.d. normal series
  (cached, seeded); `fixed_b_p_value` is `(1 + #exceed)/(1 + R)`, so never 0, and NaN for a NaN
  statistic (review A). No HLN factor on this statistic.
- `:208` `dm_test`: the rectangular fallback to Bartlett when the LRV ≤ 0; HLN with t(n−1) for the
  lag-(h−1) estimators, normal for Andrews, the simulated null for fixed-b.
- `:241` `holm`. `:265-339` the size simulation: `_overlap` (sums of h shocks), the three nulls, and
  `rejection_rates`, vectorised (tested to agree with `dm_test` row by row).

### 4.2 `src/ivf/evaluation/mz.py`, `calibration.py`, `regimes.py`

- `mz.py:56` `long_run_cov`: `Γ0 + Σ w_j(Γj + Γj')`; `:83` `mz_regression`: OLS, `V = Q⁻¹SQ⁻¹/n`, the
  eigenvalue check and fallback, the Wald statistic against (0, 1); `:118` `simulate_mz`.
- `calibration.py:33` `multipliers`: pairs sorted by **target date**, cumulative sums, one
  `searchsorted(..., side="right")` per cutoff (target ≤ c); `:51` `calibrate`.
- `regimes.py:27` `attach` maps the label at `origin_date`; `:47` `metric_rows`.

### 4.3 `src/ivf/evaluation/report.py`

- `:63` `BENCHMARKS`; `:75` `scored_rows` (dev: origin **and** target before `holdout_start`);
  `:91` `attach_targets` (y at the **target** date).
- `:113` `dm_table`: pairs with `model_a < model_b` alphabetically and `d = L_a − L_b`, so a **positive
  `mean_d` means `model_a` is worse**; `lower_loss` names the better model; every method; Holm per family.
- `:149` `mz_table`; `:173` `calibration_summary` (rule 6); `:188-213` `HOLDOUT_LOSSES` and
  `holdout_check` (D9); `:216` `evaluate`: calibrate on every row, then score the period, same-rows
  check, tables, and on the holdout the confirmatory check with the model chosen on dev rows.
- `:271` `run_report`: the holdout refusal comes before the protocol or any file is read; the data
  check; `choose_run`; `pre_holdout_inputs` for dev; writes through `repo_guard.write_file`.

## 5. Data flow

```
processed/run_id=<id>/forecast.parquet ─┐
interim (rows < holdout_start only) ────┼─> target_value (y, v), regime_label (at the origin)
                                         │
                                         └─> attach y at target_date ─> calibrate (k per cutoff, targets <= c)
                                                 ─> scored dev rows (origin, target < holdout_start)
                                                 ─> metric (all + regimes; mean, median, mean_cal)
                                                 ─> dm_test (pairs + calibration; 4 estimators; Holm)
                                                 ─> mz_regression (3 covariances; rv21 in 2 spaces)
                                                 ─> calibration, calibration_summary
                                         ─> processed/run_id=<id>/evaluation/period=dev/*.parquet + report_meta.json
```

The architecture diagram's `ivf.evaluation` node and results store were updated (amended M5).

## 6. Pitfalls and their guards

| Pitfall | Guard |
|---|---|
| A test that is "exact" for the overlap but not for state dependence | the size simulation includes the `state` null; `test_dm_size_at_h21_with_overlapping_errors` asserts the primary passes it and the rectangular estimator does not |
| statsmodels' `diebold_mariano_test` (Bartlett, `max(h−1, n^(1/3))` lags) | not used; own estimators, MZ checked against statsmodels' HAC to 1e-10 |
| The look-ahead in calibration: pairs selected by **origin** ≤ c include targets after c | pairs by target date; a poisoned-future test; the "by origin" mutant is killed |
| Scoring a dev origin whose target is in the holdout | `scored_rows` keeps origin and target < `holdout_start`; the "drop the target condition" mutant is killed |
| A holdout value reaching a dev output | `pre_holdout_inputs`; the command test spies on `build_target_values` and `build_regime_labels` and re-runs on interim values ×100 on and after `holdout_start` (every table identical); both leak mutants killed |
| A fixed-b p-value of exactly 0 | `(1 + #exceed)/(1 + R)`; tested |
| A calibration variant silently replacing the model | a separate `forecast`/model id; tested that `yhat` is untouched |

## 7. Measured results

### 7.1 DM size (O6)

`uv run python tools/probes/dm_mz_size.py 2000 1000`, 2026-10-29, seed 20261028, h = 21, nominal 5%,
2,000 simulated samples per cell (MC SE 0.0049), 6.0 s:

| Null | n | Rectangular + HLN | Newey-West (h−1) + HLN | Andrews | **Fixed-b** |
|---|---:|---:|---:|---:|---:|
| `ma`: linear MA(20) differential | 583 | 0.0665 | 0.1195 | 0.1155 | 0.0705 |
| `ma` | 6,090 | 0.0480 | 0.1090 | 0.0630 | 0.0485 |
| `sq_error`: two equally good forecasts, overlapping errors | 583 | 0.0635 | 0.0940 | 0.0970 | 0.0585 |
| `sq_error` | 6,090 | 0.0560 | 0.0915 | 0.0685 | 0.0545 |
| `state`: one model's accuracy follows a persistent state | 583 | 0.1135 | 0.1745 | 0.1545 | 0.1025 |
| `state` | 6,090 | 0.1060 | 0.1830 | 0.0920 | **0.0560** |

- **The requested check** (two equally good forecasts with overlapping errors, h = 21): the primary
  fixed-b test rejects 0.0545 at the dev size (0.0585 at 583); Newey-West at lag h−1 rejects 0.0915,
  so the M0 kernel was badly oversized and is not used.
- **Rule (pre-registered before the run):** smallest worst-case size error across the three nulls at
  n = 6,090. Fixed-b 0.0560, Andrews 0.0920, rectangular 0.1060, Newey-West 0.1830.
- The fixed-b critical value (|t| 95% quantile, 20,000 draws, n = 1,000, seed 20261028) is 2.2458.
  Across 20 seeds (20261028..20261047) its mean is 2.2290, sd 0.0195 (review A's own 20 seeds: 2.2368,
  sd 0.0108; not reconciled). **The comparison with Kiefer-Vogelsang's tables is not verified** (amended
  M5, review A: the first version claimed agreement with a cubic quoted from memory).
- **Still oversized at the holdout's size** under the `state` null (0.1025). Recorded for the M6 look.
- The test suite repeats the `sq_error` and `state` cells with 1,000 samples (0.9 s).

### 7.2 MZ size (O5)

Same command, 1,000 samples per cell (MC SE 0.0069): a rational forecast of an AR(1) level (φ = 0.98,
so α = 0 and β = 1 hold), joint Wald at 5%:

| n | Hansen-Hodrick | Newey-West (h−1) | Andrews |
|---:|---:|---:|---:|
| 583 | 0.1660 | 0.2600 | 0.2660 |
| 6,090 | **0.0650** | 0.1430 | 0.0920 |

Hansen-Hodrick is primary for the dev evaluation. At 583 every covariance is about three times too
liberal: a persistent regressor with overlapping errors needs more than 583 origins.

### 7.3 The real run (counts and timings only, D2)

2026-10-29, uncommitted M5 code on `a481168`, `/usr/bin/time`; re-run after the review A fixes (same rows,
wall 3.67 s, peak RSS 887,596 KB):

| Step | Result |
|---|---|
| `uv run python -m ivf.evaluation report` | **exit 0**; wall 6.49 s (4.2 s inside the step); peak RSS 847,144 KB |
| Source run | the newest with forecasts; 7 models on `vix_level` and 7 on `rv21` (every milestone ≤ 4 model) |
| Holdout | no value dated on or after `holdout_start` enters any output: targets and regimes from the 8,689 sessions before it (printed), shown by the ×100 poison test on synthetic data; the frozen `data_end` check reads the full interim tables (amended M5, review A: was "no interim row … is read"). 0 forecast rows with an origin on or after it; `holdout_looks 0` |
| Scored origins | vix_level h = 1 / 5 / 21: 6,110 / 6,106 / 6,090; rv21: 6,090 (M4's counts) |
| Rows written | metric 504, dm_test 896, mz_regression 210, calibration 2,716, calibration_summary 28 |
| Fallbacks | 0 `bartlett_fallback` rows in `dm_test` and in `mz_regression`; 133 of 2,716 fold multipliers in warm-up |
| Test decisions (counts) | pairwise DM tests Holm-significant at 5%: fixed-b 15 of 168, rectangular 75, Newey-West 78, Andrews 62. MZ (raw forecasts) rejected at 5%: Hansen-Hodrick 20 of 35. Calibration "helps" (rule 6): 1 of 28 |

The bandwidth choice changes most pairwise conclusions on real data (62 of 168 decisions differ
between fixed-b and rectangular). That is the sensitivity ADR O6 asked to report; the values stay local.

### 7.4 Synthetic data (publishable)

`uv run python -m ivf.evaluation --synthetic report --print-tables` on the M4 synthetic run (4.4 s;
same row counts as 7.3). At h = 21, squared error, the primary (fixed-b) test; each cell is the model,
then who has the lower mean loss (`lower_loss`), then the Holm-adjusted p (re-checked against
`dm_test` after review A):

| Track | Against `rw` | Against HAR |
|---|---|---|
| vix_level | `har_log`, `arima_log`, `arima_garch_log`: the model, 0.0010; `lstm_log`: the model, 0.0018; `hist_mean`: `rw`, 0.31; `hist_mean_252`: `rw`, 0.15 | `arima_log`, `arima_garch_log`: the model, 0.71; `lstm_log`: `har_log`, 0.71; `hist_mean`: `har_log`, 0.056; `hist_mean_252`: `har_log`, 0.053 |
| rv21 | `garch`, `gjr`: the model, 0.0021; `egarch`: the model, 0.0084; `har_rv`: the model, 0.0121; `lstm_log`: the model, 0.0015; `vix_implied`: `rw`, 0.0010 | `garch`, `gjr`, `egarch`: the model, 1.00; `lstm_log`: `har_rv`, 1.00; `vix_implied`: `har_rv`, 0.0010 |

- With the rectangular estimator `hist_mean` vs `rw` would be "significant" (0.027) and `har_log` vs
  `hist_mean` 0.0000 instead of 0.056: of 168 pairwise decisions, 29 differ between the two estimators.
- **MZ (Hansen-Hodrick, h = 21):** `har_log` α 0.23, β 0.99, p 0.93; `rw` β 0.65, p < 0.001 (the random
  walk ignores mean reversion); on `rv21`, `garch` fails in level space (β 0.89, p 0.001) and passes in
  variance space (p 0.33): it forecasts `E[v]`, and its vol number is the RMS vol (concept 13).
- **Calibration (O-M4-7):** helps **0 of 28** under rule 6. RMSE fell for 9, QLIKE for 2. The LSTM's
  rv21 RMSE rose (4.0589 → 4.0823): a multiplier learned from past errors does not remove a bias that
  depends on the level. `vix_implied` on rv21: RMSE 9.02 → 6.71 but QLIKE 0.40 → 0.62, so not "helps".

### 7.5 Mutation checks

Every new test was run against deliberate bugs in a scratch copy (`<session scratchpad>/m5/mut`,
`tests/test_evaluation_m5.py -x`, then the notebook test). First round: **43 of 43 killed** (36, then
the 4 survivors led to stronger tests; 3 notebook mutants). After review A, the whole list was re-run
on the fixed code with review A's 5 mutants and 10 more: **55 of 55 killed** in the test file (one,
"a calibrated variant may be the dev best", survived first and was killed by a stronger
`test_holdout_check_by_hand`), plus the notebook's warm-up mutant: **59 of 59** in all.

| Mutant | Killed by |
|---|---|
| HLN `n+1−h`; normal p instead of t(n−1); rectangular weights → Bartlett; LRV without the 2; autocovariance divisor n−j | `test_hln_factor_by_hand`, `test_dm_rectangular_by_hand` |
| FFT autocovariance without the mean removed | `test_fft_autocovariances_equal_the_direct_sums` |
| no rectangular fallback; Newey-West bandwidth h−1 | `test_dm_rectangular_falls_back_…` |
| Andrews' QS constant; α(1) without (1+ρ)² | `test_andrews_bandwidth_by_hand` |
| fixed-b lags floor; fixed-b null rectangular; *(round 2)* p-value without +1 | `test_fixed_b_null_is_seeded_…` (renamed in review A) |
| primary = rectangular | `test_dm_size_at_h21_with_overlapping_errors` |
| *(round 2)* simulated errors summing h−1 shocks | `test_the_simulated_ma_errors_are_sums_of_h_consecutive_shocks` |
| Holm without the step-down max; multiplier m−k−1; NaN counted in m | `test_holm_by_hand_and_against_statsmodels` |
| QLIKE ratio inverted | `test_loss_series_average_to_the_metric_losses` |
| MZ `2Γj`; Hansen-Hodrick → Bartlett | `test_mz_hansen_hodrick_lag_one_by_hand` |
| MZ Wald against (0, 0); V not divided by n | `test_mz_exact_linear_case_white_covariance_by_hand` |
| MZ no PD fallback | `test_mz_hansen_hodrick_falls_back_when_not_positive_definite` |
| *(round 2)* MZ primary = Newey-West | `test_mz_size_with_a_rational_forecast_at_h21` |
| calibration target < c; pairs by origin date; k = Σy/Σyhat; warm-up ≤; v_hat not squared; calibration replaces yhat | `test_calibration_multiplier_by_hand` |
| regime at the target date; no suppression; dev keeps targets ≥ holdout_start; y at the origin date | `test_regime_cells_label_at_the_origin_and_suppress_small_cells` |
| `mean_cal` not reported; d = L_b − L_a; Holm family across estimators | `test_evaluate_dev_and_the_synthetic_holdout_path` |
| holdout refusal removed | `test_report_refuses_the_real_holdout_before_reading_anything` |
| *(round 2)* dev reads the holdout inputs | `test_report_command_on_the_synthetic_pipeline` |
| notebook: HLN `n+1−h`; 21 lags instead of 20; `holdout_looks` printed as 1; *(review A)* warm-up 252 printed as 63 | `test_m5_notebook_runs_top_to_bottom_on_synthetic_data` |
| *(review A)* HLN applied to the fixed-b statistic | `test_dm_fixed_b_statistic_by_hand` |
| *(review A)* fixed-b p one-sided; NaN guard removed | `test_fixed_b_null_is_seeded_and_the_p_value_is_monotone` |
| *(review A)* rule 6 without the QLIKE condition; with the raw p instead of Holm | `test_calibration_verdict_follows_rule_6` |
| *(review A)* regime labels built from post-holdout data | `test_report_command_on_the_synthetic_pipeline` (the label spy) |
| *(review A)* `lower_loss` inverted; `descriptive_only` never set (DM, MZ); holdout check with the best chosen on holdout rows | `test_evaluate_dev_and_the_synthetic_holdout_path` |
| *(review A)* holdout check: best = highest loss; compared with dev `rw`; QLIKE on every target; `rw`-best not None; a calibrated variant may be the dev best | `test_holdout_check_by_hand` |

### 7.6 Gates

Run on 2026-10-29 after the review A fixes, from the repo root, uncommitted M5 code on `a481168`; the
printed last lines:

```
$ uv run pytest -q
903 passed in 111.86s (0:01:51)
$ uv run python tools/build_notebooks.py --check
ok: 6 notebook(s) in /home/rajib2k5/dev/implied-vol-forecasting/notebooks match their specs
$ uv run python tools/export_app_data.py --mode synthetic --check
ok: 1 file(s) in /home/rajib2k5/dev/implied-vol-forecasting/app/public/data/synthetic match a fresh synthetic export
$ uv lock --check
Resolved 142 packages in 0.79ms
$ uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out --check --no-png
ok: 5 spec(s) up to date in docs/architecture/out
$ uv run python tools/licensing/scan_real_rows.py
clean: 0 lines flagged in 254 text file(s) against 7 raw snapshot(s)
```

`tests/test_evaluation_m5.py` alone: 26 passed in about 6 s. The suite grew from 875 (M4) to 903: 26
new tests here (23, then 3 after review A) and 2 in `tests/test_notebooks.py` (notebook 05 runs; 05 in
the byte-equality check). `git status --porcelain --ignored` lists only this milestone's files (none
under the data root) and the usual ignored folders (`.venv/`, `.pytest_cache/`, `.learning/`,
`app/node_modules/`, `app/dist/`, `app/coverage/`, `graphify-out/`, `__pycache__/`).

## 8. Answers to the predict-first questions

1. About 9%, not 5% (7.1: 0.0915). Bartlett weights `1 − j/21` count each overlapping lag only
   partly, so the variance is too small and the statistic too large.
2. No. Rectangular is exact only when the dependence stops at lag 20; a persistent state makes it
   reject 10.6% at n = 6,090. Fixed-b (many lags, honest critical values) gets 5.6%.
3. About one (21 × 0.05 = 1.05). Holm's correction keeps the chance of any false "significant" pair at
   5%.
4. Below 1. VIX mean-reverts; the random walk forecasts today's level, so after a high day it is too
   high and after a low day too low: outcomes move less than forecasts (synthetic: β = 0.65).
5. `k` then uses the outcomes it is scored on: in-sample, so it always looks at least as good. Fit
   `k_c` at each cutoff on pairs whose **target** is dated ≤ c, and apply it to the next fold.

## 9. Explain it back

1. Why is the long-run variance, not the mean, the hard part of a DM test at h = 21?
2. What did the size simulation measure, and what rule chose the estimator?
3. Why does fixed-b have less power than the rectangular estimator, and when would you accept that?
4. What does Holm control, and what is one "family" here?
5. A model passes MZ in variance space and fails in level space. What does that tell you?
6. Why does the calibration multiplier select pairs by target date, and what would selecting by
   origin date leak?
7. Why does M5 not look at the holdout, and what will the look in M6 have to report about test size?

## 10. Decisions waiting on you

| # | Question | Default (taken 2026-10-29, owner can override) |
|---|---|---|
| O-M5-1 (= ADR 0001 O6) | Primary DM estimator | Fixed-b, b = 0.1 (measured, 7.1); block bootstrap not measured |
| O-M5-2 (= ADR 0001 O5) | Primary MZ covariance | Hansen-Hodrick (measured, 7.2) |
| O-M5-3 | Multiple-comparison control | Holm per table; MCS not implemented |
| O-M5-4 | When to look at the holdout | Once, at the end of M6, forecasts and backtest together; M5 tests the path on synthetic data only |
| O-M5-5 | Size at the holdout's n = 583 (DM state null 0.10, MZ 0.17) | Pre-registered (D9, amended M5, review A): holdout DM and MZ descriptive only; confirmatory = the dev-best model (RMSE; and QLIKE on rv21) beats `rw` on the holdout; no model choice uses holdout results. Implemented in `holdout_check`, tested on synthetic rows |
| O-M5-6 | Regime DM tests | Not computed (D6) |
| O-M5-7 (= O-M4-7) | Bias calibration | Rule 6 pre-registered; the variant is reported, never adopted silently; synthetic: helps 0 of 28 |
| O-M5-8 | The model-race export | Deferred to the polish step (it spans the holdout) |
