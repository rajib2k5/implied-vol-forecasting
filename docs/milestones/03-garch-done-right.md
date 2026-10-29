# Milestone 3: GARCH done right

**Status:** milestone 3 of 6: GARCH(1,1), GJR-GARCH and EGARCH on S&P 500 returns forecasting
`rv21`, compared with VIX as a forecast of the same quantity; the `rv21` baselines deferred from M2
(`rw`, `har_rv`, `vix_implied`); ARIMA-GARCH (an AR with a constant on ln VIX, GARCH errors; amended M3, review A: first built on its daily changes) as a `vix_level` model; rule L2
for the window target `rv21` (O-M2-7); the `python -m ivf.models garch` step; notebook 03; concept
notes 09 and 10. Dev period only: **no row on or after `holdout_start` is read**. No real-data number
is published.

**Purpose.** To explain how a daily variance model becomes a forecast of 21-session realised
volatility with the right units, why the GARCH family is fitted on returns and never on the VIX level,
and how a window target is kept from leaking into training. Read in order: answer
[Predict before you look](#1-predict-before-you-look); read sections 2 to 7; compare with
[the answers](#8-answers-to-the-predict-first-questions); then [Explain it back](#9-explain-it-back).

**Licensing (ADR 0001 D2).** Numbers below are hand-worked or measured on **synthetic** data. For
real data this walkthrough records only exit codes, row counts, timings and convergence, fallback and
model-choice counts; the losses are measured locally and not published.

**Line numbers are a snapshot**, resolved against the file text by a script when this was written.

**Amended M3, review A.** An adversarial review attacked the eight claims of the first M3 report. It
refuted the ARIMA-GARCH design (d = 1), partly refuted the EGARCH claim (unscaled residuals; the
"noise" wording), and found three mutants that survived silently. [What review A found](#what-review-a-found)
names each with its fix; every statement changed because of it says "(amended M3, review A)". The
ledger is [docs/validation/M3-review-A-garch.md](../validation/M3-review-A-garch.md).

**Companions.** [ADR 0001](../adr/0001-python-core-versioned-json-app.md) (4.4 and 4.5 carry the M3
amendments), [data model](../architecture/data-model.md) (`fit`, `forecast` amended M3), concept notes
[09 the GARCH family and variance aggregation](../concepts/09-garch-family-and-variance-aggregation.md)
and [10 GARCH on IV is a category error](../concepts/10-garch-on-iv-is-a-category-error.md),
[milestone 2](02-baselines.md).

## What review A found

| Claim or area | Verdict | Fix |
|---|---|---|
| 1. Poisoning after the cutoff changes no fit and no forecast at c | not refuted (two folds, huge and NaN poison) | — |
| 2. `har_rv` uses τ ≤ c − 21, and the restriction binds | not refuted | — |
| 3. `G = E_t[v]` in rv21 units | not refuted | — |
| 4. Our recursion equals `arch`'s to 1e-9 | not refuted (≤ 1.3e-15 measured) | — |
| 5. EGARCH product formula; under t "a simulated mean is noise" | **partly refuted**. The formula holds (3.3e-15 against brute force), but the t-MGF diverges only through \|z\| of a few hundred (1.08 up to 200, 211.5 up to 400), so a simulated mean is a stable truncated value; and the residuals were not rescaled (mean z² 0.9896 under t) | Residuals rescaled to unit RMS before `egarch_mgf`, `z_rms` recorded; wording fixed in `garch.py`, D4, concept 09, ADR |
| 6. Forecasts at t use data ≤ t, including `arima_garch_log` | not refuted | — |
| 7. The holdout is unread | not refuted | — |
| 8. The synthetic numbers, and the ARIMA-GARCH design | **design refuted**: d = 1 was worse than the random walk at every horizon, d = 0 better (O-M3-4) | `arima_garch_log` is d = 0 (AR with a constant on `100·ln VIX`, GARCH errors); d = 1 an option. Default taken 2026-10-29, owner can override |
| Mutants | **3 survived silently**: EGARCH M from raw percent residuals; the AR residual shifted by one (the `FILTER_RTOL` check refused every p > 0 candidate without a trace, and on the homoskedastic synthetic VIX the shift barely moved the variances); `forecastable` with 21 returns | `test_egarch_mgf_uses_the_standardised_residuals_rescaled_to_unit_rms` (arch's `std_resid` as oracle), `test_every_candidate_is_usable_on_clean_synthetic_data` (AR data with GARCH shocks) and a `candidates_refused` column in the `fit` table, `test_an_origin_needs_the_22_returns_t_minus_21_to_t_and_vix_at_t` |

## What the first real run found

The first real run (2026-10-29) **exited 1**: `arima_garch_log`'s first fit raised "too few changes of
ln IV". Its sample was "the gap-free run of closes ending at the cutoff". Real VIX has isolated
missing closes in burn-in (allowed by G3), and the run after the last of them was too short to fit at
the first cutoff (the count and position stay local, as in M2). **How it went undetected:** the synthetic market has no gaps, and no M3 test gave
a GARCH model a missing value; the rv21 track passed because real `^GSPC` has no missing close. Fix:
missing values are skipped (decision D7), with two regression tests
(`test_a_missing_return_is_skipped_in_the_fit_and_in_the_forecast`,
`test_arima_garch_fits_through_a_missing_close_shortly_before_the_cutoff`), and the old rule is a
mutant that those tests kill (section 7.3).

## 1. Predict before you look

1. A GARCH(1,1) with alpha + beta = 0.98 forecasts tomorrow's variance at twice its long-run level.
   Is its 21-session `G` closer to twice or to once the long-run value?
2. VIX is a forecast of the next 30 days' volatility. As a forecast of `rv21`, should it be biased
   high, low, or not at all?
3. `har_rv` is fitted at cutoff c on pairs (origin τ, `rv21` at τ + 21). What is the last τ it may use?
4. On synthetic VIX (ln VIX an AR(1) with constant shock variance), what alpha + beta does a GARCH(1,1)
   fitted to the VIX **level** report?
5. ARIMA-GARCH was first built on the daily changes of ln VIX (d = 1). On mean-reverting data, would it
   beat the random walk at h = 21? And with d = 0?

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| GARCH family | `src/ivf/models/garch.py` | `arch`'s recursion re-implemented for forecasting, the multi-step paths, the 21-session aggregate, the innovation rule, `GarchRv`, `ArimaGarch`, the category-error demo |
| rv21 track | `src/ivf/models/rv21.py` | `Rv21Inputs`, the target dated at its window end, `training_origins` (L2), `rw`, `har_rv`, `vix_implied` |
| Runner | `src/ivf/evaluation/walkforward.py` (`run_rv21`) | folds, fits on inputs up to c, the two L2 assertions, same rows, `var_method` and `v_hat` from the model |
| Registry | `configs/models.yaml`, `src/ivf/models/registry.py` | six new `model_def` rows, `rw` gains `rv21`, the `garch` settings block, `models_for` (the expected model list per target) |
| Command | `src/ivf/models/cli.py` (`run_garch`) | `python -m ivf.models [--synthetic] garch`: both tracks, counts only unless `--print-metrics`, tables merged into the run |
| Notebook | `notebooks/03_garch.ipynb` (spec in `tools/build_notebooks.py`) | the aggregate by hand, the rv21 track, the synthetic truth floor, ARIMA-GARCH, the category error |
| Tests | `tests/test_garch.py` (40), `tests/test_rv21.py` (18), `tests/test_notebooks.py` (+3) (amended M3, review A: 29 and 17 before it) | hand-worked numbers, `arch` as an oracle, L2, the holdout, truncation, same rows, the command |

**App.** No export is added: forecasts reach the app with the holdout run (M2's reason, unchanged).

## 3. Decisions

Each default below was taken on 2026-10-29; the owner can override any of them.

| # | Decision | Rejected | Would win if |
|---|---|---|---|
| D1 | **`arch` 8.0.0 estimates; this project's copy of its recursion forecasts.** The copy reproduces `arch`'s conditional variances to `FILTER_RTOL` = 1e-7 at every fit (else the candidate is refused); measured ≤ 1e-15 in tests | `arch`'s `fix(...).forecast` inside the fold: its variance bounds use the whole sample's variance and maximum, so a forecast at t is not a pure function of data ≤ t | `arch` exposed a filter without sample-wide bounds |
| D2 | **Zero mean.** `rv21` removes no mean, so `G_t = (252/21)·Σ σ²_{t+k|t}/100²` is `E_t[v]` exactly | Constant mean: forecasts `E[(r−μ)²]`, short of the target by `μ²` a day, and the GJR indicator would be on `r − μ` | The target removed the mean |
| D3 | **Innovations: normal and Student-t fitted at every cutoff; the lower AIC wins; a tie keeps normal.** Pre-registered in `configs/models.yaml` before any M3 forecast was scored | Always t (assumes the answer); always normal QMLE (consistent for the variance parameters but blind to tails); a likelihood-ratio test (ν = ∞ is on the boundary, so the χ² reference is wrong); BIC (aims at the true model, not the best forecast; M2 used AIC) | The owner prefers a fixed distribution for comparability across folds |
| D4 | **EGARCH 21-step expectation as the exact mean over the training standardised residuals, rescaled to unit RMS** (`egarch_mgf`; amended M3, review A: they were not rescaled, mean(z²) was 0.9896 under t) | `arch`'s simulation under the fitted t: the true expectation is infinite, but only through \|z\| of a few hundred, so the simulated mean is a stable truncated value with Monte Carlo noise and seed dependence (review A; the first version called it "noise"); `arch`'s bootstrap: Monte Carlo noise around the same quantity; the normal closed form: thinner tails than the residuals. The empirical mean is exact for the rescaled residuals, not for the fitted distribution | Innovations were normal by rule (then the normal closed form is exact) |
| D5 | **GJR's factor `α + γ/2 + β`**, the model's own `E[z² 1(z<0)] = 1/2` for a symmetric z, as `arch` | The empirical share from the residuals | A skewed distribution were fitted |
| D6 | **`har_rv` in variance space**: windows 1, 5, 22 of `252 r²`, OLS, direct at h = 21, floored at the smallest training target, `var_method = native` | Log space (`ln r²` is −∞ on a zero return); vol space (not `E_t[v]`); no floor (a non-positive forecast is a missing row, an error) | Intraday realised variance were available |
| D7 | **Missing values are skipped**, never filled: the recursion steps from the last observed shock to the next (amended M3, first real run) | The gap-free run ending at the cutoff (too short at the first real cutoff; the run failed); the longest gap-free segment (discards most of burn-in at early folds, and the sample jumps when the longest segment changes); a likelihood with missing-shock updates (`arch` cannot) | Gaps were frequent rather than a few isolated burn-in closes |
| D8 | **`arima_garch_log` = AR(p) with a constant on `100·ln VIX` (d = 0) + GARCH(1,1) on its innovations**, p ∈ {0,1,2} and the distribution chosen together by AIC on a common sample (first 2 values held back), a non-stationary AR refused; the lognormal mean scored, the median stored (amended M3, review A) | d = 1, the AR on `100·Δ ln VIX` with no constant (the first version: worse than the random walk at every horizon on the synthetic market, M2's `arima_log` lesson; kept as the option `garch.d: 1`); MA terms (`arch` has AR means only) | ln VIX had a unit root, which the random walk already covers |
| D9 | **One `garch` step runs both tracks**, the VIX track with M2's five baselines too, so every `vix_level` model is scored on the same rows in one run; `fit`, `forecast`, `metric` are **merged** by (model, target) into the run's tables | Overwriting (M2's behaviour: the second step deletes the first's rows); a separate run_id per step (the store's key is the run) | — |
| D10 | **L2 for the window target** (O-M2-7): inputs up to c only; a supervised pair needs `τ ≤ c − 21`; the runner asserts `train_last_origin + 21 ≤ c` | Storing `forward_rv21` on the origin's row and trusting the slice (the M2 trap) | — |
| D11 | **ARIMA-GARCH on VIX only** (the protocol target) | All six indices like M2's `baselines` | The owner wants secondary targets in the GARCH track (O-M3-2) |

## 4. Code walkthrough by `file:line`

### 4.1 `src/ivf/models/garch.py`

- [`src/ivf/models/garch.py:90`](../../src/ivf/models/garch.py#L90): `arch`'s start value, the first 75 squared values with weights `0.94^i`;
  the same at every cutoff.
- [`src/ivf/models/garch.py:111`](../../src/ivf/models/garch.py#L111): the recursion, element i using `e[:i]` only; the last element is
  the one-step forecast past the sample. The GJR sign term [`src/ivf/models/garch.py:132`](../../src/ivf/models/garch.py#L132);
  EGARCH's `sqrt(2/π)` [`src/ivf/models/garch.py:143`](../../src/ivf/models/garch.py#L143).
- [`src/ivf/models/garch.py:148`](../../src/ivf/models/garch.py#L148) and [`src/ivf/models/garch.py:159`](../../src/ivf/models/garch.py#L159): the k-step paths;
  EGARCH's product [`src/ivf/models/garch.py:175`](../../src/ivf/models/garch.py#L175).
- [`src/ivf/models/garch.py:179`](../../src/ivf/models/garch.py#L179): `G = (252/21)·Σ/100²`, refusing any window but 21.
- [`src/ivf/models/garch.py:262`](../../src/ivf/models/garch.py#L262): one `arch` fit; the usability checks [`src/ivf/models/garch.py:302`](../../src/ivf/models/garch.py#L302)
  and [`src/ivf/models/garch.py:310`](../../src/ivf/models/garch.py#L310). [`src/ivf/models/garch.py:316`](../../src/ivf/models/garch.py#L316) is the AIC rule.
- [`src/ivf/models/garch.py:333`](../../src/ivf/models/garch.py#L333): decision D7, with the first real run's failure in its docstring.
- [`src/ivf/models/garch.py:357`](../../src/ivf/models/garch.py#L357): `fit` on returns up to the cutoff; `one_step` [`src/ivf/models/garch.py:413`](../../src/ivf/models/garch.py#L413)
  is `σ²_{t+1|t}` from returns up to t.
- [`src/ivf/models/garch.py:425`](../../src/ivf/models/garch.py#L425), [`src/ivf/models/garch.py:438`](../../src/ivf/models/garch.py#L438) and [`src/ivf/models/garch.py:446`](../../src/ivf/models/garch.py#L446): the iterated AR mean (with
  its constant for d = 0) and `s² = Σ w²_{h−j} σ²_{t+j|t}/100²`, `w = ψ` (d = 0) or `Ψ` (d = 1)
  [`src/ivf/models/garch.py:540`](../../src/ivf/models/garch.py#L540); the d = 0 AR stationarity check
  [`src/ivf/models/garch.py:237`](../../src/ivf/models/garch.py#L237) (amended M3, review A).
- [`src/ivf/models/garch.py:547`](../../src/ivf/models/garch.py#L547): concept note 10's numbers.

### 4.2 `src/ivf/models/rv21.py` and the runner

- [`src/ivf/models/rv21.py:63`](../../src/ivf/models/rv21.py#L63): `rv21` on the row of its window's **end**, where it is observed.
- [`src/ivf/models/rv21.py:68`](../../src/ivf/models/rv21.py#L68): `τ ≤ c − 21`, the heart of O-M2-7; used at
  [`src/ivf/models/rv21.py:147`](../../src/ivf/models/rv21.py#L147).
- [`src/ivf/models/rv21.py:77`](../../src/ivf/models/rv21.py#L77): the same-rows rule (22 returns and VIX at t), data only.
- [`src/ivf/evaluation/walkforward.py:182`](../../src/ivf/evaluation/walkforward.py#L182): `fit(inputs.upto(c))` [`src/ivf/evaluation/walkforward.py:234`](../../src/ivf/evaluation/walkforward.py#L234),
  and the window-target assertion [`src/ivf/evaluation/walkforward.py:243`](../../src/ivf/evaluation/walkforward.py#L243).
- [`src/ivf/evaluation/walkforward.py:313`](../../src/ivf/evaluation/walkforward.py#L313): the frame takes `v_hat` and `var_method` from the model.

### 4.3 The command and the registry

- [`src/ivf/models/cli.py:212`](../../src/ivf/models/cli.py#L212): the expected lists from [`src/ivf/models/registry.py:125`](../../src/ivf/models/registry.py#L125),
  M2's check, `pre_holdout_inputs`, both tracks, [`src/ivf/models/cli.py:69`](../../src/ivf/models/cli.py#L69).
- [`src/ivf/models/registry.py:111`](../../src/ivf/models/registry.py#L111): the `garch` block, read strictly.

## 5. Data flow

```
interim ──check.frozen_check──► 𝒯 ──fold_schedule──► 97 dev folds ──pre_holdout_inputs (date < holdout_start)
   │
   ├─ rv21 track: r = Δ ln ^GSPC, VIX ──► per fold c: fit(inputs ≤ c) ──► forecast at t ∈ [c, c+62] with inputs ≤ t
   │      garch/gjr/egarch: σ²_{t+1|t} ──► 21-step path ──► G_t = 12·Σ/10⁴ (native)
   │      har_rv (pairs τ ≤ c−21), rw (rv21 ending at t), vix_implied ((V_t/100)²)
   │      target: rv21 at t+21 (window t+1..t+21), scored if before holdout_start
   └─ vix_level track (VIX, h = 1, 5, 21): M2's five baselines + arima_garch_log (AR + constant on 100·ln VIX, GARCH errors)
          │
          same rows per (target, h) against registry.models_for ──► metric ──► merged into processed/run_id=<id>/
```

The architecture spec's `models`, `configs` and `processed` labels were updated; the pipeline's shape did not change.

## 6. Pitfalls and their guards

| Pitfall | Guard | Test |
|---|---|---|
| A window target leaks into training (O-M2-7) | Inputs ≤ c; `training_origins`; the runner's window assertion | `test_poisoned_returns_after_the_cutoff_change_no_fit_and_no_forecast_at_the_cutoff`, `test_har_rv_trains_only_on_origins_whose_window_ends_by_the_cutoff`, `test_the_runner_refuses_a_fit_whose_last_target_window_ends_after_the_cutoff`, `test_every_rv21_fit_ends_at_its_cutoff` |
| A forecast at t uses a return after t (L1) | The own recursion, indexed by the count of returns ≤ t | `test_truncation_rv21_forecasts_at_t_do_not_depend_on_data_after_t` (4 origins, 6 models), `test_garch_forecast_at_t_is_the_aggregate_of_its_own_recursion` |
| Reading the holdout | `pre_holdout_inputs`; two refusals in `run_rv21` | `test_poisoning_every_holdout_row_changes_nothing_in_the_rv21_track` (42 of 63 scored in the last fold), `test_run_rv21_refuses_holdout_sessions_and_holdout_folds` |
| Units: percent², 252/21, mean | Zero mean; `aggregate` | `test_a_constant_1_percent_variance_aggregates_to_rv21_of_twenty_one_1_percent_returns`, `test_garch_21_session_aggregate_by_hand`, `test_the_target_of_origin_t_is_the_window_t_plus_1_to_t_plus_21` |
| ARIMA-GARCH's mean, variance and d | d = 0 by default; ψ (d = 0) or Ψ (d = 1) weights | `test_d0_forecast_by_hand`, `test_d0_matches_arch_s_own_mean_and_variance_at_every_horizon` (arch as oracle at h = 1, 5, 21), `test_ols_residuals_by_hand`, the d = 1 option's hand and oracle tests (amended M3, review A) |
| The recursion is not the model `arch` estimated | `FILTER_RTOL` check at every fit | `test_our_recursion_reproduces_arch_in_sample_and_its_one_step_forecast` (3 kinds × 2 distributions), `test_the_analytic_path_equals_arch_s_analytic_forecast` |
| EGARCH's infinite expectation under t; residuals not of unit scale | Empirical M over residuals rescaled to unit RMS | `test_egarch_mgf_by_hand`, `test_egarch_expected_variance_equals_brute_force_enumeration_of_every_shock_path`, `test_egarch_mgf_uses_the_standardised_residuals_rescaled_to_unit_rms` (amended M3, review A) |
| A non-stationary or unconverged fit used; a candidate refused silently | `fit_candidate` refuses it (GARCH persistence, d = 0 AR stationarity); previous fold kept; `candidates_refused` in the `fit` table (amended M3, review A) | `test_every_candidate_is_usable_on_clean_synthetic_data`, `test_a_non_stationary_ar_candidate_is_refused`, `test_ar_stationary_by_hand`, `test_a_non_stationary_fit_is_not_usable`, `test_garch_rv_fit_fails_when_no_candidate_is_usable`, `test_choose_keeps_the_lowest_aic_among_usable_fits_and_a_tie_keeps_normal` |
| A missing value filled, or a short sample | Skipped (D7) | the two regression tests above |
| Models scored on different rows | data-only `forecastable` (22 returns and VIX at t); `check_same_rows` with `registry.models_for` | `test_an_origin_needs_the_22_returns_t_minus_21_to_t_and_vix_at_t` (amended M3, review A), `test_every_rv21_model_forecasts_the_same_origins_and_a_vix_gap_drops_the_origin_for_all` |
| The second step deletes the first's rows | `_merge_write` | `test_garch_command_merges_into_the_baselines_tables` |

## 7. Measured results

### 7.1 The real run (counts and timings only, D2)

Three runs on 2026-10-29 from the repo root, uncommitted M3 code on `ab223f4`, under
`/usr/bin/time -v uv run python -m ivf.models garch`, nothing else heavy running.

| Run | Result |
|---|---|
| First | **exit 1** after 41.6 s wall: `arima_garch_log`'s first fit raised (section "What the first real run found"). The rv21 track had finished: 97 folds, 36,666 rows, Student-t chosen at all 97 cutoffs by each GARCH model |
| Second (after decision D7; `arima_garch_log` still d = 1) | exit 0; wall 3 min 57.5 s; peak RSS 519,628 KB; same row counts as the third; 0 of 582 `arima_garch_log` candidates refused; AR order 2 at all 97 cutoffs |
| **Third (amended M3, review A: d = 0, rescaled EGARCH residuals)** | **exit 0**; wall 3 min 2.8 s; peak RSS 617,736 KB. Seconds per model: `arima_log` 157.7, `arima_garch_log` 8.2, `egarch` 5.3, `gjr` 3.5, `garch` 3.4, every other model under 0.4 |
| rv21 track | 97 folds fitted, 0 skipped, 6,111 origins (97 × 63, all of dev), 0 not forecastable, 36,666 forecast rows (6,111 × 6 models); `har_rv` floored 0 forecasts |
| vix_level track (VIX) | 97 folds, 6,111 origins, 0 not forecastable, 109,998 forecast rows (6,111 × 3 horizons × 6 models) |
| Totals | 146,664 forecast rows, 146,376 scored (target observed and before `holdout_start`), 1,164 fit rows (97 × 12 models), **0 with `train_last_target_date` after the cutoff**, 33 metric rows |
| Convergence and fallback | `garch`, `gjr`, `egarch`, `arima_garch_log`: 97 of 97 converged each; 0 of 194 candidates refused for each GARCH-family model; **15 of 582 for `arima_garch_log`, all the p = 0 Student-t candidate** (a constant mean on ln VIX, GARCH persistence ≥ 1: the level's persistence leaking into the variance, concept note 10), at 15 cutoffs, never the chosen one; **no fallback** in any fit; the EGARCH rescale applied in 97 of 97 fits |
| Model choices (counts) | Student-t at all 97 cutoffs for `garch`, `gjr`, `egarch` and `arima_garch_log`; AR order 2 at all 97 cutoffs |
| Holdout | printed `no row on or after it is read`; the run reads `pre_holdout_inputs` only |

The tables were merged into the run's partition under `$VOL_DATA_DIR`; the `run_id`, the gap
positions and every loss stay local (D2).

### 7.2 Synthetic data (publishable)

`uv run python -m ivf.models --synthetic garch --print-metrics` on 2026-10-29, right after the third
real run (amended M3, review A): exit 0, wall 2 min 52.2 s, peak RSS 623,164 KB, the same row counts as
the real run, no fallback, 0 fits past their cutoff, 0 candidates refused. Normal innovations were
chosen at all 97 cutoffs by `garch`, `gjr` and `egarch` (the synthetic returns **are** normal
GARCH(1,1)); `arima_garch_log` chose normal at 84 and t at 13, AR order 1 at 96 and 2 at 1 (the truth
is an AR(1)). The truth row is the known GARCH(1,1) recursion aggregated the same way (scratch script
on the stored rows; notebook 03 section 3 does it for its folds).

**rv21, dev, h = 21** (n = 6,090 scored rows per model; QLIKE on variance is the primary loss for these `native` forecasts):

| Model | QLIKE | vs truth | RMSE | bias (y − ŷ) |
|---|---:|---:|---:|---:|
| truth `E_t[v]` | 0.1379 | — | 3.9769 | −0.8149 |
| `garch` | 0.1396 | +1.2% | 4.0005 | −0.8519 |
| `gjr` | 0.1399 | +1.5% | 4.0070 | −0.8604 |
| `har_rv` | 0.1409 | +2.2% | 4.0238 | −0.9269 |
| `egarch` | 0.1415 | +2.6% | 3.9957 | −0.7785 |
| `rw` | 0.2164 | +57% | 4.5260 | −0.0093 |
| `vix_implied` | 0.4023 | +192% | 9.0165 | −4.8612 |

- The GARCH family is within 3% of the floor; the correct model (`garch`) is closest, and GJR's
  extra gamma costs little (fitted about −0.005). `rw` is unbiased in vol points but far worse.
- `egarch` before the review A rescale: QLIKE 0.141499, RMSE 3.9962; after: 0.141490, 3.9957 (the
  synthetic residuals are normal, so their RMS was already close to 1).
- **Even the truth has a bias of −0.81 vol points.** A `native` forecast is the RMS vol
  `100·sqrt(E_t[v])`, which is above `E_t[y]` by Jensen (ADR 0001 4.3), so a negative bias in vol
  points is expected and RMSE does not rank these forecasts as means. QLIKE does.
- `vix_implied` means nothing here: synthetic VIX is independent of synthetic returns. On real data
  its bias is the variance risk premium (measured locally).

**vix_level, dev, RMSE in vol points** (the same run; `arima_log`'s numbers equal M2's to 4 decimals;
the d = 1 column is the first version, from the run before review A):

| h | `arima_garch_log` mean (d = 0) | median (d = 0) | first version, d = 1, mean | `rw` | `arima_log` | `har_log` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1798 | 1.1804 | 1.1872 | 1.1857 | 1.1798 | 1.1803 |
| 5 | 2.5373 | 2.5437 | 2.6159 | 2.5992 | 2.5375 | 2.5389 |
| 21 | 4.4981 | 4.5306 | 5.0864 | 4.9454 | 4.5002 | 4.5133 |

- With d = 0, ARIMA-GARCH is the best VIX model at every horizon, by a hair over `arima_log`
  (4.4981 against 4.5002 at h = 21) and 9.0% below the random walk at h = 21; M2's truth floor there
  was 4.4848. The GARCH errors add little on the synthetic market, whose ln VIX shocks are
  homoskedastic; on real data VIX changes cluster.
- The lognormal mean beats the median on RMSE at every horizon, as for M2's log models.
- The first version (d = 1) was worse than the random walk at every horizon (RMSE +0.1%, +0.6%,
  +2.9%): the variance it added grew like a random walk's while the truth reverts (phi = 0.98). That is
  review A's refutation of claim 8 (O-M3-4, closed).

### 7.3 Mutation checks

Every new test was mutation-checked in a scratch copy of the repo (not the checkout): each mutant
applied alone, `tests/test_garch.py` and `tests/test_rv21.py` run, the file restored
(`mutate.py` in the session scratchpad).

- **Round 1: 33 of 36 killed, 3 survived.** (1) ARIMA-GARCH's Ψ weights in the wrong order and (2)
  its one-step variance taken one change ahead (look-ahead) both survived because the hand test used
  a constant variance path and no test truncated `arima_garch_log`'s data; (3) HAR-RV's weekly window
  4 instead of 5 survived because the hand-worked returns gave the same mean over 4 and 5 sessions.
  Fixes: `test_arima_garch_variance_weights_a_moving_garch_path_by_hand` (now `test_d1_option_...`),
  `test_arima_garch_forecast_at_t_does_not_depend_on_closes_after_t`, and new numbers in
  `test_har_rv_design_by_hand`.
- **Round 2: 36 of 36 killed**, then one more for a test added after it (the 504/503 boundary of
  `run_rv21`): **37 of 37**, one of them a sanity mutant (the aggregate doubled).
- **Review A round (amended M3, review A).** The reviewer's four extra mutants (three had survived),
  ten for the d = 0 model and the new column, and the 37 above, re-pointed where the code moved: **48 of
  51 killed** at first. Three survived: (1) the AR residual shifted by one, again, because the new
  "every candidate usable" test used homoskedastic synthetic VIX, where the variance recursion hardly
  depends on the residuals; (2) the d = 0 AR stationarity check removed (no test produced a
  non-stationary AR); (3) `ols_residuals` ignoring the constant (the gap test computed its expected
  value with the same function). Fixes: that test now uses an AR(1) whose shocks are GARCH(1,1);
  `test_a_non_stationary_ar_candidate_is_refused`; `test_ols_residuals_by_hand`. Then **51 of 51
  killed**, including the reviewer's four. New mutants: EGARCH M from raw residuals; M without the
  unit-RMS rescale; the AR residual shifted; `forecastable` with 21 returns; `choose` by log-likelihood;
  d = 0 forecast without the constant; d = 0 with the cumulative Ψ; d = 0 history missing `x_t`; AR
  stationarity unchecked; the d = 0 backcast from the raw level; `ols_residuals` without the constant;
  the residual without the constant; `candidates_refused` counting usable candidates; the registry
  accepting any d.

| Area | Mutants (all killed in round 2) |
|---|---|
| Recursion | backcast decay 0.95; GJR gamma on every shock; GJR's i = 0 term `γ·bc`; EGARCH without `sqrt(2/π)` |
| Aggregation | path starting one step late; `252/22`; `/100` instead of `/100²`; GJR persistence with `γ` not halved; EGARCH product as `M(1)^(k−1)`; EGARCH drift with one term too many |
| Innovation rule | largest AIC; a tie keeps t; stationarity not checked |
| L1 and gaps | one-step variance from `r_{t+1}`; indexing by session instead of by observed return; a missing return filled with 0; ARIMA-GARCH back to the run after the last gap (the first real run's rule) |
| ARIMA-GARCH | Ψ in the wrong order; variance one change ahead; mean from `x_{t−1}`; AIC over orders on different samples |
| rv21 models | HAR-RV target one session early; weekly window 4; never floors; `vix_implied` from `V_{t−1}`; `rw` from `rv21` at t−1; `forecastable` ignoring VIX |
| Runner and command | fit on inputs up to c + 21 (the O-M2-7 leak); the window-target assertion off; targets at t + 20; holdout-date refusal off; holdout-fold refusal off; native `v_hat` ignored; every row labelled `plugin`; the 504 boundary at `<=`; overwrite instead of merge |

Not tried, as equivalent: removing `check_same_rows` from `run_rv21` (rows are built from the
runner's own origins, so no model can drop one) and passing a forecast one more session of inputs
(every model indexes by origin).

### 7.4 Gates

Run from the repo root on 2026-10-29 with the M3 changes uncommitted, after the review A fix round
(amended M3, review A; the first round's run had `839 passed in 99.11s` and 234 scanned files).

| Command | A pass means | Result |
|---|---|---|
| `uv run pytest -q` | All Python tests pass | `851 passed in 287.38s` (790 after M2: 61 new, in `test_garch.py` 40, `test_rv21.py` 18, `test_notebooks.py` 3). The time is inflated: another project's processes held the load average near 30 during the run |
| `uv run python tools/build_notebooks.py --check` | The four notebooks equal their specs | `ok: 4 notebook(s) in <repo>/notebooks match their specs` |
| `uv run python tools/export_app_data.py --mode synthetic --check` | The committed fixture is current (M3 adds no export) | `ok: 1 file(s) in <repo>/app/public/data/synthetic match a fresh synthetic export` |
| `uv lock --check` | No dependency added (`arch==8.0.0` was pinned in M0) | `Resolved 142 packages` |
| `uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out`, then with `--check --no-png` | The architecture spec's M3 labels are rendered | `built 5 spec(s) ... (5 file(s) changed, 0 removed, 0 kept)`; then `ok: 5 spec(s) up to date in docs/architecture/out` |
| `uv run python tools/licensing/scan_real_rows.py` | No real value in any file git would commit | `clean: 0 lines flagged in 235 text file(s) against 7 raw snapshot(s)` |
| `git status --porcelain --ignored` | No data file in the repo | no parquet or CSV; the store is under `$VOL_DATA_DIR` only |
| `cd app && npm run verify` | | Not run: M3 does not touch `app/` |

## 8. Answers to the predict-first questions

1. Closer to twice than you might think: `Σ = 21 + (1 − 0.98²¹)/0.02 = 38.29` against 21 at the
   long-run level, so `G` is 1.82 times the long-run value (concept 09).
2. High: VIX² is a risk-neutral expectation that includes the variance risk premium (concept 06), so
   `vix_implied` should show a negative bias (y − ŷ < 0). On synthetic data VIX is independent of
   S&P 500 returns, so its rows there measure nothing about the premium.
3. `c − 21`: its target window `c−20..c` ends on the cutoff. `c − 20` would need the return at `c + 1`.
4. 0.977, against a truth of no clustering at all (concept note 10).
5. With d = 1, no: the mean does not revert, and the lognormal correction adds a variance that grows
   like a random walk's, so its mean forecast was worse than the random walk at every horizon (review
   A). With d = 0, yes: the AR reverts to the mean, and on the synthetic market it is within 0.3% of
   M2's `arima_log` (section 7.2).

## 9. Explain it back

1. Why is the 21-session forecast a sum of variances and not of volatilities?
2. Why does the forecast use this project's recursion rather than `arch`'s?
3. What exactly would leak if `rv21` were stored on its origin's row and `fit` got `y[:c+1]`?
4. Why can a simulation not give EGARCH-t's 21-day expected variance?
5. Why does GARCH on the VIX level report strong clustering on data that has none?

## 10. Decisions waiting on you

| # | Decision | Default taken (2026-10-29, owner can override) |
|---|---|---|
| O-M3-1 | The innovation rule (D3): AIC between normal and t at every cutoff | AIC, tie → normal. Real run: t at all 97 cutoffs for all three models |
| O-M3-2 | ARIMA-GARCH on the secondary indices (VIX9D … VVIX)? | No: VIX only (D11) |
| O-M3-3 | The uncalibrated `garch`/`gjr`/`egarch` rows on `vix_level` and the `*_cal` maps of ADR 0001 4.4 | Not in M3's brief; open for M5 with Mincer-Zarnowitz |
| O-M3-4 | `arima_garch_log`'s d (amended M3, review A). d = 1 was worse than the random walk at every horizon on the synthetic market (RMSE 1.1871/2.6158/5.0857 against 1.1857/2.5992/4.9454 at h = 1/5/21); d = 0 was 1.1798/2.5373/4.4975 (coordinator's re-measurement) | **Closed: d = 0** (`garch.d: 0`); d = 1 remains an option |
| O-M3-5 | Move real results into the repo (pending data-policy confirmation) | Not done: real results stay under `$VOL_DATA_DIR` |
