# Milestone 2: baselines

**Status:** milestone 2 of 6: the walk-forward runner, the five baseline models (random walk,
two historical means, ARIMA on log IV, HAR on log IV), the four losses, the forecast store, the
`python -m ivf.models baselines` command, notebook 02 and concept notes 07 and 08. Dev period
only: **no row on or after `holdout_start` is read**. No real-data number is published.

**Purpose.** To explain how a forecast is made and scored without looking ahead, and what the
baselines are that every later model must beat. Read in order: answer
[Predict before you look](#1-predict-before-you-look); read sections 2 to 7; compare with
[the answers](#8-answers-to-the-predict-first-questions); then [Explain it back](#9-explain-it-back).

**Licensing (ADR 0001 D2).** Numbers below are hand-worked or measured on **synthetic** data. For
real data this walkthrough records only that the command ran (exit code, row counts, timings);
the losses are measured locally and not published. The owner has a data-policy change pending;
until it is confirmed, real results stay under `$VOL_DATA_DIR`.

**Line numbers are a snapshot**, resolved against the file text by a script when this was
written.

**Companions.** [ADR 0001](../adr/0001-python-core-versioned-json-app.md) (4.5 carries the M2
amendment), [data model](../architecture/data-model.md) (`fit`, `forecast`, `metric` amended M2),
concept notes [07 ARIMA](../concepts/07-arima-and-information-criteria.md) and
[08 HAR](../concepts/08-har-and-heterogeneous-memory.md), [milestone 1](01-data-and-eda.md).

**Amended M2, review A.** An adversarial review attacked the eight claims of the first M2 report.
It refuted one (the KPSS d rule), found a hole in the same-rows guard behind a second, and found
two mutants the builder's own list had not tried. [What review A found](#what-review-a-found)
names each with its fix; every statement changed because of it says "(amended M2, review A)". The
ledger is [docs/validation/M2-review-A-baselines.md](../validation/M2-review-A-baselines.md).

## What review A found

| Claim or area | Verdict | Fix |
|---|---|---|
| 1. No row on or after 2024-07-01 is read (poisoning test) | not refuted | — |
| 2. Truncation test, 5 models, 4 origins | not refuted | — |
| 3. `train_last_target_date` ≤ `cutoff_date` (L2) | not refuted, **open for M3**: "the array ends at the cutoff" holds only because each value sits on the row of the date it is observed; a window target stored on its origin's row (M1's `forward_rv21`) would leak | Stated in both module docstrings, the ADR and O-M2-7 below |
| 4. ARIMA's prediction uses data ≤ t and matches the AR(1) closed form | not refuted | — |
| 5. Each loss is minimised where the docstring says | not refuted | — |
| 6. Same rows hold everywhere | **partly refuted**: the output held, but `check_same_rows` compared only the models present, so a model missing every h = 1 row passed | Every expected model must be present in every (target, horizon) group; test added |
| 7. KPSS is a defensible d rule | **REFUTED**. On stationary AR(1) series it chose d = 1 on 52% (phi 0.98, n 2,570), 24% (0.98, 8,700), 76% (0.99, 2,570) and 52% (0.99, 8,700) of seeds (review A, 100 each; the coordinator's fresh seeds: 47%, 28%, 70%, 50%). Forcing d = 1 on the synthetic law raised h = 21 RMSE 17% (5.006 to 5.845) with a level-dependent bias. The first real run chose d = 1 at all 97 VIX cutoffs: what KPSS does to a persistent stationary series | `arima_log` uses **d = 0** (ARMA with a constant, (p, q) by AIC); KPSS kept as the option `d_rule: kpss`. Default taken 2026-10-29, owner can override (O-M2-2) |
| 8. 97 dev folds, 2000-03-15, 2024-06-28, 42 of 63 scored at h = 21 | not refuted | — |
| Mutants outside the builder's list | **2 survived**: KPSS and the ARIMA fit on the whole series instead of the longest gap-free run (no test gave ARIMA a missing close), and the 504 minimum-training boundary (`<` against `<=`) | Three tests for the gap and missing-value paths, one for the 503/504 boundary; 9 new mutants, all killed (section 7.3) |

## 1. Predict before you look

1. At h = 1, which baseline has the lowest RMSE on VIX, and by roughly how much?
2. At h = 21, does ARIMA on log VIX beat the random walk? What decides it?
3. The lognormal correction raises every log-model forecast. Which of RMSE, MAE, MAPE and QLIKE
   get better, and which get worse?
4. The last dev fold ends on 2024-06-28. How many of its 63 origins are scored at h = 21?
5. A run on data truncated at origin t: which forecasts must equal the full run's, and why does
   that catch look-ahead?

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| Model registry | `configs/models.yaml`, `src/ivf/models/registry.py` | `model_def` rows and the baseline settings, read strictly |
| Baselines | `src/ivf/models/baselines.py` | `fit(y[:cutoff+1])` and `forecast(fit, y, origins)` for each model |
| Losses | `src/ivf/evaluation/losses.py` | bias, RMSE, MAE, MAPE, QLIKE exactly as ADR 0001 4.3 |
| Runner | `src/ivf/evaluation/walkforward.py` | folds, the pre-holdout inputs, fit and forecast per fold, same-rows check, scoring |
| Command | `src/ivf/models/cli.py` | `python -m ivf.models [--synthetic] baselines`: reads interim, runs `check`, writes the store |
| Store | `ROOT/processed/run_id=<id>/` | `run_input`, `calendar` (with `fold_id`), `fold`, `fit`, `forecast`, `metric`; `runs.parquet` |
| Notebook | `notebooks/02_baselines.ipynb` (spec in `tools/build_notebooks.py`) | the last 4 dev folds of VIX, losses, the synthetic truth as a floor |
| Tests | `tests/test_losses.py`, `tests/test_baselines.py`, `tests/test_walkforward.py`, `tests/test_notebooks.py` | hand-worked numbers, look-ahead rules, the command end to end |

**App.** The app does not show forecasts yet. Its forecast view, `model_race.json` (data model
5.3), plots dev context **and the holdout**, with Mincer-Zarnowitz rows from M5; M2 may not read
the holdout, so no export is added and the synthetic fixture is unchanged. It arrives with the
holdout run.

## 3. Decisions

Each default below was taken on 2026-10-29; the owner can override any of them.

| # | Decision | Rejected | Would win if |
|---|---|---|---|
| D1 | **statsmodels `SARIMAX`** for ARIMA, with our own order search | `pmdarima.auto_arima` (pinned already): refuses NaN in the series, and real VIX has core gaps in burn-in (G3) and VIX1Y/VVIX have gaps in dev (G5); freezing parameters inside a fold would still go through statsmodels | The data had no gaps and its stepwise search were wanted |
| D2 | **d = 0 (an ARMA with a constant on ln IV), (p, q) ∈ {0,1,2}² by AIC** at every cutoff (amended M2, review A) | d by KPSS at 5% (the first M2 default; review A measured it choosing d = 1 on 24-76% of stationary AR(1) series as persistent as ln VIX; kept as `d_rule: kpss`); AIC across d (likelihoods of different data); BIC (aims at the true order, not the best forecast) | ln IV had a unit root, which the random walk baseline already covers |
| D3 | **ARIMA fallback**: previous fold's model, or ARIMA(0,1,0) at the first fold | Dropping the fold (breaks same rows); RW in levels (a different model) | — |
| D4 | **Evaluate the lognormal mean** `exp(μ + s²/2)`; store `exp(μ)` too and score both in `metric` (`forecast = mean/median`) | Evaluate the median (biased low as a mean); store only one (hides what MAE rewards) | The project switched its primary loss to MAE |
| D5 | **HAR direct**, one OLS per horizon, `s² = RSS/(n−4)` constant across origins | Iterating the h = 1 model (compounds misspecification); a heteroskedastic `s²` (a variance model: M3) | Vol-of-vol were modelled |
| D6 | **Historical means in levels**, frozen at the cutoff; fixed window **252 sessions** of the calendar | A window counted in closes (moves with gaps); a log-space mean (would need a back-transform) | — |
| D7 | **Forecast at the fold level** (`y` up to the fold's last origin), guarded by the truncation test | Calling each model once per origin with `y[:t+1]` (structurally safe, about 10 times slower for ARIMA) | A model whose forecast path cannot be tested by truncation |
| D8 | **Secondary targets** `<symbol>_level` for VIX9D, VIX3M, VIX6M, VIX1Y, VVIX | Adding them to `protocol.yaml` (changes the frozen `protocol_hash`, R1) | The owner wants them as protocol targets (O-M2-1) |
| D9 | **`min_train_sessions = 504`** before a symbol's fold is fitted; an origin needs `t-21..t` complete | Fitting from the first close (a 30-session ARIMA is noise) | — |
| D10 | **Implied vol only**; `rv21` baselines (`rw`, `har_rv`) move to M3 with the GARCH track | Doing them here | The owner wants the rv21 track before M3 |
| D11 | **Losses printed only with `--print-metrics`**; the default output is counts and timings | Printing losses always (easy to paste a real number into a tracked file) | — |

## 4. Code walkthrough by `file:line`

### 4.1 The registry: `configs/models.yaml`, `src/ivf/models/registry.py`

- [`src/ivf/models/registry.py:53`](../../src/ivf/models/registry.py#L53): exact keys, enumerations for `family`,
  `fit_space`, `multistep`; the settings block. Not part of `protocol_hash` (the file says why).
- [`src/ivf/models/registry.py:48`](../../src/ivf/models/registry.py#L48): `VIX` → `vix_level`, the protocol target;
  the other symbols → secondary targets.

### 4.2 The models: `src/ivf/models/baselines.py`

- The module docstring states the contract: `fit` gets `y[:cutoff+1]` only, `forecast` may use
  `y[:t+1]` at origin t.
- [`src/ivf/models/baselines.py:76`](../../src/ivf/models/baselines.py#L76): mean `exp(μ+s²/2)` as `yhat`, median
  `exp(μ)` as `yhat_naive`, the 80% interval `exp(μ ∓ 1.2816·s)`.
- [`src/ivf/models/baselines.py:110`](../../src/ivf/models/baselines.py#L110): `yhat = y_t`.
- [`src/ivf/models/baselines.py:123`](../../src/ivf/models/baselines.py#L123): the window is counted in calendar
  sessions ([`src/ivf/models/baselines.py:138`](../../src/ivf/models/baselines.py#L138)), nulls skipped.
- [`src/ivf/models/baselines.py:156`](../../src/ivf/models/baselines.py#L156) and [`src/ivf/models/baselines.py:172`](../../src/ivf/models/baselines.py#L172):
  the pairing line [`src/ivf/models/baselines.py:192`](../../src/ivf/models/baselines.py#L192) is where L2 lives
  inside the model (the array ends at the cutoff, so `tau + h <= cutoff`).
- [`src/ivf/models/baselines.py:227`](../../src/ivf/models/baselines.py#L227): KPSS on the longest gap-free run (G2), used
  only under the option `d_rule: kpss`.
- [`src/ivf/models/baselines.py:250`](../../src/ivf/models/baselines.py#L250): the d rule
  [`src/ivf/models/baselines.py:313`](../../src/ivf/models/baselines.py#L313) (d = 0 unless the option is set), the AIC argmin
  [[src/ivf/models/baselines.py|if ok and (best is None or aic < best[0])]], the two fallbacks,
  and the iterated forecast [`src/ivf/models/baselines.py:381`](../../src/ivf/models/baselines.py#L381)
  (`dynamic=True`: steps after `t+1` use the model's own predictions, never the data).

### 4.3 The losses: `src/ivf/evaluation/losses.py`

- [`src/ivf/evaluation/losses.py:66`](../../src/ivf/evaluation/losses.py#L66): `v/v̂ − ln(v/v̂) − 1` in variance space.
- The module docstring's table: which forecast each loss rewards. Under a lognormal, MAPE is
  minimised by `exp(μ−s²)`, MAE by `exp(μ)`, RMSE by `exp(μ+s²/2)`, QLIKE by `exp(μ+s²)`: ever
  higher. Tested on a quantile grid (`test_each_loss_rewards_the_forecast_the_docstring_says`).
- [`src/ivf/evaluation/losses.py:81`](../../src/ivf/evaluation/losses.py#L81): suppression below `min_cell_n` (ADR 4.8).

### 4.4 The runner: `src/ivf/evaluation/walkforward.py`

- [`src/ivf/evaluation/walkforward.py:52`](../../src/ivf/evaluation/walkforward.py#L52): dates only; reuses M1's
  `calendar.first_cutoff`.
- [`src/ivf/evaluation/walkforward.py:92`](../../src/ivf/evaluation/walkforward.py#L92): M1's `eda_inputs`, reused.
- [`src/ivf/evaluation/walkforward.py:112`](../../src/ivf/evaluation/walkforward.py#L112): the two refusals
  ([`src/ivf/evaluation/walkforward.py:130`](../../src/ivf/evaluation/walkforward.py#L130) and
  [`src/ivf/evaluation/walkforward.py:144`](../../src/ivf/evaluation/walkforward.py#L144)), the minimum-training rule, the
  fit on [`src/ivf/evaluation/walkforward.py:160`](../../src/ivf/evaluation/walkforward.py#L160)], the fallback, and the
  runtime L2 assertion.
- [`src/ivf/evaluation/walkforward.py:210`](../../src/ivf/evaluation/walkforward.py#L210): refuses a missing or non-positive
  forecast (a missing row is an error, ADR 4.5); `target_date = 𝒯[t + h]`.
- [`src/ivf/evaluation/walkforward.py:247`](../../src/ivf/evaluation/walkforward.py#L247) (every expected model in every
  group; amended M2, review A), [`src/ivf/evaluation/walkforward.py:269`](../../src/ivf/evaluation/walkforward.py#L269),
  [`src/ivf/evaluation/walkforward.py:275`](../../src/ivf/evaluation/walkforward.py#L275).

### 4.5 The command: `src/ivf/models/cli.py`

- [`src/ivf/models/cli.py:57`](../../src/ivf/models/cli.py#L57): the data root guard, `check.frozen_check` (the
  same refusal as `eda`), `run_id` from M1's `run_id_for`, the store written through
  `repo_guard.write_file`, and `process.upsert_run_row` (factored out of `run_eda` in M2).

## 5. Data flow

```
interim (iv, price) ──check.frozen_check──► master calendar 𝒯 ──fold_schedule──► 97 dev folds (dates only)
        │                                          │
        └──pre_holdout_inputs (date < holdout_start)──► y per symbol on dev sessions
                                                   │
          per fold c: fit(y[:c+1]) ──► forecast at origins c..c+62 (data ≤ t) ──► rows (yhat, yhat_naive, mu, s2)
                                                   │
          same-rows check ──► scored = target observed and dated < holdout_start ──► metric (dev, all)
                                                   │
          processed/run_id=<id>/{run_input, calendar, fold, fit, forecast, metric}.parquet + runs.parquet
```

The architecture diagram's store labels were updated for M2 (`configs/models.yaml`; the M2
tables); the pipeline's shape did not change.

## 6. Pitfalls and their guards

| Pitfall | Guard | Test |
|---|---|---|
| A fit sees data after its cutoff (L2) | The runner slices `y[:c+1]`; a runtime assertion on `train_last_target` | `test_every_fit_ends_at_its_cutoff_rule_l2`, `test_a_fit_does_not_change_when_data_after_its_cutoff_changes` |
| A forecast at t uses data after t (L1) | Trailing windows; ARIMA `dynamic=True` | the truncation test (4 origins, every model), `test_arima_forecast_matches_the_ar1_closed_form_and_ignores_data_after_the_origin` |
| Reading the holdout | `pre_holdout_inputs`; two refusals in `run_symbol` | `test_poisoning_every_holdout_row_changes_nothing` (every holdout close ×100: identical output), `test_run_symbol_refuses_a_holdout_session_or_a_holdout_fold` |
| Scoring a target in the holdout | `y` ends before `holdout_start`; `scored` filters on `target_date` too | the poisoning test: the last dev fold scores `63 − h` rows |
| Models scored on different rows | data-only origin rules; `check_same_rows` with the expected models; a NaN forecast raises | `test_same_rows_check_refuses_a_missing_origin`, `test_same_rows_check_refuses_a_model_missing_a_whole_horizon`, `test_a_missing_forecast_is_an_error_not_a_dropped_row`, `test_gaps_and_late_starts_are_rules_on_the_data_not_on_the_model`, `test_min_train_sessions_boundary_504_fits_503_does_not` |
| A missing close filled or bridged | Kalman missing-value path; KPSS on the longest gap-free run | `test_arima_treats_a_missing_close_as_missing_not_as_the_previous_value`, `test_arima_fits_through_missing_training_closes`, `test_kpss_runs_on_the_longest_gap_free_run` |
| Evaluating the median as if it were the mean | `yhat = exp(μ+s²/2)`; median stored separately | `test_lognormal_forecast_reports_the_mean_as_yhat_and_the_median_as_naive` |
| A failed fit silently dropping a fold | previous-fold fallback, recorded; first-fold failure is an error | `test_a_failed_fit_keeps_the_previous_fold_and_a_failed_first_fit_is_an_error`, both ARIMA fallback tests |

## 7. Measured results

### 7.1 The real run (counts and timings only, D2)

Two runs on 2026-10-29 from the repo root, uncommitted M2 code on `fecf88a`, after M1's real
pipeline. Both wrote the same `run_id` partition (same commit, dirty tree and inputs), so the
second replaced the first.

| Run | Result |
|---|---|
| First (ARIMA d by KPSS), `/usr/bin/time -v uv run python -m ivf.models baselines` | **exit 0**; wall 3 min 53.6 s; peak RSS 650,312 KB; ARIMA 229.5 s of it. **d = 1 at all 97 VIX cutoffs** (a count, read from the `fit` table before the re-run) |
| Second (amended M2, review A: d = 0), same command under `/usr/bin/time`, run **at the same time as the synthetic re-run** (so the timing is inflated) | **exit 0**; wall 741.7 s; peak RSS 643,372 KB; ARIMA 734.5 s of it. d = 0 at all 375 ARIMA fits (97 for VIX); all 375 converged; no fallback in any of the 1,875 fits; 0 fits with `train_last_target_date` after the cutoff |
| Rows (identical in both runs) | VIX (`vix_level`): 97 folds fitted, 0 skipped, 6,111 origins (97 × 63, all of dev), 0 not forecastable, 91,665 forecast rows. All six symbols: 353,055 forecast rows, 352,200 scored (target observed and before `holdout_start`), 1,875 fit rows, 126 metric rows |
| Holdout | both printed `no row on or after it is read`; the latest `origin_date` in the store is 2024-06-28 |

Per-index fold, origin and gap counts, the `run_id` and every loss are printed locally and not
published (D2: the gap counts and the `run_id` are real-data-derived).

### 7.2 Synthetic data (publishable)

`uv run python -m ivf.data --synthetic fetch|build-interim|verify|check`, then
`uv run python -m ivf.models --synthetic baselines --print-metrics`. Re-run on 2026-10-29 after
the review A fix (d = 0), concurrently with the real re-run: exit 0, wall 750.6 s, peak RSS
641,840 KB, 355,320 forecast rows (354,510 scored), 1,880 fit rows, no fallback, d = 0 at all 376
ARIMA fits. The synthetic ln VIX is an AR(1) with `phi = 0.98` (true d = 0), so the best possible
forecast is known: its RMSE is the floor in the first column (computed from the stored rows and
`SyntheticParams`, scratch script).

VIX, dev, RMSE in vol points (n = 6,110 / 6,106 / 6,090 scored rows at h = 1 / 5 / 21):

| h | truth | `arima_log` | `arima_log` median | `har_log` | `har_log` median | `rw` | `hist_mean` | `hist_mean_252` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1794 | 1.1798 | 1.1805 | 1.1803 | 1.1810 | 1.1857 | 5.9514 | 6.4516 |
| 5 | 2.5338 | 2.5375 | 2.5440 | 2.5389 | 2.5451 | 2.5992 | 5.9551 | 6.4947 |
| 21 | 4.4848 | 4.5002 | 4.5341 | 4.5133 | 4.5432 | 4.9454 | 5.9662 | 6.6438 |

- ARIMA is within 0.4% of the floor at every horizon and HAR within 0.7%; the random walk is
  10% above it at h = 21, where mean reversion matters. The historical means ignore the state.
- **Which back-transform each loss rewards**, `arima_log` on VIX (`--print-metrics`):

  | h | loss | `mean` (scored) | `median` | lower |
  |---:|---|---:|---:|---|
  | 1 | MAE | 0.896661 | 0.896814 | mean |
  | 5 | MAE | 1.942957 | 1.940412 | median |
  | 21 | MAE | 3.504057 | 3.492034 | median |
  | 21 | MAPE | 0.192293 | 0.187051 | median (also at h = 1 and 5) |
  | 21 | QLIKE | 0.104243 | 0.108509 | mean (also at h = 1 and 5) |
  | 21 | RMSE | 4.500182 | 4.534126 | mean (also at h = 1 and 5) |

  RMSE and QLIKE prefer the mean, MAPE the median at every horizon, MAE the median once `s²`
  is large enough to matter (h = 5, 21), as the losses docstring predicts.
- **What the KPSS rule chose in the first synthetic run** (amended M2, review A: the rule is no
  longer the default). VIX: d = 1 at 2 of 97 cutoffs. The shorter synthetic tenors, the same
  AR(1) through a fixed map, got d = 1 more often: VIX9D at 15 of 45 cutoffs, VIX3M at 12 of 51,
  VIX6M at 14 of 57, VIX1Y at 10 of 61; it was wrong every time (true d = 0). That run's VIX
  ARIMA RMSE at h = 21 was 4.5013 against 4.5002 now.
- Timing: the first synthetic run took 425 s alone (ARIMA 421 s); the re-run's 750.6 s ran
  beside the real re-run.

### 7.3 Mutation checks

Every new test was mutation-checked in a scratch copy of the repo (not the checkout): each
mutant was applied alone, `tests/test_losses.py`, `tests/test_baselines.py` and
`tests/test_walkforward.py` run, and the file restored. 36 mutants; **34 killed** on the first
round. **2 survived**: HAR's `s²` divided by n instead of n−4, and HAR forecasting h = 5 with the
h = 1 coefficients (every forecast test used h = 1). `test_har_s2_is_the_rss_of_its_own_forecasts_over_n_minus_4_per_horizon`
was added, and both were then killed: **36 of 36 of the builder's own list**. That list was not
complete: review A found two more that survived (KPSS and the ARIMA fit on the whole series
instead of the longest gap-free run; the 504 boundary at `<=`).

**Review A fix round (amended M2, review A).** Tests were added for both, and for the other
changes of the round; 9 new mutants, applied the same way, **9 killed**: KPSS on the whole series;
KPSS on the first run; a missing close filled with the previous one in the forecast; missing
closes filled in the fit; `d_rule` ignored (always KPSS); the KPSS rule inverted; the 504 boundary
at `<=`; the missing-model check off; the expected-models argument ignored. The first round's
"ARIMA d rule inverted" mutant now targets the `kpss` option's line. The first round's other 35
mutants were not re-run after the fix round.

| Area | Mutants (all killed) |
|---|---|
| Losses | RMSE without the root; MAPE over `yhat`; QLIKE ratio inverted; QLIKE in vol space; bias sign; suppression at `n <= min_cell_n` |
| Back-transform | no correction; `+s²` instead of `+s²/2`; interval with `s²` instead of `s` |
| Historical means | window one session longer; NaN read as 0 |
| HAR | regressors one session late (look-ahead); target h−1 ahead; week window 4; `s²` over n; h = 1 coefficients at every h |
| ARIMA | `dynamic=False` (uses data after t); prediction one step early; d rule inverted; largest AIC; unconverged fits accepted; no constant at d = 0; closed-form fallback with mean `|dx|` |
| Runner | folds anchored on `dev_start`; fit sees `c+1` (L2); gap rule ignored; minimum-training rule inverted; `target_date` one session early; median rows for every model; same-rows check off; holdout-date refusal off; holdout-fold refusal off; `pre_holdout_inputs` returning holdout rows; fallback not recorded; calendar `fold_id` off by one; NaN forecast not refused |

One mutant was not tried because it is equivalent: removing `target_date < holdout_start` from
`scored` changes nothing, since `y` already ends before `holdout_start` (the condition is a
second guard).

### 7.4 Gates

Run from the repo root on 2026-10-29 with the M2 changes uncommitted, after the review A fix
round (amended M2, review A; the first round's run had 784 tests and 224 scanned files).

| Command | A pass means | Result |
|---|---|---|
| `uv run pytest -q` | All Python tests pass | `790 passed in 80.76s` (740 before M2: 50 new, in `test_losses.py` 13, `test_baselines.py` 21, `test_walkforward.py` 15, `test_notebooks.py` 1) |
| `uv run python tools/build_notebooks.py --check` | The three notebooks equal their specs | `ok: 3 notebook(s) in <repo>/notebooks match their specs` |
| `uv run python tools/export_app_data.py --mode synthetic --check` | The committed fixture is current (M2 adds no export) | `ok: 1 file(s) in <repo>/app/public/data/synthetic match a fresh synthetic export` |
| `uv lock --check` | No dependency added (statsmodels was pinned in M0) | `Resolved 142 packages` |
| `uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out`, then with `--check --no-png` | The architecture spec's two M2 labels are rendered | first round: `built 5 spec(s) ... (5 file(s) changed, 0 removed, 0 kept)`; after the fix round: `ok: 5 spec(s) up to date in docs/architecture/out` |
| `uv run python tools/licensing/scan_real_rows.py` | No real value in any file git would commit | `clean: 0 lines flagged in 226 text file(s) against 7 raw snapshot(s)` |
| `git status --porcelain --ignored` | No data file in the repo | no parquet, CSV or real JSON; the forecast store is under `$VOL_DATA_DIR` only |
| `cd app && npm run verify` | | Not run: M2 does not touch `app/` |

## 8. Answers to the predict-first questions

1. On synthetic data (section 7.2) the random walk, ARIMA and HAR are within about 1% of each
   other at h = 1: one session of mean reversion is tiny next to one session of noise. The real
   ranking is measured locally.
2. On mean-reverting data, yes: with d = 0 ARIMA reverts to the mean, and on the synthetic
   market it is far closer to the floor than the random walk at h = 21 (section 7.2). With d = 1
   it would be close to a random walk, which is why d is fixed at 0 (amended M2, review A).
3. RMSE improves (it is consistent for the mean, which the correction targets); MAE and MAPE
   can get worse (they reward the median and below); QLIKE on `(yhat/100)²` improves (it rewards
   an even higher forecast). The `median` rows of the metric table show each effect.
4. `63 − 21 = 42`: the last 21 origins target sessions on or after 2024-07-01.
5. Every forecast at origin t. A model that peeks at t+1 gets different inputs in the two runs;
   one that does not gets identical ones. It works because the folds are passed in, not derived
   from the data.

## 9. Explain it back

1. Why is `fit` given `y[:cutoff+1]` and not the whole series with a cutoff index?
2. Why can AIC choose p and q but not d?
3. Why does every model lose an origin when only HAR lacks the inputs for it?
4. Why is a forecast whose target falls in the holdout stored but not scored?
5. What would the truncation test miss?

## 10. Decisions waiting on you

| # | Decision | Default taken (2026-10-29, owner can override) |
|---|---|---|
| O-M2-1 | Make VIX9D, VIX3M, VIX6M, VIX1Y and VVIX protocol targets? That changes `protocol_hash` and needs a new freeze record. | No: secondary targets defined in `configs/models.yaml` |
| O-M2-2 | How `arima_log` chooses d (amended M2, review A). Measured on stationary AR(1) series at fold sample sizes, KPSS at 5% chose d = 1 on 52% (phi 0.98, n 2,570), 24% (0.98, 8,700), 76% (0.99, 2,570) and 52% (0.99, 8,700) of seeds (review A, 100 each; fresh seeds 47%, 28%, 70%, 50%). The first real run, under KPSS, chose d = 1 at all 97 VIX cutoffs, which is what KPSS does to a persistent stationary series. Alternatives: KPSS (`d_rule: kpss`), or d by an ADF-type test | **d = 0** (`d_rule: d0`); the random walk covers the unit-root case |
| O-M2-3 | The fixed window of `hist_mean_252` | 252 sessions |
| O-M2-4 | `min_train_sessions` for late-starting indices | 504 |
| O-M2-5 | `rv21` baselines here or in M3 | M3 |
| O-M2-6 | Move real results into the repo (pending data-policy confirmation) | Not done: real results stay under `$VOL_DATA_DIR` |
| O-M2-7 | Re-check L2 for the window target `rv21` in M3: M2's "the array ends at the cutoff" holds only because each value sits on the row of the date it is observed (review A) | Open: an L2 test of its own when `rv21` arrives |
