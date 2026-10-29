# Milestone 4: the LSTM

**Status:** milestone 4 of 6: `lstm_log`, an LSTM in PyTorch (CPU, `torch==2.14.1`, ADR 0002) over the
22 sessions `t-21..t` with HAR's regressors beside its last hidden state, on both tracks: `vix_level`
(VIX, h = 1, 5, 21) and `rv21` (h = 21). Same folds, same origins, same losses as M2/M3; a 5-seed
ensemble; early stopping on a purged validation slice (rules L3, L4); a refit every 4th cutoff (ADR
0001 O2, measured); the `python -m ivf.models lstm` step; the `forecast_member` table; notebook 04;
concept note 11. Dev period only: **no row on or after `holdout_start` is read**. No real-data number
is published.

**Purpose.** To explain how a neural forecaster is fitted inside a walk-forward protocol without
seeing the future (inputs, training pairs, the scaler, the early-stopping data), how its log forecast
becomes a scored mean, how the seeds combine, and what the refit schedule costs. Read in order: answer
[Predict before you look](#1-predict-before-you-look); read sections 2 to 7; compare with
[the answers](#8-answers-to-the-predict-first-questions); then [Explain it back](#9-explain-it-back).

**Licensing (ADR 0001 D2).** Numbers below are hand-worked or measured on **synthetic** data. For
real data this walkthrough records only the exit code, row counts, timings, and seed and epoch
statistics; the losses are measured locally and not published.

**Line numbers are a snapshot**, taken with `grep -n` when this was written.

**Companions.** [ADR 0001](../adr/0001-python-core-versioned-json-app.md) (L4, D5 seeds, O2 and O10
carry the M4 amendments), [ADR 0002](../adr/0002-compute-and-numeric-stack.md) (the measured fit
times), [data model](../architecture/data-model.md) (`fit`, `forecast`, `forecast_member`, amended
M4), concept note [11 an LSTM for volatility, and leakage](../concepts/11-lstm-for-volatility-and-leakage.md),
[milestone 3](03-garch-done-right.md).

## What review A found

An adversarial review (Opus) attacked eight claims of the first M4 report; the ledger is
[docs/validation/M4-review-A-lstm.md](../validation/M4-review-A-lstm.md). Every statement changed
because of it says "(amended M4, review A)".

| Claim | Verdict | Fix |
|---|---|---|
| C1 truncation bit for bit, including frozen folds | **test refuted (high)**: both truncation tests used `refit_every` 63, so no frozen fold was ever truncated; a frozen-fold look-ahead mutant passed all 21 tests | `refit_every` 126 in both tests, the frozen flag asserted, truncation inside the frozen fold; the mutant is killed (section 7.4) |
| C6 a frozen fold uses its training cutoff's network, scaler and `s2` | **tests refuted (medium)**: they checked metadata and counts only; "refit the scaler at the frozen cutoff" passed | review A's P6 as a test: `mu` and `s2` equal fold 0's network, stored scaler and `s2` by hand, exactly |
| C4 diagnosis of the rv21 mean-vs-median result | **refuted (medium)**: `s2` does not overstate the variance and the skew is small; `mu` is biased high | section 7.3 rewritten with the measured numbers; bias calibration is O-M4-7 for M5 |
| (unasked) "a 0.9% RMSE gap is inside one seed's wobble" | **refuted**: a per-row spread compared with an aggregate gap; all five members lose to HAR | sentence removed here and in notebook 04, which now prints each member's RMSE |
| C2 poisoning validation leaves the weights unchanged | scope: true at one epoch only | stated in section 6 and concept note 11 |
| C8 the timing table | the first-cutoff VIX row came from a run with the pre-D5 seed | marked, and re-timed with the committed seeds (section 7.1) |

The builder also corrected a label of the brief: the refit-cadence question is ADR 0001 **O2**; ADR
0001's **O10** is whether `lstm_log` is publishable (still open).

## 1. Predict before you look

1. At cutoff c = 1,000, with H = 21 and 252 validation sessions, which origins train the network and
   which validate it?
2. The scaler is fitted on training plus validation instead of training alone. Does that leak future
   data into a forecast made after c?
3. One network is trained at the cutoff of fold k and applied unchanged at folds k+1..k+3. Is that a
   leak, a handicap, or neither?
4. Five seeds are averaged. Should the ensemble beat the average member in RMSE? Always?
5. Given the network nests a HAR-type regression, should it beat `har_log` on the synthetic market,
   where ln VIX is an AR(1)?

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| The model | `src/ivf/models/lstm.py` | features (trailing), targets (log changes), `split` (L2 + L4), `Scaler` (L3), `Net`, `train_member` (one seed, early stopping), row-wise `predict`, `member_seed` (ADR 0001 D5), `LstmLog` (vix_level) and `LstmRv` (rv21), the refit schedule |
| Runner | `src/ivf/evaluation/walkforward.py` | one `fit` row per member (`_fit_rows`); `n_members` and each member's `yhat` on the forecast frame |
| Model interface | `src/ivf/models/baselines.py` | `Fit.members`, `Forecast.members` (both default `None`: M2/M3 models unchanged) |
| Registry | `configs/models.yaml`, `src/ivf/models/registry.py` | the `lstm_log` row (targets `vix_level`, `rv21`, milestone 4), the `lstm` block read strictly (`load_lstm_settings`) |
| Command | `src/ivf/models/cli.py` (`run_lstm`, `member_table`) | `python -m ivf.models [--synthetic] lstm [--workers N] [--tracks ...] [--print-metrics]`: both tracks, the same-rows check against the **stored** table, `forecast_member`, merged tables |
| Notebook | `notebooks/04_lstm.ipynb` (spec in `tools/build_notebooks.py`) | the slices by hand, both tracks beside the baselines, the seed spread, determinism |
| Tests | `tests/test_lstm.py` (22; amended M4, review A: was 21), `tests/test_notebooks.py` (+1, and 04 in the byte-equality check) | hand-worked features, targets, split and seeds; the four leakage doors as behaviours; determinism in and out of a worker; truncation bit for bit; refit schedule; holdout; the command |

**App.** No export is added: forecasts reach the app with the holdout run (M2's reason, unchanged).

## 3. Decisions

Each default below was taken on 2026-10-29; the owner can override any of them.

| # | Decision | Rejected | Would win if |
|---|---|---|---|
| D1 | **One model id, `lstm_log`, on both tracks** (as `rw`); `LstmLog` for `run_symbol`, `LstmRv` for `run_rv21` | A separate `lstm_rv` id | The owner wants one registry row per target |
| D2 | **LSTM(1 → 32) over `t-21..t`, its last state and HAR's regressors into one linear head.** With the LSTM part at zero it is a linear HAR-type model, so HAR is the floor it must clear | ADR 0002's benchmark shape (an LSTM on the window alone: it must relearn the HAR averages); stacked or attention models (ADR 0002: no CPU budget) | The plain LSTM matched HAR without the help, or a GPU pool were adopted |
| D3 | **The VIX track sees VIX only** (ln VIX window, HAR averages of ln VIX), the information set of `har_log` and `arima_log`; returns, rv5 and rv21 enter on the `rv21` track. The brief listed trailing rv among the inputs: on the VIX track it is left out on purpose, so a VIX-level comparison is about the model, not the data | Returns and rv on the VIX track (needs a second input path through `run_symbol`) | The owner wants the LSTM's best shot rather than a like-for-like comparison (O-M4-3) |
| D4 | **One network, three outputs (h = 1, 5, 21), direct**, with H = 21 for the purge at every h (≥ each h) | One network per horizon: 3× the training; the h = 1 network could use a 1-session purge | Per-horizon early stopping mattered (each output's best epoch differs) |
| D5 | **Predict the log change from an anchor** (`x_t`, or `ln rv21_t`), MSE on standardised targets; `mu = anchor + output`; `s2` = validation MSE of the members' mean `mu`; **`yhat = exp(mu + s2/2)`** (`plugin`, M2's mean), `yhat_naive = exp(mu)` | A Gaussian NLL head (a per-origin variance, but a known-unstable fit and a second output to trust); MSE in levels (dominated by spikes, unlike every log model); `native` `E[v] = exp(2mu + 2s2)` on `rv21` (QLIKE-consistent, but RMSE is primary and M2's rule is the mean) | M5 needs calibrated intervals or ranks by QLIKE |
| D6 | **L4 as written:** validation = the last 252 usable origins (`v0..c−H`), training `τ ≤ v0 − H`; the last training target is dated `v0` ([`lstm.py:153`](../../src/ivf/models/lstm.py#L153)) | A random validation split (validates on the past of the training rows); a gap of 0 (20 of 21 returns shared at h = 21); a 2H gap (one reading of L4's wording: costs 21 more recent origins and removes no extra overlap) | The owner reads L4 as a 2H gap |
| D7 | **The scaler on the training slice only** (L3): each sequence channel, static column and target column ([`lstm.py:348`](../../src/ivf/models/lstm.py#L348)) | Training + validation (moves the training inputs by the validation block's mean, concept 02) | — |
| D8 | **5 members; the forecast averages their `mu` in log space;** each member's own `exp(mu_m + s2_m/2)` stored in `forecast_member`. Seeds from ADR 0001 D5 ([`lstm.py:105`](../../src/ivf/models/lstm.py#L105)) | One seed (one draw of the initialisation); the median of members (no gain at 5); averaging `yhat` in levels (inconsistent with `s2` measured on log errors); a seed list in `models.yaml` (the first draft; ADR 0001 D5 had already fixed the scheme) | — |
| D9 | **Refit every 252 sessions (every 4th cutoff), frozen network in between** (ADR 0001 O2; the brief called this O10, which in ADR 0001 is the publication question). The frozen network keeps its scaler and `s2`; `params.refit = false` | A retrain at every cutoff: about 2.8 h of single-thread training (derived, section 7.1), ≈ 34 min only if 5 cores are idle; every 8th cutoff (cheaper, a 2-year-old network) | The box is idle and the owner wants every cutoff (`lstm.refit_every: 63`; about 4× the training time) |
| D10 | **Determinism:** 1 thread per member, deterministic algorithms, `manual_seed` + a seeded CPU `Generator`; members train in a spawn pool (`--workers`, default 5) with identical bytes; **`predict` one row per call** ([`lstm.py:280`](../../src/ivf/models/lstm.py#L280)) | Batched prediction: outputs moved by up to 1.2e-8 log points with the batch size (measured, 13 of 63 rows at hidden 4, 41 at hidden 32), so the truncation test could only use a tolerance; several threads per member (ADR 0002: changes the low bits) | Prediction time mattered (row-wise costs ≈ 0.1 s per 63 origins and member) |
| D11 | **Same rows against the stored table:** the `lstm` step forecasts with the shared runner, then checks `check_same_rows` on the run's stored forecasts plus its own, with `registry.models_for(·, 4)`; it refuses (exit 1) if any expected model is missing and writes nothing ([`cli.py:419`](../../src/ivf/models/cli.py#L419)) | Re-running every M2/M3 model inside `lstm` (≈ 3 min more, duplicates `garch`) | — |
| D12 | **Hyper-parameters fixed before scoring** (rule L6): hidden 32, batch 64, Adam 1e-3, at most 50 epochs, patience 5, from ADR 0002's benchmark and round numbers | Any search (every variant is a trial, ADR 0001 dev purge) | — |

## 4. Code walkthrough by `file:line`

### 4.1 `src/ivf/models/lstm.py`

- [`lstm.py:105`](../../src/ivf/models/lstm.py#L105): a member's seed, `SeedSequence(protocol.seed, spawn_key=(key32("lstm_log"), yyyymmdd, m))`;
  `key32` [`lstm.py:100`](../../src/ivf/models/lstm.py#L100) never uses Python's salted `hash`.
- [`lstm.py:124`](../../src/ivf/models/lstm.py#L124) and [`lstm.py:132`](../../src/ivf/models/lstm.py#L132): the features, each a trailing window
  ending at t (`lagged` [`lstm.py:114`](../../src/ivf/models/lstm.py#L114), `build.trailing`).
- [`lstm.py:142`](../../src/ivf/models/lstm.py#L142): targets `ln level[t+h] − anchor[t]`, NaN past the array, so a pair whose target is
  after the cutoff does not exist in the fit's data.
- [`lstm.py:153`](../../src/ivf/models/lstm.py#L153): `split`, the heart of L2 and L4: `tau_max = c − H` (line 160) and
  `last_train = v0 − H` (line 162).
- [`lstm.py:169`](../../src/ivf/models/lstm.py#L169): the `Scaler`, fitted at [`lstm.py:348`](../../src/ivf/models/lstm.py#L348) on `seq[train]` only.
- [`lstm.py:197`](../../src/ivf/models/lstm.py#L197): the network; [`lstm.py:211`](../../src/ivf/models/lstm.py#L211): one thread and
  deterministic algorithms for the block, the caller's settings restored after.
- [`lstm.py:235`](../../src/ivf/models/lstm.py#L235): one member: seed (line 242), shuffled batches, a validation pass per epoch,
  the best epoch's weights kept, stop after `patience` epochs without a new best (line 269).
- [`lstm.py:280`](../../src/ivf/models/lstm.py#L280): `predict`, row by row (D10).
- [`lstm.py:327`](../../src/ivf/models/lstm.py#L327): the refit schedule: the previous network, unchanged, while fewer than
  `refit_every` sessions have passed since its training cutoff.
- [`lstm.py:337`](../../src/ivf/models/lstm.py#L337): one training cutoff: slices, complete rows, scaler, members, `s2` of the
  members' mean on validation (line 360); `train_last_target = last validation origin + H = c`.
- [`lstm.py:385`](../../src/ivf/models/lstm.py#L385): the forecast: the members' mean `mu`, `lognormal_forecast(mu, s2)` (M2's
  back-transform, line 387), each member's own `yhat`.

### 4.2 The runner, the command and the registry

- [`walkforward.py:289`](../../src/ivf/evaluation/walkforward.py#L289): one `fit` row per member, `member` 1..K with its `seed` and
  `epochs_run`; [`walkforward.py:327`](../../src/ivf/evaluation/walkforward.py#L327): `n_members`; line 333 carries the members'
  `yhat` to the command (not a stored forecast column).
- The runner's L2 assertions are unchanged and apply to the LSTM: `train_last_target ≤ c` on both
  tracks and `train_last_origin + 21 ≤ c` on `rv21`.
- [`cli.py:352`](../../src/ivf/models/cli.py#L352): `run_lstm`: the M2 data check, `pre_holdout_inputs`, both tracks, the
  same-rows check against the stored table (line 419), `forecast_member` from
  [`cli.py:319`](../../src/ivf/models/cli.py#L319), merged writes (line 428).
- [`registry.py:135`](../../src/ivf/models/registry.py#L135): the `lstm` block, unknown keys refused, `window ≤ 22` (a longer
  window would drop origins the other models forecast).

## 5. Data flow

```
interim ──check.frozen_check──► 𝒯 ──fold_schedule──► 97 dev folds ──pre_holdout_inputs (date < holdout_start)
   │
   ├─ vix_level (VIX): x = ln VIX ──► seq x_{t-21..t}; static [x_t, xw_t, xm_t]; target x_{t+h} − x_t (h = 1, 5, 21)
   └─ rv21: r = Δ ln ^GSPC, VIX ──► seq 100·r_{t-21..t}; static [ln VIX_t, ln rv5_t, ln rv21_t]; target ln rv21_{t+21} − ln rv21_t
          │
          per training cutoff c (every 252 sessions): usable τ ≤ c − 21 ─► validation last 252 ─► training τ ≤ v0 − 21
          scaler(training) ─► 5 members (seed = D5(c, m)), early stopping on validation ─► s2 = val MSE of mean mu
          per fold: forecast at t ∈ [c_k, c_k + 62] with data ≤ t: mu = mean_m mu_m; yhat = exp(mu + s2/2)
          │
          same rows vs the stored table (registry.models_for(·, 4)) ──► metric ──► merged into processed/run_id=<id>/
                                                                       └─► forecast_member (one row per seed)
```

The architecture spec's `models` and `processed` labels were updated (M4 amendment); the pipeline's
shape did not change: one more step after `garch`, writing into the same run.

## 6. Pitfalls and their guards

| Pitfall | Guard | Test |
|---|---|---|
| A feature reaches past t (L1) | Trailing windows only; row-wise `predict` | `test_vix_features_by_hand`, `test_rv_features_by_hand`, `test_truncation_forecasts_at_t_do_not_depend_on_data_after_t` (4 origins, **bit for bit**), `test_rv21_truncation_and_the_window_target_rule`, `test_poisoned_closes_after_the_cutoff_change_no_fit_and_no_forecast_at_the_cutoff` |
| A training or validation target after the cutoff (L2; the window target) | `split` (`τ ≤ c − H`); NaN targets past the array; the runner's assertions | `test_split_by_hand`, `test_log_targets_by_hand`, the poisoned-future test, `test_rv21_training_never_reads_a_return_after_the_first_validation_origin` (`train_last_origin = c − 21`) |
| Validation overlapping training (L4) | `last_train = v0 − H` | `test_values_only_validation_reads_change_neither_the_scaler_nor_the_weights_and_the_last_training_target_does`, its rv21 twin: poison after `v0` → same weights bit for bit; poison `v0` itself → different weights. **Scope (amended M4, review A):** bit-identical weights hold at **one epoch** only; with early stopping, the validation values legitimately choose which epoch is kept, so poisoning them can change the kept weights without any leak |
| A frozen fold leaking, or re-fitting its scaler or `s2` (amended M4, review A) | `_frozen` copies the trained fit; `_mu` uses the trained network's scaler | the two truncation tests now truncate inside a **frozen** fold (`refit_every` 126, the refit flag asserted false); `test_a_frozen_fold_applies_the_trained_network_scaler_and_s2_exactly` (review A's P6) |
| The scaler sees validation (L3) | `Scaler.fit(seq[train], …)` | `test_the_scaler_is_fitted_on_the_training_slice_only` (means by hand) and the poison test above |
| Early stopping keeps the last, not the best | `best_state` | `test_early_stopping_keeps_the_best_epoch_and_stops_after_patience` (validation MSE of the kept weights = the recorded minimum; stops at best + patience) |
| The wrong back-transform or seed mean | `lognormal_forecast` on the members' mean `mu` | `test_back_transform_and_seed_combination` (`s2` re-derived from the forecast path) |
| Non-deterministic training; thread count leaking out | `torch_state`; seeds from D5 | `test_determinism_same_inputs_same_bytes_and_the_thread_count_is_restored`, `test_a_worker_pool_gives_the_same_bytes_as_this_process`, `test_member_seeds_follow_adr_0001_d5` (reproduces ADR 0001's measured draw 0.45037008942388645) |
| Reading the holdout | `pre_holdout_inputs`; the runner's refusals | `test_poisoning_every_holdout_row_changes_nothing` (both tracks; 21 of 63 origins of the last fold unscored) |
| A frozen network forecasting fewer rows | The runner calls `forecast` at every fold | `test_refit_schedule_freezes_the_network_between_training_cutoffs` (trained, frozen, trained; same rows as `rw`) |
| Scoring the LSTM on other rows than M2/M3 | The same-rows check against the stored table | `test_lstm_command_refuses_without_the_other_models_then_merges_on_the_same_rows` (exit 3 with no run, exit 1 with models missing, then exit 0) |

## 7. Measured results

### 7.1 The training-time measurement behind O2 (D9)

Measured on 2026-10-29 with a scratch timing script (one member, `LstmLog`/`LstmRv.fit` on the
synthetic market, which has the real calendar and sample sizes), the committed settings, **one
thread**, while another project's 20-process pool kept the load average at 13 to 16 of 20 cores:

| Training cutoff | Track | Training origins | Epochs run (best) | s / epoch | One member |
|---|---|---:|---:|---:|---:|
| first dev cutoff (index 2,578) | vix_level | 2,265 | 9 (4)* | 0.442 | 4.0 s |
| | rv21 | 2,264 | 25 (20) | 0.294 | 7.3 s |
| fold 48 (index 5,602) | vix_level | 5,289 | 7 (2) | 0.782 | 5.5 s |
| | rv21 | 5,288 | 17 (12) | 0.652 | 11.1 s |
| last dev cutoff (index 8,626) | vix_level | 8,313 | 10 (5) | 1.252 | 12.5 s |
| | rv21 | 8,312 | 21 (16) | 1.036 | 21.8 s |

*Derived from these:* a training cutoff costs about (7.3 + 13.4) s × 5 seeds ≈ 104 s of one-thread
training, so a retrain at all 97 dev cutoffs is about 2.8 h on one core and about 34 min with 5 idle
cores, too close to the brief's 45-minute budget on a box another project was loading. **Default
(O-M4-1 = ADR 0001 O2): retrain every 4th cutoff** (25 trainings per track), which the runs below
measured. Row-wise prediction costs about 0.1 s per 63 origins and member (measured), about 2 min
per run.

| Run (5 workers × 1 thread, refit every 252 sessions) | Wall time | Training (all members, pool wall) | Peak RSS |
|---|---|---|---|
| Synthetic, `--synthetic lstm --print-metrics` (load average 100 to 300 from the other project) | 15 min 48 s | vix_level 383.1 s, rv21 518.0 s | 803,824 KB |
| Real, `lstm` (load average falling from about 90 to 5 during the run) | 13 min 25 s | vix_level 561.5 s, rv21 217.2 s | 849,872 KB |

\* The table was measured with torch seed 1, before the seeds moved to ADR 0001 D5 (D8), so its
epoch counts are not the committed run's (amended M4, review A). Re-timed with the committed seeds:
member 1 at the first dev cutoff runs 7 epochs (best 2) in 4.2 s (load average 27). The derived
totals use the per-epoch times and are unchanged in kind.

Under a lighter load the same run should be faster; it was not re-measured on an idle box.

### 7.2 The real run (counts and timings only, D2)

Two runs on 2026-10-29 from the repo root, uncommitted M4 code on `8318a1a`, each under
`/usr/bin/time -v`, while another project's process pool was loading the box (load average 260 at
the start, 5 at the end):

| Step | Result |
|---|---|
| `uv run python -m ivf.models garch` (to put every M2/M3 row in this code state's run) | **exit 0**; wall 5 min 44.8 s; peak RSS 517,424 KB; 146,664 forecast rows, 1,164 fit rows, 0 with `train_last_target_date` after the cutoff (M3's counts, unchanged) |
| `uv run python -m ivf.models lstm` | **exit 0**; wall 13 min 25.3 s; peak RSS 849,872 KB; printed `no row on or after it is read` |
| vix_level track | 97 folds fitted, **25 trained, 72 frozen**, 6,111 origins, 0 not forecastable, 18,333 forecast rows, 5 members; training 561.5 s (pool wall, all members), fit + forecast 577.8 s |
| rv21 track | 97 folds, 25 trained, 72 frozen, 6,111 origins, 0 not forecastable, 6,111 forecast rows, 5 members; training 217.2 s, fit + forecast 223.0 s |
| Epochs (trained folds, 125 members per track) | vix_level: epochs run min/median/max 6/14/50 (at least one member reached `max_epochs` without stopping), best epoch 1/9/46; rv21: 6/19/41, best 1/14/36 |
| Totals | 24,444 forecast rows, 24,396 scored, 122,220 `forecast_member` rows, 970 `fit` rows (97 × 5 × 2), **0 with `train_last_target_date` after the cutoff**; the same-rows check passed against the stored table for every milestone ≤ 4 model on `vix_level` (7 models) and `rv21` (7 models) |
| Holdout | the run reads `pre_holdout_inputs` only |

The tables were merged into the run's partition under `$VOL_DATA_DIR`; the `run_id`, the losses and
the seed spread stay local (D2).

### 7.3 Synthetic data (publishable)

`uv run python -m ivf.models --synthetic garch --print-metrics` (2 min 47 s), then
`uv run python -m ivf.models --synthetic lstm --print-metrics`, 2026-10-29: exit 0 both; 24,444
LSTM forecast rows (18,333 + 6,111), 24,396 scored, 122,220 member rows, 970 fit rows (97 folds ×
5 members × 2 tracks), **0 with `train_last_target_date` after the cutoff**, same rows as every
milestone ≤ 4 model on both targets. 25 trained and 72 frozen folds per track. Epochs run (trained
folds, all members): vix_level min/median/max 6/10/29, best epoch 1/5/24; rv21 6/15/46, best 1/10/41.

Losses on the same scored rows (`forecast = mean`, the scored lognormal mean; n = 6,110 / 6,106 / 6,090
at h = 1 / 5 / 21 on VIX, 6,090 on rv21):

| Target, h | `lstm_log` | `har_log` / `har_rv` | best M2/M3 model | `rw` |
|---|---|---|---|---|
| vix_level 1, RMSE | 1.1815 | 1.1803 | `arima_garch_log` 1.1798 | 1.1857 |
| vix_level 5, RMSE | 2.5482 | 2.5389 | `arima_garch_log` 2.5373 | 2.5992 |
| vix_level 21, RMSE | 4.5525 | 4.5133 | `arima_garch_log` 4.4981 | 4.9454 |
| vix_level 21, QLIKE | 0.1067 | 0.1050 | `arima_garch_log` 0.1040 | 0.1293 |
| rv21 21, RMSE | 4.0589 | 4.0238 | `egarch` 3.9957 | 4.5260 |
| rv21 21, QLIKE | 0.1524 | 0.1409 | `garch` 0.1396 | 0.2164 |
| rv21 21, MAE | **3.0812** (lowest) | 3.1540 | `egarch` 3.1006 | 3.3828 |

What this says (synthetic only; the real losses stay local):

- **The LSTM beats the random walk everywhere and loses to HAR everywhere in RMSE**, by 0.1% (h = 1)
  to 0.9% (h = 21) on VIX and 0.9% on rv21. On this market ln VIX *is* an AR(1), so HAR's regressors
  carry everything knowable; the network adds estimation noise. That was the expected answer to
  question 5, and it is the honest null result the comparison was built to allow (D3, D12).
- **The loss to HAR holds for every seed** (amended M4, review A). Each member's own RMSE: rv21
  4.0747 to 4.1493, VIX h = 21 4.5408 to 4.5926, all above HAR's 4.0238 and 4.5133. The 5-seed mean
  beats every member on rv21 (4.0589) but not at VIX h = 21 (4.5525 against the best member's 4.5408):
  averaging is guaranteed to help the squared error of `mu` in logs, not the RMSE of `yhat`. The per-forecast spread across members (sd / yhat: VIX median
  0.37%, max 9.3%; rv21 median 2.6%, max 20.3%) is a per-row quantity and says nothing by itself
  about an aggregate RMSE gap; the first version compared the two and was wrong.
- **On rv21 the lognormal correction hurts RMSE:** the median `exp(mu)` scores 4.0367 against the
  mean's 4.0589 (on VIX the mean wins at every h, as M2 found). The cause is **`mu` biased high, not
  an inflated `s2`** (amended M4, review A; the first version guessed skewed errors or an overstated
  `s2`, and both guesses were wrong). Measured on the 6,090 scored rows: the mean log error
  `ln y − mu` is −0.0206; `s2` *understates* the realised log MSE (mean `s2` 0.0659 against 0.0732;
  per training cutoff, the median ratio is 0.865 and 15 of 25 are below 1); skew is +0.178; the bias is
  worst on high forecasts (mean log error +0.020, −0.035, −0.047 in the low, middle and high terciles
  of `yhat`); and the RMSE-optimal multiplier on `exp(mu)` is k* = 1.0046 against the applied
  `exp(s2/2)` ≈ 1.034. The scored forecast stays the pre-registered mean (rule L6); **bias calibration
  of the LSTM (e.g. Mincer-Zarnowitz on `mu`) is an M5 item** (O-M4-7).
- QLIKE on rv21 is worst among the fitted models, as expected for a `plugin` forecast scored on
  variance (biased low by `Var_t(y)`, ADR 0001 4.3), while its MAE and MAPE are the lowest.

### 7.4 Mutation checks

Every new test was run against deliberate bugs in a scratch copy of the repo
(`<session scratchpad>/m4/mut`, `tests/test_lstm.py` and the notebook test, `-x`); each
row names the first test that failed. **33 of 33 mutants killed** (29 in the first round, 4 after review A).

| Mutant | Killed by |
|---|---|
| validation gap H − 1 (`last_train = v0 − H + 1`); `tau_max = c − H + 1` | `test_split_by_hand` |
| scaler fitted on training + validation | `test_the_scaler_is_fitted_on_the_training_slice_only` |
| the ln VIX sequence one session ahead | `test_vix_features_by_hand` (and, run without `-x`, the poisoned-future, truncation, refit, holdout and command tests) |
| rv5 one session ahead | `test_rv21_truncation_and_the_window_target_rule` |
| targets one session late | `test_log_targets_by_hand` |
| no lognormal correction; `s2` of member 1; the ensemble = member 1; every member seeded alike (in process) | `test_back_transform_and_seed_combination` |
| every member seeded alike (pool) | `test_a_worker_pool_gives_the_same_bytes_as_this_process` |
| no `manual_seed` | `test_values_only_validation_reads_…` (two fits differ) |
| thread count not restored | `test_determinism_same_inputs_same_bytes_and_the_thread_count_is_restored` |
| refit `≥` → `>`; never frozen | `test_refit_schedule_freezes_the_network_between_training_cutoffs` |
| last epoch kept, not best; patience off by one | `test_early_stopping_keeps_the_best_epoch_and_stops_after_patience` |
| batched `predict` | `test_truncation_forecasts_at_t_do_not_depend_on_data_after_t` |
| one `fit` row instead of one per member (twice: before and after the seed change); seed keyed by the first date | `test_one_fit_row_per_member_and_n_members_on_every_forecast` |
| `key32` little-endian; member left out of the spawn key | `test_member_seeds_follow_adr_0001_d5` |
| members numbered from 0; the command's same-rows check off; the command expecting the milestone-3 list | `test_lstm_command_refuses_without_the_other_models_then_merges_on_the_same_rows` |
| `pre_holdout_inputs` returning holdout rows | the poisoned-future test (the runner refuses the rows) |
| the notebook printing the wrong gap | `test_m4_notebook_runs_top_to_bottom_on_synthetic_data` |
| *(review A)* a frozen fold's anchor one session ahead, VIX (review A's M1, which survived the first truncation test) | `test_truncation_forecasts_at_t_do_not_depend_on_data_after_t` |
| *(review A)* the same on rv21 | `test_rv21_truncation_and_the_window_target_rule` |
| *(review A)* a frozen fold refits the scaler at its own cutoff (review A's M2, which survived); a frozen fold halves `s2` | `test_a_frozen_fold_applies_the_trained_network_scaler_and_s2_exactly` |

One planned test was dropped: monkeypatching `split` one session later did **not** trip the runner's
window assertion, because the extra validation origin's target is past the array (NaN) and is
dropped by the complete-rows filter. The data a fit receives is the real guard; the hand-worked
`split` test kills that mutant instead.

### 7.5 Gates

Run on 2026-10-29 after the review A fixes, from the repo root; the printed last lines:

```
$ uv run pytest -q
875 passed in 113.93s (0:01:53)
$ uv run python tools/build_notebooks.py --check
ok: 5 notebook(s) in .../notebooks match their specs
$ uv run python tools/export_app_data.py --mode synthetic --check
ok: 1 file(s) in .../app/public/data/synthetic match a fresh synthetic export
$ uv lock --check
Resolved 142 packages in 0.77ms
$ uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out --check --no-png
ok: 5 spec(s) up to date in docs/architecture/out
$ uv run python tools/licensing/scan_real_rows.py
clean: 0 lines flagged in 241 text file(s) against 7 raw snapshot(s)
```

`tests/test_lstm.py` alone: 22 passed in about 29 s (tiny networks, one or two seeds, one to twelve
epochs). `git status --porcelain --ignored`: the M4 files only, plus ignored caches; nothing under
the data root is in the repo.

## 8. Answers to the predict-first questions

1. Usable origins τ ≤ 979 (target dated ≤ 1,000); validation 728..979; training 0..707, whose last
   target (707 + 21) is dated 728, the first validation origin.
2. Not into a forecast after c: every row of the validation slice is dated ≤ c. It does leak into
   model selection: the training inputs are standardised with the validation block's mean, so the
   early-stopping loss is measured on data the training already "saw" through the scaler (L3).
3. A handicap, not a leak: the frozen network was trained on data up to an earlier cutoff, and its
   inputs at each origin are data ≤ t. Its weights are up to three folds (a year) old.
4. In expected squared error of `mu`, yes, never worse than the average member (the error of a mean
   is at most the mean of the errors, by convexity). Not for every loss and every row: the scored
   `yhat` adds `s2/2`, and RMSE in vol points is not the log MSE. See section 7.3 for the measured
   spread.
5. Not necessarily: on an AR(1) in logs the HAR regressors already carry what can be known, and the
   LSTM adds variance (seeds, early stopping on a short slice). Section 7.3 has the answer measured.

## 9. Explain it back

1. Draw the time line of one training cutoff: usable origins, validation, training, the gap, where
   the last training target and the last validation target sit.
2. Why does the head see HAR's regressors, and what does that make the LSTM's job?
3. Name the four doors through which a neural forecaster sees the future (concept 11), and the test
   that shuts each.
4. Why is `predict` row by row, and what did a batch change?
5. Why is the refit schedule a handicap and not a leak?

## 10. Decisions waiting on you

| # | Question | Default (taken 2026-10-29, owner can override) |
|---|---|---|
| O-M4-1 (= ADR 0001 O2; the brief's "O10") | Refit cadence of the LSTM | Every 4th cutoff (`refit_every: 252`), frozen between; measured basis in 7.1 |
| O-M4-2 (= ADR 0001 O10) | Is `lstm_log` in the publication set P? | Open: the `p_tie.py` re-probe was not run; the rows stay local |
| O-M4-3 | Returns and rv as inputs on the VIX track (D3) | No: VIX only, like `har_log` |
| O-M4-4 | The LSTM on the secondary indices (VIX9D … VVIX) | No: VIX only (cost, like O-M3-2) |
| O-M4-5 | The L4 gap: H (taken) or 2H | H: the last training target is dated at the first validation origin |
| O-M4-6 | Move real results into the repo (pending data-policy confirmation) | Not done: real results stay under `$VOL_DATA_DIR` |
| O-M4-7 (review A) | Bias-calibrate the LSTM's `mu` (rv21: biased high, worst on high forecasts) | Not in M4: the mean forecast is pre-registered (L6); an M5 calibration item |
