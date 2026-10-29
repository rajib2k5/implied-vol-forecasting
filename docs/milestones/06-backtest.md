# Milestone 6: the VXX/SVXY backtest

**Status (2026-10-29): built, dev run done, review A fixed (not re-reviewed).** The look at the
holdout is **not** taken: the path (preflight, record, holdout refits, M5's holdout report, backtest)
is built and tested end to end on synthetic data, and the owner runs it after this milestone is
pushed. **`holdout_looks` is 1**: the M6 build printed holdout values before its settings were written
(section 7.3), so the planned look is the second exposure and is reported with that caveat.

Theory: [14, VIX ETPs and segments](../concepts/14-vix-etps-roll-yield-and-segments.md) and
[15, PSR, DSR and trial counting](../concepts/15-psr-dsr-and-trial-counting.md).

## What review A found

Adversarial review A (Opus), 2026-10-29; ledger in `docs/validation/M6-review-A-backtest.md`.

| Finding | Verdict | Fix |
|---|---|---|
| The look record was written before checking that holdout forecasts exist, so a doomed run spent the look | **refuted** (high) | a preflight of every precondition before the record; a failed preflight exits 2 and writes nothing (tested for each precondition) |
| No command wrote forecasts at holdout origins | gap | `ivf.models.holdout`, inside the guarded look: specs frozen, refits continue at every holdout cutoff |
| Holdout values printed during the build, before the M6 settings | **breach** | recorded in ADR 0001 4.6 and section 7.3; `holdout_looks` 0 -> 1; the look carries a caveat |
| VXX: 22 days with open = high = low = close (2018-03-16 .. 2018-07-31) and one volume-0 day in the usable window | data quality | stale-print check and rule D12: VXX's usable segment starts at 2018-08-01 |
| Mutants R2 (a missing held return as 0), R3 (the VIX of t+1 at origin t), R4 (the forecast cut dropped) survived | **untested** | three tests; R4 needed holdout-origin rows in the test store to be observable |
| "25 mutants, all killed": the harness had a syntax error, a no-op mutant and a mutant without its target | **refuted** | harness rewritten; re-measured (section 6) |
| Real values in two docstrings/notes; data-model said `price_daily`; O-M6-2 | docs | removed; `tradable_daily`; O-M6-2 closed (typeset eq. 2 shows sqrt(T - 1), (g4 - 1)/4, Pearson) |

## What review B found (the one-look path)

Adversarial review B (Opus), 2026-10-29, narrow: the one-look path. Ledger: `docs/validation/M6-review-B-holdout-path.md`.

| Finding | Verdict | Fix |
|---|---|---|
| The holdout segment checks ran after the record: synthetic VXX x4 from 2024-07-24 passed the preflight, spent the look and failed in the backtest stage | **refuted** (high) | the segment checks to `data_end` are in the preflight, problems named without values; tested with that case |
| No way to check the preconditions without risking the look | gap | `--preflight-only` (real and synthetic): prints problems or "preflight ok", writes nothing |
| A failed stage made the look unrecoverable | gap | `--resume` at the same commit, protocol, config and inputs, from the failed stage; attempts appended to the record (D14) |
| A holdout fit could silently fall back to an older fit | **untested** | `check_no_fallback` fails the stage; tested with an injected fit failure, then resumed |
| No caveat in the holdout metas; "synthetic holdout path" printed for any data; stale docstrings | labels | caveat and both protocol hashes in `report_meta.json` and `backtest_meta.json`; the line follows `data_mode`; docstrings fixed |
| Mutant M1 (LSTM warm-up `>=` -> `>`) survived | **untested** | `lstm_warmup` and its test |
| Warm starts restarted at the last dev fold: ARIMA holdout forecasts moved by up to 0.03 vol points against the full chain | **refuted** | chain from the first dev fold for every non-LSTM model, with the dev-reproduction check (D15) |
| The dev run's protocol was never compared with the current one | gap | accepted only if it differs in `holdout_looks` alone (D16) |

## 1. Predict before you look

Written before the first real run (PR/FAQ H4): after costs, no forecast-driven rule will have a DSR
distinguishable from zero. Short volatility (SVXY) earns the roll most days, so "always long SVXY" will
be hard to beat on Sharpe and will have the worst drawdowns; "always long VXX" will lose most of its
value. The rule's best hope is to stand aside (or go long VXX) before spikes.

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| Config | `configs/backtest.yaml` | the pre-registered rules, costs, model-choice rule, splits to check, trial-log location, holdout rule |
| Segments | `configs/instruments.yaml`, `src/ivf/backtest/segments.py` | VXX and SVXY as `tradable`; segments with leverage; membership of a return; the data checks |
| Data | `src/ivf/backtest/data.py` | `interim/tradable_daily.parquet` from the raw snapshots; the synthetic VXX/SVXY |
| Engine | `src/ivf/backtest/engine.py` | signal at t, fill at the close of t+1, earn t+1 -> t+2 (L7); costs per side; exact net return |
| Strategy | `src/ivf/backtest/strategy.py` | config loader, dev-only model choice, band signal, the period window, trial log, summary, holdout verdict |
| Metrics | `src/ivf/backtest/metrics.py`, `sharpe.py` | return, vol, Sharpe, Sortino, Calmar, drawdown, turnover, hit rate, PSR, DSR (`sharpe.py` copied, MIT) |
| Command | `src/ivf/backtest/cli.py` | `python -m ivf.backtest [--synthetic] prepare` and `run [--period dev\|holdout] [--one-look]` |
| Notebook | `notebooks/06_backtest.ipynb` (spec in `tools/build_notebooks.py`) | synthetic only: the book by hand, segments, a dev backtest, the deflation |
| Tests | `tests/test_backtest.py` (34 after review B), `tests/test_notebooks.py` (+1, and 06 in the byte check) | listed in section 6 |

Changed outside the package: `ivf.data` `build-interim` now passes only `role: underlying` tickers to
`price_daily` (`src/ivf/data/cli.py:85`), so the tradables move no M1-M5 table or run_id;
`report.run_report` takes `one_look` (`src/ivf/evaluation/report.py`), passed only by the backtest's
guarded look; three registry tests list VXX and SVXY explicitly.

## 3. Decisions

Each default below was taken on 2026-10-29; the owner can override any of them.

| # | Decision | Rejected alternative | Would win if |
|---|---|---|---|
| D1 | **Primary rule: rv21 forecast minus VIX, band 2 vol points.** Long VXX above +2, SVXY below -2, else cash (PR/FAQ Q3). The rv21 version is the variance risk premium; futures already price the expected VIX change (concept 14). | VIX-level forecast minus VIX as the primary | VIX futures did not already price expected VIX changes. It is kept as trial 5 |
| D2 | **Model per track by a rule fixed before any run:** lowest dev loss in M5's `metric` (QLIKE on rv21, RMSE on vix_level, forecast `mean`, scheme `all`), ties alphabetical; `vix_implied` (rv21) and `rw` (vix_level) excluded because forecast minus VIX is then identically 0. The run prints the ids; if `rw` wins rv21 it says so. | Choosing the model by backtest Sharpe | Never: that is selection on the outcome |
| D3 | **Timing L7:** fill at the close of t+1, first return t+1 -> t+2; `lag_sessions < 2` is refused by the loader | Fill at the next open | An open price series were in the repo with the same provenance; the gap risk overnight is part of the trade either way |
| D4 | **Costs:** 4 bp half-spread + 1 bp commission = 5 bp per side on \|change in weight\|; a switch is 2 sides; sensitivity 0, 5, 10, 20 bp; net = (1 - c)(1 + gross) - 1 | gross - cost (first order) | Never needed: the exact form is as simple |
| D5 | **Cash earns 0.** No rate series is in the repo; Sharpe is of the raw return | A T-bill series | One is added with provenance; then `cash_rate_annual` would be a series, not a constant |
| D6 | **Segments:** VXX +1x from 2018-08-01 (D12); SVXY -1x to the close of 2018-02-27 (unusable) and -0.5x from that close (usable; its open end from the issuer's fund page, not from data). Dev window: returns 2018-08-02 to 2024-06-28 (was 2018-02-28 before D12) | Rebuilding -0.5x before 2018 from index returns | VXX history before 2018 were in scope (it is not on Yahoo) |
| D7 | **Exposure check is a gate on usable segments only;** on unusable ones it is reported. Amended after the first real run, which exited 1 on the unusable -1x segment (section 7.3) | A gate on every segment | The -1x segment were used |
| D8 | **DSR:** E[SR] = 0, V = max(cross-trial variance, 1/T), N = distinct trials in the log; `dsr_paper_var` also kept | V = cross-trial variance only (the paper) | Trials were many and independent, so their spread were a fair V |
| D9 | **Trials:** 5 pre-registered forecast strategies (bands 0, 1, 2, 4 on rv21; band 2 on vix_level); benchmarks and cost levels are not trials; the holdout adds none | Counting each cost level as a trial | The default cost had been chosen after looking |
| D10 | **Holdout rule (confirmatory, pre-registered):** the primary "held up" if its holdout Sharpe at the default cost is above 0 and above both always-long benchmarks'. PSR and DSR there are descriptive | PSR >= 0.95 on the holdout | The holdout were long enough for that test to have power (it is about 2.3 years) |
| D11 | **The one look is one guarded command** (amended, review A): real `--period holdout` needs `--one-look`; then preflight (every precondition, nothing written on failure) -> create-only record (**the look is spent from here**) -> holdout refits and forecasts of every dev-store model -> M5's holdout report -> the backtest. The record's `status` says how far it got (`complete`, `failed at <stage>`, or `running <stage>` after a hard crash) | The first version: record first, then a run that could not succeed | Never: a precondition that can fail must fail before the look is spent |
| D12 | **Stale prints** (added after the first dev run, review A): a session with open = high = low = close, or volume 0, is stale; a usable segment holds none on dev data, so VXX's usable segment starts at the first close after its last one (2018-08-01). Holdout stale prints are counted and reported, never move the window | Stale days untradable (forced cash) | Stale prints were scattered through the sample, not a start-up cluster |
| D14 | **Resume** (review B): after `failed at <stage>`, `--resume` (+ `--one-look` on real data) continues from that stage, reusing the stored output of the completed stages, only at the look's own commit, protocol, backtest config and input snapshots. Still the one look: the same frozen code on the same data. A `running <stage>` record (hard crash) is resumed only after the owner marks it failed by hand | No resume: a fault after the record ends the project's holdout | Never preferable: a resume cannot change any frozen choice |
| D15 | **Warm starts from the first dev fold** (review B): every non-LSTM model re-runs the whole dev chain before the holdout (about +200 s on the full synthetic set, review B's measurement), and the recomputed dev forecasts must equal the stored ones within 1e-12 vol points before any holdout forecast is used. The LSTM restarts at its last dev retrain (a full chain costs hours) | Restart at the last dev fold | Warm starts had no effect on the optimum (review B measured up to 0.03 vol points for ARIMA) |
| D16 | **Dev-run protocol** (review B): the run's `protocol_hash` must equal the current protocol re-hashed with some earlier `holdout_looks` (0..current); anything else is refused. Both hashes are in every meta | Ignoring the hash | — |
| D13 | **`holdout_looks` 1** (review A's rule): the build printed holdout values before every M6 setting was written; `holdout_looks_before_look: {real: 1}` and `holdout_caveat` in the config; the look refuses if the protocol says otherwise | Keeping 0 and a footnote | No setting had been written after the print |

## 4. Code walkthrough by `file:line`

- `engine.py:37` `positions`: the weight on return date d comes from `sessions[i(d) - lag]` (`:44`); a
  missing signal raises instead of becoming cash.
- `engine.py:57` `daily`: turnover with the pre-start weights 0 (`:69`), so the first entry is charged;
  the exact net return (`:71`); drawdown from a peak of at least 1.
- `strategy.py:73` `load_config`: exact keys, `lag < 2` refused (`:77`), unique ids, the primary present.
- `strategy.py:103` `select_model`: M5's ranking (sort by id, then a stable sort by loss).
- `strategy.py:118` `band_signal`: strict inequalities, so a value at ±band is cash; NaN stays NaN.
- `strategy.py:143` `window`: usable return dates of the period, and they must be consecutive sessions.
- `strategy.py:183`-`:207`: trial rows, `count_trials` (distinct `(strategy_id, params)`), `trial_srs`.
- `strategy.py:210` `summarise`: metrics, then `deflate` with the period's trials.
- `segments.py:85` `segment_of_return`: both closes inside; `:110` `check_segments` (gap, split at
  `:160`, exposure at `:149`), reading only rows up to `read_until`.
- `metrics.py:61` `deflate`: `var_sr = max(var_cross, 1/T)` (`:69`).
- `sharpe.py`: verbatim from `sharpe-observatory/scripts/metrics.py` lines 36, 39, 42-54, 84-94,
  97-112, 122-139, 142-144, 147-149 at commit 080a044, with the MIT notice.
- `cli.py:209` `cmd_dev`: the frozen `data_end` check, then the cut to dates before `holdout_start`
  (`:219`) before the segment check, the model choice and the signals (`_backtest`, `:135`).
- `cli.py:233` `preflight`: every precondition of the look, reading dates, keys and configs only;
  `cli.py:318` `cmd_look`: preflight (`:323`), the create-only record (`:336`), then forecasts, M5's
  report and the backtest inside one `try`, the record's `status` updated after each stage.
- `models/holdout.py:84` `run_holdout_forecasts`: warm-up + holdout folds through the dev runners
  (`walkforward.run_symbol` / `run_rv21` with `allow_holdout=True`, `walkforward.py:123`, `:195`), warm-up
  output discarded; `:75` `lstm_start_cutoff`; `:45` `build_track_models`.
- `segments.py:117` `stale_prints`, checked at `:146`.
- Review B: `cli.py` `preflight` (segment checks with `values=False`, resume checks, `run_protocol`),
  `cmd_look` (stages, `--resume`, `--preflight-only`); `models/holdout.py` `check_reproduces_dev`,
  `check_no_fallback`, `lstm_warmup`.

## 5. Data flow (dev; the look adds holdout refits under `period=holdout/`, M5's holdout tables and `backtest/period=holdout/`)

```
yfinance VXX, SVXY --fetch--> raw/yfinance/{VXX,SVXY}/<sha>.csv + provenance.jsonl
   --prepare--> interim/tradable_daily.parquet (price_daily schema; not price_daily itself)
interim (VIX close) + processed/run_id=<id>/forecast.parquet + evaluation/period=dev/metric.parquet
   --run--> [cut to < holdout_start] -> segment checks -> model choice -> signals -> engine (L7, costs)
   --> processed/run_id=<id>/backtest/period=dev/{backtest_daily, strategy_summary, segment_check}.parquet
       + backtest_meta.json;  ROOT/backtest/trial_log.jsonl (append-only). All local (ADR 0001 D2).
```

## 6. Pitfalls and their guards (mutation-checked after review B: 56 mutants, 56 killed)

Review A refuted the first count ("25 mutants, all killed"): that harness had a syntax error, a no-op mutant and a
mutant without its target, and R2, R3 and R4 survived the reviewer's own run. Re-measured on the final tree with a
rewritten harness: 44 mutants (the 24 earlier ones, a combined leak mutant, R2, R3, R4, R4b, and 15 on the look,
the preflight, the stale check and the runners), each killed by a named test. After review B: 57 mutants
(13 more on the look path, B1-B13); N4 survived because the preflight's separate presence check of
holdout closes duplicated the segment check's missing-close test, so that check was removed (with N4)
and N2 and B1 re-run: 56 mutants, 56 killed. The other results were measured on the tree just before
that removal.

| Pitfall | Guard (test) |
|---|---|
| Filling at the close the signal used, or earning that day's return | hand-worked book; lag 0 and lag 1 mutants killed (`test_pnl_by_hand_with_next_close_fills`); the loader refuses lag 1 |
| Costs missing on the first entry, a switch charged as one side, first-order costs | the same hand-worked book (three mutants); `test_costs_are_charged_on_every_change_of_position` |
| A value at the band edge trading; a missing forecast silently becoming cash | `test_dead_band_is_cash_inside_and_at_the_edge`; `test_a_missing_signal_stops_the_engine` |
| Excess instead of Pearson kurtosis; sqrt(T) for sqrt(T - 1); ddof; dropping the 1/T floor | PSR and DSR by hand with `statistics.NormalDist` (`test_psr_by_hand_on_a_four_day_series`, `test_dsr_by_hand`) |
| A return straddling the SVXY cut | `test_segment_boundaries_from_the_config` (the "end close only" mutant is killed) |
| A gap, an unadjusted split, a wrong leverage | `test_segment_checks_catch_a_gap_an_unadjusted_split_and_a_wrong_exposure` |
| The window running into the holdout, or bridging a hole | `test_the_engine_never_leaves_the_usable_segment` |
| A holdout value reaching a dev output | the command re-run with every VIX, S&P, ETP price and forecast dated on or after `holdout_start` x100: every dev table identical; a mutant that drops the cut and uses a full-sample max is killed |
| Benchmarks or holdout runs counted as trials | `test_dev_run_reads_nothing_from_the_holdout_and_the_synthetic_look_works` (log lines = 5 x 4; unchanged by the holdout run) |
| A real look by accident, from a dirty or unpushed tree, or twice | `test_the_real_look_needs_the_flag_a_clean_pushed_tree_and_happens_once` |
| A doomed look spending the record (review A, finding 1) | `test_a_failed_preflight_writes_no_record` (5 cases); a race after a passed preflight: `test_the_record_is_create_only_even_after_a_passed_preflight` |
| The look end to end: holdout refits of every dev-store model, M5's holdout report, the backtest, no new trials, never twice | `test_the_one_look_end_to_end_on_synthetic_data` (rv21 `rw`, `har_rv`; vix_level `rw`, `har_log`: no GARCH or LSTM in the test) |
| A crash mid-look hiding that the look was spent | `test_a_crash_mid_look_leaves_a_record_that_says_so` |
| A missing return while holding read as 0 (review A, R2) | `test_a_missing_return_while_holding_stops_the_engine` |
| The forecast compared with the VIX of another day (review A, R3) | `test_the_signal_compares_the_forecast_with_the_vix_of_its_own_origin` |
| Stale prints inside a usable segment | the segment-check test (O = H = L = C, volume 0, and a holdout stale print counted, not failed) |
| The LSTM's retrain schedule restarting at the holdout | `test_the_lstm_continues_from_its_last_dev_retrain`, `test_the_lstm_warmup_starts_at_the_fold_of_its_last_retrain` |
| An unadjusted holdout split found only after the look is spent (review B) | `test_a_failed_preflight_writes_no_record[unadjusted_split]` (VXX x4 from 2024-07-24; problems named, no values) |
| A precondition check that writes or spends something | `test_preflight_only_writes_nothing_and_is_not_a_look` (file tree identical, byte for byte and by mtime) |
| A failed stage ending the holdout for good, or a resume that changes the code | `test_a_failed_look_resumes_at_the_failed_stage` |
| A holdout fit silently falling back | `test_a_holdout_fit_that_falls_back_fails_the_stage_loudly` |
| A warm-up chain that does not reproduce dev | `test_the_holdout_chain_must_reproduce_the_stored_dev_forecasts`, `test_a_look_whose_chain_does_not_reproduce_dev_stops_before_any_holdout_forecast`; the end-to-end test asserts every stored dev row was recomputed |
| A dev run made under another protocol | `test_a_dev_run_may_differ_from_the_protocol_only_in_holdout_looks` |

Mutation runs: `/tmp/claude-1000/-home-rajib2k5-dev/650c02f9-8119-48bf-910f-cad779202fc6/scratchpad/m6/mutate2.py`
(a scratch copy per mutant; `tests/test_backtest.py`, and `tests/test_walkforward.py` + `test_baselines.py` for the runners' default),
results in `.../m6/mutants2_final.txt`.

## 7. Measured results

### 7.1 Synthetic (publishable)

`uv run python -m ivf.backtest --synthetic run --print-tables`, 2026-10-29, synthetic run
`85a849917197` (models chosen by the rule: rv21 `garch`, vix_level `arima_garch_log`), re-run after
D12 on returns 2018-08-02 .. 2024-06-28 (1,486 sessions). The synthetic VIX has no state-dependent
risk premium (`ivf.data.synthetic`), so there is nothing to find:

| Strategy (5 bp) | Sharpe (ann.) | PSR0 | DSR (N = 5) | Fills |
|---|---:|---:|---:|---:|
| primary `fvix_rv21_b2` | -0.27 | 0.26 | 0.03 | 169 |
| always long SVXY | 2.07 | 1.00 | 1.00 | 1 |
| always long VXX | -2.07 | 0.00 | 0.00 | 1 |

The two benchmarks have Sharpe ratios of opposite sign and equal size, as the toy's SVXY is exactly
-0.5x its VXX each day (a check on the engine).

### 7.2 The real-data steps (counts, dates and timings only, D2)

2026-10-29, uncommitted M6 code on `4da5b45`, `/usr/bin/time`:

| Step | Result |
|---|---|
| `uv run python -m ivf.data fetch --symbols VXX SVXY` | exit 0; two new snapshots with provenance; wall 1.7 s |
| `uv run python -m ivf.backtest prepare` | exit 0; VXX 2,203 rows 2018-01-25 .. 2026-10-29; SVXY 3,790 rows 2011-10-04 .. 2026-10-29; wall 1.35 s |
| `uv run python -m ivf.backtest run` (first) | **exit 1**: the exposure check failed on the unusable -1x SVXY segment (D7); nothing written, no trial logged |
| `uv run python -m ivf.backtest run` (after D7) | **exit 0**; wall 2.28 s (0.7 s inside the step); peak RSS 460,288 KB |
| Segments read | VXX 1,617 closes and SVXY 1,595 to 2024-06-28, 0 missing; 4 splits checked; 0 violations |
| Dev window (first run, before D12) | returns 2018-02-28 .. 2024-06-28: 1,594 sessions (origins two sessions earlier); 0 forecast rows on or after `holdout_start` in the store |
| Fills at 5 bp (entries from cash included) | `fvix_rv21_b2` 194, `b0` 66, `b1` 150, `b4` 165, `fvix_vix_b2` 100; benchmarks 1 each; cash 0 |
| Trials | n_trials = 5 (20 log lines: 5 strategies x 4 cost levels) |
| Rows | backtest_daily 51,008 (8 strategies x 4 costs x 1,594), strategy_summary 32, segment_check 7 |
| **Re-run after review A** (`uv run python -m ivf.backtest run`, D12, `holdout_looks` 1) | **exit 0**; wall 1.92 s (0.5 s inside); VXX 1,487 closes from 2018-08-01 and SVXY 1,595 to 2024-06-28, 0 missing, 0 stale; 4 splits; 0 violations; returns 2018-08-02 .. 2024-06-28: 1,486 sessions; fills at 5 bp: `fvix_rv21_b2` 178, `b0` 61, `b1` 130, `b4` 155, `fvix_vix_b2` 98, benchmarks 1, cash 0; n_trials still 5 (40 log lines: the rule set did not change, so no trial was added); backtest_daily 47,552 |

The model ids the rule chose and every return, Sharpe, PSR and DSR stay local
(`processed/run_id=<id>/backtest/period=dev/`), pending the owner's data-policy change. Review A's
stale-print counts (dev data, dates only): VXX 22 open = high = low = close days inside the first usable
window, 2018-03-16 .. 2018-07-31 (longest run 7 sessions, from 2018-04-09), and 3 volume-0 days since
2018-01-25; SVXY none.

### 7.3 A holdout exposure during the build (review A)

Recorded in ADR 0001 4.6 ("A holdout exposure during the M6 build"). Between the VXX/SVXY fetch
(16:57:19 CDT) and 16:58:09 CDT on 2026-10-29, two inspection scripts printed holdout values: VXX and
SVXY daily returns on 2024-07-23 and 2024-07-24; the full-history list of days with an absolute return
above 30%, which included VXX on 2024-08-05; the yfinance split list (VXX 2024-07-24 included); one
OLS slope of SVXY's returns on VXX's over 2024-07-01 .. 2026-10-28; full-history row counts, last
dates and missing-session counts. No values are repeated here. Every M6 setting was written later
(`configs/backtest.yaml` 17:01:56, the D7 exposure rule 17:12:02, the segments 17:18). By review A's
rule `holdout_looks` is now 1, and the planned look is the **second exposure**: its outputs carry
`holdout_caveat`. The SVXY segment's open end is justified by the issuer's fund page (ProShares
n.d., `docs/concepts/00-background.md`), not by that slope; it was not re-read in M6.

### 7.4 The look on the full synthetic model set (review B fix round)

`VOL_DATA_DIR=<copy of review B's pre-look store> uv run python -m ivf.backtest --synthetic run --period
holdout`, 2026-10-29, `/usr/bin/time`, uncommitted code on `4da5b45`; every M2-M4 model (12 non-LSTM on two
tracks, `lstm_log` on both). **exit 0; wall 269.15 s; peak RSS 1,134,504 KB.** Preflight: segment check
to 2026-10-26 pass. Holdout forecasts 264.4 s: the non-LSTM models re-ran the dev chain from the first
dev fold, the LSTM one warm-up fold from its last retrain; **146,916 recomputed dev-fold forecasts equal
the stored ones within 1e-12 vol points**, GARCH, ARIMA and LSTM included; 0 holdout fits fell back;
16,324 holdout forecast rows on 10 holdout folds. Review B's look on the earlier code (restart at the
last dev fold) took 132.13 s, so the full chain costs about +137 s here (review B measured 199.5 s for the
chain on its own). The real look will take longer (more data per fit; ADR 0001 O2).

Real dev re-run after review B (`uv run python -m ivf.backtest run`): **exit 0**; wall 1.84 s; the
same window, fills and row counts as after review A; the dev run's protocol `c170a7eb9453` accepted
against the current `f0fa347c36bd` (they differ only in `holdout_looks`); n_trials 5 (60 log lines).

## 8. Open items

| # | Item | Default (taken 2026-10-29, owner can override) |
|---|---|---|
| O-M6-1 | **Closed (review A; warm starts amended by review B, D15).** The holdout forecasts are a stage of the look itself (`ivf.models.holdout`): specifications and hyper-parameters frozen, refits continue at every holdout cutoff (orders, parameters, scaler moments, the LSTM on its `refit_every` schedule, the `*_cal` multipliers in M5's report). Not measured: the real look's run time. Measured on the full synthetic model set: section 7.4 | — |
| O-M6-2 | **Closed (review A):** the typeset eq. (2) shows sqrt(T - 1) and (g4 - 1)/4, Pearson kurtosis | — |
| O-M6-6 | The SVXY fund page (the segment's open end) was not re-read in M6 | re-read it before the look |
| O-M6-3 | VXX Series B inception date | to verify against the issuer; it precedes the usable window either way |
| O-M6-4 | Cost assumptions (4 + 1 bp) | not measured; the sensitivity table brackets them |
| O-M6-5 | The app's `equity_curves.json` (data-model 5.5) | not built in M6 |
