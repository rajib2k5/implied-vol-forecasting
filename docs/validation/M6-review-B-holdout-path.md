# M6 review B: the one-look path

Adversarial review (Opus) of the one-look path after the review A fix round, 2026-10-29, narrow scope.
Its scratch folder (`m6-review-b/`) holds probes and logs (`harness/chain.py`, `poison.py`, `poison2.py`,
`build.sh`, `build.log`, `chain.log`, `h_full_warm.log`, `full_look.out`, `inj1.out`/`.err`), and no
ledger file. The findings below are as relayed by the orchestrator; the reviewer's verdict text
replaces this section when supplied.

| # | Finding (as relayed) | Probe | Fix (M6 fix round B) |
|---|---|---|---|
| 1 | High: the holdout segment checks ran in stage 3, after the record; synthetic VXX x4 from 2024-07-24 passed the preflight, wrote the record, then failed at the holdout backtest | `inj1.out`/`.err` | segment checks to `data_end` in the preflight, problems named without values; test with the same x4 case |
| 2 | No preflight-only mode | — | `--preflight-only`, tested to write nothing |
| 3 | No resume after `failed at <stage>` | — | `--resume` at the same commit, protocol, config and inputs; attempts in the record; test with a fault at the M5 stage, then resume |
| 4 | Holdout fallbacks (e.g. a dead LSTM worker -> `previous_fold`) were silent | — | `check_no_fallback`; test with an injected fit failure |
| 5 | Labels: no caveat in the metas; "synthetic holdout path" printed for any data; stale docstrings in `report.py` | — | fixed |
| 6 | Mutant M1 (LSTM warm-up `>=` -> `>`) survived | — | `lstm_warmup` + test |
| 7 | Warm starts restarted at the last dev fold; vs the full chain, holdout forecasts of `arima_log` moved by up to 0.031 vol points, `arima_garch_log` 0.013, GARCH family below 2e-6 (13,992 rows; about 1% of rows moved by more than 0.01); the full chain took 199.5 s | `chain.log` | non-LSTM chain from the first dev fold, with a 1e-12 dev-reproduction check |
| 8 | `choose_evaluated_run` ignored `protocol_hash` | — | accepted only if the protocol differs in `holdout_looks` alone; both hashes in the metas |

The reviewer's full synthetic look on its code state (`full_look.out`): exit 0, 132.13 s, peak RSS
1,079,024 KB. The numbers in row 7 are synthetic-data measurements (publishable, ADR 0001 D2).

## The reviewer's verdicts (from its final report; added by the orchestrator)

Narrow review B of the one-look path, 2026-10-29, adversarial-reviewer (Opus). The real look was never
run and no real data was read beyond configs; all runs used a scratch copy with synthetic roots.
`uv run pytest -q`: `931 passed in 127.24 s`.

| # | Verdict | Evidence | Severity / fix |
|---|---|---|---|
| 1 preflight before record | **REFUTED** (high) | the preflight checked only that holdout closes exist; the holdout segment checks (VXX 2024-07-24 split adjustment, SVXY exposure slope to `data_end`, missing closes) ran in stage 3. Synthetic VXX ×4 from 2024-07-24: preflight passed, record written, holdout forecasts and M5 report written, then exit 1, `failed at holdout backtest` | Run the segment check to `data_end` in the preflight, pass/fail only. The race guard (link-based create-only write) SURVIVES |
| 2 full-model synthetic look | Run SURVIVES; labels **REFUTED** (medium) | 7 models per track, exit 0, 132.13 s, RSS 1.08 GB, 16,324 holdout rows on 10 folds, same-rows checks passed, LSTM retrained twice per track, `descriptive_only` True in `dm_test` and `mz_regression`, 5 `holdout_check` rows; real look estimated at 2.5–4 min | `report_meta.json` and `backtest_meta.json` had no caveat key; `report.py:348` printed "synthetic holdout path" on every holdout run; docstrings said "holdout_looks stays 0" |
| 3 refits use data ≤ cutoff, frozen specs | SURVIVES | ×100 on every value after 2025-04-01: for 5,292 rows with origin ≤ that date, max \|Δyhat\| = 0.0 for all 12 models; 66 fits with cutoff ≤ it identical; LSTM last dev retrain 8626, look retrains at 8878 and 9130 (refit_every counted across the boundary) | Low: restarting at the last dev fold changed ARIMA/GARCH warm starts (holdout yhat up to 0.031 `arima_log`, 0.013 `arima_garch_log`; 0.99% of rows > 0.01 vol points) |
| 4 crash leaves `failed at <stage>`; dry run | Record and docs SURVIVE; no dry run **REFUTED** (medium) | real `--period holdout` without `--one-look` exits 2 before the preflight; no preflight-only flag; after a failed stage every rerun was refused, so one transient error lost the look; Ctrl-C leaves `running <stage>` (documented) | Add `--preflight-only` and a resume; from code reading: an LSTM worker death later falls back silently to `previous_fold` (`walkforward.py:164-169`) and the look never checked `fallback` |
| 5 holdout_looks = 1 | SURVIVES | real 1 against expected 1, synthetic 0 against 0; `freeze_record_violations` [] for both; a moved `data_end` with only the hash rewritten is still caught | Low: `choose_evaluated_run` ignored `protocol_hash` |
| 6 dev backtest after D12 | SURVIVES | every interim value ≥ 2024-07-01 ×100 (tradable 1,166, iv 3,498, price 583 rows): dev `backtest_daily` (47,552×14), `strategy_summary`, `segment_check` identical; returns 2018-08-02..2024-06-28 (1,486); D12 disclosed | — |

Mutants of `holdout.py` against `tests/test_backtest.py`: M1 (warm-up `cutoff_idx >= start` → `>`)
**survived** (26/26 pass); M2 (warm-up rows kept in the holdout tables) and M3 (`lstm_start_cutoff`
max → min) killed.
