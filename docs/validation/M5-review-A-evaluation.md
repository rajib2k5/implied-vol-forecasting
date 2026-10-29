# M5 review A: evaluation (claims ledger and findings)

The adversarial review (Opus) of milestone 5's first report. The ledger below is the reviewer's,
copied as written (pre-registered claims, all starting REJECTED); its probes (`probe1_poison.py`,
`probe2_fixedb.py`, `probe3.py`, `mutate_a.py`) stayed in `<session scratchpad>/m5-review-a/`. The
findings and fixes after it were relayed by the orchestrator; the fixes are in
[the walkthrough](../milestones/05-evaluation.md#what-review-a-found). Real-data probes printed only
return codes, booleans and counts (ADR 0001 D2).

# Claims ledger (pre-registered; mode: single-context sequential fallback)
All material; all start REJECTED.

C1a dev report outputs are invariant to x100 poisoning of interim iv/price rows dated >= 2024-07-01.
   Probe: copy real root to scratch twice, poison one, run `report` on both, compare every output parquet (booleans). Falsifier: mutant passing full iv/price to build_target_values -> outputs differ.
C1b the dev report reads no interim row dated >= 2024-07-01 (literal).
   Probe: instrument read_interim / frozen_check: does run_report load and pass rows >= holdout_start to any function? Falsifier: n/a (structural).
C1c `--period holdout` on real data exits 2 before read_interim. Probe: run with read_interim patched to raise; rc == 2.
C2a fixed-b 95% |t| quantile = 2.2458 (seed 20261028, R=20000, n=1000) and is reproducible across processes. Probe: two fresh processes.
C2b that quantile is within MC error of the asymptotic value. Probe: other seeds, n=4000. Channel A: KV cubic (no offline primary source -> expect UNVERIFIABLE).
C2c size of fixed-b at n=6090 ~0.055 under sq_error and state nulls. Probe: fresh seed, 1000 reps.
C2d p-value and critical value come from the same null; HLN not applied to fixed-b. Probe: dm_test fixed_b stat == mean/sqrt(lrv/n) by hand with M=ceil(.1n) Bartlett; p == (1+#)/(R+1).
C3a rectangular dm_stat equals hand formula at h=5 random d. C3b HLN only on rectangular/bartlett. C3c sign convention stated and consistent.
C4a MZ HH and NW cov == statsmodels HAC (uniform/bartlett, use_correction False) to 1e-10 at h=21, n=2000 (full matrix). C4b Wald == statsmodels wald_test with the same cov. C4c fallback triggers iff min eig <= 0 and is recorded.
C5 calibration k_c uses only pairs with target_date <= c; fold origins >= c; rv21 target_date - origin = 21 sessions. Probe on real forecast frame (booleans) + poison y after c on synthetic.
C6 Holm == statsmodels multipletests holm on random p incl ties; adjusted p monotone in sorted order; fixed-b p >= 1/(R+1), also for NaN/inf t.
C7 regime labels from date-of-origin data only; n<20 suppressed. Probe: poison iv after origin, labels at origin unchanged; protocol min_cell_n.
C8 synthetic "helps 0 of 28" recomputed independently from metric+dm tables.
Extra: pytest count; 3+ new mutants; doc honesty at n=583.

## Findings (as relayed by the orchestrator) and what changed

| # | Finding | Fix (amended M5, review A) |
|---|---|---|
| 1 | **The primary statistic was untested** (C2d): a mutant applying the HLN factor to the fixed-b statistic only left all 23 M5 tests passing | `test_dm_fixed_b_statistic_by_hand` (n = 11, M = 2, arithmetic in comments; also the sign and two-sidedness) |
| 2 | Rule 6 untested: "helps without the QLIKE condition" and "helps uses the raw p, not Holm" survived | `test_calibration_verdict_follows_rule_6` on a constructed frame |
| 3 | "Regime labels built from post-holdout data" survived: the leak test watched `build_target_values` only | the command test spies on both builders and re-runs the report on interim data poisoned (×100) on and after `holdout_start`: every output table is identical |
| 4 (C2b) | **Refuted:** "the fixed-b 95% quantile is within Monte Carlo error of Kiefer-Vogelsang's cubic". The reviewer's 20 seeds: mean 2.2368, sd 0.0108, about 10 SE below the quoted 2.2606; its own implementation 2.2250. The cubic was quoted from memory | assertion and claim removed; a seeded reproducibility test of the shipped null instead; docs give the simulated value, its seed spread, and "KV comparison not verified" |
| 5 | `fixed_b_p_value(nan)` returned 1/(R+1), the smallest p-value | returns NaN; tested |
| 6 (C1b) | **Refuted as worded:** "the dev report reads no interim row ≥ holdout_start". The frozen `data_end` check reads the full interim tables; the stdout line was a fixed string | reworded to "no value dated ≥ holdout_start enters any output" (the poison test); the line now prints the last session actually used |
| 7 (C3c) | Sign convention not stated where the numbers are | `dm_test.lower_loss`; "positive `mean_d` means `model_a` is worse" in the code, the data model and the docs; section 7.4's directions re-checked against the table |
| 8 | Notebook 05 used a 63-pair warm-up against the pre-registered 252 | labelled and printed as a synthetic illustration |
| 9 | At n ≈ 583 (the holdout) fixed-b DM rejects about 0.10-0.11 under the state null and MZ Hansen-Hodrick about 0.17 | pre-registered (O-M5-5): holdout DM and MZ are descriptive only (`descriptive_only`); the confirmatory check is directional (`holdout_check`), implemented and tested on synthetic data |

**Re-measurement by the orchestrator** (before the fixes): the mutant "HLN applied to fixed-b only"
(`dm.py`, the fixed-b branch of `dm_test`) → `tests/test_evaluation_m5.py`: **23 passed**. After the
fixes it is killed by `test_dm_fixed_b_statistic_by_hand` (walkthrough 7.5).

**Builder's own re-measurement of finding 4** (`fixed_b_null(0.1, 1000, 20000, seed)` for seeds
20261028..20261047): the 95% quantile of |t| has mean 2.2290 and sd 0.0195; the shipped seed gives
2.2458. The two seed studies disagree on the sd (0.0108 against 0.0195); neither was reconciled.
