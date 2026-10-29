<!-- The reviewer's ledger as it reported it, copied in the M2 fix round. Absolute paths are replaced
by <session scratchpad>; nothing else is changed. The fixes are in docs/milestones/02-baselines.md,
"What review A found". -->

# M2 review A: baselines (ledger)

Reviewer: adversarial-reviewer (Opus), 2026-10-29. Target: the uncommitted M2 tree on top of fecf88a.
Mode: single-context sequential fallback. The two channels are the ADR, docs and source on one side
and runtime probes on the other. Statsmodels docs were not consulted (no network). Probes are in
`<session scratchpad>/m2-review-a/` (not committed). No repo file was modified. The scratch copy's
checksum was verified after the mutants were restored.

**Result.** Six of the eight claims survive. Claim 7 is refuted, two of the reviewer's three mutants
survived, and the same-rows check has a hole.

**Gate.** `uv run pytest -q` printed `784 passed in 78.96s`. In the scratch copy, the three new test
files gave `43 passed`.

| # | Verdict | Evidence (command → printed) | Sev | Minimal fix |
|---|---|---|---|---|
| 1 | SURVIVES | `p1_holdout.py`: poisoned 3,498 interim rows ×100 and ran `cli.run_baselines` (VIX and VIX3M, all 5 models, last 2 dev folds) → `y identical: True` for both; forecast, fit and metric tables identical. Aside: the M1 `frozen_check` passed with holdout VIX ×100 (rc 0). | — | — |
| 2 | SURVIVES | `p2_trunc.py`: configured ARIMA grid, a one-session gap at c+5, origins c, c−1, gap−1, gap+22 → 15 rows each, max\|diff\| ≤ 1.07e-14. | — | — |
| 3 | SURVIVES | Real run: 0 of 1,500 fits have `train_last_target_date > cutoff_date`. Premise check: ADR:722 defines L2 on a point target, so no "h=21 target window" exists for `*_level`. Forward risk: `rv21` (ADR:542) is a window target, and the claim at baselines.py:8-9 and walkforward.py:16 ("L2 holds by construction because the array ends there") would not hold for it in M3. | low (M3) | Recheck L2 when `rv21` arrives. |
| 4 | SURVIVES | `p4_arima.py`: against the AR(1) closed form, mean and variance errors ≤ 8.9e-16 at h=1, 5, 21. The h=21 variance is 13.512 × the one-step variance, which equals the closed form. RW variance = h·σ² exactly. Changing data after t gives a change of 0. Falsifier `dynamic=False` → mean error 0.18, post-t change 1.6. HAR's s² is the per-horizon direct RSS/(n−4), so it is h-step. The historical mean uses no s² (a level forecast). The reviewer's mutant "variance at h=1 used for every h" is killed by `test_arima_forecast_matches_the_ar1_closed_form...`. | — | — |
| 5 | SURVIVES | `p5_losses.py`, 400k lognormal(ln 20, 0.09) draws: argmins 18.28 / 19.99 / 20.90 / 21.86 against formulas 18.28 / 20.00 / 20.92 / 21.88 (MAPE / MAE / RMSE / QLIKE). `qlike(20,25)` equals the hand value. The code matches ADR:597, Patton's b=−2 form on v=(y/100)². Patton's robustness does not apply: the target is the observed index, not a proxy. The real caveat, that the plug-in v̂ sits below E[v] by Var/1e4, is disclosed at losses.py:21-22. | — | — |
| 6 | SURVIVES, with a hole | Real and synthetic stores: all 18 (target, h) groups have 5 models with identical origin sets. However, `check_same_rows` accepted a frame where model c had no h=1 rows ("ACCEPTS"), because walkforward.py:252 only compares the models that are present. | low | Require every expected `model_id` in every group. |
| 7 | **REFUTED** (as a defensible rule with honest docs) | `p7_kpss.py`, 100 reps per cell, rate at which d=1 is chosen on a stationary AR(1): φ=0.98: 52% / 37% / 24% at n=2570 / 5000 / 8700; φ=0.99: 76% / 67% / 52%; φ=0.995: 87% / 85% / 77%. Real pre-holdout ln VIX AR(1) slope lies in [0.98, 0.995] (boolean True). So 97 of 97 d=1 on real data is what KPSS does to a stationary series that persistent, not evidence of a unit root. The synthetic 2 of 97 comes from one path with nested windows: one draw, not 97. Bias (`p7b_bias.py`, synthetic φ=0.98, h=21): d=1 RMSE 5.85 vs d=0 5.01 (+17%); mean error +0.87 vs +0.14 vol points; +5.45 when x_t > mean+1sd and −2.34 when x_t < mean−1sd. | **med-high** | Default d=0 for `arima_log`, or disclose. |
| 8 | SURVIVES | Real and synthetic fold tables: 97 dev folds, first cutoff 2000-03-15, last origin 2024-06-28. In the last fold, targets before 2024-07-01: h=21 42 of 63, h=5 58, h=1 62. | — | — |

**Mutants the builder did not list** (run in the scratch copy):

- **M1**, ARIMA using the one-step variance at every h: killed.
- **M2**, KPSS on the whole series instead of the longest gap-free run (baselines.py:236):
  **survived**, 43 passed. Real `kpss` raises `ValueError` on NaN, and real VIX has gaps in burn-in
  (G3). No test gives ARIMA a training series that contains NaN, so neither that rule (G2) nor the
  filter's missing-value path is pinned. The real run's exit 0 is the only evidence those paths work.
- **M3**, the minimum-training rule changed from `<` to `<=` (walkforward.py:148): **survived**. The
  504 boundary is not pinned; the test's nearest case is 474.

**Docs claims the evidence does not support:**

- docs/milestones/02-baselines.md:279 and docs/concepts/07-arima-and-information-criteria.md:99-100
  cite "31% … n = 9,000" as KPSS's risk. At φ=0.98 that understates the rate in the early folds (52%
  at n≈2570), and at φ 0.99–0.995 the rate is 52–87%.
- 02-baselines.md:279 sets the default for O-M2-2 on the synthetic result in section 7.2 (d=1 at 2 of
  97). Nothing in the walkthrough says the real VIX run chose d=1 at all 97 cutoffs.
- 02-baselines.md:256-257 ("with d = 1 it is close to a random walk") and concept 07:101-103 are
  qualitatively honest, but give no size.
- 02-baselines.md:220-227, "36 of 36" mutants killed: true for their list only. M2 and M3 above
  survive.

**Most important fix:** O-M2-2. State that the real VIX run chose d=1 at every cutoff, give the
measured KPSS rates at fold sample sizes, and either switch the default to d=0 or say plainly that
real `arima_log` is a d=1 model behaving like a random walk at h=21. Then add an ARIMA test whose
training series contains NaN.

**Orchestrator's check (2026-10-29).** Before acting on claim 7, the orchestrator re-measured the
KPSS d=1 rate on fresh seeds (60 per cell, `kpss_p_value` from the repo): φ=0.98: 28/60 at n=2570,
17/60 at n=8700; φ=0.99: 42/60 at n=2570, 30/60 at n=8700. The refutation holds.
