# M6 review A: the VXX/SVXY backtest

Adversarial review (Opus) of the first M6 report, 2026-10-29. The reviewer's probes are in its scratch
folder (`m6-review-a/`: `claims-ledger.md`, `mut.py`, `poison_real.py`, `probe_engine.py`, `probe_seg.py`,
outputs `out_R*.txt`). That folder holds the **pre-registered claims ledger** (copied verbatim below)
and the mutant outputs; it holds no file with the verdicts. The verdicts below are as relayed by the
orchestrator, which verified finding 1 itself. If the reviewer's own verdict text is supplied later,
it replaces section 2.

## 1. The pre-registered ledger (verbatim, `claims-ledger.md`)

```
# Pre-registered ledger (mode: single-context sequential fallback), 2026-10-29
C1 lag: weight on return d = signal at sessions[i(d)-2]; loader refuses lag<2. Probe: hand book + mutants lag0/lag1/lag3(off-by-one) in scratch copy; falsifier: tests must fail.
C2 cost: c*|dw| per side, first entry charged, switch=2 sides; net=(1-c)(1+g)-1 equals return on equity when cost is paid on traded notional at fill close. Probe: independent equity simulation with notional-based cost.
C3 leakage: x100 poison on >=holdout_start rows of VIX, S&P, ETP prices, forecasts leaves every dev table identical. Probe: re-run real-data dev pipeline on scratch data root copy, compare. Falsifier: mutant removing forecast cut / sessions cut.
C4 segments: every used return has both closes in one usable segment of both ETPs; window unbroken; split days adjusted. Probe: recompute on synthetic + real dates/booleans only.
C5 sharpe.py verbatim + MIT; PSR/DSR = BLdP 2014 eq.2 with Pearson kurtosis, sqrt(T-1). Probe: line diff; hand value via NormalDist.
C6 N=5 honest: trial_log + runs + mtimes. C7 model choice uses dev metric only, config predates run.
C8 holdout guard: refused w/o --one-look, create-only record; second look refused. Probe: CLI on scratch root.
```

## 2. Findings (as relayed by the orchestrator) and the fix round

| # | Finding | Severity | Fix (M6 fix round, 2026-10-29) |
|---|---|---|---|
| 1 | `cli.py` wrote the look record before checking that holdout forecasts exist, so a doomed run consumed the one look; the old test showed it | high, verified by the orchestrator | `preflight` checks every precondition before the record; a failed preflight exits 2 and writes nothing; tested for five preconditions plus the real-data ones (dirty tree, unpushed commit), and the create-only write under a race |
| 2 | No step forecast the holdout folds | gap | `ivf.models.holdout` inside the guarded look: specs frozen, refits continue at every cutoff; order preflight -> record -> refits -> M5 holdout report -> backtest; tested end to end on synthetic data, including a mid-look failure leaving `failed at <stage>` |
| 3 | Holdout values printed during the build before the M6 settings were written | breach | ADR 0001 4.6 and the walkthrough 7.3 record what, when and why; `holdout_looks` 0 -> 1 by the reviewer's rule; the look carries a caveat; the SVXY segment's open end rests on the issuer's fund page |
| 4 | VXX stale prints (22 OHLC-equal days 2018-03-16..2018-07-31, one volume-0 day) in the usable window | medium | stale-print check and rule D12 (VXX usable from 2018-08-01), disclosed as added after the first dev run; real dev re-run: exit 0, no new trial |
| 5 | Mutants R2, R3 survived; the author's harness had a syntax error, a no-op mutant and a mutant without its target | untested / refuted count | tests for R2 and R3, and R4 made observable; harness rewritten (`m6/mutate2.py`); re-measured |
| 6 | Real values in `segments.py` and concept 14; data-model said `price_daily`; O-M6-2 open | docs | removed; `tradable_daily`; O-M6-2 closed |

Reviewer mutant outputs (its `out_R*.txt`): R0 control survived (as a control must); R1 (lag off by one)
killed; R2 survived; R3 survived; R4 (the forecast cut dropped; named "forecast_cut_and_iv_cut_dropped") survived; R5 did not run (syntax
error in the probe); R6 (same-day fill) killed.

## 3. The reviewer's verdicts (from its final report; added by the orchestrator)

Mode: single-context sequential fallback (dual-confirmed (fallback) at best). Scratch:
`<session scratchpad>/m6-review-a/`. No repo file changed. `uv run pytest -q`: `920 passed in 311.87s`.

| # | Verdict | Evidence | Severity / fix |
|---|---|---|---|
| C1 lag of 2 sessions | SURVIVES | `probe_engine.py`: `origin == sessions[i-2] for all: True`; real data `probe_seg.py`: `True`. Mutants same-day fill (R6) and lag 3 (R1) killed | Gap (medium): R3, `vix.shift(-1)` in `strategy.signals`, survived (15 passed): the forecast-to-VIX alignment was unpinned |
| C2 costs | SURVIVES | an independent equity book (cost = c·\|Δw\|·pre-trade equity) matched `engine.daily` (`allclose True`); turnover `[1,0,2,1,0,1,2]`; gap to cost on executed notional ≤ 9.73e-07 a day (≈ c²) | Wording: "exact" holds under the pre-trade-equity convention |
| C3 no holdout value reaches dev | SURVIVES (real data) | `poison_real.py`: every value dated ≥ holdout_start ×100 plus 40 fake holdout forecasts ×100 → `identical after poison: all True`; falsifier (one dev forecast changed) → `False` | R4 (forecast cut removed) survived: the cut is redundant, not the guard the docs describe |
| C4 segments | SURVIVES on dates; data-quality hole | returns 2018-02-28..2024-06-28, n = 1,594, consecutive; `adj_close != close` 0 rows; no unlisted jump > 0.6, 0 NaN returns | Medium: VXX had 22 days with open = high = low = close (2018-03-16..2018-07-31) and 1 zero-volume day in the usable window. The SVXY 8-K and the Series B inception date were not verified offline |
| C5 sharpe.py, PSR/DSR | SURVIVES | all 8 cited spans of `080a044:scripts/metrics.py` found verbatim, MIT notice present; hand values PSR0 0.9416898371, DSR 0.6544429294, Pearson kurtosis 11.07370354; the typeset eq. 2 shows √(T−1) and (γ̂₄−1)/4 | O-M6-2 can close |
| C6 N = 5 | SURVIVES on the artifacts | trial log: 20 lines from one run (5 strategies × 4 costs); the failed first run exited at the segment check before the model choice and the log write, so it computed no strategy P&L; the D7 change was a data gate | `runs.parquet` holds model runs only; REPL activity cannot be checked |
| C7 model choice fixed in advance | SURVIVES | `backtest.yaml` written 17:01:56, before the synthetic (17:06) and real (17:12:04) runs; metric table `period == ['dev']` only | — |
| C8 holdout guard | SURVIVES, with a high-severity design flaw | refused without `--one-look` before any read; record create-only (R5 `how="replace"` killed) | High: the record was created before checking that holdout forecasts exist, so a run certain to fail would consume the one look (`tests/test_backtest.py:319-323` showed it) |
| 8a "frozen" | — | ADR 0001 :756 "hyper-parameters frozen", :791 parameters re-estimated at every cutoff, holdout included; :693 holdout folds; O2 counts 97 dev + 10 holdout refits | Specifications frozen, refits continue; O-M6-1's "every model frozen" was ambiguous |
| 8b holdout prints | Protocol breach of the M1 kind (ADR :786-788), undisclosed | grep found nothing in the walkthrough, ADR or log | Record what, when; raise `holdout_looks` to 1 if any M6 setting was written after the print; the look stays possible with a caveat |

Other findings: "25 mutants, all killed" (06-backtest.md:86) cited a harness that did not parse (`SyntaxError` at line 32; M26 a no-op; M25 lacking its target); R2 (a missing return while holding treated as 0) survived; data-model.md named `price_daily.adj_close` as the return source; `segments.py:19-21` and concept 14:76-78 published real return and slope values. The reviewer's own breach: a header peek printed two raw VXX rows dated 2018 (dev period) to its console; they are not reproduced anywhere.
