# M0 review B: maths, protocol and documents (evidence-gated validation ledger)

| | |
|---|---|
| Target | ADR-0001; `docs/architecture/data-model.md`, `*.mmd`, `specs/*.yaml`; concepts 01-03; `docs/prfaq.md`; `docs/milestones/00-scaffold-and-design.md`; `README.md`; `schemas/app-data/*.json` (untracked tree, no commits, 2026-10-28) |
| Environment | uv 0.11.25, Python 3.12.13, numpy 2.5.4, scipy 1.18.1, arch 8.0.0, pandas 3.1.0, jsonschema 4.26.0, node 22.23.1; `exchange_calendars` 4.13.2 in a scratch venv only; `@mermaid-js/mermaid-cli` 11.17.0 in a scratch dir only |
| Mode | Tier 2 (dual channel) for [T2] claims, Tier 1 elsewhere. Channel A = a separate docs-only subagent. Channel B = the reviewing context (independent derivation and computation). C7 and C8 = separate probe-only subagents. **Caveat:** Channel A's report reached the reviewing context before Channel B's probes ran. Probe designs were pre-registered first, but agreement is "dual-confirmed (fallback mode)", not independent confirmation. |
| Ledger | Pre-registered 2026-10-28T13:40Z: 8 claim groups, 40 rows, all material except C5.4. Rows added after registration are marked **ADDED**. |
| Default | Every material claim started REJECTED. |

## Verdict: FAIL

- **2 claim groups are REJECTED:** C4.2 (licensing rules) and C1.9 (MAPE).
- **3 claim groups are refuted in part:** C2 (the premise behind the DM kernel), C5 (a broken link; an unverified fact stated as fact) and C8 (ERD contradictions).
- **2 leaf claims are UNRESOLVED:** the VIX 16:15 close and the gzip table.
- **Everything else is ACCEPTED,** including every number in the three Reproduce blocks, every calendar figure, and the DM sizes.

## Evidence table

A = primary-source channel (records quote source, page and equation). B = computation channel (records give the command and its verbatim output). Every B probe that counts as evidence was paired with a falsifier that it failed on.

| ID | Claim | A | B | Gate | Sev |
|---|---|---|---|---|---|
| C1.1 | `252/21·Σr²` is annualised variance per 252-session year; 20.78 / 15.87 / 44% | derivation | concept-01 block prints 20.7846; falsifier 365/21 differs | ACCEPTED | — |
| C1.2 | Cboe VIX formula and the App. 3 example give 13.93 | CONFIRMED: Cboe *Volatility Index Methodology* v6.0 (26 Feb 2026) App. 3 pp.10-14 ("VIX = 100 x 0.13927842 = 13.93"); Mathematics Methodology v5.0 eq.(1) | 13.927842; swapped weights give 12.7151 and 252/21 annualisation gives 13.8321 (both discriminate) | ACCEPTED | — |
| C1.2d | VIX official close at 16:15 ET | UNVERIFIABLE: v6.0 §4 p.8 gives only "Between 09:31 a.m. ET and 4:15 p.m. ET" | n/a | UNRESOLVED | LOW |
| C1.3 | Log-contract strip (Demeterfi et al. 1999); with jumps the strip prices `2Σ(e^J−1−J)`, not `ΣJ²` | CONFIRMED: DDKZ EQ 26 p.19, cited in v6.0 p.5. The jump form is not printed in Carr-Wu (2004 WP) Thm 1 | J=−0.1 gives 0.010000 vs 0.009675. strip−QV = −0.000325 ≈ +J³/3; the sign flips for J>0 | ACCEPTED (jump form by derivation) | LOW: "differ by about −J³/3" holds only as QV−strip |
| C1.4 | VRP definition, units and decomposition; 0.011369, 2.39, 10.66 | derivation | `0.011369 2.39 10.66` | ACCEPTED | — |
| C1.5 | GARCH k-step closed form and 21-step sum 42.609012; G=0.051131; ŷ=22.61; naive 23.81 | derivation | brute − closed = 0.0; arch 8.0.0 analytic path vs recursion max 3.3e-16; falsifier `φ^k` gives 8.5e-3 | ACCEPTED | — |
| C1.6 | RMS bias ≈ `100·Var(v)/(8v̂^{3/2})`; 0.471 vs 0.492; plug-in gap is exactly `Var(y)/10⁴` | derivation | MC 0.4919 exact, 0.4704 Taylor; coefficient 1/4 gives 0.94; gap equals Var(y)/1e4 to 16 digits | ACCEPTED | — |
| C1.7 | Lognormal mean and median; 20.635; MC 20.6408 (SE 0.0052); median 20.0048; 18.83 | derivation | 20.640822, SE 0.005245, median 20.004830 | ACCEPTED | — |
| C1.8 | QLIKE `v/v̂ − ln(v/v̂) − 1` is ≥0, minimised at E[v], and robust to a noisy proxy | CONFIRMED: Patton 2011 eq.(24) at b=−2 p.252; Definition 1 needs a "conditionally unbiased volatility proxy" | argmin = E[target] for both truth and proxy; ranking stable under QLIKE but flips under squared-log loss (falsifier) | ACCEPTED | LOW: this is Patton's normalised eq.(24), not his eq.(6) "QLIKE"; the ADR omits the unbiased-proxy condition |
| **C1.9** | "RMSE, MAE, MAPE and bias use ŷ (fine for plugin)" (ADR:221-222); "each kind is unbiased in its own space" (concept 01:174) | **REFUTED:** Gneiting (2011, arXiv v2 p.27): absolute percentage error elicits "the median of order -1" | **REFUTED:** lognormal s=0.25: E[y]=20.635, MAPE minimiser 18.785, MAE minimiser 19.999. MAPE of the mean forecast is 0.2118 against 0.2038 for the median | **REJECTED** | MED |
| C2.1 | Label rule: h leaking pairs (not h−1); tables c=100,h=21 and c=500,h=5 | derivation | 21 pairs (τ 80..100), max lag 21; 5 pairs (496..500) | ACCEPTED | — |
| C2.2 | Optimal h-step errors are MA(h−1); RW acf (h−k)/h; sim 0.9521 … 0.0080 | CONFIRMED: DM 1994 TWP p.4 "at most (k-1)-dependent"; HLN p.281 | concept-02 block reproduces | ACCEPTED | — |
| C2.3 | Rectangular LRV truncated at h−1 (DM); Bartlett `1−j/h` gives 14.016 vs 21 (SE 0.817); h=5 gives 3.4 vs 5 | CONFIRMED: DM pp.4-5 "uniform, or rectangular, lag window", S(T)=k−1; NW eq.(5) `1 − j/(m+1)` | 14.015873, 0.81696 | ACCEPTED | — |
| C2.4 / C6.8 | HLN factor `√((n+1−2h+h(h−1)/n)/n)` with t_{n−1}; 0.9648; derived for DM's rectangular estimator | CONFIRMED: HLN 1997 eq.(9) p.283; eqs.(1), (5)-(8): "The estimator employed by Diebold and Mariano can then be written as" | 0.964837 | ACCEPTED. The **[unverified]** tags at ADR:342 and concept 02:149 can now cite HLN eq.(5)-(9) | LOW |
| C2.5 | DM size 0.0636/0.1189 (n=583), 0.0542/0.1090 (2000); sq-err 0.0590/0.0896, 0.0553/0.0938; rect never ≤0 | n/a (measured) | Own code, seed 777, 20,000 reps per cell: 0.0641/0.1154, 0.0553/0.1128, 0.0619/0.0913, 0.0529/0.0905; rect≤0 0 times. Every gap ≤0.0038, within 3 MC SE. Falsifier: plain t-test size 0.61-0.67 | ACCEPTED | — |
| C2.6 | (ADR:335) "Optimal h-step errors are MA(h−1), **so** the loss differential's autocovariances vanish beyond lag h−1" | UNVERIFIABLE: DM and HLN *assume* it (HLN p.282: "all autocorrelations of order h or higher of the sequence d_t are zero"); no source derives it for nonlinear losses | Pre-registered exchangeable GARCH design: size 0.0533/0.0480, my attack failed. However, rect≤0 occurred 3 times in 20,000 | see C2.6b | — |
| **C2.6b ADDED** | The same premise when the two models' relative accuracy depends on a persistent vol state, with E[d]=0 unconditionally | N/A | **REFUTED:** log-variance AR(1) φ=0.98. acf(d) at lags 21/30/50 = 0.089/0.066/0.046; population LRV 24.4 vs 16.8 when truncated at h−1. rect+HLN size **0.1258** (n=583) and **0.1103** (n=2000); symmetric control 0.0563/0.0507 | **REFUTED (single-channel counterexample)** | MED |
| C3.1 | 251.50 sessions/yr; 21 sessions span 29-38 days (mean 30.50); 30 days hold 15-22 sessions (mean 20.64) | not run | XNYS 2000-2025: 6,539/26 = 251.50; 29/38/30.5011; 15/22/20.6356 | ACCEPTED (B only; the C7 agent independently got 6,539 and 30.501 with the same library) | — |
| C3.2 | Trailing 24 months to 2026-10-26 start 2024-10-28 and exclude 2024-08-05 | not run | 24m before = 2024-10-26, so the first session is 2024-10-28; excludes 08-05 | ACCEPTED | — |
| C3.3 | 583 = 9·63+16 (10 folds); 97 dev folds; first cutoff 2000-03-15; first fit 2,579; spike origins 07-05/07-29/08-02 and 2025-03-10 (→04-08); 835; 3,977 | not run | all reproduced exactly | ACCEPTED | LOW: dev origins 2000-01-03..2000-03-14 (50 sessions) belong to no fold |
| C4.1 | Cumulative RW h=1 errors rebuild the path | derivation | max abs error 0.0; falsifier (h=5) gives 3.99 | ACCEPTED | — |
| **C4.2** | ≥20 dates per cell, plus the refresh rule, plus sibling blanking, prevent recovering any single day | n/a (logic) | **REFUTED by construction** (attacker sees only `n` and 4-dp `bias`; all rules obeyed), see below | **REJECTED** | HIGH |
| C5.1 | Every annotated number in the concept 01/02/03 Reproduce blocks | — | All three blocks rc=0; every value matches (output verbatim in Appendix A) | ACCEPTED | — |
| C5.2 | Numbers outside Reproduce blocks | — | GARCH-on-AR(1) ranges [0.923,0.977], [0.946,0.977], [0,0.0315], [0.983,1.029]; MC median/SE; 47 and 446 ulps; run_ids fbe9bc2a1017/797fa6aed9c3/31c8d4c13221; `test_schemas.py` "49 passed"; `--check` "ok: 1 file(s)" rc 0 | ACCEPTED | — |
| C5.2g | Concept 03 §7 gzip byte table | — | No command and no data spec, so it cannot be re-made. Its ratios (2.85/1.77/1.57) are internally consistent | UNRESOLVED | LOW |
| C5.3 | No "to verify" claim is stated as fact elsewhere | — | ADR:189 says "16:15 versus 16:00 ET, to verify in M1", but concept 02:70 says "VIX closes at 16:15 ET, after VXX at 16:00" as fact | REFUTED | LOW |
| C5.4 | Links resolve | — | 401 relative links checked. `00-background.md` is missing: concept 01:4, 02:4, 03:4, milestone:9, README:74, LEARNING_LOG:33. prfaq.md:106 cites it as the **source** of H1 | REFUTED | MED |
| C6.1 | VIX = 30-day expected S&P 500 volatility from SPX options | CONFIRMED: v6.0 p.3 and p.5 | (C1.2) | ACCEPTED | LOW: the document title is "Cboe Volatility Index Methodology", not "VIX Index Methodology" |
| C6.2 | VXX is an ETN on rolling 1-month short-dated VIX futures; reverse splits; series change | CONFIRMED: Barclays 424B2 (13 Feb 2026). Original matured 30 Jan 2019; Series B was VXXB until 2 May 2019. 1:4 splits 2010-11-09, 2012-10-05, 2013-11-08, 2016-08-09, 2017-08-23, 2021-04-23, 2023-03-07, 2024-07-24 | n/a | ACCEPTED (A only, factual) | — |
| C6.3 | SVXY −1x → −0.5x in Feb 2018 | CONFIRMED: ProShares 8-K / press release 26 Feb 2018, "effective as of close of business on February 27, 2018" | n/a | ACCEPTED: the first −0.5x session is 2018-02-28 | — |
| C6.4 | XIV terminated in 2018 | CONFIRMED: Credit Suisse 6-K 6 Feb 2018. Acceleration event 5 Feb, valuation 15 Feb, payment 21 Feb 2018 | n/a | ACCEPTED | — |
| C6.5 | HAR = Corsi (2009) | CONFIRMED: JFEc 7(2) eq.(8) p.181; weekly 5-day, monthly 22-day | n/a | ACCEPTED (Corsi's HAR is on realized *volatility*; the repo's use on ln V is disclosed) | — |
| C6.6 | QLIKE penalises under-forecasting more | CONFIRMED: Patton p.252 "heavier penalty … on under-prediction (b < 0)" | v=1: v̂=0.5 gives 0.3069 vs v̂=1.5 gives 0.0721; log-symmetric 0.1487 vs 0.1065 | ACCEPTED | — |
| C6.7 | The PR/FAQ presents no number as a measured finding | — | Every number is a rule, date, plan, hypothesis, or "derived, not measured" (the "about 2 h" is labelled) | ACCEPTED | LOW: prfaq:27 "Every model forecasts the VIX 1, 5 and 21 sessions ahead" contradicts the registry (`har_rv`, `vix_implied` are rv21-only, h=21) |
| C7 | Sampled walkthrough citations; P-answers vs ADR | — | See C7 below | REFUTED in part | LOW-MED |
| C8 | Spec names, ERD vs data model, JSON contract vs schemas, renders | — | See C8 below | REFUTED in part | HIGH (one item) |

### C4.2: how a single day is recovered under the rules as written

The attacks below ran on a simulated VIX-like series. The attacker sees only the published `n` and the 4-decimal `bias` of each `metric` cell. The script is `probes/c4_licensing_attack.py`, kept in the review scratchpad; it is not in the repo.

```
A1 version E1: n=400  n*bias=12.5600  true V_E1-V_s=12.5600  bound=0.0200
A1 refresh +20 (rule met): n2*b2-n1*b1=-5.7560  true V_E2-V_E1=-5.7700  bound=0.0410
A1 with one anchor V_s=11.59 (e.g. a dated close quoted in prose, allowed by D2): V_E2 = 18.394 (true 18.38)
A1 FALSIFIER (hist-mean model): n2*b2-n1*b1=87.0520 vs V_E2-V_E1=-5.7700
A2 refresh E1=1102->E2=1126, per-regime gains {'low': 0, 'mid': 1, 'high': 23}; published n_all - sum n_regime = 1 date(s)
A2 recovered error sum -0.6154; true single-day error at origin [1125] = -0.6300; bound 0.0525
A3 n_all=400, n_contango=254, n_backw=145 (both >=20, nothing blanked); residual n=1
A3 recovered 1.7102  true error on 905: 1.7000  bound 0.0399
```

- **A1, telescoping.** The random walk's h=1 bias over any run of consecutive origins equals `(V_end − V_start)/n`, however large n is. So the premise "the difference is then itself an aggregate over ≥ 20 dates" does not protect anything:
  - each version reveals a two-day contrast;
  - each refresh reveals `V_E2 − V_E1`;
  - one dated close quoted in prose (which D2 allows) turns these contrasts into levels.
  The same holds for always-long VXX: `max_drawdown`, and `ann_return` if it is geometric, depend only on two prices.
- **A2, the refresh rule against the partition.** This attack is not contingent on anything unusual.
  - A refresh adds 24 dates: `all` gains 24 and updates, `high` gains 23 and updates, and `mid` gains 1, so it "keeps its previously committed values and n".
  - The published `n_all − Σ n_regime` is then 1, and the residual sum is that day's error.
  - No cell was blanked, so secondary suppression never fires.
- **A3, null labels.** `contango ∪ backwardation` does not partition `all` on days whose `term_structure` label is null. ADR 4.8 defines null only before VIX3M's first date, and `data_check` does not require VIX3M. One null day in a period is enough to recover it.

The D2 principle that no *daily series* can be rebuilt survives: what leaks is sparse. The stated rule-level guarantee does not.

### C7: walkthrough citations and P-answers

The check sampled 45 of 298 distinct citations with `random.Random(20261028)`. 42 MATCH, 2 PARTIAL, 1 MISMATCH. All 324 link anchors equal their displayed text. Every number in P1-P7 is on the line it cites. The C7 agent's findings, with the ones I re-checked marked:

- **The "11 '(amended M0)' notes" count is wrong (milestone:61).** ADR-0001 has **25**; I re-counted with `grep -o "amended M0" | wc -l`.
- **`app/src/lib/project.test.ts:87` is cited for "a non-synthetic source marked committed is flagged".** That test never calls the validator. The flagging test is at `:103`, `it('flags committed vendor data'…)`. I re-checked both lines.
- **`tests/test_contract.py:304` is cited for "a malformed envelope reports its real problem".** The matching test is at `:315`; I re-checked both lines.
- **`RibbonCanvas.tsx:58` is off by one:** the attribute is on `:59`.
- **P2 "2.0 to 2.3 times"** does not follow from the 33.0 s and 14.7 s quoted beside it, which give 2.24. **"About 8 times heavier"** holds for the download (8.5×) but not the venv (3.9×).
- **Several ranges stop short of the supporting text:** "28 months" is on ADR:300; "107" is on ADR-0002:375; "plain GPU loop slower" is on ADR-0002:147.

### C8: diagrams and the contract

**What holds**

- Every `.mmd` file and the README's inline block render with mermaid-cli 11.17.0 (exit 0, no error graphic). Two deliberately broken copies exit 1.
- `erd.yaml` and `erd.mmd` agree exactly: 33 entities, 296 columns, 51 relations.
- data-model.md §5 matches the six schemas field for field: types, required-ness, enums and the 2,097,152-byte limit. All 10 patterns end in `$(?!\n)`.

**What fails** (agent-reported; the items marked "verified" I checked in the files myself)

- **HIGH, verified: `target_value → forecast` is `1 : 0..*`.** This is in data-model.md:239, `erd.yaml:479` and `erd.mmd` (`||--o{`). It requires a target row for every forecast, but data-model.md:209 and ADR:275 require forecast rows at unobserved targets, and the truncation test depends on them.
- **MED, verified: `forecast_member.member` points to `fit.member`, but `forecast_member` has no `fold_id`.** That column is part of fit's key, so the stated FK to `fit(model_id, fold_id, member)` cannot resolve. See `erd.yaml:289` and data-model.md:210.
- **MED, verified: the `.mmd` headers say "GENERATED from … edit the YAML, not this file".** No generator exists, and `tools/diagrams/` is absent. `erd.yaml:2` itself says "hand-kept".
- **MED: the ERD's JSON entities drift from the schemas.**
  - It lacks `dataset` and `benchmark_model_id`.
  - It types `values` as `cube[]`, where the schema has an object.
  - It adds FKs that data-model.md does not state.
- **LOW-MED:**
  - `rv21_back`, `vov21` and `iv_rv_spread` are nullable in the ERD but NOT NULL in data-model.md (by its default rule).
  - `ivf.evaluation.splits` is named in `architecture.yaml:32` but nowhere in ADR-0001 or data-model.md.
  - `raw/{sha256}.csv` in `data-flow.yaml:26` drops `{source}/{symbol}`.

## Found without being asked

1. **Unhandled edge case:** `*_cal` produces `v̂ = a_h + b_h·G` by unconstrained OLS. A negative `v̂` makes `ŷ = 100·√v̂` and QLIKE undefined, and no rule covers it. LOW.
2. **Ambiguous wording at ADR:319 (L4):** "An h-session purge separates the last inner-train target from the first validation origin" can be read as a gap of h sessions after the last target. Non-overlap needs only `last target ≤ first validation origin`, which is a purge of h *origins*. LOW.
3. **Side effects during this review, none by the reviewing context:**
   - `app/dist/` and `app/coverage/` were rewritten at about 14:05Z by another process, after this ledger was frozen.
   - The C8 agent wrote 17 `.pyc` files into `.venv/…/yaml/__pycache__/`, then removed them.

## Required fixes

1. **C4.2.** Make the D2 rules atomic per scheme and version, and cover the remaining leak paths:
   - If any cell of a scheme keeps its old values, keep `all` and every cell of that scheme at their old values too.
   - Publish the null-label residual as its own cell, under the same suppression rules.
   - Either accept and document the two-point leak of telescoping statistics (RW h=1 bias; benchmark drawdown and CAGR), or suppress those statistics.
2. **C1.9.** State that MAPE is minimised below the mean (the "median of order −1") and so favours low forecasts, as the ADR already does for MAE. Then decide whether MAPE ranks plug-in mean forecasts at all.
3. **C2.6b.** Change ADR:335 from "so" to "assume". Add a state-dependent null to the M5 size simulation, and report DM results' sensitivity to the bandwidth.
4. **C8.** Change `target_value → forecast` to `0..1 : 0..*` (data-model.md:239, erd.yaml:479, erd.mmd). Add `fold_id` to `forecast_member`, or restate the FK. Remove the "GENERATED" headers.
5. **C5.4.** Write `docs/concepts/00-background.md`, or remove the six links and the PR/FAQ H1 source citation.
6. **C7.** Change "11" to "25" at milestone:61. Fix the `project.test.ts` (`:87` → `:103`) and `test_contract.py` (`:304` → `:315`) citations.

## Flags

- Two items were added after pre-registration:
  - C2.6b.
  - The C1.2 falsifier. My first falsifier ("drop T-weights") gave 13.9255, which also rounds to 13.93, so it was PROBE_INVALID and was replaced.
- The C1.3 path check was dominated by noise (PROBE_INVALID); only the analytic per-jump check counts.
- My pre-registered refutation of C2.6 failed. The premise holds when the two models are exchangeable under common GARCH vol.
- I compared designs with the authors' `ivf-design/dm_size.py` only after writing `channel-b.md`. It uses the same corr-0.8 construction.

## Not validated

- Legal sufficiency of D2: this is a technical attack only, not legal advice.
- MZ Wald size (O5).
- The authors' 20,000-run seed: I used my own seed, so the agreement is within Monte Carlo error, not exact.
- Publisher versions of DM (JBES 1995), Newey-West (1987), Gneiting (JASA 2011) and Carr-Wu (RFS 2009). Channel A cited the authors' working papers.
- Any VXX reverse split after the 13 Feb 2026 filing.
- The official Cboe definition of the VIX closing time.

## Appendix A: Reproduce-block output (run verbatim from the repo root, rc=0)

```
concept 01: 20.784609690826528 1.0009935422222145 | 13.927842240212474 | 42.60901173000046 22.612123756073984 |
            0.01136918592399945 22.360679774997898 20.634868149982054 | 0.47080444859994053 0.49242250247064234 | 20.640822228056894
concept 02: [0.9521, 0.5277, 0.0573, 0.0091, 0.008] | 14.015873015873016 0.8169601027735746 | 0.9648366685724423 |
            14.85 15.649999999999999 | 1.538752730835173 | 2.507593636441685
concept 03: 137 | SyntaxError | [NaN] 2.67 2.68 | -5.343281372915953e-12 0.0 1.1368683772161603e-13 |
            0.416 0.3974999999999991 | 3421073489 0.45037008942388645
```
