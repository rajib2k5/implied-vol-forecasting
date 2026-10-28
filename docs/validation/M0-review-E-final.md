# M0 review E: final attack on what reviews A-D left unattacked (evidence-gated ledger)

| | |
|---|---|
| Target | Working tree of `~/dev/implied-vol-forecasting`, 2026-10-28 16:08Z to 16:50Z. There are no commits, so there is no SHA. **The tree changed during the review.** Another session rewrote `README.md`, `docs/milestones/00-scaffold-and-design.md`, `docs/prfaq.md`, `LEARNING_LOG.md` and `.gitignore` between 16:39Z and 16:43Z. The ADR, `data-model.md`, `app/DATA.md`, `tools/export_app_data.py` and `app/scripts/guardDist.mjs` did not change. E1.a1 was re-run on the final tree. Checksums: `scratchpad/reviewE/shas_start.txt` and `shas_now.txt` |
| Environment | uv 0.11.25, Python 3.12.13, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, jsonschema 4.26.0. Node 22.23.1, npm 10.9.8, Vite 7.3.7, vitest 4.1.12, TypeScript 5.8.3. Linux 7.0.0. Pin: `scratchpad/reviewE/env-pin.txt` |
| Protocol | evidence-gated-validation. Tier 2 (dual channel) for E1 (the D2 argument). Tier 1 (run it, with falsifiers) for E2 (export guard), E3 (deploy guard) and E4 (suites and mutants) |
| Mode | Channel A: one subagent with no code execution, working from derivation, repo text and primary sources (`scratchpad/reviewE/channel-a.md`). Channel B: this context, execution only. **Caveat:** A's report reached this context before B ran its first probe. B's probe commands and pass conditions were pre-registered before A started (`claims-ledger.md`). B's verdict mapping is therefore not blind to A. Rows marked **ADDED** were designed after A reported. A disclosed two process slips: one `git log --oneline` (read-only), and a grep that printed a few `docs/validation/` lines A says it did not use |
| Default | Every claim started as "broken" |
| Side effects | This ledger is the only repo file written by this review. Every attack ran on scratch copies: `reviewE/g/repo`, `reviewE/m/repo`, `reviewE/a/app` and `reviewE/b/app`. **One slip:** a timing probe ran `python -I` in the repo. `-I` ignores `PYTHONDONTWRITEBYTECODE`, so it created `tools/__pycache__/` (gitignored). That folder did not exist at the start, and it was deleted again. The final checksum diff shows only the other session's five edits. No mounts were left (`mount \| grep scratchpad` = 0). Every scratch copy of a mutated file was restored and its sha256 re-checked |

`scratchpad` = `<session scratchpad>`. Every number below comes from a simulated path or from the calendar. No real market data was used, and none is quoted.

## Verdict

**D2 as written: FAIL.** No single-day guarantee survives anywhere, and P's rounding and table mapping are exact. But four claims the user would rely on are false or untestable:
- the stated barrier for scale-tied models;
- the "unknown scale" mitigation in residual 1;
- the information-bound margin;
- the objective's undefined "material part".

**Export guard: PASS** for review D's attack, which is now refused. Two new bypasses need a deliberate setup (LOW). **Deploy guard: PASS** for review D's 11 vectors (10 stopped, as claimed). New vectors either break the build, belong to documented classes, or are deliberate. One is undocumented (LOW). **Suites: PASS**: 365 pytest and 290 vitest tests. All 6 mutants were killed.

| # | Claim | Gate | Sev | Fix before M0 commit, or user's decision |
|---|---|---|---|---|
| E1.a1 | No surviving guarantee or old-rule text | **REJECTED** (dual) | LOW-MED (MED at start; partly fixed mid-review) | Fix |
| E1.a2 | The objective is precise | **REJECTED** (dual) | MED | User: define "material part" |
| E1.b1 | P closed, enumerated, rounded; reports use only P | Mapping **ACCEPTED**; "closed" **REJECTED** (dual) | LOW-MED | Fix wording |
| E1.b2 | Nothing real-data-derived outside P is committed | **REJECTED** (dual) | LOW | Fix wording |
| E1.c1 | Scale-equivariant models: P invariant to V → cV | **ACCEPTED** (dual) | – | – |
| E1.c2 | Scale-tied models leak no level (the SPX barrier) | **REJECTED** (dual) | **MED** (A: HIGH) | Fix text; user decides on the rows |
| E1.c3 | `*_cal` intercepts leak no level | **ACCEPTED** (dual) | – | – |
| E1.c4 | LSTM with train-only scaling is equivariant | **UNRESOLVED** (depends on M4's inputs) | LOW-MED | User/M4: inputs, then residual 2 |
| E1.c5 | Half-life leaks nothing material | **ACCEPTED** (dual) | – | – |
| E1.c6 | The ratio system pins no close or contrast | **UNRESOLVED** (A UNVERIFIABLE, B local evidence) | LOW | Residual 1 stands |
| E1.c7 | Information bound | **REJECTED** (dual) | LOW-MED | Fix or delete |
| E1.d1 | R1 coherent with git history | Git half **ACCEPTED**; enforcement **REJECTED** | LOW-MED | Fix: commit the probes |
| E1.e1 | Q coherent; no doc example violates it | No-violation half **ACCEPTED**; coherence half **REJECTED** (LOW) | LOW | Fix wording |
| E1.e2 | Q's boundary rule protects against anchoring | **REJECTED** (dual) | LOW-MED | Fix rationale |
| E1.f1 | The residual list is honest | **REJECTED** (dual) | **MED** (A: HIGH) | Fix |
| E2 | Export guard subfolder fix | Review D's attack **closed**; race and overlay bypasses | LOW (latent to M1) | User (residual) |
| E3 | Deploy guard | 10/11 **confirmed**; new vectors documented or deliberate, except V20 | LOW | Fix one doc line |
| E4 | Suites and 6 mutants | 365 / 290 pass; **0 survivors** | – | – |

## E1. ADR-0001 D2 (Tier 2)

### E1.a1: surviving old-rule text (REJECTED, dual)

- **B probe:** `grep -rniE` over the whole repo, excluding `docs/validation/` (`a1_grep.txt` at 16:1xZ, `a1_grep_final.txt` at 16:45Z). Falsifier: the same pattern finds 30 hits in the pre-fix ADR (`reviewD-fix/adr.before.md`).
- **No single-day guarantee survives.** Every "single day" mention is history: ADR:90, :181, :190; concept 03:116-131; milestone :297 and :329; LEARNING_LOG:67.
- **At the start** these were live old-rule text:
  - README:42-45: the old R4 "corrected only if 0 or at least 20 days … changed" and "rules R1 to R4";
  - README:91: "holdout RMSE by model" as a permitted figure, which is outside P;
  - milestone:56, :258, :358, :369. The §9 decision item asked the user to accept "the one-release rule R1 to R4 … (holdout RMSE by model, say)";
  - LEARNING_LOG:67;
  - .gitignore:24-26.
- **The other session fixed all of those during the review.**
- **Still live at 16:45Z:**
  - `docs/prfaq.md:64`: "losses and test statistics only (no bias, drawdowns or other signed contrasts), never refreshed". P is ratios, Sharpe/PSR/DSR, turnover and a half-life. R1 also allows a new release for a new `data_end`.
  - `docs/prfaq.md:114`: "refreshed releases".
  - `docs/concepts/00-background.md:54`: "only whole-period aggregates may be committed (ADR-0001 D2, R2–R3)". This is broader than P (it would admit a period mean) and cites deleted rules.
  - `docs/architecture/graph/GRAPH_REPORT.md:124, :315`: generated; still says "R1 to R4" and "cells of at least 20 dates".
  - `app/src/lib/architecture.ts:163`: comment only.
- **Channel A:** REFUTED, MED, on the start-of-review tree. It found the same list.

### E1.a2: "material part" is undefined (REJECTED, dual)

- **B:** "material" appears 4 times (ADR:86, :190; concept 03:106; DATA.md:44). None has a threshold, a count or a test.
- **A:** REFUTED, MED. "Reconstruct" is also undefined: exact, to a tolerance, or up to scale? No probe can test the objective as written, and residual 1 concedes it is unproven.
- **This is the user's call.** Example: "no window of 5 or more consecutive closes recoverable within ±1 vol point".

### E1.b1 and E1.b2: closure (mapping ACCEPTED; "closed" REJECTED)

- **Mapping, B (`b1_closure.py`).** All 9 P rows carry a rounding. All 19 value columns in data-model "Committed reports" map to a P row with the same rounding. Falsifier: one added `mean_vix (2 dp)` column gave `not mapped: 1`.
- **Mapping, A.** Same result, checked row by row. **ACCEPTED.**
- **"Closed" fails (both channels):**
  - ADR:132: "EDA findings in prose must be qualitative **or unitless**". Skew and kurtosis are unitless, yet the same paragraph (ADR:130) bans them. The clause also admits ACFs, ADF/KPSS p-values and regime frequencies (which are regime cells) into prose.
  - ADR:130-131 bans "AR φ", but P publishes the half-life `ln 0.5 / ln φ`, a bijection of φ.
  - The DM "benchmark" is not named in P (ADR:121). The error surface uses `rw`, while prfaq:19 names two bars, rw and HAR.
- **E1.b2, outside P.** `reports/provenance.json` commits the raw-file sha256 list (data-model:273; ADR:253 says "no data values"). `run_id` hashes that list, and `data_end` is chosen from data presence. Harm is LOW: a hash only confirms a guessed file. But it falsifies "the closed list of every real-data-derived number" (ADR:111-112).
- Also, `.gitignore` and `tests/test_gitignore.py:108` keep `reports/regimes/deep/table.agg.csv` committable. The allow-list is broader than P's five files (LOW).

### E1.c1: equivariant models (ACCEPTED, dual)

**B:** re-ran `p_set.py` and `p_falsify.py` unchanged (`reviewE/c1/`, sha256 identical to the review D copy):

```
(b) scale c=0.8: P entries that differ (V-equivariant models + strategy) = 0 of 114
(b) scale c=1.25: ... = 0 of 114     (b) scale c=2.0: ... = 0 of 114
(b') with the SPX-driven model included, c=1.25: 22 differ, all for model 'ewma_spx'? True
(c) delta=0.25 vol pts: P identical under +/-delta on 5/604 single days (0.8%)
old-R3 statistic under c=1.25 -> eda_sd 7.166->8.957 CHANGED, rmse_rw 1.485->1.857 CHANGED, mz_beta 0.983 same
```

**A:** the lemma holds for ratios, DM (squared error and QLIKE), the MZ Wald test and the half-life. Two caveats:
- a band in vol points in `sign(ŷ − V_t) with band` (data-model:100) breaks equivariance;
- so does an absolute floor on `v̂`.

### E1.c2: scale-tied models leak level statistics given public SPX (REJECTED, dual)

**The claims under test.** ADR:174-176 and residual 2 (ADR:194-195) say that using the VIX-SPX tie "needs the SPX return series, which is vendor data and is not committed".

**Channel A (primary source).** That is no barrier: FRED SP500 publishes "Daily, Close" with 10 years of history (fred.stlouisfed.org/series/SP500, retrieved 2026-10-28). Reading it is not redistributing it.

**Channel B (`reviewE/c2/p_tie.py`, `p_tie_seeds.py`; simulated).** The rv21 track is `rw` (= `rv21_back`, a pure SPX function), `vix_implied` (`v̂ = (V_t/100)²`) and an EWMA stand-in for GARCH. An attacker who holds the SPX returns computes the random walk's absolute losses exactly. Multiplying the committed ratios back gives absolute level-unit statistics of VIX:

```
c=0.8: 9 of 18 rv21 entries differ; models: ['vix_implied']   FALSIFIER (vix_implied removed) c=1.25: 0 of 9 differ
recovered absolute RMSE(vix_implied) = 4.4028 vol pts (true 4.4007); MAE = 3.4275 (true 3.4266)
10 seeds: RMSE(V - rv21) recovered, median |err| 0.053%; holdout mean V: est1 (mean y + MAE) median 9.58%,
          est2 (V ~ c * EWMA vol, c from the published QLIKE) median 4.29%, max 13.69%
```

- The pre-registered survival bar was a mean-level error above 20%. Every seed came in under it.
- The pre-registered MAE-based estimator 1 is crude. Its bias is `2·mean((y − V)+)`, and it is exact when VIX ≥ subsequent RV on every origin; the simulation's VRP is weak (V ≥ y on only 60-79% of origins).
- **What leaks:** period-level VIX statistics, namely the mean level and the RMS VIX-RV gap. No daily path leaks. The objective says isolated closes are "not the harm", which is why I put severity at **MED**. A says HIGH.
- **The text is wrong and must change before the commit.** The user should then decide either to drop `vix_implied` (rv21) and the uncalibrated `garch`/`gjr`/`egarch` (`vix_level`) rows from P, or to state what they reveal.
- **Where A and B diverge:** A also calls ADR:159-161 ("the attacks' inputs are absent … absolute RMSE") false. B finds that sentence true as scoped: the inputs of review D's `p1_shift` and `p2_r3` (`vix_level` EDA moments, MZ β/R², `vix_level` RMSEs) are still absent. The recovered absolute RMSE is a different quantity.

### E1.c3: `*_cal` intercepts (ACCEPTED, dual)

- **A:** OLS gives `(a, b) → (c²a, c²b)` under V → cV, and `b → b/s²` under SPX r → s·r, so `v̂` is equivariant in V and invariant in SPX.
- **B:**

```
h=1: 9 entries; changed over 3 V-scales: 0; over 2 SPX-scales: 0; FALSIFIER fixed a=0.01, c=1.25: 7 changed
h=5: ... 0; 0; FALSIFIER 6 changed       h=21: ... 0; 0; FALSIFIER 8 changed
```

- **Caveat (A):** the docs do not say how a non-positive `a + bG` is floored. An absolute floor would break equivariance.

### E1.c4: LSTM (UNRESOLVED: depends on M4's input choice)

- The docs never list `lstm_log`'s inputs.
- **B** used a tiny float32 net with train-only z-scaling:

```
inputs none         : 0 of 81 entry-comparisons changed (3 horizons x 3 scales)
inputs rv21_back    : 0 of 81 changed      <- pre-registered falsifier: PROBE_INVALID (it could not fail)
inputs iv_rv_spread : 65 of 81 changed     <- ADDED, after A derived that this input breaks equivariance
inputs none, float64: 0 of 81 changed
```

- So **A was right** that `rv21_back` cannot break it and `iv_rv_spread` (a column of the `feature` table) does.
- **For M4:** if `lstm_log` takes `iv_rv_spread`, a fixed-threshold regime flag, or an absolute clip, it is scale-tied. Residual 2 (ADR:194) must then list it.

### E1.c5: half-life (ACCEPTED, dual)

- **A:** 2 s.f. over 5-500 sessions gives 181 values, 7.50 bits; the measure is shift-invariant.
- **B:** `p_sens.py` reproduces the ADR's flip shares exactly (dm_stat 98.8%, dm_p 85.9%, rel_qlike 85.4%, strategy 2.0%). The half-life flipped on **0 of 604** days at ±0.25.
- The only problem is ADR:131's "not AR φ" (see b1).

### E1.c6: does the ratio system pin anything? (UNRESOLVED)

**A:** UNVERIFIABLE. Scale invariance removes only the all-ones direction, so close ratios such as `V_e/V_s` get no protection from it. The telescoping term sits inside every `rel_qlike` denominator.

**B** (`reviewE/c6/p_jac*.py`): a numerical Jacobian of unrounded equivariant P with respect to ln V, every day unknown (73 entries, rank 55, 3,084 unknowns). The residual is the share of a direction outside P's row space: 1 means invisible to P, 0 means pinned to first order.

```
all-ones residual 1.000000 (expect 1: scale invariance)
seed 20261028: lowest single-day residual (3062 = day after the last origin, 0.0974), next 0.3981; median 0.9991
               h=1 ln(V_e/V_s) residual 0.5815; rounding blur on ln V_e (linearised): worst 0.177, rms 0.060
seed 1: lowest 0.4528 (day 2570, not a boundary); ln(V_e/V_s) 0.8178; blur worst 1.42
seed 2: lowest 0.4234 (day 2974, not a boundary); ln(V_e/V_s) 0.8444; blur worst 0.52
FALSIFIER (a row equal to the telescoping term added): h=1 endpoint residual 0.000000
```

- On one path, one boundary day is nearly identifiable to first order. On two other paths it is not, and the most exposed days are interior.
- Rounding blurs every linear read-out by 6-48% in log units, and the read-out also needs the rest of the path.
- **Nothing material is demonstrated.** "Pins nothing" is not proven either, which is exactly residual 1. Left to the user as stated.

### E1.c7: information bound (REJECTED, dual)

**B** (`c7/p_bits.py`):

```
non-rw rows 38: ratios 152, DM pairs 76, MZ booleans 84; P bound = 3488.8 bits (ADR: ~3,500)
ADR denominator 583*13 + 583*11 = 13992 (ADR: ~14,000)
simulated holdout: sd of daily change 1.573 -> Gaussian max-entropy rate 9.34 bits/close; 583 closes -> 5448 bits
ratio P bound / VIX-only entropy (sim) = 0.64; ADR's implied ratio = 0.25
```

- The arithmetic and the counts are right.
- **A:** the measure is wrong. 13 bits is a close's marginal entropy, but the series carries its entropy rate. The Gaussian is the maximum-entropy case, so 9.34 is an upper bound and the true margin is at most 1.6×.
- SPX bits do not belong in the denominator, because SPX is public.
- "P cannot encode the series" still holds, but the margin is about 1.6×, not 4×. A capacity bound also says nothing about "any material part": a quarter of closes is about 550 bits.

### E1.d1: R1 (git half ACCEPTED; enforcement REJECTED)

- **Coherent with git (both channels):** withdraw-only fits append-only history, and no commit exists yet.
- **Not enforced (both channels):**
  - no test or CI step fails if a `reports/*.agg.csv` value changes after it is first committed;
  - R1's own safeguard, "rerun the P probes below" (ADR:140-141), points at probes that live only in this session's `/tmp` scratchpad ("not in the repo", ADR:154). They use simulated data only and can be committed.
- **Nested releases.** A derives that a new `data_end` isolates the new block's mean loss to 0.9-9%. That holds for absolute losses. P publishes ratios, whose unknown denominators do not cancel, so B disputes the figure. R1 defers this to a future ADR anyway. UNRESOLVED, LOW.

### E1.e1 and E1.e2: rule Q

- **No committed doc quotes a dated level (B grep).** ADR:145's example, 2024-08-05, is 24 sessions after `holdout_start` and 25 after the last dev origin (`numpy.busday_count`, Jul-4 holiday). That is outside the 21-session window. A planted 2024-07-15 is at 9 and would be flagged. **ACCEPTED.**
- **Coherence (LOW).** "The last origin of the holdout" depends on the horizon (h = 1 vs 21), and `data_end` is not frozen yet.
- **E1.e2 (REJECTED, dual).** Review D's `p1_shift.py`, re-run, takes the anchor as a plain number (`attack(c, n, Vs_anchor)`), so where the anchor came from does not matter.
  - ADR:88 calls closes "public facts that the news reports", and A cites FRED VIXCLS.
  - The rationale at ADR:149-152 ("a boundary quote would anchor …") therefore protects nothing.
  - Q is harmless as minimisation. Its stated reason is wrong.

### E1.f1: residual honesty (REJECTED, dual)

Residuals 3-5 are honest. These are not:
1. **Residual 1 (ADR:192-193), "unknown scale".** False for the full P: `vix_implied` (rv21) plus public SPX pins the scale (E1.c2), and so does any public close.
2. **Residual 2 and ADR:174-176, "needs the SPX series".** That is no barrier (E1.c2).
3. **Consequences (ADR:569-570), "licence-clean by construction: no file in the repo tree ever holds real daily data".** The guards are tripwires with bypasses (E2, E3) and their own residuals (ADR:226-229, 237-240).
4. **ADR "Guards" (ADR:230-231), "fails on any symlink under app/".** Top-level `node_modules/` is skipped, and V20b (below) passes with the marker.

## E2. Export guard subfolder fix (Tier 1)

`reviewE/g/e_guard.py` ran against the scratch copy, with real mode enabled by the same test-only `generate` patch review D used. Mount cases ran under `unshare -rm`.

```
B3 bind of repo/app/public at outside path: rc=3                              <- review D's bypass, now refused
R1 repo folder created+bound after the walk: rc=0 landed in repo=True       <- race
R1 control (folder existed before the walk): rc=3
O1 overlay, upper layer inside repo, merged outside: rc=0 landed in repo=True
O2 overlay, lower = repo/app/public, upper outside: rc=0 landed in repo=False landed in upper=True
U  repo folder with mode 000 / 311 / 644: rc=3 / 3 / 3
U  folder deleted mid-walk (before its stat): rc=EXC FileNotFoundError ...  <- uncaught, but nothing written
L1 symlink loop a->b->a and a link to '..' inside repo: rc=0 (0.00s)
D1 depth 300: rc=0 (0.01s)      D2 depth 2500 (path > PATH_MAX 4096): rc=3 (0.25s)
P1 101353 folders in the repo copy: _directories_inside 0.91s   (real repo: 5,931 folders, 0.103 s)
```

| Attack | Result | Severity (latent until M1, single-user machine) |
|---|---|---|
| Review D's bind mount of `app/public` | Refused | Closed |
| Race: a repo folder created and bound in the window between the walk and the check | Written into the repo | LOW. Needs a deliberate concurrent mount in a window of about 0.1 s. It is the TOCTOU class ADR:226 accepts |
| Overlay whose upper (writable) layer is inside the repo | Written into the repo | LOW. Needs a deliberately built overlay and mount privilege. Not covered by "bind mount" in ADR:223 |
| Overlay with the repo as the read-only lower layer | Safe: the write lands outside | – |
| Unreadable folder (000, d-wx, r--); path > PATH_MAX | Refused, rc 3 | Fail-closed |
| Folder deleted mid-walk | Uncaught `FileNotFoundError` traceback, nothing written | LOW (cosmetic) |
| Symlink loop; 100k folders | Correct; 0.9 s | – |
| FUSE (bindfs or similar) | **UNVERIFIABLE**: no FUSE tool installed | – |

## E3. Deploy guard (Tier 1)

`reviewE/b/p_e3.sh` ran on a scratch copy of `app/`. For each vector: `tsc -b`, `vite build`, then `guardDist`, then a grep of `dist/` for the sentinel.
- **E3.1:** review D's V1-V8, V10 and V11 fail the guard; V9 (`VITE_*`) passes. **10 of 11 stopped, as claimed.**
- **E3.2, new vectors:**

| Vector | guard | Reaches dist | Judgement |
|---|---|---|---|
| V2b markerless JSON in `app/realdata/`, imported | 0 | yes | Documented (DATA.md:107). The realistic M1 shortcut |
| V12 real values relabelled `"synthetic"` in `src/data/` | 0 | yes | Deliberate; documented. Nothing detects provenance in `src/`: vitest 290/290 passed. The same file in `public/data/synthetic/` **is** caught by `export_app_data.py --check` ("unexpected: … not produced by this exporter", rc 1), which pytest runs |
| V13 `.json5` in `src/lib`, `?raw` | 0 | yes | Deliberate (`?raw`); documented class |
| V13b `.jsonc` in `src/data`, `?raw` | 1 | – | Stopped (data-file-not-json) |
| V14 `.geojson`; V18/V18b `.txt`/`.dat` in `public/` | 0 | yes | Documented class (non-JSON in `public/` outside `public/data/`). Realistic for a dropped vendor file |
| V15 `.mjs` data module | 0 | yes | **Blocked by the build**: `tsc` rc 2, so `npm run build` and `verify` fail |
| V16 `new URL('../realdata/x.json', import.meta.url)`, < 4 KB | 0 | base64 data URL in the JS | Documented class (markerless JSON outside `src/`). Note: the > 4 KB version (V16b) **is** caught, as `dist/assets/vix-*.json` |
| V17 `?url` import of a small markerless JSON | 0 | base64 in the JS | Same as V16 |
| V19 `public/Data/vix.csv` (case variant) | 0 | yes | LOW. The data-dir rule is case-sensitive |
| **V20 / V20b** `node_modules/ivf-real` → outside folder (`npm link`), **with the marker** | **0 / 0** | yes | **Undocumented.** It contradicts ADR:230 "any symlink under app/". LOW: the Docker build would fail (`npm ci` drops the link), so this leaks only through a local `dist/` upload |

## E4. Suites and mutants (Tier 1)

- **pytest.** `PYTHONDONTWRITEBYTECODE=1 uv run --locked pytest -q -p no:cacheprovider`, in the repo: **`365 passed in 13.60s`**.
- **npm run verify.** Run on a byte-identical copy (`reviewE/a/app`), existing lockfile, no install: rc 0.
  - eslint passed.
  - vitest: **`Test Files 19 passed (19)`, `Tests 290 passed (290)`**.
  - Coverage: **98.52 / 95.92 / 98.41 / 99.38**.
  - Guard: `scanned 5 file(s) in dist/ and 37 elsewhere … 3 JSON file(s), all synthetic`.

| M | Mutation (newest code) | Result |
|---|---|---|
| M1 | `_directories_inside` returns only the root's identity | KILLED (3 failed) |
| M2 | `os.walk` without `onerror` | KILLED (1 failed) |
| M3 | drop `os.chmod(temporary, existing_mode)` | KILLED (1 failed) |
| M4 | `notSyntheticReason` accepts any object | KILLED (10 failed) |
| M5 | symlink test only for `.json` names | KILLED (7 failed) |
| M6 | drop the data-dir non-JSON rule | KILLED (2 failed) |

**No survivors.** One note: M2's only killer is `skipif(os.geteuid() == 0)`, so M2 would survive a root CI runner (LOW; there is no CI config yet). Scripts and outputs: `reviewE/m/run_py_mutants.py`, `py_mutants_full.out`, `reviewE/b/run_js_mutants.py`, `js_mutants.out`.

## Found without being asked

1. **The P probes are not in the repo** (MED). ADR:154 cites `p_set.py` "(not in the repo)", and R1 requires rerunning it. It lives in a session `/tmp` folder that will be deleted. It uses simulated data only, so commit it (with `p_sens.py` and `p_falsify.py`).
2. **Review ledgers are committed docs.** They fall under D2 too: none may quote a real-data-derived number outside P. This one quotes none. Reviews C and D were not re-audited for this.
3. **`reports/**/*.agg.csv` is committable at any depth**, including a `regimes/` path that `tests/test_gitignore.py:108` requires (LOW). P names five files; the allow-list could name them.
4. **`_directories_inside`'s timing claim** ("about 0.07 s … about 5,900 folders") re-measured at 0.103 s for 5,931 folders. Fine.

## Classification

**Real defects to fix before the M0 commit** (text, all cheap):
- ADR D2: residual 1's "unknown scale", residual 2's and ADR:174-176's SPX "barrier", the "Information bound" bullet, Q's rationale (ADR:149-152), "licence-clean by construction" (ADR:569), "any symlink under app/" (ADR:230), "or unitless" in ADR:132, and "AR φ" in ADR:131.
- Commit the P probes.
- Stale text: prfaq:64 and :114, concept 00:54, `GRAPH_REPORT.md` (regenerate).
- Name the DM benchmark in P.

**Documented residuals the user decides on:**
- Define "material part".
- Keep or drop the scale-tied rows (`vix_implied` on rv21; uncalibrated GARCH on `vix_level`), now knowing they reveal period-level VIX statistics given public SPX.
- M4's LSTM inputs.
- An R1 enforcement check.
- Residual 1 (E1.c6: local exposure exists, recovery not shown).
- Export-guard race and overlay (LOW).
- Deploy-guard documented classes, plus V19 and V20 (LOW).

## Not validated

- Legal sufficiency of D2.
- Real data: every E1 number is from a simulated path. The VRP share and real holdout length would change the c2 error bands, not the identities.
- FUSE mounts; macOS/APFS and Docker Desktop mounts; the Docker image build (reasoned about, not built).
- Channel A's FRED citations (SP500, VIXCLS) were not re-fetched by B.

## Single most important fix

Rewrite D2's scale argument (residuals 1-2, ADR:169-176). Daily S&P 500 closes are public. With them, P's `vix_implied` (rv21) ratios give the absolute RMS VIX-RV gap to about 0.05%, and the holdout mean VIX to a few percent (simulated, 10 seeds). The "unknown scale" mitigation and the "SPX is not committed" barrier are both false. Then let the user decide whether those rows stay in P.

## One item to delete

The "Information bound" bullet (ADR:182-187). It plays the role the deleted guarantee played: a reassuring number. It is computed on the wrong measure (marginal entropy, plus SPX bits that are public), so its margin is about 1.6×, not 4×. And it says nothing about the "material part" the objective is about.
