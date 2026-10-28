# M0 review D: attack on the fixes for reviews A and B (evidence-gated ledger)

| | |
|---|---|
| Target | Working tree of `~/dev/implied-vol-forecasting` on 2026-10-28, 14:47Z to 15:35Z. No commits exist, so there is no SHA. **The tree changed during the review** (another session edited the ADR, README, milestone doc, `docs/architecture/README.md` and added `docs/architecture/graph/`). D2 R1-R4 did not change. Start and end checksums: `scratchpad/reviewD/shas_start.txt`, `shas_now.txt`, `shas_end.txt` |
| Environment | uv 0.11.25, Python 3.12.13, numpy 2.5.4, statsmodels 0.15.0, jsonschema 4.26.0; Node 22.23.1, Vite 7.3.7, vitest 4.1.12. Pin file: `scratchpad/reviewD/env-pin.txt` |
| Protocol | evidence-gated-validation. Tier 2 (dual channel) for C1 (licensing) and C5 (statistics); Tier 1 (run it) for C2, C3, C4, C6, C7 |
| Mode | Channel A = two separate no-execution subagents (C1: derivation from the ADR and data model only, barred from `docs/validation/` and the scratchpads; C5: primary sources and installed statsmodels source). Neither saw any Channel B output. Channel B = this context, execution only. **Caveat:** B's probes were pre-registered (14:55Z) before A reported, but were run after A's reports reached this context, so B's verdict-mapping is not blind to A |
| Ledger | Pre-registered in `scratchpad/reviewD/claims-ledger.md` and `mutants.md` before any probe ran. Rows added later are marked **ADDED** |
| Default | Every claim started as "the fix is broken" |
| Side effects | The only repo file written is this ledger. All attacks ran on scratch copies (`scratchpad/reviewD/repo`, `scratchpad/reviewD/appc`). `uv run pytest` ran in the repo with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`; the end checksum diff shows no file changed by this review. `npm run verify` ran on a byte-identical copy of `app/` (`diff -rq` rc 0, excluding `node_modules`, `dist`, `coverage`), so the repo's `app/dist` was not rebuilt |

`scratchpad` = `<session scratchpad>`. Raw outputs are in `scratchpad/reviewD/p*.txt`, `channel-a-c1.md`, `channel-a-c5.md`, `mutant_results.json`.

## Verdict: FAIL

| # | Fix claim | Verdict | Sev |
|---|---|---|---|
| 1 | D2 one-release rule: no single day recoverable by linear differencing; R3 removes every telescoping statistic | **REFUTED** (dual) | **HIGH** |
| 2 | Export guard by `(st_dev, st_ino)`, `.git` ancestors, `O_EXCL` + `os.replace`, link refusal | **REFUTED in part**: review A's attacks are closed; a bind mount of a repo *subdirectory* still writes into the repo | MED (latent until M1; LOW realism) |
| 3 | `.gitignore` covers vendor data and secrets | **REFUTED in part**: review A's own `notebooks/fred.json` is still not ignored, plus other realistic names | MED (live now) |
| 4 | Deploy guard keeps real data out of `dist/` | **REFUTED for realistic accidents**: 7 of 11 vectors reach `dist/` with the guard green | MED |
| 5 | Statistics fixes (consistency table, DM paragraph, statsmodels) | **SURVIVES** (dual), with two LOW wording items | LOW |
| 6 | ERD fixes | **SURVIVES** for the three named fixes; "yaml ≡ mmd ≡ data-model" **REFUTED in part** | LOW-MED |
| 7 | Test suites | **SURVIVES**: 323 pytest, 248 vitest, verify rc 0. One non-equivalent mutant of the new contract walk survives | LOW |

## Claim 1: D2 licensing (Tier 2)

### The attack (C1.1, C1.2): shifted sets telescope in second moments

The ADR's proof says each per-day quantity `g` is summed over one fixed set per period. But EDA statistics are per **session** over `O_p`, while forecast statistics are per **origin** over `O_p` with targets dated `t + h`. For the random walk (`ŷ_t = V_t`) at h = 1, `V_d²` is summed over `O_p` (EDA sd) and over `O_p + 1` (the targets, through MZ). Every input below is listed as committed by ADR R3 and data-model "Committed reports":

- `S_xx = (n−1)·sd²` from the EDA sd of V over `O_p`;
- `S_yy = β²·S_xx / R²` from the random walk's MZ β and R² at h = 1;
- `S_yy − S_xx = (V_e − V_s)(V_e + V_s − 2V̄) − (V_e − V_s)²/n`: a function of the first origin `V_s` and the day after the last origin `V_e` only;
- with the random walk's RMSE as a third equation, `(V_e − V_s)² = n²·RMSE² − n·(S_xx(1 − 2β) + S_yy)`, and both closes follow without any quoted value (two candidates; Channel A shows the sign is fixed because dev's `V_e` and the holdout's `V_s` are the same day, 2024-07-01).

Probe `p1_shift.py` (simulated log-AR(1) VIX path, holdout n = 562, closes at 2 dp), verbatim excerpts:

```
seed=20261028  true V_s=24.30 V_e=13.77
identity check: Syy-Sxx (true) = 73.464737   two-point formula = 73.464737
[full precision] (b) no anchor: (V_s, V_e) candidates = [(20.765, 31.295), (24.3, 13.77)]
[4dp (mean 4dp)] f recovered = 77.279293 (two-point truth 73.464737, error +3.8146)
  (a) one anchor V_s=24.30: V_e candidates = [31.505, 13.554]  (true 13.77)
  (b) no anchor: a^2 = -715.0027 (true 110.8809); (V_s, V_e) candidates = []
seed=1  [4dp] (a) one anchor: V_e candidates = [25.634, 9.114] (true 25.48)
seed=2  [4dp] (a) one anchor: V_e candidates = [21.294, 10.939] (true 10.88)
FALSIFIER (historical-mean forecast instead of V_t): f recovered = 842775.515283 vs truth 73.464737
```

- **At full float precision** (the pandas `to_csv` default; no document sets decimals for `reports/*.csv`, the "Rounding at export" table covers app JSON only) two dated closes are recovered exactly from five committed numbers, with no quote. Channel A derived the same, plus a third date (2000-03-15) from the dev period.
- **At 4 dp** the no-anchor solve fails (catastrophic cancellation), but one quoted close at a boundary date (2024-07-01 is the holdout start) gives the other to within 0.06-0.22 vol points in 3 of 3 seeds. ADR line 91 allows dated index quotes *because* "no committed statistic is a two-point contrast of an index that a quote could anchor". That premise is false.
- Channel A (derivation, independent): REFUTED, HIGH; also shows 8- and 10-day block contrasts at h = 5 and 21, and that dev targets fall on holdout sessions, so "the two periods' sets are disjoint" holds for origins only.
- **Gate: REJECTED (dual).** The ADR "Guarantee" (lines 130-141), R2's sentence "The same per-day quantity is therefore never summed over two different committed sets" (line 126), concept 03:148 and README:42 ("Nothing reconstructable is published") are all false as written.

### C1.3: R3's exclusions are cosmetic (probe `p2_r3.py`)

```
(i) DM mean_d = -0.2277728011; RMSE_a^2 - RMSE_b^2 = -0.2277728011; from 4dp RMSEs: -0.227620
(ii) QLIKE_rw = 0.009588052144; mean((V_t+1/V_t)^2) - 2 ln(V_e/V_s)/n - 1 = 0.009588052144
(iii) mean r = -0.0024819995; sharpe_ann*ann_vol/252 = -0.0024819995
falsifier (i), different row sets: mean_d = -0.1690387063 vs -0.2277728011
```

- DM `mean_d` is not committed but equals the difference of two committed squared RMSEs, exactly, with sign.
- The random walk's committed QLIKE contains the telescoping term `−2·ln(V_e/V_s)/n`. It is not isolable by itself (no committed statistic gives `Σ(V_{t+1}/V_t)²`), so this is a counterexample to "R3 removes every g that telescopes", not a leak on its own.
- The strategy mean return is recoverable (and data-model commits `ann_return` directly, which ADR R3's committed list omits).
- Channel A adds: the random-walk `bias` and MZ α are recoverable at full precision through the C1.1 solve. **Gate: REJECTED (dual), MED.**

### C1.4: R4 corrections (Channel A derivation, plus probe `p12_r4.py`, ADDED)

R4 is "checked by diffing the local forecast tables". A vendor restatement of the close on the first session after the last scored origin changes a **target**, not a forecast row:

```
forecast rows (origins in O_p) that differ between releases: 0 -> R4 as checked in the ADR: 0 changed days, replacement allowed
P1 no-anchor solve, old release (V_s, V_e) candidates: [(18.0, 19.64), (22.7936, 21.1536)]  truth: (18.0, 19.64)
P1 no-anchor solve, new release (V_s, V_e) candidates: [(18.0, 21.01), (22.7936, 19.7836)]  truth: (18.0, 21.01)
```

- R4's check procedure counts 0 changed days, so the cell would be replaced, and both the old and the new close are recoverable (full precision).
- "Changed" has no size threshold, and the EDA, strategy and data-check cells, which do not come from forecast tables, have no defined check.
- R1 ("never rewritten or partially updated") contradicts R4 ("may be replaced").
- **Gate: REJECTED. MED.**

### C1.5: review B's attacks re-run (`p3.txt`)

Review B's `c4_licensing_attack.py` and `cross_scheme.py` reproduce their output byte for byte (A1 `V_E2 = 18.394 (true 18.38)`, A2 `-0.6154` vs `-0.6300`, A3 `1.7102` vs `1.7000`, A4 `-10.8614` vs `-10.8600`). Each needs an input that R1-R3 now forbid: `bias` (R3), a refresh (R1), regime cells (R2). **SURVIVES** for those four attacks as constructed. A1's telescoping is reopened through second moments (C1.1).

**Strategies (Channel A only, conditional, MED).** Two committed strategies whose positions differ on fewer than 20 days give `(N/252)·Δann_return + cost terms = Σ_{D} Δw·R`, a strictly linear combination that survives 4-dp rounding (about ±2e-4 on one daily return). The days in `D` are not dated by anything committed. No rule requires committed strategies to differ pairwise on 0 or at least 20 days. **Gate: UNRESOLVED.** There is no B probe: real signals are needed to judge realism.

### C1.6: the docs do not all state the same rule

Checked on the tree at 15:29Z. Contradictions:

| Where | Says | Conflict |
|---|---|---|
| `app/DATA.md:42` | Aggregates may be committed "if every cell aggregates at least 20 dates. Smaller cells are blanked" | The superseded per-cell rule |
| `app/src/lib/architecture.ts:19` | `real-aggregate`: "every cell covers at least `min_cell_n` dates" | Same; this text drives the app's architecture page |
| `schemas/app-data/manifest.schema.json:87` | "aggregate = cells of >= min_cell_n dates" | Same; the ADR now says the aggregate (error surface) is never committed |
| `docs/architecture/specs/architecture.yaml:36,99`, `architecture.mmd:21,77`, `docs/architecture/README.md:47,103` | `ivf.reports` → `reports/`: "n >= 20, else blank" | Omits R1-R4; describes the old per-cell gate |
| `docs/prfaq.md:101` | DM tests "for every model against both benchmarks (the two benchmarks against each other)" at h = 1, 5, 21 | ADR R3 and data-model: DM "against the benchmark model only"; rv21 models have h = 21 only |
| `README.md:43` | `reports/` "holds losses and test statistics only" | R3 also commits EDA level means, Sharpe, vol, skew, kurtosis, turnover, `ann_return` |
| ADR R3 committed list vs data-model `strategy_summary.csv` | ADR omits `ann_return`, `sr_daily`, `e_max_sr`, `cost_bps` | data-model commits them |
| ADR R1 vs R4 | "never rewritten or partially updated" vs "may be replaced" | Internal |

`docs/milestones/00-scaffold-and-design.md` still described the old refresh rule at the start of this review (line 66, "refreshes must add at least `min_cell_n` new dates per changed cell"). It was rewritten by another session during the review and is now consistent. **REFUTED, MED.**

## Claim 2: export guard (Tier 1)

**Review A's `attack_guard.py`, run unchanged against a current copy (`p5_attack_guard.txt`):**
- A1-A8: unchanged (refused, or correctly accepted when physically outside).
- **B1** symlinked target: `rc=3 repo fixture overwritten=False`. **B2** hard-linked target: `rc=3 … overwritten=False`. Both previously gave rc 0 and overwrote the fixture.
- B3 (TOCTOU: a parent becomes a symlink between the check and the write) still lands: `rc=0 file landed in repo=True`.
- Falsifier: the same script against the pre-fix exporter (`scratchpad/export_app_data.py.orig`) gives B1/B2 `rc=0 … overwritten=True` (`p5_falsifier.txt`).

**New attacks (`p6_guard.py`, `p6_casefold.sh`, under `unshare -rm` where a mount was needed):**

| Attack | Result |
|---|---|
| `..` after a symlink; out-dir symlink to a folder whose ancestor is the repo; `nonexistent/..` then into the repo; `VOL_DATA_DIR=<repo>/app` | REFUSED |
| Bind mount of the repo root; existing subfolder through it | REFUSED (review A's bypass is closed) |
| Case variants `…/ivf`, `…/Ivf` of repo `…/IVF` on a casefold tmpfs | REFUSED (review A's bypass is closed) |
| NFD spelling of an NFC folder on ext4 | ACCEPTED, correctly: a different, non-existent name |
| FIFO at the target | Replaced by a regular file, rc 0, nothing read from the FIFO |
| `/dev/null` symlink at the target; hard link to a non-repo file | REFUSED (rc 3) |
| Directory at the target name | REFUSED, but with the message "it has 2 hard links" (misleading; LOW) |
| Read-only out dir | Uncaught `PermissionError` traceback, no temp file left (LOW) |
| Mode bits | New file 0o644 at umask 022 and 0o600 at umask 077. **An existing 0o600 file becomes 0o644** after a re-export: the docstring's "like write_bytes did" holds only for new files (LOW) |
| **Bind mount of `repo/app/public` at an outside path** | **ACCEPTED**, `rc=0 landed in repo=True`, `repo files changed: ['…/repo/app/public/data/real/manifest.json']` |

- **Root cause** (`tools/export_app_data.py:181-188`). The walk compares each *path* ancestor with the repo root's `(dev, ino)`. A mount of a subfolder never exposes the repo root, or a `.git` entry, as a path ancestor.
- `docs/milestones/00-scaffold-and-design.md:131` and the `_identity` docstring both say bind mounts are handled. Only a bind of the root is.
- **Realism is LOW** (a container volume or devcontainer mount of a repo subfolder), and the attack is latent until M1 adds a real generator.
- **TOCTOU** is documented in `tools/export_app_data.py:246-250` only. ADR D2 "Guards" still says "Real JSON is never written inside the repo tree".
- **Gate: REFUTED in part (MED claim, LOW realism).**

## Claim 3: `.gitignore` (Tier 1)

`git -c core.excludesFile=/dev/null check-ignore -q --no-index` on a pre-written list (`p7_names.txt` → `p7_out.txt`). `core.ignorecase` is unset (false on Linux).

- **Still committed, from review A's own list:** `notebooks/fred.json`. FRED's API returns JSON.
- **Still committed, vendor data:**
  - `VIX_HISTORY.CSV` (upper-case extension);
  - `vix_history.json`, `notebooks/vix.json`, `data.json`, `vix.jsonl`;
  - `VIX_History.txt`, `.tsv`, `.zip`, `.csv.bz2`, `.csv.xz`, `.csv.zst`;
  - `prices.npy`, `.npz`, `.joblib`, `.arrow`, `.pq`, `.ods`, `.xlsm`;
  - `notebooks/explore_vix.ipynb` (a new notebook with outputs; `build_notebooks.py --check` catches it only in CI, after the push);
  - `figures/*.png`;
  - `app/src/data/term-structure.json`, `app/realdata/*.json`, `app/data/*.json`;
  - `tests/fixtures/vix_aug2024.csv` and `reports/VIX_History.agg.csv`: the two exemptions are by name only.
- **Still committed, secrets:** `.envrc` (direnv, a common home for `FRED_API_KEY`), `FRED_API_KEY.txt` (`*api_key*` is case-sensitive), `id_ed25519` (only `id_rsa*` is listed), `client_secret.json`, `token.json`, `service-account.json`, `.git-credentials`, `.aws/credentials`, `credentials.yaml`, `cert.p12`.
- **Committed files are not ignored (C3.2):** a sweep over every file in `docs schemas notebooks tests app/src app/public app/scripts tools src` printed no "WRONGLY IGNORED". `git ls-files --others --exclude-standard` lists both fixtures, the notebook and all six schemas. **SURVIVES.**
- **Gate: REFUTED in part, MED.**

## Claim 4: deploy guard (Tier 1)

Probe `p8_guarddist.sh`, on a scratch copy of `app/`. For each vector it runs `tsc -b`, `vite build` and `guardDist`, then greps `dist/` for a sentinel (`p8_out.txt`):

```
V1 marker JSON in src/ imported by TS                guard=1 | sentinel in dist | FAIL [real-marker]
V2 marker JSON in app/realdata/ (outside src)         guard=0 | sentinel in dist: index-*.js | literal marker in dist: []
V3 src/data/realdir -> symlink to an outside real dir guard=0 | sentinel in dist: index-*.js | literal marker in dist: []
V4 CSV in src/ imported with ?raw                     guard=0 | sentinel in dist
V5 real JSON without the marker (df.to_json) in src/  guard=0 | sentinel in dist
V6 gzipped marker JSON in public/data/                guard=0 | dist/data/term_structure.json.gz
V7 .js file with the marker in public/                guard=1 | FAIL [real-marker]
V8 import.meta.glob eager of ../realdata/*.json       guard=0 | sentinel in dist
V9 VITE_* env var baked into the bundle               guard=0 | sentinel in dist
V10 src/data/leak.json is a FILE symlink to outside   guard=1 | FAIL [symlink]
V11 marker object in a .ts module in src/             guard=1 | FAIL [real-marker]
```

**Realistic accidents the guard misses:**
- **V2/V8.** An exporter-made real JSON (marker included) is copied anywhere under `app/` outside `src/` and `public/` and imported. Vite strips the literal marker, as the guard's own header says. The path is also neither gitignored nor dockerignored, so the same file reaches git and the Docker image.
- **V3.** A *directory* symlink in `src/`. The src/ scan applies `rel.endsWith('.json')` before the symlink test (`guardDist.mjs:106`), so a link named `realdir` is skipped. The stated reason for check 3 ("a link can point at data outside the repository") applies to `src/` just as much. Docker's `COPY` would leave the link dangling and fail the build, so this one leaks through `npm run verify` and a static upload of `dist/`, not through the image.
- **V5.** Real data without the marker. `TermStructurePage` consumes the provisional non-envelope fixture shape, which has no `data_mode` field at all. Swapping in a real file of that shape is the obvious M1 shortcut.

Not counted (deliberate obfuscation or unlikely): V4, V6, V9. The tripwire's documented scope is "the marker", and it does catch V1, V7, V10 and V11. **Gate: REFUTED for realistic accidents, MED.**

## Claim 5: statistics (Tier 2)

| ID | Claim | A (primary source, `channel-a-c5.md`) | B (run) | Gate |
|---|---|---|---|---|
| C5.1 | Squared error is consistent for the mean | CONFIRMED: Gneiting arXiv:0912.0902v2 p.12, §5 p.27 | `p9`: lognormal argmin 20.6375 = sample mean 20.6375; gamma 20.0092 = 20.0092 | ACCEPTED |
| C5.2 | QLIKE `v/v̂ − ln(v/v̂) − 1` is consistent for the mean of v | CONFIRMED: Patton 2011 eq.(24) p.252, p.250; unbiased-proxy condition Def 1 p.248 | argmin 0.045332 = mean of v 0.045332 (plug-in `(ȳ/100)²` = 0.042591) | ACCEPTED. LOW: the table's "Ranks correctly" for "a conditional-mean forecast" is false for a plug-in ŷ under QLIKE (6% low here); the bullets below the table do disclose it |
| C5.3 | Absolute error → median | CONFIRMED: Gneiting §5 p.27 | 20.0050 = median 20.0050; gamma 18.3514 = 18.3514 | ACCEPTED |
| C5.4 | APE → "median of order −1", below the median; lognormal `exp(μ − s²)` | CONFIRMED (Thm 2.7, §5 p.27). The lognormal value and the universal "below" are UNVERIFIABLE in sources | 18.8027 vs 1/y-weighted median 18.8016 vs analytic 18.7883; gamma 13.3754 vs 13.3741. Falsifier (denominator x): 21.2903 / 22.0217. ADR figures re-run with `mape_check.py`: 20.635 / 20.000 / 18.787; MAPE 0.21177 / 0.20373 / 0.19744 | ACCEPTED (lognormal value: B plus derivation; "below the median" follows from the monotone likelihood ratio of 1/y) |
| C5.5 | DM and HLN *assume* zero d-autocorrelation beyond h−1 | CONFIRMED: DM p.135; HLN pp.281-282, "it will be assumed … all autocorrelations of order h or higher … are zero" | Review B's `c26d_sv.py` re-run unchanged: acf(d) at 21/30/50 = 0.089/0.066/0.046; LRV 24.4 vs 16.8 | ACCEPTED |
| C5.6 | State-dependent sizes 0.1258 / 0.1103; control 0.0563 / 0.0507; Bartlett L = 100/200 | n/a (measurement) | Exact match: `0.1258`, `0.1103`, `0.0563`, `0.0507`; L=100 `0.106/0.0682/0.1635/0.1088`; L=200 `0.1622/0.0782/0.2098/0.1073`. 6,000 reps (MC SE 0.0028), not the 20,000 of the first table; same seed, so this is a reproduction | ACCEPTED (B only) |
| C5.7 | statsmodels 0.15.0 `diebold_mariano_test`: default lags `max(h−1, ceil(n^{1/3}))`; Bartlett even at `lags = h−1` | CONFIRMED: `_stattools.py:2934-2935`; call chain to `weights_bartlett` (`sandwich_covariance.py:543`); HLN only with `harvey_adj=True`, default False | `p10`: lags=20 gives −1.470737 = Bartlett(20) −1.470737 ≠ rectangular(20) −1.469288; h=1 default matches Bartlett(9) and only L=9; signature `(y, forecast_a, forecast_b, *, lags=None, criterion='mse', power=2, harvey_adj=False, horizon=1)` | ACCEPTED |
| C5.7b **ADDED** | Review C's new O6 sentence: `OLS(d,1).fit(cov_type="HAC", kernel="uniform", maxlags=h−1, use_correction=False)` is the rectangular estimate | not audited | `p10b`: uniform −1.469288 = rectangular(20) −1.469288; Bartlett kernel −1.470737 | ACCEPTED (B only) |
| C5.8 | "HLN (1997) eqs. 5-8 write that estimator, and eq. 9 gives the factor" | eq.(5) is the estimator and eq.(9) the factor (p.283), CONFIRMED; **eqs.(6)-(8) derive its expectation, they do not write it** | — | REFUTED as worded, LOW |

## Claim 6: ERD (Tier 1)

- **C6.1 `target_value → forecast` is `0..1 : 0..*` (SURVIVES).** It reads that way in `erd.yaml:483`, `erd.mmd:395` (`|o--o{`) and the data-model Cardinality table.
- **C6.2 `forecast_member` has `fold_id` (SURVIVES).** It is in all three: `erd.yaml` (forecast_member), `erd.mmd:230` and `data-model.md` row `forecast_member`.
- **C6.4 no `.mmd` header claims a generator (SURVIVES).** All four now say "Hand-synced with specs/…yaml … edit both".
- **C6.3 "erd.yaml ≡ erd.mmd ≡ data-model.md" (REFUTED in part, LOW-MED).**
  - I re-ran review B's `c8b_erd.py` unchanged (`p11_erd.txt`). yaml ≡ mmd holds: 33 entities, 51 relations, no yaml-vs-mmd line. A falsifier copy, with the cardinality reverted and `fold_id` removed, produced 3 yaml-vs-mmd lines.
  - yaml ≠ data-model: review B's MED item "the ERD adds FKs that data-model.md does not state" is not fixed. `metric`, `dm_test` and `mz_regression` → `target_def`; `eda_stat` → `iv_index_meta`; `run` → `eda_stat` / `data_check` (`erd.yaml:469-470`). `surfaces.values` and `dm_p_vs_benchmark` are nullable in the yaml but required objects in the schema.
  - The parser's 6 "HIGH" feature-nullability lines are false positives on the amended prose. I checked by hand that the yaml, the mmd and data-model.md:202 agree.
  - Both documents mark `har_w` (t−4..t) and `har_m` (t−21..t) NOT NULL. They are undefined for the first 4 and 21 sessions, which is exactly the reason `rv21_back` is nullable (LOW).

## Claim 7: test suites (Tier 1)

- **pytest.** `uv run --locked pytest -q -p no:cacheprovider` in the repo: **`323 passed in 9.87s`**. The scratch copy also gave 323 passed.
- **npm run verify.** Run on the identical copy, with no install (existing `node_modules` from the lockfile): rc 0.
  - eslint passed.
  - vitest: **`Test Files 19 passed (19)`, `Tests 248 passed (248)`**.
  - Coverage (statements / branches / functions / lines): **98.51 / 95.86 / 98.41 / 99.38**.
  - The build passed, and guard:dist reported `scanned 5 file(s) in dist/, 1 in public/, 1 JSON in src/`.
- **Mutants.** 9 pre-registered mutants (`run_mutants.py`). Each file was restored and its sha256 re-checked:

| M | Mutation | Result |
|---|---|---|
| D1 | drop the `.git` ancestor check | killed (6 failed) |
| D2 | walk `directory.parents` only | killed (4) |
| D3 | `st_nlink > 2` | killed (3) |
| D4 | drop `O_EXCL` | killed (1) |
| D5 | `_check_json` recurses into `list` only, not `tuple` | killed (1) |
| **D6** | `_check_json` checks only the **first** key/value of each dict | **SURVIVED** (122 passed) |
| D7 | `REAL_MARKER` exact JSON spelling only | killed (3) |
| D8 | src/ not scanned | killed (3) |
| D9 | `public/data/real` not checked | killed (4) |

D6 is not equivalent (`d6_demo.py`). With the mutant, `{"a": 1, "b": np.int64(2)}` gives `BARE TypeError: Object of type int64 is not JSON serializable`, and `{"a": 1, "z": {10: 1}}` serialises with an int key. These are exactly the two bugs (review A C4.1 and X-F) that `_check_json` was added to fix. Every test puts the bad value at the first or only key. **LOW.**

## Found without being asked

1. **No precision rule for `reports/*.csv`** (HIGH, part of claim 1). Whether the C1.1 attack needs a quote depends entirely on it.
2. **Export guard usability** (LOW). Any hard-link snapshot backup of `$VOL_DATA_DIR` (rsnapshot-style) makes every export fail with exit 3, because each file then has `st_nlink > 1`.
3. **`tests/test_export.py::test_cli_real_mode_in_repo_exits_3_and_writes_nothing`** still asserts that `app/public/data/real` does not exist (review A X-E, unchanged).
4. **The ADR changed during this review** (review C amendments to 4.1, 4.8 and O6). D2 did not change; the O6 addition was checked as C5.7b.

## Required fixes

1. **D2 (HIGH).**
   - Restate the guarantee: the one-set-per-period argument does not cover statistics whose per-day quantities are shifts of each other over the same origin set.
   - At minimum, do not commit both EDA level moments over `O_p` and the random walk's MZ β/R² (and RMSE) at h = 1.
   - Set and justify decimals for `reports/*.csv`, with a stated bound per attack.
   - Drop the "quotes are safe" premise in the D2 table.
   - Make R4 count target changes, and give it a size threshold.
   - Reconcile R1 with R4.
2. **Docs (MED).** Fix the 8 contradictions in the C1.6 table.
3. **Deploy guard (MED).**
   - In `src/`, flag every symlink, not only `*.json` ones.
   - Scan every JSON or CSV under `app/` outside `node_modules`, not just `src/`.
   - Say plainly in `app/DATA.md` that real data without the marker is invisible to the guard.
4. **`.gitignore` (MED).** Add `*.json` under `notebooks/`, `*.CSV`, `*.txt`/`*.tsv`/`*.zip` vendor names, numpy/joblib/arrow caches, `.envrc`, `id_ed25519*`, `client_secret*.json`, `token.json`, `.git-credentials`, `.aws/`. Or invert to an allow-list for data folders.
5. **Export guard (LOW).** Document that the bind-mount check covers the repo root only. Move the TOCTOU residual into ADR D2 "Guards".
6. **Test gap (LOW).** Add a `_check_json` case with the bad value at a non-first key.
7. **Citation (LOW).** ADR 4.8: "HLN eq. 5 writes that estimator; eqs. 6-8 give its expectation; eq. 9 the factor".

## Not validated

- **Legal sufficiency of D2.**
- **Real data.** Every C1 probe ran on a simulated path. Real VIX levels, the real holdout length and the real dev n would change the 4-dp error bands, but not the identities.
- **Realism of the C1.5 strategy leak.** Judging it needs real signals.
- **macOS and Windows behaviour of the export guard.** APFS normalisation-insensitivity, Docker Desktop mounts and WSL paths were not tested.
- **Docker.** The Dockerfile path was reasoned about (`COPY` behaviour), not built.
- **Gneiting's JASA pagination** (403). arXiv v2 was used.

## Single most important fix

Rewrite D2's guarantee and R2/R3, and stop committing the pair that telescopes: EDA level moments over `O_p` together with random-walk MZ/RMSE at h = 1. With CSV precision unspecified, two dated VIX closes per period are exactly recoverable from five committed numbers, with no quote needed.

## One item to delete

The D2 "Guarantee" paragraph (ADR lines 130-141) and its copy in concept 03:148. It presents a proof that does not hold. That false assurance is what licenses quoting dated index levels in prose.
