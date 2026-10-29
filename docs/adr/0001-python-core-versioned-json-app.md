# ADR 0001: Python core, versioned JSON contract, licence-safe data, one evaluation protocol

| | |
|---|---|
| Status | **Proposed** (2026-10-27, milestone 0). Becomes Accepted when open item O1 (holdout start) is settled **and the user accepts D2**, the data-licensing rule, which is the user's decision (amended M0, review D). **Defaults in effect from M1, which the owner can override (amended M1):** `holdout_start` is the fixed 2024-07-01 (O1); D2 as proposed, with O7 and O8 "no" and O9 the proposed M1/M2 definition; CPU torch (ADR 0002). The orchestrator set these defaults; the status stays Proposed until the owner confirms them. |
| Scope | Milestones 0-6 |
| Related | [`docs/architecture/data-model.md`](../architecture/data-model.md), [`schemas/app-data/`](../../schemas/app-data/), [`docs/architecture/specs/erd.yaml`](../architecture/specs/erd.yaml) |

## Context

- The project forecasts implied volatility (VIX first) with naive, ARIMA, HAR, GARCH-family and LSTM
  models. It compares them on RMSE, MAE, MAPE and QLIKE, plus Diebold-Mariano (DM) and
  Mincer-Zarnowitz (MZ) tests, and trades the forecasts on VXX/SVXY.
- The repo is **public**. Every input is restricted:
  - Cboe index CSVs are licensed for personal, non-commercial use.
  - FRED `*CLS` series are re-hosted Cboe data, so the same restriction applies.
  - Yahoo Finance data (via yfinance) is for personal use.
- The React app has four views: a 3D term-structure ribbon, a model race, a 3D error surface, and an
  equity curve. Three of the four plot daily series, so they cannot ship real data.
- Without one shared protocol, models end up compared on different origins, windows or target
  definitions. A DM test across such rows is meaningless, and nothing visible flags it.
- The user is learning. Each rule below is written so it can be restated in an interview: the rule,
  then why.

## Decision

### D1. Separation: Python computes, JSON carries, the app renders

| Component | Owns | Must not |
|---|---|---|
| `src/ivf/` | Loading, features, fitting, forecasting, metrics, tests, regimes, backtest | Import or write anything under `app/` |
| `tools/export_app_data.py` | Read processed parquet, reshape to the schema, round, validate, write JSON | Compute a new number (beyond rounding and reshaping) |
| `app/` | Rendering. Pure UI transforms live in `app/src/lib/`: index lookups, decimation, colour scales, formatting | Compute a metric, forecast, regime or flag |
| `schemas/app-data/` | The single source of truth for the contract | Be edited without bumping `schema_version`. Every edit bumps at least PATCH, including description-only edits (amended M0). |

**Why.** A number the app computes is a second implementation that nobody tests against the first.
With this split, a reviewer re-measures every published number in one place (Python). The app is
then tested only for whether it draws the data correctly.

**Envelope.** Every JSON file the app reads is `{envelope fields…, payload}`:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string, semver `MAJOR.MINOR.PATCH` | The contract version, shared by all files. It starts at `1.0.0`. |
| `generated_at` | string, RFC 3339 UTC (`…Z`) | When the export ran |
| `source_commit` | string, 40 hex | `git rev-parse HEAD` of the code that produced the data. With no commit yet (or no git repo) it is 40 zeros, git's null object id, and `source_dirty` is `true`. (amended M0) |
| `source_dirty` | boolean | `true` if the working tree had uncommitted changes |
| `data_mode` | `"synthetic"` \| `"real"` | Where the inputs came from. The app shows it as a badge on every view. |
| `dataset` | string id | Which file this is (`manifest`, `term_structure`, …) |
| `payload` | object | File-specific content |

**Semver rules for `schema_version`.** One version covers the whole contract, because the producer
and consumer live in one repo and change together.

| Bump | When |
|---|---|
| MAJOR | Removing or renaming a field. Changing a field's type, units, meaning, array layout or index base. Adding a value to a closed enum the app branches on (`data_mode`, `license_class`, strategy `kind`, `period`, `var_method`). |
| MINOR | Adding an optional field. Adding a new dataset file (the app ignores datasets it does not know). Adding new ids in open-id fields (`model_id`, `target_id`, `strategy_id`, regime ids) is not a schema change at all. |
| PATCH | Changes to descriptions, titles or examples only. The set of valid documents does not change. |

`1.0.0` is first released by the M0 commit. Edits made before that commit, including the M0
amendments, do not bump it (amended M0).

**Tolerant reader (amended M0).** The schemas are strict (`additionalProperties: false`), so a
document from a newer MINOR fails an older copy of them. The two sides therefore validate
differently:
- **Producer.** The exporter validates strictly against the schemas of its own version, before
  writing. The same-commit contract tests (pytest on the export, and vitest on the committed fixture)
  do the same.
- **App at runtime.** The app does **not** validate against its bundled JSON Schema. It checks that
  the data's MAJOR equals its own and that the fields it uses are present and well typed.
  - It ignores fields it does not know, so data with a higher MINOR still renders.
  - If the data's MINOR is lower, the app hides features added later and shows "re-export to see
    this".

The envelope schema pins the major version (`^1\.`), so a MAJOR bump cannot happen without
deliberately editing that pattern.

### D2. Data licensing: publish a closed set of unitless results, never the series

**Status of D2: the user's decision (amended M0, review D).** D2 stays **Proposed** until the user
accepts it. It is an engineering rule, not legal advice; nobody has assessed whether it satisfies the
vendors' terms.

**Owner decision on history (amended M1, 2026-10-29).** M1 review A found one real, dated VIX
level in the pushed M0 commit `43ded87` (a test value and its mention in the M0 review-A ledger).
The owner decided: keep the strict rule, with no real vendor values or data files (including
`prices.json` and `eda.json`) in any new commit and made-up values in tests, and leave public
history as it is. So `43ded87` is not rewritten or force-pushed. The working tree holds a made-up
value, and `tools/licensing/scan_real_rows.py` must report 0 hits on the tree before every push.

**Objective (restated, amended M0, review D).** Nothing committed, including git history, lets
anyone reconstruct the daily series of any vendor index or ETP (Cboe, FRED, Yahoo), or any material
part of it.
- Isolated individual closes are not the harm: they are public facts that the news reports.
- We still minimise them (rule Q).
- Earlier drafts promised more: that no single day could be recovered. Reviews B and D each found a
  new linear combination that broke that promise. Guaranteeing it is a research problem in
  statistical disclosure control, so D2 no longer claims it.

**"Material part": proposed definition (amended M0, review E; the user's decision, O9).** A material
part of a vendor series is recoverable if someone can compute either of the following. They may use
everything ever committed plus any public data except that series itself (for example SPX closes from
FRED). The series itself is excluded because VIX closes are also public (FRED VIXCLS), and with them
anything is trivially "recoverable".
- **M1.** Any run of 5 or more consecutive daily values, to within ±1 vol point (for ETPs, ±1% of
  price).
- **M2.** Any statistic of the series in level units (vol points or prices), for a period or a
  sub-period, to within ±10%. Examples: a mean, sd, RMS gap, quantile or absolute loss.

Isolated single values are outside the definition, as in the objective.

*How P (below) is argued, not proven, to comply.*
- **M2.** Every VIX-dependent number in P is unchanged when the whole VIX path is rescaled (V → cV)
  with public SPX held fixed. This is the membership rule for P's rows, measured below. Any
  level-unit statistic scales with c. Two series that differ only by c therefore give the same P and
  the same public SPX but different level statistics, so P plus SPX cannot determine one. The
  argument holds only while every row really is scale-free: it is measured on simulated paths for the
  probe's models, and argued by equivariance for the rest of the registry.
- **M1.** P alone fixes no level, for the same reason. A level could come from a quoted close
  (rule Q) or from outside knowledge. A run would then still need P to pin 5 consecutive daily
  log-ratios to within about ±5%. Review E's linearised probe found at most one nearly identifiable
  direction per simulated path, before rounding. Rounding blurs such read-outs by 6-48% in log units.
  So no run of 5 has been shown recoverable, and none is ruled out (residual 1).

| Artefact | Real data | Synthetic data |
|---|---|---|
| Raw CSVs, interim and processed parquet | Never committed. They live in `$VOL_DATA_DIR`. | Never committed. They are regenerable from the seed. |
| Per-date forecasts, errors, actuals, equity curves | **Never committed** | Committed inside the fixture JSON |
| Real-data-derived numbers in `reports/*.agg.csv`, the README, docs or screenshots | **Only the publication set P below, at its stated rounding.** Nothing else. (amended M0, review D) | Not restricted (the synthetic fixture and its numbers) |
| App JSON | Written **outside the repo** to `$ROOT/app-data/real/`. The Vite dev server serves it in dev only. | Committed to `app/public/data/synthetic/` |
| Screenshots and figures | Only figures drawn entirely from P values (e.g. holdout RMSE ratio by model). No real error surface, histogram, time plot or regime cell. (amended M0, review D) | Allowed for any view. The SYNTHETIC badge must be visible. |
| Dated index levels quoted in prose | Only under rule Q below (amended M0, review D) | — |

**Why per-date errors count as series.** Take the random-walk forecast at h=1. Its error is
`e_t = V_{t+1} − V_t`, so the cumulative sum of the errors rebuilds the VIX path up to a constant.
The always-long-VXX equity curve is the VXX price path, rescaled. Each JSON file therefore carries a
`license_class` in the manifest:
- `series` covers the ribbon, the model race and the equity curve;
- `aggregate` covers the error surface. Its real version is not a daily series, but it holds regime
  cells and level-unit metrics, so it stays local like every real JSON (D2 P; amended M0, review D).

**Publication set P (amended M0, review D; rows restricted and closure fixed in review E).** This is
the closed list of every real-data-derived number that may ever be committed. All of them are
**unitless**: ratios, test statistics, probabilities, booleans, counts and time spans. None carries
vol points or prices. Each is computed per whole period (`dev`, `holdout`), per target and per horizon
(or per strategy, or per index), over the period's evaluable origins. No regime or sub-period cells are
allowed.

*Which rows (amended M0, review E).* A model × target row is in P only if every number in it is
unchanged when the whole VIX path is rescaled (V → cV) with the **public** SPX series held fixed.
- Daily SPX closes are public (FRED SP500). A row whose forecast ties the VIX scale to SPX therefore
  reveals level-unit VIX statistics.
- Review E measured this on 10 simulated seeds (`tools/licensing/p_tie_seeds.py`). From the
  `vix_implied` rv21 ratios plus SPX, it recovered the holdout mean VIX to a median error of 4.29%
  (max 13.69%) and the RMS VIX-RV gap to 0.053%.
- Scale-tied rows stay **local**. That is the orchestrator's default; the user may opt in (O8).
- The benchmark for every ratio and every DM test is the random walk `rw`. For `vix_level` that means
  ŷ = V_t. For `rv21` it means ŷ = `rv21_back(t)`, a function of SPX only.

| Target | Rows in P | Local only (scale-tied) |
|---|---|---|
| `vix_level`, h = 1, 5, 21 | `hist_mean`, `arima_log`, `har_log`, `arima_garch_log`, `garch_cal`, `gjr_cal`, `egarch_cal`. `rw` contributes only its MZ booleans and `n`. | Uncalibrated `garch`, `gjr`, `egarch` (SPX-only forecasts of a VIX target) |
| `rv21`, h = 21 | `har_rv`, `garch`, `gjr`, `egarch`: SPX-only forecasts of an SPX-only target, so they carry no VIX information. `rw` contributes only its MZ booleans and `n`. | `vix_implied` |

Conditional members:
- **`lstm_log`** is in P only if M4's inputs keep it scale-free, re-probed with
  `tools/licensing/p_tie.py` (c4). With `iv_rv_spread`, a fixed-threshold regime flag, or an absolute
  clip it is scale-tied and stays local. Measured: 65 of 81 entries changed with `iv_rv_spread`, and
  0 of 81 with no extra inputs or with `rv21_back`.
- **`*_cal`** is in P only if a non-positive `a + b·G` is handled in a scale-free way, for example a
  floor proportional to the training mean of v. An absolute floor makes it scale-tied.
- **Strategies** are in P only if driven by in-P models, with any band relative to the level (e.g. a
  percentage of V_t), never in vol points.

| Quantity | Per | Rounding | Note |
|---|---|---|---|
| RMSE and QLIKE of each in-P model **as a ratio to `rw`'s** | Model, target, horizon, period | 3 significant figures | Primary comparisons (4.3). A ratio removes the level scale. |
| MAE and MAPE as ratios to `rw`'s | Same | 3 s.f. | Flagged in every table as not consistent for the mean (4.3) |
| DM statistic, HLN-corrected, and its p-value, each in-P model against `rw` | Model, target, horizon, period, loss (`sq_error`, `qlike`) | 2 dp each | |
| MZ tests: "β = 1 not rejected" and "α = 0 and β = 1 not rejected", at 5% | Model (in-P rows and `rw`), target, horizon, period | Booleans | Pass/fail only. No coefficients and no p-values. |
| Sharpe (annualised, from arithmetic simple returns), PSR, DSR | In-P strategy, period | 2 dp | All three are scale-free |
| Number of trials | Strategy, period | Exact | |
| Turnover (annualised) | Strategy, period | 2 s.f. | |
| AR(1) half-life of ln(index), in sessions | Index, period | 2 s.f. | M1's persistence finding; shift-invariant in ln(index). It is a one-to-one function of the AR(1) φ (φ = 0.5^(1/half-life)), so **P publishes φ in effect**, at the half-life's precision. That is intended: φ is unitless and unchanged by rescaling (amended M0, review E). 2 s.f. over roughly 5-500 sessions is about 180 values, under 8 bits. |
| `n` (the count behind each number above) | Each row | Exact | Fixed by the protocol calendar |

**Not in P,** for example:
- anything in vol points or prices (levels, RMSE, MAE, bias);
- every moment (means, standard deviations, skew, kurtosis of levels or returns);
- every coefficient (MZ α and β, regression slopes and intercepts), except the AR(1) φ that the
  half-life encodes;
- every return (mean, annualised, CAGR);
- drawdowns, extremes and quantiles;
- regime cells and regime frequencies;
- ACFs and ADF/KPSS statistics or p-values;
- the scale-tied rows above;
- provenance hashes of the real raw files and the `run_id` built from them (D3). They are
  real-data-derived, so they stay local (amended M0, review E).

**EDA prose (amended M0, review E: replaces "qualitative or unitless", which let skew, ACFs and
regime frequencies in).** About real data, a document may make **only** the statements below. Each
is qualitative (a direction, or yes/no), and none carries a number except E2's half-life from P.

| # | Allowed statement |
|---|---|
| E1 | The index shows volatility clustering: yes or no |
| E2 | Its AR(1) half-life is about N sessions, with N from P |
| E3 | It mean-reverts: yes or no |
| E4 | The VIX term structure is usually upward-sloping (contango) and inverts in stress episodes: yes or no, with no frequencies |
| E5 | Implied variance usually exceeds subsequent realised variance (a positive VRP): yes or no |
| E6 | Vol-of-vol rises with the level: yes, or "not shown" (amended M1, review B: the test that keeps its size when shocks cluster has little power then, so failing to reject is not a "no"). "Yes" only when the fitted AR(1) phi of ln VIX is at most 0.985 (amended M1, final check): under skewed GARCH(0.03, 0.965) shocks with no level effect (200 paths, n = 9,000) the test rejected on 0.0% at phi 0.98, 2.0% at 0.985, 15.0% at 0.99 and 44.5% at 0.995, so above 0.985 it says "not shown" with that reason. And only on at least 6,000 level/next-shock pairs (`eda.E6_MIN_N`; default decision 2026-10-29): at n = 2,000 the test said "yes" on 24 of 100 null paths inside the cutoff; at 4,000, 10 of 200 (the rule's bound, so not taken); at 6,000, at most 3 of 200 |
| E7 | ln VIX is judged stationary at 5% by ADF and KPSS: the verdict only, with no statistic or p-value |

Anything else needs an ADR amendment. Results of local-only (scale-tied) rows may be described only
by direction (e.g. "uncalibrated GARCH is biased low against VIX"), with no numbers.

**R1. One release per frozen `data_end`; no corrections (amended M0, review D; replaces the old
R1-R4).**
- P is committed once, in one commit, for the frozen protocol.
- A value found wrong after release is **withdrawn**: a later commit deletes it and adds a prose
  erratum that gives no replacement number. Git history keeps the old value, which is why it is
  never corrected in place: the old and new values together would be a second publication.
- New values for the same quantities need a new ADR and a new release for a new `data_end`. That ADR
  must rerun the committed probes in `tools/licensing/` on the union of every release.
- Nothing enforces R1 automatically yet.

**Q. Quoting dated index levels in prose: minimisation, not protection (amended M0, review D;
rationale corrected in review E).**
- A dated close may be quoted only for a **named market event**, cited to a public source, e.g.
  "VIX closed at X on 2024-08-05 (Cboe)".
- Never a VXX or SVXY price.
- Never a session within 21 sessions (h_max) of `train_start`, the first dev origin,
  `holdout_start`, or `data_end`. That window covers the last origin at every horizon.
- **What Q does and does not do.** Closes are public facts (e.g. FRED VIXCLS), and review D's anchor
  solve accepts any number, wherever it came from. So Q protects nothing against someone who looks a
  close up elsewhere.
  - It is hygiene. Our own documents do not supply anchors near the boundaries, where the telescoping
    identities found so far live, and quoted values stay few.
  - Under the proposed definition (O9), a committed quote is one of the things M1 must reckon with.

**What P reveals, measured (amended M0, review D; probes committed in review E).**
`tools/licensing/p_set.py` computes exactly P at the rounding above. It uses simulated data only.
- The path is a simulated log-AR(1) VIX path (seed 20261028, 2,500 training sessions, 562 holdout
  origins).
- The published rows are `rw`, `hist_mean`, `ar1_log` and `har_log`, plus one sign strategy on
  VXX-like returns.
- `ewma_spx`, a stand-in for a scale-tied row such as uncalibrated GARCH on `vix_level`, is added
  only to show why such rows are excluded.

Findings:
- **The attacks' inputs are absent.** P has 114 published numbers for the period. None is an input
  to review B's A1-A4 (bias, refreshes, regime cells) or to review D's `p1_shift` and `p2_r3`: EDA
  level mean or sd, MZ β or R², absolute RMSE or QLIKE, DM `mean_d`, or strategy vol and returns.
  - Without absolute RMSEs, `RMSE_a² − RMSE_b²` is known only in units of `RMSE_rw²`.
  - The telescoping `−2·ln(V_e/V_s)/n` sits inside `QLIKE_rw`, which P publishes only as the
    denominator of ratios.
  - The mean return needs `ann_vol`, which is not in P.
- **The published rows are scale-free.** Rescaling the whole VIX path and the strategy returns by
  c = 0.8, 1.25 and 2.0 left **0 of 114** P entries changed. Two paths that differ by any scale give
  identical P, so no function of P alone can return a level.
  - **Falsifier** (`tools/licensing/p_falsify.py`): the same rescale does change level-unit
    statistics (EDA sd 7.166 → 8.957, random-walk RMSE 1.485 → 1.857, ann_vol 0.6359 → 0.7949), so
    the test would catch them. MZ β (0.983) did not change: it is unitless, but it is still excluded
    from P, because review D's attack used it together with level moments.
  - `*_cal` (`tools/licensing/p_tie.py`, c3): 0 entries changed over three VIX scales and two SPX
    scales. With a fixed absolute intercept as falsifier, 6-8 changed.
- **Scale-tied rows are not scale-free, and SPX is public.** Adding `ewma_spx` made **22 of 147**
  entries change under a VIX rescale, all of them that model's. With public SPX, such rows reveal
  level-unit VIX statistics (`p_tie.py`, `p_tie_seeds.py`).
  - Absolute RMSE of `vix_implied` against RV: recovered 4.4028 against a true 4.4007.
  - Holdout mean VIX: median error 4.29% over 10 seeds.
  - That is why these rows are not in P. (amended M0, review E: the earlier text said exploiting
    them "needs the SPX return series, which is vendor data and is not committed". SPX is public,
    so that was no barrier.)
- **P is sensitive to single days.** Moving one close by ±0.25 vol points flipped at least one
  rounded P entry on **599 of 604** sessions in play, and ±1.0 on all 604
  (`tools/licensing/p_sens.py`).
  - By class: DM statistic 98.8% of sessions, DM p-value 85.9%, MAPE ratio 75.3%, QLIKE ratio 74.0%,
    MAE ratio 72.5%, RMSE ratio 30.0%, strategy entries 2.0%. Review E found the half-life flipped on
    0 of 604.
  - Sensitivity is not recovery, but it is why D2 does not promise that single days are protected.

(amended M0, review E: the "information bound" bullet is deleted. It used the marginal entropy of a
close where the series' entropy rate applies, it counted public SPX bits, and it said nothing about a
material part.)

**Residual (stated, not closed; amended M0, review E).**
1. **Joint inversion is not excluded.** Scale invariance removes only one direction, the overall
   level. Ratios between days, such as `V_e/V_s`, get no protection from it: the telescoping
   `ln(V_e/V_s)/n` sits inside every `rel_qlike` denominator.
   - Review E's linearised probe (E1.c6, unrounded P) found one boundary day nearly identifiable to
     first order on one of three simulated paths. On the other two, the most exposed days were
     interior. Rounding blurs every such read-out by 6-48% in log units.
   - No recovery has been demonstrated, and none is ruled out.
   - Once a close is known (committed under Q, or public), the level is known, so scale invariance is
     no mitigation for whatever P does pin.
2. **Scale-tied rows are excluded, not harmless.** If the user opts in (O8), they reveal level-unit
   VIX statistics given public SPX, as measured above. The membership conditions for `lstm_log`,
   `*_cal` and strategies must be re-checked when those are built (M3-M6).
3. **Strategies that differ on a few days.** Their Sharpe, PSR and DSR differ only through those
   days' returns, measured in units of each strategy's own sd and rounded to 2 dp. This is not a
   recovery of returns, but it is noted (review D, C1.5).
4. **Quoted closes** disclose themselves (rule Q).
5. **Legal sufficiency** has not been assessed.

**Rejected alternatives.**
- **Per-cell minimums plus refresh and suppression rules** (M0, then review B). Each round of review
  found a new differencing identity: A1-A4, then the shifted level moments in review D.
- **Committing level-unit statistics with a written proof of safety.** The review D proof failed.
- **Atom-based regime cells** (O7). They are only safe if every statistic is also safe, and P
  avoids having to prove that by publishing no level units at all.
- **Publishing scale-tied rows** (O8): measured to reveal level-unit statistics given public SPX
  (amended M0, review E).

**Synthetic fixture.** `ivf.data.synthetic` writes Cboe-shaped CSVs and Yahoo-shaped price tables
into `$ROOT/synthetic/raw/`. The **same** loaders, pipeline and exporter then run on them. This
means CI exercises the real code path, not a mock.
- The generator's parameters are hand-set round numbers, commented in code. They are **not fitted**
  to real data.
- The generator has a two-factor log-vol term structure (fast and slow factor), so contango and
  backwardation both occur. Index returns are driven by the same factors, with a positive variance
  premium so the VRP gap exists, and VXX/SVXY are approximated by roll yield.

**Guards.**
- The exporter refuses to write real JSON inside the repo tree, so `git add -A` and `vite build` do
  not pick it up. As a second layer, `.gitignore` also lists `app/public/data/real/`.
  - The exporter compares the output folder and each existing ancestor with the (device, inode)
    of the repo root and of every folder below it, and refuses any ancestor holding a `.git` entry,
    so a symlink, a case variant, or a bind mount of the repo or of any folder in it is caught
    (amended M0, review D: the first fix compared with the root only, and a bind mount of
    `app/public` got through).
  - **Residual, accepted (amended M0, review D):** the check and the write are separate steps, a
    time-of-check-to-time-of-use (TOCTOU) gap. Someone who swaps a parent folder for a symlink in
    between can still redirect the write (`tools/export_app_data.py`). That needs write access to a
    parent of the output folder, which already means owning this account.
  - **Residual, accepted, same TOCTOU and mount class (amended M0, review E):**
    - Identity-set race: a repo folder created and bind-mounted in the roughly 0.1 s between the
      folder walk and the check gets written into.
    - Overlay mount: one whose writable upper layer is inside the repo gets written into.
    - Both need a deliberate concurrent setup and mount privilege.
- `npm run verify` (and the Dockerfile build stage) runs `app/scripts/guardDist.mjs`, which fails on
  any symlink under `app/` outside `node_modules/` (amended M0, review E: an `npm link` symlink there
  is not checked; the Docker build, whose `npm ci` drops such links, is the backstop), any file under `app/` (outside `node_modules/`, `coverage/`, `.git/`,
  and docs/tests that only describe it) carrying the `"data_mode":"real"` marker, any `.json` under
  `src/`, `public/` or `dist/` whose top-level `data_mode` is not `"synthetic"` (positive
  allowlist), and any non-JSON file in a `data/` folder (amended M0, review D: it originally scanned
  `dist/` for the marker only, and review D got real data past it in 7 of 11 vectors; 10 of 11 are
  now stopped).
  - **Residual:** it is a tripwire for accidents, not a defence against deliberate obfuscation. It
    cannot see values baked in through `VITE_*` env vars, real data pasted into a `.ts` literal, a
    JSON that labels itself synthetic, or markerless JSON outside `src/`, `public/` and `dist/`
    (`app/DATA.md` lists all of them).
  - **Residual, accepted (amended M0, review E):**
    - A case-variant data folder (`public/Data/`) escapes the case-sensitive data-folder rule.
    - An `npm link` symlink under `node_modules/` is not checked (see above).
    - Real values relabelled `"synthetic"` in `src/` are invisible to the guard. In
      `public/data/synthetic/`, the exporter's `--check` catches them.
    - A small markerless asset imported with `?raw`, `?url` or `new URL(…)` is inlined into the JS
      bundle as base64.
- **The data pipeline refuses a data root inside the repo (amended M1, review A).** Review A found
  that only the exporter had the guard: `VOL_DATA_DIR=<repo>/localdata` let `fetch`, `build-interim`
  and `eda` write raw snapshots, raw hashes and every EDA number into untracked, unignored files of
  the checkout. The guard now lives in `ivf/repo_guard.py`, one implementation shared by the
  exporter and by every pipeline stage that writes under `ROOT` (the `python -m ivf.data` commands
  and their library entry points), with the same (device, inode) and `.git` rules, the same exit
  code 3 and the same residuals as above.
  - **Every folder written into is checked, at write time (amended M1, review B).** Review B kept
    `ROOT` outside the repo but made `ROOT/raw` (and `ROOT/eda`) a symlink into the checkout:
    `fetch` wrote 8 snapshots and `eda` its report into it, exit 0, because only `ROOT` was checked.
    Now the root check also covers `raw/`, `interim/`, `processed/`, `eda/` and `app-data/`, so
    those stop the command before its first write; and every file, from the pipeline and from the
    exporter, is written by one function, `ivf.repo_guard.write_file`, which checks the folder it
    writes into when it writes, opens it, checks that the open folder is the checked one (same
    device and inode, same resolved path) and writes through that handle. This narrows the TOCTOU
    residual above to a folder *moved into* the repo after it was opened.
- **A local scan for copied real data (amended M1, review A).** `tools/licensing/scan_real_rows.py`
  compares every text file git would commit with the raw snapshots under `ROOT` (whole lines, whole
  raw rows, dated values, file sizes and their totals, raw hashes) and prints only `file:line`.
  Review A found two real 1990 VIX rows in a test fixture, byte for byte, and the total size of the
  raw files in the M1 walkthrough; the scan then found more real closes in the M1 fixtures and one
  real VIX level in an M0 test (`tests/test_schemas.py`, committed in the M0 commit). All are now
  made-up values. **Residual:** it runs only where the raw data are (not in CI), it is a pre-commit
  gate by convention, and a value rounded to one decimal or paraphrased in words escapes it.
  - **Other spellings (amended M1, review B).** Review B planted stand-in values in 16 shapes and 9
    escaped entirely: `date(...)` objects, epoch-millisecond keys, lists of closes without dates,
    pretty JSON with the value more than 3 lines from its date, compact, slash-ISO and textual
    dates, 1-decimal values and values divided by 100. The scan now reads every date spelling it
    names (ISO with `-`, `/` or `.`; US and day-first; compact; `(Y, M, D)` calls; textual; epoch
    seconds to nanoseconds), matches a value at its own decimals, rounded either way, or divided by
    100, looks up to 12 lines away for the nearest dated line, and flags runs of 3 or more values
    equal to consecutive sessions of one raw series with no date at all (5 or more at one decimal).
    Every planted row of all 16 shapes is caught, and of 24 of 25 shapes added in the fix; the 25th,
    a 1-decimal value whose date is on another line, stays out by design. **Residual, stated:** values transformed beyond rounding
    and /100 (returns, logs, differences, sums, truncation); a lone value with no date nearby and no
    run (fewer than 3, or 5 at one decimal); a one-decimal value whose date is on another line;
    two-digit years and date spellings not listed; scientific notation and numbers in words; and
    ignored, binary or non-UTF-8 files (images, PDFs, parquet).
  - **M1 final check.** Three more planted shapes escaped: a decimal-comma CSV
    (`DD.MM.YYYY;16,94`), Excel serial dates with a value, and every k-th session's close. The
    decimal comma is now read as a number (not with exactly 3 digits after it); the day-first
    dotted date already was. **Residual, stated:** Excel serial dates and every-k-th-session
    lists stay uncaught (`tools/licensing/README.md`, "still not caught").
- The README carries a Cboe trademark and terms notice, plus a "not financial advice" line.

### D3. Storage

`ROOT = $VOL_DATA_DIR/implied-vol-forecasting`. `VOL_DATA_DIR` defaults to `~/data/vol-projects`.
Synthetic runs use the same layout under `ROOT/synthetic/`.

| Layer | Location | Format | Committed | Written by |
|---|---|---|---|---|
| raw | `ROOT/raw/{source}/{symbol}/{sha256}.csv` and `ROOT/raw/provenance.jsonl` | Bytes as downloaded, plus an append-only JSONL log | No | `ivf.data.fetch` |
| interim | `ROOT/interim/{table}.parquet` | Parquet: typed, deduplicated, one table per source kind, not joined | No | `ivf.data.build_interim` |
| processed | `ROOT/processed/run_id=<id>/{table}.parquet` and `ROOT/processed/runs.parquet` | Parquet, hive-partitioned by `run_id` | No | The pipeline stages |
| reports | `reports/*.agg.csv` | CSV | **Yes**: publication set P only (D2), real data, clean tree. The provenance record (run row, raw sha256 list, protocol) is **not committed**. It stays local in `ROOT/reports-provenance/<run_id>.json`, because the raw hashes, and the `run_id` built from them, are real-data-derived and outside P. Committed reports carry only `protocol_hash` and `source_commit`. (amended M0, review E) | `ivf.reports` |
| app data | `app/public/data/synthetic/*.json` (synthetic) or `ROOT/app-data/real/*.json` (real) | JSON | Synthetic only | `tools/export_app_data.py` |

**Fetch idempotency.**
1. Each download goes to a temp file and is hashed with SHA-256. It is renamed to `{sha256}.csv`
   only if that file is absent. Identical content is therefore stored once, and old snapshots are
   kept, so a result can always be rebuilt from the exact bytes it used.
2. Every fetch appends one line to `provenance.jsonl`: `fetch_id, source, symbol, url, final_url,
   fetched_at, http_status, bytes, sha256, raw_path, changed, tool_version`. The line is written
   even when the content is unchanged.
   - JSONL is append-only, so a crash cannot corrupt earlier records. Rewriting a parquet file
     could.
   - `final_url` records the target of Cboe's 307 redirect.
3. Re-running `fetch` within `--max-age` (default 1 day) of the last fetch of a symbol makes no
   network call. `--force` overrides this.
4. `build_interim` uses the latest snapshot per `(source, symbol)`, unless `--snapshot` pins one.
   Every interim row keeps its `raw_sha256`.
5. `ivf data verify` re-hashes every raw file and fails on any mismatch.

**As built in M1 (amended M1).** The command is `uv run python -m ivf.data verify`; the code is
`ivf.data.raw` and `ivf.data.fetch`. Five details the text above left open:
- **The bytes stored for yfinance.** yfinance returns a DataFrame, so "bytes as downloaded" is
  defined by `fetch.yfinance_csv_bytes`: header `date,open,high,low,close,adj_close,volume`, the
  New York session date, prices with six decimals, volume as an integer, an empty field for a
  missing value, rows sorted by date, `\n` line ends. The index must be tz-aware
  `America/New_York` at midnight, with exactly those six columns; anything else is refused, not
  stored. The request is `history(period="max", interval="1d", auto_adjust=False, actions=False)`,
  recorded as the pseudo-URL `yfinance://^GSPC?...`.
- **"Rename only if absent"** is `os.link` from the temporary file to `{sha256}.csv`, which
  refuses an existing name in one atomic step; the temporary file is then removed.
- **A skipped fetch** (younger than `--max-age`) makes no network call and writes no log line:
  nothing was fetched.
- **A failed fetch** (HTTP status other than 200, an empty body, an empty DataFrame) stores
  nothing and logs nothing, and `fetch` exits 5 after trying the other symbols.
- **`verify`** also fails on a snapshot no log line accounts for, on a file in `raw/` that is not
  a snapshot, and on a log line whose file is missing or whose `bytes` disagree. A leftover
  `.tmp-*` file from an interrupted fetch is a warning only.

**Provenance chain.** Each link records the key of the one before it:
- raw `sha256` → interim `raw_sha256`;
- interim → `run_input(run_id, raw_sha256)`;
- `run_input` → `run(run_id, protocol_hash, source_commit, source_dirty)`;
- `run` → the local `ROOT/reports-provenance/<run_id>.json` and the JSON manifest's `run_id`. Real manifests are local too. (amended M0, review E: provenance is never committed)

**Identities.**
- `run_id` is the first 12 hex characters of the SHA-256 of
  `{protocol_hash, source_commit, source_dirty, data_mode, sorted raw sha256s}`. The same inputs
  give the same `run_id`, so a re-run overwrites its own partition. `run_id` is provenance, like
  `source_commit`, and is therefore excluded from `--check` comparisons (amended M0: run_id added
  to check-ignored provenance fields).
- `protocol_hash` is the SHA-256 of the canonical (sorted-key) JSON of the protocol config.

### D4. Evaluation protocol (shared by every model)

#### 4.1 Calendar and information set

- **Master calendar 𝒯 (amended M0).** The NYSE (`XNYS`) session calendar from `exchange_calendars`,
  from `train_start` to `data_end`, extended by the scheduled sessions after `data_end` that are
  needed to date the longest-horizon targets. `t+h` means the h-th session after `t` in 𝒯.
  - 𝒯 is computed from dates alone, never from which data happen to be present. Fold cutoffs and
    target dates therefore do not change when the data are truncated or refreshed. This is what lets
    the truncation test (4.7) compare like with like.
  - M1's `data_check` must find a VIX close and a `^GSPC` close on every session of 𝒯. If any is
    missing, M1 decides the gap policy explicitly; a gap is never silently dropped or filled.
  - **Gap policy (amended M1).** M1 measured the real files on 2026-10-29 (counts and dates
    are `data_check` output, local only under D2). Cboe's VIX file has no close on a few XNYS
    sessions, all in burn-in, years before the first fold cutoff; the `^GSPC` file is complete
    on 𝒯; FRED VIXCLS lacks most of the same sessions, so the FRED fallback cannot close them and
    is not used. The rules
    (`ivf/data/check.py`):
    - **G1.** A session of 𝒯 in `[train_start, data_end]` without a VIX close or a `^GSPC` close
      is a *core gap*. It stays a gap: never filled, interpolated, or removed from 𝒯. It is null
      in every table built on 𝒯.
    - **G2.** Anything computed over a window containing a gap is null: a return needs two
      consecutive sessions, a 21-session sum needs all 21. Pairs (an AR(1) fit, an ACF lag) are
      formed on 𝒯; a pair with a gap is dropped, never bridged to "the previous row that exists".
      Tests that need an unbroken series (ADF and KPSS lag and difference consecutive values) run
      on the longest gap-free segment of their sample, and report its length and dates (amended
      M1, review A: the first M1 version dropped the missing values and so joined the neighbours of
      every gap).
    - **G3.** Core gaps are allowed only in burn-in, at least 22 sessions (the `har_m` window,
      the longest trailing window) before the first fold cutoff, so no feature at a forecast
      origin and no scored target can touch one. A later core gap fails `check` (exit 1); it would
      need a missing-origin rule under 4.5's "same rows", which is a decision, not a default. The
      measured gaps satisfy G3. M2's fitters must accept nulls in burn-in training rows (or say
      how they drop them).
    - **G4.** A row on a date that is not an XNYS session (Cboe has published VIX on NYSE holidays
      since 2022, and on one earlier unscheduled NYSE closure) is not on 𝒯, so nothing downstream
      reads it. It stays in interim and is counted as `not_a_session`.
    - **G5.** The other indices (VIX9D, VIX3M, VIX6M, VIX1Y, VVIX) are not core: their gaps are
      null and counted (`missing_in_cboe`) and never fail `check`. VIX1Y misses a few US
      bond-market holidays; VVIX is sparse in its first months.
  - **`data_end` (amended M1)** is the last session with both core closes **settled**: the fetch
    that produced them ran on a later New York calendar day than the session (yfinance can return a
    bar for a session still in progress; not observed in M1). `check --freeze-data-end` writes it into
    `configs/protocol.yaml` once and refuses to move it. Frozen on 2026-10-29 at **2026-10-28**.
    **Re-validated on every run (amended M1, review A):** `check` and `eda` fail (exit 1) unless the
    frozen value is an XNYS session with both core closes present in the interim tables being read,
    each from a fetch made on a later New York day. Review A found the rule applied only when
    freezing: a frozen `data_end` fetched the same day, or a hand edit, passed both commands.
    **Checked against its freeze record (amended M1, review B):** a hand edit moving `data_end` back
    to an earlier *settled* session still passed, since such a session meets every rule above. The
    freeze now also writes `configs/protocol.freeze.json` (the value and the `protocol_hash` it
    froze), and `check`, `eda` and notebook 01 fail when the protocol no longer matches it (the
    notebook stops). It is a tripwire for an accidental edit, not a lock: editing both files on
    purpose is a new protocol, which rule R1 allows. The record of the 2026-10-28 freeze was written
    on 2026-10-29 by `check --freeze-data-end`, which records an earlier freeze only while the data
    still support the frozen value.
  - `exchange_calendars` becomes a pinned dependency when 𝒯 is implemented. The calendar figures in
    4.4 and 4.5 were measured with it.
- **Information set at origin t.** Every value dated ≤ t, as of that session's close. VIX's daily
  close is struck after the SPX cash close (16:15 versus 16:00 ET; not verified in M1, which found no
  primary source in the time it had: still open, amended M1); both count as
  "close of t". Nothing dated t+1 or later may be used.
- **Vintage limitation.** History is the vintage as fetched, not point-in-time. Cboe restatements
  are rare. `^GSPC` is a price index, so it is not dividend-adjusted. VXX/SVXY use split-adjusted
  closes, because both had splits. Review C found VXX reverse splits on 2021-04-23, 2023-03-07 and
  2024-07-24, and SVXY splits on 2018-09-18 and 2024-04-11 (yfinance `splits`; issuer confirmation
  for SVXY pending); M6 verifies every date against the issuer before trusting adjusted closes
  (amended M0, review C).

#### 4.2 Targets (forecast for origin t at horizon h targets y(t+h))

| `target_id` | Value at date d | Units | Horizons | Models |
|---|---|---|---|---|
| `vix_level` | `V_d` = VIX close | Vol points (annualised %, e.g. 18.25) | 1, 5, 21 | All IV models |
| `rv21` | `100·sqrt((252/21)·Σ_{j=0}^{20} r²_{d−j})`, where `r_d = ln(SPX_d / SPX_{d−1})` (decimal, no mean removed) | Vol points | 21 only, so the window is exactly the 21 sessions after the origin | GARCH track, and VIX-as-forecaster |

Every target also has a variance-space value `v = (y/100)²`: annualised variance in decimal units.

#### 4.3 Point forecast, spaces and losses

- **Every forecast row has a vol-point forecast `ŷ` and a variance forecast `v̂`, with
  `ŷ = 100·√v̂` always (amended M0).** Models may fit in any space (log VIX, returns, variance).
  `var_method` records which of the two numbers is the model's own forecast.
  - **`plugin`** (RW, historical mean, ARIMA, HAR, LSTM): `ŷ = E_t[y]`, the conditional mean, and
    `v̂ = (ŷ/100)²`.
    - A log-space model with forecast mean μ and variance s² reports `ŷ = exp(μ + s²/2)`, the
      lognormal mean. `exp(μ)` would be the median, which is biased low.
    - As a forecast of `E_t[v]`, the plug-in is low by exactly `Var_t(y)/10⁴`, the *conditional*
      variance, because `E_t[y²] = E_t[y]² + Var_t(y)`. That identity holds only when `ŷ` is the
      true conditional mean.
  - **`native`** (the GARCH family's `G`, `vix_implied`, `*_cal`): `v̂ = E_t[v]`, and `ŷ = 100·√v̂`
    is defined as the **root-mean-square** vol, `√E_t[y²]`.
    - By Jensen, `√E_t[y²] ≥ E_t[y]`, so as a mean `ŷ` is biased high. By a second-order Taylor
      expansion the bias is about `100·Var_t(v) / (8·v̂^{3/2})`.
- **What this means for scoring (amended M0, review B).** A loss ranks forecasts correctly only
  for the statistic it is *consistent* for (Gneiting 2011): the statistic whose true value minimises
  the expected loss.

  | Loss | Consistent for | Effect on a conditional-mean forecast |
  |---|---|---|
  | Squared error (RMSE) | The **mean** of y | Ranks correctly |
  | QLIKE (variance space) | The **mean** of v | Ranks a `native` forecast of `E_t[v]` correctly. A `plugin` forecast `(ŷ/100)²` is biased low by `Var_t(y)/10⁴` even when `ŷ` is the true mean, so QLIKE does not rank plug-in forecasts as means (amended M0, review D) |
  | Absolute error (MAE) | The **median** of y | For right-skewed y the median is below the mean, so MAE can prefer a forecast that is biased low |
  | Absolute percentage error (MAPE) | The "median of order −1": the median of y's distribution reweighted by 1/y (Gneiting 2011) | Below the median, so MAPE prefers low forecasts even more strongly |

  - **Measured** (`uv run python tools/probes/mape_check.py`: 2,000,000 draws of a lognormal y
    with median 20 and log-sd 0.25), with the matching review B figures in parentheses:
    - Squared error is minimised at 20.635: the mean, `exp(μ + s²/2)`.
    - Absolute error is minimised at 20.000: the median (review B: 19.999).
    - Percentage error is minimised at 18.787, against the analytic `exp(μ − s²)` = 18.788
      (review B: 18.785).
    - MAPE of the mean forecast is 0.2118, against 0.2037 for the median (review B: 0.2118 and 0.2038)
      and 0.1974 for 18.788.
  - **What the project does.**
    - The **primary** comparisons are RMSE (`plugin` forecasts, the mean of y) and QLIKE (`native`
      forecasts, the mean of v). The DM tests use exactly these two losses (4.8).
    - MAE and MAPE are reported because the brief asks for them. Every table that shows them
      flags them as **not consistent for the mean**: a better MAE or MAPE can come from a forecast
      that is biased low.
    - Comparing `plugin` and `native` forecasts under the other kind's loss also includes the Jensen
      term described above. Reports say so, and MZ shows it as β ≠ 1.
    - We report these effects rather than correct them: see "Alternatives considered".

| Loss | Space | Formula | Note |
|---|---|---|---|
| bias | Level | mean(`y − ŷ`) | Positive means the model under-forecasts. Local diagnostic only: never committed for real data (not in D2 P; amended M0, review D). |
| RMSE | Level (vol points) | sqrt(mean((`y − ŷ`)²)) | **Primary** for `plugin` forecasts: consistent for the mean |
| MAE | Level (vol points) | mean(\|`y − ŷ`\|) | Consistent for the median, **not** the mean. Reported because the brief asks for it, and flagged as such. (amended M0, review B) |
| MAPE | Level, as a fraction | mean(\|`y − ŷ`\| / `y`) | Consistent for the median of order −1, which lies below the median. Rewards low forecasts. Reported and flagged as for MAE. `y > 0` always holds for VIX and RV21. (amended M0, review B) |
| QLIKE | Variance | mean(`v/v̂ − ln(v/v̂) − 1`) | **Primary** for `native` forecasts. This is Patton's (2011) normalised form, his eq. 24 at b = −2: ≥ 0, and 0 only when `v̂ = v`. Needs `v, v̂ > 0`. It ranks consistently when the target is a conditionally unbiased proxy, such as RV21. |

#### 4.4 GARCH: what is comparable to a 30-day implied vol

- The GARCH family (GARCH, GJR, EGARCH) is fitted on **daily SPX log returns**, scaled to percent for
  the optimiser. It is never fitted on the VIX level.
  - VIX is already a conditional-volatility forecast. Its level is persistent and positive, not a
    zero-mean shock.
  - Fitting GARCH to it is a category error, explained in `docs/concepts/`.
- From the daily variance forecasts `σ²_{t+k|t}` (decimal², so divide `arch`'s percent² output by
  10⁴), define the annualised variance forecast for the 21-session window that starts after session
  `t+a`:

  `G_t(a) = (252/21) · Σ_{k=a+1}^{a+21} σ²_{t+k|t}`
- **Why 21 sessions and 252.** VIX² is the risk-neutral variance of the next 30 *calendar* days,
  annualised per calendar year. Measured on the NYSE calendar, 2000-2025 (via `exchange_calendars`):
  - A 30-calendar-day window holds 15-22 sessions, mean **20.64**.
  - 21 sessions span 29-38 calendar days, mean **30.50**.
  - There are **251.50** sessions per year.

  If variance accrues per session, both conventions estimate *variance per year*: weekend variance
  shows up in Monday's close-to-close return. So `G` and `(V/100)²` are on the same scale, to within
  the 252-versus-251.5 convention.
- Two comparisons, because P and Q are different measures:

  | Track | GARCH forecast | Compared with | What it shows |
  |---|---|---|---|
  | `rv21` (P vs P) | `v̂ = G_t(0)`, so `ŷ = 100·sqrt(G_t(0))`, the RMS vol (4.3) | RV over the 21 sessions after t, along with `vix_implied` (`v̂ = (V_t/100)²`) and the other RV models | Whether GARCH or VIX better forecasts realised variance. VIX is expected to be biased high. `vix_implied` is scale-tied, so its rows stay local (D2; amended M0, review E). |
  | `vix_level` (P vs Q) | `garch`, `gjr`, `egarch` uncalibrated: `100·sqrt(G_t(h))`. This is the window `VIX_{t+h}` prices. | `V_{t+h}` | **The VRP gap.** Q-variance usually exceeds P-variance, so the raw GARCH forecast should be biased low. MZ shows it as α ≠ 0 or β ≠ 1. These rows are scale-tied, so they are evaluated locally and published only as a direction (D2; amended M0, review E). |
  | `vix_level` | `*_cal`: `v̂ = a_h + b_h·G_t(h)` | `V_{t+h}` | The map from P to Q. `a_h, b_h` are fitted by OLS on training pairs whose target date ≤ cutoff. With an intercept, OLS fits the means exactly, so the average gap is `mean(v) − mean(G) = a_h + (b_h − 1)·mean(G)`. That equals `a_h` only if `b_h = 1`. It is the average VRP only if `G` is unbiased for realised variance; otherwise it also contains GARCH's own bias (concept 01). (amended M0) |

#### 4.5 Walk-forward design

| Rule | Value | Why |
|---|---|---|
| Window | **Expanding** from `train_start`, for every model | Vol regimes such as 2008 and 2020 are rare, and rolling windows throw them away. Expanding windows are also simpler to explain. |
| Fold length (refit cadence) | **63 sessions** (about a quarter), the same for every model | The first fit already has 2,579 sessions (1990-01-02..2000-03-15, measured), and the window only grows, so a quarter of new data barely moves the parameters. It also keeps the LSTM affordable. |
| Fold anchoring | Cutoffs `c_k = idx(holdout_start) + 63k` with `c_k ≥ idx(dev_start)`, indexed on 𝒯. The fold schedule is a pure function of (𝒯, `dev_start`, `holdout_start`, `fold_len`), with no data involved. It is computed once and **passed into** every run, so a run on truncated data uses the same cutoffs as the full run. (amended M0) | No fold straddles dev and holdout, and truncated runs stay comparable (4.7) |
| Fit at cutoff `c_k` | Parameters come from data dated ≤ `c_k`. A supervised pair `(x_τ, y_{τ+h})` is usable only if `τ + h ≤ c_k` (label availability). | Without it, the h training origins `c_k−h+1 … c_k` would carry labels dated `c_k+1 … c_k+h`, up to **h** sessions past the cutoff. For `rv21`, the pair at `τ = c_k` contains the entire window the fold's first forecast must predict. (amended M0: was "h−1") |
| Within a fold | Origins `t ∈ [c_k, c_{k+1})`. Parameters are frozen. State and inputs update daily with data ≤ t: Kalman state, GARCH recursion, HAR regressors, LSTM input window. | Every forecast uses all the information available at t, and no more |
| Multi-step | Each model's native method, recorded in `model_def.multistep`. Iterated for ARIMA and GARCH; direct for HAR, the LSTM and calibrations. | |
| Key | `(run_id, model_id, target_id, origin_date, horizon)`. `target_date = 𝒯[idx(origin)+h]` is stored and checked. | |
| Evaluable origin | `idx(origin) + h ≤ idx(data_end)`. This filters **scoring**, not production: the pipeline forecasts at every origin up to the last available date, including origins whose target is not yet observed. Such rows have no `target_value` and are not scored. (amended M0) | A truncated run must still emit its forecast at the truncation date (4.7) |
| Same rows | For each (target, h, period), every model has a scored forecast at exactly the same set of origins. A failed fit falls back to the previous fold's parameters (`fit.fallback`). A missing row is an error, never a silent drop. | DM and MZ compare like with like |

On the measured NYSE calendar (`dev_start` 2000-01-03, `holdout_start` 2024-07-01), this gives
**97 dev folds**, with the first cutoff on 2000-03-15. Measured on 2026-10-27, the holdout ran to
2026-10-26: 583 sessions, or 10 folds, the last one partial.

The 50 sessions 2000-01-03..2000-03-14 belong to no fold. This is **intended**: `dev_start` is a
lower bound, not the first origin. Anchoring folds on `holdout_start` puts the first cutoff at
2000-03-15, and those 50 sessions are burn-in. They are training data for the first fit, never
forecast origins, and `calendar.period` labels them `burn_in`. (amended M0, review B)

**As built in M2 (amended M2).** `ivf.evaluation.walkforward` and `ivf.models.baselines`; the
walkthrough is `docs/milestones/02-baselines.md`. What M2 settles:
- **Fold ids** count from the first dev cutoff: 0..96 are the dev folds, 97 onwards the holdout
  folds. The `fold` table holds all of them (dates only) and `calendar.fold_id` is filled.
- **The holdout stays unread in M2.** The baselines run dev folds only, on M1's `eda_inputs`
  (sessions and interim rows strictly before `holdout_start`); the runner refuses a later date or
  a holdout fold. A dev forecast whose target is on or after `holdout_start` is stored without a
  target and is not scored, as an unobserved target would be. The holdout run is a later, single
  look (4.6).
- **Same rows, made explicit for the non-core indices.** An origin is forecastable when its
  close and the 22 closes `t-21..t` are present (the `har_m` window, G2), and a symbol's fold is
  fitted only when at least `min_train_sessions` (504) of its closes are dated ≤ the cutoff.
  Both rules depend on the data, never on the model, so every model has the same rows; both are
  counted in the run output. For VIX in dev neither removes anything (G3).
- **Nulls in training rows (answers G3).** ARIMA's Kalman filter treats a missing close as a
  missing observation; HAR drops a pair with a null in `t-21..t` or at `t+h`; the historical
  means skip nulls and count what they used.
- **Fallback** (`fit.fallback`): `none`, `previous_fold`, or, for `arima_log`'s first fold only,
  `rw_log` (ARIMA(0,1,0), whose variance has a closed form).
- **Log-space models** report `ŷ = exp(μ + s²/2)` as above and also store `exp(μ)` as
  `yhat_naive`; the metric table scores both (`forecast = mean | median`) so the effect of the
  back-transform on each loss is visible. Only `mean` is the model's forecast.
- **ARIMA order** is re-selected at every cutoff on training data only: (p, q) ∈ {0,1,2}² by
  AIC among converged fits, with **d = 0 and a constant** (an ARMA on ln IV, which reverts to its
  mean); the unit-root case is the random walk's. (amended M2, review A: d was chosen by KPSS at
  5%, which picked d = 1 on 17-52% of stationary AR(1) series with phi = 0.98 at fold sample sizes
  and at all 97 real VIX cutoffs; KPSS remains an option, `d_rule: kpss`, not the default.)
  **HAR** is direct, one OLS per horizon, with `s² = RSS/(n−4)`.
- **L2 and window targets (open, M3).** L2 holds by construction in M2 because every value sits
  on the row of the date it is observed. A window target stored on its origin's row (as M1's
  `forward_rv21`) would not; `rv21` needs its own L2 test in M3. (amended M2, review A)
- **Secondary targets.** The other implied-vol indices of `configs/iv_indices.yaml` are
  forecast as `<symbol>_level` with the same calendar, folds and rules. They are not protocol
  targets (adding one would change the frozen `protocol_hash`); owner decision O-M2-1.

#### 4.6 Periods and the holdout

| Parameter | Value |
|---|---|
| `train_start` | 1990-01-02, the first VIX date. Verified in M1 on Cboe's file; `check` re-verifies it on every run (amended M1). |
| `dev_start` | 2000-01-03, so the first fit has about 10 years of data |
| `holdout_start` | A **fixed date** in `configs/protocol.yaml`. **Proposed: 2024-07-01** (see O1). |
| `data_end` | Frozen once in M1 at the last session with complete core data: **2026-10-28** (amended M1; rule in 4.1). Changing it creates a new `protocol_hash`. |
| Period of a forecast | `holdout` if `origin_date ≥ holdout_start`, else `dev` |

- **Why a fixed date and not "the last 24 months".** A rolling definition moves every time the data
  is refreshed. The holdout then quietly changes, and anything already seen leaks into model choices.
- **Why 2024-07-01.** The plan asks for a holdout that "includes Aug-2024 and Apr-2025". Measured on
  2026-10-27, the last complete session was 2026-10-26, so "the last 24 months" would start on
  2024-10-28 and **exclude Aug-2024**. Those two requirements conflict.
  - With 2024-07-01, the 5-Aug-2024 spike is a holdout *target* at every horizon. Its h=21 origin is
    2024-07-05, h=5 is 2024-07-29 and h=1 is 2024-08-02 (measured).
  - The early-April-2025 spike is also inside: its h=21 origin is 2025-03-10.
  - The holdout is about 28 months, not 24.
- **Dev purge.** Anything chosen by looking at evaluation results is a hyper-parameter: model
  choice, architecture, thresholds, strategy rules. Those choices use only dev rows with
  `target_date < holdout_start`. The last h dev origins of each horizon are dropped from tuning.
- **One look.** The holdout is scored with hyper-parameters frozen. If a hyper-parameter changes
  after the holdout has been seen, `protocol.holdout_looks` increments and the report must print it.
- **M1's EDA computes nothing on holdout rows (amended M1, review A).** Every EDA statistic uses
  sessions and interim rows strictly before `holdout_start`, so no origin and no target date is in
  the holdout (`ivf.data.process.eda_inputs`; `eda_report` refuses a holdout row). Its samples are
  `pre_holdout` (burn-in and dev together), `burn_in` and `dev`, the calendar's own periods. The
  first M1 version printed and stored holdout statistics (half-life, ADF/KPSS, spread, contango)
  before any model existed, a look this rule forbids, and its "dev" sample silently included
  burn-in. Holdout descriptives wait for the scored holdout run. `holdout_looks` stays 0.
  A re-tuned strategy becomes a new `strategy_def` row (rows are retired, never deleted), so DSR's
  `n_trials` counts it automatically.
- **In-sample criteria are part of fitting.** AIC order selection, MLE parameters, scaler
  moments and the `*_cal` OLS are re-estimated at every cutoff, holdout included, using only data
  available at that cutoff.

#### 4.7 No-look-ahead rules (each one becomes a test)

| # | Rule |
|---|---|
| L1 | The information set at t is data dated ≤ t. Features use trailing windows only: no centred rolling, no full-sample normalisation, no `bfill`. `ffill` is allowed only from a series' own past. |
| L2 | Label availability: training pairs need `target_date ≤ cutoff` (4.5). |
| L3 | Scalers, imputers and clipping thresholds are fitted at each cutoff, on the rows used for gradient or OLS updates only. For the LSTM, that means inner-train rows, not the early-stopping rows. |
| L4 | The LSTM's early-stopping block is the last 252 training origins whose targets are ≤ cutoff. An h-session purge separates the last inner-train target from the first validation origin. |
| L5 | Regime labels are computed from data ≤ their date, with thresholds fixed in config, not sample quantiles. That makes them usable as conditioning variables and as trading inputs. |
| L6 | Hyper-parameters are chosen on dev rows with `target_date < holdout_start` (4.6). |
| L7 | Backtest: a signal from origin t is executed at the close of t+1, so it first earns the return from t+1 to t+2. |
| **Test** | **Truncation test.** Forecasts at origin t from a run on data truncated at t must equal the full run's forecasts at t (the LSTM to within tolerance). This one test catches most look-ahead bugs. It works only because the fold schedule is passed in, not derived from the data (4.5), and because forecasts are produced at origins whose target is not yet observed (4.5, "Evaluable origin"). It cannot catch a leak that is identical in both runs, such as a threshold chosen by looking at the full sample. (amended M0) |

#### 4.8 Regimes, tests and suppression

| Item | Rule |
|---|---|
| `vix_level` regimes (labelled at the origin) | `low`: V < 15. `mid`: 15 ≤ V < 25. `high`: V ≥ 25. These are fixed round numbers, not quantiles. |
| `term_structure` regimes | `backwardation` if VIX > VIX3M, else `contango`. Null before VIX3M's first date (to verify in M1). Review C found Cboe's `VIX3M_History.csv` starts 2009-09-18, so this regime is null for roughly the first decade of the dev period unless M1 adds an earlier VXV source; M1 measures it and decides (amended M0, review C). **M1 decision (amended M1):** M1 confirmed 2009-09-18 on the file and adds no earlier source. The regime is null before then (no `regime_label` row), so regime cells for `term_structure` cover only the later part of dev. An earlier VXV series would be a second licensed source with its own splice risk, for a regime that never reaches committed reports (O7). |
| Diebold-Mariano | `d_t = L_a − L_b` per origin, on identical rows. Losses: squared error (level) and QLIKE. The long-run variance uses the **rectangular** estimator, with the HLN correction and a Bartlett fallback, as set out below the table. (amended M0: was Newey-West) |
| Mincer-Zarnowitz | `y = α + β·ŷ + ε`, with a joint Wald test of α=0 and β=1 using NW lag h−1. Run in level space for `vix_level` and in variance space for `rv21`. The Bartlett kernel's downward bias (below) applies here too: see O5. (amended M0) |
| Suppression | In the local `metric` table and the local app, any cell with n < `min_cell_n` (20) is written as null, with its n kept, because so small a cell is unreliable. Regime cells never reach committed real reports at all (D2 P), so the M0 secondary-suppression rule is superseded (amended M0, review D). The holdout's `high` regime may be thin. The dev period is reported alongside it, labelled "dev (used for tuning)". (amended M0, review B) |

**DM long-run variance (amended M0).**
- **Assumption, not a consequence (amended M0, review B).** DM, and HLN (1997, p. 282), *assume* that
  the loss differential's autocorrelations vanish beyond lag h−1. Optimal h-step *errors* are
  MA(h−1), but that does not carry over to `d_t` for nonlinear losses. It fails when the two models'
  relative accuracy depends on a persistent volatility state, which is the normal case for vol
  forecasts.
- **Estimator.** The original Diebold-Mariano (1995) estimator, with equal (rectangular) weights
  truncated at lag h−1: `LRV = γ̂₀ + 2·Σ_{j=1}^{h−1} γ̂_j`.
- **Small-sample correction.** Multiply DM by the Harvey-Leybourne-Newbold factor
  `√((n + 1 − 2h + h(h−1)/n) / n)` and use t_{n−1} p-values. HLN derived it for DM's equal-weight
  estimator: HLN (1997) eq. 5 writes that estimator, eqs. 6-8 derive its expectation, and eq. 9
  (p. 283) gives the factor. This was confirmed from the primary source by review B's documentation
  channel (ledger C2.4) and review D (C5.8). (amended M0, review B: the [unverified] tag is removed;
  amended M0, review D: equation roles corrected)
- **Fallback.** The rectangular estimate is not guaranteed positive. If it is ≤ 0, use Bartlett
  (Newey-West) weights with lag h−1 for that row, and flag the row (`dm_test.lrv_method`, added in
  M5).
- **Why not Bartlett by default.** With lag h−1, Bartlett weights `1 − j/h` under-estimate even in
  population. For the triangular autocorrelation of overlapping random-walk errors at h = 21:
  `1 + 2·Σ_{j=1}^{20}(1 − j/21)² = 1 + 2·2870/441 = 14.02`, against a true 21 (ratio 0.667,
  confirmed numerically).
- **Measured** (20,000 simulations under H0, h = 21, nominal 5%, seed 20261028; Monte Carlo SE
  ≈ 0.0015):

  | d_t | n | Rectangular + HLN | Bartlett + HLN |
  |---|---:|---:|---:|
  | Overlapping RW error (linear) | 583 | 0.0636 | 0.1189 |
  | Overlapping RW error (linear) | 2,000 | 0.0542 | 0.1090 |
  | Squared-error differential, error correlation 0.8 | 583 | 0.0590 | 0.0896 |
  | Squared-error differential, error correlation 0.8 | 2,000 | 0.0553 | 0.0938 |

  The rectangular estimate was never ≤ 0 in these runs. These designs satisfy the assumption above.
- **When the assumption fails (amended M0, review B).** Review B's ledger, row C2.6b, used a
  state-dependent null: model A's error scales with a log-variance AR(1) state (φ = 0.98, sd 0.5),
  model B's is homoskedastic, and `E[d] = 0` unconditionally. I re-ran its probe (`c26d_sv.py`, same
  seed) and got the same output. This is a reproduction, not independent confirmation.
  - `acf(d)` at lags 21, 30 and 50 is 0.089, 0.066 and 0.046.
  - The long-run variance is 24.4·γ₀ with all lags, but only 16.8·γ₀ when truncated at h−1.

  | Null, h = 21, nominal 5% | n | Rectangular + HLN | Bartlett, L = 100 | Bartlett, L = 200 |
  |---|---:|---:|---:|---:|
  | Symmetric control | 583 | 0.0563 | 0.1060 | 0.1622 |
  | Symmetric control | 2,000 | 0.0507 | 0.0682 | 0.0782 |
  | State-dependent | 583 | **0.1258** | 0.1635 | 0.2098 |
  | State-dependent | 2,000 | **0.1103** | 0.1088 | 0.1073 |

  So DM p-values from this protocol can be about 2.5 times too liberal. A wider Bartlett window does
  not fix it at these sample sizes. M5 must choose a remedy by measurement (O6) and report how
  sensitive each DM result is to the bandwidth.

### D5. Determinism and reproducibility

| Concern | Rule |
|---|---|
| Python | 3.12 via `uv`. Exact `==` pins in `pyproject.toml`, `uv.lock` committed, CI runs `uv sync --locked`. |
| Node | Node 22. `package-lock.json` committed, CI runs `npm ci`. |
| Seeds | One `protocol.seed`. Each component gets `np.random.SeedSequence(seed, spawn_key=(key32(model_id), cutoff_yyyymmdd, member))`, where `key32(s)` is the first 4 bytes of `sha256(s.encode("utf-8"))` read as an unsigned big-endian integer. Streams are therefore reproducible and independent across models, folds and LSTM members. (amended M0) **Why stable keys:** a positional `model_idx` changes every seed when `models.yaml` is reordered, and `fold_id` changes when `dev_start` moves; a model id and a cutoff date do not. Python's built-in `hash()` is salted per process and must not be used. Measured: `(key32("har_log"), 20240701, 1)` drew 0.45037008942388645 in two separate processes, while `hash("har_log")` differed between them. Torch uses `manual_seed` and `use_deterministic_algorithms(True)`. EGARCH multi-step forecasts are expected to need simulation in `arch`, which must be seeded (to verify in M3). |
| Fixture platform | Torch is CPU-only (ADR 0002). The thread count is always set explicitly, because BLAS reduction order, and therefore the last bits of the floats, can depend on it. The committed synthetic fixture uses 1 thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), so it reproduces across machines. Real runs may use ADR 0002's faster setting. `fit.device` is still recorded, and `CUBLAS_WORKSPACE_CONFIG` matters only if the GPU variant is ever enabled. |
| `--check` modes | `export_app_data.py --mode synthetic --check` regenerates into a temp dir and compares with the committed fixture. It ignores `generated_at`, `source_commit`, `source_dirty` and `run_id` (amended M0: run_id added to check-ignored provenance fields), and allows numeric differences of one unit in the last exported decimal. Also: `build_notebooks.py --check` (notebook sync), `ivf data verify` (raw hashes), and `ivf reports --check` (local only, since it needs real data). |
| Contract tests | pytest validates exporter output against `schemas/app-data/` with `jsonschema` (`Draft202012Validator` plus a `referencing.Registry` holding the envelope). vitest is to validate the committed fixture with `new Ajv2020({strict: true, allowUnionTypes: true})` from `ajv/dist/2020`. **That TS/Ajv side does not exist yet; it arrives with the first real JSON in M1** (amended M0). The schemas use `pattern`, not `format`, so neither side needs a formats plugin. Semantic invariants (array lengths, index ranges, sort order) are checked by the exporter before it writes, and asserted by the `app/src/lib` loaders. |
| Regex dialect (amended M0) | JSON Schema patterns are ECMA-262 regexes, but python-jsonschema runs them with Python's `re`, where `$` also matches before a final `"\n"`. **Every pattern therefore ends in `$(?!\n)`**: "end, and no newline follows", which behaves the same in both engines. Patterns also use `[0-9]`, not `\d`. **Measured before the fix:** a trailing `"\n"` on 11 pattern-checked fields (every distinct pattern) was accepted by python-jsonschema and rejected by Ajv 8.20.0. **After the fix:** both reject all 11. Python `re.search` and node `RegExp(p, "u")` agree on 45 of 45 cases: 9 patterns × {valid, `+\n`, `+\n\n`, leading `\n`, `+\r`}. `tests/test_schemas.py` asserts the rejection per field, plus that every pattern in the schemas is exercised. It was mutation-checked: un-guarding any one pattern fails 2-3 tests, and dropping a case fails the coverage test. |
| JSON hygiene | `json.dumps(allow_nan=False)`: NaN becomes `null` explicitly, because bare `NaN` is not valid JSON. Each field is rounded to a fixed number of decimals (listed in the data model). Dates are `YYYY-MM-DD` session dates; instants are UTC. |
| Clean-tree rule | Files in `reports/` must come from a run with `source_dirty = false`: commit the code, run, then commit the reports. The synthetic fixture may be dirty, because CI's `--check` is what proves it matches the code. |

## Alternatives considered

| Decision | Rejected alternative | Why rejected | Would win if |
|---|---|---|---|
| D1 | The app computes metrics in TypeScript from forecasts | Two implementations of every metric drift apart, and the reviewer must re-measure both | The app needed interactive what-if recomputation (it does not) |
| D1 | One version per file | More bookkeeping for no benefit, since producer and consumer share a repo | Files were consumed by third parties on separate release cycles |
| D1 | Array-of-objects JSON | Larger, because every row repeats every key. It is also slower to turn into typed arrays for three.js. | Payloads were tiny and read by humans |
| D2 | Commit real JSON to a public repo | Redistributes Cboe/Yahoo data, against their terms | We held a redistribution licence |
| D2 | A private repo for real outputs | Splits the project, and the user asked for public | Real-data screenshots became essential |
| D2 | App fetches Cboe live in the browser | CORS, terms and rate limits, and the app would then compute | Cboe offered a licensed public API |
| D2 | Write real JSON to `app/public/data/real/` (gitignored) | `vite build` copies `public/` into `dist/`, so one deploy leaks it | — (kept only as the second-layer `.gitignore` guard) |
| D3 | Overwrite one raw file per symbol | Loses the bytes a past result came from, so results cannot be reproduced | Sources were immutable |
| D3 | A database (DuckDB or SQLite) as the store | Another tool to learn. Parquet plus pyarrow is enough here: the forecast table is estimated at under 10⁶ rows. | Tables grew past memory, or ad-hoc SQL became central |
| D4 | Rolling window | Drops the rare crisis regimes, and the parameters get noisier | A measured structural break makes old data harmful (worth a sensitivity run) |
| D4 | Refit every session or every 21 sessions | 63× or 3× the fits for a negligible parameter change. Monthly refitting triples LSTM cost. | Measured parameter drift between quarterly refits exceeds about 1 SE, or an LSTM fit takes ≤ 5 s |
| D4 | Refit every 252 sessions | Parameters go stale through a crisis year | Compute was the binding constraint |
| D4 | Evaluate GARCH against VIX directly, with no RV track | Mixes P and Q, so the VRP gap would read as model error | — |
| D4 | QLIKE using each model's own `E[v]` everywhere | Needs a predictive variance from RW, historical mean and the LSTM, which they do not have. The rule would be uneven. | Every model produced predictive distributions |
| D4 | Correct `native` vol forecasts to a mean, `ŷ ≈ 100·(√v̂ − Var_t(v)/(8·v̂^{3/2}))` (amended M0) | Needs a forecast of `Var_t(v)`, the variance of variance. GARCH could simulate one, but `vix_implied` and `*_cal` have none, so the correction would be uneven. It would also add a second-order approximation to every native row. | Every native model produced a predictive distribution of `v` |
| D4 | DM with Newey-West (Bartlett) weights at lag h−1 (amended M0) | Under-estimates the long-run variance of MA(h−1) differentials. Measured size was 0.09-0.12 at a nominal 0.05 (4.8). | The rectangular estimate were often ≤ 0 on real data (then raise the Bartlett bandwidth instead) |
| D4 | "Last 24 months" as the rule | It moves with every data refresh, and at today's date it excludes Aug-2024 | The user prefers exactly 24 months over including Aug-2024 (O1) |
| D4 | Regime thresholds from sample quantiles | Full-sample quantiles are look-ahead when used as a trading input | Regimes were used only for after-the-fact description |

## Consequences

- **Good.**
  - Every published number has one implementation, one provenance chain and one schema.
  - CI runs the real pipeline end-to-end on synthetic data.
  - The public repo is designed to stay licence-clean. The D2 guards keep real daily data out of the
    repo tree, but they are tripwires with stated residuals, not a proof. (amended M0, review E:
    "by construction" removed)
- **Cost.**
  - Real-data views exist only on the user's machine.
  - README screenshots of the ribbon, race and equity views show synthetic data, with a badge.
  - The holdout is about 28 months, not 24 (pending O1).
  - The 63-session cadence is a compromise and is revisited in M4 with measured LSTM fit times.
- **Follow-ups.**
  - M1 (done, amended M1): `train_start` 1990-01-02 verified; first dates on Cboe's files VIX9D
    2011-01-04, VIX3M 2009-09-18, VIX6M 2008-01-02, VIX1Y 2007-01-03 (and VVIX 2006-03-06);
    `tenor_days`: VIX's 30 calendar days is stated in Cboe's VIX methodology, and VIX9D 9, VIX3M 93,
    VIX6M 184 and VIX1Y **366** days in Cboe's "Selected SPX Target Expected Volatility Term Indices
    Methodology" (version 7.0, last revised March 4, 2026, section 2.1, pp. 5-6; VIX1Y uses the "366
    Day Bracket Method"). M1 first wrote 365 and called 9, 93, 184 and 365 unverified for want of a
    primary source; that was a search failure, not a fact (amended M1, review A). `data_end` frozen
    at 2026-10-28. The original follow-up read:
  - M1: verify `train_start`, the VIX3M, VIX9D, VIX6M and VIX1Y start dates, and `tenor_days`, then
    freeze `data_end`.
  - M3: verify `arch` EGARCH multi-step behaviour.
  - M6: verify the VXX series switch and SVXY leverage-cut dates against the issuers' documents.

## Open items for the orchestrator

| # | Question | Recommendation |
|---|---|---|
| O1 | Holdout start: a fixed 2024-07-01 (about 28 months, includes Aug-2024 and Apr-2025) or exactly 24 months (starts 2024-10-28 on data to 2026-10-26, excludes Aug-2024)? | **Fixed 2024-07-01.** The stress events are the reason for the holdout. Also restate the plan's "24 months" as "since July 2024". |
| O2 | LSTM refit cadence. ADR 0002's compute estimate assumes 24 refits; this protocol gives about 107 (97 dev + 10 holdout). | **Keep 63 sessions for the LSTM too.** Scaling ADR 0002's figure (about 25 min on a CPU pool for 18,000 model-epochs) linearly to 107 × 3 × 5 × 50 ≈ 80,000 model-epochs gives about 2 h. That is derived, not measured, and it ignores the expanding training set. Measure in M4. If it is too slow, refit the LSTM every 4th fold and reuse its weights in between, recorded in `fit`. This handicaps the LSTM and leaks nothing. ADR 0002's "24 refits" should be updated. |
| O3 | Real-data time-series figures in `docs/` (e.g. a VIX history plot)? | **No.** Use synthetic figures, or link to Cboe's own charts. |
| O4 | Real JSON outside the repo (`ROOT/app-data/real/`) instead of the brief's `app/public/data/` | **Accept.** It is the only placement a `vite build` cannot leak. |
| O5 (amended M0) | MZ uses Newey-West at lag h−1, the same kernel the DM amendment rejected. Overlapping MA(h−1) errors bias its Wald test the same way. | **Decide in M5 by measurement.** Run the same simulation for the MZ Wald size, then choose between Hansen-Hodrick (rectangular, lag h−1, with a fallback when the covariance matrix is not positive definite) and Newey-West with a longer bandwidth. Not changed in M0 because MZ size was not measured. |
| O6 (amended M0, review B) | DM size under state-dependent loss differentials: 0.1258 at n = 583 (4.8) | **Decide in M5 by measurement.** Add the state-dependent null to the M5 size simulation, then compare three candidates: an automatic bandwidth (Andrews 1991), fixed-b critical values (Kiefer-Vogelsang), and a block bootstrap of `d_t`. Report every DM result's sensitivity to the bandwidth. Do not call `statsmodels.tsa.stattools.diebold_mariano_test` (0.15.0) for the 4.8 estimator: it defaults to `lags = max(h-1, ceil(n**(1/3)))` and uses Bartlett (HAC) weights even with `lags=h-1`, so it computes the fallback, not the rectangular estimate (measured in docs/concepts/02, Reproduce block; amended M0). The rectangular estimate is available as `OLS(d, 1).fit(cov_type="HAC", cov_kwds={"maxlags": h-1, "kernel": "uniform", "use_correction": False})`, with the HLN factor applied by hand; on overlapping 21-step errors (n = 583) it reproduced a hand-coded rectangular t to 6 dp (-0.913913, against -1.094819 Bartlett) (amended M0, review C). |
| O7 (amended M0, review B) | Should real-data **regime** numbers ever be public? D2 P keeps them local. | **No.** Regime cells would need a separate disclosure analysis that P avoids (amended M0, review D). This is the user's call. |
| O8 (amended M0, review E) | Publish the scale-tied rows (`vix_implied` on `rv21`; uncalibrated `garch`, `gjr`, `egarch` on `vix_level`) in P? | **No (orchestrator default).** Measured leak, on 10 simulated seeds with public SPX (`tools/licensing/p_tie_seeds.py`): the holdout RMS VIX-RV gap to a median 0.053%, and the holdout mean VIX to a median 4.29% (max 13.69%). These are period-level VIX statistics, not daily values. Opting in breaks M2 of the proposed definition (O9). The user's call. |
| O9 (amended M0, review E) | What is a "material part" of a vendor series (the D2 objective)? | **Adopt the proposed definition in D2:** M1 (a run of 5 or more consecutive values within ±1 vol point or ±1%) and M2 (any level-unit statistic within ±10%), using committed content plus public data other than the series itself. P is argued, not proven, to comply. The user's decision. |
| O10 (amended M0, review E) | Is `lstm_log` in P? | **Decide in M4** from its inputs, re-probed with `tools/licensing/p_tie.py` (c4). With `iv_rv_spread`, a fixed-threshold flag or an absolute clip, it is scale-tied and stays local. |
