# 03 · Reproducibility and data contracts

What it takes for a number to be re-made by someone else, and for the app to read it without
guessing. Background: [00 · Background](00-background.md). Numbers are computed with the
commands shown, or cited to the ADR that measured them.

## 1. First principles: three meanings of "the same"

1. **Same environment**: the same library bytes (pins, lockfile).
2. **Same numbers**: the same arithmetic order and random streams (seeds, threads, tolerances).
3. **Same shape**: the consumer reads what the producer writes (a versioned schema).

On top, licensing decides *what* may be published, and provenance records *which* code made it.

## 2. Exact pins and the lockfile

`==` pins in `pyproject.toml` record **intent**; `uv.lock` records every wheel's URL and SHA-256,
the **bytes**, and CI is to run `uv sync --locked` (ADR-0001 D5; no CI workflow yet).
`grep -c '^\[\[package\]\]' uv.lock` prints **137**, ADR-0002's count (136 packages plus the project).

A version string does not identify a binary. ADR-0002 measured `torch==2.14.1` resolving to five
different aarch64 wheel files, and the `+cpu` and `+cu130` builds training to different weights
with the same seed and one thread. The lock, not the version, picks the build.

## 3. Deterministic bytes

The exporter writes `json.dumps(doc, allow_nan=False, sort_keys=True, separators=(",", ":"))`
(`src/ivf/contract.py:220`; default `ensure_ascii`, so bytes do not depend on the platform encoding).

- **Key order.** `json.dumps({'b':1,'a':2}) == json.dumps({'a':2,'b':1})` is `False`: same content,
  different bytes and SHA-256. With `sort_keys=True` it is `True`.
- **NaN is not JSON** (RFC 8259). Python's default writes `[NaN, Infinity]`; Node 22's
  `JSON.parse("[NaN]")` throws `SyntaxError`; `allow_nan=False` raises, forcing an explicit `null`.
- **Rounding needs a tolerance.** Let $q$ be one unit in the last exported decimal. Rounding moves a
  value by at most $q/2$, so if two runs differ by $\lvert a-b\rvert < q$, then
  $\lvert \mathrm{round}(a) - \mathrm{round}(b)\rvert \le \lvert a-b\rvert + q < 2q$; being a multiple of $q$,
  it is at most $q$. Example: `2.675` is stored as 2.67499999999999982236 and rounds to 2.67; the next
  double up (+4.4e-16) rounds to 2.68. Hence ADR-0001 D5's one-unit `--check` tolerance, sound only
  if run-to-run noise is far below $q$.

## 4. Floating point, threads and seeds

Floating-point addition is not associative: `(0.1+0.2)+0.3` is `0.6000000000000001`,
`0.1+(0.2+0.3)` is `0.6`. Summing the same $10^6$ standard normals (seed 0), a forward loop lands
47 ulps from the correctly rounded sum (`math.fsum`), a backward loop 446 ulps, `np.sum` on it.
A multi-threaded reduction combines per-thread partial sums, so the thread count changes the
bracketing and the last bits. ADR-0002 measured: same lock, machine and thread count gave
**bitwise identical** LSTM weights; 1 and 2 threads agreed, 4 differed. Consequences:

- Tests compare model outputs with **tolerances**; conclusions are checked across seeds, not hashes.
- Set the thread count explicitly (ADR-0002: torch's default 20 threads ran 6.7–20× *slower* than
  8). The committed fixture uses 1 thread (ADR-0001 D5).
- Random streams come from **stable keys**: `SeedSequence(seed, spawn_key=(key32(model_id),
  cutoff_yyyymmdd, member))`, `key32` being the first 4 bytes of `sha256(model_id)`. A list index
  would change every seed when `models.yaml` is reordered; Python's `hash()` is salted per process
  (two runs printed different `hash('har_log')`). Measured: `key32('har_log')` = 3421073489, and
  root seed 20261028, cutoff 20240701, member 1 draws 0.45037008942388645, as the ADR records.
- A seed can be ignored: `arch`'s EGARCH simulation needs `rng=`, not `random_state=` (ADR-0002).

## 5. Semver for a data contract

ADR-0001 D1 versions the whole contract with one `schema_version`. The useful lens is *who breaks*:

| Bump | ADR-0001 D1 examples | Why |
|---|---|---|
| MAJOR | rename/remove a field; change type, units or index base; add a closed-enum value (`data_mode`, `period`, …) | An old app can misread or crash, e.g. on an enum value it has no branch for. |
| MINOR | add an optional field or a dataset file | Old apps work **if they ignore what they do not know**. |
| PATCH | descriptions, titles, examples | Valid documents unchanged. Every schema edit bumps at least PATCH. |

`1.0.0` is released by the M0 commit; the envelope pattern pins MAJOR = 1. **Tolerant reader:** the
schemas are strict (`additionalProperties: false`, e.g. `schemas/app-data/manifest.schema.json:15`),
so a MINOR-newer document fails an older copy. The producer and same-commit tests validate strictly;
the app at runtime checks only the MAJOR and the fields it uses, and ignores unknown ones.

## 6. JSON Schema as a two-sided contract test

One set of schema files, two validators, so the producer cannot ship what the consumer rejects.
Python validates the re-parsed export bytes (`jsonschema` 4.26.0, `src/ivf/contract.py:260-286`,
`tools/export_app_data.py:147-150`); vitest with `Ajv2020({strict: true})` arrives in M1.

Both must read a pattern the same way. JSON Schema specifies ECMA-262 regexes, but Python's
`jsonschema` uses `re`, where `$` also matches before a final newline: a trailing `"\n"` on
`schema_version`, `source_commit`, `generated_at` or `run_id` once passed Python and failed Node's
`new RegExp(p, "u")`. **Every pattern now ends in `$(?!\n)`**; re-measured, all four are rejected,
and `tests/test_schemas.py` gives 49 passed (ADR-0001 D5 reports Ajv 8.20.0 agreeing; cited).
Patterns use `[0-9]`, since Python's `\d` also matches non-ASCII digits.

## 7. Columnar vs array-of-objects

A model-race-sized synthetic panel (seed 0): 835 origins × 11 series = 9,185 values at 3 decimals,
written with the repo's `serialise` (command under Reproduce):

| Layout | Bytes | gzip level 1 | gzip level 6 |
|---|---:|---:|---:|
| Columnar (one array per series, shared `origin_idx0`) | 58,903 | 26,755 | 22,786 |
| Array of objects (keys repeated per row) | 167,809 | 47,320 | 35,764 |

Raw 2.85×; gzipped 1.77× at level 1 (nginx's default; `app/nginx.conf:21-22` sets none) and 1.57×
at level 6, as repeated keys compress well. A column also maps onto a `Float32Array`.

## 8. Licensing as an architecture constraint

**Objective** (ADR-0001 D2, *Proposed* until the user accepts it; not legal advice): nothing
committed, git history included, lets anyone reconstruct a vendor daily series or a material part
of it; isolated closes are public facts, not the harm. **"Material part" is the user's open decision
(O9).** Proposed, given everything committed plus public data other than the series itself: **M1**,
a run of ≥ 5 consecutive values to ±1 vol point (±1% for ETPs); **M2**, any level-unit statistic to ±10%.

**Per-date errors are a series.** The $h=1$ random walk's errors $e_t = V_{t+1} - V_t$ give
$V_t = V_0 + \sum_{s<t} e_s$: $[1.5, -1.7, 5.5, -1.2]$ rebuilds $[20.0, 21.5, 19.8, 25.3, 24.1]$ exactly.

**The lesson: four assumptions that failed.** A published mean times its $n$ is a sum, and sums
subtract; each review broke the previous rule.

1. **≥ 20 dates per cell** (M0). A cell going from 583 to 584 dates gives $n_2\bar b_2 - n_1\bar b_1$,
   the new day's error, to $(n_1+n_2)\times 0.00005 = 0.058$ at 4 dp (simulated: 0.416, recovered 0.3975).
2. **Refresh by ≥ 20 dates, blank sibling cells** (review B). The $h=1$ random walk's bias
   telescopes to $(V_e - V_s)/n$; regime partitions still isolated a few days.
3. **"Losses, not signed contrasts"; "no single day recoverable"** (review D). The EDA sd sums
   $V_d^2$ over the origins, the random walk's MZ fit at $h=1$ sums $V_{d+1}^2$ over them shifted by
   one; the difference depends only on $V_s$, $V_e$, and the random walk's RMSE then gives both. And
   DM `mean_d` $= RMSE_a^2 - RMSE_b^2$; mean return $=$ Sharpe × vol / 252.
4. **"An SPX-tied row needs SPX, which is not committed"** (review E). SPX is public (FRED SP500):
   from `vix_implied` rv21 ratios plus SPX, review E recovered the holdout mean VIX to a median
   4.29% (max 13.69%) over 10 seeds (`tools/licensing/p_tie_seeds.py`).

So D2 no longer promises single-day protection; it publishes an **allow-list of unitless numbers**.

**Publication set P.** Per whole period, target and horizon (or strategy, or index); no regime
cells. **Row rule:** a model's row is in P only if every number in it is unchanged when the whole VIX
path is rescaled ($V \to cV$) with *public* SPX held fixed. The benchmark for every ratio and DM test
is `rw`: $\hat y = V_t$ for `vix_level`, `rv21_back(t)` for `rv21`.

- **In P:** `hist_mean`, `arima_log`, `har_log`, `arima_garch_log`, `*_cal` on `vix_level`; `har_rv`
  and the GARCH family on `rv21` (SPX-only forecasts of an SPX-only target); `rw`'s MZ booleans and `n`.
- **Local only (scale-tied; O8, recommended no):** uncalibrated GARCH family on `vix_level`;
  `vix_implied` on `rv21`.
- **Conditional:** `lstm_log` (O10, decided in M4 with `tools/licensing/p_tie.py`); `*_cal` with a
  scale-free floor on $\hat v$; strategies from in-P models with relative bands.

| Quantity | Rounding |
|---|---|
| RMSE, QLIKE, MAE, MAPE of each in-P model as a **ratio to `rw`'s** | 3 s.f. |
| DM statistic (HLN-corrected) and p-value against `rw`, per loss | 2 dp |
| MZ: "β = 1 not rejected" and "α = 0 and β = 1 not rejected" at 5% | booleans |
| Strategy Sharpe, PSR, DSR | 2 dp |
| Turnover; AR(1) half-life of ln(index) in sessions (so, in effect, φ: intended) | 2 s.f. |
| Number of trials; `n` | exact |

Nothing else: no vol points, prices, moments, coefficients, returns, drawdowns, extremes or
quantiles; real-data EDA prose is limited to the qualitative statements E1–E7. **R1:** one commit
per frozen `data_end`; a wrong value is withdrawn, never replaced. **Q:** dated closes only for
named, cited events, never VXX/SVXY or near a period boundary; hygiene, not protection, since
closes are public anyway.

**Why ratios: scale invariance.** For a scale-equivariant model (forecasting $cV$ gives $c$ times
the forecast) every error scales by $c$: RMSE and MAE do, their ratios to `rw` do not; MAPE, QLIKE,
DM and MZ tests are unchanged. So P alone cannot return a level, but a forecast that ignores $V$'s
scale breaks this. Measured below: `rw` RMSE 1.4324, 1.7905, 2.8648 at $c = 1, 1.25, 2$; the
historical mean's ratio 4.05 each time; a fixed-level stand-in for an SPX-only forecast 4.04, 4.85,
7.97. `tools/licensing/p_set.py` found 0 of 114 P entries changed.

**Residuals, stated not closed:** scale invariance removes only the overall level; ratios between
days are unprotected (`ln(V_e/V_s)/n` sits in every QLIKE-ratio denominator), and one known close,
quoted or public, fixes the level. Moving one close by ±0.25 vol points flipped some rounded entry
on 599 of 604 sessions (`tools/licensing/p_sens.py`). No recovery of a material part is shown, none
ruled out. Also: opted-in scale-tied rows, strategies differing on a few days, quotes, legality.

## 9. `run_id`, provenance and `--check`

`run_id` is the first 12 hex of the SHA-256 of `{protocol_hash, source_commit, source_dirty,
data_mode, sorted raw sha256s}` (ADR-0001 D3; `tools/export_app_data.py:111-119`). Measured: commit
`111…1` clean gives `fbe9bc2a1017`, `222…2` clean `797fa6aed9c3`, `111…1` dirty `31c8d4c13221`.

**Why `--check` ignores it.** A file cannot contain the hash of the commit that contains it (the
hash covers the file's bytes: a fixed point, infeasible to find). So the fixture's `source_commit`
names an earlier commit (or forty zeros, dirty), `run_id` inherits that, and comparing them would
fail on every commit. `--check` drops those four fields (`tools/export_app_data.py:53-54`, `334-340`);
run today, it printed `ok: 1 file(s) …`, exit 0. A *real* `run_id` hashes real raw files, so it stays
local (D2).

## Reproduce

```bash
grep -c '^\[\[package\]\]' uv.lock                                   # 137
node -e 'try{JSON.parse("[NaN]")}catch(e){console.log(e.name)}'       # SyntaxError
uv run python - <<'EOF'
import json, math, numpy as np
print(json.dumps([float('nan')]), round(2.675,2), round(math.nextafter(2.675,3),2))  # [NaN] 2.67 2.68
r=np.random.default_rng(0).standard_normal(10**6); ex=math.fsum(r)
f=0.0
for v in r: f+=float(v)
print(f-ex, float(np.sum(r))-ex, math.ulp(ex))                        # -5.3e-12 0.0 1.1e-13
n1=583; e=np.round(np.random.default_rng(0).normal(0,1.2,n1+1),3)
b1,b2=round(float(e[:n1].mean()),4),round(float(e.mean()),4); print(e[-1],(n1+1)*b2-n1*b1)  # 0.416 0.3975
import hashlib; k=int.from_bytes(hashlib.sha256(b'har_log').digest()[:4],'big')
print(k, np.random.default_rng(np.random.SeedSequence(20261028,spawn_key=(k,20240701,1))).random())  # 3421073489 0.45037...
import gzip; from ivf.contract import serialise                       # §7 table
g=np.random.default_rng(0); n=835; M=['rw','hist_mean','arima_log','har_log','arima_garch_log','garch','gjr','egarch','garch_cal','lstm_log']
a=np.round(15+np.cumsum(g.normal(0,0.5,n)).clip(-8,40),3); C={k:np.round(a+g.normal(0,1,n),3) for k in M}
col={'origin_idx0':0,'n':n,'actual':a.tolist(),'series':[{'model_id':k,'yhat':C[k].tolist()} for k in M]}
aoo=[{'origin_idx':i,'actual':float(a[i]),**{k:float(C[k][i]) for k in M}} for i in range(n)]
for d in (col,aoo): b=serialise(d); print(len(b),len(gzip.compress(b,1,mtime=0)),len(gzip.compress(b,6,mtime=0)))  # 58903 26755 22786 / 167809 47320 35764
x=np.zeros(600); e=np.random.default_rng(0).normal(0,0.07,600)        # §8 scale invariance: log-AR(1) path
for t in range(1,600): x[t]=0.98*x[t-1]+e[t]
for c in (1,1.25,2):   # RMSE scales by c; hist-mean's ratio to rw does not; a fixed-level forecast's does
    W=c*20*np.exp(x); y=W[1:]; r=lambda f: np.sqrt(np.mean((y-f)**2)); rw=r(W[:-1])
    print(c, round(rw,4), f"{r(np.cumsum(W)[:-1]/np.arange(1,600))/rw:.3g}", f"{r(np.full(599,20.0))/rw:.3g}")  # 1.4324/1.7905/2.8648; 4.05 each; fixed level 4.04/4.85/7.97
EOF
```

## Assumptions

- The same lock on the same platform yields the same wheels.
- Run-to-run numeric noise is far below one unit of the last exported decimal.
- The app's loaders check the fields they use (M1); the tolerant reader relies on it.
- Every in-P row is scale-free given public SPX (probed on simulated paths, argued for the rest).

## Where it breaks

- **Across platforms** the lock pins wheels, not CPU kernels: expect low-bit differences.
- **D2 is an allow-list plus process rules**; joint inversion is not ruled out (§8).
- **Regex dialects** beyond `$`: any new pattern needs the `$(?!\n)` ending and a test case.

## How this repo implements it

- Code today: `src/ivf/contract.py` (envelope `:136-165`, canonical text `:205-229`, `serialise`
  `:232-234`, validation `:260-315`, `NO_COMMIT` `:56`); `tools/export_app_data.py` (provenance
  `:53-54`, `run_id` `:111-119`, `--check` `:334-399`, no float tolerance yet `:376-378`);
  `schemas/app-data/`; `tests/test_schemas.py`.
- Decisions: [ADR-0001](../adr/0001-python-core-versioned-json-app.md) D1–D3, D5;
  [ADR-0002](../adr/0002-compute-and-numeric-stack.md).
- Not yet: the Ajv side of the contract test and the float tolerance (M1), seeds (M2+).

## Self-check

1. Why can't the committed fixture's `source_commit` equal the commit that contains it, and what
   does that imply for `--check`?
2. Two runs produce 0.12344999 and 0.12345001 for a field exported with 4 decimals. What does
   `--check` see, and should it fail?
3. Why does D2 publish RMSE only as a ratio to the random walk's, never in vol points?

<details>
<summary>Answers</summary>

1. The commit hash covers the file's bytes, so the file would contain a hash of itself. It records
   an earlier commit (or zeros plus dirty), `run_id` hashes that, and `--check` excludes both.
2. They export as 0.1234 and 0.1235 (measured): one unit apart from a raw gap of 2e-8. §3's lemma
   caps such flips at one unit, so the tolerance passes it; an exact comparison would fail on noise.
3. The ratio is the same for every rescaled path $cV$ (RMSE 1.4324 → 2.8648 at $c = 2$, ratio 4.05
   both times), so P alone returns no level; absolute RMSE helped recover two closes in review D. It
   holds only for rows that scale with $V$: an SPX-tied row's ratio moves with $c$ and, SPX being
   public, leaks the level.

</details>
