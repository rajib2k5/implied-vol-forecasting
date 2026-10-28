# M0 adversarial review A: code and numerics (evidence-gated ledger)

| | |
|---|---|
| target | working tree of `~/dev/implied-vol-forecasting` on 2026-10-28; no commits exist yet, so there is no SHA |
| environment | Python 3.12.13 (uv); numpy 2.5.4, pandas 3.1.0, statsmodels 0.15.0, arch 8.0.0, pmdarima 2.1.1, torch 2.14.1+cpu, jsonschema 4.26.0, referencing 0.37.0; Node 22.23.1, npm 10.9.8; NVIDIA GB10 (compute capability 12.1), driver 580.178.04. The pin file is `scratchpad/reviewA/env-pin.txt` |
| protocol | evidence-gated-validation. Tier 2 (dual channel) for C4, C5 and C6; Tier 1 (runtime only) for the rest |
| mode | Channel A was a separate docs-only subagent: no execution, and it never saw the runtime results. Channel B was the reviewer, who ran probes only. Neither channel saw the other's results before writing its own verdicts. Probes were pre-registered in `scratchpad/reviewA/claims-ledger.md` before any doc was read |
| default verdict | every claim started as REFUTED and was conceded only on evidence produced in this session |
| claim count | 38 ledger rows: 37 material, all with falsifiers. Plus 26 mutants |
| reviewer constraints | the repo source was never edited. Mutations and attacks ran in a scratch copy (`scratchpad/reviewA/repo`, `scratchpad/reviewA/app`). The only file written in the repo is this ledger. Running the prescribed `npm run verify` rebuilt the gitignored `app/dist/` and `app/coverage/` |

`scratchpad` is `<session scratchpad>`. Raw outputs are in `scratchpad/reviewA/b_*.txt|jsonl` (Channel B) and `scratchpad/reviewA/channel-a.md` (Channel A, 34 records with verbatim quotes).

## Verdict

**FAIL.** Material refutations:
- C3: the real-mode guard. MED, and latent until M1.
- C8.1: `.gitignore` does not cover vendor CSVs or common secret files. MED, and live now.
- X-A: the D2 deploy guard is described in the present tense but does not exist. MED, latent.

LOW refutations: C1.2 (3 non-equivalent surviving mutants), C7.6 (src/lib purity), C4.1 edge cases, C5.5 and C6.1 wording.

Everything else survived the attack.

## Evidence table

A = documentation channel, B = runtime channel. "B-only" means Channel A returned UNVERIFIABLE because the claim is a measurement (split rule: probe-scoped). B-only results are single-channel evidence, not dual-confirmed.

| ID | Claim | A | B (command → output) | Gate / verdict | Sev |
|---|---|---|---|---|---|
| C1.1 | `uv run pytest -q` passes with 207 tests | n/a (T1) | `uv run pytest -q -p no:cacheprovider`: `207 passed in 10.59s`, rc 0. The scratch copy also gave 207 passed | SURVIVES | – |
| C1.2 | the suite kills mutants in config, contract, exporter and notebook builder | n/a | `mutate.py` ran 26 pre-registered mutants with pycache cleared and sha256 restored and checked: **20 killed, 6 survived** (table below) | **REFUTED (partial)**: 3 survivors are non-equivalent untested branches | LOW |
| C1.3 | no source-grep tests, no assertion-free tests, no helper reimplementations | n/a | AST scan: 7 tests have no `assert`. All 7 are positive-validation tests that pass by not raising, each paired with a rejection test. Hash literals recomputed with `printf … \| sha256sum`: protocol_hash `03fe7772…`, run_ids `fbe9bc2a1017`, `807b96c9b5dd`, `31c8d4c13221`, `886e7eacc485`, and `efbd0040…` all match. No helper reimplements code under test | SURVIVES, with two weak tests: `test_later_milestone_subpackages_say_when_they_arrive` asserts on docstring prose; `test_default_repo_is_this_project` would pass if `git_state()` always returned the placeholder | LOW |
| C2.1 | `build_notebooks.py --check` passes | n/a | `ok: 1 notebook(s) … match their specs`, rc 0 | SURVIVES | – |
| C2.2 | `export_app_data.py --mode synthetic --check` passes | n/a | `ok: 1 file(s) … match a fresh synthetic export`, rc 0 | SURVIVES | – |
| C2.3 | `uv lock --check` passes | n/a | `Resolved 137 packages`, rc 0. The lock has 137 `[[package]]` entries | SURVIVES | – |
| C2.4 | notebook bytes are deterministic across runs | n/a | 5 fresh processes (PYTHONHASHSEED random/0/12345/999, plus LC_ALL=C TZ=Asia/Kolkata, cwd `/` or `/tmp`): all `529792f1…` = the committed file. Falsifier (random cell ids): two runs gave `60bdc630…` and `13c0c65d…` | SURVIVES | – |
| C2.5 | the notebook executes headless | n/a | `uv run jupyter nbconvert --to notebook --execute …`: rc 0, 4/4 code cells executed, 0 error outputs. Output includes the envelope, the `'paper' is not one of` violation and the NaN refusal | SURVIVES | – |
| C3 | the real-mode guard cannot be made to write real data inside the repo | n/a | `attack_guard.py` plus user-namespace probes (details below). **Refused correctly:** symlinked dir, `..`, relative path, trailing `/` and `/.`, `//`, `VOL_DATA_DIR`/`HOME` pointing in. **Accepted while physically inside the repo:** a case variant on a case-insensitive FS (casefold tmpfs: `…/ivf/app/src/data` with repo `…/IVF`, same inode); a bind mount of the repo. **Wrote into the repo** with the M1 real generator simulated: output-file symlink and output-file hard link each overwrote the committed synthetic fixture with `data_mode: real`; a TOCTOU parent swap landed `repo/app/public/data/real/manifest.json`. In M0 itself real mode exits 4 and writes nothing (B4) | **REFUTED** | MED (latent until M1) |
| C4.1 | `serialise()` output has sorted keys, compact separators, ASCII-only bytes, and rejects NaN/Inf | CONFIRMED (json docs; `encoder.py:354,365-373`; `_json.c:1321-1324`) | Adversarial inputs (astral chars, lone surrogate, `-0.0`, `1e16`, `2**60`): ASCII-only, no whitespace, re-dump stable. `np.float64('nan')` raises ContractError. **But:** int keys sort numerically (`{10:1,9:2}` gives `{"9":2,"10":1}`), so `canonical_json_sha256` gives two hashes for one JSON document (`b3f50f7b…` vs `616552ed…`) and re-serialising is not idempotent. np.int64, np.float32, np.bool_, pd.NA, Timestamp, set and mixed str/int keys raise a bare `TypeError`, not `ContractError` | SURVIVES for str-keyed JSON-native input; edge cases REFUTED | LOW |
| C4.2 | serialisation is byte-deterministic | CONFIRMED (b); C4.2c "equal input": REFUTED as worded, since `{True:0}=={1:0}` but they serialise differently | 4 hash seeds give 4 different key orders and identical `serialise` sha256 every time. The falsifier (raw `json.dumps`) differs on every seed | SURVIVES (dual-confirmed) | – |
| C4.3 | Draft 2020-12 is used | CONFIRMED (`validators.py:812-856`; spec §8.2.3.1) | `Draft202012Validator(` is in `_validate`. A `$ref` sibling keyword is enforced: 1 error under 2020-12, 0 under the Draft7 falsifier | SURVIVES (dual-confirmed) | – |
| C4.4 | `$ref` resolves locally, with no network | CONFIRMED with caveats (`referencing/_core.py:683-684`; `validators.py:285-286,463-464`) | Validation succeeds under `unshare -rn` (`Errno 101` for sockets). Socket/urlopen/requests patches recorded 0 calls. An unregistered https `$ref` raises `_WrappedReferencingError`, which is **not** a `ContractError`/`ValueError`, so `check()` would traceback instead of reporting | SURVIVES (dual-confirmed) | LOW (exception type) |
| C4.5 | pattern verdicts agree between python-jsonschema and Ajv 8 / `RegExp(p,"u")`, including trailing `\n` | CONFIRMED (`_keywords.py:218` uses `re.search`; ES2024 §22.2.2.4; Ajv `lib/vocabularies/code.ts:95-104`) | `js/gen_cases.py` + `js/diff.mjs`, Ajv **8.20.0** installed in scratch only: **32,151** cases × 9 patterns (suffixes `\n \n\n \r \r\n U+2028 U+2029 U+0085 NUL BOM`…, non-ASCII digits, Kelvin sign, lone surrogates, 3,000 random strings per pattern) gave **0 disagreements** with ADR options and with defaults. Falsifiers: unguarded `$` gave 21 disagreements; `\d` gave 20 | SURVIVES (dual-confirmed) | – |
| C4.6 | whole-document verdicts agree | CONFIRMED (no semantic difference for integer, uniqueItems, const/enum, minLength, exclusiveMinimum, additionalProperties next to `$ref`) | **9,523** documents (5 datasets × every leaf × 29 substitutions, plus key deletions and extra keys): 1,080 valid in both, **0 disagreements** | SURVIVES (dual-confirmed) | – |
| C4.7 | Ajv 8 compiles every schema | CONFIRMED (strict-mode docs; `validate/index.ts:293-305`) | `new Ajv2020()` and `new Ajv2020({strict:true, allowUnionTypes:true})`: 6 schemas compiled, 0 strict warnings | SURVIVES (dual-confirmed). Note: **no TS/Ajv validator exists in `app/`**, see X-B | – |
| C5.1 | cu130 torch runs on the GB10 | CONFIRMED at build-config level (`.ci/manywheel/build_env_setup.py@v2.14.1:77-79`, sm_120); Tegra caveat open | `probe_gpu.py` in `envprobe/torch-cu130`: arch list includes `sm_120`; copy, matmul, `x+1`, `zeros` OK; LSTM fwd+bwd vs CPU `|dloss| 2.62e-06`, `max|dgrad| 6.91e-06` | SURVIVES (dual-confirmed) | – |
| C5.2 | cu126 fails with "no kernel image" although `is_available()` is True | CONFIRMED (`build_env_setup.py:73-75`: 12.6 aarch64 = {80, 90}) | `is_available True`, arch `['sm_80','sm_90']`; `x+1`, `torch.zeros(device=cuda)` and LSTM raise `CUDA error: no kernel image is available`. A matmul of CPU-made tensors copied to the device works (ADR wording holds); `randn(device=cuda)` fails | SURVIVES (dual-confirmed) | – |
| C5.3 | GPU is at most about 2.3x faster at hidden 32–64 | UNVERIFIABLE (measurement) | Single fit, 3 runs, median steady epoch: hidden 32 GPU/CPU8 = **0.86x**, hidden 64 = **1.26x**. 20-model pool, GPU 5 procs vs CPU 20×1: **2.30x** and **1.96x** | SURVIVES (B-only) | – |
| C5.4 | default 20 threads is 6.7–20x slower than 8 | CONFIRMED from source that the default = physical cores (`ParallelCommon.cpp:110-128`); the factor is UNVERIFIABLE | Median of 3: hidden 32 **13.0x**, hidden 64 **9.1x**. `get_num_threads()` = 20. Noisy (20-thread epochs 0.84–2.94 s; load average 2→11) | SURVIVES (B-only for the factor) | – |
| C5.5 | PyPI `torch==2.14.1` on Linux aarch64 is the CUDA build, about 3 GB | CONFIRMED as cu130 (`cuda-toolkit==13.0.3` in requires_dist); "about 3 GB of NVIDIA libs" REFUTED as worded: NVIDIA wheels 2.45 GB, total 3.14 GB | `envprobe/torch-pypi` (lock source `pypi.org/simple`) imports `2.14.1+cu130`, CUDA 13.0; all GPU ops OK. Lock wheel sum **3,163.2 MB** (nvidia+cuda* **2,456.5 MB**, torch 454.0, triton 226.5); site-packages 5.1 GB | ADR wording ("3,163 MB in total") SURVIVES; "~3 GB of NVIDIA libs" REFUTED | LOW |
| C5.6 | arch 8.0.0: EGARCH analytic forecast for h>1 raises | CONFIRMED (`volatility.py:2758-2759`) | `ValueError: Analytic forecasts not available for horizon > 1`, also with no `method=`; h=1 works | SURVIVES (dual-confirmed) | – |
| C5.7 | `random_state=` is ignored by `method="simulation"` | CONFIRMED (`base.py:1494-1495`; `volatility.py:766-780,2814`) | `RandomState(0)` twice gives 1.169838 and 1.162032. Bootstrap with `RandomState(0)` twice gives the same value | SURVIVES (dual-confirmed) | – |
| C5.8 | `rng=` seeds the simulation | CONFIRMED (signature `rng(size)`, `base.py:1490-1493`) | A fresh seeded-Generator `rng` twice gives 1.168110 = 1.168110; seed 1 gives 1.178083. Falsifier (one rng object reused) gives 1.168110 vs 1.150783 | SURVIVES (dual-confirmed) | – |
| C6.1 | pmdarima 2.1.1 works on this stack and recovers simple orders | CONFIRMED for the wheel and lower-only bounds; runtime UNVERIFIABLE from docs | `probe_pmd.py`, 3 seeds × n=1000: AR(1) (1,0,0) 3/3; MA(1) (0,0,1) 3/3; white noise (0,0,0) 3/3; **AR(2) gave (1,0,2)/(1,0,3)/(1,0,2), never (2,0,0)**. `ndiffs` gives 1 for a random walk and 0 for white noise. With `-W error::DeprecationWarning -W error::FutureWarning` the result is identical | SURVIVES; "recovers simple orders" overstates AR(2) (the ADR reports (1,0,2) honestly) | LOW |
| C6.2 | `update` and `predict(return_conf_int=True)` work | CONFIRMED (`arima.py:748-753,895`) | nobs 520 after update; conf_int shape (5,2), brackets the forecast, widths non-decreasing. `predict(n_periods=np.int64(5))` gives `TypeError: n_periods must be an int` | SURVIVES (dual-confirmed) | – |
| C7.1 | `npm run verify` is green | n/a | rc 0: eslint OK; **19 files / 219 tests passed**; coverage 98.31 / 95.52 / 97.22 / 99.23 (statements / branches / functions / lines); build OK | SURVIVES | – |
| C7.2 | coverage thresholds are 85/85/75 and enforced | n/a | Config reads `lines: 85, functions: 85, branches: 75`. No `v8 ignore`, `istanbul ignore`, `.skip`, `.only` or `todo` anywhere in `src`. Scratch with lines 100 gave rc 1 `Coverage for lines (99.23%) does not meet global threshold (100%)` | SURVIVES | – |
| C7.3 | three.js is only in the lazy chunk | n/a | `index-6CC54T4Z.js` (263 kB): 0 WebGLRenderer / BufferGeometry / `@react-three`. `RibbonCanvas-AomOMehq.js`: 6 / 27 / 1. Only `import("./RibbonCanvas-…js")`. No static three import in `src`. Falsifier (static import in main.tsx) put WebGLRenderer=6 in the main chunk | SURVIVES | – |
| C7.4 | every dependency is pinned exactly | n/a | 22/22 match `^\d+\.\d+\.\d+$`; lock root and installed versions agree; no overrides | SURVIVES | – |
| C7.5 | the 3D page does not crash without WebGL | n/a | 3 mutants (`useState(() => true)`, boundary `failed:false`, `detectWebGL` without catch): all killed (2, 2 and 1 failing tests). R3F 9.8.1 routes async renderer errors through `configure(...).catch(setError)` and then `if (error) throw error` (`react-three-fiber.esm.js:61,118`), which `RenderBoundary` catches | SURVIVES (jsdom; no real no-WebGL browser run) | – |
| C7.6 | `src/lib` is pure: no I/O, wall clock or unseeded randomness | n/a | `src/lib/storage.ts:33` reads `window.localStorage`. `src/lib/metrics.ts:56,76` default `now = Date.now()` and persist to localStorage. `storage.test.ts` touches the jsdom localStorage. These are not covered by the documented shell/chart/diagram exception. No `Math.random` | **REFUTED** (undocumented exception; the clock is injectable) | LOW |
| C8.1 | `.gitignore` excludes all vendor data and secrets | n/a | `git check-ignore -v --no-index`. **NOT ignored:** `VIX_History.csv`, `notebooks/VIX_History.csv`, `reports/VIX_History.csv`, `VIXCLS.csv`, `raw/cboe/VIX/abc.csv`, `*.feather`, `*.pkl`, `*.h5`, `*.xlsx`, `*.sqlite`, the yfinance cache, `notebooks/fred.json`, `app/src/data/term-structure.real.json`, `app/public/data/term_structure.json`, `credentials.json`, `kaggle.json`, `.netrc`, `.npmrc`, `.pypirc`, `secrets.toml`, `*.pem`, `id_rsa`, `fred_api_key.txt`. **Ignored:** `/data/`, `*.parquet`, `*.csv.gz`, `app/public/data/real/`, `.env*` | **REFUTED**: only `*.csv.gz` is covered, not `*.csv` | MED |
| C8.2 | the committed synthetic fixtures are not ignored | n/a | Neither `app/src/data/term-structure.synthetic.json` nor `app/public/data/synthetic/manifest.json` is ignored. Falsifier (unanchored `data/` via `core.excludesFile`) ignores both | SURVIVES | – |
| C8.3 | `git ls-files --others --exclude-standard` lists no real market data | n/a | 108 files; the only data-like files are the 2 synthetic fixtures. `npm run gen:fixture` in scratch reproduces `term-structure.synthetic.json` byte for byte (sha256 `916e3cc3…`). `tests/test_schemas.py:144` holds one level, `23.39` for 2024-08-05; its provenance could not be checked here (no local VIX data) | SURVIVES | – |

### C1.2 mutants (`scratchpad/reviewA/mutate.py`, output `b_mutants.jsonl`)

| M | mutation | result |
|---|---|---|
| M1 | `VOL_DATA_DIR` empty no longer counts as unset | killed (`test_empty_env_var_counts_as_unset`) |
| M2 | repo-marker `all` → `any` | killed |
| **M3** | drop `start = start.resolve()` in `find_repo_root` | **SURVIVED**: a relative `start` such as `Path("notebooks")` has no `.parents` above cwd; untested |
| M4 | project dirname changed | killed |
| M5 | `--untracked-files=no` | killed (`test_dirty_flag_follows_the_working_tree`) |
| **M6** | a failing `git status` no longer means dirty | **SURVIVED**: the documented "cannot show clean, so say dirty" branch is untested |
| M7 | no UTC conversion | killed |
| M8 | `ensure_ascii=False` | killed |
| M9 | `fullmatch` → `match` on the commit id | killed (SHA-256 repo test) |
| M10 | drop the envelope pre-check for an unknown dataset | killed |
| M11 | `dict(payload)` → `payload` (aliasing) | survived (benign) |
| M12 | sibling-prefix bug (`startswith`) | killed |
| M13 | `.resolve()` → `.absolute()` | killed |
| M14 | drop the bool/int distinction | killed |
| M15 | `check()` no longer reports unexpected files | killed |
| M16 | `check()` skips schema validation | killed |
| M17 | generate before the guard | killed (exit 4 instead of 3) |
| **M18** | `mkdir(exist_ok=False)` | **SURVIVED**: the documented regenerate command (`--mode synthetic` into the existing fixture folder) is never exercised by a test. Verified it works on the original code: wrote 832 bytes, then `--check` rc 0 |
| M19 | `run_id` not treated as provenance | killed |
| M20 | id format `cell-0` | killed |
| M21 | semantic instead of byte compare | killed (trailing-whitespace test) |
| M22 | notebook `check()` no longer reports unexpected files | killed |
| M23 | `--check` returns 0 on drift | killed |
| M24 | drop `nbformat.validate` | survived (equivalent for the current specs) |
| M25 | guard applied to the wrong mode | killed |
| M26 | glob `*.json` | survived (equivalent while all files end in `.schema.json`) |

### C3 attack log (`scratchpad/reviewA/b_c3.txt` plus the user-namespace runs)

```
A1 symlink dir -> repo/app                 REFUSED
A2 dotdot from outside                     REFUSED
A3 sibling prefix repo-other               ACCEPTED  (correct: physically outside)
A4 relative app/public/data/real (cwd=repo) REFUSED
A5 trailing slash / '/.'                   REFUSED
A6 case variant on case-SENSITIVE ext4     ACCEPTED  (correct: different directory)
A7 double slash                            REFUSED
A8 VOL_DATA_DIR / HOME pointing in (4 forms) REFUSED
bind mount (unshare -rm, mount --bind repo view): guard ACCEPTED, same inode as repo/app/src/data: True, file visible inside repo: True
casefold tmpfs (unshare -rm, chattr +F): repo .../cf/IVF; '.../cf/ivf/app/src/data' ACCEPTED, same-dir-as-repo: True
B1 out/manifest.json symlink -> repo file  rc=0  repo fixture overwritten=True; data_mode now=real
B2 out/manifest.json hardlink to repo file rc=0  repo fixture overwritten=True; data_mode now=real
B3 TOCTOU parent becomes symlink           rc=0  file landed in repo=True
B4 M0 CLI real outside repo                rc=4  wrote=False
```

Root cause: `tools/export_app_data.py:157-159` decides by path text, using `resolve()` and then `repo in resolved.parents`. Lines 271-273 then write with `Path.write_bytes`, which follows a symlink or hard link at the final component and re-walks the path after the check. A fix would compare device and inode identity (`os.path.samefile` on the nearest existing ancestor) and write through a temp file plus `os.replace` (or `O_NOFOLLOW`, with a check that `st_nlink == 1`).

## Found outside the claims

- **X-A (MED, latent): the D2 deploy guard does not exist.** ADR 0001 D2 says "`npm run verify` fails if any file in `dist/` has `"data_mode":"real"`" in the present tense, and the comment in `tests/test_contract.py::test_real_mode_spelling_matches_the_deploy_guard` says verify greps for it. In fact `verify` is lint + coverage + build only. Measured in scratch: with `public/data/real/manifest.json` containing `"data_mode":"real"`, `npm run verify` gave rc 0 and the file was in `dist/data/real/`. `app/DATA.md:43-44` admits this ("Guard still to build"). `app/.dockerignore` also does not exclude `public/data/real`, so a Docker build would ship it.
- **X-B (LOW):** `src/ivf/contract.py:18-19` says the files "are validated again on the TypeScript side". Nothing in `app/` imports Ajv or reads `schemas/app-data` (only `app/node_modules/ajv` 6.15.0, a transitive dependency of eslint). ADR 0001 says this arrives in M1.
- **X-C (LOW):** the app's own fixture `app/src/data/term-structure.synthetic.json` is not an envelope. It has `schema_version: 1` (an integer), and `contract.validate_document` rejects it with 7 errors. Two incompatible "term structure" contracts coexist in M0; DATA.md calls the shape provisional.
- **X-D (LOW, for M3):** in arch 8.0.0, `method="Simulation"` or `"SIMULATION"` is accepted and silently runs the bootstrap path: `RandomState(0)` twice gives identical values. Both channels agree (`volatility.py:756` validates the lower-cased string, `:766` dispatches on the raw one). Always pass lower-case.
- **X-E (LOW):** `test_cli_real_mode_in_repo_exits_3_and_writes_nothing` asserts that `app/public/data/real` does not exist. A developer who has that gitignored folder would get a spurious failure.
- **X-F (LOW):** `serialise` and `canonical_json_sha256` raise a bare `TypeError` for numpy and pandas scalars, which M1 exporters will meet. They are also not canonical for int keys (C4.1).

## Not validated

- macOS or Windows behaviour of the guard. The case-insensitivity bypass was reproduced on a Linux casefold tmpfs, not on APFS or NTFS.
- A real browser without WebGL. Only jsdom tests and reading the R3F source.
- The GPU 40-model and multi-process tables beyond one 20-model comparison (2 repeats). The determinism hashes in ADR 0002 §5. cu128 and cu132.
- Whether `23.39` in `tests/test_schemas.py:144` is a real Cboe close.
- Channel A's open Tegra caveat on the cubin compatibility rule for GB10. Runtime success (C5.1) makes it moot on this box.

## Single most important fix

Add vendor-data patterns to `.gitignore` before M1 starts downloading: at least `*.csv` with a `!tests/fixtures/**/*.csv` exception, plus `*.feather`, `*.pkl`, `*.h5`, `*.xlsx`, `*.sqlite`, the yfinance cache and the common secret files. Today a `VIX_History.csv` saved from a notebook, or at the repo root, would be swept into this public repo by `git add -A`.

## One item to delete

`tests/test_package.py::test_later_milestone_subpackages_say_when_they_arrive`. It asserts on docstring prose and guards no behaviour.
