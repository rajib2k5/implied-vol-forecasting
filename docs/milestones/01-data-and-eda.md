# Milestone 1: data and EDA

**Status:** milestone 1 of 6: the data pipeline (fetch, raw store, interim tables, verify), the
master calendar 𝒯 and its gap check, the frozen `data_end`, the features and targets, the EDA
module, a seeded synthetic generator with known truth, notebook 01, and the real-data
term-structure export. No model is fitted. No real-data number is published.

**Purpose.** To explain, line by line, how the data gets from Cboe and Yahoo to a checked,
calendar-aligned table, and what the first statistics say. Read in order: answer
[Predict before you look](#1-predict-before-you-look) on paper; read sections 2 to 7; compare
with [the answers](#8-answers-to-the-predict-first-questions); then answer
[Explain it back](#10-explain-it-back), whose answers are not here.

**Licensing (ADR 0001 D2, O3).** Every number below is measured on **synthetic** data, with the
command and the date. For real data the docs say "measured locally, not published (D2)". The
real results exist only under `$VOL_DATA_DIR` on the machine that ran them.

**Line numbers are a snapshot** of the files when this was written, each resolved against the
file text by a script, not typed by hand. Later milestones move lines.

**Companions.** [ADR 0001](../adr/0001-python-core-versioned-json-app.md) (D3 and D4.1 carry the
M1 amendments), [data model](../architecture/data-model.md), concept notes
[04](../concepts/04-volatility-clustering-and-the-acf.md),
[05](../concepts/05-persistence-half-life-and-unit-roots.md) and
[06](../concepts/06-variance-risk-premium-and-vol-of-vol.md),
[milestone 0](00-scaffold-and-design.md).

**Amended M1, review A.** An adversarial review of this milestone refuted six of its claims, showed
that a seventh ("63 of 63 mutations caught") guaranteed less than it seemed, and found two problems
nobody had claimed anything about. [What review A found](#what-review-a-found) names each one with
its fix; every statement changed because of it says "(amended M1, review A)". The ledger is
[docs/validation/M1-review-A-data-and-eda.md](../validation/M1-review-A-data-and-eda.md).

**Amended M1, review B.** A second adversarial review attacked the review-A fixes. It refuted four
of them outright: E6's new bootstrap was valid only for iid shocks, the network guard still had
open routes, the licensing scan was blind to most spellings of a date, and the repo guard did not
see a symlinked subfolder. It refuted parts of two more: a hand-moved `data_end` and leftovers in
the half-life interval, the raw store and the docs. [What review B found](#what-review-b-found)
names each one with its fix; every statement changed because of it says "(amended M1, review B)".
The ledger, with the fix round's notes, is
[docs/validation/M1-review-B-fixes.md](../validation/M1-review-B-fixes.md).

---

## What review A found

| Claim in the first version | Verdict | Fix |
|---|---|---|
| "No test touches the network" (`tests/conftest.py`) | **Refuted.** The guard was a fixture that only the data tests asked for. `connect_ex`, DNS (`getaddrinfo`) and `curl_cffi`, the HTTP client yfinance 1.7 uses, all reached a local server; deleting the guard's `connect` patch failed none of the 578 tests. | Installed for the whole session (`pytest_configure`), blocking `connect`, `connect_ex`, `sendto`, `create_connection`, `getaddrinfo` and curl_cffi; one test per route against a 127.0.0.1 listener. A subprocess is still unguarded: a known limit (§6, pitfall 14). **Amended M1, review B:** it was not the only route left. Review B sent a UDP datagram and opened a TCP connection through `sendmsg`, sent a DNS query through `gethostbyname`, connected through `socket.SocketType`, and saw yfinance swallow the refusal, so a test could pass silently ([What review B found](#what-review-b-found)). |
| E6 "uses the next AR(1) shock and shows no false effect under skewed shocks" | **Refuted.** The statistic is centred, but its Spearman p-value rejected on 30% of 300 stationary paths with iid exponential shocks and no level effect (5.3% with the true shocks), and the verdict said "yes" on 11%; this walkthrough's own seed 2 gave p = 2.1e-5. | The p-value is a residual bootstrap that refits the AR(1) on every path: size 5.3% (skewed) and 3.3% (normal), power 99-100% (§7.3). A test bounds the size. |
| Concept 06: E6's statistic "with iid shocks it is zero, whatever their shape", and its p-value fails because VIX is persistent | **Refuted** (review A's ledger, claim 16; added to this table by review B, which found it missing). The statistic is centred, but the test fails because of the estimated `phi`, not persistence: with the true shocks the Spearman size is 5.3%. | Concept 06 §2 rewritten: the fitted shock carries `(phi - phi_hat) ln V_t`, and the bootstrap refits `phi` on every path. |
| "No real-data number committed or about to be" | **Refuted.** Two real 1990 VIX rows, byte for byte, in `tests/test_data_fetch.py`; more real closes (VIX, VVIX, S&P 500) in `tests/test_data_interim.py`; the real raw files' total size in §3 of this walkthrough. The new scan then found one more, committed in M0: a real VIX level in `tests/test_schemas.py`, repeated in M0 review A's ledger. | Made-up values with hashes recomputed by `sha256sum`; the sentence deleted; the M0 test and ledger edited; `tools/licensing/scan_real_rows.py` added to the pre-commit gates (§7.5). The M0 commit still holds the old value (§11). |
| "`data_end` is chosen by an enforced rule" | **Refuted.** The rule ran only when freezing: a frozen `data_end` fetched on the same New York day, or edited by hand, passed `check` and `eda`. | `check` and `eda` re-validate it on every run (exit 1). |
| "Gaps are never bridged" | **Refuted for ADF and KPSS**, which ran on the series with its gaps dropped, joining their neighbours (ADR 0001 G2 forbids it). | Both run on the longest gap-free segment, and the report says which. |
| `tenor_days` "9, 93, 184, 365: no Cboe primary source" | **Refuted.** Cboe's term-index methodology (v7.0, pp. 5-6) states 9, 93, 184 and **366**. | VIX1Y is 366, cited in `configs/iv_indices.yaml`, ADR 0001 and the data model. |
| "63 of 63 mutations were caught" | **True as recorded, weak as evidence.** Of 11 mutants picked from the claims rather than the code, 9 passed all 578 tests: trailing or one-session-late RV in the spread, HC0 for HC1, a 90% interval called 95%, KPSS with a trend, the E6 shock without its intercept, EDA past `data_end`, `verify` ignoring symlinks, and the guard without `connect`. | A test with hand-worked values kills each (§7.4, second table). |
| (not claimed) | **Found:** the pipeline wrote wherever `VOL_DATA_DIR` pointed. `VOL_DATA_DIR=<repo>/localdata` put raw hashes and every EDA number into unignored files of the checkout. | One guard, `src/ivf/repo_guard.py`, shared with the exporter; every pipeline stage refuses a data root inside the repo with exit 3. |
| (not claimed) | **Found:** the EDA looked at the holdout. It printed and stored holdout statistics before any model existed (ADR 0001 4.6 allows one look, with frozen models), and its "dev" sample included burn-in. | The EDA sees only sessions and rows before `holdout_start`; its samples are `pre_holdout`, `burn_in` and `dev`. |
| `history(raise_errors=True)` | **Deprecated** in yfinance 1.7.0 (a `DeprecationWarning`). | `yfinance.config.debug.hide_exceptions = False` for the call, restored after. |

---

## What review B found

Review B attacked the review-A fixes. Every refutation below was checked again before it was
acted on (the measurement that checked it is named); none turned out wrong. One of its proposed
fixes was: a wild bootstrap, which fails on skewed shocks (§7.3).

| Claim after review A | Verdict | Fix |
|---|---|---|
| E6's bootstrap "has size about 5% and power of 99% or more; it is valid" | **Refuted: valid only for iid shocks.** Under GARCH(1,1) shocks with no level effect at all it rejected at 5% on 39.5% (alpha 0.10, beta 0.88) and 21% (0.05, 0.90) of 200 paths, and E6 said "yes" on 18% and 11.5% against a nominal 2.5%. VIX's shocks cluster, so the real verdict rested on a false assumption. Checked: 47.0% and 19.7% on 300 other paths (§7.3). | A circular **block** bootstrap of the fitted shocks, block length by Politis and White's rule on `abs(shock)`, still refitting the AR(1) on every path. Size at most 5.7% on ten nulls, among them skewed and asymmetric GARCH; power 95% and 99.7% where the shocks do not cluster, but **18%** where they do, so E6 now says "yes" or "not shown", never "no" (ADR 0001 E6, amended). A wild bootstrap and a HAC test were rejected (§7.3). |
| "During tests, only a subprocess or a direct `_socket` call reaches the network" | **Refuted.** A UDP `sendmsg` delivered its datagram; a TCP `sendmsg` with `MSG_FASTOPEN` connected; `gethostbyname`, `gethostbyname_ex` and `getnameinfo` were open (strace saw a query to port 53); `socket.SocketType` connected. And yfinance *swallows* `NetworkBlocked`, logging "possibly delisted", which the conftest docstring said no library did: a test that forgot its fake network would pass with no data. Checked: the bypass probe again, before and after. | `sendmsg` (AF_INET and AF_INET6) and `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo` are refused. Every refusal is now recorded and fails the test in which it happened, in setup, call or teardown, also when the code caught it; one during collection is a collection error; one anywhere else fails the session. Only `expect_blocked()` excuses a refusal, and it fails if nothing inside it was refused. Limits, stated in the docstring: subprocesses, `_socket` used directly (`SocketType`, `super(socket.socket, s).connect`), C code other than libcurl, AF_UNIX. |
| The licensing scan has "no blind spot for reformatted floats, JSON, Markdown, notebooks, yfinance or split lines" | **Refuted.** 9 of 16 planted shapes (30 synthetic stand-in rows each) were missed entirely: `date(1990, 5, 25)` objects (this repo's own test idiom), pandas epoch-millisecond keys, lists of closes without dates, pretty JSON with the value more than 3 lines from its date, `19900525`, `1990/05/25`, "May 25, 1990", 1-decimal values and values divided by 100. Checked: the same probe, before the change. | The scan reads every date spelling it names, matches a value rounded either way or divided by 100, looks for the nearest dated line up to 12 lines away, and adds rule 6: a run of 3 or more values equal to consecutive sessions of one raw series flags with no date at all (5 or more at one decimal). Every planted row of all 16 shapes now flags, and of 24 of 25 shapes added in the fix; the 25th, a 1-decimal value whose date is on another line, stays out by design (allowing it added a coincidence on the real store and one on the synthetic). The residual is listed in ADR 0001 D2. On the real tree it found one coincidence (a comment of made-up regime-edge inputs), which was reworded. |
| "No writer can write under a data root inside the repo" | **Refuted.** With `ROOT` outside the repo but `ROOT/raw` a symlink into the checkout, `fetch` wrote 8 snapshots into it and `build-interim` and `check` read them, all exit 0; with `ROOT/eda` symlinked, `eda` wrote `eda_report.json` into the checkout, untracked and not ignored. Only `ROOT` was checked. Checked: switching off the subfolder check (mutant G1) or the write-time check (G4) makes 5 and 10 of the new tests fail, writing into the stand-in checkout. | The root check also covers `raw/`, `interim/`, `processed/`, `eda/` and `app-data/`, before the first write. Every writer (fetch, the provenance log, interim, processed, the EDA report, synthetic, the real export, and the `data_end` freeze) now writes through `repo_guard.write_file`, which checks the folder at write time and writes through the folder it checked (§4.10). A symlinked `raw/` or `eda/` exits 3 with nothing written. |
| A frozen `data_end` "is re-validated on every run" | **Survived, with a gap.** A hand edit moving `data_end` back to an earlier settled session passed `check` and `eda` (exit 0), since such a session meets every rule; and the notebook printed VIOLATED and computed everything anyway. Checked: the same edit on the synthetic protocol, exit 0 and 0 without the new record (mutants R5, R6). | `check --freeze-data-end` also writes a freeze record, `configs/<protocol>.freeze.json`, with the value and the `protocol_hash`; `check`, `eda` and the notebook fail when the protocol no longer matches it, and the notebook stops. |
| Leftovers: the half-life interval, the raw store | **Refuted (minor).** White noise gave `half_life_ci = (nan, 0.245)`; `fetch --force` with a torn log stored the snapshot, then failed to log it, leaving an orphan; no folder `fsync` after the link. | An interval end `phi <= 0` gives a half-life of 0, by a stated convention (`eda.half_life_bound`). `fetch` checks the log before it downloads or stores; `verify` names an orphan and how to fix it; `write_file` flushes the folder after every new name. |
| The review-A table and the ledger "state the refutations plainly" | **Partly refuted.** The network row named only the subprocess limit; review A's concept-06 refutation was missing from the table. Review B also found an absolute local path with the user name in `tools/diagrams/VENDORED.md`, and `docs/architecture/data-model.md` listed `eda_stat` values (`mean`, `sd`) that nothing writes. | All four fixed here: the row, the added row, "a sibling checkout of the kit" plus the public URL, and the exact list `process.eda_stat_rows` writes. |
| Survived: no EDA row on or after `holdout_start`; V1h and V2b equivalent; real ADF/KPSS on the longest gap-free segment; the gates; all 89 `file:line` links | **Survived** | none |

---

## 1. Predict before you look

Write a number or yes/no for each. Answers: [section 8](#8-answers-to-the-predict-first-questions).

**P1.** Cboe's VIX history starts on 1990-01-02. On how many NYSE sessions since then would you
expect no VIX close at all: none, a handful, or hundreds?

**P2.** Does Cboe publish a VIX close on days the NYSE is closed?

**P3.** An AR(1) on ln VIX has `phi = 0.98`. What is the half-life in sessions? And at `phi = 0.99`?

**P4.** You simulate a *stationary* AR(1) with `phi = 0.98` for 9,000 sessions and run KPSS at
5%. On what share of seeds does KPSS wrongly reject stationarity?

**P5.** The series `1, 2, gap, 4, 5`. What is its lag-1 autocorrelation if you keep the gap, and
what if you call `dropna()` first?

**P6.** Shocks to ln VIX are iid but skewed (big jumps up, small steps down). Does the level
`V_t` correlate with the size of the next day's move `|ln V_{t+1} - ln V_t|`?

**P7.** Can the average implied-minus-realised gap be positive in vol points and negative in
variance units over the same days?

---

## 2. What we built and why

| Area | Files |
|---|---|
| Configuration | `configs/protocol.yaml` (real, `data_end` frozen), `configs/protocol.synthetic.yaml`, their freeze records `configs/protocol.freeze.json` and `configs/protocol.synthetic.freeze.json` (amended M1, review B), `configs/iv_indices.yaml`, `configs/instruments.yaml`; `src/ivf/protocol.py` reads them strictly |
| Raw store and fetch | `src/ivf/data/raw.py` (content-addressed snapshots, the JSONL log, `verify`), `src/ivf/data/fetch.py` (Cboe and yfinance, one network function) |
| Interim | `src/ivf/data/build_interim.py`: `iv_index_daily` and `price_daily` parquet |
| Calendar and checks | `src/ivf/data/calendar.py` (𝒯 from XNYS), `src/ivf/data/check.py` (data_check, gap policy, settled `data_end`) |
| Features and EDA | `src/ivf/features/build.py` (features, targets, regimes), `src/ivf/eda.py` (pure statistics), `src/ivf/data/process.py` (processed tables and the local report) |
| Synthetic data | `src/ivf/data/synthetic.py`: AR(1) ln VIX and GARCH(1,1) returns with known parameters, written in Cboe's and yfinance's shapes |
| Command line | `src/ivf/data/cli.py`, `src/ivf/data/__main__.py`: `uv run python -m ivf.data fetch / build-interim / verify / check / eda` |
| Tools | `tools/export_app_data.py --mode real` writes `term_structure.json` and `manifest.json` to `ROOT/app-data/real/`; `tools/build_notebooks.py` adds notebook 01 |
| Licensing guards (amended M1, review A) | `src/ivf/repo_guard.py`: the in-repo refusal, shared by the exporter and the pipeline, and since review B `write_file`, the one function that writes a file; `tools/licensing/scan_real_rows.py`: a local scan for real data copied into the repo |
| Tests | ten new files: `tests/test_data_fetch.py`, `test_data_interim.py`, `test_protocol_calendar.py`, `test_data_check.py`, `test_features.py`, `test_eda.py`, `test_synthetic_pipeline.py`, and after review A `test_network_guard.py`, `test_process.py`, `test_repo_guard.py`, after review B `test_scan_real_rows.py`; `tests/conftest.py` (the session-wide network guard); additions to `test_export.py`, `test_notebooks.py`, `test_gitignore.py`, `test_schemas.py` |
| Probes | `tools/probes/e6_size_power.py`, `tools/probes/e6_designs.py` (review B) and `tools/probes/kpss_holdout_size.py`: simulations behind §7.3 (no data read) |
| Docs | this walkthrough; concept notes 04 to 06; the "(amended M1)", "(amended M1, review A)" and "(amended M1, review B)" text in ADR 0001; the two review ledgers in `docs/validation/` |

The flow is one direction: `fetch` → `raw/` → `build-interim` → `interim/` → `check` and `eda` →
`processed/` and `eda/`. `tools/` reads `ivf`; `ivf` never reads `tools/` or `app/`.

---

## 3. Why this design

| Choice | Alternatives | Why | Cost |
|---|---|---|---|
| Raw files named by their SHA-256, plus an append-only JSONL log (ADR 0001 D3) | Overwrite one file per symbol; a database | A result can always be rebuilt from the exact bytes it used, and identical downloads are stored once. A torn JSONL line damages only itself. | Disk grows by one snapshot per changed download (the size of the real files is a real-data figure, not published: D2; amended M1, review A). |
| "Store only if absent" by `os.link` from a temporary file | Check `exists()`, then `rename` | `link` refuses an existing name in one atomic step, so two fetches cannot clobber each other. `rename` silently overwrites on POSIX. | Needs a filesystem with hard links (any Linux or macOS disk; not some network mounts). |
| One network function, `fetch.network_get`, replaced in tests; a network guard installed for the whole pytest session (amended M1, review A: it was a fixture only some tests used, and it left four routes open), which records every refusal and fails the test it happened in (amended M1, review B: yfinance swallowed the refusal, and `sendmsg` and the older DNS functions were open) | Mock `requests` and `yfinance` separately; a guard per test file; trust the exception to propagate | One seam to replace, and a guard that turns a forgotten mock into a failure instead of a download, also in collection and in fixtures, even when a library catches the exception. | The real network path is exercised only by running `fetch` by hand. A subprocess, `_socket` used directly, or C code with its own sockets, is not guarded. |
| The bytes stored for yfinance are a fixed CSV (`yfinance_csv_bytes`) | Pickle; `df.to_csv()` defaults; parquet | Same data, same bytes, same hash, on any machine. Pickle and parquet bytes depend on library versions; `to_csv` defaults print floats with `repr`, whose length varies. | Six decimals round yfinance's float64 by under 5e-7; a log return moves by under 2e-9. |
| Strict parsers that fail with a file and line | `pandas.read_csv` with type inference | A format change at Cboe (reordered columns, day-first dates, a `nan`) becomes an error, not silently wrong numbers. | A harmless new format also stops the pipeline until a person looks. |
| Gap policy G1-G5: gaps stay null; core gaps only deep in burn-in | Forward-fill; fill from FRED; drop the session from 𝒯 | Filling invents a zero move; dropping a session changes 𝒯 and every later index; FRED VIXCLS lacks most of the same sessions (checked locally). | Burn-in training rows contain nulls that M2's fitters must handle. |
| `data_end` = last session with both core closes *settled* (fetched on a later New York day), re-validated on every `check` and `eda` (amended M1, review A), and compared with a freeze record (amended M1, review B) | The last row present in both files; checking only when freezing; the data rules alone | A same-day fetch could hold an intraday price; the rule needs no exchange hours. Checked every run, a same-day refetch cannot slip through; the record catches a hand edit to another valid session, which the data rules cannot. | At most one session is lost when fetching after the close on the same day. One more committed file per protocol; a deliberate protocol change must edit the record too. |
| 𝒯 from `exchange_calendars` XNYS | The union of dates present in the data | ADR 0001 4.1: dates alone, so truncation and refresh never move a cutoff. Measured: the calendar reproduces every M0 figure (section 7). | Cboe's holiday rows are not on 𝒯 (G4). |
| E6 measured as the rank correlation of V_t with abs(next AR(1) shock), its p-value from a bootstrap that refits the AR(1) on every path (amended M1, review A) and resamples the shocks in circular blocks whose length follows Politis and White's rule on `abs(shock)` (amended M1, review B) | Trailing 21-session sd; same-day move; next raw move; scipy's Spearman p-value; fit on one half, test on the other; iid draws (review A); a wild bootstrap; a stationary bootstrap; a HAC test of `rho = 0` | With skewed iid shocks, the same-day and next-raw measures correlate with the level although the shock size never depends on it. The Spearman p-value treats the fitted shocks as true and rejected 30% of the time with no effect. iid draws lose volatility clustering and rejected on 47% of GARCH paths with no effect; a wild bootstrap keeps clustering but loses the skew; a test of `rho = 0` cannot have the right size once skewed shocks cluster, because `rho` is then not 0 (§7.3). | About 1.4 s per EDA run (999 paths of 8,700 sessions, measured); a seed makes the p-value reproducible. **Little power when the shocks cluster** (18% against a strong level effect), so E6 says "yes" or "not shown", never "no". A stationary bootstrap would do as well; it is not vectorised as simply. |
| The data root must be outside the repo, checked by device and inode in one shared function (amended M1, review A); its subfolders too, and every folder again when a file is written into it, through one write function that writes through the folder it checked (amended M1, review B) | `.gitignore` only; a check in the exporter only; a check of the root only (review A); refusing every symlink below `ROOT` | `.gitignore` covered `/data/` only; the pipeline wrote raw hashes and EDA numbers into the checkout when told to, and later through a symlinked `raw/` or `eda/`. Refusing all symlinks would forbid a legitimate data disk mounted by link. | One repo walk (0.1 s) per process, cached; then a few `stat` calls per file. Parquet is serialised in memory before it is written (the tables are small). |
| The EDA uses only sessions before `holdout_start` (amended M1, review A) | Full, dev and holdout samples | ADR 0001 4.6 allows one look at the holdout, with models frozen; descriptives computed now would inform M2's choices. | No holdout descriptives until the holdout is scored. |
| ADF and KPSS on the longest gap-free segment (amended M1, review A) | Drop the missing values (the first version) | Both lag and difference consecutive values; dropping a gap joins its neighbours, which G2 forbids. | On real data the sessions before the last burn-in gap are left out of these two tests. |
| HC1 standard errors for `phi` | Plain OLS; HAC | Shocks to ln VIX are larger in stress; HC1 stays honest then (11-14 times OLS on a heteroskedastic test series). | Ignores residual autocorrelation; HAC would cover it but needs a bandwidth choice. |
| Three extra ratio columns in `feature` (`vix9d_vix`, `vix3m_vix6m`, `vix6m_vix1y`) | Only the data model's `vix_vix3m` | The brief asks for the ratios across all five tenors; shorter over longer keeps "above 1 = inverted" for every segment. | The data model and ERD must add them (listed for the diagram owner). |
| Real export computes the panel in `ivf.features.build.term_structure_panel`; the tool only rounds and reshapes | Compute the ratio and flag in the tool | ADR 0001 D1: the exporter computes nothing new. | One more function in `ivf`. |
| `--check` ignores the manifest's `files[].sha256` and `bytes`, and verifies them against the files on disk instead | Compare them like any field | They hash bytes that contain `generated_at` and `source_commit`, so they change on every export. | One extra check function. |
| Synthetic VIX and S&P 500 drawn independently | ADR 0001's two-factor fixture with a priced premium | Each EDA statistic then has a closed form to test against. | The synthetic "premium" is a level difference, not a state-dependent premium. |

---

## 4. Code walkthrough by `file:line`

Each entry: the line, what it does, why. **Mutation** notes come from a scratch copy of the repo
(section 7.4): break one line, run the milestone's test files, restore.

### 4.1 Configuration: `configs/` and `src/ivf/protocol.py`

- [`configs/protocol.yaml:18`](../../configs/protocol.yaml#L18): `data_end`, frozen by
  `check --freeze-data-end` on 2026-10-29. The comment above it says how; the dates are quoted. Its
  freeze record, `configs/protocol.freeze.json`, was written on 2026-10-29 by the same command in
  the review-B round, because the value was frozen before records existed (amended M1, review B).
- [`src/ivf/protocol.py:125`](../../src/ivf/protocol.py#L125): refuses an unquoted YAML date with "write the date in
  quotes". YAML turns `1990-01-02` into a `date` object, which has no JSON spelling, so
  `protocol_hash` would be undefined. **Mutation:** accepting date objects fails one test.
- [`src/ivf/protocol.py:144`](../../src/ivf/protocol.py#L144): exact keys (a typo is an error, not a silent
  default) and the order `train_start <= dev_start < holdout_start <= data_end`. **Mutation:**
  accepting unknown keys fails one test.
- [`src/ivf/protocol.py:73`](../../src/ivf/protocol.py#L73): the SHA-256 of the canonical JSON of the parsed
  file (ADR 0001 D3). `tests/test_protocol_calendar.py` compares it with `sha256sum` run on a
  hand-written JSON string.
- [`src/ivf/protocol.py:286`](../../src/ivf/protocol.py#L286): rewrites the one `data_end: null` line, keeps the
  comments, reads the file back, and refuses to move a frozen value. **Mutation:** allowing the
  move fails `test_a_frozen_data_end_is_never_moved`. Both files are now written by
  `repo_guard.write_file` (atomic rename, flushed; amended M1, review B).
- [`src/ivf/protocol.py:210`](../../src/ivf/protocol.py#L210) and
  [`src/ivf/protocol.py:244`](../../src/ivf/protocol.py#L244) (amended M1, review B): the freeze also
  writes `configs/<protocol>.freeze.json`, the value and the `protocol_hash` it froze, and `check`,
  `eda` and the notebook fail when the protocol no longer matches it: a moved `data_end`, any other
  edit after the freeze, a `data_end` typed in with no record, or a frozen value set back to null
  (which [`src/ivf/protocol.py:316`](../../src/ivf/protocol.py#L316) also refuses to freeze again). Review B
  moved `data_end` back by hand to an earlier settled session and both commands returned 0. It is
  a tripwire, not a lock: editing both files on purpose is a new protocol, which rule R1 allows.
  **Mutations:** R1 to R6 in §7.4.

### 4.2 The raw store: `src/ivf/data/raw.py`

- [`src/ivf/data/raw.py:46`](../../src/ivf/data/raw.py#L46): a symbol becomes a folder name, so `../VIX`
  is refused. **Mutation:** removing the check fails five tests.
- [`src/ivf/data/raw.py:100`](../../src/ivf/data/raw.py#L100): hash the bytes, then
  [`src/ivf/data/raw.py:120`](../../src/ivf/data/raw.py#L120): `write_file(..., how="new")` writes a
  temporary file in the folder, flushes it, *links* it to `{sha256}.csv` and flushes the folder
  (amended M1, review B: the folder flush was missing, and the write now goes through the repo
  guard's write-time check). **Mutation:** `os.replace` instead of `os.link` overwrites a planted
  file and fails two tests (R1 before review B, G13 after).
- [`src/ivf/data/raw.py:139`](../../src/ivf/data/raw.py#L139): refuses to append after a torn last line (no
  final newline), because the new line would be glued to it
  ([`src/ivf/data/raw.py:124`](../../src/ivf/data/raw.py#L124)); `fetch_one` now calls the same check
  before it downloads or stores anything, so a torn log no longer leaves an orphan snapshot
  (amended M1, review B). **Mutations:** removing the refusal fails two tests (F4); dropping the
  early call fails one (F1).
- [`src/ivf/data/raw.py:151`](../../src/ivf/data/raw.py#L151): every line must have exactly the twelve fields
  of `raw_fetch` and a consistent `raw_path`; an unreadable line is an error, never skipped.
  **Mutation:** skipping it fails one test.
- [`src/ivf/data/raw.py:175`](../../src/ivf/data/raw.py#L175): the latest `fetched_at`, ties to the later line.
  **Mutation:** comparing time only (first line wins a tie) fails one test.
- [`src/ivf/data/raw.py:199`](../../src/ivf/data/raw.py#L199): re-hashes every `{sha256}.csv` and cross-checks the log
  both ways. A snapshot no line refers to is reported as an orphan, with the likely cause (a crash
  between storing and logging) and the fix (`fetch --force` logs it again, or delete it; amended
  M1, review B). **Mutations:** skipping the re-hash fails one test; skipping the orphan check
  fails two (F2).

### 4.3 Fetch: `src/ivf/data/fetch.py`

- [`src/ivf/data/fetch.py:79`](../../src/ivf/data/fetch.py#L79): the only network access in `ivf`. Cboe's
  `cdn.cboe.com` URL answers 307 to `cdn-api.cboe.com`; `requests` follows it and `final_url`
  records the target. yfinance is called with `auto_adjust=False, actions=False`, and raises on a
  failure because `yfinance.config.debug.hide_exceptions` is off for the call
  ([`src/ivf/data/fetch.py:109`](../../src/ivf/data/fetch.py#L109)) and restored after; the first
  version passed `raise_errors=True`, deprecated in yfinance 1.7.0 (amended M1, review A).
  **Mutations:** leaving the switch on, passing `raise_errors`, or not restoring it each fail
  `test_yfinance_is_asked_to_raise_on_failure_without_the_deprecated_keyword`.
- [`src/ivf/data/fetch.py:127`](../../src/ivf/data/fetch.py#L127): the exact bytes stored for a DataFrame
  (ADR 0001 D3, amended M1). **Mutations:** dropping the sort, or writing four decimals, each fail
  `test_yfinance_frame_becomes_exactly_these_bytes`.
- [`src/ivf/data/fetch.py:182`](../../src/ivf/data/fetch.py#L182): the log check
  ([`src/ivf/data/fetch.py:201`](../../src/ivf/data/fetch.py#L201); amended M1, review B), the freshness check
  ([`src/ivf/data/fetch.py:206`](../../src/ivf/data/fetch.py#L206)), then the download, the status check, the store
  and one log line even for unchanged bytes. **Mutations:** no freshness skip fails two tests; no
  status check fails two; logging only changed content fails two.
- [`src/ivf/data/fetch.py:247`](../../src/ivf/data/fetch.py#L247): `1d`, `12h`, `30m`, `0`; anything else is an
  error.

### 4.4 Interim: `src/ivf/data/build_interim.py`

- [`src/ivf/data/build_interim.py:163`](../../src/ivf/data/build_interim.py#L163): three accepted layouts. The header
  must be exactly `DATE,OPEN,HIGH,LOW,CLOSE`, or `DATE,CLOSE`, or `DATE,<SYMBOL>` (VVIX's file is
  `DATE,VVIX`). **Mutation:** accepting any five-column header as OHLC survived the first test
  run; the test now includes `DATE,CLOSE,HIGH,LOW,OPEN` (same columns, other order) and fails.
- [`src/ivf/data/build_interim.py:110`](../../src/ivf/data/build_interim.py#L110): one date format per file, M/D/YYYY or
  ISO; a mixed file or a day-first date is an error. **Mutation:** accepting mixed formats fails
  one test.
- [`src/ivf/data/build_interim.py:132`](../../src/ivf/data/build_interim.py#L132): `float("nan")` parses in Python, so finiteness
  is checked separately. **Mutation:** removing it fails two tests.
- [`src/ivf/data/build_interim.py:147`](../../src/ivf/data/build_interim.py#L147): an identical repeated row is kept once and
  counted; a date with two different values is an error. **Mutation:** keeping the first fails one
  test.
- [`src/ivf/data/build_interim.py:251`](../../src/ivf/data/build_interim.py#L251): the latest fetch per
  `(source, symbol)` or the `--snapshot` pin. **Mutation:** ignoring pins fails one test.
- [`src/ivf/data/build_interim.py:292`](../../src/ivf/data/build_interim.py#L292): re-hashes the snapshot before parsing
  ([`src/ivf/data/build_interim.py:319`](../../src/ivf/data/build_interim.py#L319))
  and writes both tables with explicit Arrow schemas (`date32`, nullable OHLC), serialised in
  memory and written by `repo_guard.write_file`
  ([`src/ivf/data/build_interim.py:265`](../../src/ivf/data/build_interim.py#L265); amended M1, review B). **Mutations:**
  skipping the hash check fails one test; merging two sources silently fails one; writing around
  `write_file` fails one (G8).

### 4.5 The master calendar: `src/ivf/data/calendar.py`

- [`src/ivf/data/calendar.py:28`](../../src/ivf/data/calendar.py#L28): XNYS sessions from `exchange_calendars`. It
  filters a pandas 3.1 deprecation warning that the library triggers.
- [`src/ivf/data/calendar.py:54`](../../src/ivf/data/calendar.py#L54): `c_k = idx(holdout_start) + 63k`, the earliest
  with `c_k >= idx(dev_start)`: 2000-03-15. **Mutation:** rounding k up fails two tests.
- [`src/ivf/data/calendar.py:69`](../../src/ivf/data/calendar.py#L69): 𝒯 = sessions to `data_end` plus `h_max`
  more; `period` is `burn_in` before the first cutoff, `dev` to `holdout_start`, `holdout` after;
  `fold_id` stays null until M2 numbers folds. **Mutations:** `h_max - 1` extra sessions fails one
  test; labelling the cutoff itself `burn_in` fails one.

### 4.6 The data check: `src/ivf/data/check.py`

- [`src/ivf/data/check.py:71`](../../src/ivf/data/check.py#L71): gap policy G3's margin, the `har_m` window.
  **Mutation:** 21 fails the boundary test, which drops the session 22 and then 21 before the
  cutoff.
- [`src/ivf/data/check.py:78`](../../src/ivf/data/check.py#L78): the New York date of a fetch. **Mutation:** using
  the UTC date fails two tests (a fetch at 03:59 UTC is still the previous day in New York).
- [`src/ivf/data/check.py:105`](../../src/ivf/data/check.py#L105): per symbol, missing sessions, rows on non-sessions
  and unsettled rows; the `train_start` check
  ([`src/ivf/data/check.py:182`](../../src/ivf/data/check.py#L182)); core gaps; G3; the
  `data_end` candidate. **Mutations:** ignoring `^GSPC` in core gaps fails three tests; dropping
  the `train_start` check fails one; ignoring settlement fails three.
- [`src/ivf/data/check.py:248`](../../src/ivf/data/check.py#L248): the gap and coverage summary, terminal only.
  Its stream defaults to `sys.stdout` *at call time*
  ([`src/ivf/data/check.py:255`](../../src/ivf/data/check.py#L255)): pitfall 4.
- [`src/ivf/data/check.py:223`](../../src/ivf/data/check.py#L223), called at
  [`src/ivf/data/check.py:200`](../../src/ivf/data/check.py#L200): a frozen `data_end` must be a session
  with both core closes, each from a fetch on a later New York day, in the data being read now
  (amended M1, review A). **Mutations:** skipping the re-check fails five tests; `>` for `>=`
  (a same-day fetch counted as settled) fails three; `check` treating a frozen value as unfrozen
  fails one; `eda` doing so fails one.
- [`src/ivf/data/check.py:279`](../../src/ivf/data/check.py#L279): exit 1 on a policy violation; `--freeze-data-end`
  writes the candidate once and never moves a frozen value. A frozen value with no record yet is
  recorded while the data still support it ([`src/ivf/data/check.py:295`](../../src/ivf/data/check.py#L295)),
  and the record's violations join the gap policy's, so a mismatch prints `gap policy: VIOLATED`
  (amended M1, review B).
- [`src/ivf/data/check.py:212`](../../src/ivf/data/check.py#L212) (amended M1, review B): `data_check` with the frozen
  `data_end` re-validated, plus the freeze record. `eda` and the notebook run exactly this before
  computing anything. **Mutations:** `eda` checking the data only fails one test (R5); `check`
  ignoring the record fails one (R6).

### 4.7 Features and targets: `src/ivf/features/build.py`

- [`src/ivf/features/build.py:45`](../../src/ivf/features/build.py#L45): returns only between consecutive sessions of
  𝒯; a gap makes two returns missing. **Mutation:** forward-filling closes first (bridging the
  gap) fails one test.
- [`src/ivf/features/build.py:53`](../../src/ivf/features/build.py#L53): `rolling(window, min_periods=window)`: trailing
  and all-or-nothing. **Mutations:** `min_periods=1` fails four tests; `center=True` (look-ahead)
  fails ten, among them the truncation test.
- [`src/ivf/features/build.py:67`](../../src/ivf/features/build.py#L67): ADR 0001 4.2 exactly: `100*sqrt((252/21) *
  sum r^2)`, no mean removed. **Mutation:** `252/20` fails six tests.
- [`src/ivf/features/build.py:76`](../../src/ivf/features/build.py#L76): the only forward shift, for targets and ex-post
  EDA. **Mutation:** an off-by-one fails two tests.
- [`src/ivf/features/build.py:84`](../../src/ivf/features/build.py#L84): the data model's columns plus three ratios.
  **Mutation:** `HAR_MONTH = 21` fails two tests.
- [`src/ivf/features/build.py:112`](../../src/ivf/features/build.py#L112): `vix_level` and `rv21`, with
  `v = (y/100)^2`, no row where there is no value.
- [`src/ivf/features/build.py:142`](../../src/ivf/features/build.py#L142): fixed edges (rule L5); backwardation
  only when VIX > VIX3M. **Mutation:** `v <= 15` as low fails one test.

### 4.8 The EDA: `src/ivf/eda.py`

- [`src/ivf/eda.py:39`](../../src/ivf/eda.py#L39): pairwise-complete on the calendar. **Mutations:** dropping NaNs
  first fails one test; dividing each lag by its own count fails five.
- [`src/ivf/eda.py:71`](../../src/ivf/eda.py#L71): **Mutation:** dropping the `n - k` weights fails two
  tests.
- [`src/ivf/eda.py:144`](../../src/ivf/eda.py#L144) with [`src/ivf/eda.py:90`](../../src/ivf/eda.py#L90): OLS on
  calendar pairs, HC1 errors, the interval mapped through the half-life. **Mutations:** bridging
  gaps fails one test; breaking the HC1 branch fails eight (every caller of the default, `vol_of_vol` included); inverting the half-life fails three.
  HC0 in place of HC1, and a 90% interval reported as 95%, passed every test of the first version
  (review A); `test_hc1_standard_error_and_95_percent_interval_by_hand` (4 pairs, worked in
  elevenths) now fails on each. An interval end at or below 0 maps to a half-life of 0
  ([`src/ivf/eda.py:122`](../../src/ivf/eda.py#L122); amended M1, review B): the half-lives such an interval
  allows reach down to 0 as `phi` falls to 0, so 0 is the honest lower end, where the first version
  gave nan (white noise: `(nan, 0.245)`). The point estimate keeps nan for `phi <= 0`, which is no
  decay. **Mutations:** H1 and H2 in §7.4.
- [`src/ivf/eda.py:224`](../../src/ivf/eda.py#L224): ADF and KPSS, the verdict needs both, run on the
  longest gap-free segment ([`src/ivf/eda.py:207`](../../src/ivf/eda.py#L207); amended M1, review A).
  **Mutations:** a verdict from ADF alone survived the first run; the test with a mean shift (both
  tests reject) now kills it. Dropping the gaps instead (the first version) fails one test; KPSS
  with `regression="ct"` passed every test of the first version and now fails one (a trend that
  "c" rejects and "ct" accepts).
- [`src/ivf/eda.py:279`](../../src/ivf/eda.py#L279): the ex-post gap in vol points and variance.
  **Mutation:** an unsquared variance gap fails three tests.
- [`src/ivf/eda.py:539`](../../src/ivf/eda.py#L539): the annualised rolling sd, and E6 through
  [`src/ivf/eda.py:442`](../../src/ivf/eda.py#L442): `rho` of the level with the next AR(1) shock
  ([`src/ivf/eda.py:369`](../../src/ivf/eda.py#L369)) and its bootstrap p-value
  ([`src/ivf/eda.py:388`](../../src/ivf/eda.py#L388)). The first version used the next raw move; a
  mutation to the same-day move survived, and the skewed-shock test that now kills it also showed
  the raw-move measure was itself misleading (pitfall 6). Its Spearman p-value was refuted by
  review A (pitfall 13). **Mutations:** the shock without its intercept fails three tests; the
  bootstrap without the refit, or without the intercept, or scipy's p-value back, or a one-tailed
  p each fail the size test or a hand-worked one.
- [`src/ivf/eda.py:402`](../../src/ivf/eda.py#L402) and [`src/ivf/eda.py:428`](../../src/ivf/eda.py#L428) (amended M1,
  review B): the bootstrap resamples the fitted shocks in circular blocks of `L` consecutive
  sessions, `L` from Politis and White's rule (with the 2009 correction) applied to `abs(shock)`,
  rounded, at least 1. On the shocks themselves the rule sees no dependence even under strong
  clustering; on their size it does. iid shocks get `L` = 1, review A's bootstrap exactly. Review A's
  iid draws rejected on 47% of GARCH paths with no level effect (§7.3). The report records `L` as
  `e6_block`. **Mutations:** B1 to B7 in §7.4; B1, the rule replaced by `L` = 1 (review A's design),
  fails `test_e6_size_under_garch_shocks_with_no_level_effect_stays_near_5_percent`, which counts
  1 rejection in 100 for the shipped test and 44 for `block=1` on the same paths.
- [`src/ivf/eda.py:610`](../../src/ivf/eda.py#L610): contango is VIX <= VIX3M, the regime rule's
  complement. **Mutation:** a strict `<` fails one test.

### 4.9 Synthetic data: `src/ivf/data/synthetic.py`

- [`src/ivf/data/synthetic.py:69`](../../src/ivf/data/synthetic.py#L69): round, hand-set parameters with their
  arithmetic in comments; not fitted to real data.
- [`src/ivf/data/synthetic.py:142`](../../src/ivf/data/synthetic.py#L142) and
  [`src/ivf/data/synthetic.py:158`](../../src/ivf/data/synthetic.py#L158): the closed forms the tests compare
  with. **Mutations:** dropping the `alpha*beta` term fails one test; `offset/slope` fails two.
- [`src/ivf/data/synthetic.py:243`](../../src/ivf/data/synthetic.py#L243): the same `store_snapshot` and
  `append_provenance` as `fetch`, so the synthetic files run the real path.

### 4.10 Processing and the command line

- [`src/ivf/data/process.py:54`](../../src/ivf/data/process.py#L54): ADR 0001 D3's `run_id`, with the raw hashes
  sorted. **Mutation:** keeping input order fails one test.
- [`src/ivf/data/process.py:256`](../../src/ivf/data/process.py#L256): re-runs `data_check` with the
  frozen `data_end` re-validated and the freeze record compared (`check.frozen_check`; amended M1,
  review B) and refuses on a violation, then writes the processed tables, each with `run_id`
  first, and the local `eda/<run_id>/eda_report.json`, all through `repo_guard.write_file`. It
  prints only the qualitative E-statements and the half-life at 2 s.f., the forms D2 would allow;
  E6 prints "yes" or "not shown" ([`src/ivf/data/process.py:349`](../../src/ivf/data/process.py#L349); amended M1,
  review B), with the reason for "not shown", and "yes" only when the fitted AR(1) `phi` is at most
  `eda.E6_PHI_MAX` = 0.985 (`eda.e6_verdict`; amended M1, final check, §7.3).
- [`src/ivf/data/process.py:87`](../../src/ivf/data/process.py#L87): what the EDA may see, sessions and
  rows before `holdout_start` only; [`src/ivf/data/process.py:114`](../../src/ivf/data/process.py#L114)
  refuses a holdout row (amended M1, review A). **Mutations:** keeping holdout sessions fails nine
  tests; dropping the refusal fails one.
- [`src/ivf/data/process.py:102`](../../src/ivf/data/process.py#L102): the spread's forward window,
  t+1..t+21. **Mutations:** trailing RV and a window one session late passed every test of the
  first version (same mean on stationary data); each now fails two tests that check the window
  origin by origin. Sessions past `data_end` in the tables fail one test.
- [`src/ivf/data/cli.py:126`](../../src/ivf/data/cli.py#L126): `--synthetic` switches the data root and the protocol;
  exit codes 0, 1 (a check failed), 2 (usage), 3 (input missing or malformed, or the licensing
  guard refused the data root), 5 (a download failed). Every command first refuses a data root
  inside the repo ([`src/ivf/data/cli.py:143`](../../src/ivf/data/cli.py#L143); amended M1, review A),
  and so do the library entry points `fetch_one`, `write_synthetic_raw`, `build_interim`,
  `run_check` and `run_eda`. **Mutations:** removing the command-line call fails 14 tests; removing
  any one library call fails one.
- [`src/ivf/repo_guard.py:155`](../../src/ivf/repo_guard.py#L155) and
  [`src/ivf/repo_guard.py:191`](../../src/ivf/repo_guard.py#L191) (amended M1, review B): the root check now also
  resolves `raw/`, `interim/`, `processed/`, `eda/` and `app-data/` under the root, so a symlinked
  `eda/` stops `eda` before `processed/` is touched. Then every file is written by `write_file`:
  it resolves and checks the folder it writes into ([`src/ivf/repo_guard.py:171`](../../src/ivf/repo_guard.py#L171))
  before creating anything, creates it, opens it, checks that the folder it holds open has the
  identity and the resolved path it checked, and writes through that handle (`dir_fd`), so a
  folder swapped for a link after the check is not written through. A refusal at write time exits
  3 too ([`src/ivf/data/cli.py:159`](../../src/ivf/data/cli.py#L159)), and the
  fetch loop passes it on instead of counting it as a failed download (exit 5). The repo's folder
  identities are read once per process (0.1 s). **Mutations:** G1 to G13 in §7.4.

### 4.11 The real export: `tools/export_app_data.py`

- [`tools/export_app_data.py:199`](../../tools/export_app_data.py#L199): dates from the first session on
  which all five tenors exist (VIX9D starts last) to `data_end`; levels at 2 decimals, the ratio
  at 4; the flag from the unrounded ratio, computed in `ivf`. **Mutations:** starting at the first
  tenor fails one test; 3 decimals fails one.
- [`tools/export_app_data.py:184`](../../tools/export_app_data.py#L184): invariants the schema cannot express.
  **Mutation:** dropping the tenor-order check fails one test.
- [`tools/export_app_data.py:233`](../../tools/export_app_data.py#L233): needs `ROOT/interim/` and a frozen protocol,
  else exit 4 with the step to run. The licensing guard of M0 runs first and behaves as before: an
  in-repo folder is still refused with exit 3 (`test_real_export_cli_writes_outside_the_repo_and_never_inside`).
  Its code moved to [`src/ivf/repo_guard.py:104`](../../src/ivf/repo_guard.py#L104) and
  [`src/ivf/repo_guard.py:80`](../../src/ivf/repo_guard.py#L80), shared with the pipeline's
  [`src/ivf/repo_guard.py:155`](../../src/ivf/repo_guard.py#L155) (amended M1, review A); the tool keeps
  its old names as aliases, so its tests are unchanged. **Mutations:** no `.git` check fails seven
  tests; no device-and-inode check fails 21. `write_files` now writes each file through
  `repo_guard.write_file` ([`tools/export_app_data.py:368`](../../tools/export_app_data.py#L368)),
  checking the folder at write time in real mode (amended M1, review B); its tests patch
  `repo_guard.os` and `repo_guard.secrets` where they patched the tool's.
- [`tools/export_app_data.py:63`](../../tools/export_app_data.py#L63) and
  [`tools/export_app_data.py:390`](../../tools/export_app_data.py#L390): pitfall 9. **Mutations:** comparing
  the listed hash fails one test; dropping the consistency check fails one.

### 4.12 Notebook 01 and the tests

- [`tools/build_notebooks.py:155`](../../tools/build_notebooks.py#L155): the spec. `load()` uses real data
  when `ROOT` has it, else the synthetic market, and says which; `IVF_NOTEBOOK_DATA=synthetic`
  forces synthetic. The check cell raises when `check` reports VIOLATED
  ([`tools/build_notebooks.py:250`](../../tools/build_notebooks.py#L250); amended M1, review B:
  the first version printed VIOLATED and computed everything below it). **Mutation:** printing
  instead of raising fails `test_m1_notebook_stops_when_check_reports_violated` (N1).
- [`tests/conftest.py:115`](../../tests/conftest.py#L115): the network guard, installed for the whole
  session at [`tests/conftest.py:278`](../../tests/conftest.py#L278) (amended M1, review A: it was a
  fixture only the data tests requested). `tests/test_network_guard.py` tries each route against a
  listener on 127.0.0.1; removing any one patch fails its test (§7.4), except
  `create_connection`'s, which is also blocked through `getaddrinfo` and `connect`. Amended M1,
  review B: `sendmsg` and the four older DNS functions are refused too, every refusal is recorded
  ([`tests/conftest.py:195`](../../tests/conftest.py#L195)) and charged to the test, setup or teardown it happened
  in ([`tests/conftest.py:305`](../../tests/conftest.py#L305)), also when the code under test caught
  it, as yfinance does; a test of the guard marks its refusals as expected with
  [`tests/conftest.py:228`](../../tests/conftest.py#L228), which fails if nothing was refused. The inner tests
  that show a swallowed refusal failing run in a separate pytest session (`pytester`). A new test
  temporarily lifts the `getaddrinfo` and `connect` patches, so deleting `create_connection`'s own
  patch now fails a test as well.
- [`tests/test_scan_real_rows.py:132`](../../tests/test_scan_real_rows.py#L132)
  (amended M1, review B): the scan, run on a synthetic raw store and a throw-away git repository in
  `tmp_path`, flags a planted value in each date spelling and each number spelling, flags runs
  without dates, leaves the same numbers on other dates alone, and never prints a value.
- [`tests/test_features.py:103`](../../tests/test_features.py#L103):
  ADR 0001 4.7's truncation test, applied to features.
- [`tests/test_protocol_calendar.py:182`](../../tests/test_protocol_calendar.py#L182):
  the calendar reproduces M0's independent measurements.

---

## 5. Data flow

```
Cboe CSVs (307 -> cdn-api.cboe.com)      yfinance ^GSPC DataFrame
        |                                         | fetch.yfinance_csv_bytes (defined bytes)
        +------------- fetch.network_get ---------+
                             |
   raw.store_snapshot: ROOT/raw/{source}/{symbol}/{sha256}.csv   (only if absent: os.link)
   raw.append_provenance: ROOT/raw/provenance.jsonl              (one line per fetch)
                             |
   build_interim: latest snapshot per (source, symbol) or --snapshot; re-hash; strict parse
                             |
   ROOT/interim/iv_index_daily.parquet, price_daily.parquet, build_report.json
                             |
   check.data_check on 𝒯 (calendar.xnys_sessions): coverage, gap policy G1-G5, settled data_end
        |  --freeze-data-end -> configs/protocol.yaml (committed: a date, not a measurement)
        v
   process.run_eda: calendar.master_calendar, features.build to data_end;
        |  the EDA (eda.*) on sessions before holdout_start only (review A)
        |-> ROOT/processed/runs.parquet, processed/run_id=<id>/{run_input,calendar,data_check,
        |   feature,target_value,regime_label,eda_stat}.parquet
        |-> ROOT/eda/<run_id>/eda_report.json                (local only, D2)
        v
   tools/export_app_data.py --mode real -> ROOT/app-data/real/{manifest,term_structure}.json
   (refuses any folder inside the repo: exit 3)
```

The synthetic run is the same picture under `ROOT/synthetic/`, with `synthetic.write_synthetic_raw`
in place of the two downloads. Every stage starts with `repo_guard.require_data_root_outside_repo`:
a `ROOT` inside the repo stops it with exit 3 before anything is read or written (amended M1,
review A).

---

## 6. Pitfalls and their guards

| # | Pitfall | How it bites | Guard |
|---|---|---|---|
| 1 | yfinance can return a bar for a session still in progress | An intraday price is frozen as a close in `data_end` | The settled rule, [`src/ivf/data/check.py:78`](../../src/ivf/data/check.py#L78) (not observed in M1: the fetch ran before the open) |
| 2 | Cboe publishes VIX on some NYSE holidays | A holiday row would add a session and shift every index | 𝒯 from dates alone; holiday rows counted, never read (G4), [`src/ivf/features/build.py:35`](../../src/ivf/features/build.py#L35) |
| 3 | `dropna()` before lagging | A two-session move counted as one; ACF and phi biased | Pairs on the calendar, [`src/ivf/eda.py:90`](../../src/ivf/eda.py#L90) |
| 4 | A default argument `out=sys.stdout` binds the stream at import | The summary went to a stale stream; the notebook test passed alone and failed in the full suite (found in M1) | Resolve at call time, [`src/ivf/data/check.py:255`](../../src/ivf/data/check.py#L255) |
| 5 | `plt.show()` under pytest on a desktop | pytest hung for minutes connected to the X server (found in M1: a GUI backend had been chosen by an earlier import) | No `plt.show()` in the notebook; the test switches to Agg, [`tests/test_notebooks.py:176`](../../tests/test_notebooks.py#L176) |
| 6 | Measuring "vol-of-vol rises with the level" with the same-day or next raw move | Skewed shocks alone make both positive (section 7.3) | The next AR(1) shock, [`src/ivf/eda.py:369`](../../src/ivf/eda.py#L369) |
| 7 | KPSS on a persistent series | Rejects a true stationary AR(1) with `phi = 0.98` on 31% of seeds | The verdict needs both tests; the nulls are printed with every result, [`src/ivf/eda.py:203`](../../src/ivf/eda.py#L203) |
| 8 | An unquoted date in YAML | Becomes a `date` object; `protocol_hash` undefined | [`src/ivf/protocol.py:125`](../../src/ivf/protocol.py#L125) |
| 9 | `--check` against a manifest whose file hashes include provenance | Every re-export looks stale | [`tools/export_app_data.py:63`](../../tools/export_app_data.py#L63) plus an on-disk consistency check |
| 10 | A hash typed from memory | A test asserted a wrong SHA-256 for `x\n` until it failed; the test was right to fail | Every hash in the tests comes from `sha256sum`, with the command in a comment |
| 11 | Renaming a field and running only the nearest test file | `vol_of_vol`'s p-value field was renamed; `tests/test_eda.py` passed, but the notebook spec still read the old name and only the full suite failed (found in M1, and it would have invalidated the notebook mutation K6) | Run the whole suite after any rename; the notebook-execution test, [`tests/test_notebooks.py:168`](../../tests/test_notebooks.py#L168) |
| 12 | Real numbers drifting into docs or tests | Breaks D2. Review A found real closes in the fixtures and a real byte total here | Docs quote synthetic numbers only; fixtures are made up; real results go to `ROOT/eda/`; `tools/licensing/scan_real_rows.py` and `git status --porcelain --ignored` before every commit (section 7.5) |
| 13 | A textbook p-value for a statistic built on estimated parameters (found by review A) | Spearman's p-value took the fitted shocks for the true ones; with skewed shocks it rejected on 30% of paths with no effect | A bootstrap that repeats the estimation on every path, [`src/ivf/eda.py:442`](../../src/ivf/eda.py#L442) |
| 14 | A network guard that some tests request and that patches one of several routes (review A) | `connect_ex`, DNS and curl_cffi stayed open, and deleting the main patch failed no test; after review A, `sendmsg` and `gethostbyname` still were (review B) | Installed for the session, one test per route, [`tests/conftest.py:278`](../../tests/conftest.py#L278). Known limits: a subprocess (the pipeline and diagram tests start some), `_socket` used directly (`socket.SocketType`) and C code other than libcurl run unguarded |
| 15 | A licensing guard on one writer only (review A) | The pipeline wrote raw hashes and EDA numbers into the checkout | One function for every writer, [`src/ivf/repo_guard.py:155`](../../src/ivf/repo_guard.py#L155) |
| 16 | Computing EDA on the holdout (review A) | Later model choices would have seen it (ADR 0001 4.6) | [`src/ivf/data/process.py:87`](../../src/ivf/data/process.py#L87); `eda_report` refuses holdout rows |
| 17 | A rule enforced only by the tool that writes the value (review A) | A hand edit or a same-day refetch of a frozen `data_end` passed `check` and `eda` | Re-validated every run, [`src/ivf/data/check.py:223`](../../src/ivf/data/check.py#L223) |
| 18 | "Not found" written as "does not exist" (review A) | `tenor_days` of VIX1Y was 365 and three tenors were called unverified, though Cboe states them | Cite the document and page; say "not checked" when the search ran out |
| 19 | A resampling scheme that assumes iid when the data cluster (found by review B) | Review A's iid bootstrap rejected on 47% of GARCH paths with no level effect, and the real E6 verdict rested on it | Blocks of consecutive shocks, [`src/ivf/eda.py:402`](../../src/ivf/eda.py#L402); size measured on ten nulls, power stated (§7.3) |
| 20 | Checking only the root of a tree that gets written into (found by review B) | A symlinked `raw/` or `eda/` sent snapshots and the EDA report into the checkout | The subfolders before the first write, and every folder at write time, [`src/ivf/repo_guard.py:191`](../../src/ivf/repo_guard.py#L191) |
| 21 | A frozen value that a hand edit can move to another valid value (found by review B) | `data_end` moved back to an earlier settled session passed every data rule | The freeze record, [`src/ivf/protocol.py:244`](../../src/ivf/protocol.py#L244) |
| 22 | A library that swallows the exception a guard raises (found by review B) | yfinance logged "possibly delisted" and returned an empty frame; the test passed | Every refusal recorded and charged to its test, [`tests/conftest.py:305`](../../tests/conftest.py#L305) |
| 23 | A scan that knows one spelling of a date (found by review B) | `date(1990, 5, 25)`, epoch-ms keys and lists of closes without dates passed it | Every listed date spelling and runs of consecutive raw values, [`tools/licensing/scan_real_rows.py:377`](../../tools/licensing/scan_real_rows.py#L377) |

---

## 7. Measured results

### 7.1 The pipeline and the calendar

| What | Command | Result | Date |
|---|---|---|---|
| Real pipeline | `uv run python -m ivf.data fetch`, `build-interim`, `verify`, `check` | All exit 0; gap policy OK; coverage and gap counts printed. **Measured locally, not published (D2).** Re-run after review A (fetch without `--force`: 7 skipped, nothing downloaded; then `build-interim`, `verify`, `check`, `eda`): all exit 0, the frozen `data_end` re-validated, no holdout row read. Re-run after review B (amended M1, review B): the same, plus `check --freeze-data-end`, which recorded the 2026-10-28 freeze in `configs/protocol.freeze.json`; all exit 0 | 2026-10-29 |
| `train_start` is the first VIX date | `check` (row `train_start_is_first_vix`) | Yes: 1990-01-02 | 2026-10-29 |
| First dates on Cboe's files | `build-interim` | VIX9D 2011-01-04, VIX3M 2009-09-18, VIX6M 2008-01-02, VIX1Y 2007-01-03, VVIX 2006-03-06 (availability dates, not values) | 2026-10-29 |
| Cboe 307 redirect | `fetch` (`final_url`) | `cdn.cboe.com` → `cdn-api.cboe.com` for all six files | 2026-10-29 |
| `data_end` | `check --freeze-data-end` | 2026-10-28; `protocol_hash` c170a7eb9453... | 2026-10-29 |
| Calendar vs ADR 0001 (M0) | `pytest tests/test_protocol_calendar.py` | 2,579 sessions to 2000-03-15; first cutoff 2000-03-15; 97 dev folds; 50 burn-in sessions from 2000-01-03; 583 holdout sessions to 2026-10-26: all as M0 measured | 2026-10-29 |
| Real EDA and real export | `uv run python -m ivf.data eda`; `tools/export_app_data.py --mode real` and `--mode real --check` | Ran; outputs in `ROOT/eda/` and `ROOT/app-data/real/`. **Measured locally, not published (D2).** | 2026-10-29 |
| `tenor_days` | Cboe's VIX methodology; Cboe, "Selected SPX Target Expected Volatility Term Indices Methodology", v7.0 (last revised 2026-03-04), section 2.1, pp. 5-6 | VIX 30; VIX9D 9, VIX3M 93, VIX6M 184, VIX1Y **366** calendar days (amended M1, review A: the first version said 365 and "no Cboe primary source found"; the source was read again for this fix) | 2026-10-29 |
| VIX close time (16:15 ET) | | **Not measured** | |

### 7.2 EDA on the synthetic market

`VOL_DATA_DIR=<scratch> uv run python -m ivf.data --synthetic fetch` then `build-interim`,
`verify`, `check`, `eda` (all exit 0); read `eda/<run_id>/eda_report.json`. 2026-10-29, re-run
after review A. Truth: `phi = 0.98` (half-life 34.31), GARCH `alpha = 0.08, beta = 0.90`
(`rho_1(r²) = 0.205`), contango probability 0.857, vol of ln VIX 0.952 a year, variance gap 0.0132.

The EDA now uses only the 8,689 sessions before `holdout_start` (1990-01-02..2024-06-28):
`pre_holdout` is all of them, `burn_in` (2,578) and `dev` (6,111) the calendar's periods inside it
(amended M1, review A: the first version's samples were full, "dev" (which was burn-in plus dev,
the `pre_holdout` column below) and holdout).

| Statistic | pre_holdout | burn_in | dev |
|---|---:|---:|---:|
| `phi_hat` (95% CI) | 0.9813 (0.9773, 0.9854) | 0.9824 (0.9752, 0.9897) | 0.9808 (0.9759, 0.9857) |
| Half-life, sessions (95% CI) | 36.8 (30.2, 47.1) | 39.1 (27.6, 66.9) | 35.8 (28.4, 48.2) |
| ADF p / KPSS p / stationary at 5% | 5.0e-15 / > 0.10 / yes | 6.0e-05 / 0.084 / yes | 9.3e-12 / > 0.10 / yes |
| Variance gap (implied - realised) | +0.0152 | +0.0098 | +0.0175 |
| Implied above realised | 74.3% | 70.0% | 76.1% |
| Contango share (VIX <= VIX3M) | 84.7% | none (VIX3M starts 2009) | 84.7% |

Clustering (pre-holdout): ACF lag 1 of `r` -0.0090 (Ljung-Box Q(21) = 18.7, p = 0.60); of `r²`
0.174 (Q(21) = 3,809); of `|r|` 0.152 (Q(21) = 2,821). Persistence: ACF of ln VIX at lag 21 0.686
against `0.98^21 = 0.654`. Vol-of-vol: mean annualised 0.931; the E6 measure (rank correlation of
the level with the next AR(1) shock, 8,688 pairs) -0.0005 with bootstrap p = 0.92 (999 paths), and
review A's version of E6 said "no" there, against +0.30 in vol points. Amended M1, review B: with the block bootstrap, `L` = 2
and p = 0.924, and E6 says "not shown" (re-run 2026-10-29 on the same synthetic files). ADF and KPSS ran on the whole sample of each column:
the synthetic market has no gaps.

M1 no longer computes holdout statistics. At holdout size the E7 verdict would say little anyway:
on 100 stationary paths of 583 sessions it said "not stationary" 77 times (§7.3; concept 05 §4).

### 7.3 Estimator behaviour (scratch scripts, 2026-10-29)

| What | Result |
|---|---|
| `phi_hat`, `phi = 0.98`, n = 20,000, 20 seeds | mean 0.97997, sd 0.00124; 95% CI covered 0.98 on 19 of 20 |
| `rho_1(r²)`, GARCH 0.10/0.80, n = 200,000, 20 seeds | mean 0.1389 against 0.14, sd 0.0054 |
| `rho_1(r²)`, GARCH 0.08/0.90, n = 9,272, 20 seeds | mean 0.193 against 0.205, range 0.136 to 0.255 |
| Contango share, n = 200,000, 20 seeds | mean 0.8564 against 0.8569, sd 0.0064 |
| Variance gap, n = 200,000, 20 seeds | mean 0.01323 against 0.01319, sd 0.00064 |
| KPSS rejects a stationary AR(1), 100 seeds | `phi = 0.5`: 7%; `phi = 0.9`: 8%; `phi = 0.98` (n = 9,000): 31% |
| E6 measures with iid skewed shocks (no level effect), 5 seeds | next AR(1) shock: abs(rho) <= 0.0095; next raw move: +0.105 to +0.114; same-day move: +0.163 to +0.172 |
| E6 test as shipped after review A (iid bootstrap, B = 999), 300 paths each, n = 9,000, `phi = 0.98`; the review-A version of `tools/probes/e6_size_power.py` (about 22 min; amended M1, review A). **Superseded by the table below (review B)** | **Size**, iid normal shocks: rejects at 5% on 3.3% (10/300); the old Spearman p-value 4.0%. iid exponential shocks: **5.3% (16/300)**; Spearman **30.7%**; the E6 verdict "yes" (rho > 0, p < 0.05, nominal 2.5%) 2.3% against 9.3%. `rho` mean -0.0047, sd 0.0194 (normal: -0.0004, 0.0104). **Power**, shock sd x min(e^{0.5(x-mu)}, 3): 100% (300/300), Spearman 100%; the weaker min(e^{0.15(x-mu)}, 3): 99.0% (297/300), Spearman 99.0% |
| E6 alternatives, 60 paths, n = 9,000, B = 199 (scratch pilot) | fitting `phi` on one half and testing the other: 30.0% under exponential shocks, no better than Spearman (26.7%); the bootstrap 3.3% |
| E6 test in the suite: 100 paths of 1,000 sessions, B = 99 (`test_e6_size_under_skewed_iid_shocks_stays_near_5_percent`, 1.2 s) | bootstrap 5/100; Spearman 30/100. The test fails above 12/100 |
| E6 under clustering in the suite (amended M1, review B): 100 paths of 1,000 sessions, GARCH(0.10, 0.88), B = 99 (`test_e6_size_under_garch_shocks_with_no_level_effect_stays_near_5_percent`, 2.3 s) | shipped 1/100; `block=1` (review A's design) 44/100 on the same paths. The test fails above 12/100, or if `block=1` rejects fewer than 25 |
| KPSS, ADF and E7 at holdout size: `phi = 0.98`, n = 583, 100 seeds; `uv run python tools/probes/kpss_holdout_size.py` | KPSS rejects 61, ADF rejects 44, E7 "not stationary" 77 of 100 (review A measured the same) |

#### E6 when the shocks cluster (amended M1, review B)

Every row is simulated (no file is read), ln V an AR(1) with `phi = 0.98`, mean ln 18, shock sd
0.06, n = 9,000, B = 999, 300 paths per row, seed 20261029 for the resampling. "Rejects" is the
two-sided bootstrap p below 0.05. The acceptance bound for a design was a size of at most 8% on
every null. 2026-10-29.

**Is "no level effect" the same as `rho = 0`?** Not when skewed shocks cluster. One path of
10^6 sessions per seed, 8 seeds, `rho` without a bootstrap (`uv run python
tools/probes/e6_population_rho.py`, about 4 s):

| Shocks, no level effect | mean `rho` (sd over 8 seeds) |
|---|---:|
| iid normal; iid exponential minus 1 | +0.0006 (0.0007); +0.0003 (0.0015) |
| symmetric GARCH (0.05, 0.90); (0.10, 0.88) | +0.0012 (0.0008); +0.0020 (0.0022) |
| skewed GARCH (exponential innovations), (0.05, 0.90); (0.10, 0.88) | **+0.1331** (0.0017); **+0.2471** (0.0035) |

A big up-move raises the level and, through the clustering, the next shock sizes. So any test of
"`rho` = 0" (iid, wild or HAC alike) must reject on skewed GARCH, the HAC test included; only a
bootstrap that reproduces the clustering can treat it as the null.

**Designs compared on the same paths** (`uv run python tools/probes/e6_designs.py SCHEME SCENARIO
300 SEED0`, SEED0 = 201000 for `normal` and 1,000 more for each next row in this order; about 30 s
a row, 3.5 to 5.5 minutes for the Python-loop block rows of the scenarios without clustering). Rejection rate at 5%:

| Scenario | iid (review A) | wild, Rademacher | wild, Mammen | circular blocks | stationary bootstrap |
|---|---:|---:|---:|---:|---:|
| iid normal | 3.3% | 4.3% | 3.7% | 3.0% | 3.0% |
| iid exponential (skewed) | 5.3% | **26.0%** | **10.7%** | 4.7% | 4.0% |
| GARCH (0.05, 0.90) | **18.7%** | 5.7% | **8.7%** | 0.0% | 0.0% |
| GARCH (0.10, 0.88) | **47.3%** | 7.3% | **10.0%** | 0.0% | 0.0% |
| skewed GARCH (0.05, 0.90) | **100%** | **100%** | **100%** | 1.3% | 0.3% |
| skewed GARCH (0.10, 0.88) | **100%** | **100%** | **100%** | 0.0% | 0.0% |
| level effect k = 0.5 (power) | 100% | 100% | 100% | 93.3% | 90.7% |
| level effect k = 0.15 (power) | 99.3% | 99.0% | 99.0% | 99.3% | 99.3% |

Mean block length by the rule: 1.4 to 1.7 on the iid rows, 58 to 66 for GARCH (0.05, 0.90), 180
to 205 for (0.10, 0.88), 89 to 102 and 170 to 195 for the skewed GARCH rows, 58 to 66 under the
k = 0.5 effect. **Decision:** circular blocks. The wild bootstraps keep each `abs(e_hat)` and so the
clustering, but flip signs, which throws the skew away (26% on skewed iid shocks). The stationary
bootstrap does as well as circular blocks; its random block lengths are harder to vectorise, and
nothing measured favours it. A HAC test of "`rho` = 0" was not built: the skewed GARCH rows show
that `rho` is not 0 there, so its size would go to 100% with n. What would make a rejected design
win: a wild bootstrap, if VIX's shocks were symmetric; review A's iid draws, if they did not cluster
(review B: on real data the Ljung-Box test on squared AR(1) shocks gave p < 1e-6).

**The shipped test against review A's, on the same paths** (`uv run python
tools/probes/e6_size_power.py 300 9000`, 13 m 47 s with 18 processes; `block=1` is review A's
design):

| Scenario | Shipped: rejects | review A: rejects | `rho` mean (sd) | block `L`: mean [min, max] |
|---|---:|---:|---:|---:|
| H0 iid normal | 5.3% (16/300) | 5.7% | +0.0010 (0.0107) | 1.6 [1, 4] |
| H0 iid exponential | 2.3% (7/300) | 2.7% | -0.0019 (0.0169) | 1.6 [1, 4] |
| H0 GARCH (0.05, 0.90) | 0.0% (0/300) | **19.7%** | +0.0008 (0.0160) | 67.3 [10, 141] |
| H0 GARCH (0.10, 0.88) | 0.0% (0/300) | **47.0%** | +0.0003 (0.0287) | 203.5 [140, 255] |
| H0 skewed GARCH (0.05, 0.90) | 0.3% (1/300) | **100%** | +0.1294 (0.0250) | 105.5 [50, 202] |
| H0 skewed GARCH (0.10, 0.88) | 0.0% (0/300) | **100%** | +0.2409 (0.0370) | 194.9 [125, 259] |
| H0 skewed GARCH (0.03, 0.965), volatility half-life 138 | 0.0% (0/300) | **100%** | +0.1783 (0.0353) | 245.9 [174, 268] |
| H0 GJR: normal innovations, volatility rises more after up-moves (0.02, 0.12, 0.88) | 0.0% (0/300) | **100%** | +0.1264 (0.0193) | 145.1 [94, 215] |
| H0 iid exponential, `phi = 0.995` | 5.7% (17/300) | 6.3% | -0.0077 (0.0185) | 1.6 [1, 4] |
| H0 iid exponential, 60 sessions missing | 4.3% (13/300) | 5.0% | -0.0005 (0.0191) | 1.5 [1, 4] |
| H1 shock sd x min(e^{0.5(x - mu)}, 3) | **95.0%** (285/300) | 100% | +0.1590 (0.0139) | 67.5 [2, 225] |
| H1 shock sd x min(e^{0.15(x - mu)}, 3) | **99.7%** (299/300) | 99.7% | +0.0486 (0.0106) | 1.6 [1, 4] |
| H1 the k = 0.5 effect on top of GARCH (0.05, 0.90) | **18.0%** (54/300) | 100% | +0.1524 (0.0185) | 142.2 [75, 222] |

The E6 verdict ("yes": `rho > 0` and p < 0.05) counts are in the probe's output; under every null
they are at most 9 of 300 for the shipped test. **What the table says.** The size holds on all ten
nulls (at most 5.7%), and under clustering the test is conservative (0 to 0.3%). The price is power
when the shocks cluster: a strong level effect on top of GARCH clustering is found on 18% of paths,
because the blocks, 142 sessions long there, keep the level effect inside them along with the
clustering. In this statistic the two leave the same trace (the level is itself a weighted sum of
past shocks), so no resampling scheme can separate them; a volatility model can (§11). Hence E6
reads "yes" or "not shown".

#### E6 when the level is very persistent too (amended M1, final check)

The final check found a null the ten above missed: skewed GARCH(0.03, 0.965) shocks AND a level
`phi` of 0.99 or more. The blocks (mean about 245, near the rule's cap of 284) are shorter than
the level's memory, and the test rejects with no level effect. 200 fresh paths per row, n = 9,000,
B = 999 (`uv run python tools/probes/e6_size_power.py 200 9000 skew98both skew985both skew99both
skew995both lnorm995both`, 3 m 27 s with 18 processes; seeds 400000 on):

| Null: skewed GARCH (0.03, 0.965), no level effect | rejects (= "yes") | `phi_hat` mean [min, max] | "yes" if `phi_hat` <= 0.975 / 0.98 / 0.985 | block `L` mean |
|---|---:|---:|---:|---:|
| `phi` = 0.98 | 0.0% (0/200) | 0.9791 [0.9721, 0.9878] | 0 / 0 / 0 | 245.4 |
| `phi` = 0.985 | 2.0% (4/200) | 0.9840 [0.9771, 0.9902] | 0 / 0 / 0 | 245.6 |
| `phi` = 0.99 | **15.0%** (30/200) | 0.9890 [0.9810, 0.9944] | 0 / 0 / 0 | 244.7 |
| `phi` = 0.995 | **44.5%** (89/200) | 0.9943 [0.9900, 0.9986] | 0 / 0 / 0 | 245.3 |
| `phi` = 0.995, standardised lognormal innovations | **34.5%** (69/200) | 0.9935 [0.9788, 1.0016] | 0 / 0 / 0 | 233.2 |

Power with each cutoff, the three alternatives above (300 paths, review B's seeds, so the
"rejects" column repeats the table above exactly): k = 0.5: 14 / 164 / 285 of 300; k = 0.15: 4 /
158 / 299; k = 0.5 on GARCH: 0 / 28 / 54.

**Decision (pre-registered rule):** `eda.E6_PHI_MAX` = 0.985, the largest of 0.975, 0.98 and 0.985
at which every null in the probe rejects on at most 8% and says "yes" on at most 5% (the ten review-B
nulls at most 5.7% and 9 of 300; the two new ones at `phi` <= 0.985, 0.0% and 2.0%). Above it, E6
says "not shown" with the reason "persistence above the size-checked range", and still reports
`rho` and p. 0.985 costs no power at `phi` = 0.98; 0.98 would cost about half of it, because
`phi_hat` scatters around the true 0.98. **Rejected:** a longer block. It was not measured; the bar
was size at most 8% on every null and power at least 80% at k = 0.5 under GARCH, where the current
block already has 18%. **What would change this:** a size measurement above 0.985 that holds, or a
volatility model (§11).

**Short samples.** The suite's size test for this null runs at n = 2,000 (100 paths, B = 99,
`test_e6_verdict_keeps_its_size_under_persistent_skewed_garch_through_the_cutoff`, about 4 s): 55
of 100 paths significant, 2 "yes" through the cutoff. But at that length the test is oversized
even at `phi` = 0.98 (27 of 100 significant, 24 "yes" through the cutoff; at n = 1,000, 50 and 39;
at n = 4,500, 3 of 60), which the fix round found in its own check. So E6 also has a minimum
sample, `eda.E6_MIN_N` (default decision, 2026-10-29). Rule: the smallest of 4,000, 6,000 and
8,000 at which E6 says "yes" (with the `phi` cutoff) on at most 10 of 200 fresh paths, B = 999, on
each of three nulls. `uv run python tools/probes/e6_size_power.py 200 N skew98both skew985both
skew` with `E6_PROBE_SEED_OFFSET` = 1,000,000, 2,000,000 and 3,000,000 for N = 4,000, 6,000 and
8,000. "Yes" out of 200, two-sided rejections in brackets:

| Sessions (pairs = sessions - 1) | skewed GARCH (0.03, 0.965), `phi` 0.98 | the same, `phi` 0.985 | iid exponential, `phi` 0.98 |
|---:|---:|---:|---:|
| 4,000 | **10** (12) | 5 (24) | 0 (9) |
| 6,000 | 1 (1) | 0 (10) | 3 (9) |
| 8,000 | 0 (0) | 0 (5) | 5 (13) |

**Decision:** `E6_MIN_N` = 6,000 pairs. The rule picks 4,000, but only on the boundary: 10 of 200
at `phi` 0.98, the `phi` gate is what holds `phi` 0.985 there (24 rejections, 5 "yes"), and the
measured paths had 3,999 pairs, one fewer than the minimum. So the orchestrator took the next size,
6,000, where the counts were 1, 0 and 3 of 200. Below it E6 says "not shown" with the reason
"sample below the size-checked length (n < 6000 pairs)". The pre-holdout VIX sample has about 8,688 pairs,
so its verdict is unchanged ("not shown" before and after, exit 0). **What would change it:** a
re-run that gives more than 10 of 200 at 6,000.

### 7.4 Mutation checks

In a scratch copy (`rsync` without `.venv`, `node_modules` and `.git`, then `git init`), each
mutation was applied alone and the listed test files run with `PYTHONPATH` pointing at the copy.
The baseline in the copy was `578 passed`. An earlier run, on an earlier version of the code,
left three mutants alive: I1 (any five-column header), E7 (a verdict from ADF alone) and the
same-day E6 variant. Section 4 names the tests added to kill them; the table below is the run
on the final code (2026-10-29).
**63 of 63 mutations were caught.** That number measured the mutants I chose. Review A chose 11
from the claims instead, and 9 of them passed all 578 tests; the second table below is the run
after its fixes (amended M1, review A).

| Mutation | File | Test files run | Result | A failing test |
|---|---|---|---|---|
| R1 store: os.link -> os.replace (overwrites an existing snapshot) | `src/ivf/data/raw.py` | test_data_fetch | 2 failed, 46 passed in 2.53s | `test_identical_bytes_are_stored_once_but_every_fetch_is_logged` |
| R2 fetch: no max-age skip | `src/ivf/data/fetch.py` | test_data_fetch | 2 failed, 46 passed in 2.53s | `test_a_fresh_symbol_is_skipped_with_no_network_call_and_no_log_line` |
| R3 fetch: log only changed content | `src/ivf/data/fetch.py` | test_data_fetch | 2 failed, 46 passed in 2.49s | `test_identical_bytes_are_stored_once_but_every_fetch_is_logged` |
| R4 fetch: no HTTP status check | `src/ivf/data/fetch.py` | test_data_fetch | 2 failed, 46 passed in 2.49s | `test_a_failed_download_stores_nothing_and_logs_nothing` |
| R5 yfinance bytes: unsorted rows | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 47 passed in 2.86s | `test_yfinance_frame_becomes_exactly_these_bytes` |
| R6 yfinance bytes: 4 decimals | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 47 passed in 2.85s | `test_yfinance_frame_becomes_exactly_these_bytes` |
| R7 append: no torn-line refusal | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 47 passed in 2.43s | `test_a_torn_last_line_blocks_the_next_append` |
| R8 verify: no re-hash | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 47 passed in 2.44s | `test_verify_catches_a_changed_byte` |
| R9 verify: no orphan check | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 47 passed in 2.46s | `test_verify_catches_a_file_no_fetch_accounts_for` |
| R10 provenance: skip unreadable lines | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 47 passed in 2.84s | `test_an_unreadable_provenance_line_is_an_error_not_skipped` |
| R11 latest: time only, first line wins ties | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 47 passed in 2.36s | `test_latest_fetch_is_the_latest_time_and_on_a_tie_the_later_line` |
| R12 symbol: no path check | `src/ivf/data/raw.py` | test_data_fetch | 5 failed, 43 passed in 2.55s | `test_a_symbol_cannot_become_a_path_outside_the_store` |
| I1 cboe: any header accepted as OHLC | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.31s | `test_anything_unexpected_fails_loudly_with_a_location` |
| I2 numbers: nan/inf accepted | `src/ivf/data/build_interim.py` | test_data_interim | 2 failed, 42 passed in 0.32s | `test_anything_unexpected_fails_loudly_with_a_location` |
| I3 duplicates: conflicting date keeps the first | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.32s | `test_a_date_with_two_different_values_is_an_error` |
| I4 levels: <= 0 accepted | `src/ivf/data/build_interim.py` | test_data_interim | 3 failed, 41 passed in 0.32s | `test_anything_unexpected_fails_loudly_with_a_location` |
| I5 dates: mixed formats accepted | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.33s | `test_anything_unexpected_fails_loudly_with_a_location` |
| I6 pins ignored | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.31s | `test_the_latest_snapshot_is_used_unless_one_is_pinned` |
| I7 build: no hash check of the snapshot | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.32s | `test_a_snapshot_whose_bytes_changed_is_refused` |
| I8 build: two sources merged silently | `src/ivf/data/build_interim.py` | test_data_interim | 1 failed, 43 passed in 0.32s | `test_two_sources_for_one_symbol_are_an_error_in_milestone_1` |
| C1 first_cutoff rounds k up | `src/ivf/data/calendar.py` | test_protocol_calendar | 2 failed, 25 passed in 1.92s | `test_calendar_matches_the_figures_adr_0001_measured_in_milestone_0` |
| C2 extension h_max - 1 | `src/ivf/data/calendar.py` | test_protocol_calendar | 1 failed, 26 passed in 1.90s | `test_master_calendar_reaches_h_max_sessions_past_data_end_with_dense_indices` |
| C3 period: cutoff is burn_in | `src/ivf/data/calendar.py` | test_protocol_calendar | 1 failed, 26 passed in 1.89s | `test_periods_are_burn_in_before_the_first_cutoff_dev_until_holdout_then_holdout` |
| C4 freeze moves a frozen value | `src/ivf/protocol.py` | test_protocol_calendar | 1 failed, 26 passed in 1.95s | `test_a_frozen_data_end_is_never_moved` |
| C5 unquoted YAML dates accepted | `src/ivf/protocol.py` | test_protocol_calendar | 1 failed, 26 passed in 1.95s | `test_a_malformed_protocol_is_refused` |
| C6 protocol: unknown keys accepted | `src/ivf/protocol.py` | test_protocol_calendar | 1 failed, 26 passed in 1.90s | `test_a_malformed_protocol_is_refused` |
| K1 spill 21 instead of 22 | `src/ivf/data/check.py` | test_data_check | 1 failed, 13 passed in 1.55s | `test_gap_policy_g3_boundary_is_22_sessions_before_the_first_cutoff` |
| K2 settled by UTC date | `src/ivf/data/check.py` | test_data_check | 2 failed, 12 passed in 1.55s | `test_data_end_is_the_last_session_settled_before_the_fetch_day_in_new_york` |
| K3 core gap ignores ^GSPC | `src/ivf/data/check.py` | test_data_check, test_synthetic_pipeline | 3 failed, 24 passed in 14.48s | `test_gap_policy_g3_boundary_is_22_sessions_before_the_first_cutoff` |
| K4 no train_start check | `src/ivf/data/check.py` | test_data_check | 1 failed, 13 passed in 1.55s | `test_train_start_must_be_the_first_vix_date` |
| K5 candidate ignores settlement | `src/ivf/data/check.py` | test_data_check | 3 failed, 11 passed in 1.57s | `test_data_end_is_the_last_session_settled_before_the_fetch_day_in_new_york` |
| K6 summary stream bound at import | `src/ivf/data/check.py` | test_notebooks, test_synthetic_pipeline | 1 failed, 31 passed in 19.19s | `test_m1_notebook_runs_top_to_bottom_on_synthetic_data_when_there_is_no_real_data` |
| S1 rv21 annualises 252/20 | `src/ivf/features/build.py` | test_features | 6 failed, 9 passed in 1.33s | `test_rv21_of_21_returns_of_one_percent_is_15_8745` |
| S2 windows accept partial data | `src/ivf/features/build.py` | test_features | 4 failed, 11 passed in 1.34s | `test_rv21_of_21_returns_of_one_percent_is_15_8745` |
| S3 returns bridge gaps (ffill) | `src/ivf/features/build.py` | test_features | 1 failed, 14 passed in 1.37s | `test_a_missing_close_makes_two_returns_missing_and_is_never_bridged` |
| S4 har_m window 21 | `src/ivf/features/build.py` | test_features | 2 failed, 13 passed in 1.32s | `test_feature_columns_and_first_empty_rows_match_the_data_model` |
| S5 forward off by one | `src/ivf/features/build.py` | test_features, test_eda | 2 failed, 42 passed in 2.33s | `test_forward_shifts_values_back_by_h_and_pads_with_nan` |
| S6 centred windows (look-ahead) | `src/ivf/features/build.py` | test_features | 10 failed, 5 passed in 1.32s | `test_rv21_of_21_returns_of_one_percent_is_15_8745` |
| S7 regime low inclusive | `src/ivf/features/build.py` | test_features | 1 failed, 14 passed in 1.38s | `test_regime_labels_use_fixed_edges_and_strict_backwardation` |
| S8 backwardation at ratio 1 (the shared rule) | `src/ivf/features/build.py` | test_features | 2 failed, 13 passed in 1.35s | `test_regime_labels_use_fixed_edges_and_strict_backwardation` |
| S9 vix_vix3m inverted | `src/ivf/features/build.py` | test_features | 1 failed, 14 passed in 1.38s | `test_ratios_put_the_shorter_tenor_on_top_and_iv_rv_spread_is_in_variance_units` |
| E1 acf drops NaN first | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.52s | `test_acf_never_pairs_values_across_a_gap` |
| E2 acf divides each lag by its own count | `src/ivf/eda.py` | test_eda | 5 failed, 24 passed in 2.41s | `test_acf_of_1_2_3_4_is_0_25_at_lag_1` |
| E3 Ljung-Box without the n-k weights | `src/ivf/eda.py` | test_eda | 2 failed, 27 passed in 2.33s | `test_ljung_box_on_1_2_3_4_is_q_0_5` |
| E4 AR(1) pairs bridge gaps | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.44s | `test_ar1_forms_pairs_on_the_calendar_not_across_a_gap` |
| E5 half-life inverted | `src/ivf/eda.py` | test_eda | 3 failed, 26 passed in 2.34s | `test_half_life_is_ln_half_over_ln_phi` |
| E6 HC1 is OLS | `src/ivf/eda.py` | test_eda | 8 failed, 21 passed in 2.30s | `test_ar1_on_an_exact_halving_series_finds_phi_one_half` |
| E7 stationary verdict ignores KPSS | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.48s | `test_when_kpss_rejects_the_verdict_is_not_stationary_even_if_adf_rejects` |
| E8 variance gap not squared | `src/ivf/eda.py` | test_eda | 3 failed, 26 passed in 2.44s | `test_spread_by_hand` |
| E9 vov annualised with 250 | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.40s | `test_vov_is_annualised_with_sqrt_252` |
| E10 E6 measure uses the same-day shock | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.30s | `test_e6_measure_ignores_skewed_shocks_that_fool_the_simpler_measures` |
| E12 E6 measure uses the next raw move | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.32s | `test_e6_measure_ignores_skewed_shocks_that_fool_the_simpler_measures` |
| P2 runs.parquet appends instead of replacing | `src/ivf/data/process.py` | test_synthetic_pipeline | 1 failed, 12 passed in 15.56s | `test_rerunning_eda_on_the_same_inputs_replaces_its_own_run_row` |
| E11 contango strict | `src/ivf/eda.py` | test_eda | 1 failed, 28 passed in 2.38s | `test_term_structure_by_hand` |
| Y1 GARCH r2 acf formula drops alpha*beta | `src/ivf/data/synthetic.py` | test_eda, test_synthetic_pipeline | 1 failed, 41 passed in 18.57s | `test_garch_squared_returns_cluster_with_the_closed_form_acf_while_returns_do_not` |
| Y2 contango probability uses offset/slope | `src/ivf/data/synthetic.py` | test_eda, test_synthetic_pipeline | 2 failed, 40 passed in 16.72s | `test_contango_share_recovers_the_closed_form` |
| Y3 GARCH recursion uses r_t | `src/ivf/data/synthetic.py` | test_eda, test_synthetic_pipeline | 2 failed, 40 passed in 17.13s | `test_garch_squared_returns_cluster_with_the_closed_form_acf_while_returns_do_not` |
| X1 ribbon starts at the first tenor | `tools/export_app_data.py` | test_export | 1 failed, 86 passed in 17.17s | `test_term_structure_starts_when_every_tenor_exists_and_rounds_as_the_data_model_says` |
| X2 check compares listed sha256 | `tools/export_app_data.py` | test_export | 1 failed, 86 passed in 15.15s | `test_real_check_passes_on_its_own_output_and_ignores_the_listed_hash` |
| X3 no manifest/file consistency | `tools/export_app_data.py` | test_export | 1 failed, 86 passed in 15.21s | `test_real_check_catches_a_term_structure_that_does_not_match_its_manifest` |
| X4 no tenor sort check | `tools/export_app_data.py` | test_export | 1 failed, 86 passed in 15.22s | `test_term_structure_invariants_the_schema_cannot_express_are_checked` |
| X5 levels rounded to 3 dp | `tools/export_app_data.py` | test_export | 1 failed, 86 passed in 15.10s | `test_term_structure_starts_when_every_tenor_exists_and_rounds_as_the_data_model_says` |
| P1 run_id keeps input order | `src/ivf/data/process.py` | test_synthetic_pipeline | 1 failed, 12 passed in 16.72s | `test_the_eda_run_id_changes_with_the_inputs` |

**After review A** (amended M1, review A). A fresh scratch copy (`rsync` without `.venv` and
`node_modules`), baseline `634 passed`; each mutant applied alone, the listed files run with the
repo's interpreter and `PYTHONPATH` pointing at the copy, the file restored and its SHA-256 checked
(review A's `probes/mutate.py`). V1-V11 are review A's survivors (or their successor in the new
design); the rest test the fixes. 2026-10-29.

| Mutation | File | Test files run | Result | A failing test |
|---|---|---|---|---|
| V1 guard: no `socket.socket.connect` patch | `tests/conftest.py` | test_network_guard | 2 failed, 9 passed | `test_socket_connect_is_blocked` |
| V1b guard: no `connect_ex` patch | `tests/conftest.py` | test_network_guard | 1 failed, 10 passed | `test_socket_connect_ex_is_blocked` |
| V1c guard: no `getaddrinfo` patch | `tests/conftest.py` | test_network_guard | 1 failed, 10 passed | `test_dns_lookup_is_blocked` |
| V1d guard: no curl_cffi `Curl.perform` patch | `tests/conftest.py` | test_network_guard | 1 failed, 10 passed | `test_curl_cffi_the_yfinance_client_is_blocked` |
| V1e guard: not installed for the session | `tests/conftest.py` | test_network_guard | 9 failed, 2 passed | `test_the_guard_is_on_for_a_test_that_did_not_ask_for_it` |
| V1f guard: no `AsyncCurl.add_handle` patch | `tests/conftest.py` | test_network_guard | 1 failed, 10 passed | `test_curl_cffi_async_requests_are_blocked` |
| V1g guard: no `sendto` patch | `tests/conftest.py` | test_network_guard | 1 failed, 10 passed | `test_udp_sendto_is_blocked` |
| V1h guard: no `create_connection` patch | `tests/conftest.py` | test_network_guard | **11 passed: equivalent.** `create_connection` calls `getaddrinfo` and `connect`, both blocked; the patch stays for its clearer message | none |
| V2 EDA tables run past `data_end` | `src/ivf/data/process.py` | test_process, test_synthetic_pipeline | 1 failed, 19 passed | `test_rows_after_data_end_change_nothing_the_eda_writes` |
| V2b interim rows after `data_end` not filtered | `src/ivf/data/process.py` | test_process, test_synthetic_pipeline | **20 passed: equivalent.** `build.align` reads only the sessions it is given, so the filter changed nothing; it was removed and the comment says why | none |
| V3 spread with trailing RV (window t-20..t) | `src/ivf/data/process.py` | test_process, test_synthetic_pipeline | 2 failed, 18 passed | `test_forward_rv21_at_origin_t_covers_exactly_the_21_sessions_after_t` |
| V4 spread window one session late (t+2..t+22) | `src/ivf/data/process.py` | test_process, test_synthetic_pipeline | 2 failed, 18 passed | `test_the_spread_pairs_each_vix_with_the_rv_of_its_own_next_21_sessions` |
| V5 HC0 in place of HC1 | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_hc1_standard_error_and_95_percent_interval_by_hand` |
| V6 a 90% interval reported as 95% | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_hc1_standard_error_and_95_percent_interval_by_hand` |
| V8 KPSS with `regression="ct"` | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_kpss_tests_level_stationarity_not_trend_stationarity` |
| V9 E6 shock without its intercept | `src/ivf/eda.py` | test_eda | 3 failed, 34 passed | `test_ar1_shocks_on_an_exact_line_are_zero_and_need_the_intercept` |
| V9b E6 bootstrap shock without its intercept | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_e6_size_under_skewed_iid_shocks_stays_near_5_percent` |
| V9c E6 bootstrap without the refit | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_e6_size_under_skewed_iid_shocks_stays_near_5_percent` |
| V9d E6 p-value back to scipy's Spearman | `src/ivf/eda.py` | test_eda | 2 failed, 35 passed | `test_e6_size_under_skewed_iid_shocks_stays_near_5_percent` |
| V9e E6 p-value one-tailed (no factor 2) | `src/ivf/eda.py` | test_eda | 2 failed, 35 passed | `test_two_sided_bootstrap_p_by_hand` |
| V11 `verify` does not flag symlinks | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 48 passed | `test_verify_flags_a_symlink_even_when_its_target_has_the_right_bytes` |
| H1 EDA inputs keep holdout sessions | `src/ivf/data/process.py` | test_process, test_synthetic_pipeline, test_notebooks | 9 failed, 30 passed | `test_eda_inputs_stop_the_day_before_holdout_start` |
| H2 `eda_report` accepts holdout rows | `src/ivf/data/process.py` | test_process | 1 failed, 6 passed | `test_eda_report_refuses_a_holdout_session_or_row` |
| F1 a frozen `data_end` never re-validated | `src/ivf/data/check.py` | test_data_check, test_process | 5 failed, 22 passed | `test_a_frozen_data_end_fetched_on_the_same_new_york_day_fails` |
| F2 `eda` checks `data_end` as unfrozen | `src/ivf/data/process.py` | test_process | 1 failed, 6 passed | `test_check_and_eda_fail_when_the_frozen_data_end_is_not_settled` |
| F3 a same-day fetch counted as settled (`>` for `>=`) | `src/ivf/data/check.py` | test_data_check, test_process | 3 failed, 24 passed | `test_a_frozen_data_end_fetched_on_the_same_new_york_day_fails` |
| F4 `check` treats a frozen `data_end` as unfrozen | `src/ivf/data/check.py` | test_data_check, test_process | 1 failed, 26 passed | `test_check_and_eda_fail_when_the_frozen_data_end_is_not_settled` |
| G1 command line without the repo guard | `src/ivf/data/cli.py` | test_repo_guard | 14 failed, 4 passed | `test_every_command_refuses_a_data_root_inside_the_repo_with_exit_3` |
| G2-G6 `fetch_one`, `build_interim`, `write_synthetic_raw`, `run_check`, `run_eda` without it (one at a time) | `src/ivf/data/*.py` | test_repo_guard | 1 failed, 17 passed, each | `test_the_library_writers_refuse_too_not_only_the_command_line` |
| G7 shared guard without the `.git` check | `src/ivf/repo_guard.py` | test_repo_guard, test_export | 7 failed, 98 passed | `test_another_checkout_is_refused_by_its_git_entry` |
| G8 shared guard without the device-and-inode check | `src/ivf/repo_guard.py` | test_repo_guard, test_export | 21 failed, 84 passed | `test_every_command_refuses_a_data_root_inside_the_repo_with_exit_3` |
| U1 ADF/KPSS on the series with gaps dropped | `src/ivf/eda.py` | test_eda | 1 failed, 36 passed | `test_unit_root_tests_use_the_longest_gap_free_segment_and_never_join_across_a_gap` |
| T1 VIX1Y tenor 365 | `configs/iv_indices.yaml` | test_protocol_calendar, test_export | 2 failed, 117 passed | `test_the_registries_list_five_tenors_in_order_plus_vvix_and_the_sp500` |
| Y1 yfinance: `hide_exceptions` left on | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 49 passed | `test_yfinance_is_asked_to_raise_on_failure_without_the_deprecated_keyword` |
| Y2 yfinance: `raise_errors=True` passed | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 49 passed | `test_yfinance_is_asked_to_raise_on_failure_without_the_deprecated_keyword` |
| Y3 yfinance: the setting not restored | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 49 passed | `test_yfinance_is_asked_to_raise_on_failure_without_the_deprecated_keyword` |

38 mutants: 36 caught, 2 equivalent (V1h, V2b). The D4.4 and D4.6 calendar figures are now under
test too (`test_calendar_matches_the_session_figures_of_adr_0001_4_4`,
`test_the_holdout_origins_adr_0001_4_6_names`); before, only D4.5's were. The scan tool is not a
test (CI has no raw data). Its falsifier: run on review A's pre-fix copy of the repo, the final
version flags 30 lines in 196 files: this walkthrough's byte-total line, five lines of
`tests/test_data_fetch.py` (the two VIX rows, their `printf` comment and two S&P 500 rows), 22
lines of `tests/test_data_interim.py`, `tests/test_schemas.py:144` (M0) and its mention in M0
review A's ledger. On the fixed tree it prints `clean` (§7.5).

**After review B's fixes (amended M1, review B).** Every new or changed test was checked the same
way: a fresh scratch copy (`rsync` without `.venv` and `node_modules`), one mutant at a time,
`PYTHONPATH` at the copy, each file restored and its SHA-256 compared afterwards; the copy matched
the final tree file for file. 104 mutants: 34 for E6, the half-life, the repo guard, the raw store
and the freeze record (prefixes B, H, G, F, R, N, V); 32 for the network guard (NG), 30 of them
re-run on a fresh copy with the same counts, and two (`getaddrinfo` patch removed, guard never
installed) taken from the first run only, because without them a test reaches the real network; 38
for the scan (SC). **103 caught, 1 equivalent** (NG `tryfirst`: the conftest's hook wrappers already
run outermost). Two notes. G1's `raw` case fails only because the stand-in checkout has no
configs; with them, the write-time check refuses it anyway (exit 3, nothing written), which the
deeper-symlink test shows: that is the second layer doing its job, so G1 counts as caught by its
`eda`, `interim` and `processed` cases. And V1h (`create_connection` without its own patch) from
the table above is no longer equivalent: a new test lifts the `getaddrinfo` and `connect` patches
for a moment, and it fails. R1 above ("store: os.link -> os.replace") now lives in
`repo_guard.write_file`; it is G13 below.

| Mutation | File | Test files run | Result | A failing test |
|---|---|---|---|---|
| B1 E6 block rule -> 1 (review-A iid design) | `src/ivf/eda.py` | test_eda | 1 failed, 46 passed | `test_e6_size_under_garch_shocks_with_no_level_effect_stays_near_5_percent` |
| B2 E6 block rule on the shocks, not abs(shocks) | `src/ivf/eda.py` | test_eda | 2 failed, 45 passed | `test_e6_block_length_is_1_for_iid_shocks_and_long_for_clustered_ones` |
| B3 circular blocks without the wrap (clipped) | `src/ivf/eda.py` | test_eda | 1 failed, 46 passed | `test_circular_blocks_by_hand` |
| B4 blocks of 1 drawn per value (iid inside the helper) | `src/ivf/eda.py` | test_eda | 2 failed, 45 passed | `test_circular_blocks_by_hand` |
| B5 block < 1 accepted | `src/ivf/eda.py` | test_eda | 1 failed, 46 passed | `test_e6_block_must_be_positive` |
| B6 constant-size guard removed | `src/ivf/eda.py` | test_eda | 1 failed, 46 passed | `test_e6_block_length_is_1_for_iid_shocks_and_long_for_clustered_ones` |
| B7 report block 0 (vol_of_vol drops the block) | `src/ivf/eda.py` | test_synthetic_pipeline | 1 failed, 12 passed | `test_the_eda_on_synthetic_files_recovers_the_known_truth` |
| H1 half-life bound nan for phi <= 0 (first version) | `src/ivf/eda.py` | test_eda | 3 failed, 44 passed | `test_a_confidence_interval_end_maps_to_a_half_life_bound_even_outside_zero_one` |
| H2 CI mapped with half_life_of | `src/ivf/eda.py` | test_eda | 1 failed, 46 passed | `test_hc1_standard_error_and_95_percent_interval_by_hand` |
| G1 root check without DATA_SUBFOLDERS | `src/ivf/repo_guard.py` | test_repo_guard | 5 failed, 31 passed | `test_a_data_subfolder_symlinked_into_the_repo_exits_3_and_writes_nothing` |
| G2 write_file: no check before makedirs | `src/ivf/repo_guard.py` | test_repo_guard | 4 failed, 32 passed | `test_a_deeper_symlink_is_caught_at_write_time_and_the_command_exits_3` |
| G3 write_file: no check of the folder held open | `src/ivf/repo_guard.py` | test_repo_guard | 1 failed, 35 passed | `test_a_folder_swapped_for_a_link_after_the_check_is_caught_before_the_write` |
| G4 write_file: both checks off (the review-A design) | `src/ivf/repo_guard.py` | test_repo_guard | 10 failed, 26 passed | `test_a_deeper_symlink_is_caught_at_write_time_and_the_command_exits_3` |
| G5 no folder fsync after replace/link | `src/ivf/repo_guard.py` | test_repo_guard | 1 failed, 35 passed | `test_write_file_flushes_the_folder_after_the_new_name_appears` |
| G6 append follows a symlink | `src/ivf/repo_guard.py` | test_repo_guard | 1 failed, 35 passed | `test_appending_to_a_log_that_is_a_symlink_is_refused` |
| G7 processed parquet bypasses write_file | `src/ivf/data/process.py` | test_repo_guard | 1 failed, 35 passed | `test_every_writer_checks_the_folder_it_writes_into_at_write_time` |
| G8 interim parquet bypasses write_file | `src/ivf/data/build_interim.py` | test_repo_guard | 1 failed, 35 passed | `test_every_writer_checks_the_folder_it_writes_into_at_write_time` |
| G9 raw snapshot bypasses write_file | `src/ivf/data/raw.py` | test_repo_guard | 3 failed, 33 passed | `test_a_deeper_symlink_is_caught_at_write_time_and_the_command_exits_3` |
| G10 real export writes with outside_repo=False | `tools/export_app_data.py` | test_repo_guard | 1 failed, 35 passed | `test_every_writer_checks_the_folder_it_writes_into_at_write_time` |
| G11 CLI lets a write-time refusal escape | `src/ivf/data/cli.py` | test_repo_guard | 2 failed, 34 passed | `test_a_deeper_symlink_is_caught_at_write_time_and_the_command_exits_3` |
| G12 fetch loop turns a refusal into exit 5 | `src/ivf/data/cli.py` | test_repo_guard | 1 failed, 35 passed | `test_a_write_time_refusal_in_the_real_fetch_loop_exits_3_not_5` |
| F1 fetch does not check the log before storing | `src/ivf/data/fetch.py` | test_data_fetch | 1 failed, 51 passed | `test_a_torn_log_stops_a_forced_fetch_before_the_download_and_before_anything_is_stored` |
| F2 verify accepts an orphan | `src/ivf/data/raw.py` | test_data_fetch | 2 failed, 50 passed | `test_verify_names_the_orphan_a_crash_between_storing_and_logging_leaves` |
| F3 verify: new temporary names are problems | `src/ivf/data/raw.py` | test_data_fetch | 1 failed, 51 passed | `test_verify_only_warns_about_a_leftover_temporary_file` |
| R1 no freeze record check | `src/ivf/protocol.py` | test_protocol_calendar, test_data_check | 5 failed, 55 passed | `test_moving_a_frozen_data_end_back_by_hand_is_a_violation` |
| R2 record hash not compared | `src/ivf/protocol.py` | test_protocol_calendar | 1 failed, 38 passed | `test_another_edit_after_the_freeze_is_a_violation_too` |
| R3 freeze writes no record | `src/ivf/protocol.py` | test_protocol_calendar | 4 failed, 35 passed | `test_freezing_writes_a_record_of_the_value_and_the_hash` |
| R4 freeze again after data_end removed | `src/ivf/protocol.py` | test_protocol_calendar | 1 failed, 38 passed | `test_removing_a_frozen_data_end_is_a_violation_and_blocks_a_new_freeze` |
| R5 eda checks the data only, not the record | `src/ivf/data/process.py` | test_data_check | 1 failed, 20 passed | `test_moving_the_frozen_data_end_back_to_a_settled_session_fails_check_and_eda` |
| R6 check command ignores the record | `src/ivf/data/check.py` | test_data_check | 1 failed, 20 passed | `test_moving_the_frozen_data_end_back_to_a_settled_session_fails_check_and_eda` |
| N1 notebook carries on after VIOLATED | `tools/build_notebooks.py` | test_notebooks | 3 failed, 17 passed | `test_m1_notebook_stops_when_check_reports_violated` |
| G13 new-file mode replaces instead of linking | `src/ivf/repo_guard.py` | test_data_fetch | 2 failed, 50 passed | `test_identical_bytes_are_stored_once_but_every_fetch_is_logged` |
| F4 torn last line accepted | `src/ivf/data/raw.py` | test_data_fetch | 2 failed, 50 passed | `test_a_torn_last_line_blocks_the_next_append` |
| V1 E6 verdict prints no instead of not shown | `src/ivf/data/process.py` | test_synthetic_pipeline | 1 failed, 12 passed | `test_every_command_succeeds_on_synthetic_data` |
| NG guard: no sendmsg patch | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_tcp_fast_open_through_sendmsg_is_blocked` |
| NG guard: no connect patch | `tests/conftest.py` | test_network_guard | 3 failed, 20 passed | `test_create_connection_is_blocked_by_its_own_patch` |
| NG guard: no connect_ex patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_socket_connect_ex_is_blocked` |
| NG guard: no sendto patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_udp_sendto_is_blocked` |
| NG guard: no create_connection patch | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_a_reference_taken_before_the_guard_is_still_refused` |
| NG guard: no getaddrinfo patch | `tests/conftest.py` | test_network_guard | 5 failed, 18 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: no gethostbyname patch | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_removing_the_guard_restores_the_real_functions` |
| NG guard: no gethostbyname_ex patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_the_older_dns_functions_are_blocked` |
| NG guard: no gethostbyaddr patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_the_older_dns_functions_are_blocked` |
| NG guard: no getnameinfo patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_the_older_dns_functions_are_blocked` |
| NG guard: no Curl.perform patch | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_curl_cffi_the_yfinance_client_is_blocked` |
| NG guard: no AsyncCurl.add_handle patch | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_curl_cffi_async_requests_are_blocked` |
| NG guard: family check gone (AF_UNIX refused too) | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_unix_sockets_still_work` |
| NG guard: makereport does not charge | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: collect report does not charge | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_while_collecting_is_a_collection_error` |
| NG guard: sessionfinish keeps exit 0 | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_outside_any_test_fails_the_session` |
| NG guard: no terminal summary section | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_outside_any_test_fails_the_session` |
| NG guard: refusal not queued | `tests/conftest.py` | test_network_guard | 4 failed, 19 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: expect_blocked ignored | `tests/conftest.py` | test_network_guard | 18 failed, 5 passed | `test_a_reference_taken_before_the_guard_is_still_refused` |
| NG guard: expect_blocked: no 'nothing refused' check | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: no setup context | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: no call context | `tests/conftest.py` | test_network_guard | 2 failed, 21 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: no teardown context | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: no collect context | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_while_collecting_is_a_collection_error` |
| NG guard: wasxfail kept | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: failed report overwritten | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: failed report not charged | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_fails_the_test_in_every_phase` |
| NG guard: during() does not restore | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_outside_any_test_fails_the_session` |
| NG guard: take() ignores node/phase | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_a_swallowed_refusal_outside_any_test_fails_the_session` |
| NG guard: makereport not tryfirst | `tests/conftest.py` | test_network_guard | **23 passed: equivalent.** conftest hook wrappers already run outermost | none |
| NG guard: guard never installed | `tests/conftest.py` | test_network_guard | 21 failed, 2 passed | `test_a_reference_taken_before_the_guard_is_still_refused` |
| NG guard: remove() leaves inherited patches | `tests/conftest.py` | test_network_guard | 1 failed, 22 passed | `test_removing_the_guard_restores_the_real_functions` |
| SC scan: date: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 16 failed, 28 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: MM/DD/YYYY family | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 4 failed, 40 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: day-first reading | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: compact YYYYMMDD | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: (Y, M, D) calls | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: JS months from 0 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: textual month first | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 3 failed, 41 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: textual day first | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 3 failed, 41 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: epoch | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 3 failed, 41 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: date: epoch ms unit | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: window: FAR_LINES 12 -> 3 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_pretty_json_record_puts_the_value_lines_away_from_its_date` |
| SC scan: window: FAR_LINES 12 -> 13 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_date_out_of_reach_is_not_compared` |
| SC scan: window: NEAR_LINES 3 -> 0 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_every_date_within_three_lines_counts_not_only_the_nearest` |
| SC scan: window: all dates within FAR, not nearest | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_date_out_of_reach_is_not_compared` |
| SC scan: window: long lines lend dates | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_date_out_of_reach_is_not_compared` |
| SC scan: run: RUN_MIN 3 -> 4 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 4 failed, 40 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: run: RUN_MIN 3 -> 2 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 4 failed, 40 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: run: oldest first only | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_three_consecutive_closes_without_dates_are_flagged` |
| SC scan: run: stride 1 only | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: run: no in-between condition | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: run: unmatched between always allowed | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_one_decimal_runs_need_five_sessions` |
| SC scan: run: unmatched between never allowed | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: run: rule 6 not reported | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 5 failed, 39 passed | `test_numbers_between_two_sessions_must_belong_to_them` |
| SC scan: 1 decimal: no 1-decimal readings | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 4 failed, 40 passed | `test_a_dated_value_is_flagged_in_every_number_spelling` |
| SC scan: 1 decimal: other lines allowed | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_one_decimal_value_needs_its_date_on_the_same_line` |
| SC scan: 1 decimal: RUN_MIN_ONE_DECIMAL 5 -> 4 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_one_decimal_runs_need_five_sessions` |
| SC scan: 1 decimal: RUN_MIN_ONE_DECIMAL 5 -> 6 | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_one_decimal_runs_need_five_sessions` |
| SC scan: /100: no rescaled reading | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_a_dated_value_is_flagged_in_every_number_spelling` |
| SC scan: number: no thousands separators | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_a_dated_value_is_flagged_in_every_number_spelling` |
| SC scan: number: no leading point | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_a_dated_value_is_flagged_in_every_number_spelling` |
| SC scan: number: full stop glued on | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_a_close_next_to_its_date_is_flagged_in_every_date_spelling` |
| SC scan: number: half-open rounding interval | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 2 failed, 42 passed | `test_a_dated_value_is_flagged_in_every_number_spelling` |
| SC scan: output: prints the line's text | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_the_command_prints_only_file_and_line_and_never_a_value` |
| SC scan: listing: ignored files scanned | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_the_command_prints_only_file_and_line_and_never_a_value` |
| SC scan: rule 1 off | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_rule_1_a_line_found_verbatim_in_a_raw_file` |
| SC scan: rule 2 off | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_rule_2_a_whole_raw_row_inside_a_bytes_literal` |
| SC scan: rule 4 off | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_rule_4_a_raw_file_size_or_total_with_separators` |
| SC scan: rule 5 off | `tools/licensing/scan_real_rows.py` | test_scan_real_rows | 1 failed, 43 passed | `test_rule_5_a_raw_snapshot_hash` |

### 7.5 Gates (run before every commit)

Run from the repo root on 2026-10-29, after the review B fixes (amended M1, review B). They
replace the review-A run's table, whose differences were: 635 tests, the scan clean on 204 files,
the diagrams unchanged, and no `check --freeze-data-end` (there was no freeze record yet). The
first run's table differed further: 578 tests, no diagram or scan row, and a `fetch --force` that
stored nothing new and logged 7 unchanged fetches (14 log lines for 7 files).

| Command | A pass means | Result |
|---|---|---|
| `uv run pytest -q` | All Python tests pass | `730 passed in 62.79s` (635 after review A, 578 before it; the count includes 2 tests in `tests/test_diagrams.py`, which belong to the diagram kit, not to M1) |
| `uv run python tools/build_notebooks.py --check` | Both notebooks equal their specs | `ok: 2 notebook(s) in <repo>/notebooks match their specs` |
| `uv run python tools/export_app_data.py --mode synthetic --check` | The committed fixture is current | `ok: 1 file(s) in <repo>/app/public/data/synthetic match a fresh synthetic export` |
| `uv lock --check` | The lock matches `pyproject.toml` | `Resolved 142 packages` |
| `uv run --group diagrams python tools/diagrams/build.py docs/architecture/specs --out docs/architecture/out` then the same with `--check --no-png` | The rendered diagrams match their specs | The specs gained review B's changes (write_file, the subfolder check, the freeze record, the scan's run rule); `out/` rebuilt (10 files changed); then `ok: 5 spec(s) up to date` |
| `uv run python tools/licensing/scan_real_rows.py` | No real row, dated value (any listed date spelling), run of consecutive raw values, raw file size or raw hash in any file git would commit (local only: it needs the raw data) | `clean: 0 lines flagged in 210 text file(s) against 7 raw snapshot(s)` (exit 0). Before the reworded comment in `tests/test_features.py`, the extended scan flagged that one line: made-up regime-edge inputs (an edge minus 0.01) six lines below a dated line, a coincidence |
| `uv run python -m ivf.data check --freeze-data-end` (once), then `fetch` (no `--force`), `build-interim`, `verify`, `check`, `eda` | The real pipeline end to end, inside `--max-age`, with the freeze recorded | exit 0 each. The first command printed `recorded the frozen data_end 2026-10-28 ... in protocol.freeze.json`; then 7 `skipped` (nothing downloaded); `verify` ok for 7 raw files; `gap policy: OK`, no frozen-`data_end` or freeze-record violation; `eda` prints 5 E-statements and the burn-in and dev half-lives, no holdout line. Counts and values: measured locally, not published (D2). |
| `uv run python tools/export_app_data.py --mode real` then `--mode real --check` | Real JSON written outside the repo, through `write_file`, and reproducible | exit 0; `ok: 2 file(s) ... match a fresh real export` |
| `uv run python tools/export_app_data.py --mode real --out-dir app/public/data/real` and `VOL_DATA_DIR=$PWD/localdata uv run python -m ivf.data fetch` | Both licensing guards refuse the repo | exit 3 each; neither folder exists afterwards |
| `git status --porcelain --ignored` | No raw, interim, processed or real JSON file in the repo | Ignored: `.venv/`, `.pytest_cache/`, the `__pycache__/` folders, `app/node_modules/`, `app/dist/`, `app/coverage/`, `graphify-out/` and `.learning/`. Untracked or modified: code, tests, configs (with the two freeze records), docs, probes, notebook 01, the diagram kit and its outputs. No raw, interim, processed, EDA or real JSON file |
| Every `file:line` link in this walkthrough and concept notes 04 to 06 | Each lands on what the text says | 111 distinct links, resolved by a script against the file text after the review-B changes moved lines, and each target line read: every one lands on the function or statement its text names |
| `cd app && npm run verify` | | Not run: M1 did not touch `app/` |

---

## 8. Answers to the predict-first questions

**P1. A handful.** Cboe's VIX file has no close on a few XNYS sessions, all in burn-in. The count
and the dates are `data_check` output, local only (D2). FRED VIXCLS lacks most of the same
sessions (checked locally), so the ADR's FRED fallback could not close them. Gap policy G1-G5
keeps them null.

**P2. Yes.** Since 2022 Cboe's file has VIX rows on NYSE holidays such as Memorial Day and
Thanksgiving, and one on an earlier unscheduled closure. They are not sessions of 𝒯, so the
pipeline counts them (`not_a_session`) and never reads them (G4).

**P3. 34.31 and 68.97 sessions.** `ln 0.5 / ln 0.98 = -0.693147 / -0.020203 = 34.31`;
`ln 0.5 / ln 0.99 = -0.693147 / -0.010050 = 68.97`. One point of `phi` doubles the half-life.

**P4. About 31%,** six times the nominal 5% (100 seeds, section 7.3). For `phi = 0.5` it was 7%.
KPSS's automatic bandwidth is too short for so much persistence.

**P5. 0.4 kept, 0.3 after `dropna()`.** With the gap: pairs `(2,1)` and `(5,4)`, sum 4, over a
lag-0 sum of 10. After `dropna()`, 4 is paired with 2, a two-session move.

**P6. Yes, falsely.** With exponential shocks the next raw move had rank correlation about +0.11
with the level, from the mean-reversion pull, though the shock size never depended on the level.
The next AR(1) *shock* had abs(rho) <= 0.0095. That is why E6 uses the shock (section 7.3). And a
small `rho` is not the whole story: its textbook p-value still rejected on 30% of such paths,
which is why the p-value comes from a bootstrap (amended M1, review A). With clustered shocks, as
VIX's are, it must also resample them in blocks, and then it can say "yes" or "not shown" but
not "no" (amended M1, review B; section 7.3).

**P7. Yes.** Variance weights high-volatility days far more than vol points do. If realised
volatility is more dispersed than implied, a zero variance gap leaves a positive vol-point gap
(Jensen; concept 06 §2).

---

## 9. Concepts used

- [04 · Volatility clustering, the ACF and Ljung-Box](../concepts/04-volatility-clustering-and-the-acf.md):
  the ACF formula, pairwise-complete lags, Ljung-Box, the GARCH(1,1) ACF of `r²`.
- [05 · Persistence, the AR(1) half-life and unit roots](../concepts/05-persistence-half-life-and-unit-roots.md):
  half-life, its interval, HC1, Kendall bias, ADF against KPSS.
- [06 · The variance risk premium and vol-of-vol](../concepts/06-variance-risk-premium-and-vol-of-vol.md):
  the ex-post gap in two units, Jensen, the E6 measure.
- From M0: [01 · Targets and annualisation](../concepts/01-forecast-targets-and-annualisation.md)
  (`rv21`, units), [02 · Walk-forward and look-ahead](../concepts/02-walk-forward-evaluation-and-look-ahead.md)
  (rule L1, the truncation test), [03 · Reproducibility and data contracts](../concepts/03-reproducibility-and-data-contracts.md)
  (content addressing, `protocol_hash`, `run_id`).

---

## 10. Explain it back

Answer in your own words, naming the function and the line. No answers here.

**E1.** Walk through `uv run python -m ivf.data fetch` run twice within an hour, then once with
`--force` when Cboe's file has not changed. What does each run do to `raw/` and to
`provenance.jsonl`, and which lines decide it?

**E2.** Why does `store_snapshot` use `os.link` and not `exists()` followed by `rename`? Describe
the race the second version has.

**E3.** State gap policy G1 to G3. Why is a gap allowed 22 sessions before the first fold cutoff
but not 21? What would break in M2 if a core gap fell inside the holdout?

**E4.** How is `data_end` chosen, and why "settled"? What does `--freeze-data-end` refuse to do,
and why does ADR 0001 rule R1 care?

**E5.** Explain the truncation test in `tests/test_features.py`. Which mutation does it catch,
and what kind of look-ahead can it not catch?

**E6.** ADF rejects and KPSS rejects on ln VIX. What do you conclude, and why is KPSS the weaker
witness for a series this persistent? Give the measured rejection rate.

**E7.** Why does the E6 vol-of-vol measure use the next AR(1) shock and not the next raw move?
Give the mechanism and the measured numbers.

**E8.** Name three things in this milestone that keep real Cboe data out of the public repo, and
one thing that is only a rule, not a mechanism.

**E9.** The E6 statistic is centred near zero under iid skewed shocks, yet its Spearman p-value
rejected on 30% of paths. Where does the extra spread come from? Why does a bootstrap that refits
the AR(1) on each path fix the test, when fitting on one half and testing on the other does not?

**E10.** List every route by which a test process can reach the network, say which ones
`tests/conftest.py` closes, and name the ones it cannot. How would you show a route is closed?
Why is raising `NetworkBlocked` not enough on its own (amended M1, review B)?

**E11.** Review A's E6 bootstrap had the right size on iid shocks and 47% on GARCH shocks with no
level effect. Explain why, then explain what a circular block bootstrap keeps that iid draws lose,
how its block length is chosen, and why it is applied to `abs(shock)` and not the shock. What does
the 18% in §7.3 mean for how E6 is worded?

**E12.** `ROOT` is outside the repo but `ROOT/eda` is a symlink into it. Walk through what
`python -m ivf.data eda` does now, and name the two layers that stop it. Why does `write_file`
check the folder it holds open, not only the path it was given?

**E13.** Someone edits `data_end` in `configs/protocol.yaml` from 2026-10-28 to an earlier session
on which both closes exist and were settled. Which rule of `check` catches it, and why could none
of the data rules?

---

## 11. Decisions waiting on you

- **The M1 defaults** (O1 holdout 2024-07-01; D2 as proposed with O7 and O8 "no" and O9 the
  proposed definition; CPU torch). They are in effect; ADR 0001 stays Proposed until you confirm.
- **`data_end` = 2026-10-28.** Frozen. Changing it is a new protocol and, after a release, a new
  ADR (rule R1).
- **Protocol fields not yet set:** `lstm_members` (M4) and `cost_bps` (M6) are in the data model
  but not in `configs/protocol.yaml`. Adding them changes `protocol_hash`, so they should be set
  before any release of P.
- **The synthetic fixture** still uses the M0 stand-in protocol in `tools/export_app_data.py`.
  Switching it to `configs/protocol.synthetic.yaml` changes the committed manifest's
  `protocol_hash`; it is a separate change.
- **Real-data E-statements.** D2 would allow E1 to E7 (yes/no) and the half-life at 2 s.f. from a
  clean-tree run. M1 publishes none of them.
- **E6 is now "yes" or "not shown" (amended M1, review B).** ADR 0001's E6 row said "yes or no".
  The test that keeps its size under volatility clustering has 18% power against a strong level
  effect when the shocks cluster, so a non-rejection cannot be read as "no". The ADR row is amended;
  confirm the wording. A test with power under clustering needs a volatility model (a GARCH with
  the level as a regressor, tested by likelihood ratio), which fits milestone 2's GARCH work; it
  would be valid only as far as that model fits VIX's shocks.
- **E6 answers only up to `phi_hat` = 0.985 (amended M1, final check).** Above it the test's size
  was measured to fail (15% at 0.99, 44.5% at 0.995; §7.3), so E6 says "not shown" with that
  reason. Real ln VIX has a pre-holdout `phi_hat` between 0.98 and 0.99, close to the cutoff.
- **E6 needs at least 6,000 pairs (`eda.E6_MIN_N`; default decision 2026-10-29, yours to
  override).** Shorter samples were oversized inside the cutoff (24% "yes" at n = 2,000 with `phi`
  0.98); 4,000 passed the rule only on the boundary (10 of 200), so 6,000 was taken (§7.3).
- **Known limit: `eda` does not re-hash interim against raw** (M1 final check). It reads
  `interim/` as it finds it, so a hand-edited interim parquet passes `eda`; only `verify` checks the
  raw snapshots against their hashes, and nothing rebuilds interim to compare.
- **Unverified:** VIX's 16:15 ET close. (`tenor_days` is verified since review A: 9, 30, 93, 184,
  366.)
- **A real value in the M0 commit (review A).** The M0 commit `43ded87` holds one real VIX level,
  dated, in `tests/test_schemas.py`, and names it in `docs/validation/M0-review-A-code-numerics.md`.
  Both files now hold a made-up value. **Decided by the owner on 2026-10-29:** public history stays
  as it is (no rewrite, no force-push), so `43ded87` keeps that one value. The rule for every new
  commit is strict: no real vendor values or data files, made-up values in tests, and the real-data
  scan at 0 hits before each push. Recorded in ADR 0001 D2.
- **The first M1 run looked at the holdout (review A).** It printed and stored holdout statistics
  on real data before any model existed. The stored report has since been overwritten by the
  pre-holdout one (same `run_id`), but the terminal output was seen. No hyper-parameter exists yet,
  so `holdout_looks` stays 0; say if you want that look recorded.
- **Known limits of the network guard:** a subprocess started by a test (the pipeline and diagram
  tests start some) runs without it, and so does `_socket` used directly (`socket.SocketType`) or a
  C extension other than libcurl (amended M1, review B).
- **Diagrams (amended M1, review B).** This bullet used to list what the specs did not show after
  review A; the specs have since gained `ivf.repo_guard`, the scan, the pre-holdout EDA edge and the
  `sample` section, and in the review-B round `write_file`, the subfolder check, the freeze record
  and the scan's run rule. `out/` was rebuilt with the vendored kit. The `--check` gate compares
  outputs with specs, not specs with code, so this stays a reading task at each review.
