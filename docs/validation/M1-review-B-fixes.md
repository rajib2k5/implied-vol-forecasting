<!-- Copied in the review-B fix round from the reviewer's ledger in the session scratchpad. Absolute paths
are replaced by <session scratchpad>; nothing else in the review is changed. The fix round's notes follow
the review, under "Fix round". -->

# M1 review B: adversarial ledger (fixes made after review A)

Reviewer: adversarial-reviewer (Opus), 2026-10-29. Target: the uncommitted tree on top of 43ded87,
after the review-A fix round. Method: evidence-gated validation, **mode: single-context sequential
fallback**. The docs channel was the repo's own docs, the ADR and the installed source. No
external documentation was retrieved. The runtime channel was the probes below. The claims came from
the orchestrator, and no claims file was pre-registered before the probes ran. Treat the
convergence as "dual-confirmed (fallback mode)" at best.

Licensing: real data was read only. Real-data probes print booleans only (`probes/p_real.py`,
`probes/p_block.py real`, `probes/p_scan_ext.py`). Every number below is synthetic, a count or a
calendar figure.

Probes are in `<session scratchpad>/m1-review-b/probes/` (not committed). Mutations ran in a scratch copy (`<session scratchpad>/m1-review-b/repo/`) through
`probes/mut.py`, which restores each file and checks its SHA-256. The real repo was not modified.
The gates ran in the real repo and wrote only `.venv`/cache files.

| # | Claim | Verdict | Evidence (what ran, what it printed) | Proposed fix | Severity |
|---|---|---|---|---|---|
| 1 | E6 bootstrap: size ~5% (skewed), ~3% (normal), power >= 99%; the bootstrap is valid | **REFUTED: valid only for iid shocks.** The iid size and power numbers SURVIVE | Fresh seeds, n=9,000, B=999 (`probes/p_e6.py`):<br>- skewed iid: two-sided p<0.05 on **18/300 (6.0%)**, verdict "yes" 6/300<br>- normal iid: **17/300 (5.7%)** (the docs say 3.3%; the gap is seed noise)<br>- phi=0.995: 4/200<br>- skewed with 60 gaps: 8/200<br>- power k=0.5: 100/100; k=0.15: 100/100<br>**Symmetric GARCH(1,1) shocks, no level effect (population rho is 0 by symmetry, measured mean -0.0003):**<br>- α=.10, β=.88: two-sided rejection **79/200 (39.5%)**, verdict "yes" **36/200 (18%)** against a nominal 2.5%<br>- α=.05, β=.90: **42/200 (21%)**, "yes" 23/200<br>**Real data (booleans):**<br>- Ljung-Box on squared AR(1) shocks: p<1e-6 = True<br>- shipped E6 p<0.05 = True<br>- a moving-block residual bootstrap (L=100, which gave 0/100 under GARCH, so it is conservative): p<0.05 = **False**<br>So the real verdict depends on the iid assumption, and VIX shocks are not iid.<br>**What survives:**<br>- residuals are centred (eda.py:399)<br>- the paths use the fitted c and φ (eda.py:409)<br>- the statistic is refitted identically on each path (eda.py:412-413)<br>- p = 2·min((1+#)/(B+1)), two-sided as the docstring says<br>- the verdict (ρ>0 & p<0.05) is one-sided at 2.5%, and §7.3 says so<br>Mutants:<br>- scipy p-value back: 2 failed<br>- no refit: 1 failed<br>- one-tailed: 2 failed<br>- unmasked gaps: 50 passed. This one is equivalent, because the pairs are selected by `pairs[first:]` | Use a bootstrap that keeps conditional heteroskedasticity: a wild bootstrap (e* = ê·η with Rademacher η) or a calibrated block length. Measure its size under GARCH shocks. Add a GARCH H0 case to `e6_size_power.py` and to the test. Until then, state in concept 06 §4 and the walkthrough that E6's p-value assumes iid shocks and that VIX's shocks cluster. | **High.** The published E6 statement is decided by this test. |
| 2 | During tests only a subprocess or a direct `_socket` call reaches the network; deleting each patch fails a test | **REFUTED** | `probes/test_zz_bypass_b.py` ran under the session guard against a 127.0.0.1 listener:<br>- **UDP `sock.sendmsg(..., addr)`: the datagram was delivered**<br>- **TCP `sendmsg(..., MSG_FASTOPEN, addr)`: the listener accepted the connection**<br>- **`socket.gethostbyname`/`gethostbyname_ex`/`getnameinfo` are unpatched. strace during the test shows `connect(..., sin_port=htons(53))` to the resolver**<br>- `socket.SocketType(...).connect` connected. This is `_socket.socket` under a public name.<br>Blocked:<br>- `connect`, `connect_ex`, `sendto`<br>- asyncio `open_connection` and datagrams<br>- ssl<br>- urllib, requests<br>- curl_cffi sync, stream, websocket and async<br>- yfinance (it *swallowed* `NetworkBlocked` and logged "possibly delisted", contrary to the conftest docstring)<br>Deleting patches:<br>- `connect`: 2 failed<br>- `connect_ex`, `sendto`, `getaddrinfo`, `Curl.perform`, `AsyncCurl.add_handle`: 1 failed each<br>- **`create_connection`: 61 passed** (the equivalent V1h) | Patch `sendmsg` (AF_INET/AF_INET6), `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr` and `getnameinfo`. List `SocketType` among the limits. Fix the "no library swallows it" sentence. | Medium-low |
| 3 | Scanner: no false negative on review A's cases, and no blind spot for reformatted floats, JSON/Markdown/notebooks, yfinance, or split lines; never prints the match | **REFUTED** (blind spots). Review A's cases and the no-print rule SURVIVE | Synthetic store standing in for real (`probes/p_scan.py`), 30 rows per case:<br>Flagged:<br>- reformatted CSV: 30/30<br>- Markdown: 30<br>- notebook output: 30<br>- yfinance rows: 30<br>- a raw row split over 2 lines: 60/60<br>- JSON records: 26/30 (the 4 misses had a trailing zero dropped, which leaves 1 decimal)<br>- review A shapes (bytes literal, byte total with `,`/`_`, edited row): 5/5<br>**Missed (0/30 each):**<br>- `date(1990, 5, 25)` objects (the repo's own test idiom)<br>- pandas `to_json` epoch-ms keys<br>- a list of closes without dates<br>- pretty JSON with the value more than 3 lines from its date<br>- `19900525`<br>- `1990/05/25`<br>- "May 25, 1990"<br>- 1-decimal values<br>- /100 rescaling<br>Output never contained a planted value.<br>Real tree:<br>- `clean: 0 lines flagged in 204 text file(s)`<br>- my extended scan found 1 coincidence: a made-up comment value in a regime test, more than 3 lines from a date | Accept `date(Y, M, D)`, compact and slash dates. Widen NEAR_LINES for JSON. Flag runs of 5 or more consecutive raw closes without dates. The ADR residual should list these forms. | Medium-low (no leak found) |
| 4 | No writer can write under a data root inside the repo | **REFUTED** | `probes/p_guard.sh`, scratch copy:<br>- refused with rc 3: in-repo, symlinked ROOT, relative path, `..`, bind mount (`unshare -rm`), ROOT equal to the repo<br>- **ROOT outside but `ROOT/.../raw` a symlink into the repo: `fetch` rc 0, 8 files written into the checkout, then `build-interim` and `check` rc 0; `eda/` symlinked: `eda` rc 0, `eda_report.json` written into the checkout, untracked and NOT ignored.** The guard walks only from ROOT upward (repo_guard.py:89), never into the subfolders it writes.<br>- Check-then-write: swapping a symlinked ROOT after the guard put 8 files in the repo. This is a documented residual, but `cli.py:138` throws away the resolved path, so the window is the whole command.<br>- Case variants: not tested (ext4 is case-sensitive).<br>- Export: in-repo and symlinked `--out-dir` rc 3; outside rc 4 (no input); synthetic `--check` rc 0. Two M0 real-mode tests were replaced, not kept. | Check the final target folder (`layout.raw`, `.interim`, `.processed`, `.eda`) at write time, or refuse symlinks below ROOT. Pass the resolved root on. | Medium |
| 5 | The EDA gets no row on or after `holdout_start` | **SURVIVES** | `probes/p_holdout.py` (synthetic):<br>- every holdout row ×7 / ×0.1: report identical = True<br>- falsifier (the last 21 pre-holdout sessions changed): the spread changed = True<br>- a direct `eda_report` call with a holdout date: refused<br>- full-price features on pre-holdout dates: identical<br>- 21 origins without a forward rv21<br>Mutants:<br>- refusal removed: 1 failed<br>- `eda_inputs` keeps holdout rows: 3 failed<br>The notebook uses `eda_inputs` too. | none | — |
| 6 | V1h and V2b are equivalent | **SURVIVES** | V1h: 61 passed with the patch gone. The real `create_connection` calls the patched module-global `getaddrinfo`, so `NetworkBlocked` is still raised.<br>V2b: 24 extra interim rows after `data_end` with ×9 values. Feature, target, regime, eda_stat, calendar and eda_report were identical; only data_check differs (it counts unsettled rows, by design). | none | — |
| 7 | Real ADF/KPSS use the longest gap-free segment (G2 as amended) | **SURVIVES** | Real (booleans):<br>- gaps exist, all in burn-in<br>- every sample's segment is NaN-free and spans no gap<br>- pre_holdout n < n_nonmissing<br>- the segment covers all of dev<br>Mutant "drop NaNs": 1 failed. ADR G2 text matches. | none | — |
| 8 | Frozen `data_end` is re-validated on every `check`/`eda`; an unsettled date fails | **SURVIVES** (notes) | Synthetic CLI:<br>- fetched the same NY day: check rc 1, eda rc 1<br>- UTC next day but NY same day: rc 1/1<br>- NY next day 00:30: rc 0/0<br>- VIX alone unsettled: eda rc 1<br>- weekend: rc 1/1<br>- after the data: rc 1/1<br>Mutant `frozen=False` in run_eda: 1 failed.<br>Real: ok and frozen == candidate (booleans).<br>Notes:<br>- a hand edit to an *earlier* settled session passes (rc 0/0); only `protocol_hash` shows it<br>- the notebook prints "VIOLATED" and carries on | Make the notebook stop when `RESULT.ok` is False. | Low |
| 9 | Gates | **SURVIVES** | `uv run pytest -q`: `635 passed in 66.76s`, no warnings line.<br>`build_notebooks.py --check`: `ok: 2 notebook(s)`, rc 0.<br>Synthetic export `--check`: `ok: 1 file(s)`, rc 0.<br>`uv lock --check`: `Resolved 142 packages`, rc 0. | none | — |
| 10 | `file:line` links | **SURVIVES** | All 89 distinct links in the walkthrough and concepts 04-06 resolve, and the text matches the `#L` anchor. I checked 45 semantically (all the fix-agent ones: check 63/70/174/192/204/236/260, cli 121/138, fetch 79/109/202, process 55/91/106/118/158/242/256, raw 126, eda 179/183/200/255/328/347/361/417, repo_guard 57/81/99, conftest 40/99, export 65/186/201/235/409, tests 103/125, and others). Each lands on what the text describes. | none | — |
| 11 | ADR amendments and the "What review A found" table state the refutations plainly; the ledger has no real value and no absolute path | **Partly REFUTED** | Ledger:<br>- no `/home`, `/tmp` or `~/`<br>- scanner clean<br>- decimals against the real `eda_report.json`: 1 match, a calendar figure (coincidence)<br>- claim 12's se/CI match no real or synthetic sample (booleans)<br>Table:<br>- the network row says "a subprocess is still unguarded" as the limit, but claim 2 shows other routes<br>- it omits review A's refuted concept-06 sentence (fixed in concept 06:123-124 itself) | Correct the network row. Add the concept-06 item. | Low |
| 12 | Leftovers | **REFUTED** (minor) | - Half-life CI with φ_lo ≤ 0: white noise gives `half_life_ci = (nan, 0.245)`. Not fixed.<br>- `fetch --force` with a torn log: the snapshot is stored (1 file), then RawStoreError leaves an orphan. Not fixed.<br>- There is still no directory fsync after `os.link` (raw.py:126).<br>- pandas 3: no chained assignment found by grep, and the suite has no warnings.<br>- New tests: none greps source or asserts on a reimplementation. | Map φ_lo ≤ 0 to a half-life of 0. Check the torn line before `store_snapshot`. fsync the folder. | Low |

## New findings

- **N1 (high): E6's iid bootstrap is oversized under volatility clustering.** On real data the
  verdict changes with a clustering-preserving bootstrap (claim 1). This is the same class of error
  as review A's most important finding: the decision rule was tested only under the null its
  author imagined.
- **N2 (medium): the repo guard does not see a symlinked subfolder of ROOT** (claim 4).
- **N3 (low):** `tools/diagrams/VENDORED.md:3` holds an absolute local path with the username.
- **N4 (low):** yfinance swallows `NetworkBlocked` under the default `hide_exceptions`. A future
  test that forgets `network=` would pass silently, with empty data.

**Most important fix:** E6's bootstrap must respect shock clustering (a wild or block bootstrap,
with its size measured under GARCH) before any real E6 verdict is stated.

**Delete:** nothing.

---

## Fix round (2026-10-29)

Every refutation above was measured again before it was acted on; none was wrong. The walkthrough's
"What review B found" names each fix; its §7.3 has the E6 tables and §7.4 the mutants.

| # | Re-measured | Fix | Evidence |
|---|---|---|---|
| 1 | The orchestrator re-ran `probes/p_e6b.py garchmild 60 51000`: two-sided p < 0.05 on 14/60, "yes" on 9/60, rho mean +0.0006. The fix round's own 300-path runs: review A's design rejected on 19.7% (0.05, 0.90) and 47.0% (0.10, 0.88) of GARCH paths with no level effect | Circular block bootstrap, block length by Politis-White on abs(shock) (`ivf.eda.e6_block_length`), still refitting the AR(1) | Size at most 5.7% on 10 nulls (0 to 0.3% on the clustered ones); power 95.0% (k = 0.5) and 99.7% (k = 0.15); **18.0%** for k = 0.5 on top of GARCH clustering, so E6 says "yes" or "not shown" (ADR 0001 E6 amended). `tools/probes/e6_size_power.py`, `e6_designs.py`, `e6_population_rho.py` |
| | **Proposed fix rejected:** a wild bootstrap | | Rademacher: 26.0% on iid skewed shocks, 7.3% on GARCH (0.10, 0.88), 100% on skewed GARCH; Mammen: 10.7%, 10.0%, 100% |
| 2 | Bypass probe re-run before the change: `sendmsg` (UDP and TCP fast open), `gethostbyname`, `SocketType` and `super().connect` bypassed | `sendmsg`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo` refused; every refusal recorded and charged to its test (or collection, or the session); `expect_blocked()` the only exemption | Probe after: those routes blocked, 0 queries to port 53 under strace; `SocketType` and `super().connect` still bypass (documented limit). 32 guard mutants: 31 caught, 1 equivalent |
| 3 | The 16 planted shapes re-run against the old scan: the same 9 missed | Every listed date spelling, rounding either way, /100, the nearest dated line up to 12 lines, rule 6 (runs of 3, or 5 at one decimal) | Every planted row of all 16 shapes flags, and of 24 of 25 added shapes; the 25th (a 1-decimal value off its date's line) stays out by design; 8 negative controls flag nothing; the real tree flagged one coincidence (a made-up comment), reworded. 38 mutants caught |
| 4 | With the write-time check off (mutant G4), 10 of the new tests write into the stand-in checkout and fail | Root check covers raw/, interim/, processed/, eda/, app-data/; `repo_guard.write_file` checks every folder at write time and writes through the folder it checked | Symlinked raw/ and eda/ (stand-in checkout and the real checkout, dangling) exit 3 and write nothing; 13 mutants (G1-G13) caught |
| 8 | Without the freeze record (mutants R5, R6) the hand edit to 2026-10-23 passes `eda` and `check` on the synthetic protocol | `configs/<protocol>.freeze.json`; `check.frozen_check`; the notebook raises | 6 mutants (R1-R6) and N1 caught; the real record written by `check --freeze-data-end` |
| 11 | The network row and the missing concept-06 row confirmed | Both fixed in the walkthrough | |
| 12 | Half-life (nan, 0.966) on the hand-worked series; the torn-log orphan; no folder fsync | `eda.half_life_bound` (an end <= 0 maps to 0); log checked before download and store; `verify` names orphans; `write_file` fsyncs the folder | Mutants H1, H2, F1-F4, G5, G13 caught |
| N3 | Confirmed | `tools/diagrams/VENDORED.md`: "a sibling checkout of the kit" plus the public URL | |
| (data model) | `eda_stat` listed `mean` and `sd`, which nothing writes (found while fixing N3's neighbours) | The list now names exactly what `process.eda_stat_rows` writes | |

**Incidents in the fix round, stated plainly.**

- While mutation-testing the network guard in a scratch copy, the mutants that removed the
  `getaddrinfo` patch or never installed the guard let tests make real DNS lookups
  (`cdn.cboe.com`, `example.invalid`), and probably one yfinance request to Yahoo, before the inner
  yfinance test was pointed at a refusing 127.0.0.1 proxy. Nothing was printed or stored. Those two
  mutants were not re-run in the verification pass for that reason.
- A draft of the guard's new `tests/conftest.py` was written into the real checkout instead of
  the scratch copy and stayed there for one to two minutes before it was taken out again; the file
  there now is the final version, tested as recorded above.
- Real-data runs showed only exit codes, counts, booleans and dates (their full output, which
  holds the half-lives, went to scratch files outside the repo and was not displayed). The E6 verdict on real data is
  a yes/no in a form D2 allows; like the other E-statements, M1 does not publish it.

**Owner decision (2026-10-29).** The real value review A found in the pushed M0 commit `43ded87`
stays in public history: no rewrite, no force-push. New commits carry no real vendor values or data
files, tests use made-up values, and the real-data scan must show 0 hits before every push. See ADR
0001 D2, "Owner decision on history".
