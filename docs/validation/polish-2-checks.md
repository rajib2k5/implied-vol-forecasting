# Polish-2: checks

Polish-2 adds:
- the 3D error surface and the regime ribbons (schema 1.2.0);
- the raw-file gate (`tools/licensing/check_raw_files.py`);
- the O10 probe;
- the retirement of `error_surface.json`;
- the milestone 07 walkthrough, concept 16 and checkpoint 07.

Run on 2026-11-09.

**Route:**
- **App views, export and schema:** router row 2, routine code with tests. Gates, plus one hand mutant per new or changed test.
- **Raw-file gate:** router row 11, a push hook. The gate is exercised once on a constructed bad input.
- **O10:** row 4; see [its ledger](O10-lstm-equivariance.md).
- **Docs:** row 10. A verifier re-measures every number, path and command.
- No adversarial pass was routed. No number from this work goes to a decision about money, and the trust boundary (no regime content in a real export) is pinned by tests that kill mutants.

## Gates (coordinator's run, working tree)

| Gate | Output |
|---|---|
| `uv run pytest -q` | 1035 passed (after the error_surface schema retirement; 1041 before, the 6 retired parametrisations removed) |
| `cd app && npm run verify` | 26 files, 383 tests passed; coverage 98.89 / 95.88 / 98.61 / 99.47; `guard:dist OK` |
| `export_app_data.py --mode synthetic --check` | ok, 5 files |
| `results_table.py --check` | ok |
| `build_notebooks.py --check` | ok, 7 notebooks |
| `uv lock --check` | 142 packages |
| diagrams `--check --no-png` | ok, 5 specs |
| `scan_real_rows.py` | clean, 0 lines |
| `check_raw_files.py` | 0 raw vendor files tracked or staged, 0 files over 50 MB (324 checked) |

## Mutants

- **App views and export:** the builder ran 45 mutants (16 Python, 29 app), and all were killed. A7, "whole-period ratio ignores the target", survived at first. The fixture's row order hid it. The fixture was reordered, and the mutant is now killed.
- **Coordinator's own mutants:**
  - `regimes_allowed` set to always return True: 4 tests failed. The real-mode guard is pinned.
  - The raw-file gate's staged-index read dropped: 6 tests failed.
- **Raw-file gate:** the builder ran 13 mutants. Two of them were equivalent, and both are explained in its report.
- **Raw-file gate on a constructed input:** in a scratch clone, a real snapshot was copied under a harmless name and staged. The gate named it by SHA-256 and exited 1.
- **Schema retirement:** putting `error_surface` back in `EXPECTED` made 6 tests fail.

## Verifier (row 10) on the new docs: what it refuted

| Claim | Where | Measured | Fix |
|---|---|---|---|
| "The `cc04558` message says the chain ran in 2,064.8 s" | 07-polish.md | The figure is in the **df9655e** message | Corrected |
| "`lstm_log` rows qualify for P under (D2 rule) R1" | O10 ledger, ADR O10, design page | **Wrong rule.** R1 is "one release per frozen `data_end`". Membership of P is D2's "Which rows" paragraph | Corrected in all three |
| "34 s" for the O10 probe | 07-polish.md | 28 s | "about 30 s" |
| pytest and export timings | 07-polish.md | They depend on load | Timings dropped |

Everything else in the four files agreed: every number, every `file:line` reference within 2 lines, every command, and all eight quiz answers. For each quiz item, the keyed option is the only correct one.

## Corrections the coordinator made outside the new docs

- **ADR 0001 §4.4:** the sentence said KPSS "picked d = 1 on 17-52%" of the series. That was wrong: it mixed a count (17 of 60) with a percentage. The measured range at phi = 0.98 is 24-52%, from the M2 review A ledger.
- **ADR 0001 D2 (strategies):** the M6 primary rule's band is 2 vol points, so the rule is not in P. The ADR now says so.
- **`docs/architecture/README.md`:** the line said "76 relations". That was the M0 count. The current count is 79: 60 declared in the spec plus 19 drawn from foreign-key columns.
