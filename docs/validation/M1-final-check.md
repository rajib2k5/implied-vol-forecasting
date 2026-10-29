# M1 final check

An adversarial check of the M1 tree after review B's fixes (uncommitted, on top of `43ded87`),
2026-10-29. It attacked six claims; the orchestrator reproduced claim 1. Every number below comes
from simulated data or planted stand-ins; no real value appears here.

| # | Claim | Verdict | Evidence | Fix |
|---|---|---|---|---|
| 1 | E6's block bootstrap keeps its size on every null | **REFUTED** (persistent skewed clustering) | Skewed GARCH(0.03, 0.965) shocks, no level effect, `phi` = 0.995: "yes" on 47 of 100 paths (re-run by the orchestrator). The fix round's 200 fresh paths each, n = 9,000, B = 999: rejects 0.0% at `phi` 0.98, 2.0% at 0.985, **15.0%** at 0.99, **44.5%** at 0.995, **34.5%** with lognormal innovations at 0.995. The block (mean about 245, near the cap) is shorter than the level's memory | `eda.E6_PHI_MAX` = 0.985 by the pre-registered rule; above it `eda.e6_verdict` says "not shown" with the reason "persistence above the size-checked range", and `rho` and p are still reported. "Yes" through the cutoff: 0 of 200 on all five nulls; power at `phi` 0.98 unchanged (285, 299 and 54 of 300). Walkthrough §7.3 |
| 2 | The repo guard stops every write into the checkout through a link | **REFUTED** (hard link; symlinks survive) | `raw/provenance.jsonl` hard-linked to a file in the checkout: 13 lines appended to it. `O_NOFOLLOW` stops a symlink, not a hard link | `repo_guard.write_file(how="append")` checks the open descriptor (`fstat`) and refuses a file that is not regular or has more than one name, before writing; exit 3 through the CLI. `replace` and `new` write a fresh temporary file, so they never write through an existing inode (tested) |
| 3 | The `data_end` freeze record catches a hand edit | **SURVIVES** | Editing both the protocol and its freeze record on purpose passes, as documented: a tripwire, not a lock (ADR 0001; walkthrough) | None |
| 4 | Only `expect_blocked` excuses a network refusal, and only in tests of the guard | **REFUTED** (misused `expect_blocked`; all routes survive) | A test wrapping production code in `expect_blocked()` and swallowing the refusal passed. Every blocked route still held | `expect_blocked` works only in `test_network_guard.py` and `test_data_fetch.py` (`EXPECT_BLOCKED_FILES`); elsewhere it records the misuse as a refusal and fails the test, even if the test catches what it raises (tested in an inner session) |
| 5 | The real-data scan catches the planted shapes | **REFUTED, not fixed** (3 unlisted shapes; no-print survives) | Missed: a decimal-comma CSV (`DD.MM.YYYY;16,94`), Excel serial dates with a value, every k-th session's close. The scanner printed no value | Not fixed: the scanner is no longer a push gate after the owner's 2026-10-29 data decision. A decimal-comma reading was added before that decision and kept, not extended; the other two are listed as not caught (`tools/licensing/README.md`) |
| 6 | No document says E6 answers "no" | **REFUTED** (one walkthrough sentence) | The walkthrough's synthetic EDA paragraph said `so E6 says "no"` | Reworded: review A's version of E6 said "no" there. A grep of the walkthrough, ADR, concepts 04 to 06, the notebook builder and the README found no other; a test comment in `tests/test_eda.py` said the same and was reworded too |

**The fix round's own finding, fixed.** Inside the cutoff, E6 was still oversized on short samples:
skewed GARCH(0.03, 0.965), `phi` 0.98, B = 99, "yes" on 39 of 100 paths at n = 1,000 and 24 of 100
at n = 2,000. Fix (default decision, 2026-10-29): `eda.E6_MIN_N` = 6,000 pairs. The rule (the smallest of
4,000, 6,000 and 8,000 at which "yes" was at most 10 of 200 fresh paths, B = 999, on skewed GARCH at
`phi` 0.98 and 0.985 and on iid skewed shocks) gave 4,000, but only on the boundary (10, 5 and 0,
on 3,999 pairs), so the orchestrator took 6,000, where the counts were 1, 0 and 3 (walkthrough
§7.3 has the table). Below it the verdict is "not shown", "sample below the size-checked length".
The real verdict word is unchanged.

**How the defects went undetected.** Claim 1: review B's ten nulls all had a level `phi` of 0.98
(one of 0.995, with iid shocks), so no null combined persistent volatility with a persistent
level, and the real `phi_hat` (between 0.98 and 0.99) sat just outside what was measured. Claim 2:
the symlink test pinned the one link type the code handled. Claim 4: the guard's tests only used
`expect_blocked` correctly; nothing tested a wrong use.

**Still open.**

- `E6_MIN_N` = 6,000 is one step above what the rule picked: 4,000 sat on the boundary (10 of 200).
- `eda` does not re-hash interim against raw, so a hand-edited interim parquet passes `eda`
  (walkthrough §11).
- A longer E6 block was not measured.
