# History rewrite of 2026-11-09: old and new commit SHAs

The owner decided on 2026-11-09 to rewrite the history of this repository. The rewritten `main`
was pushed on 2026-11-10. GitHub `main` moved from `d4bb76f` to `df9655e`.

**Why:** the owner asked for the Claude attribution lines (`Co-Authored-By: Claude …`) to be
removed from the commit messages. Only the messages changed. Every commit got a new SHA because a
commit's SHA includes its message.

**Old SHAs on GitHub:** GitHub may still serve an old commit by its direct link for a while, until
it removes unreachable objects. Do not cite an old SHA. This file is the only tracked file that
holds them, so that an old link can be translated.

## Map

| Old SHA | New SHA | Commit |
|---|---|---|
| `43ded87b36b25e14a03aa23a06d8c05a806b8a0a` | `87dc6f81dcf0acd1886a339e59b8c653f4606801` | M0: scaffold, design, app shell and learning docs |
| `fecf88a3bd4e23754829a9bc05d5a97f4c5b95d0` | `76934ca8c8af3edaf400bde562cd053606803e51` | M1: data layer and EDA |
| `ab223f42b6689bcdcf156e995f3818b336da2608` | `1c47a477bffcf9c277cca7d5e9c7ec698f0ac7ef` | M2: baselines |
| `8318a1a7760c9e16568a311788c206b82ed59a7c` | `82a612f2ba118584e4a517df8b7c5a5fa3447637` | M3: GARCH done right |
| `a481168e61e06cc38aa1d4d432ffa385661bdc48` | `9e290a1e63c83e2ec5c744914ceb4afce62298b9` | M4: LSTM |
| `4da5b4501b0e7ee7408f599b1e1a1d7c7186f4b8` | `79e1aa12236f8416954ba78f616a6ba2088c38c8` | M5: evaluation |
| `7eeae14a163e9826eb9dcff20f2cd7dd9d77b6ae` | `47bea1a55d25eda843d73ba6d8de0cd890359167` | M6: backtest, PSR/DSR, guarded holdout look |
| `f903a01d62ba12b8978a6454fbcacad191e742c6` | `cc04558f81007422e874b439aa0d92291d2552b5` | Polish: app views, screenshots, README |
| `d4bb76fba3908ffd6d5592099afbdf224107f0be` | `df9655e614fc9949cb4564ae90576ae656029b64` | Synthetic app data exported from a clean store |

The map comes from the hub's backup,
`~/data/vol-projects/git-backups-2026-11-09/implied-vol-forecasting.commit-map` (not in the repo).

## What was measured (2026-11-10)

The local object store still holds the old commits, so each pair was compared with `git`:

- **Trees:** for all 9 pairs, `git rev-parse <old>^{tree}` equals `git rev-parse <new>^{tree}`.
  The files in every commit are byte-identical.
- **Metadata:** the author name, author email, author date and committer date are the same for
  all 9 pairs.
- **Messages:** each old message lost its `Co-Authored-By: Claude …` line (9 lines in all). Two
  messages also named an older commit and now name its new SHA: the M1 message and the synthetic
  data message.

## What this means for earlier decisions

ADR 0001 D2, "Owner decision on history" (2026-10-29), kept one real VIX level in the M0 commit's
tree and decided not to rewrite history for it. This rewrite changed messages only, so the M0
commit, now `87dc6f8`, still holds that value. The 2026-10-29 decision about the value is
unchanged.

## How the references were updated

Every short or full old SHA in a tracked file was replaced with the new SHA at the same length.
This includes the milestone walkthroughs, ADR 0001, the validation ledgers, `LEARNING_LOG.md`,
`README.md`, `docs/prfaq.md` and `app/src/lib/project.ts`. The exported app JSON names its source
commit in `source_commit`. It was re-exported from a clean checkout of a new commit, so it names a
new SHA. The quiz pins (`Code at:` in `.learning/milestones/*-checkpoint.md`) are git-ignored and
were updated in the same way.

The check, run from the repo root, prints nothing:

```bash
git grep -nE '43ded87|fecf88a|ab223f4|8318a1a|a481168|4da5b45|7eeae14|f903a01|d4bb76f' -- ':!docs/validation/history-rewrite-2026-11-09.md'
```
