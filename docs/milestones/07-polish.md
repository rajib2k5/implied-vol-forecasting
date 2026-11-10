# Milestone 7: polish

**Status (2026-11-09): built, tests green, not re-reviewed, Polish-2 not yet committed.** This
milestone has two parts. **Polish** is in two pushed commits, `cc04558` (app views, screenshots, README
results block, `tools/build_synthetic_store.py`, schema 1.1.0) and `df9655e` (the synthetic app data
exported from a store built at `cc04558`). **Polish-2** is the working tree today (`git status`: edits
to `src/ivf/app_results.py`, `src/ivf/contract.py`, the app and the schemas, plus new files under
`app/src/` and `tools/licensing/`); it is committed and pushed after this walkthrough, and the quiz
checkpoint pins that commit then. All data in this milestone is **synthetic**. No real VIX or ETP value
appears in this note.

Theory: [16, scale equivariance and publication](../concepts/16-scale-equivariance-and-publication.md).
Earlier theory it leans on: [03, reproducibility and data contracts](../concepts/03-reproducibility-and-data-contracts.md)
(section 9 on `run_id` and `--check`; the publication set P).

## 1. Predict before you look

Answer these before you read section 7. Write the answer down; the numbers are in the next sections.

1. The Evaluation view now draws a loss ratio to the random walk for each model, horizon and regime.
   Of the models other than `rw`, what share of regime cells do you expect to be below 1 (the model
   beats the random walk)? Where do you expect the worst ratio to sit: a calm regime or a stressed one?
2. A model has a ratio of 2 in one cell and 0.5 in another. If bar length were the ratio itself, drawn
   from 0, how would the two bars compare? What if it were `ln(ratio)` around a plane at 1?
3. If the same code built a **real** export, how many `regime_losses` rows and how many
   `regime_segments` would the files hold? Why?
4. The O10 probe multiplies the VIX by c = 0.8 and refits the real `LstmLog` and `LstmRv`. How many of
   its 16 compared entries change at 6 significant figures? Now give the probe a raw VIX input, and then
   a `V > 20` flag: how many change?

## 2. What we built

| Piece | File | What it does |
|---|---|---|
| Rebuild | `tools/build_synthetic_store.py` | the ten steps that build the synthetic dev store, in order, with wall times |
| Export check | `tools/export_app_data.py` | `--check` re-exports and compares; `run_id` and `source_run_id` are provenance, ignored |
| README tie | `tools/results_table.py` | prints the README results block from the committed JSON; `--check` fails if the README differs |
| Views | `app/src/pages/{ModelRace,Evaluation,Backtest,Eda}Page.tsx` | four synthetic views over the exported JSON |
| Screenshots | `tools/screenshots.py`, `docs/images/` | 16 PNGs: 7 sections plus the hero ribbon, each in 2 themes (`ls docs/images` shows 16 files) |
| Schema 1.2.0 | `src/ivf/contract.py`, `schemas/app-data/` | optional `evaluation.regime_losses`; `equity_curves.regime_segments` filled |
| Regime content | `src/ivf/app_results.py` | `regime_losses`, `regime_segments`, and the gate `regimes_allowed` |
| Error surface | `app/src/lib/errorSurface.ts`, `app/src/pages/ErrorSurface.tsx`, `app/src/pages/RibbonCanvas.tsx` | the pure geometry, the view, and the three.js canvas |
| Regime ribbons | `app/src/lib/XYChart.tsx` (bands), `app/src/lib/results.ts` (`regimeBands`) | shaded bands behind the equity curve |
| Raw-file gate | `tools/licensing/check_raw_files.py` | hashes what git holds and compares with the raw snapshots |
| O10 probe | `tools/licensing/p_tie_lstm.py` | V to cV on the real LSTM code; ledger `docs/validation/O10-lstm-equivariance.md` |
| Tests | `tests/test_app_results.py`, `tests/test_export.py`, `tests/test_check_raw_files.py`, `app/src/lib/errorSurface.test.ts`, others | listed in section 6 |

Changed in the docs: ADR 0001 O10 (decided) and `docs/architecture/data-model.md` section 5.4, which now
says `error_surface.json` was retired and never produced. The architecture files are edited by another
person's task and are only cited here.

## 3. Decisions

Each design below was chosen over the alternative in the third column.

| # | Decision | Rejected alternative | Would win if |
|---|---|---|---|
| P1 | **The app data is reproducible from a clean clone:** `build_synthetic_store.py` runs the ten steps, `export_app_data.py --mode synthetic --check` compares exactly (no float tolerance), and `source_run_id` joins `run_id` as provenance the check ignores | Pin the `source_run_id` in the committed JSON | A run id did not hash the commit. It does, so a rebuild at any other commit changed it and the check could never pass (the `cc04558` message says exactly this happened) |
| P2 | **The README's numbers are printed from the JSON** by `results_table.py`, between two markers; `--check` fails on a difference and the tool refuses any file that is not `data_mode: synthetic` | Paste numbers by hand | Never: a hand-pasted table drifts when the data is re-exported |
| P3 | **Regime content is synthetic only.** `regimes_allowed(store)` is true only for a store whose `data_mode` is exactly `synthetic`; a real payload gets no `regime_losses` key and an empty `regime_segments` | Export real regime cells with `n` and ratios, rounded | The regimes were not cut from VIX levels. They are, at fixed thresholds, so a cell or a dated run reveals vendor values (ADR 0001 D2) |
| P4 | **The mode is checked twice:** `load_store(..., data_mode=)` takes the caller's word, and the run's own `report_meta` and `backtest_meta` say which data they came from; a disagreement raises `StoreUnavailable` | Trusting the caller alone | The metas did not record the mode |
| P5 | **The error surface reads `evaluation.json`'s optional `regime_losses`;** the planned `error_surface.json` is retired (data-model.md 5.4) | A separate dataset with DM p-value cubes | Regime DM rows were computed. They are not, as built in M5 |
| P6 | **Bar length is `ln(ratio)` around a plane at ratio 1,** scaled so the largest `|ln ratio|` drawn is `height` long | Bar height = the ratio from 0 | A reader needed the ratio's size in absolute terms. On a ratio scale a doubling and a halving look unequal, and a 9x cell would flatten every cell near 1 |
| P7 | **Geometry is a pure function** (`surfaceLayout`), the canvas only uploads boxes; a table shows the same numbers when WebGL is missing or the renderer fails | Computing positions inside the canvas | jsdom could run WebGL. It cannot, so only a pure function is testable |
| P8 | **Ribbons are run-length segments** (`scheme`, `regime_id`, `start_idx`, `end_idx`) | One label per date per scheme | The view needed a label for every date. It draws spans, and segments are checkable for overlap and bounds |
| P9 | **Schema 1.2.0, a MINOR bump:** an optional field was added (ADR 0001, semver table: MINOR is "adding an optional field"). `regime_segments` already existed, filled with `[]`, so filling it changes no schema | MAJOR | A field had been removed, renamed or changed in meaning |
| P10 | **The raw-file gate compares content hashes** of what git holds with the raw snapshots, so a renamed or binary copy is caught | Match by file name, or only scan text for values (`scan_real_rows.py`) | The vendor files could only ever appear under their own names |
| P11 | **O10 is decided by measurement:** a metamorphic probe (V to cV) on the real LSTM code, with two falsifiers that must fire | Argue equivariance from the architecture alone | Never as the only evidence: the argument stays in the ledger as the explanation of the measurement |

## 4. Code walkthrough by `file:line`

Line numbers were measured with `grep -n` on 2026-11-09, on the working tree.

**Python, `src/ivf/app_results.py`**
- `:67` `REGIMES` and `:71` `SCHEMES`: M5's regime cells in display order; the schemes derived from them.
- `:124` `Store.data_mode`: the store says which data it was built from.
- `:143` `regimes_allowed`: `store.data_mode == "synthetic"`, nothing else passes.
- `:152` `load_store(layout, protocol, *, data_mode)`: keyword-only, so a caller cannot forget it;
  `:176` `stated`: the modes the run's metas state; any that differs from `data_mode` raises
  `StoreUnavailable`; `:179` the regime labels come from `build_regime_labels` on the pre-holdout inputs.
- `:298` `regime_losses`: dev, forecast `mean`, scheme not `all`; a ratio is the cell's loss over `rw`'s
  loss in the same cell; a suppressed cell keeps `n` with null losses.
- `:359` `evaluation_payload`: adds `regime_losses` only `if regimes_allowed(store)`; `:389`
  `check_evaluation`: a regime row needs a whole-period loss row for its model, and no cell repeats.
- `:417` `regime_segments`: each scheme's labels over the backtest dates, run-length encoded; a date
  with no label ends the run; `:486` the call, again behind `regimes_allowed`; `:499`
  `check_equity_curves`: segments inside the dates, in order, no overlap within a scheme.
- `src/ivf/contract.py:46` `SCHEMA_VERSION = "1.2.0"`.

**Tools**
- `tools/export_app_data.py:73` `PAYLOAD_PROVENANCE = ("run_id", "source_run_id")`; `:445`
  `_without_provenance`; `:502` `check` (exact comparison apart from provenance, no tolerance).
- `tools/build_synthetic_store.py:29` `STEPS`: the ten commands in order.
- `tools/results_table.py:70` `block`, `:119` `readme_block`, `:126` `main` (exit 0 ok, 1 the block
  differs, 2 a file is missing or not synthetic or the README has no block).
- `tools/licensing/check_raw_files.py:93` `raw_hashes` (SHA-256 of each non-empty snapshot's bytes),
  `:108` `_Blobs` (`git cat-file --batch`), `:151` `check`, `:160` the two sets, `HEAD:path` for tracked
  files and `:path` (the index blob) for staged ones, `:167` the comparison, `:135`
  `_ignored_raw_extension`.
- `tools/licensing/p_tie_lstm.py:44` `Mutant` (the two falsifiers), `:36` `settings` (the cheap config).

**App**
- `app/src/lib/results.ts:100` `DEFAULT_MODELS`, `:107` `REGIMES` (with the ribbon `shade`), `:148`
  `regimeLosses` (absent means `[]`), `:199` `regimeSegments` (parse-time checks), `:220` `regimeBands`.
- `app/src/lib/errorSurface.ts:32` `surfaceGrid` (a dense model x horizon x regime grid), `:61`
  `lowestIndex`, `:83` `DEFAULT_SURFACE`, `:133` `surfaceLayout`, `:146` where a scheme change adds a
  gap row, `:158` a null or non-positive ratio gets no box, `:162` `scale`, `:165` `length`, `:166` the
  `tall` test that decides whether the `minHeight` tile is used.
- `app/src/pages/ErrorSurface.tsx:9` the lazy import of the canvas, `:21` `SurfaceTable`, `:66`
  `ErrorSurface`, `:72` `detectWebGL`, `:79` the no-regime-rows message, `:120` the no-WebGL fallback,
  `:140` the `RenderBoundary` fallback. `EvaluationPage.tsx:90` mounts it.
- `app/src/pages/RibbonCanvas.tsx:79` `useColourTokens` (WebGL cannot read CSS variables, so the theme's
  `--series-N` values are read and re-read through a `MutationObserver`, `:86`), `:98`
  `ErrorSurfaceCanvas`, `:104` `created`, `:108` `onCreated`, `:129` `:135` `:140` the three label groups
  behind `created &&`.
- `app/src/lib/XYChart.tsx:29` `XYBands`, `:83` `half`, `:84` `bandAt`, `:161` the band `rect`.
  `BacktestPage.tsx:15` `hasRibbons`, `:16` the call to `regimeBands`.

## 5. Data flow

```
synthetic dev store  (ROOT/synthetic/processed/run_id=<id>/..., built by tools/build_synthetic_store.py)
   --app_results.load_store(data_mode="synthetic")-->  Store (+ regime labels, metas checked against data_mode)
   --regimes_allowed(store)?--> yes: regime_losses -> evaluation.json ; regime_segments -> equity_curves.json
                              > no (real): no regime_losses key ; regime_segments = []
   --tools/export_app_data.py--> app/public/data/synthetic/{manifest,model_race,evaluation,equity_curves,eda}.json
   --tools/results_table.py--> README results block      (--check ties the two)
   --parseEvaluation / parseEquityCurves--> surfaceGrid -> surfaceLayout -> ErrorSurfaceCanvas (WebGL) or SurfaceTable
                                            regimeBands -> XYChart bands
git (HEAD + index) --tools/licensing/check_raw_files.py--> SHA-256 vs the raw snapshots under ROOT/raw
```

## 6. Pitfalls and their guards

| Pitfall | Guard |
|---|---|
| **The drei `Html` labels.** The first label stayed empty when the labels mounted in the canvas's first commit (drei 10.7, fiber 9.8, React 19.3, seen in Chromium). The workaround mounts all labels after `onCreated` (`RibbonCanvas.tsx:104`, `:108`, `:129`-`:140`). **The root cause is not known**: it was not isolated to drei, fiber or React, and the workaround was checked by the screenshots, not by a unit test (jsdom has no WebGL) | The comment at `RibbonCanvas.tsx:102`-`:103` names the versions. If a dependency is upgraded, re-take the evaluation screenshot and look at the first label |
| A real export carrying regime cells | `test_a_store_that_is_not_synthetic_gets_no_regime_cells_and_no_ribbons` (`tests/test_app_results.py:185`); `test_real_export_holds_no_regime_cell_and_no_ribbon` (`tests/test_export.py:1068`), which also scans a synthetic export to prove the scan can find them |
| The mode taken on trust | `data_mode` is keyword-only and cross-checked with the run metas (`app_results.py:176`) |
| A ribbon that overlaps itself, or points outside the dates | `check_equity_curves` (`app_results.py:499`), mirrored at parse time in `regimeSegments` (`results.ts:199`) |
| A run bridging a missing label (the term structure has none before the VIX3M series exists) | `test_regime_segments_at_the_boundaries` (`tests/test_app_results.py:224`): a null label is no label and splits the run |
| A bar for a missing cell drawn as zero | `surfaceLayout` draws nothing for null or non-positive ratios (`errorSurface.ts:158`); the table shows a dash |
| A ratio of exactly 1 vanishing | the `minHeight` tile (`errorSurface.ts:166`, `:174`); `scale` 0 gives length 0 and still draws the tile |
| The two schemes read as one partition | the layout leaves a gap row where the scheme changes (`errorSurface.ts:146`); the page says they are slices |
| The README disagreeing with the JSON | `tools/results_table.py --check` |
| `--check` passing only at one commit | provenance fields exempt (`export_app_data.py:73`); content exact |
| A renamed copy of a vendor file in git | content hash (`test_staged_renamed_copy_is_caught`); a copy stored under a wrong name in the store (`test_snapshot_stored_under_a_wrong_name_is_still_matched`) |
| A staged copy that differs from the work tree | the index blob is hashed, not the file on disk (`test_staged_copy_is_caught_by_its_staged_bytes_not_the_work_tree`) |
| An empty snapshot matching every empty file | empty snapshots are skipped (`test_empty_file_is_not_a_match_for_an_empty_snapshot`) |
| A **modified** copy of a vendor file | **not caught, by design**: a one-byte change defeats a content hash (`test_modified_copy_is_not_flagged`). The text scanner `scan_real_rows.py` covers values; neither covers a transformed binary |
| An unstaged, untracked file | not checked, because git would not commit it (`test_unstaged_untracked_copy_is_not_flagged`) |
| An LSTM probe that cannot fail | the two falsifiers must change the entries; they do (section 7.3) |

Tests run for this note, 2026-11-09: `uv run pytest tests/test_app_results.py tests/test_check_raw_files.py
tests/test_export.py tests/test_contract.py -q` printed `274 passed`; `npx vitest run
src/lib/errorSurface.test.ts src/lib/results.test.ts src/lib/XYChart.test.tsx` printed 3 files and 67 tests
passed. Not run for this note: the full Python suite, the full app `npm run verify`, `tests/test_schemas.py`
(being edited elsewhere), and any new mutation run on the Polish-2 tests.

## 7. Measured results

### 7.1 The checks (commands run on 2026-11-09)

| Command | Printed |
|---|---|
| `uv run python tools/results_table.py --check` | `ok: ... README.md's results block matches the committed synthetic data` (exit 0) |
| `uv run python tools/export_app_data.py --mode synthetic --check` | `ok: 5 file(s) ... match a fresh synthetic export` (exit 0) |
| `uv run python tools/licensing/check_raw_files.py` | `raw-file gate: 0 raw vendor files tracked or staged, 0 files over 50 MB (324 files checked)` (exit 0) |

The full ten-step rebuild was not run today. The `df9655e` message says the chain ran in 2,064.8 s
for run `beb513536bef`; the working-tree manifest names run `3b44cf0297b9` with `source_dirty: true`,
so re-export at a clean commit if the JSON is to be committed (the docstring of
`tools/build_synthetic_store.py` says the same).

### 7.2 What the new files hold (read from `app/public/data/synthetic/`)

| Quantity | Value |
|---|---|
| `schema_version` in `evaluation.json` and `equity_curves.json` | `1.2.0` |
| `evaluation.losses` rows | 28 (7 race models x 3 VIX horizons, plus 7 on rv21) |
| `evaluation.regime_losses` rows | 140: 35 per panel (7 models x 5 regime cells), 4 panels |
| Null (suppressed) ratio cells | 0 |
| Rows other than `rw` | 120: 84 with an RMSE ratio below 1, 36 above 1, none exactly 1 |
| Largest RMSE ratio | 9.106721, `hist_mean_252`, VIX h = 1, low VIX |
| Smallest RMSE ratio | 0.822684, `arima_log`, VIX h = 21, high VIX |
| Their `ln` | 2.2090 and -0.1952 |
| `equity_curves` dates | 1,486 |
| `regime_segments` | 191: 126 on `vix_level`, 65 on `term_structure` |
| Cell sizes at VIX h = 1 (`rw`) | VIX level 1,605 + 3,644 + 861; term structure 3,149 + 569 |

The two schemes cover different numbers of origins (6,110 against 3,718 at VIX h = 1): the term-structure
regime has no label before the VIX3M series exists. With `ln` bars, the 9.1 cell is 2.5 long and the 0.82
cell hangs 0.22 below the plane; on a ratio scale the 0.82 cell would be a sliver beside it.

### 7.3 The O10 probe (`uv run python tools/licensing/p_tie_lstm.py`, 2026-11-09, about 30 s, exit 0)

Simulated data (seed 20261028), the cheap config the ledger states: 3 folds, 2 members, hidden 8, 4 epochs.
Entries compared: 4 per panel (RMSE ratio, QLIKE ratio, MAPE ratio, MAPE) x 4 panels = 16.

| Variant | c = 0.8, differ at 6 s.f. | c = 1.25 | at 3 s.f. (c = 0.8 / 1.25) |
|---|---|---|---|
| real model | 0 of 16 | 0 of 16 | 0 / 0 |
| falsifier `raw_level` | 16 of 16 | 16 of 16 | 2 / 1 |
| falsifier `flag20` | 16 of 16 | 16 of 16 | 15 / 12 |

The largest entry difference of the real model was 4.4e-16. Raw VIX-track forecasts divided by c differed
by at most 2.1e-14 (absolute). Not covered, from the ledger: the production config (hidden 32, 5 seeds, 50
epochs), the frozen-fit path, DM and MZ entries, and more than one simulated seed.

## 8. Explain it back

Answer in your own words. No answers are given here.

1. Explain what `--check` compares and what it ignores, and why `source_run_id` had to be moved into the
   ignored set. What does the README results tie add that `--check` does not?
2. Explain why a real export holds no regime cells or ribbons. What leaks from a cell with its `n`, or from
   a dated run, and which two checks stop a real store being read as synthetic?
3. Explain the bar encoding. Where is the plane, what does a bar of ratio 2 look like against a bar of
   ratio 0.5, and what happens to a null cell, to a ratio of exactly 1, and to a set of ratios that are
   all 1?
4. Explain the drei `Html` label problem: what was seen, what the workaround is, and what is *not*
   known. Why can the workaround not be unit tested?
5. Explain the metamorphic relation used for O10: what is changed, what must stay the same, why the
   probe needs falsifiers, and why 3 significant figures hides a leak that 6 shows.
6. Explain why filling `regime_segments` and adding `regime_losses` is a MINOR bump, and what an older
   file (without `regime_losses`) does in the app.
7. Explain the raw-file gate: what it hashes, why HEAD and the index are separate sets, why a name is not
   enough, and one thing it does not catch.

## 9. Open items

| # | Item | Default |
|---|---|---|
| O-M7-1 | The root cause of the empty first `Html` label is unknown | the post-`onCreated` mount is the workaround; re-check on any drei, fiber or React upgrade |
| O-M7-2 | The O10 production config, the frozen-fit path and DM/MZ entries were not probed | argued by equivariance in the ledger, not measured |
| O-M7-3 | The Polish-2 tests have not been mutation-checked in this note | do it before the commit |
| O-M7-4 | The committed synthetic JSON must be exported at a clean commit | `source_dirty: true` in the working-tree manifest today |
| O-M7-5 | The checkpoint's `Code at` is a placeholder until the push | the coordinator pins the SHA |
