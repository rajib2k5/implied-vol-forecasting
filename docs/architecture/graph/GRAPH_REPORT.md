# Graph Report - implied-vol-forecasting  (2026-10-28)

## Corpus Check
- 102 files · ~103,817 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 13 file(s) not represented in the graph (top: (none) 5, .mmd 5, .conf 1)

## Summary
- 1222 nodes · 2526 edges · 70 communities (59 shown, 11 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 241 edges (avg confidence: 0.89)
- Token cost: 674,120 input · 0 output

## Community Hubs (Navigation)
- Architecture Docs Hub
- Python Contract Tests
- Contract and Config Modules
- Export CLI and Provenance
- ivf Package and Config Tests
- App Shell and Project Info
- Architecture Diagram Data
- Notebook Builder
- App RNG and Term Structure
- Raw Data and Fetch Pipeline
- Manifest JSON Fields
- App tsconfig Settings
- Forecast and Fit Entities
- Diebold-Mariano Literature
- Licensing Assumptions and Open Items
- App Guide and Testing Rules
- ESLint and Package Config
- App Dev Dependencies
- tsconfig.node Settings
- Determinism and Numeric Stack
- Error Surface and Suppression
- App Charts and Heatmaps
- Publication Set P Rules
- Export Check Tests
- App Pure Library Code
- Data Provenance and Guards
- ADR 0001 Decisions
- Implied vs Realized Literature
- p_set Licensing Probe
- Real vs Synthetic Data Policy
- guardDist Deploy Guard
- App Format and Regime Helpers
- Data Flow Row Types
- HAR and Statistical Predictability
- p_tie Licensing Probe
- App JSON Files and Envelope
- Sharpe Ratio Pitfalls
- Export CLI Tests
- npm Scripts
- Loss Functions Literature
- Holdout and Protocol Config
- guardDist Tests
- App Entry and RenderBoundary
- MAPE Check and Predict-First
- Symlink Safety Tests
- Bind-Mount Race Tests
- LineChart Component
- Ribbon Canvas 3D
- Periods and Look-Ahead Rules
- Backtest and Strategy Entities
- Calendar and RV21 Target
- Export Test Fixtures
- Guards and Adversarial Reviews
- Model Registry Rows
- GARCH Family Literature
- App Runtime Dependencies
- Export Rounding and Cells
- App Commands and Verify Gate
- App HTML Entry
- Root tsconfig References
- Vite Config
- Test Setup
- Failed-Write Test
- Project Root Node

## God Nodes (most connected - your core abstractions)
1. `ADR 0001: Python core, versioned JSON contract, licence-safe data, one evaluation protocol` - 54 edges
2. `Data model (milestones 0-6)` - 34 edges
3. `example_envelope()` - 25 edges
4. `Decision: CPU-only torch 2.14.1 from the PyTorch CPU index` - 22 edges
5. `compilerOptions` - 20 edges
6. `vitest` - 20 edges
7. `Publication set P: allow-list of unitless numbers (rules R1 and Q)` - 19 edges
8. `Data provenance` - 18 edges
9. `Publication set P (closed list of unitless real-data-derived numbers)` - 17 edges
10. `Milestone 0: scaffold and design` - 17 edges

## Surprising Connections (you probably didn't know these)
- `pytest + mutation check` --semantically_similar_to--> `Testing rules: hand-worked oracles, boundary cases, mutation check, no source greps`  [INFERRED] [semantically similar]
  docs/architecture/specs/process.yaml → app/CLAUDE.md
- `App never computes results` --semantically_similar_to--> `app/src/lib loaders`  [INFERRED] [semantically similar]
  app/CLAUDE.md → docs/architecture/specs/architecture.yaml
- `Publication set P: the only real-data numbers that may be committed` --semantically_similar_to--> `Committed reports: publication set P (reports/*.agg.csv)`  [INFERRED] [semantically similar]
  app/DATA.md → docs/architecture/data-model.md
- `Milestone 0 scope and limitations` --conceptually_related_to--> `Milestone 0: scaffold and design`  [INFERRED]
  app/README.md → docs/milestones/00-scaffold-and-design.md
- `Data and licensing policy (no vendor data; one release of unitless results)` --semantically_similar_to--> `Per-date errors are a series (random-walk errors rebuild the VIX path)`  [INFERRED] [semantically similar]
  README.md → docs/concepts/03-reproducibility-and-data-contracts.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Evidence and trade-offs behind choosing CPU-only torch** — docs_adr_0002_compute_and_numeric_stack_cpu_only_torch_decision, docs_adr_0002_compute_and_numeric_stack_gpu_latency_bound, docs_adr_0002_compute_and_numeric_stack_pypi_default_torch_is_cuda, docs_adr_0002_compute_and_numeric_stack_alt_a_cuda_everywhere, docs_adr_0002_compute_and_numeric_stack_thread_count_explicit [INFERRED 0.85]
- **D2 data-licensing rule set (P, material part, scale invariance, R1, Q)** — docs_adr_0001_python_core_versioned_json_app_publication_set_p, docs_adr_0001_python_core_versioned_json_app_material_part_definition, docs_adr_0001_python_core_versioned_json_app_scale_invariance_argument, docs_adr_0001_python_core_versioned_json_app_rule_r1_single_release, docs_adr_0001_python_core_versioned_json_app_rule_q_quoting_dated_levels [INFERRED 0.85]
- **Shared evaluation protocol (calendar, walk-forward, holdout, no-look-ahead, losses)** — docs_adr_0001_python_core_versioned_json_app_master_calendar, docs_adr_0001_python_core_versioned_json_app_walk_forward_design, docs_adr_0001_python_core_versioned_json_app_holdout_period, docs_adr_0001_python_core_versioned_json_app_no_look_ahead_rules, docs_adr_0001_python_core_versioned_json_app_loss_consistency [INFERRED 0.85]
- **M0 layered guards against leaking real data (exporter, .gitignore, deploy guard)** — docs_milestones_00_scaffold_and_design_exporter_guard_walkthrough, docs_milestones_00_scaffold_and_design_gitignore_walkthrough, docs_milestones_00_scaffold_and_design_deploy_guard_walkthrough, docs_milestones_00_scaffold_and_design_leak_guards [INFERRED 0.85]
- **Causal-evaluation safeguards against look-ahead** — docs_concepts_02_walk_forward_evaluation_and_look_ahead_label_availability_rule, docs_concepts_02_walk_forward_evaluation_and_look_ahead_train_only_scaling, docs_concepts_02_walk_forward_evaluation_and_look_ahead_calendar_folds_from_dates, docs_concepts_02_walk_forward_evaluation_and_look_ahead_truncation_test, docs_concepts_02_walk_forward_evaluation_and_look_ahead_fixed_holdout [INFERRED 0.85]
- **ADR-0001 D2 licensing-driven publication rule** — docs_concepts_03_reproducibility_and_data_contracts_publication_set_p, docs_concepts_03_reproducibility_and_data_contracts_scale_invariance_ratios, docs_concepts_03_reproducibility_and_data_contracts_failed_licensing_assumptions, docs_concepts_03_reproducibility_and_data_contracts_per_date_errors_are_series, docs_concepts_03_reproducibility_and_data_contracts_licensing_residual_risks [INFERRED 0.85]
- **Reconciling variance, vol-point and level forecasts (units and Jensen terms)** — docs_concepts_01_forecast_targets_and_annualisation_garch_forecast_to_vol_points, docs_concepts_01_forecast_targets_and_annualisation_vrp_p_vs_q_decomposition, docs_concepts_01_forecast_targets_and_annualisation_jensen_variance_vs_vol, docs_concepts_01_forecast_targets_and_annualisation_var_method_plugin_native, docs_concepts_01_forecast_targets_and_annualisation_lognormal_back_transform [INFERRED 0.85]
- **No-look-ahead rules L1 to L7** — docs_architecture_readme_rule_l1, docs_architecture_readme_rule_l2, docs_architecture_readme_rule_l3, docs_architecture_readme_rule_l4, docs_architecture_readme_rule_l5, docs_architecture_readme_rule_l6, docs_architecture_readme_rule_l7 [EXTRACTED 1.00]
- **Python to app JSON contract: manifest plus four dataset files** — docs_architecture_data_model_manifest_json, docs_architecture_data_model_term_structure_json, docs_architecture_data_model_model_race_json, docs_architecture_data_model_error_surface_json, docs_architecture_data_model_equity_curves_json [EXTRACTED 1.00]
- **Layers keeping real vendor data out of the deployed app** — app_data_real_json_outside_repo, app_data_gitignore_layer, app_data_deploy_guard, app_data_dockerignore [INFERRED 0.85]

## Communities (70 total, 11 thin omitted)

### Community 0 - "Architecture Docs Hub"
Cohesion: 0.06
Nodes (55): Where results come from: versioned JSON exported by tools/export_app_data.py, Architecture page (pipeline, fold sequence, milestone process diagrams), ADR 0001: Python core, versioned JSON contract, licence-safe data, one evaluation protocol, Alternative A: CUDA torch (cu130) everywhere (rejected), Alternative B: CUDA on Linux, CPU elsewhere via platform-split sources (rejected for now), torch.cuda.is_available() True does not mean PyTorch kernels run (cu126 on GB10), cuXYZ wheel suffix and Blackwell sm_121 compatibility, Data model (milestones 0-6) (+47 more)

### Community 1 - "Python Contract Tests"
Cohesion: 0.05
Nodes (47): example_envelope(), file_entry(), git(), manifest_payload(), manifest_with(), repo(), test_a_built_envelope_passes_the_envelope_schema(), test_a_failing_git_status_means_dirty() (+39 more)

### Community 2 - "Contract and Config Modules"
Cohesion: 0.05
Nodes (25): Columnar versus array-of-objects layout, Deterministic JSON bytes (sort_keys, allow_nan=False, compact separators), Walkthrough of src/ivf/config.py and src/ivf/contract.py, data_layout(), DataLayout, find_repo_root(), repo_root(), RepoRootNotFound (+17 more)

### Community 3 - "Export CLI and Provenance"
Cohesion: 0.06
Nodes (25): Benchmark evidence: single-fit, throughput and determinism measurements, run_id and provenance (hash of protocol, commit, dirty flag, data mode, raw hashes), Walkthrough of tools/bench_lstm.py (evidence behind ADR 0002), Repo layout (src/ivf, tests, tools, schemas/app-data, app, docs), make_data(), Net, train_one(), check() (+17 more)

### Community 4 - "ivf Package and Config Tests"
Cohesion: 0.06
Nodes (25): make_fake_repo(), test_find_repo_root_accepts_a_relative_start_path(), test_find_repo_root_ignores_an_unrelated_pyproject(), test_find_repo_root_raises_outside_any_repo(), test_find_repo_root_walks_up_from_a_nested_folder(), test_importing_ivf_creates_nothing(), is_ignored(), test_committed_files_are_not_ignored() (+17 more)

### Community 5 - "App Shell and Project Info"
Cohesion: 0.09
Nodes (35): App(), STORE, APP_SLUG, DATA_SOURCES, DataSource, DISCLAIMER, Milestone, MILESTONES (+27 more)

### Community 6 - "Architecture Diagram Data"
Cohesion: 0.11
Nodes (39): DataKind, describeEdges(), FOLD_ACTORS, FOLD_ORDER, FOLD_STEPS, FoldStep, FoldStepKind, MAX_ACTION_CHARS (+31 more)

### Community 7 - "Notebook Builder"
Cohesion: 0.06
Nodes (14): notebooks/ (generated), Walkthrough of tools/build_notebooks.py (notebooks as build products), Verification commands and mutation checks, read_notebook(), test_check_detects_saved_outputs(), written(), build_notebook(), check() (+6 more)

### Community 8 - "App RNG and Term Structure"
Cohesion: 0.11
Nodes (27): gaussianFrom(), mulberry32(), buildRibbonGeometry(), businessDates(), classifyCurve(), curveLevel(), CurveShape, DataMode (+19 more)

### Community 9 - "Raw Data and Fetch Pipeline"
Cohesion: 0.08
Nodes (30): ivf.data.build_interim, CBOE index CSVs, ivf.data.fetch, FRED *CLS series (fallback), ROOT/raw/provenance.jsonl, ROOT/raw/{source}/{symbol}/{sha256}.csv, ROOT/synthetic/raw/, ivf.data.synthetic (+22 more)

### Community 11 - "Manifest JSON Fields"
Cohesion: 0.08
Nodes (23): data_mode, dataset, generated_at, payload, files, models, notice, protocol (+15 more)

### Community 12 - "App tsconfig Settings"
Cohesion: 0.09
Nodes (21): compilerOptions, allowImportingTsExtensions, jsx, lib, module, moduleDetection, moduleResolution, noEmit (+13 more)

### Community 13 - "Forecast and Fit Entities"
Cohesion: 0.17
Nodes (19): forecast var_method: native vs plugin, LSTM (mean over lstm_members seeds), rv21_back(t) equals target_value(rv21, t): one series, two roles, ivf.evaluation, forecast and fit tables, ivf.models, ROOT/processed/run_id=<id>/{table}.parquet, ivf.evaluation.splits (+11 more)

### Community 14 - "Diebold-Mariano Literature"
Cohesion: 0.14
Nodes (17): Diebold (2015) comparing predictive accuracy, twenty years later, Diebold and Mariano (1995) comparing predictive accuracy, Diebold-Mariano test (rectangular LRV with HLN factor, Bartlett fallback), Harvey, Leybourne and Newbold (1997) testing equality of prediction MSE, Mincer and Zarnowitz (1969) evaluation of economic forecasts, Mincer-Zarnowitz regression and joint Wald test, Model confidence set (optional in M5), Assumptions and where it breaks (structural breaks, vintage, cost of purging) (+9 more)

### Community 15 - "Licensing Assumptions and Open Items"
Cohesion: 0.19
Nodes (11): Fixed holdout_start 2024-07-01 (open item O1), Assumptions and where it breaks (cross-platform kernels, regex dialects, joint inversion), Open item O10: publishing lstm_log rows (decided in M4 with p_tie), Open item O8: publish scale-tied rows (recommended no), Open item O9: definition of a 'material part' of a vendor series, Publication set P: allow-list of unitless numbers (rules R1 and Q), Open questions for the author (reader, success, trading rule, holdout, publication), Milestone 0 adversarial reviews (A code, B maths, C literature, D fixes, E final attack) (+3 more)

### Community 16 - "App Guide and Testing Rules"
Cohesion: 0.16
Nodes (13): Implied Vol Forecasting: app, project guide for Claude, Non-root nginx-unprivileged container with CSP, Coverage thresholds (85% lines/functions, 75% branches on src/lib), mulberry32 seeded rng (src/lib/rng.ts), Exact-pinned dependencies installed with npm ci, App stack: React 19, TypeScript, Vite, React Three Fiber 9, drei 10, vitest, Documented UI exceptions (shell, charts, storage.ts, theme.ts), Implied Vol Forecasting: app (+5 more)

### Community 17 - "ESLint and Package Config"
Cohesion: 0.14
Nodes (16): name, private, type, version, eslint, @eslint/js, eslint-plugin-react-hooks, eslint-plugin-react-refresh (+8 more)

### Community 18 - "App Dev Dependencies"
Cohesion: 0.11
Nodes (18): devDependencies, eslint, @eslint/js, eslint-plugin-react-hooks, eslint-plugin-react-refresh, globals, jsdom, @testing-library/jest-dom (+10 more)

### Community 19 - "tsconfig.node Settings"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, lib, module, moduleDetection, moduleResolution, noEmit, noFallthroughCasesInSwitch (+9 more)

### Community 20 - "Determinism and Numeric Stack"
Cohesion: 0.15
Nodes (14): D3: storage layout (raw, interim, processed, reports, app data), D5: determinism and reproducibility (pins, seeds, --check, contract tests), Fetch idempotency and provenance chain (sha256 raw files, append-only JSONL), run_id and protocol_hash identities, ADR 0002: Compute and numeric stack, arch EGARCH: simulation needs rng= (random_state ignored) and lower-case method, pmdarima predict requires Python int n_periods (np.int64 raises TypeError), LSTM reproducibility: bitwise identical only on same lock, machine and thread count (+6 more)

### Community 21 - "Error Surface and Suppression"
Cohesion: 0.28
Nodes (15): About page, Diebold-Mariano test (rectangular LRV, HLN correction), error_surface.json (model x horizon x regime cube, M5), Mincer-Zarnowitz regression, QLIKE loss (variance space), reports/*.agg.csv files (forecast_ratios, dm_tests, mz_tests, strategy_summary, half_life), loss row (transient, no stored table), metric cell (model x horizon x regime) (+7 more)

### Community 22 - "App Charts and Heatmaps"
Cohesion: 0.21
Nodes (10): binValues(), Heatmap(), Histogram(), normalizeMatrix(), BACKWARDATION_RGB, CONTANGO_RGB, TermStructureData, detectWebGL() (+2 more)

### Community 23 - "Publication Set P Rules"
Cohesion: 0.23
Nodes (6): EDA prose allow-list (statements E1 to E7), Publication set P (closed list of unitless real-data-derived numbers), R1: one release per frozen data_end; withdraw, never correct, Scale-tied rows kept local (vix_implied on rv21; uncalibrated garch/gjr/egarch on vix_level), Walkthrough of tools/licensing and tools/probes committed probes, Licensing probes (ADR 0001 D2)

### Community 24 - "Export Check Tests"
Cohesion: 0.12
Nodes (15): parsed(), run_id_for(), test_check_detects_a_changed_payload_value_and_says_where(), test_check_detects_a_committed_file_that_breaks_the_schema_in_a_provenance_field(), test_check_detects_an_added_field(), test_check_ignores_a_different_run_id_on_its_own(), test_check_ignores_all_provenance_fields(), test_check_still_requires_run_id_to_be_12_hex() (+7 more)

### Community 25 - "App Pure Library Code"
Cohesion: 0.15
Nodes (9): Tested domain invariants (fixture, ribbon geometry, milestones, diagram data), MILESTONES in src/lib/project.ts, src/lib/termStructure.ts (geometry, colours, generator), Seeded fixture generator (mulberry32, seed 20261027), src/data/term-structure.synthetic.json fixture, Source layout (src/lib, src/pages, src/data, scripts), npm scripts (dev, verify, coverage, build, guard:dist, preview, gen:fixture), Term structure 3D page (synthetic placeholder ribbon) (+1 more)

### Community 26 - "Data Provenance and Guards"
Cohesion: 0.16
Nodes (5): Data provenance, Deploy guard guardDist.mjs (last line of defence), .dockerignore excludes public/data/real and .env*, .gitignore entry for app/public/data/real/, Provisional payload shape until the M1 exporter fixes the contract

### Community 27 - "ADR 0001 Decisions"
Cohesion: 0.21
Nodes (10): Alternatives considered (rejected options per decision), D2 residual risks (joint inversion, scale-tied opt-in, few-day strategy differences, quoted closes, legal sufficiency), D4: one evaluation protocol shared by every model, JSON envelope (schema_version, generated_at, source_commit, source_dirty, data_mode, dataset, payload), Material part of a vendor series (M1 run of 5 within +/-1 vol point, M2 level statistic within +/-10%), Q: quoting dated index levels in prose (minimisation, not protection), Semver rules for schema_version (MAJOR/MINOR/PATCH), Synthetic fixture (same loaders, pipeline and exporter on generated data) (+2 more)

### Community 28 - "Implied vs Realized Literature"
Cohesion: 0.21
Nodes (13): Carr and Wu (2009) variance risk premiums, Cboe VIX methodology and mathematics methodology (2026a, 2026b), Christensen and Prabhala (1998) relation between implied and realized volatility, Demeterfi, Derman, Kamal and Zou (1999) guide to volatility and variance swaps, IV as a forecast of RV: biased but informative, Multi-step GARCH variance forecast G_t(a), Variance risk premium (implied versus realised volatility), VIX computation and meaning (model-free variance strip) (+5 more)

### Community 29 - "p_set Licensing Probe"
Cohesion: 0.25
Nodes (8): dm(), fit_direct(), forecasts(), lrv_rect(), mz_pass(), publication_set(), sim(), strategy()

### Community 30 - "Real vs Synthetic Data Policy"
Cohesion: 0.21
Nodes (11): Real JSON (outside the repo, served by dev server only), Synthetic JSON (committed, data_mode synthetic), Where app JSON files live (synthetic committed, real outside the repo), export_app_data.py --mode synthetic --check, app/src/lib loaders, Equity curve page, Model race page, Term-structure ribbon (3D) page (+3 more)

### Community 31 - "guardDist Deploy Guard"
Cohesion: 0.24
Nodes (12): APP_ROOT, checkBuild(), DATA_DIRS, hasRealMarker(), JSON_ROOTS, listTree(), lstatOrNull(), main() (+4 more)

### Community 32 - "App Format and Regime Helpers"
Cohesion: 0.26
Nodes (9): bps(), fmt(), pct(), regimeCounts(), regimeSummary(), rgbToHex(), TermStructurePage(), behaviour (+1 more)

### Community 33 - "Data Flow Row Types"
Cohesion: 0.21
Nodes (13): backtest_daily row, calendar row, feature row at origin t, fitted model, params frozen, fold assignment, forecast row, iv_index_daily row, regime_label row (+5 more)

### Community 34 - "HAR and Statistical Predictability"
Cohesion: 0.27
Nodes (11): Corsi (2009) HAR model of realised volatility, Fernandes, Medeiros and Scharth (2014) modelling the CBOE VIX, Forecasting IV itself: statistical is not economic predictability, HAR model (daily, weekly, monthly cascade), LSTM and the evidence on machine learning for volatility, Models we build: random walk, historical mean, ARIMA, HAR, GARCH family, ARIMA-GARCH, LSTM, Stylised facts of volatility (clustering, persistence, mean reversion, leverage, heavy tails, vol-of-vol, spikes), H1: tomorrow equals today is hard to beat at h=1 (+3 more)

### Community 35 - "p_tie Licensing Probe"
Cohesion: 0.33
Nodes (7): cal_track(), ewma_var(), losses(), nn_track(), ratio_rows(), rv21(), rv21_track()

### Community 36 - "App JSON Files and Envelope"
Cohesion: 0.35
Nodes (11): equity_curves.json (equity with regime ribbons, M6), App JSON envelope (schema_version, data_mode, dataset, payload), license_class: series vs aggregate, manifest.json (read first by the app), model_race.json (forecast vs actual by horizon, M2), 2 MiB per-file size budget, term_structure.json (3D VIX term-structure ribbon, M1), iv_index_daily entity (+3 more)

### Community 37 - "Sharpe Ratio Pitfalls"
Cohesion: 0.18
Nodes (11): Bailey and Lopez de Prado (2012, 2014) PSR and deflated Sharpe ratio, Lo (2002) the statistics of Sharpe ratios, Sharpe-ratio pitfalls: annualisation, PSR and DSR, Simon and Campasano (2014) the VIX futures basis, Trading on volatility forecasts: VIX is not tradable; VXX/SVXY mechanics and M6 backtest pitfalls, VIX term structure: VIX9D to VIX1Y, contango and backwardation, Whaley (2013) trading volatility: at what cost, Glossary (VIX, term structure, realised volatility, VRP, QLIKE, DM/MZ, PSR/DSR) (+3 more)

### Community 38 - "Export CLI Tests"
Cohesion: 0.17
Nodes (11): run_cli(), test_a_folder_inside_the_repo_that_cannot_be_read_is_refused_not_skipped(), test_check_command_exits_1_on_drift(), test_check_mode_never_writes(), test_check_passes_on_the_committed_fixture(), test_cli_output_path_that_is_a_file_exits_5(), test_cli_real_check_in_repo_is_refused_too(), test_cli_real_mode_in_repo_exits_3_and_writes_nothing() (+3 more)

### Community 39 - "npm Scripts"
Cohesion: 0.18
Nodes (11): scripts, build, coverage, dev, gen:fixture, guard:dist, lint, preview (+3 more)

### Community 40 - "Loss Functions Literature"
Cohesion: 0.22
Nodes (7): Gneiting (2011) making and evaluating point forecasts, Patton (2011) volatility forecast comparison using imperfect proxies, QLIKE and Patton's noisy-proxy robustness, vol-strategy-backtester (sibling project for options-chain and surface work), Credits (Cboe, FRED, Yahoo Finance, sharpe-observatory, methods, libraries), Not financial advice, Results table (empty until milestone 2)

### Community 41 - "Holdout and Protocol Config"
Cohesion: 0.22
Nodes (10): Holdout from 2024-07-01 (keeps Aug-2024 and Apr-2025 spikes), backtest_daily: decision origin two sessions earlier (L7), returns from adj_close, protocol config (protocol_hash, dev/holdout dates, horizons, fold_len, seed), Regime schemes: vix_level (low/mid/high) and term_structure (contango/backwardation), VXX/SVXY leverage segments (tradable_segment), configs/*.yaml, instrument_meta entity, iv_index_meta entity (+2 more)

### Community 42 - "guardDist Tests"
Cohesion: 0.22
Nodes (4): cleanApp(), ADR-0001, put(), SCRIPT

### Community 43 - "App Entry and RenderBoundary"
Cohesion: 0.24
Nodes (3): RenderBoundary, react, react-dom

### Community 44 - "MAPE Check and Predict-First"
Cohesion: 0.24
Nodes (7): GARCH comparability to 30-day implied vol (G_t(a), 21 sessions, 252, rv21 vs vix_level tracks), Plugin vs native forecasts (y-hat and v-hat, var_method), Small LSTM is GPU latency-bound (parity at hidden 32, 1.3x at 64), LSTM workload: ~5,000 samples, 22-day window, hidden 32-64, batch 64, Answers to the predict-first questions, Predict-before-you-look questions P1 to P9, argmin()

### Community 45 - "Symlink Safety Tests"
Cohesion: 0.27
Nodes (8): make_link(), test_a_link_that_appears_after_the_check_is_replaced_not_written_through(), test_a_simulated_m1_real_export_cannot_overwrite_a_repo_file_through_a_link(), test_cli_refuses_a_linked_output_file_with_exit_3(), test_real_mode_refuses_a_second_name_for_the_repo_by_device_and_inode(), test_real_mode_refuses_any_folder_inside_another_git_checkout(), test_real_mode_refuses_any_path_inside_the_repo(), test_write_files_refuses_a_link_at_the_target_and_leaves_the_linked_file_alone()

### Community 46 - "Bind-Mount Race Tests"
Cohesion: 0.20
Nodes (6): real_export_through_bind_mount(), test_a_folder_that_vanishes_during_the_walk_is_refused_not_a_traceback(), test_cli_real_export_exits_3_with_a_message_when_a_repo_folder_vanishes_mid_walk(), test_real_export_through_a_bind_mount_of_a_repo_subfolder_is_refused(), test_the_same_bind_mount_of_a_folder_outside_the_repo_is_allowed(), walk_that_deletes()

### Community 47 - "LineChart Component"
Cohesion: 0.28
Nodes (4): layoutLegend(), LineChart(), pad0, Series

### Community 48 - "Ribbon Canvas 3D"
Cohesion: 0.28
Nodes (8): RibbonGeometry, RibbonCanvas(), RibbonMesh(), toLinear(), RibbonCanvas, @react-three/drei, @react-three/fiber, three

### Community 49 - "Periods and Look-Ahead Rules"
Cohesion: 0.28
Nodes (8): Master calendar T (NYSE sessions from dates alone) and information set, No-look-ahead rules L1 to L7 and the truncation test, O1: holdout start (fixed 2024-07-01 vs exactly 24 months), O5 and O6: MZ kernel and DM size remedies decided in M5 by measurement, Open items O1 to O10 for the orchestrator, Walk-forward design (expanding window, 63-session folds, label availability, same rows), Update: ~107 refits gives ~80,250 model-epochs (~111 min CPU pool, derived), Decisions waiting on the user (O1, CPU vs CUDA torch, PR/FAQ questions, D2 publication rule)

### Community 50 - "Backtest and Strategy Entities"
Cohesion: 0.36
Nodes (8): Probabilistic and Deflated Sharpe Ratio (PSR, DSR), ivf.backtest, metric, dm_test, mz_regression, backtest_daily, strategy_summary tables, sharpe-observatory scripts/metrics.py (PSR, DSR), backtest_daily entity, equity_curves.json entity, strategy_def entity, strategy_summary entity

### Community 51 - "Calendar and RV21 Target"
Cohesion: 0.25
Nodes (8): Assumptions and where it breaks (windows differ, autocorrelation, jumps, fat-tailed log errors), Calendar conventions: 365-day VIX versus 252-session RV21, RV21 realised-volatility target, Forecast targets: vix_level (h=1, 5, 21) and rv21 (h=21), Draft press release (written working backwards), Roadmap M1-M6, Milestones table (M0 current; M1-M6 planned), Status: milestone 0 of 6 (design, scaffold, app shell; no data, no models, no results)

### Community 52 - "Export Test Fixtures"
Cohesion: 0.22
Nodes (4): can_bind_mount(), exported(), fake_repo(), precious()

### Community 53 - "Guards and Adversarial Reviews"
Cohesion: 0.43
Nodes (7): D2 guards: exporter refusal, .gitignore layer, guardDist tripwire, Regimes, tests and suppression (fixed thresholds, MZ test, min_cell_n), What the five adversarial reviews found and what changed, Walkthrough of app/scripts/guardDist.mjs (deploy guard), Walkthrough of tools/export_app_data.py and its real-data licensing guard, Walkthrough of .gitignore and tests/test_gitignore.py, The four guards against leaking real data (exporter, .gitignore, npm run verify, rule P)

### Community 54 - "Model Registry Rows"
Cohesion: 0.29
Nodes (7): ARIMA and ARIMA-GARCH on ln VIX, GARCH / GJR / EGARCH on SPX returns (plus calibrated variants), HAR (Corsi) model on ln VIX, Initial model registry (illustrative; M2-M4 finalise), rw random-walk baseline (benchmark), vix_implied model: v_hat = (V_t/100)^2, Variance risk premium gap

### Community 55 - "GARCH Family Literature"
Cohesion: 0.52
Nodes (4): ARIMA-GARCH on log-VIX changes, Engle (1982) ARCH, GARCH, GJR and EGARCH on SPX returns, Hansen and Lunde (2005) does anything beat a GARCH(1,1)

### Community 56 - "App Runtime Dependencies"
Cohesion: 0.33
Nodes (6): dependencies, react, react-dom, @react-three/drei, @react-three/fiber, three

### Community 57 - "Export Rounding and Cells"
Cohesion: 0.50
Nodes (5): Rounding at export (decimals per field), app: typed array / three.js buffer, blank cell (value null, n kept), JSON column array, Decision: n >= min_cell_n (20)?

### Community 58 - "App Commands and Verify Gate"
Cohesion: 0.67
Nodes (4): App commands (dev, verify, coverage, guard:dist, gen:fixture), scripts/guardDist.mjs deploy guard, npm run verify gate (lint, tests, coverage, build, deploy guard), npm run verify deploy guard (no data_mode real in dist/)

### Community 59 - "App HTML Entry"
Cohesion: 1.00
Nodes (3): app/index.html SPA entry page, src/main.tsx entry module, div#root React mount point

## Knowledge Gaps
- **48 isolated node(s):** `eslint`, `jsdom`, `@types/react`, `@types/react-dom`, `@types/three` (+43 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 384 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ADR 0001: Python core, versioned JSON contract, licence-safe data, one evaluation protocol` connect `Architecture Docs Hub` to `Data Provenance and Guards`, `Export CLI and Provenance`, `Loss Functions Literature`, `Forecast and Fit Entities`, `Diebold-Mariano Literature`, `Licensing Assumptions and Open Items`, `App Guide and Testing Rules`, `Calendar and RV21 Target`, `Error Surface and Suppression`, `Publication Set P Rules`, `App Commands and Verify Gate`, `Implied vs Realized Literature`, `Real vs Synthetic Data Policy`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Decision: CPU-only torch 2.14.1 from the PyTorch CPU index` (e.g. with `03 · Reproducibility and data contracts` and `Exact pins and the uv.lock lockfile`) actually correct?**
  _`Decision: CPU-only torch 2.14.1 from the PyTorch CPU index` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `eslint`, `jsdom`, `@types/react` to the rest of the system?**
  _48 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Architecture Docs Hub` be split into smaller, more focused modules?**
  _Cohesion score 0.06220095693779904 - nodes in this community are weakly interconnected._
- **Why does `Data flow through M0 (Python to JSON, synthetic vs real, what the app reads)` connect `App Charts and Heatmaps` to `Contract and Config Modules`, `Export CLI and Provenance`, `App RNG and Term Structure`, `Determinism and Numeric Stack`, `Data Provenance and Guards`, `ADR 0001 Decisions`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Should `Python Contract Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05365296803652968 - nodes in this community are weakly interconnected._
- **Why does `vitest` connect `App Shell and Project Info` to `App Format and Regime Helpers`, `Architecture Diagram Data`, `App RNG and Term Structure`, `guardDist Tests`, `App Entry and RenderBoundary`, `LineChart Component`, `ESLint and Package Config`, `App Charts and Heatmaps`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._