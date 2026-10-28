# Architecture diagrams

The decisions behind these diagrams are in
[ADR 0001](../adr/0001-python-core-versioned-json-app.md). The stored tables and JSON files are
described in [`data-model.md`](data-model.md). Each diagram has a YAML spec (the source of truth)
and a Mermaid file drawn from it.

## Index

| Diagram | Spec | Mermaid | What it shows |
|---|---|---|---|
| Data model (ERD) | [`specs/erd.yaml`](specs/erd.yaml) | [`erd.mmd`](erd.mmd) | Every stored table and app JSON file, with keys and cardinalities |
| System architecture | [`specs/architecture.yaml`](specs/architecture.yaml) | [`architecture.mmd`](architecture.mmd) | End to end: external sources, the Python core (`src/ivf`), storage, tooling and the app, with the synthetic JSON (committed) kept apart from the real JSON (outside the repo) |
| Data flow | [`specs/data-flow.yaml`](specs/data-flow.yaml) | [`data-flow.mmd`](data-flow.mmd) | One datum, from a CBOE CSV row to a JSON column array. Each edge names the transformation and the no-look-ahead rule (L1 to L7) that applies |
| Milestone process | [`specs/process.yaml`](specs/process.yaml) | [`process.mmd`](process.mmd) | Build, checks, learning material, adversarial review, number re-measurement, security review, push, then the pause for the learner |
| Walk-forward fold | none (hand-written) | [`walk-forward-sequence.mmd`](walk-forward-sequence.mmd) | One fold in order: splitter, feature builder, scaler fit on train only, model fit on pairs with target date at or before the cutoff, frozen parameters, daily forecasts, evaluator, loss rows |

## System architecture, rendered inline

GitHub renders Mermaid inside a fenced block in a Markdown file. The block below is a copy of
[`architecture.mmd`](architecture.mmd), which is hand-synced with
[`specs/architecture.yaml`](specs/architecture.yaml). When you change the
spec or the `.mmd`, change this copy in the same commit. The diagram is large, so open it full
width.

```mermaid
%% Hand-synced with specs/architecture.yaml until tools/diagrams/ is vendored (M1); edit both. (amended M0, review B)
%% implied-vol-forecasting system architecture (M0-M6)
%% Shapes: source stadium, store cylinder, process box, model subroutine, artifact parallelogram, ui rounded, decision diamond, external hexagon.
flowchart TB
    subgraph external["External"]
        cboe(["<b>CBOE index CSVs</b><br/><i>cdn.cboe.com .../daily_prices/#123;SYM#125;_History.csv, 307 to cdn-api.cboe.com</i>"])
        fred(["<b>FRED *CLS series</b><br/><i>fallback: only fills dates missing from Cboe</i>"])
        yfinance(["<b>yfinance</b><br/><i>^GSPC and other underlyings, VXX, SVXY; auto_adjust=False</i>"])
        sharpe{{"<b>sharpe-observatory scripts/metrics.py</b><br/><i>PSR, expected max SR, DSR (MIT, reused)</i>"}}
    end
    subgraph core["Python core (src/ivf)"]
        fetch["<b>ivf.data.fetch</b><br/><i>temp file, SHA-256, rename to #123;sha256#125;.csv if absent; skip within --max-age 1 day</i>"]
        synthetic["<b>ivf.data.synthetic</b><br/><i>seeded two-factor generator; hand-set parameters, not fitted</i>"]
        build_interim["<b>ivf.data.build_interim</b><br/><i>typed, deduplicated, one table per source kind; latest snapshot unless --snapshot</i>"]
        verify["<b>ivf data verify</b><br/><i>re-hash every raw file; fail on any mismatch</i>"]
        features["<b>ivf.features</b><br/><i>trailing windows only (L1); also calendar, targets, regimes (L5)</i>"]
        splits["<b>ivf.evaluation.splits</b><br/><i>expanding window, 63-session folds, shared by every model</i>"]
        models[["<b>ivf.models</b><br/><i>rw, hist_mean, ARIMA, HAR, GARCH/GJR/EGARCH, ARIMA-GARCH, LSTM; params frozen within a fold</i>"]]
        evaluation["<b>ivf.evaluation</b><br/><i>RMSE, MAE, MAPE, QLIKE, bias; Diebold-Mariano; Mincer-Zarnowitz; regimes</i>"]
        backtest["<b>ivf.backtest</b><br/><i>VXX/SVXY signals executed at close of t+1 (L7), costs, PSR/DSR</i>"]
        ivf_reports["<b>ivf.reports</b><br/><i>commits only publication set P (unitless ratios, 2 dp tests); one release per frozen data_end; needs a clean tree</i>"]
    end
    subgraph storage["Storage"]
        configs[("<b>configs/*.yaml</b><br/><i>protocol, models, strategies, iv_indices, instruments; committed. Every stage reads protocol.yaml; ivf.models reads models.yaml; ivf.backtest reads strategies.yaml</i>")]
        raw[("<b>ROOT/raw/#123;source#125;/#123;symbol#125;/#123;sha256#125;.csv</b><br/><i>bytes as downloaded; never committed</i>")]
        prov[("<b>ROOT/raw/provenance.jsonl</b><br/><i>append-only, one line per fetch, written even if unchanged</i>")]
        synth_raw[("<b>ROOT/synthetic/raw/</b><br/><i>Cboe-shaped CSVs and Yahoo-shaped tables; never committed</i>")]
        interim[("<b>ROOT/interim/#123;table#125;.parquet</b><br/><i>iv_index_daily, price_daily; every row keeps raw_sha256</i>")]
        processed[("<b>ROOT/processed/run_id=#60;id#62;/#123;table#125;.parquet</b><br/><i>calendar, feature, target_value, regime_label, fold, eda_stat, data_check; runs.parquet beside it</i>")]
        forecasts[("<b>forecast and fit tables</b><br/><i>under processed/run_id=#60;id#62;/; key (run_id, model_id, target_id, origin_date, horizon)</i>")]
        results[("<b>metric, dm_test, mz_regression, backtest_daily, strategy_summary</b><br/><i>under processed/run_id=#60;id#62;/</i>")]
        reports_dir[/"<b>reports/*.agg.csv</b><br/><i>committed: publication set P only (unitless, rounded), source_dirty = false</i>"/]
        schemas[("<b>schemas/app-data/</b><br/><i>JSON Schemas: the single source of truth for the contract</i>")]
        synth_json[/"<b>app/public/data/synthetic/*.json</b><br/><i>committed fixture; data_mode synthetic</i>"/]
        real_json[/"<b>ROOT/app-data/real/*.json</b><br/><i>never in the repo tree; the Vite dev server serves it in dev only</i>"/]
    end
    subgraph tooling["Tooling"]
        export["<b>tools/export_app_data.py</b><br/><i>reshape, round, validate, write; computes no new number (D1). Ribbon levels come from interim iv_index_daily</i>"]
        export_check["<b>export_app_data.py --mode synthetic --check</b><br/><i>regenerate in a temp dir and compare with the committed fixture</i>"]
        build_nb["<b>tools/build_notebooks.py</b><br/><i>--check is the notebook sync test</i>"]
        notebooks[/"<b>notebooks/</b><br/><i>generated from src/ivf; never hand-edited</i>"/]
    end
    subgraph app["App"]
        loader["<b>app/src/lib loaders</b><br/><i>check envelope and invariants, show the data_mode badge; compute nothing</i>"]
        page_ribbon("<b>Term-structure ribbon (3D)</b><br/><i>term_structure.json, series</i>")
        page_race("<b>Model race</b><br/><i>model_race.json, series</i>")
        page_surface("<b>Error surface (3D)</b><br/><i>error_surface.json, aggregate</i>")
        page_equity("<b>Equity curve</b><br/><i>equity_curves.json, series</i>")
        dist_guard["<b>npm run verify</b><br/><i>fails if any file in dist/ has data_mode real</i>"]
    end
    cboe -->|"HTTP GET, follow 307"| fetch
    fred -->|"fallback series"| fetch
    yfinance -->|"DataFrame to CSV, hashed"| fetch
    configs -->|"URLs, tickers"| fetch
    fetch -->|"#123;sha256#125;.csv if absent"| raw
    fetch -->|"append one line"| prov
    raw -->|"re-hash"| verify
    synthetic -->|"seeded CSVs"| synth_raw
    raw -->|"latest snapshot"| build_interim
    synth_raw -->|"same loaders"| build_interim
    build_interim -->|"typed parquet"| interim
    interim -->|"iv_index_daily, price_daily"| features
    features -->|"calendar, feature, target_value, regime_label, run"| processed
    configs -->|"protocol: dev_start, holdout_start, fold_len"| splits
    processed -->|"calendar t_idx"| splits
    splits -->|"folds: cutoff, origins"| models
    processed -->|"features at t (L1)"| models
    models -->|"fit rows, forecast rows"| forecasts
    forecasts -->|"same origins for every model"| evaluation
    processed -->|"targets y, v; regimes"| evaluation
    evaluation -->|"metric, dm_test, mz_regression"| results
    forecasts -->|"signal at origin t"| backtest
    interim -->|"VXX, SVXY adj_close"| backtest
    sharpe -->|"PSR, DSR"| backtest
    backtest -->|"backtest_daily, strategy_summary"| results
    results -->|"aggregates"| ivf_reports
    ivf_reports -->|"set P only, one release"| reports_dir
    results -->|"metrics, DM, MZ, equity"| export
    forecasts -->|"yhat for the race"| export
    processed -->|"target_value, regime_label, run"| export
    schemas -->|"validate before writing"| export
    export -->|"synthetic: committed"| synth_json
    export -->|"real: outside the repo"| real_json
    export -->|"temp-dir rerun"| export_check
    synth_json -->|"compare with fixture"| export_check
    models -->|"generated from src/ivf (all modules)"| build_nb
    build_nb -->|"generate"| notebooks
    synth_json -->|"manifest.json first"| loader
    real_json -->|"dev server only"| loader
    schemas -->|"contract tests"| loader
    loader --> page_ribbon
    loader --> page_race
    loader --> page_surface
    loader --> page_equity
    synth_json -->|"vite build copies to dist/"| dist_guard
    classDef artifact fill:#f3e5f5,stroke:#7b1fa2,color:#212121
    classDef external fill:#fce4ec,stroke:#ad1457,color:#212121
    classDef model fill:#e8f5e9,stroke:#2e7d32,color:#212121
    classDef process fill:#f5f5f5,stroke:#616161,color:#212121
    classDef source fill:#e3f2fd,stroke:#1565c0,color:#212121
    classDef store fill:#fff8e1,stroke:#f9a825,color:#212121
    classDef ui fill:#e0f7fa,stroke:#00838f,color:#212121
    class reports_dir,synth_json,real_json,notebooks artifact
    class sharpe external
    class models model
    class fetch,synthetic,build_interim,verify,features,splits,evaluation,backtest,ivf_reports,export,export_check,build_nb,loader,dist_guard process
    class cboe,fred,yfinance source
    class configs,raw,prov,synth_raw,interim,processed,forecasts,results,schemas store
    class page_ribbon,page_race,page_surface,page_equity ui
```

## How to render

Every `.mmd` file renders with Mermaid CLI (needs Node and a Chromium for Mermaid CLI to drive):

```bash
npx -y @mermaid-js/mermaid-cli@11.17.0 -i docs/architecture/architecture.mmd -o /tmp/architecture.svg
```

If Mermaid CLI cannot find its bundled Chromium, pass `-p puppeteer.json`, where that file holds
`{"executablePath": "<path to a local Chromium>", "args": ["--no-sandbox"]}`. Use `-o x.png -w 2200`
for a PNG. The architecture diagram is large (about 4,400 by 4,000 px as SVG), so open the SVG and
zoom rather than reading a thumbnail.

SVG/PNG/draw.io/Lucid/Figma exports arrive when the shared diagram toolkit is vendored into `tools/diagrams/` (import-tested: no).

## Spec format

The three flow specs share one shape. The shared toolkit's final schema may differ slightly.

```yaml
kind: flow
title: ...
direction: LR | TB
lanes: [{id, label}]
nodes: [{id, label, lane, type, note?}]
edges: [{from, to, label}]
```

| `type` | Meaning | Mermaid shape |
|---|---|---|
| `source` | Data we do not own | Stadium |
| `external` | Code or a person outside the loop | Hexagon |
| `process` | Code or a step that runs | Rectangle |
| `model` | A fitted forecaster | Subroutine |
| `store` | A persistent table or file | Cylinder |
| `artifact` | A row, cell, array or committed product | Parallelogram |
| `ui` | What the learner sees | Rounded box |
| `decision` | A gate whose exits are all labelled | Diamond |

Ids match `[A-Za-z_][A-Za-z0-9_]*` and are not Mermaid keywords. Labels and notes are one line.
Each lane becomes one `subgraph`.

Until `tools/diagrams/` exists, the `.mmd` files are kept in sync with their specs by hand
(each one starts with a `Hand-synced with specs/...` comment; `walk-forward-sequence.mmd` is hand-written). When you change a spec, change
its `.mmd` in the same commit.

## Reading the data-flow diagram

| Rule | Meaning (ADR 0001, 4.7) |
|---|---|
| L1 | Features use trailing windows only. The information set at origin t is data dated at or before t. |
| L2 | A training pair is usable only if its target date is at or before the cutoff. |
| L3 | Scalers, imputers and clipping thresholds are fitted on the training rows only, at each cutoff. |
| L4 | The LSTM's early-stopping block is the last 252 training origins, separated by an h-session purge. |
| L5 | Regimes come from data at or before their date, with fixed thresholds. |
| L6 | Hyper-parameters are chosen on dev rows whose target date is before `holdout_start`. |
| L7 | A signal at origin t is executed at the close of t+1. |

## Where the diagrams go beyond ADR 0001

These are choices made while drawing, to confirm when the code lands:
- ADR 0001 names `ivf.data.fetch`, `ivf.data.build_interim`, `ivf.data.synthetic`, `ivf.reports`,
  `ivf data verify` and `tools/export_app_data.py`. The names `ivf.features`,
  `ivf.evaluation.splits`, `ivf.models`, `ivf.evaluation`, `ivf.backtest`, `tools/build_notebooks.py`
  and `notebooks/` come from the project plan and brief.
- ADR D3 says only "the pipeline stages" write the calendar, `target_value` and `regime_label`
  tables, so the architecture diagram draws them inside `ivf.features`.
- The "loss row" in the data-flow diagram has no table in the data model. It is drawn as a transient
  state inside `ivf.evaluation`.
