# implied-vol-forecasting

Forecast implied volatility (the VIX first) with a random walk, ARIMA, HAR, the GARCH family and an LSTM on identical walk-forward folds, test whether the differences are real, and backtest the forecasts on VXX and SVXY.

> **Status: milestone 0 of 6.** Design, scaffold and app shell. No data has been fetched, no model has been fitted, and there are no results yet. The sections below describe what the project is built to do; each one says which milestone delivers it.

## What it does, and what it teaches

The project is built in milestones, and each milestone has a written walkthrough in [`docs/milestones/`](docs/milestones/). The original requirement: develop and compare models that forecast implied volatility (ARIMA, GARCH, and machine learning such as an LSTM), start with exploratory analysis of volatility clustering and persistence, evaluate with RMSE, MAE and MAPE, and backtest trading strategies driven by the forecasts.

What it teaches, and when it arrives:

- **Volatility clustering and persistence** as the starting point: autocorrelation, the AR(1) half-life, mean reversion, the implied-versus-realised spread (milestone 1).
- **HAR (Corsi, 2009) as the baseline to beat**, next to the random walk, a historical mean and ARIMA on log-IV (milestone 2).
- **GARCH done right** (milestone 3). GARCH, GJR and EGARCH are fitted to the S&P 500's *returns* and turned into 21-day variance forecasts, which are compared with implied volatility. Fitting GARCH to the VIX *level* is a category error, and the project says why. ARIMA-GARCH is applied to log-IV changes.
- **An LSTM** on the same folds, with scaling fitted on the training window only, early stopping and several seeds (milestone 4).
- **RMSE and QLIKE as the primary losses**, with MAE and MAPE reported beside them because the brief asks for them. Neither is consistent for a mean forecast: MAE favours the median and MAPE favours lower forecasts, so every table flags them. Diebold-Mariano and Mincer-Zarnowitz tests at horizons of 1, 5 and 21 trading days (milestone 5).
- **Identical walk-forward splits and no look-ahead**, with a fixed-date holdout that contains the August 2024 and April 2025 volatility spikes. The proposed start is 2024-07-01; it is still an open decision ([ADR 0001](docs/adr/0001-python-core-versioned-json-app.md), item O1).
- **A costed VXX/SVXY backtest** against always-long and always-short volatility, with the Probabilistic and Deflated Sharpe ratios (milestone 6).
- **Engineering habits**: Python computes and a versioned JSON contract carries the result to the app; the app never computes a result; real data stays off the repository; every number is reproducible from a pinned environment.

An interactive React app (`app/`) draws the results: a 3D VIX term-structure ribbon, a model race, a 3D error surface and an equity curve. With real data these views run only on your machine; the committed fixture is synthetic. In milestone 0 the app has an About page, an Architecture page and a 3D placeholder driven by that fixture.

## Quick start

You need [uv](https://docs.astral.sh/uv/) 0.11 (developed with 0.11.25) and Node 22.

```bash
git clone https://github.com/rajib2k5/implied-vol-forecasting.git
cd implied-vol-forecasting
uv sync                      # Python 3.12 via uv; CPU-only torch (ADR 0002)
uv run pytest -q             # the Python tests
cd app && npm ci && npm run dev
```

`npm run verify` (inside `app/`) is the app's full gate: lint, tests with coverage thresholds, a production build, and a deploy guard that fails if real market data could ship. The remaining checks are listed in the [milestone 0 walkthrough](docs/milestones/00-scaffold-and-design.md#10-verification).

## Data and licensing

- **No vendor data is in this repository.** Cboe index history is licensed for personal, non-commercial use, and FRED's `*CLS` series and Yahoo Finance data carry the same kind of restriction. The pipeline downloads them to your own machine only.
- **Where data lives:** `$VOL_DATA_DIR/implied-vol-forecasting/`, outside the repo. `VOL_DATA_DIR` defaults to `~/data/vol-projects`.
- **No vendor series is published.** The goal is that nothing committed, git history included, lets anyone reconstruct a vendor index's daily series or a material part of it. Per-date forecasts, errors and equity curves count as the data, because the random walk's cumulative one-day errors rebuild the VIX path. They are never committed for real data.
- **One release of real results, at most, and only unitless numbers.** `reports/*.agg.csv` may hold a single set of real result tables, written once for a frozen end date. Every number in it is a ratio to the random walk, a test statistic, a pass/fail flag, a count or a time span, for a whole period (development or holdout). Never a level in vol points, a mean, a coefficient, a return, a drawdown or a regime cell. A model is included only if its numbers do not change when the VIX is rescaled, even given the public S&P 500 series; models tied to the S&P 500 (for example uncalibrated GARCH on the VIX level) stay on your machine. A wrong value is withdrawn, never corrected. An earlier promise that no single day is recoverable was deleted, because each new combination of published statistics broke it ([milestone 0, section 7](docs/milestones/00-scaffold-and-design.md#7-what-the-adversarial-reviews-found-and-what-changed)). The probes that test this run on simulated data and live in [`tools/licensing/`](tools/licensing/README.md).
- **The app ships a synthetic fixture,** generated from a seed and labelled synthetic wherever it is drawn. Real-data JSON is written outside the repo and served only by the local dev server.
- The reasoning and the guards are in [ADR 0001](docs/adr/0001-python-core-versioned-json-app.md), decision D2 (the closed publication set, rules R1 and Q). That is a technical rule, not legal advice, and D2 is still Proposed until the author accepts it, with open items O8 to O10 (scale-tied rows, "material part", the LSTM).

## Repo layout

Directories that exist today:

```
README.md  LEARNING_LOG.md  LICENSE  pyproject.toml  uv.lock
src/ivf/          the Python package; all computation lives here
                  config.py, contract.py now; data/ features/ models/ evaluation/ backtest/ arrive in M1 to M6
tests/            pytest, with hand-worked expected values
notebooks/        generated from tools/build_notebooks.py, never hand-edited
tools/            build_notebooks.py, export_app_data.py, bench_lstm.py
                  licensing/ (probes that test the publication rule, simulated data only), probes/ (small checks)
schemas/app-data/ the JSON Schemas of the Python-to-app contract
app/              React 19 + TypeScript + Vite + three.js
                  src/lib (pure, unit-tested), src/pages, src/data (synthetic fixture),
                  public/data/synthetic (the committed JSON the exporter writes),
                  scripts/guardDist.mjs (the deploy guard)
docs/             prfaq.md, adr/, architecture/, concepts/, milestones/, validation/
```

Planned, not present yet: `configs/` (protocol, models, strategies), `reports/` (aggregate result tables), `docs/images/` (screenshots). `docs/validation/` holds the adversarial-review ledgers, one per review.

## Documentation

| What | Where |
|---|---|
| The idea, written backwards from the finished project | [`docs/prfaq.md`](docs/prfaq.md) (PR/FAQ) |
| Why the design is the way it is | [ADR 0001](docs/adr/0001-python-core-versioned-json-app.md): Python core, JSON contract, licensing, storage, evaluation protocol, determinism. [ADR 0002](docs/adr/0002-compute-and-numeric-stack.md): compute and numeric stack. |
| Stored tables and JSON files | [`docs/architecture/data-model.md`](docs/architecture/data-model.md) |
| Diagrams | [`docs/architecture/`](docs/architecture/README.md), with Mermaid sources [`architecture.mmd`](docs/architecture/architecture.mmd), [`erd.mmd`](docs/architecture/erd.mmd), [`data-flow.mmd`](docs/architecture/data-flow.mmd), [`process.mmd`](docs/architecture/process.mmd) and [`walk-forward-sequence.mmd`](docs/architecture/walk-forward-sequence.mmd). The system architecture is also embedded as a rendered diagram in that folder's README. |
| Theory behind each idea | [`docs/concepts/`](docs/concepts/): [00 background](docs/concepts/00-background.md), [01 forecast targets and annualisation](docs/concepts/01-forecast-targets-and-annualisation.md), [02 walk-forward evaluation and look-ahead](docs/concepts/02-walk-forward-evaluation-and-look-ahead.md), [03 reproducibility and data contracts](docs/concepts/03-reproducibility-and-data-contracts.md) |
| Milestone walkthroughs | [`docs/milestones/`](docs/milestones/), starting with [milestone 0](docs/milestones/00-scaffold-and-design.md) |
| Adversarial reviews and what they found | [`docs/validation/`](docs/validation/): [review A, code and numerics](docs/validation/M0-review-A-code-numerics.md); [review B, maths and documents](docs/validation/M0-review-B-maths-docs.md); [review C, the literature brief](docs/validation/M0-review-C-background.md); [review D, the fixes for A and B](docs/validation/M0-review-D-fixes.md); [review E, the final attack](docs/validation/M0-review-E-final.md) |
| The course index: milestones, questions, answers | [`LEARNING_LOG.md`](LEARNING_LOG.md) |

## Results

Results arrive from milestone 2 onwards. Every number in this README is re-measured by a verifier before it is committed. Real-data results will cover whole periods only (development and holdout), from the single release described under "Data and licensing", and will be ratios to the random walk's loss rather than levels. RMSE and QLIKE are the primary comparisons; MAE and MAPE are not consistent for a mean forecast.

| Model | Target | Horizon | Period | RMSE | MAE | MAPE | QLIKE | n |
|---|---|---|---|---|---|---|---|---|
| none yet | | | | | | | | |

## Screenshots

Screenshots arrive from the final milestone. Those of time-series views will show synthetic data with a visible SYNTHETIC badge. A real-data figure may be drawn only from numbers already committed in `reports/`, for example the holdout RMSE ratio by model. The real error surface (which has regime cells), histograms and time plots stay on your machine.

## Not financial advice

This is a research and learning project. Nothing here is a recommendation or an offer to trade. Backtests are hypothetical, past performance does not predict future results, and volatility exchange-traded products can lose most of their value in a day.

## License

[MIT](LICENSE), copyright 2026 Rajib Bahar. The licence covers this repository's code and documents. It does not cover Cboe, FRED or Yahoo data, which this repository does not contain.

## Credits

- **Cboe index data** is used through Cboe's site under Cboe's terms (personal, non-commercial use) and is not redistributed here. Cboe and VIX are trademarks of Cboe Exchange, Inc. This project is independent and is not affiliated with, endorsed by or sponsored by Cboe.
- **FRED** (Federal Reserve Bank of St. Louis) is the fallback source for the same index series, and **Yahoo Finance**, through the `yfinance` library, supplies underlying and ETP prices. Both are used for personal research only.
- **`sharpe-observatory`**: the Probabilistic and Deflated Sharpe ratio code (`scripts/metrics.py`) will be reused from the author's own repository under its MIT licence, in milestone 6.
- **Methods**: HAR is Corsi (2009); QLIKE as a robust loss is Patton (2011); the Diebold-Mariano and Mincer-Zarnowitz tests are the standard forecast-comparison tools.
- **Libraries**: statsmodels, arch, pmdarima, PyTorch, pandas, numpy, scipy, React, three.js and React Three Fiber.
