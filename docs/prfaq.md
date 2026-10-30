# PR/FAQ: implied-vol-forecasting

> **Draft for owner review (revised 2026-10-29, after milestone 6).** Written working backwards at milestone 0; revised with what was measured. Rajib Bahar will edit it; every quote is a draft. **Status: milestones 0 to 6 done (code at `7eeae14`); the single guarded holdout look has NOT been taken.** Every number below is from the **synthetic** market (seed 20261028, development period); real results are measured locally and are not published under the current data policy (ADR 0001 D2). Synthetic numbers come from `uv run python tools/results_table.py` or from the milestone walkthrough named beside them. Holdout results stay `[RESULT: ...]` placeholders until the owner takes the look. ADRs: `docs/adr/`; theory: [`docs/concepts/`](concepts/).
>
> **Roadmap.** M1 data and EDA; M2 baselines (random walk, historical mean, ARIMA on log-IV, HAR); M3 GARCH family; M4 LSTM; M5 evaluation; M6 trading backtest.

## 1. Press release

**DRAFT, written in advance and revised after milestone 6. [DATE: release, a future date]** *The release has not happened. The synthetic results quoted are measured; every holdout and real-data result is still a placeholder.*

### A fair race for VIX forecasts: random walk, ARIMA, HAR, GARCH and LSTM on identical folds

*An open-source Python package and React app that forecasts the CBOE VIX at 1-, 5- and 21-day horizons, tests whether model differences are real, and backtests the forecasts on VXX/SVXY.*

**Summary.** Rajib Bahar today released implied-vol-forecasting, a public repository that forecasts the VIX (CBOE's 30-day implied volatility of the S&P 500) with models from a random walk to an LSTM, scored under one protocol. Headline (synthetic development period): at h = 21 the lowest-QLIKE VIX model, ARIMA-GARCH on ln VIX, has 0.80444 of the random walk's QLIKE, and for 21-day realised volatility GARCH has 0.645263 of it; at h = 1 the best model gains only 0.5% in RMSE. Holdout headline: `[RESULT: lowest-QLIKE model at h=21 on the holdout, and significance of its gap to the random walk — measured at the one look]`.

**The problem.** Volatility-model comparisons go wrong in predictable ways: scalers or tuning that see test data, GARCH fitted to the VIX level, no check against "tomorrow equals today", and backtests tuned until they look good.

**The solution.** One protocol for every model: identical walk-forward folds, train-only scalers, and error metrics paired with significance tests. The random walk and HAR are the bars to beat. GARCH is fitted to S&P 500 returns; the one place it touches implied volatility is ARIMA-GARCH on log-IV changes. A costed VXX/SVXY backtest reports Sharpe with its Probabilistic and Deflated versions. On the synthetic development period the pre-registered rule has Sharpe -0.2694 at 5 bps, PSR 0.2557 and DSR 0.0319 (5 trials), against -2.0665 for always-long VXX and 2.0658 for always-long SVXY. Holdout: `[RESULT: DSR of the primary rule vs always-long and always-short — measured at the one look]`.

**Author quote.** "I'm learning volatility modelling by building it, so every comparison has to be one I can defend line by line," said Rajib Bahar, project author. "`[DRAFT QUOTE: Rajib to rewrite]`"

**How it works.**

1. **Fetch and prepare.** CBOE and yfinance data are cached outside the repo; features use only data up to the forecast origin.
2. **Split.** One walk-forward splitter serves every model; a fixed-date holdout is scored once.
3. **Forecast.** VIX models forecast the VIX at 1, 5 and 21 sessions; realised-volatility models forecast `rv21` at 21. GARCH, fitted on S&P 500 returns, does both.
4. **Evaluate and show.** Score, test and backtest, then export JSON to the React app, with a 3D term-structure view.

**Customer quote (hypothetical).** *Not a real person or firm.* "I'd want to know whether my LSTM beats a random walk or only my own weak baseline. With every model on the same folds, tests could show which gaps are larger than noise." — a hypothetical volatility researcher

**Get started.**

```bash
git clone https://github.com/rajib2k5/implied-vol-forecasting.git
cd implied-vol-forecasting
uv sync
uv run python -m ivf.data --synthetic fetch          # then build-interim, check, eda (README, "How to run each step")
uv run python -m ivf.models --synthetic baselines    # then garch, lstm; then ivf.evaluation and ivf.backtest
uv run python tools/export_app_data.py --mode synthetic
cd app && npm ci && npm run dev
```

Then read `docs/milestones/`.

## What was measured (synthetic, development period)

Each hypothesis below was stated at milestone 0 and is reported whichever way it fell. These are synthetic results: they show that the machinery measures what it should on a market whose truth is known, not what the real VIX does.

| Hypothesis | Synthetic result | Where measured |
|---|---|---|
| H1: the random walk is hard to beat at h = 1 | Holds. Best RMSE ratio to the random walk 0.99503 at h = 1 (ARIMA-GARCH; ARIMA 0.995048), 0.90955 at h = 21 | `tools/results_table.py` |
| H2: implied sits above realised, and raw forecasts fail MZ | Partly. The synthetic VIX sits 4.86 vol points above later `rv21` on average (bias y − ŷ of `vix_implied` = -4.8612) but carries no information about it (MZ beta -0.0041, R² 0: the generator draws the two independently, a limit of the synthetic market). GARCH fails MZ in level space (beta 0.8888, Wald p 0.0008) and passes in variance space, because it forecasts E[v] | `app/public/data/synthetic/evaluation.json` (`losses`, `mz` rows for rv21); [milestone 5](milestones/05-evaluation.md) |
| H3: the LSTM does not beat HAR significantly | Holds, and more: the LSTM loses to HAR in every panel (RMSE ratio 0.920553 against 0.91263 at VIX h = 21; 0.896785 against 0.889032 on rv21), and each of its five seeds loses to HAR on its own | `tools/results_table.py`; [milestone 4](milestones/04-lstm.md) |
| H4: after costs no rule has a DSR distinguishable from zero | Holds for the primary rule: DSR 0.0319, Sharpe negative at every cost level (-0.2166 at 0 bps to -0.4272 at 20 bps) | `tools/results_table.py`; [milestone 6](milestones/06-backtest.md) |

The honest null results, kept as findings:

- **The LSTM loses to HAR in all seeds** (above).
- **ARIMA uses d = 0.** KPSS chose d = 1 on 24-76% of stationary AR(1) series as persistent as ln VIX, and forcing d = 1 raised h = 21 RMSE 17% ([milestone 2](milestones/02-baselines.md)).
- **GARCH sits within 1.2% of the truth floor** on synthetic `rv21` (QLIKE 0.1396 against 0.1379 for the true conditional variance), so there is little left for any model to find there ([milestone 3](milestones/03-garch-done-right.md)).
- **Walk-forward bias calibration helps in 0 of 28 cells** under the pre-registered rule ([milestone 5](milestones/05-evaluation.md)).
- **Significance is not free:** 90 of 168 model pairs differ at Holm-adjusted p < 0.05 with the fixed-b test on this long synthetic sample; the bandwidth choice changes many decisions ([milestone 5](milestones/05-evaluation.md)).
- **The ln VIX half-life** is 36.8 sessions (95% CI 30.16 to 47.14); the synthetic truth, phi = 0.98, gives ln(0.5)/ln(0.98) = 34.3 sessions, inside the interval.

What is still open: every real-data number, and every holdout number. The holdout look is the owner's single, guarded command (README, "The holdout look").

## 2. External FAQ

**What exactly is forecast?**
Two targets (ADR-0001 D4): `vix_level`, the VIX close at 1, 5 and 21 trading days (21 sessions ≈ 30 calendar days), and `rv21`, S&P 500 realised volatility over the next 21 sessions (h = 21 only). The VIX is the only implied-volatility target. VIX9D, VIX3M, VIX6M and VIX1Y feed the 3D term-structure view; VIX3M also gives a contango/backwardation flag. "IV" here means index levels.

**Which models?**
Those on the roadmap, plus calibrated GARCH variants.

**Why are the random walk and HAR the bars to beat?**
The random walk (forecast = today's value) is the minimum bar; a model that cannot beat it adds nothing. How hard it is to beat is hypothesis H1. HAR (Corsi, 2009) regresses future volatility on its daily, weekly and monthly averages; it was built for realised volatility and is applied here to the implied series.
**Why is GARCH fitted on S&P 500 returns, not the VIX level?**
GARCH models the changing variance of a return series; fitted to the VIX level it would model the variance of the VIX, not the VIX. So GARCH, GJR and EGARCH are fitted to daily S&P 500 returns and converted to an annualised 21-session variance forecast. It is compared like with like against `rv21`, and against the future VIX, where the gap between the two measures is hypothesis H2.

**How are models compared fairly?**
Every model uses the same walk-forward folds: an expanding window refitted every 63 sessions, parameters frozen within a fold. A training example counts only if its target date is on or before the fold's cutoff; otherwise overlapping multi-day targets leak future sessions. A test requires identical forecast origins for every model. RMSE (vol points) and QLIKE (variance) are the primary losses, used by the DM tests; MZ checks bias. MAE and MAPE are reported but not consistent for a mean forecast: MAE favours the median, MAPE lower forecasts.

**What does the trading backtest do?**
It maps a forecast to a VXX (long volatility) or SVXY (short volatility) position, executed at the next session's close, net of costs, and compares it with always-long VXX and always-short via SVXY using Sharpe, PSR and DSR. A good VIX forecast is not automatically a good trade: VXX and SVXY track VIX futures, not the VIX, so roll cost matters.

**Is CBOE data in the repo?**
No. CBOE permits personal, non-commercial use, so real data stays on the author's machine; per-date errors count as data (they can rebuild the VIX path). Real results enter `reports/` only as ADR-0001 D2's publication set P: rounded unitless numbers (loss ratios to the random walk, DM tests, MZ pass/fail, Sharpe-type) in one uncorrected release; rows tied to the VIX scale via public S&P 500 data stay local. The public app ships a synthetic fixture, badged SYNTHETIC; screenshots show synthetic views or P values.

**Is this financial advice?**
No. It is a research and learning project, and volatility ETPs can lose most of their value in a day.

## 3. Internal FAQ

### Data

**Sources and licensing?**
- **CBOE index CSVs:** VIX, VIX9D, VIX3M, VIX6M, VIX1Y, VXN, RVX, VXD, GVZ, OVX, VXEEM, and single-stock VXAPL, VXAZN, VXGOG, VXGS, VXIBM. Only the VIX is a target; the rest are stretch scope.
- **FRED `*CLS`** (fallback; re-hosted CBOE data, same restriction) and **yfinance** (S&P 500, VXX, SVXY; Yahoo personal use). Not legal advice (open question 5).

**What are the VXX and SVXY traps?**
- VXX is an ETN tracking a rolling one-month position in short-dated VIX futures, not the VIX; in contango it bleeds as futures roll toward spot. Reverse splits and series changes break raw prices, so returns use split-adjusted closes.
- SVXY is the ProShares Short VIX Short-Term Futures ETF; its daily target changed from -1x to -0.5x in Feb 2018, so the backtest runs only inside usable segments (dates verified at M6).
- Failed products (XIV, redeemed early in 2018) are absent, and ETP histories are far shorter than the VIX's.

### Feasibility

- **Compute.** Daily data is small. ADR-0002 chooses CPU-only torch; ADR-0001 O2 derives (not measures) about 2 h for the LSTM's fits.
- **Libraries.** `arch`, `statsmodels`, `pmdarima`, `torch`, pinned in `uv.lock`; ADR-0002 reports `pmdarima` working on Python 3.12, numpy 2 and aarch64.

### Risks

- **Look-ahead leakage.** Mitigations: shared folds, train-only fitting, tuning on development data only, and a truncation test (forecasts from data cut at t must equal the full run's).
- **Overlapping horizons.** At h = 5 and 21, neighbouring errors are autocorrelated (up to h-1 lags), so a naive DM test overstates significance. Mitigation: a lag h-1 variance estimator with a small-sample correction. ADR-0001 records that it can still over-reject when volatility clusters (O6); M5 picks a remedy by measurement.
- **Regime dependence.** The holdout may contain few volatility spikes, so conclusions could hinge on a handful of days. Mitigations: a regime breakdown at M5 (kept local), and a fixed holdout start of 2024-07-01, proposed so Aug-2024 and Apr-2025 are inside (open question 4).
- **Scenario: the random walk is not beaten.** That would still be a valid result: extra complexity bought nothing detectable here.
- **Backtest overfitting.** Keeping the best of many rules inflates Sharpe. Mitigation: DSR counts every variant tried.

### Success metrics

Process metrics, pass or fail:

1. One command reproduces every table in `reports/` from a clean clone after the data fetch; `verifier` runs it.
2. A mutation-checked test fails if any model's forecast origins differ from the others'; a truncation test passes for every model.
3. DM tests (squared error and QLIKE) for every model against the random walk, with no empty cells: h = 1, 5, 21 for the VIX-level target and h = 21 for the realised-variance target (the publication set P in [ADR 0001](adr/0001-python-core-versioned-json-app.md), D2, lists the scale-free rows).
4. Every number in the README and this document is re-measured by `verifier`; the adversarial ledger is committed in `docs/validation/`.

Hypotheses to test, not promised results; each is reported whichever way it falls.
- H1: "tomorrow equals today" is hard to beat at h = 1, because implied volatility is highly persistent (sources: [background, section 3](concepts/00-background.md)).
- H2: implied variance exceeds later realised variance on average (VRP), so raw GARCH forecasts sit below the future VIX and MZ shows intercept not 0 or slope not 1.
- H3: the LSTM does not beat HAR significantly.
- H4: after costs, no forecast-driven rule has a DSR distinguishable from zero.

### What we will not do

- No intraday data, live trading or paper trading.
- No claim of profitability.
- No vendor series, or real results outside D2's set P, in the repo or a deployed app.
- No options-chain or volatility-surface work; that is a sibling project (vol-strategy-backtester).

## Open questions for the author

1. **Reader.** Who is this for first: an interviewer, a quant peer, or you in a year?
2. **Success.** Is a defended "nothing beats the random walk" a success, or only beating HAR?
3. **Trading rule.** Which matters most: forecast minus current VIX (with a dead band), term-structure slope and roll, or a regime filter?
4. **Holdout.** Accept ADR-0001's proposed fixed start of 2024-07-01 (about 28 months; includes Aug-2024 and Apr-2025), not a trailing 24 months (starts 2024-10-28; excludes Aug-2024)?
5. **Publication.** Real results stay on your machine except ADR-0001 D2's set P. Define a "material part" of a vendor series (O9); publish scale-tied rows (O8; recommended no)?

## Glossary

- **VIX:** CBOE's 30-day expected S&P 500 volatility, from SPX options, annualised, in vol points.
- **Term structure:** implied volatility by maturity. **Contango:** longer maturities higher; **backwardation:** the reverse.
- **Realised volatility:** annualised volatility of daily returns (`rv21`: next 21 sessions).
- **VRP:** variance risk premium, the gap between implied and later realised variance.
- **QLIKE:** a variance-space loss penalising under-forecasts more.
- **DM / MZ:** Diebold-Mariano tests equal forecast accuracy; Mincer-Zarnowitz regresses outcomes on forecasts (unbiased: intercept 0, slope 1).
- **PSR / DSR:** probability a Sharpe ratio beats a benchmark; DSR also deflates for strategies tried.
