# PR/FAQ: implied-vol-forecasting

> **Milestone 0 draft, written working backwards.** Rajib Bahar will edit it; every quote is a draft. **Status today (M0): design, scaffold and app shell only; no data fetched, no model fitted.** Results are bracketed `[RESULT: ...]` placeholders, filled at M5/M6 and re-measured by `verifier`. ADRs: `docs/adr/`; theory: [`docs/concepts/`](concepts/).
>
> **Roadmap.** M1 data and EDA; M2 baselines (random walk, historical mean, ARIMA on log-IV, HAR); M3 GARCH family; M4 LSTM; M5 evaluation; M6 trading backtest.

## 1. Press release

**DRAFT, written in advance. [DATE: Milestone 6 release, a future date]** *None of this has happened yet.*

### A fair race for VIX forecasts: random walk, ARIMA, HAR, GARCH and LSTM on identical folds

*An open-source Python package and React app that forecasts the CBOE VIX at 1-, 5- and 21-day horizons, tests whether model differences are real, and backtests the forecasts on VXX/SVXY.*

**Summary.** Rajib Bahar today released implied-vol-forecasting, a public repository that forecasts the VIX (CBOE's 30-day implied volatility of the S&P 500) with models from a random walk to an LSTM, scored under one protocol. Headline: `[RESULT: lowest-QLIKE model at h=21 on the holdout, and significance of its gap to the random walk — measured at M5]`.

**The problem.** Volatility-model comparisons go wrong in predictable ways: scalers or tuning that see test data, GARCH fitted to the VIX level, no check against "tomorrow equals today", and backtests tuned until they look good.

**The solution.** One protocol for every model: identical walk-forward folds, train-only scalers, and error metrics paired with significance tests. The random walk and HAR are the bars to beat. GARCH is fitted to S&P 500 returns; the one place it touches implied volatility is ARIMA-GARCH on log-IV changes. A costed VXX/SVXY backtest reports Sharpe with its Probabilistic and Deflated versions: `[RESULT: DSR of the best forecast-driven rule vs always-long and always-short — measured at M6]`.

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
uv run [PIPELINE COMMAND — named at M1]
cd app && npm ci && npm run dev
```

Then read `docs/milestones/`.

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
