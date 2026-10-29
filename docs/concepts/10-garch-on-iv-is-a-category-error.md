# 10 · Why fitting GARCH on the IV level is a category error

ADR 0001 4.4 says the GARCH family is "never fitted on the VIX level", and calls doing so a category
error. This note says why in plain words, then shows it on synthetic data where the truth is known.
Background: [04 · Clustering](04-volatility-clustering-and-the-acf.md),
[05 · Persistence](05-persistence-half-life-and-unit-roots.md),
[09 · The GARCH family](09-garch-family-and-variance-aggregation.md). Code:
`garch.garch_on_the_level_demo` in [`src/ivf/models/garch.py`](../../src/ivf/models/garch.py);
notebook 03, section 5.

## 1. The argument

- **GARCH is a model of a shock's variance.** Its input is a series that has (close to) zero mean and
  no autocorrelation, whose *size* clusters: a return. Its output is a variance, `sigma2_t`.
- **VIX is already the output of that kind of model**, not an input to one. It is the market's
  forecast of the next 30 days' volatility: a positive, persistent *level* (concept 05: an AR(1) on
  ln VIX has phi near 0.98). Feeding it to GARCH asks "what is the variance of a volatility
  forecast's deviation from its mean?", which is not the question being forecast.
- **The level breaks the model's assumptions in a way that does not show as a failed fit.** The fit
  converges and gives sensible-looking numbers. The damage is in their meaning:
  1. The "shocks" `V_t − mu` are the level's deviations from its mean. They are as persistent as the
     level, so they are autocorrelated, which GARCH assumes they are not.
  2. Their squares are persistent too, so the fitted alpha + beta comes out near 1. That looks like
     strong volatility clustering. It is the level's persistence, relabelled.
  3. The mean equation is a constant, so the forecast of the level is the historical mean, whatever
     VIX is today: the weakest M2 baseline.
  4. Its "variance forecast" is in vol points² of a level deviation. Aggregating it like `rv21`
     (concept 09) produces a number with no counterpart in the target.
- **The right spaces.** For the vol-of-vol question, give ln VIX a mean model first and put GARCH on
  its innovations, which are close to shocks: that is `arima_garch_log` (an AR with a constant on
  ln VIX, GARCH errors; amended M3, review A: it was an AR on `Δ ln VIX`). For the volatility
  question, model the returns VIX is a forecast of: `garch`, `gjr`, `egarch` on S&P 500 returns,
  compared with VIX on the `rv21` target.

## 2. The demonstration (synthetic data, publishable)

The synthetic market's ln VIX is an AR(1) with phi = 0.98 and **constant** innovation variance
(`ivf.data.synthetic`). So the truth is: there is no volatility clustering in VIX at all, and today's
VIX is the best simple guide to tomorrow's. `garch_on_the_level_demo` fits a constant-mean GARCH(1,1)
to the level, and a zero-mean GARCH(1,1) to `100 Δ ln VIX`, on the 8,689 synthetic sessions before
`holdout_start` (measured 2026-10-29, notebook 03 section 5 prints the same numbers):

| Quantity | GARCH on the level | GARCH on Δ ln VIX | Truth |
|---|---:|---:|---|
| alpha + beta (level) / alpha (changes) | **0.9771** | 0.0095 | no clustering: alpha = 0 |
| Lag-1 autocorrelation of standardised residuals | **0.8512** | −0.0170 | the shocks are white |
| One-step RMSE of the mean forecast, vol points | **6.0565** (the constant mean) | — | random walk: 1.1769 |

- On the level, GARCH "finds" clustering that is not there (0.977), its residuals are not shocks
  (0.851), and its forecast is five times worse than "tomorrow equals today".
- On the log changes, the same model correctly finds nothing (alpha 0.0095) and white residuals.

`tests/test_garch.py::test_garch_on_the_vix_level_shows_the_category_error_on_synthetic_data` pins
the direction of each number on the first 4,000 sessions.

## 3. What would make it not a category error

If the question were "how volatile is the VIX level around its mean", a model of the level's
*innovations* would answer it: an AR or HAR mean on ln VIX first, then GARCH on its residuals. That is
an ARIMA-GARCH, which is what `arima_garch_log` is (d = 0, a mean-reverting AR; amended M3, review A).
GARCH on the raw level skips the mean model, so the persistence leaks into the variance.

## Self-check

1. Why does the fitted alpha + beta come out near the level's AR(1) phi?
2. The level GARCH converges and its parameters are significant. Why is that not evidence it is right?
3. Which two models in this project answer the two questions the level GARCH confuses?
4. On real data VIX changes *do* cluster. Does that rescue GARCH on the level?

Sources: Engle (1982); Bollerslev (1986) for what GARCH models; ADR 0001 4.4 for the project's rule.
The demonstration is this project's own (synthetic data, `ivf.data.synthetic`).
