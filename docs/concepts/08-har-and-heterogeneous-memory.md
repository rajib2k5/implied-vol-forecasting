# 08 · HAR: heterogeneous memory from three averages

Why a regression on three averages of past volatility forecasts so well, how `har_log` fits it
per horizon, and where it is weak. Background: [00 · Background](00-background.md) (HAR row),
[04 · Clustering and the ACF](04-volatility-clustering-and-the-acf.md),
[07 · ARIMA](07-arima-and-information-criteria.md). Code:
[`src/ivf/models/baselines.py`](../../src/ivf/models/baselines.py) (`Har`, `har_design`).

## 1. First principles

- Traders work on different clocks. Some react to today, some to this week, some to this month.
  Corsi (2009) built volatility as the sum of three components, one per clock: a "heterogeneous"
  market.
- Each clock is represented by an average of past volatility over its window: 1 session, 5
  sessions (a week), 22 sessions (a month).
- A linear regression on those three averages is a restricted AR(22): 22 lags, but only three
  free slopes. It decays fast at first and slowly later, which mimics long memory (an ACF that
  dies off slowly) without a long-memory model.

That shape suits VIX: concept 05 found its ACF decays more slowly than one AR(1) can match.

## 2. The maths

With `x_t = ln VIX_t` and the windows of ADR 0001 (the `feature` table's `har_w` and `har_m`):

- `xw_t = mean(x_{t-4}, ..., x_t)` (5 sessions),
- `xm_t = mean(x_{t-21}, ..., x_t)` (22 sessions),

`har_log` fits, **separately for each horizon h** (the direct method):

`x_{t+h} = b0 + b_d x_t + b_w xw_t + b_m xm_t + e_{t+h}`

by ordinary least squares on every training pair `(t, t+h)` with `t + h <= cutoff` (rule L2) and
no gap in `t-21..t` or at `t+h` (G2). The h-step error variance is the residual variance
`s2 = RSS / (n - 4)`, and the forecast is `mu = b0 + b_d x_t + b_w xw_t + b_m xm_t`, back-
transformed exactly as ARIMA's: `yhat = exp(mu + s2/2)` (evaluated), `yhat_naive = exp(mu)`.

Hand-worked (tested): `x = 3.0` for 21 sessions then `3.5`; at that origin `xw = 3.1`,
`xm = 66.5/22 = 3.022727`. With `b = (0.1, 0.5, 0.3, 0.15)` and `s2 = 0.01`,
`mu = 0.1 + 1.75 + 0.93 + 0.453409 = 3.233409` and `yhat = exp(3.238409) = 25.4931`.

**Direct versus iterated.** ARIMA forecasts h steps by iterating its one-step model; HAR fits one
regression per h. Direct forecasts do not compound a misspecified one-step model, at the price of
one fit per horizon and of overlapping errors for h > 1 (consecutive h-step errors share h-1
shocks, so they are MA(h-1)). The overlap leaves the OLS coefficients and `s2` consistent; it
only makes their standard errors larger than OLS reports (`har_log` does not use OLS standard
errors).

**Why on logs.** Corsi's HAR is on realised volatility. On ln VIX the model is linear in a
quantity whose shocks are closer to symmetric, and the forecast stays positive. The price is the
back-transform.

## 3. Assumptions

- **Constant `s2`.** The h-step error variance is the same at every origin. When VIX is high its
  log moves more (vol-of-vol, concept 06), so the lognormal correction is too small after spikes
  and too large in calm. An ARIMA-GARCH or a HAR with a variance equation would fix it (M3).
- **Linear in logs, one set of slopes** for the whole expanding window.
- **No look-ahead in the averages.** All three are trailing (rule L1); a window with a gap is
  missing, never shortened.

## 4. Where it breaks

- **The first 21 sessions after a gap** have no monthly average, so no HAR forecast; to keep the
  same rows for every model (ADR 0001 4.5), no model forecasts there. For VIX in dev this never
  happens (G3). For VIX1Y and VVIX, which have gaps, it removes 22 origins per gap.
- **Collinearity.** The three regressors are strongly correlated (each average contains the
  shorter one). The forecasts are stable; the individual slopes are not, and should not be read
  as "how much traders on each clock matter".
- **Regime shifts** that change the slopes (a spike decays faster than calm drifts) are averaged
  away.

## 5. In this project

- `har_design` and `Har`: [`src/ivf/models/baselines.py`](../../src/ivf/models/baselines.py).
- Tests: exact recovery of a constructed h-step relation for h = 1, 5 and 21, the hand-worked
  forecast, the design's trailing windows (`tests/test_baselines.py`), and the truncation and L2
  tests (`tests/test_walkforward.py`).
- Primary source: Corsi, F. (2009), *Journal of Financial Econometrics* 7(2), 174-196 (in the
  references of concept 00).

## Self-check

1. Write HAR as an AR(22): what are the coefficients on `x_{t-1}` and on `x_{t-10}`?
2. Why does `har_log` fit three separate regressions instead of iterating the h = 1 one?
3. A gap in VIX1Y on session g: which origins lose their HAR forecast, and why do the other
   models lose those origins too?
