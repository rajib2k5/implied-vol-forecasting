# 07 · ARIMA on log VIX, and choosing its order by AIC

What an ARIMA model says about VIX, how `arima_log` picks its order at every refit, and what the
back-transform from log space does. Background: [00 · Background](00-background.md),
[05 · Persistence and unit roots](05-persistence-half-life-and-unit-roots.md). Code:
[`src/ivf/models/baselines.py`](../../src/ivf/models/baselines.py) (`Arima`). Numbers are from
**synthetic** data or hand-worked; real-data results stay local (ADR 0001 D2).

## 1. First principles

- An **AR** term says today's value is partly yesterday's: `x_t = c + phi*x_{t-1} + e_t`. With
  `|phi| < 1` the series is pulled back to its mean `c/(1-phi)`; that is mean reversion.
- An **MA** term says today's shock is partly yesterday's surprise: `+ theta*e_{t-1}`. It lets a
  shock fade on a different schedule from the AR pull.
- **I** (integrated, `d = 1`) says model the *changes* `x_t - x_{t-1}` instead of the level. Then
  there is no mean to return to: every shock is permanent, as in a random walk.
- ARIMA(p, d, q) has p AR terms, q MA terms and d differences. Box and Jenkins (1970) made it the
  standard univariate forecasting model.

Why log VIX and not VIX: VIX is positive and its swings grow with its level (vol-of-vol rises
with vol, concept 06). The log is unbounded below and its swings are closer to constant, which is
what a Gaussian ARIMA assumes.

## 2. The maths

**Model.** `phi(L)(1 - L)^d x_t = c + theta(L) e_t`, `e_t ~ N(0, sigma2)`, with `x = ln VIX` and
`L` the lag operator. `arima_log` uses a constant `c` when `d = 0` (the long-run mean of ln VIX)
and none when `d = 1`: a constant on the differences is a drift, which would make log vol trend
for ever.

**Fit.** Exact Gaussian maximum likelihood through the Kalman filter (statsmodels `SARIMAX`). The
filter treats a missing close as a missing observation: nothing is filled, and the state is
carried across the gap by the model, never by the neighbouring value (gap policy G2).

**Multi-step forecast (iterated).** From the state at origin t, the filter predicts `t+1`, then
feeds its own prediction in to get `t+2`, and so on to `t+h`. It returns a mean `mu_h` and a
forecast-error variance `s2_h`. For an AR(1) with mean `m`:

- `mu_h = m + phi^h (x_t - m)`: the deviation shrinks by `phi` per session;
- `s2_h = sigma2 (1 - phi^(2h)) / (1 - phi^2)`: it grows with h towards the variance of x.

Hand-worked (and tested in `tests/test_baselines.py`): `m = 3`, `phi = 0.9`, `sigma2 = 0.01`,
`x_t = 3.5` gives `mu_5 = 3 + 0.59049*0.5 = 3.295245` and
`s2_5 = 0.01*(1 - 0.348678)/0.19 = 0.034280`.

**Back to vol points.** If `x_{t+h}` is normal with mean `mu` and variance `s2`, then
`y = exp(x)` is lognormal and

| Statistic | Formula | Which loss rewards it |
|---|---|---|
| median | `exp(mu)`, the naive back-transform | MAE |
| mean | `exp(mu + s2/2)`, the lognormal correction | RMSE |
| mean of `y^2`, as a vol | `exp(mu + s2)` | QLIKE on `(yhat/100)^2` |

The project evaluates the mean (ADR 0001 4.3, `var_method = plugin`) and stores the median beside
it as `yhat_naive`. At `s2 = 0.04` the two differ by `exp(0.02) - 1 = 2.0%`; at h = 21 on ln VIX
`s2` is several times larger, so the correction matters most where forecasts are hardest.

**Information criteria.** A model with more parameters always fits the training data at least as
well, so the in-sample likelihood cannot choose the order. Akaike's criterion (Akaike 1974, to
verify: not yet in the repo's references) penalises each parameter:
`AIC = 2k - 2 ln L`, with k parameters and maximised likelihood L. The smallest AIC wins. AIC
aims at the best one-step predictive fit; BIC (`k ln n - 2 ln L`) penalises more and aims at the
true order. AIC is the usual choice for forecasting.

**AIC cannot choose d.** With `d = 1` the likelihood is of the differences; with `d = 0`, of the
levels. They are likelihoods of different data, so their AICs are not comparable. The usual
recipe, Hyndman and Khandakar's (2008, to verify: not yet in the repo's references), which is
also what `pmdarima.auto_arima` does by default, picks d first with a unit-root test, then (p, q)
by AIC. `arima_log` keeps the AIC step but fixes d = 0 (section 3), because the unit-root
test is unreliable on a series this persistent (section 5).

## 3. What `arima_log` does at every cutoff

1. Training data: ln of the closes from the first close to the cutoff, nothing later.
2. **d = 0** (`d_rule: d0`, the default): an ARMA(p, q) with a constant on ln IV, which reverts to
   its mean. The unit-root case is the random walk baseline's. The option `d_rule: kpss` takes
   d = 1 when KPSS (null: level stationarity) rejects at 5% on the longest gap-free run of the
   training data; section 5 says why it is not the default (amended M2, review A).
3. **(p, q)**: every pair in `0..2 x 0..2` with that d (9 fits). The smallest AIC among fits
   that converged wins; a tie goes to the smaller order. Each candidate starts from its own
   previous-fold estimate (a warm start: same optimum, fewer iterations).
4. **Fallback**: no candidate converged → keep the previous fold's model; at the first fold →
   ARIMA(0,1,0), whose variance has a closed form (the mean squared daily change).
5. The chosen order, every candidate's AIC and convergence flag, the d rule (and the KPSS
   p-value under `d_rule: kpss`) and the parameters go into the `fit` table (`params_json`, `hyperparams_json`).

Inside the fold the parameters are frozen; only the filter state moves as each new close arrives.

## 4. Assumptions

- **Gaussian, homoskedastic shocks in ln VIX.** VIX spikes are sharper than its decays, and its
  vol-of-vol moves. `s2` is then right on average and wrong at any given time: too small after a
  spike, too large in calm. The lognormal correction inherits that error.
- **Parameters are known.** `s2_h` ignores estimation error. With thousands of observations it
  is small next to the shock variance.
- **One model for the whole expanding window.** 1990 and 2024 share one set of parameters (ADR
  0001 4.5 chose expanding windows on purpose).

## 5. Where it breaks

- **KPSS over-rejects for persistent series, so it is not the default d rule** (amended M2,
  review A). On a stationary AR(1) it chose d = 1 on 52% of seeds with `phi = 0.98` and n = 2,570
  (about the first VIX fold's sample), 24% at n = 8,700 (about the last), and 76% and 52% with
  `phi = 0.99` (review A, 100 seeds each; the coordinator's fresh seeds gave 47%, 28%, 70% and
  50%). With d = 1 the forecast never returns to the mean: on the synthetic law at h = 21 RMSE
  rose 17% (5.006 to 5.845) and the bias became level-dependent (+5.45 after highs, −2.34 after
  lows). The first real M2 run, under the KPSS rule, chose d = 1 at all 97 VIX cutoffs; that is
  what KPSS does to a persistent stationary series, not evidence of a unit root. The ledger is
  `docs/validation/M2-review-A-baselines.md`.
- **AIC can overfit a little.** It picks a larger model than the truth with positive
  probability even in large samples. With the grid capped at (2, 2) the cost is a few extra
  parameters.
- **A near-cancelling AR and MA pair** (`phi ≈ -theta`) fits about as well as no pair at all,
  so the chosen order can flip between folds with almost no change in the forecast. Compare
  forecasts, not orders.

## 6. In this project

- `Arima.fit` and `Arima.forecast`: [`src/ivf/models/baselines.py`](../../src/ivf/models/baselines.py).
- The rule's settings: `configs/models.yaml`, `baselines.arima` (`d_rule`, the grid).
- Tests: `tests/test_baselines.py` (closed forms, including through a missing close; both d
  rules; KPSS on the longest gap-free run; AIC argmin; both fallbacks) and the
  truncation test in `tests/test_walkforward.py`.

## Self-check

1. Why may two candidates with d = 0 be compared by AIC, but not a d = 0 and a d = 1 candidate?
2. An AR(1) with `phi = 0.98`: by how much has a deviation from the mean shrunk after 21
   sessions? What does ARIMA(0,1,0) say instead, and why did the KPSS rule keep choosing it?
3. MAE prefers `exp(mu)` to `exp(mu + s2/2)`. Is the forecast with the lower MAE the better
   forecast of the mean?
