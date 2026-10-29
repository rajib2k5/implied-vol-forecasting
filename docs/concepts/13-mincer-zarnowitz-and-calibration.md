# 13. Mincer-Zarnowitz, and calibrating a biased forecast

*Milestone 5. Code: `src/ivf/evaluation/mz.py`, `src/ivf/evaluation/calibration.py`. Walkthrough:
[milestone 5](../milestones/05-evaluation.md).*

## Is the forecast the conditional mean?

DM ranks two forecasts. Mincer & Zarnowitz (1969) ask something about one forecast on its own:
regress the outcome on the forecast,

```
y_{t+h} = alpha + beta * yhat_t + e
```

If `yhat` is the conditional mean of y, then `alpha = 0` and `beta = 1`: on average the outcome is
the forecast, whatever its level. The joint Wald test of `(alpha, beta) = (0, 1)` checks both at once
(chi-squared with 2 degrees of freedom).

How to read a failure:

| Result | Meaning |
|---|---|
| `beta < 1` | Forecasts move too much: high forecasts are too high, low ones too low. Typical of a random walk at long horizons, which ignores mean reversion. |
| `beta > 1` | Forecasts move too little (over-smoothed). |
| `alpha ≠ 0` with `beta = 1` | A constant bias. |

A passing MZ test does not make a forecast good: a constant equal to the long-run mean has
`alpha = 0`, `beta = 1` in a large sample and is useless. MZ is a calibration check, DM a ranking.

## The covariance, again

The residuals of an h-step regression overlap exactly as the DM differentials do, so the standard
errors need the same long-run treatment. The scores `x_t · e_t` (x = (1, yhat)) are correlated up to
lag h−1 when the forecast is rational. Three candidates (ADR 0001 O5):

* **Hansen-Hodrick**: equal weights up to lag h−1. Exact for that dependence; can fail to be
  positive definite (then Newey-West is used and the row says `bartlett_fallback`).
* **Newey-West** at lag h−1: the M0 rule; biased down for the same reason as in DM.
* **Newey-West with Andrews' automatic bandwidth.**

Milestone 5 simulated a rational forecast of a persistent level (φ = 0.98, h = 21). At the dev sample
size (6,090) the Wald test rejected the true (0, 1) at a nominal 5%: Hansen-Hodrick 6.5%, Andrews
9.2%, Newey-West 14.3%. So Hansen-Hodrick is primary. At the holdout's size (583) all three were far
too liberal (16.6% for Hansen-Hodrick), so holdout MZ p-values are reported as descriptive only
(pre-registered in M5, amended M5, review A).

## Level space and variance space

For `rv21` the regression runs twice: in vol points (`y` on `yhat`) and in variance (`v` on `v_hat`).
GARCH forecasts the mean of the **variance**, `E[v]`; its vol-point number `100·sqrt(v_hat)` is the
root-mean-square vol, which by Jensen sits above the mean vol (concept 09). So GARCH can pass in
variance space and fail in level space, and on the synthetic market it does: at h = 21, GARCH's
level-space Wald p-value is 0.001 and its variance-space one 0.33.

## Calibrating a biased forecast (O-M4-7)

M4 found the LSTM's `rv21` forecast biased high on the synthetic market. One fix is a multiplier:
`yhat_cal = k · yhat`. The danger is fitting k on the same data it is scored on, which makes any
model look better. The rule here (pre-registered, `calibration.py`):

* at each fold cutoff c, fit `k_c = sum(y·yhat) / sum(yhat²)` (the least-squares multiplier) on the
  model's own past forecasts **whose targets are dated ≤ c**;
* apply `k_c` to the forecasts of the fold that starts at c; with fewer than 252 past pairs, k = 1;
* score the result as a **separate variant** (`mean_cal`, `<model>+cal`), never in place of the model;
* say it helps only if RMSE and QLIKE both fall and a DM test of raw against calibrated rejects
  after Holm.

On the synthetic market it helped none of the 28 (model, target, horizon) rows: the multiplier
learned from past errors did not carry over well enough to beat the raw forecast on both losses.
A one-number correction cannot fix a bias that changes with the level (the LSTM's was worst on high
forecasts).

## Check yourself

1. A forecast has `alpha = 2`, `beta = 0.9` and a Wald p-value of 0.4. Is it unbiased?
2. Why can GARCH pass MZ in variance space and fail it in vol points?
3. Why must `k_c` use only targets dated ≤ c, not origins ≤ c?
4. Calibration lowers RMSE but raises QLIKE. Under the rule, does it help?
