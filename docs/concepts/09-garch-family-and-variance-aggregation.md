# 09 · The GARCH family and variance aggregation

How GARCH, GJR and EGARCH turn yesterday's return into tomorrow's variance, how 21 daily variances
become one number comparable with `rv21` and with VIX, and the three places where the units go wrong.
Background: [00 · Background](00-background.md) (the model table and the multi-step formula),
[01 · Targets and annualisation](01-forecast-targets-and-annualisation.md),
[04 · Clustering](04-volatility-clustering-and-the-acf.md). Code:
[`src/ivf/models/garch.py`](../../src/ivf/models/garch.py), [`src/ivf/models/rv21.py`](../../src/ivf/models/rv21.py).

## 1. First principles

- A daily return is a shock: it has (almost) zero mean, and its *size* is what clusters. Big moves
  follow big moves (concept 04). GARCH writes that down: today's variance is a weighted mix of a
  constant, yesterday's squared shock and yesterday's variance (Engle 1982; Bollerslev 1986).
- Because the mix is recursive, a forecast of the variance several days ahead decays from today's
  level back to a long-run level, at a speed set by the **persistence** (alpha + beta).
- Two refinements. **GJR** gives negative shocks an extra weight gamma (Glosten, Jagannathan & Runkle
  1993): falls raise volatility more than rises, the leverage effect. **EGARCH** models the *log*
  of the variance, so it is positive without constraints, with a size term and a sign term (Nelson 1991).
- The quantity we forecast, `rv21`, is the realised volatility of the next 21 sessions. Its square is
  a **sum** of 21 squared returns. So the forecast of it is a sum of 21 forecasts of daily variance,
  annualised the same way. That is all "variance aggregation" means.

## 2. The maths

With `e_t = 100 r_t` (percent; `arch` is fitted on these) and `sigma2` in percent²:

| Model | One-step recursion | k-step mean, from `s1 = sigma2_{t+1|t}` |
|---|---|---|
| GARCH(1,1) | `sigma2_t = omega + alpha e_{t-1}² + beta sigma2_{t-1}` | `sigma2_{t+k} = omega + (alpha + beta) sigma2_{t+k-1}` |
| GJR(1,1,1) | `+ gamma e_{t-1}² 1(e_{t-1} < 0)` | factor `alpha + gamma/2 + beta` (`1/2 = E[z² 1(z<0)]` for a symmetric z) |
| EGARCH(1,1,1) | `ln sigma2_t = omega + alpha(|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1} + beta ln sigma2_{t-1}` | `exp(omega Σ_{i<k-1} beta^i + beta^(k-1) ln s1) · Π_{i<k-1} M(beta^i)`, `M(c) = E[exp(c g(z))]` |

`z = e / sigma` is the standardised shock and `g(z) = alpha(|z| - sqrt(2/pi)) + gamma z`.

**The aggregate.** `rv21` at `t + 21` in variance space is `v = (252/21) Σ_{k=1}^{21} r_{t+k}²` (decimal,
no mean removed; ADR 0001 4.2). With a zero-mean return model, `E_t[r_{t+k}²] = sigma2_{t+k|t} / 100²`, so

    G_t = (252/21) · Σ_{k=1}^{21} sigma2_{t+k|t} / 100²        ŷ = 100 · sqrt(G_t)

`G_t` is `E_t[v]` exactly. Worked numbers (both are tests):

- GARCH with omega 0.02, alpha 0.08, beta 0.90 has a long-run daily variance of 0.02/0.02 = 1. From
  `s1 = 2`: `Σ = 21 + (1 − 0.98^21)/0.02 = 21 + 17.287 = 38.287`, so `G = 12 × 38.287 / 10⁴ = 0.045945`
  and `ŷ = 21.43` vol points.
- A constant 1 percent² a day gives `G = 12 × 21 / 10⁴ = 0.0252`, `ŷ = 15.8745`: exactly `rv21` of 21
  returns of 0.01. That is the units check against the target's own definition.

**Why `ŷ` is the RMS vol.** `100·sqrt(E_t[v])` is not `E_t[100·sqrt(v)]`: by Jensen it is larger.
`G` is the model's own forecast of the variance (`var_method = native`, ADR 0001 4.3), and QLIKE on
variance is the loss that ranks it fairly (Patton 2011). RMSE on vol points is reported, with that caveat.

## 3. Three places where the units go wrong

1. **Percent²** (`×100` returns): forgetting `/100²` makes `G` 10,000 times too big.
2. **Annualisation.** `252/21` and a 21-term sum. `252/22` with 21 terms, or 21 terms with `×252`
   (no division), each move every forecast by a constant factor that RMSE punishes but a correlation would not.
3. **The mean.** `rv21` does not remove the mean, so the model of `E[r²]` must not either: a constant-mean
   GARCH forecasts `E[(r − mu)²]`, short by `mu²` a day. The project uses a zero mean (decision D2 of
   the M3 walkthrough); the difference is small for daily index returns, but it is a definition, not a rounding.

## 4. Assumptions, and where they break

- **Stationarity.** The k-step formulas need the persistence below 1; otherwise the long-run level
  does not exist and the 21-day sum grows without bound. A fit at or above 1 is refused.
- **EGARCH with Student-t shocks has no 21-day expectation, in theory.** `M(c) = E[exp(c alpha |z|)]`
  is infinite for any polynomial-tailed z (the exponential beats the tail), so `E_t[sigma2_{t+k}]` is
  infinite for k ≥ 2. In practice the divergence comes only from |z| of a few hundred: review A measured,
  for t(6) and c = 0.1, a partial integral of 1.08 up to |z| = 200 and 211.5 up to 400. So a simulated
  mean is a stable, truncated value, with Monte Carlo noise. The project takes M as the exact mean over
  the training standardised residuals, **rescaled to unit RMS** so that `E[z²] = 1` holds (amended M3,
  review A): deterministic, finite, fixed at the cutoff, and exact for that empirical distribution, not
  for the fitted one. Derived here from the recursion; a literature reference is to verify.
- **The proxy is noisy.** `rv21` from 21 daily returns is a noisy measure of the true variance, so even
  a perfect forecast has a sizeable loss (Andersen & Bollerslev 1998); QLIKE stays consistent under such
  noise (Patton 2011).
- **P versus Q.** `G` forecasts the *realised* (P) variance. VIX² is a *risk-neutral* (Q) expectation of
  the same window, usually higher: the variance risk premium (concept 06). Scoring VIX as an `rv21`
  forecast (`vix_implied`) therefore shows a bias, by design.

## 5. In this project

- `garch.filter_variance` is `arch`'s recursion (tested to 1e-9 against `arch` 8.0.0), run through
  returns up to each origin with parameters frozen at the cutoff. `arch`'s own filter is not used for
  forecasting because its variance bounds use the whole sample.
- `garch.variance_path` and `garch.aggregate` are the table above; the GJR and GARCH paths equal
  `arch`'s analytic forecast; EGARCH's equals a brute-force enumeration of every shock path.
- Normal or Student-t innovations: both are fitted at every cutoff and the lower AIC is kept.
- `rv21.HarRv` is the HAR of concept 08 on realised *variance* (daily, weekly, monthly averages of
  `252 r²`), a strong non-GARCH competitor; `rv21.RandomWalkRv` uses the last 21 sessions' `rv21`.

## Self-check

1. Why does the 21-day forecast add variances, not volatilities?
2. A GARCH with alpha + beta = 0.98 starts at twice its long-run variance. How far above the long-run
   `G` is its 21-day `G`?
3. Why is a zero-mean model the one that matches `rv21`'s definition?
4. Why can a simulation not rescue EGARCH with Student-t shocks?
5. Which loss ranks a forecast of `E_t[v]` correctly, and why not RMSE on `ŷ`?

Sources: Engle (1982); Bollerslev (1986); Glosten, Jagannathan & Runkle (1993); Nelson (1991);
Hansen & Lunde (2005); Andersen & Bollerslev (1998); Patton (2011); Corsi (2009). The empirical-MGF
expectation is a deterministic form of filtered historical simulation; a reference for that name is to verify.
