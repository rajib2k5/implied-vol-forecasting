# 04 · Volatility clustering, the ACF and the Ljung-Box test

Why tomorrow's *size* of move is predictable when its *direction* is not, and how milestone 1
measures that. Background: [00 · Background](00-background.md). Numbers come from the commands
shown, run on 2026-10-29, on **synthetic** data. Real-data results are measured locally, not
published (ADR 0001 D2).

## 1. First principles

- A daily log return is `r_t = ln(S_t / S_{t-1})`. Its sign is close to a coin flip. If it were
  predictable, traders would trade it away.
- Its **size** is not a coin flip. Big moves come in bunches, and so do quiet days. That is
  *volatility clustering*. A calm year and a crisis year look like different worlds.
- Clustering shows up as a pattern in `r²` or `|r|`, not in `r`. So we test `r`, `r²` and `|r|`
  separately and compare.
- This is the whole reason volatility forecasting works. GARCH and HAR models are machines for
  turning "recent moves were big" into "the next ones will be big too".

## 2. The maths

### The sample autocorrelation (ACF)

For a series `x_1 .. x_n` with mean `m`, the lag-k autocorrelation is

```
rho_k = sum_t (x_t - m)(x_{t-k} - m) / sum_t (x_t - m)^2
```

Every lag divides by the same lag-0 sum. That is the textbook estimator and the one statsmodels
uses (`adjusted=False`).

**Worked number** (the test is `test_acf_of_1_2_3_4_is_0_25_at_lag_1`): `x = 1, 2, 3, 4`. Then
`m = 2.5`, the deviations are `-1.5, -0.5, 0.5, 1.5`, the lag-0 sum is
`2.25 + 0.25 + 0.25 + 2.25 = 5`, and the lag-1 sum is `(-0.5)(-1.5) + (0.5)(-0.5) + (1.5)(0.5) = 1.25`.
So `rho_1 = 1.25 / 5 = 0.25`.

**Gaps.** Milestone 1 keeps a missing session as a gap (gap policy G2). A lag-k pair needs both
sessions, k sessions apart *on the calendar*, to have a value. With `x = 1, 2, gap, 4, 5` the
pairs are `(2,1)` and `(5,4)`, so `rho_1 = (2 + 2)/10 = 0.4`. Deleting the gap first would pair 4
with 2 and give 0.3: a two-session move counted as one. The test is
`test_acf_never_pairs_values_across_a_gap`.

**How big is "big"?** If `x` is independent noise, `rho_k` is roughly normal with sd `1/sqrt(n)`.
For `n = 9,272` sessions that is `1/96.3 = 0.0104`, so a 95% band is about `±0.0204`.

### The Ljung-Box test

One autocorrelation can look large by luck. Ljung-Box pools the first `m` of them:

```
Q(m) = n (n + 2) * sum_{k=1}^{m} rho_k^2 / (n - k)
```

Under the null "no autocorrelation up to lag m", `Q(m)` is roughly chi-squared with `m` degrees
of freedom. A tiny p-value rejects the null.

**Worked numbers.**
- `x = 1, 2, 3, 4`: `n = 4`, `rho_1 = 0.25`, so `Q(1) = 4 * 6 * 0.0625 / 3 = 0.5` and
  `p = P(chi2_1 > 0.5) = 0.4795` (`test_ljung_box_on_1_2_3_4_is_q_0_5`).
- `n = 100`, `rho_1 = 0.2`: `Q(1) = 100 * 102 * 0.04 / 99 = 4.1212`, `p = 0.0423`.

The test `test_ljung_box_agrees_with_statsmodels` checks the formula against
`statsmodels.stats.diagnostic.acorr_ljungbox`, an independent implementation.

### What GARCH(1,1) predicts for the ACF of r²

With `r_t = sqrt(h_t) z_t`, `h_t = omega + alpha r_{t-1}^2 + beta h_{t-1}` and normal `z_t`,
the ACF of `r²` is (Bollerslev 1988):

```
rho_1 = alpha (1 - alpha beta - beta^2) / (1 - 2 alpha beta - beta^2),   rho_k = rho_1 (alpha + beta)^(k-1)
```

It needs a finite fourth moment: `3 alpha^2 + 2 alpha beta + beta^2 < 1`.

**Worked number:** `alpha = 0.10, beta = 0.80` gives
`rho_1 = 0.10 * (1 - 0.08 - 0.64) / (1 - 0.16 - 0.64) = 0.10 * 0.28 / 0.20 = 0.14`, and
`rho_5 = 0.14 * 0.9^4 = 0.0919`. On a simulated path of 200,000 returns the sample gave values
inside `±0.025` of both; over 20 seeds the sample `rho_1` had sd 0.0054
(`test_garch_squared_returns_cluster_with_the_closed_form_acf_while_returns_do_not`).

### On the synthetic market

`uv run python -m ivf.data --synthetic fetch`, then `build-interim`, `verify`, `check` and `eda`,
with `VOL_DATA_DIR` set to a scratch folder (2026-10-29, re-run after review A). The EDA uses the
8,689 sessions before `holdout_start`, 8,688 returns (amended M1, review A: the first version used
every session to `data_end`, holdout included). The generator uses `alpha = 0.08`, `beta = 0.90`,
so the true `rho_1(r²) = 0.08 * 0.118 / 0.046 = 0.2052`.

| Series | ACF lag 1 | Ljung-Box Q(21) | p |
|---|---:|---:|---:|
| `r` | -0.0090 | 18.7 | 0.60 |
| `r²` | 0.1736 | 3,809 | 0.0 (below double precision) |
| `abs(r)` | 0.1519 | 2,821 | 0.0 |

`r` passes as noise; `r²` and `|r|` fail by a mile. That is the clustering signature.

## 3. Assumptions

- **Ljung-Box assumes independence under the null**, not just zero correlation. For returns
  with GARCH effects, `r` can be uncorrelated yet dependent, and the test on `r` itself can then
  reject too often (a result from the literature; not measured here). We use it on `r` only as a
  contrast, never as a model check.
- **The chi-squared reference is asymptotic.** With 9,000 sessions and `m = 21` it is fine.
- **Stationarity.** An ACF assumes the mean and variance do not drift. Returns roughly satisfy
  it; prices do not, which is why we never take the ACF of the S&P 500 level.

## 4. Where it breaks

- **Heavy tails make the r² ACF noisy and biased low.** With `alpha + beta = 0.98` the fourth
  moment is barely finite (`3(0.0064) + 2(0.072) + 0.81 = 0.9732`). Over 20 seeds of 9,272
  sessions the sample `rho_1(r²)` had mean 0.193 against a true 0.205, and ranged from 0.136 to
  0.255 (scratch script, 2026-10-29). So a single path cannot pin GARCH parameters from its ACF.
  The milestone uses the ACF for a yes/no answer (statement E1), not to estimate parameters.
- **A huge Q is not a model.** It says "not noise", not "GARCH(1,1)". Asymmetry (GJR) and long
  memory (HAR) also give a slowly decaying ACF of `r²`.
- **Gaps lower the effective n.** Each missing session removes up to two returns and several
  lag pairs. The code uses the count of non-missing values as `n`, a small approximation when
  gaps are few.

## 5. In this project

- [`src/ivf/eda.py:39`](../../src/ivf/eda.py#L39): the pairwise-complete ACF. Missing values contribute zero to the
  sums, so no pair crosses a gap.
- [`src/ivf/eda.py:71`](../../src/ivf/eda.py#L71): `Q(m)` for every `m`, with chi-squared p-values.
- [`src/ivf/features/build.py:45`](../../src/ivf/features/build.py#L45): returns only between consecutive sessions of 𝒯.
- [`src/ivf/data/synthetic.py:142`](../../src/ivf/data/synthetic.py#L142): the closed form the tests compare with.
- [`src/ivf/data/process.py:154`](../../src/ivf/data/process.py#L154): the three series in the local report.
- Tests: `tests/test_eda.py` (hand-worked values, statsmodels agreement, GARCH recovery).

## Self-check

1. Why does Ljung-Box on `r` say "noise" while on `r²` it says "not noise"? What does that tell
   you about the *sign* and the *size* of returns?
2. A colleague computes the ACF after `dropna()`. What goes wrong at a gap, and how big is the
   error for `x = 1, 2, gap, 4, 5`?
3. Why is the sample ACF of `r²` a poor way to estimate GARCH's `alpha` and `beta`?
