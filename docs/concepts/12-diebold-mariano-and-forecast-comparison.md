# 12. Diebold-Mariano, and comparing many forecasts honestly

*Milestone 5. Code: `src/ivf/evaluation/dm.py`, `src/ivf/evaluation/report.py`. Walkthrough:
[milestone 5](../milestones/05-evaluation.md).*

## The question

Model A's RMSE is 4.51 and model B's is 4.95. Is A better, or did it get a lucky stretch? The
Diebold-Mariano (DM) test (Diebold & Mariano 1995) answers that with one series. At each origin t,
take the loss of A minus the loss of B:

```
d_t = L_A,t - L_B,t          (squared error, or QLIKE)
```

If the two are equally accurate, `E[d] = 0`. The test is a t-test on the mean of `d`:
`DM = mean(d) / sqrt(LRV / n)`. Negative means A has the smaller loss.

## Why the variance is the hard part

A 21-session forecast made today and one made tomorrow share 20 of their 21 future sessions, so
their errors overlap. `d_t` is then correlated with `d_{t-1}`, ..., `d_{t-20}`, and the plain variance
of `d` understates how noisy its mean is. The **long-run variance** (LRV) adds the autocovariances:

```
LRV = gamma_0 + 2 * sum_j w_j * gamma_j
```

The weights `w_j` and how many lags to use are the whole design problem:

| Estimator | Weights | What goes wrong |
|---|---|---|
| Rectangular at lag h−1 (DM 1995) | 1 up to lag 20 | Exact when `d` is correlated only up to lag 20, as for two forecasts whose only dependence is the overlap. Can come out negative (then Bartlett is used). |
| Newey-West (Bartlett) at lag h−1 | `1 − j/21` | Shrinks lags that matter fully: at h = 21 it captures about two thirds of the true LRV of an overlapping error (ADR 0001 4.8), so the test rejects about twice as often as it should. |
| Andrews (1991) automatic | Bartlett, bandwidth from an AR(1) fit to `d` | Adapts to persistence, but is still biased downward when the persistence is long. |
| Fixed-b (Kiefer-Vogelsang) | Bartlett with a bandwidth that is a fixed share b of the sample (here 10%) | Uses many lags, and accepts that the LRV is then noisy: the critical values come from the statistic's own distribution under that bandwidth, not from the normal. |

The Harvey-Leybourne-Newbold (1997) factor `sqrt((n + 1 − 2h + h(h−1)/n)/n)`, with t(n−1) p-values,
corrects the rectangular estimator's small-sample bias. It was derived for that estimator, so the
project applies it only to the two lag-(h−1) estimators.

## The trap: relative accuracy that follows a state

DM's lag-(h−1) truncation *assumes* `d` has no autocorrelation past lag h−1. For volatility
forecasts that is often false. If model A is worse when markets are calm and better when they are
stressed, `d` follows the volatility regime, which lasts months. The overlap is then the smaller
part of the dependence. Milestone 5 simulated exactly that (one model's errors scaled by a
persistent state, φ = 0.98) and measured at the dev sample size: the rectangular test rejected a
true null 10.6% of the time at a nominal 5%, the fixed-b test 5.6%
([walkthrough, 7.1](../milestones/05-evaluation.md#71-dm-size-o6)). That is why the project's primary
DM test is fixed-b.

The price is power. A noisier variance estimate means fatter-tailed critical values (the simulated 5% value is
about 2.25 instead of 1.96 at b = 0.1; its agreement with Kiefer-Vogelsang's tables is not verified),
so fewer differences are declared significant. And even fixed-b is not size-controlled on a sample as
short as the holdout's (about 583 origins), so holdout DM p-values are descriptive only.

## Many comparisons at once

Seven models give 21 pairs per table. At 5% each, about one pair would look "significant" by chance
even if every model were equally good. **Holm's** step-down rule (Holm 1979; not cited in this repo
before, to verify) controls the chance of *any* false rejection in a family:

1. sort the m p-values ascending;
2. multiply the k-th smallest by (m − k + 1);
3. make the results non-decreasing in that order, and cap at 1.

Reject where the adjusted p-value is below 5%. Holm is valid however the tests are correlated (and
pairs that share a model are). Its family here is one table: the pairs of one (target, horizon, loss,
estimator).

The alternative considered was Hansen, Lunde and Nason's (2011) **Model Confidence Set**: the set of
models that cannot be told apart from the best. It answers a better question ("which models are
best?"), but it needs a block bootstrap of the loss differentials, whose block length runs into the
same persistence problem as the LRV, and its size here was not measured.

## Reading a DM row

The sign convention (amended M5, review A): `d = L_a − L_b`, where `model_a` is the pair's
alphabetically first model. A **positive** `mean_d` means `model_a` has the larger loss, so `model_a` is
worse; the `lower_loss` column names the better one.

(Illustrative numbers.) `model_a = har_log`, `model_b = rw`, `loss = sq_error`, `mean_d = −3.2`, `p_holm = 0.001`: HAR's
squared error is on average 3.2 vol-points² smaller than the random walk's, and that difference
survives the multiple-comparison correction. A row with `p_holm = 0.7` says the data cannot tell the
two apart; it does not say they are equal.

## Check yourself

1. Why is `d_t` autocorrelated even when each model's one-step errors are not?
2. What does Newey-West at lag h−1 get wrong for an MA(h−1) differential, in one sentence?
3. Why does a fixed-b test need its own critical values?
4. Holm on p = (0.01, 0.04, 0.03, 0.005): which survive at 5%? (Answer: 0.005 → 0.02 and 0.01 → 0.03;
   0.03 and 0.04 both → 0.06.)
