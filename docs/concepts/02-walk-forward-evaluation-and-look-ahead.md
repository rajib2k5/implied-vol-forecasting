# 02 · Walk-forward evaluation and look-ahead

How to score a forecaster as it would have scored in real time, and how a backtest quietly sees
the future. Background: [00 · Background](00-background.md) (§5.2 on Diebold–Mariano).
Targets and units: [01](01-forecast-targets-and-annualisation.md). Numbers are computed with the
commands shown, or cited to the ADR that measured them.

## 1. First principles

**Train/test split.** Fit once on the early history, score on the rest: one frozen model, one test
block, parameters years stale by the end.

**Walk-forward.** Pick cutoffs $c_0 < c_1 < \dots$. At each cutoff $c_k$, fit using only data dated
$\le c_k$, then forecast from every origin $t \in [c_k, c_{k+1})$. Two choices:

- **Window.** *Expanding* (since `train_start`) keeps rare regimes such as 2008 and 2020; *rolling*
  (last $W$ sessions) adapts to breaks but forgets crises. ADR-0001 §4.5 expands for every model.
- **Refit cadence.** Every 63 sessions (about a quarter). Within a fold the **parameters are
  frozen**, but the **state** (GARCH recursion, HAR regressors, LSTM input window) updates daily
  with data $\le t$, so each forecast uses everything known at $t$.

**Identical folds for every model.** Model comparison is a *paired* problem. For the loss
differential $d_t = L_{a,t} - L_{b,t}$,

$$
\operatorname{Var}(d_t) = \operatorname{Var}(L_a) + \operatorname{Var}(L_b) - 2\operatorname{Cov}(L_a, L_b).
$$

Two vol models' losses blow up on the same spike. With sd 10 each and correlation 0.9,
$\operatorname{sd}(d) = \sqrt{100+100-180} = 4.47$, against $\sqrt{200} = 14.14$ unpaired. Pairing
needs identical origins and information: what the Diebold–Mariano (DM) test needs in M5.

## 2. Origin, horizon, target, and label availability

Index sessions on the master calendar $\mathcal{T}$. A forecast at **origin** $t$ for **horizon**
$h$ targets $y_{t+h}$, the value at the $h$-th session after $t$. A supervised training pair is
$(x_\tau, y_{\tau+h})$. The **label-availability rule** (ADR-0001 §4.5, L2): at cutoff $c$,

$$
(x_\tau, y_{\tau+h}) \text{ is usable only if } \tau + h \le c .
$$

**Worked timeline**, $h = 21$, cutoff at session $c = 100$:

| pair origin $\tau$ | target date $\tau+21$ | usable at $c = 100$? |
|---:|---:|---|
| 78 | 99 | yes |
| 79 | 100 | yes, the last one |
| 80 | 101 | no: label dated 1 session after the cutoff |
| … | … | no |
| 100 | 121 | no: this *is* the target of the fold's first forecast (origin 100) |

Without the rule, the 21 pairs $\tau = 80,\dots,100$ leak labels up to 21 sessions past the cutoff;
for $RV_{21}$ the pair at $\tau = 100$ holds $r_{101},\dots,r_{121}$, the whole window the first
forecast must predict. The rule costs 21 pairs out of thousands.

## 3. The look-ahead rules L1–L7

| Rule (ADR-0001 §4.7) | How it would leak without the rule |
|---|---|
| L1 trailing windows only | `rolling(21, center=True).mean()` at $t$ averages $t-10..t+10$. `bfill` fills a missing VIX3M at $t$ with the close of $t+1$. |
| L2 label availability | The timeline above. |
| L3 scalers fitted per cutoff on update rows only | An LSTM scaler fitted on inner-train **plus** early-stopping rows moves the training inputs by the validation block's mean (§5 shows the size). |
| L4 early-stopping block with an $h$-session purge | Validation targets after the cutoff pick the epoch from future labels. Without the purge, inner-train and validation labels overlap (shared $RV_{21}$ returns), so validation loss is optimistic. |
| L5 fixed regime thresholds | "Low vol = below the full-sample 33rd percentile" used in 2003 encodes the 2008 and 2020 distribution. |
| L6 hyper-parameters chosen on dev only | Picking HAR's lag set by holdout RMSE turns the holdout into a training set (§7). |
| L7 trade at the close of $t+1$ | *If* the VIX close is struck at 16:15 ET, after VXX's 16:00 close (ADR-0001 §4.1: to verify in M1; Cboe's methodology only says VIX is disseminated until 4:15 p.m.), a signal from VIX at $t$'s close cannot be filled at VXX's $t$ close. |

## 4. The truncation test as a property test

If the pipeline is causal, a forecast at origin $t$ cannot depend on any data after $t$. So for
**every** $t$:

$$
\text{forecast}_t\big(\text{run on data}_{\le t}\big) = \text{forecast}_t\big(\text{run on all data}\big)
$$

(the LSTM to within a tolerance). It is a *property*, so test many origins, not one hand-picked
date; a seeded random sample does that without adding `hypothesis`. A sketch for M2, not existing
code:

```python
for t in rng.choice(origins, size=50, replace=False):      # seeded numpy Generator
    full, cut = run(data, folds), run(data[data.date <= t], folds)
    assert_close(cut.forecast_at(t), full.forecast_at(t))
```

It catches centred windows, `bfill`, full-sample scaling and label leaks. It **cannot** catch a
leak identical in both runs: a threshold chosen by eye from the full sample, a hyper-parameter tuned
on the holdout (L6), a data restatement. Two rules in ADR-0001 §4.1 and §4.5 make it possible:

- **Calendar and folds come from dates, not data.** $\mathcal{T}$ is the NYSE calendar from
  `exchange_calendars`. Cutoffs `c_k = idx(holdout_start) + 63k` are anchored *backwards*, so a run
  truncated before `holdout_start` could not derive them; the schedule is a pure function of
  ($\mathcal{T}$, `dev_start`, `holdout_start`, `fold_len`), passed into every run (`folds` above).
- **Forecasting is separate from scoring.** Forecasts are made at every origin, including those with
  unobserved targets; the "evaluable origin" rule (`idx(origin) + h ≤ idx(data_end)`) only filters
  scoring, so a run truncated at $t$ still forecasts at $t$.

## 5. Scaling: fit on train only

**Worked example.** Train = $[12, 14, 13, 15]$, then a spike, test = $[35, 40]$. Population sd
(as in scikit-learn's `StandardScaler`):

- Train-only: mean 13.5, sd 1.118; training z-scores $[-1.34, 0.45, -0.45, 1.34]$.
- Full sample: mean $129/6 = 21.5$, sd 11.442; training z-scores $[-0.83, -0.66, -0.74, -0.57]$,
  all negative. The scaled training data now "know" that a higher level is coming.

An AR(1) in z-space with $\phi = 0.9$ forecasts $\hat x = m + \phi(x_t - m) = (1-\phi)m + \phi x_t$
(the sd cancels). From $x_t = 15$: train-only $13.5 + 0.9 \times 1.5 = 14.85$; full-sample
$21.5 + 0.9 \times (15 - 21.5) = 15.65$. The 0.8 gap is exactly $(1-\phi)(21.5 - 13.5)$: reversion
to a mean only the future reveals. An estimated $\phi$ would carry the leak too.

## 6. Overlapping forecasts and why DM needs HAC

Consecutive $h$-step forecasts share target windows. For a random walk $V_t$ with i.i.d.
increments $\varepsilon$ (variance $\sigma^2$), the $h$-step no-change forecast error is

$$
e_t = V_{t+h} - V_t = \sum_{j=1}^{h}\varepsilon_{t+j},
\qquad
\operatorname{Cov}(e_t, e_{t+k}) = (h-k)\,\sigma^2 \;\; (0 \le k < h), \;\; 0 \text{ for } k \ge h .
$$

So $e_t$ is MA($h-1$) with autocorrelation $(h-k)/h$. For $h = 21$: 0.9524 at lag 1, 0.5238 at
lag 10, 0.0476 at lag 20. A 200,000-step simulation (seed 0) gave 0.9521, 0.5277, 0.0573, then
0.0091 at lag 21 and 0.0080 at lag 25 (noise).

The variance of a mean of such errors uses the **long-run variance**
$\sum_{k=-(h-1)}^{h-1}\gamma_k = \sigma_e^2\big[1 + 2\sum_{k=1}^{h-1}(h-k)/h\big] = h\,\sigma_e^2$.
The i.i.d. formula is therefore too small by a factor $\sqrt{21} = 4.58$ in standard error, and a
DM test would reject far too often. DM's test estimates the long-run variance with a HAC estimator
truncated at lag $h-1$. **Kernel choice matters.** Newey–West's Bartlett weights $1 - k/h$
at lag $h-1$ give, even in population, $\sigma_e^2[1 + (h-1)(2h-1)/(3h)] = 14.02\,\sigma_e^2$ instead
of $21\,\sigma_e^2$ here: a standard error 0.817 times the truth. Equal (rectangular) weights,
$\hat\gamma_0 + 2\sum_{j=1}^{h-1}\hat\gamma_j$, are exact for MA($h-1$) in population. ADR-0001 §4.8
(as amended) therefore uses:

- the **rectangular** estimator at lag $h-1$, as in the original DM paper;
- the Harvey–Leybourne–Newbold factor $\sqrt{(n + 1 - 2h + h(h-1)/n)/n}$ with $t_{n-1}$ p-values
  ($= 0.9648$ for $n = 583$, $h = 21$), derived by HLN (1997) for exactly this estimator: eq. 5
  writes the estimator, eqs. 6–8 derive its expectation, eq. 9 (p. 283) gives the factor;
- a **Bartlett fallback**, flagged per row, if the rectangular estimate is $\le 0$.

The ADR's simulation (20,000 null runs, $h = 21$, $n = 583$, nominal 5%) measured rejection
rates of 0.0636 (rectangular + HLN) and 0.1189 (Bartlett + HLN); cited, not re-run.

**The lag-$(h-1)$ cut-off is an assumption.** DM and HLN (p. 282) *assume* $d_t$'s autocorrelations
vanish beyond lag $h-1$. Optimal *errors* are MA($h-1$), but that need not carry over to $d_t$ under
nonlinear losses, and it fails when relative accuracy depends on a persistent vol state, the normal
case here. Review B's state-dependent null ($E[d] = 0$; one model's error scaled by a log-variance
AR(1), $\phi = 0.98$) gave rectangular + HLN rejection rates of **0.1258** ($n = 583$) and **0.1103**
($n = 2{,}000$), against 0.0563 and 0.0507 for a symmetric control (ADR-0001 §4.8; cited, not re-run).
P-values can be about 2.5 times too liberal; open item O6 has M5 pick a remedy by measurement and
report bandwidth sensitivity. Mincer–Zarnowitz still uses Newey–West at lag $h-1$ (O5). For real
data only the DM statistic and p-value against `rw` are published (2 dp), and MZ only as pass/fail
booleans, for rows that are scale-free given public SPX; `mean_d` $= RMSE_a^2 - RMSE_b^2$ never is
(ADR-0001 D2; [03 §8](03-reproducibility-and-data-contracts.md)).

**Off-the-shelf DM is not this estimator.** In the pinned statsmodels 0.15.0 (source read in the
repo `.venv`), `diebold_mariano_test` defaults to `lags = max(h-1, ceil(nobs**(1/3)))`, which is 9
at $h = 1$, $n = 583$, and it regresses $d_t$ on a constant with `cov_type="HAC"`: Bartlett weights.
With overlapping 21-step random-walk errors as $d_t$ and `lags=20` it returned $t = -1.3318$, equal
to hand-coded Bartlett; rectangular gives $-1.0440$. M5 must pin the lags and code the rectangular
estimator.

## 7. The holdout

ADR-0001 §4.6 and open item O1 propose a **fixed** `holdout_start` of 2024-07-01, not "the last 24
months". Measured on 2026-10-27 (ADR-0001): a trailing window would start 2024-10-28 and miss the
5-Aug-2024 spike, which 2024-07-01 makes a holdout target at every horizon. The holdout ran to
2026-10-26: 583 sessions $= 9 \times 63 + 16$, i.e. 10 folds, the last partial. A rolling
definition moves at every refresh, sliding seen dates into "out of sample". The 50 sessions
2000-01-03..2000-03-14 are burn-in: training data, never origins (ADR-0001 §4.5).

**Touched once.** Each look-then-change is a selection step. If $N$ skill-free models have
independent standard-normal holdout $t$-statistics, the best has expected $t = E[\max_{i\le N} Z_i]$:
1.54 for $N = 10$, 2.51 for $N = 100$ (below; correlation between real models shrinks this). Hence
`protocol.holdout_looks`, retired-not-deleted `strategy_def` rows for DSR's `n_trials`, and the dev
purge (tuning uses dev rows with `target_date < holdout_start`).

## Reproduce

```bash
uv run python - <<'EOF'
import math, numpy as np
from scipy import integrate, stats
h=21; X=np.cumsum(np.random.default_rng(0).standard_normal(200_000+h)); e=X[h:]-X[:-h]; e-=e.mean()
print([round(float(e[:-k]@e[k:]/(e@e)),4) for k in (1,10,20,21,25)])   # 0.9521 0.5277 0.0573 0.0091 0.008
print(1+(h-1)*(2*h-1)/(3*h), math.sqrt(14.0159/21))                     # 14.0159 0.817
print(math.sqrt((583+1-2*h+h*(h-1)/583)/583))                            # 0.9648 (HLN factor)
x=np.array([12.,14,13,15,35,40]); tr=x[:4]
print(tr.mean()+0.9*(15-tr.mean()), x.mean()+0.9*(15-x.mean()))          # 14.85 15.65
for N in (10,100): print(integrate.quad(lambda z: z*N*stats.norm.pdf(z)*stats.norm.cdf(z)**(N-1),-10,10)[0])  # 1.5388 2.5076
from statsmodels.tsa.stattools import diebold_mariano_test as dm        # statsmodels' kernel and default lags:
W=np.cumsum(np.random.default_rng(1).standard_normal(604)); d=W[21:]-W[:-21]; z=0*d; c=d-d.mean(); G=[c[j:]@c[:583-j]/583 for j in range(21)]
t=lambda w: d.mean()/np.sqrt(sum(w[j]*G[j]*(1+(j>0)) for j in range(21))/583)
print(dm(d,z,d,lags=20,horizon=21,criterion=lambda y,f:y-f).statistic, t([1-j/21 for j in range(21)]), t([1]*21), dm(d,z,d,horizon=1).lags)  # -1.3318 -1.3318 -1.0440 9
EOF
```

## Assumptions

- Data dated $t$ is fully known at $t$'s close (ADR-0001 §4.1); history is the vintage as fetched.
- A quarterly refit loses little: ADR-0001 §4.5 cites a first fit of 2,579 sessions (measured).
- The dependence of $d_t$ ends at lag $h-1$: assumed, and known to fail under vol clustering (§6).

## Where it breaks

- **Structural breaks**: an expanding window averages over them.
- **Vintage**: restated data is a leak the truncation test cannot see.
- **Purging is not free**: it drops the newest labels, which matter most just after a spike.

## How this repo implements it

- Protocol: ADR-0001 §4.1 (calendar), §4.5–§4.7 (folds, holdout, L1–L7, truncation test), §4.8
  (DM, MZ) in [the ADR](../adr/0001-python-core-versioned-json-app.md).
- Storage: `fold`, `fit` (`train_last_target_date ≤ cutoff_date` is L2's tested witness) and
  `forecast` in [data-model.md](../architecture/data-model.md) §4.
- **Code:** none yet (folds and the truncation test in M2, DM in M5). Only the protocol constants
  exist: `tools/export_app_data.py:64-74` (`M0_PROTOCOL`).

## Self-check

1. Cutoff $c = 500$, $h = 5$. Which training pairs may the fit use, and what would including
   $\tau = 500$ let the model see?
2. You compare two models' $h = 5$ forecasts on 1,000 origins with a plain $t$-test on $d_t$. What is
   wrong, by roughly how much, and does Newey–West with lag 4 fully fix it?
3. Why a fixed `holdout_start`, and why must the holdout be scored only once?

<details>
<summary>Answers</summary>

1. Pairs with $\tau \le 495$. The five pairs $\tau = 496..500$ have targets $501..505$ after the cutoff;
   $\tau = 500$ targets session 505, the exact target of the fold's first forecast (origin 500).
2. Overlapping 5-step errors are MA(4). For random-walk errors the lag-1 autocorrelation is
   $4/5 = 0.8$ and the long-run variance is $5\sigma_e^2$, so the naive standard error is too small by
   $\sqrt5 = 2.24$. Bartlett weights at lag 4 give $1 + 4 \times 9/15 = 3.4$ instead of 5: the standard
   error is still $\sqrt{3.4/5} = 0.82$ of the truth. Hence rectangular + HLN, Bartlett as fallback,
   and even that only holds if $d_t$'s dependence really stops at lag $h-1$.
3. A trailing window moves with each refresh, leaking seen dates into "out of sample" (and on
   2026-10-27 it excluded Aug-2024). Every look-then-change is selection: the best of 10 skill-free
   models has expected $t \approx 1.54$; `holdout_looks` and DSR's `n_trials` record it.

</details>
