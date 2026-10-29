# 05 · Persistence, the AR(1) half-life and unit-root tests

How long a VIX shock lasts, how to put a number and an interval on it, and how to test whether
VIX returns to a mean at all. Background: [00 · Background](00-background.md). Numbers come from
the commands shown, run on 2026-10-29, on **synthetic** data. Real-data results are measured
locally, not published (ADR 0001 D2); only the half-life per whole period is in the publication
set P, and M1 publishes nothing.

## 1. First principles

- VIX today is a very good guess for VIX tomorrow. That is **persistence**: the ACF of VIX
  decays slowly.
- But VIX does not wander off for ever. After a spike it drifts back down; after a calm spell it
  drifts up. That is **mean reversion**.
- An AR(1) model holds both ideas in one number, `phi`. Each session a deviation from the
  long-run mean shrinks to `phi` times itself. `phi` near 1 means slow reversion; `phi = 1` means
  none at all (a random walk, a *unit root*).
- We model **ln VIX**, not VIX. Logs turn "VIX rises 20%" into the same step at any level, and
  ln VIX is closer to normal. The half-life is the same under any rescaling of VIX
  (`ln(cV) = ln c + ln V`), which is why ADR 0001 D2 allows it in the publication set P.

## 2. The maths

### AR(1) and the half-life

```
x_t = c + phi * x_{t-1} + e_t,      long-run mean mu = c / (1 - phi)
x_{t+h} - mu = phi^h (x_t - mu)     (expected path, no new shocks)
```

A deviation halves when `phi^h = 0.5`, so

```
half-life = ln(0.5) / ln(phi)   sessions
```

**Worked numbers** (`test_half_life_is_ln_half_over_ln_phi`):
- `phi = 0.98`: `ln 0.5 / ln 0.98 = -0.693147 / -0.020203 = 34.31` sessions, about 7 weeks.
- `phi = 0.5`: exactly 1 session. `phi = 0.25`: 0.5 sessions.
- `phi >= 1`: never halves (infinite). `phi <= 0`: the path oscillates; no half-life.

The function is very steep near 1: `phi = 0.97` gives 22.8 sessions and `phi = 0.99` gives 69.0.
A small error in `phi` is a big error in the half-life.

### Estimating phi with an interval

OLS of `x_t` on `(1, x_{t-1})` over every pair of **consecutive sessions** where both values
exist. The interval for `phi` is `phi_hat ± 1.96 se`. Because the half-life is an increasing
function of `phi` on `(0, 1)`, mapping both ends of `phi`'s interval through the formula gives
the half-life's interval exactly, without a delta method. An upper end at or above 1 gives an
infinite upper half-life: the data cannot rule out a unit root at that confidence. A lower end at
or below 0 gives a lower half-life of **0** (amended M1, review B; the first version gave nan, so
white noise reported `(nan, 0.245)`): the half-lives the interval allows are those of its `phi` in
`(0, 1)`, and `ln(0.5)/ln(phi)` falls to 0 as `phi` falls to 0. The point estimate stays undefined
for `phi <= 0`, which is no decay. Worked numbers: the interval `(-1.2152, 0.4880)` maps to
`(0, ln 0.5/ln 0.4880) = (0, 0.9660)` sessions.

The standard error is White's heteroskedasticity-robust HC1 by default. Shocks to ln VIX are
larger in stressed markets, and plain OLS errors then understate the uncertainty. On a test
series whose shock size grows with `|x_{t-1}|`, HC1 errors were 11 to 14 times the OLS errors
(seeds 0-4); on homoskedastic data they agree within 10%
(`test_hc1_and_ols_errors_agree_on_homoskedastic_data_and_differ_on_heteroskedastic`).

**OLS is biased low in small samples.** Kendall (1954): the bias in `phi_hat` is about
`-(1 + 3 phi)/n`. At `phi = 0.98, n = 9,000` that is `-3.94/9000 = -0.00044`, which moves the
half-life from 34.31 to about 33.6 sessions. At `n = 583` (a holdout-sized sample) it is
`-3.94/583 = -0.0068`, and the half-life at `phi = 0.9732` is 25.5 sessions: a 26% shortfall from
bias alone.

**Recovery** (`test_ar1_recovers_phi_and_the_interval_covers_it`): 20,000 simulated sessions
with `phi = 0.98`. Over 20 seeds `phi_hat` had sd 0.00124 and the half-life sd 2.1 sessions; the
95% interval covered 0.98 on 19 of 20 seeds (scratch script, 2026-10-29).

**On the synthetic market** (`uv run python -m ivf.data --synthetic eda`, 2026-10-29, re-run after
review A; truth `phi = 0.98`, half-life 34.31). The EDA sees only sessions before `holdout_start`
(amended M1, review A): `pre_holdout` is all of them, `burn_in` and `dev` are the calendar's
periods inside it. The first version also fitted the holdout, a look ADR 0001 4.6 forbids before
the models are frozen, and its "dev" row (origin < 2024-07-01) was in fact burn-in plus dev, the
`pre_holdout` row below.

| Sample | n pairs | phi_hat (95% CI) | Half-life, sessions (95% CI) |
|---|---:|---|---|
| pre_holdout (1990-01-02..2024-06-28) | 8,688 | 0.9813 (0.9773, 0.9854) | 36.8 (30.2, 47.1) |
| burn_in (to 2000-03-14) | 2,577 | 0.9824 (0.9752, 0.9897) | 39.1 (27.6, 66.9) |
| dev (2000-03-15..2024-06-28) | 6,110 | 0.9808 (0.9759, 0.9857) | 35.8 (28.4, 48.2) |

Each interval covers the truth; the shorter burn-in has the wider one.

### Unit-root tests: two tests, opposite nulls

| Test | Null hypothesis | A small p-value means |
|---|---|---|
| ADF (augmented Dickey-Fuller) | a unit root: no mean reversion | evidence **for** mean reversion |
| KPSS | (level) stationarity: mean reversion | evidence **against** mean reversion |

ADF regresses `Δx_t` on `x_{t-1}` plus lagged differences and tests whether the coefficient on
`x_{t-1}` (which is `phi - 1`) is zero. Its statistic does not have a t distribution under the
null; it uses Dickey-Fuller critical values (about -2.86 at 5% with a constant). KPSS measures
how far the partial sums of `x_t - mean` wander, scaled by a long-run variance estimate.

"Stationary at 5%" (statement E7 in ADR 0001 D2) means ADF rejects **and** KPSS does not.

**Measured** (scratch script, 2026-10-29, 100 seeds each):

| Simulated series | ADF rejects | KPSS rejects | Verdict "stationary" |
|---|---:|---:|---:|
| AR(1), `phi = 0.5`, n = 5,000 | 100% | 7% | 93% |
| AR(1), `phi = 0.9`, n = 5,000 | 100% | 8% | 92% |
| AR(1), `phi = 0.98`, n = 9,000 | 100% | 31% | 69% |
| AR(1), `phi = 0.98`, n = 583 (holdout-sized; re-run 2026-10-29 after review A) | 44% | 61% | 23% |
| Random walk, n = 2,000 (20 seeds) | 0% | 100% | 0% |

## 3. Assumptions

- **One AR(1) for the whole sample.** The half-life is a single number, so it assumes the
  speed of reversion is the same in 1990, 2008 and 2024. The per-period estimates (burn-in,
  dev) are a rough check of that.
- **Linear, Gaussian-ish dynamics in ln VIX.** Spikes are sharp and decays slow; an AR(1)
  averages the two speeds.
- **HC1 errors assume no autocorrelation in the residuals.** If an AR(1) is too simple (HAR's
  premise), the residuals are autocorrelated and the interval is too narrow.
- **ADF and KPSS need an unbroken series.** Both lag and difference consecutive values, so
  `unit_root_tests` runs them on the longest run of sessions with no gap, and reports its length
  and dates beside the number of values it left out (gap policy G2). Amended M1, review A: the
  first version dropped the missing values instead, which joins the neighbours of every gap into
  one fake one-session move, the bridge G2 forbids. On real data the core gaps are all in burn-in,
  so the pre-holdout test drops the sessions before the last of them.

## 4. Where it breaks

- **KPSS over-rejects for persistent series.** In the table above it rejected a true stationary
  AR(1) with `phi = 0.98` on 31% of seeds, six times its nominal 5%. Its automatic bandwidth is
  too short for that much persistence, and a longer one does not cure it (review A: about 137
  lags still rejected on 12-20%). So for ln VIX, a "not stationary" verdict from KPSS is weak
  evidence.
- **At holdout size the E7 verdict is uninformative.** On 100 stationary AR(1) paths with
  `phi = 0.98` and n = 583, KPSS rejected on 61 and ADF on only 44, so E7 said "not stationary"
  on 77 of 100 (table above). E7 is therefore stated on the pre-holdout sample only, never per
  holdout (amended M1, review A).
- **ADF cannot tell `phi = 0.99` from `phi = 1` in a short sample.** Its power falls as `phi`
  nears 1. Not rejecting is not proof of a unit root.
- **The half-life is unstable near 1.** Its interval's upper end runs to infinity once the
  upper end of `phi`'s interval reaches 1. Report the interval, not only the point.
- **Regime changes.** If VIX reverts fast after spikes and slowly in calm, one `phi` describes
  neither. The burn-in estimate differing from the dev estimate is a hint, not a test.

## 5. In this project

- [`src/ivf/eda.py:90`](../../src/ivf/eda.py#L90): pairs on the calendar, dropping any pair with a gap.
- [`src/ivf/eda.py:109`](../../src/ivf/eda.py#L109) and [`src/ivf/eda.py:144`](../../src/ivf/eda.py#L144): OLS, HC1,
  and the interval mapped through the half-life.
- [`src/ivf/eda.py:224`](../../src/ivf/eda.py#L224): ADF (constant, AIC lag order) and KPSS (constant,
  automatic bandwidth), on the longest gap-free segment
  ([`src/ivf/eda.py:207`](../../src/ivf/eda.py#L207)). statsmodels clips the KPSS p-value to [0.01, 0.10] and warns; the code
  keeps the side of the clip in `kpss_p_bound`.
- [`src/ivf/eda.py:203`](../../src/ivf/eda.py#L203): the two nulls, stated in words in every report.
- [`src/ivf/data/process.py:242`](../../src/ivf/data/process.py#L242): the eda_stat row
  that ADR 0001 D2 allows `reports/half_life.agg.csv` to publish (per period, 2 s.f., clean tree).
  Milestone 1 writes it locally only.
- [`src/ivf/data/synthetic.py:109`](../../src/ivf/data/synthetic.py#L109): the known-truth generator.

## Self-check

1. `phi_hat = 0.975` with a 95% interval `(0.965, 0.985)`. Give the half-life and its interval.
2. ADF rejects and KPSS rejects. What do you conclude about ln VIX, and why is KPSS the weaker
   witness here?
3. Why is the half-life of ln VIX in the publication set P, while the mean of VIX is not?
