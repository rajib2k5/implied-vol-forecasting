# 06 · The implied-minus-realised spread (variance risk premium) and vol-of-vol

Why VIX is usually above the volatility that follows it, how to measure the gap without fooling
yourself, and how volatile volatility itself is. Background: [00 · Background](00-background.md)
§2.1 and [01 · Forecast targets](01-forecast-targets-and-annualisation.md) (units, `rv21`, Jensen).
Numbers come from the commands shown, run on 2026-10-29, on **synthetic** data. Real-data
results are measured locally, not published (ADR 0001 D2).

## 1. First principles

- **VIX is a price, not a forecast.** `VIX²/10⁴` is close to the fair rate of a 30-day variance
  swap. Option sellers bear crash risk, so they charge for it. The price of variance is then,
  on average, above the variance that follows.
- **The gap is the variance risk premium (VRP).** Ex post, for each day `t`:
  `(V_t / 100)² - (RV_{t→t+21} / 100)²`, implied variance minus the realised variance of the next
  21 sessions. A positive average says insurance was, on average, overpriced.
- **It is a description of the past, not an input.** The realised part uses the 21 sessions
  *after* `t`. It is fine for EDA and forbidden as a model feature (rule L1). The model-safe
  version is the trailing spread `iv_rv_spread`, which uses the 21 sessions *before* `t`.
- **Vol-of-vol** is how much VIX itself moves. Two views: realised (the sd of daily changes in
  ln VIX, annualised) and implied (VVIX, the VIX of VIX options).

## 2. The maths

### Units

A vol-point value `y` (VIX = 20 means 20% a year) has the variance `v = (y/100)²`. The realised
target (ADR 0001 4.2) is

```
rv21(d) = 100 * sqrt((252/21) * sum_{j=0}^{20} r_{d-j}^2)       r_d = ln(SPX_d / SPX_{d-1}), no mean removed
```

The forward value for origin `t` is `rv21(𝒯[t+21])`: its window is exactly the 21 sessions
after `t`.

**Worked numbers** (`test_rv21_of_21_returns_of_one_percent_is_15_8745`,
`test_spread_by_hand`):
- 21 returns of 0.01: `100 * sqrt(12 * 21 * 0.0001) = 100 * sqrt(0.0252) = 15.8745` vol points.
- `V = 20` and the next RV = 15: the gap is 5 vol points, or `0.04 - 0.0225 = 0.0175` in variance.
- `V = 10` and the next RV = 12: -2 vol points, or `0.01 - 0.0144 = -0.0044`.
  Mean of the two: 1.5 vol points, 0.00655 in variance; implied above realised on half.

### Why report both vol points and variance

They can disagree in sign. The mean of `V - RV` and the mean of `V² - RV²` weight days
differently: variance weights the high-vol days far more. By Jensen, a variance gap of zero
still leaves a positive vol-point gap when `RV` is more dispersed than `V`. Concept 01 §4 has
the identity.

### The known truth in the synthetic market

The generator draws ln VIX and S&P 500 returns **independently**: an AR(1) with
`mu = ln 18`, stationary sd 0.30, and a GARCH(1,1) with unconditional variance
`0.16²/252` per session. Then the expected variance gap is a difference of two levels:

```
E[(V/100)^2] = exp(2 mu + 2 s^2) / 10^4 = 324 * exp(0.18) / 10^4 = 0.038790
E[(RV/100)^2] = 252 * omega / (1 - alpha - beta) = 0.16^2 = 0.0256
gap = 0.013190
```

On 200,000 simulated sessions the estimate came within `±0.003`; over 20 seeds it had sd
0.00064 (`test_spread_recovers_the_known_variance_gap`). Because the two series are independent,
this "premium" does not move with the state. Real VRP does; the synthetic data cannot show it.

**On the synthetic market** (`uv run python -m ivf.data --synthetic eda`, 2026-10-29, re-run after
review A). The EDA sees only sessions before `holdout_start` (amended M1, review A: the first
version also reported a holdout sample, a look at the holdout that ADR 0001 4.6 forbids before the
models are frozen):

| Sample | n origins | Mean gap, vol points | Median | Mean gap, variance | Implied above realised |
|---|---:|---:|---:|---:|---:|
| pre_holdout (1990-01-02..2024-06-28) | 8,668 | +4.20 | +3.95 | +0.0152 | 74.3% |
| burn_in | 2,578 | +2.63 | +2.59 | +0.0098 | 70.0% |
| dev | 6,090 | +4.86 | +4.65 | +0.0175 | 76.1% |

The pre-holdout sample has 21 fewer origins than sessions (8,689): the forward windows of its last
21 origins would end in the holdout, so they are not computed.

### Vol-of-vol

The realised version: the sd of `d ln V` over 21 sessions, times `sqrt(252)`.

**Worked number** (`test_vov_is_annualised_with_sqrt_252`): a window of alternating `±0.1` steps
(11 up, 10 down) has sample sd 0.10235; times `sqrt(252) = 15.8745` gives 1.6248, i.e. 162% a
year.

For an AR(1) in ln VIX, `sd(d ln V) = sigma * sqrt(2 / (1 + phi))`. With the generator's
`sigma = 0.059699` and `phi = 0.98`: `0.059699 * 1.005038 = 0.060000` per session, 0.9525 a year.
A 21-session sd is biased low; over 20 seeds the mean was 0.9446 (sd 0.0016). On the synthetic
market path: 0.931.

**"Vol-of-vol rises with the level" (statement E6)** is measured as the rank correlation `rho` of
`V_t` with the size of the *next AR(1) shock* to ln VIX, `|ln V_{t+1} - c - phi ln V_t|`, with `c`
and `phi` fitted on the same sample. Two simpler measures mislead when shocks are skewed, as VIX's
are (it spikes up and decays down):

| Measure, iid exponential shocks (no level effect), 5 seeds of 200,000 | Rank correlation with V_t |
|---|---:|
| Next AR(1) shock (the E6 measure) | -0.005 to +0.010 |
| Next raw move, abs(ln V_{t+1} - ln V_t) (contains the pull back to the mean) | +0.105 to +0.114 |
| Same-day move, abs(ln V_t - ln V_{t-1}) (a high level is made by a big up-move) | +0.163 to +0.172 |

In vol points, `|V_{t+1} - V_t|`, the correlation is positive by construction (synthetic: +0.30),
because the same percentage move is more points at a higher level. Only the log shock says
something about VIX.

**Testing it: the p-value must come from a bootstrap that refits the AR(1)** (amended M1, review A).
With iid shocks the E6 `rho` is *centred* near zero whatever their shape (mean -0.005 over 300
skewed paths), but it is not spread like the correlation of independent pairs, so scipy's Spearman
p-value is wrong for it. The fitted shock is not the true one:

```
shock_hat_{t+1} = e_{t+1} + (c - c_hat) + (phi - phi_hat) * ln V_t
```

The last term grows with the level, and `phi_hat - phi` comes from the same shocks. With skewed
shocks the ranks of `|shock_hat|` react strongly to that small level-dependent shift (most
exponential shocks sit near their lower bound, where a tiny shift reorders many of them). Review A
measured it on 300 stationary paths with iid exponential shocks and **no level effect**: Spearman's
p-value rejected at 5% on 30% of them, against 5.3% when the true shocks are used; the sd of `rho`
was 0.0187 against 0.0105 for independent pairs. The first version of this note said "with iid
shocks it is zero, whatever their shape" and blamed the p-value only on VIX's persistence; the
statistic is centred, but that sentence was wrong about the test. Fitting `phi` on one half and
testing on the other does not cure it (30% on 60 paths, scratch run): the out-of-sample error in
`phi_hat` is still in every shock.

The fix reproduces the estimation step. Each bootstrap path starts at the first observed ln V,
follows `x*_t = c_hat + phi_hat * x*_{t-1} + e*_t`, takes the data's gaps, and goes through the same
fit and the same `rho`. The p-value is where the observed `rho` falls among 999 such draws
(two-sided, equal tails, `(1 + count)/(B + 1)`).

**The shocks must also keep their clustering** (amended M1, review B). Review A's version drew the
`e*` one at a time, with replacement (iid). That is right only if the shocks are iid, and VIX's are
not: big moves come in runs (volatility clustering, concept 04). Under clustering both series in
`rho` are persistent, the level through `phi` and the shock size through the clustering, and the
correlation of two persistent series spreads far wider than iid draws allow (the same reason two
independent random walks often look correlated). Review B simulated symmetric GARCH(1,1) shocks,
where the level and the next shock size are unrelated by symmetry, and the iid bootstrap rejected
at 5% on 39.5% of 200 paths (47.0% of 300 other paths in the table below).

With *skewed* clustered shocks it is worse: the population `rho` is not even zero. A big up-move
raises the level and, through the clustering, the size of the next shocks, so the two correlate
(+0.13 and +0.25 for the two GARCH settings below, n = 10^6) although the level never enters the
volatility equation. Any test of "`rho` = 0" calls that a level effect. A wild bootstrap
(`e* = e_hat * eta`, random signs) keeps every `|e_hat|` in place, so it keeps the clustering, but it
throws the skew away and fails on skewed shocks.

The shipped bootstrap cuts the fitted shocks into **blocks** of `L` consecutive sessions and puts
the blocks back together in random order (a circular block bootstrap). Inside a block, whatever
the shock sizes do together, and what that does to the level, is kept. Between blocks the order is
random, so the level a path has reached says nothing about the next block's shock sizes. `L`
comes from Politis and White's rule applied to `|shock|` (`arch.bootstrap.optimal_block_length`):
iid shocks get `L` = 1, which is review A's bootstrap exactly, and clustered shocks get blocks of
about 70 to 250 sessions. So the null is "the level goes with the next shock size no more than the
shocks' own short-range dependence explains".

Measured with the shipped code and review A's on the same paths (`block=1`), n = 9,000,
phi = 0.98, 300 paths each, B = 999 (`uv run python tools/probes/e6_size_power.py 300 9000`;
walkthrough §7.3 has every row and the designs that were rejected):

| Paths (shock size never depends on the level unless stated) | Shipped: p < 0.05 | Review A (iid draws) |
|---|---:|---:|
| iid normal | 5.3% | 5.7% |
| iid exponential (skewed) | 2.3% | 2.7% |
| GARCH(1,1), alpha 0.05, beta 0.90 | 0.0% | 19.7% |
| GARCH(1,1), alpha 0.10, beta 0.88 | 0.0% | 47.0% |
| skewed GARCH, the same two settings | 0.3%, 0.0% | 100%, 100% |
| GJR (volatility rises more after up-moves) | 0.0% | 100% |
| an effect: shock sd x min(e^{0.5(x - mu)}, 3) | **95.0%** | 100% |
| a weaker one: shock sd x min(e^{0.15(x - mu)}, 3) | **99.7%** | 99.7% |
| the 0.5 effect on top of GARCH(0.05, 0.90) clustering | **18.0%** | 100% |

(Scipy's Spearman p-value, the test review A replaced, rejected on 30.7% of the skewed iid paths.)

So the block bootstrap keeps its size on every null tried, and has the power of review A's where the
shocks do not cluster. **Where they cluster, it has little power**: on the last row it found a strong
level effect on only 18% of paths. Clustering and a level effect leave the same trace in this
statistic, because the level is itself a sum of past shocks; the blocks keep both. So E6 can say
"yes" (the level goes with the next shock size beyond what the clustering explains) or "not shown",
never "no". On the synthetic market (pre-holdout, 8,688 calendar pairs, `L` = 2): `rho` = -0.0005,
bootstrap p = 0.92; E6 says "not shown", as it should. And it answers only while the fitted `phi`
is at most 0.985 (amended M1, final check; section 4).

## 3. Assumptions

- **VIX and RV21 are on the same scale.** VIX covers 30 calendar days annualised by calendar
  time; RV21 covers 21 sessions annualised by 252. ADR 0001 4.4 shows the two agree to within the
  252-versus-251.5 convention.
- **RV21 is a noisy proxy.** 21 squared returns estimate the realised variance with error; a
  single day's crash dominates. The *mean* gap is robust to that noise; any single day's gap is
  not.
- **The ex-post gap mixes premium and surprise.** Over a short sample, one unexpected crash
  makes realised exceed implied and can turn the mean gap negative. Only a long sample separates
  a premium from luck.

## 4. Where it breaks

- **Short periods.** The holdout has about 560 origins, and its 21-session windows overlap, so
  it holds only about 27 independent windows. One crash month moves its mean a lot.
- **Vol points vs variance.** A conclusion that holds in one can fail in the other (Jensen).
  The report gives both.
- **VVIX is sparse early.** Cboe's VVIX history has many missing sessions in its first months
  (counted locally by `data_check`). Correlations with VVIX use pairwise-complete sessions only.
- **The E6 measure is still a correlation.** The bootstrap fixes its *size*, also when the shocks
  cluster (amended M1, review B); it does not make the AR(1) right. A misfit mean (VIX reverting
  faster after spikes than in calm, say) leaves part of the pull in the "shock", and the bootstrap,
  which simulates the fitted AR(1), cannot see that. A "yes" says the next shock grows with the
  level *given an AR(1) mean*, beyond what the shocks' clustering explains; a richer mean model is
  milestone 2's job.
- **"Not shown" is not "no"** (amended M1, review B). When the shocks cluster, as VIX's do, the
  block bootstrap found a strong level effect on 18% of paths. Separating "the level drives the
  volatility" from "volatility clusters and the shocks are skewed" needs a model of the volatility
  (a GARCH with the level as a regressor, say), which is milestone 2's GARCH work, not a resampling
  scheme.
- **Too persistent to answer** (amended M1, final check). When both the level and the volatility
  are very persistent, the blocks (about 245 sessions, near the rule's cap) are shorter than the
  level's memory, and the test rejects far too often with no level effect at all. Skewed
  GARCH(0.03, 0.965) shocks, 200 paths of 9,000 sessions, B = 999
  (`uv run python tools/probes/e6_size_power.py 200 9000 skew98both skew985both skew99both
  skew995both lnorm995both`, about 3.5 minutes):

  | AR(1) `phi` | rejects at 5% | "yes" once `phi_hat` <= 0.985 is required |
  |---:|---:|---:|
  | 0.98 | 0.0% (0/200) | 0 |
  | 0.985 | 2.0% (4/200) | 0 |
  | 0.99 | **15.0%** (30/200) | 0 |
  | 0.995 | **44.5%** (89/200) | 0 |
  | 0.995, lognormal innovations | **34.5%** (69/200) | 0 |

  So E6 says "yes" only when the fitted `phi` is at most 0.985 (`eda.E6_PHI_MAX`), and "not
  shown" with the reason "persistence above the size-checked range" otherwise; `rho` and p are
  still reported. The cutoff costs no power at `phi` = 0.98 (95.0%, 99.7% and 18.0% on the three
  alternatives, unchanged). It does not fix short samples: at n = 2,000 the same shocks with `phi`
  = 0.98 gave "yes" on 24 of 100 paths through the cutoff (B = 99). So E6 also needs at least
  6,000 level/next-shock pairs (`eda.E6_MIN_N`). At 4,000 "yes" reached 10 of 200 paths on one
  of three nulls, the bound itself; at 6,000 it was 1, 0 and 3 of 200. The pre-holdout VIX sample has about 8,700 pairs.

## 5. In this project

- [`src/ivf/features/build.py:67`](../../src/ivf/features/build.py#L67): `rv21`, trailing, no mean removed.
- [`src/ivf/features/build.py:76`](../../src/ivf/features/build.py#L76): the shift that turns `rv21` into the forward
  target; for targets and ex-post EDA only.
- [`src/ivf/features/build.py:105`](../../src/ivf/features/build.py#L105): the model-safe trailing spread.
- [`src/ivf/eda.py:279`](../../src/ivf/eda.py#L279): the ex-post gap in vol points and variance, share
  positive, per period mask.
- [`src/ivf/eda.py:539`](../../src/ivf/eda.py#L539): annualised rolling vov, the two misleading measures
  for contrast, and VIX against VVIX.
- [`src/ivf/eda.py:442`](../../src/ivf/eda.py#L442): the E6 measure and its bootstrap, which refits the
  AR(1) on every path and resamples the shocks in blocks; [`src/ivf/eda.py:402`](../../src/ivf/eda.py#L402): the block
  length rule; [`src/ivf/eda.py:369`](../../src/ivf/eda.py#L369): the fitted shocks.
- [`src/ivf/data/synthetic.py:69`](../../src/ivf/data/synthetic.py#L69): the parameters behind the known truth.

## Self-check

1. Why is the forward spread a fine EDA statistic but a forbidden feature? Name the rule.
2. Over one holdout, the mean gap is +2 vol points but -0.001 in variance. Is that possible?
   What does it say about the days with high RV?
3. Why does `|ΔV|` (in points) correlate with the level even when ln VIX has iid shocks? And why
   does the next raw move `|Δ ln V|` too, once the shocks are skewed?
4. The E6 `rho` is centred near zero under iid skewed shocks, yet scipy's p-value for it rejects on
   30% of paths. Where does the extra spread come from, and why does refitting `phi` inside each
   bootstrap path fix the test when fitting it on the other half of the sample does not?
5. Symmetric GARCH shocks give `rho` = 0 in the population, yet review A's iid bootstrap rejected on
   47% of such paths. Why? And why can no test of "`rho` = 0" have the right size once the
   clustered shocks are also skewed? (Amended M1, review B.)
