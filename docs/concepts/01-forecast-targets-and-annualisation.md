# 01 · Forecast targets and annualisation

What is forecast, in what units, and why a GARCH number and a VIX number are not directly
comparable. Background: [00 · Background](00-background.md) (§1 VIX, §2.1 the VRP, §5.1 losses). Numbers are computed with the
commands shown or cited to the ADR that measured them.

## 1. First principles

### Variance adds; volatility does not

Let $r_t = \ln(S_t/S_{t-1})$ be a daily log return. Over $n$ sessions the return is
$R = \sum_{k=1}^{n} r_{t+k}$, and

$$
\operatorname{Var}(R) = \sum_{k=1}^{n}\operatorname{Var}(r_{t+k}) + 2\sum_{j<k}\operatorname{Cov}(r_{t+j}, r_{t+k}).
$$

If returns are **uncorrelated**, the covariance terms vanish and variances add. If, in
addition, every session has the **same** variance $\sigma_d^2$, then
$\operatorname{Var}(R) = n\sigma_d^2$ and $\operatorname{sd}(R) = \sigma_d\sqrt{n}$. That is
square-root-of-time scaling. "i.i.d." is sufficient but stronger than needed: what matters is
zero autocorrelation and constant variance. Volatility never adds; variance does.

**Units used throughout** (ADR-0001 §4.2): a vol-point value $y$ (VIX units, e.g. 20 means 20%
a year) has variance-space value $v = (y/100)^2$, an annualised decimal variance.
VIX = 20 means $v = 0.04$; the variance over 30 calendar days is $0.04 \times 30/365 = 0.003288$,
a 30-day standard deviation of 5.73%; per session (252 convention) the sd is $20/\sqrt{252} = 1.26\%$.

### What VIX is

Cboe describes VIX as the "market's expectation of 30-day forward looking volatility" (Cboe
Volatility Index Methodology v6.0, 26 Feb 2026). For each of two expiries (near and next term) Cboe computes

$$
\sigma^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{RT} Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0}-1\right)^2 ,
$$

where $Q(K_i)$ are mid-quotes of out-of-the-money puts and calls, $F$ the forward, $K_0$ the
first strike at or below $F$, and $T$ is in **minutes / 525,600** (a 365-day year). It then
interpolates the two total variances to 30 days ($N_{30} = 43{,}200$ minutes) and annualises:

$$
\text{VIX} = 100\sqrt{\left[T_1\sigma_1^2\frac{N_{T_2}-N_{30}}{N_{T_2}-N_{T_1}} + T_2\sigma_2^2\frac{N_{30}-N_{T_1}}{N_{T_2}-N_{T_1}}\right]\frac{N_{365}}{N_{30}}}.
$$

The bracket is the risk-neutral variance of the next 30 **calendar** days; $N_{365}/N_{30}$
makes it per calendar year. Cboe's Appendix 3 inputs reproduce their 13.93 (below: 13.9278).

### Why VIX² is (approximately) a variance-swap rate

With continuous paths, Itô gives $d\ln S_t = dS_t/S_t - \tfrac12\sigma_t^2\,dt$:

$$
\frac{1}{T}\int_0^T \sigma_t^2\,dt = \frac{2}{T}\left(\int_0^T \frac{dS_t}{S_t} - \ln\frac{S_T}{S_0}\right).
$$

Under the pricing measure $Q$, $E^Q[\int dS/S] = rT$; with $F_0 = S_0e^{rT}$ this gives
$E^Q[\tfrac1T\int\sigma^2dt] = \tfrac2T E^Q[-\ln(S_T/F_0)]$. A Taylor expansion with remainder
writes the log payoff as a forward plus a strip of options weighted by $1/K^2$:

$$
-\ln\frac{S_T}{F} = -\frac{S_T-F}{F} + \int_0^F \frac{(K-S_T)^+}{K^2}\,dK + \int_F^\infty \frac{(S_T-K)^+}{K^2}\,dK .
$$

So the fair variance-swap strike is $\frac{2e^{rT}}{T}\left[\int_0^F \frac{P(K)}{K^2}dK + \int_F^\infty \frac{C(K)}{K^2}dK\right]$
(Demeterfi et al., 1999, cited by Cboe). The Cboe formula is its
discretised version; its second term corrects for splitting the strip at $K_0$ instead of $F$.
"Approximately", because: strikes are discrete and truncated; two expiries are interpolated; and
**with jumps** $J$ in $\ln S$ the strip prices
$\frac1T E^Q[\int\sigma^2dt + 2\sum(e^J-1-J)]$ rather than the quadratic variation
$\frac1T E^Q[\int\sigma^2dt + \sum J^2]$ (QV minus strip is about $-J^3/3$ per jump: positive for
$J<0$, negative for $J>0$).
For $J = -0.1$: $J^2 = 0.010000$ versus $2(e^J-1-J) = 0.009675$; crash-type jumps put VIX² slightly
below a true variance-swap rate.

## 2. The realised target: $RV_{21}$

ADR-0001 §4.2 defines, for date $d$,

$$
RV_{21}(d) = 100\sqrt{\frac{252}{21}\sum_{j=0}^{20} r_{d-j}^2}, \qquad r_d = \ln\frac{SPX_d}{SPX_{d-1}},
$$

with no mean removed. With $h = 21$ only, origin $t$ targets $r_{t+1},\dots,r_{t+21}$, exactly the
window after it; $252/21 = 12$ makes the sum a variance per 252-session year.

**Worked example.** Twenty returns of ±1% and one of 4%:
$\sum r^2 = 20(10^{-4}) + 16(10^{-4}) = 0.0036$; $\times 12 = 0.0432$; $RV_{21} = 100\sqrt{0.0432} = 20.78$.
All 21 at 1% gives $100\sqrt{12 \times 0.0021} = 15.87$; the 4% day supplies $16/36 = 44\%$ of the variance.

**Calendar conventions.** VIX annualises per 365 calendar days; $RV_{21}$ per 252 sessions.
ADR-0001 §4.4 measured, on the NYSE calendar 2000–2025: 251.50 sessions per year; 21 sessions
span 29–38 calendar days (mean 30.50); a 30-calendar-day window holds 15–22 sessions (mean 20.64).
Derived check, not re-measured: $21 \times 365.25/251.5 = 30.50$. If variance accrues per session
(weekend variance shows in Monday's return), both are "variance per year". The gap is small:
$\sqrt{252/251.5} = 1.000994$, 0.02 vol points at 20.

## 3. From a GARCH forecast to vol points, and the variance risk premium

GARCH(1,1) on daily returns in percent: $\sigma_{t+1}^2 = \omega + \alpha\varepsilon_t^2 + \beta\sigma_t^2$.
With persistence $\phi = \alpha+\beta < 1$ and long-run variance $\bar\sigma^2 = \omega/(1-\phi)$,

$$
\sigma^2_{t+k\mid t} = \bar\sigma^2 + \phi^{\,k-1}\left(\sigma^2_{t+1\mid t} - \bar\sigma^2\right),
\qquad
\sum_{k=1}^{21}\sigma^2_{t+k\mid t} = 21\bar\sigma^2 + \left(\sigma^2_{t+1\mid t}-\bar\sigma^2\right)\frac{1-\phi^{21}}{1-\phi}.
$$

The conditional variances are **summed**, not $21\sigma^2_{t+1|t}$, because variance is not
constant. ADR-0001 §4.4 then forms
$G_t(a) = \frac{252}{21}\sum_{k=a+1}^{a+21}\sigma^2_{t+k|t}$ (decimal², i.e. `arch` output $/10^4$),
stores it as the variance forecast $\hat v = G_t(a)$, and reports $\hat y = 100\sqrt{\hat v}$ (§4).

**Worked example.** $\omega = 0.02$, $\alpha = 0.10$, $\beta = 0.88$ (percent²), so $\phi = 0.98$,
$\bar\sigma^2 = 1.0$ (15.87 vol points long-run). Today $\sigma^2_{t+1|t} = 2.25$ (1.5% a day).

- $\phi^{21} = 0.654256$; $(1-0.654256)/0.02 = 17.2872$; sum $= 21 + 1.25 \times 17.2872 = 42.609$.
- $G_t(0) = 12 \times 42.609/10^4 = 0.051131$; $\hat y = 100\sqrt{0.051131} = 22.61$.
- Naive $\sqrt{252 \times 2.25/10^4} = 23.81$ overstates it, because variance mean-reverts.

`arch` 8.0.0's analytic path, `arch_model(y, mean='Zero').fix([0.02,0.10,0.88]).forecast(horizon=21)`
on a simulated series, matched this recursion to 2.2e-16.

**The variance risk premium.** In consistent units (annualised decimal variance, same window):

$$
VRP_t = \left(\frac{VIX_t}{100}\right)^2 - E^P_t\!\left[v^{RV}_{t\to t+21}\right].
$$

VIX is a $Q$-expectation; GARCH estimates a $P$-expectation. Decompose the gap:

$$
\left(\frac{VIX_t}{100}\right)^2 - G_t = \underbrace{VRP_t}_{\text{price of variance risk}} + \underbrace{E^P_t[v^{RV}] - G_t}_{\text{GARCH model error}} .
$$

A *perfect* GARCH still misses VIX by $VRP_t$. With $VIX_t = 25$ and $G_t = 0.051131$ as the true
$P$-expectation, $VRP_t = 0.0625 - 0.051131 = 0.011369$. In vol points the gap is
$25 - 22.61 = 2.39$, **not** $100\sqrt{VRP_t} = 10.66$: a difference of variances is not a
difference of vols. Hence ADR-0001's $P$-vs-$P$ track (GARCH vs $RV_{21}$) beside the $P$-vs-$Q$
track (GARCH vs VIX), whose error contains the VRP. At horizon $h$, by iterated expectations, the expected
error of $100\sqrt{G_t(h)}$ against $V_{t+h}$ in variance space is $E_t[VRP_{t+h}]$ plus model bias.
The `*_cal` models fit $\hat v = a_h + b_h G_t(h)$ by OLS; the average gap they absorb,
$\overline{v} - \overline{G} = a_h + (b_h-1)\overline{G}$, is $a_h$ only if $b_h = 1$, and the average
VRP only if $G$ is unbiased (ADR-0001 §4.4).

## 4. Jensen: variance and vol forecasts are not interchangeable

For concave $\sqrt{\cdot}$, $E[\sqrt X] \le \sqrt{E[X]}$. Exactly, for a vol-point target $y$:

$$
100\sqrt{E[v]} = \sqrt{E[y^2]} = \sqrt{E[y]^2 + \operatorname{Var}(y)} \;\ge\; E[y].
$$

**Two-point example.** $y \in \lbrace 10, 30\rbrace$ equally likely: $E[y] = 20$, but
$E[v] = (0.01+0.09)/2 = 0.05$ and $100\sqrt{0.05} = 22.36$, the root-mean-square vol. Conversely
the plug-in $(E[y]/100)^2 = 0.04$ is low by exactly $\operatorname{Var}(y)/10^4 = 0.01$; with
conditional moments, by $\operatorname{Var}_t(y)/10^4$ when $\hat y = E_t[y]$.

**How the repo handles it** (ADR-0001 §4.3). Every forecast row stores both numbers, always linked
by $\hat y = 100\sqrt{\hat v}$; `var_method` says which one is the model's own forecast:

- `plugin` (random walk, historical mean, ARIMA, HAR, LSTM): $\hat y = E_t[y]$ and
  $\hat v = (\hat y/100)^2$, low as a variance forecast by $\operatorname{Var}_t(y)/10^4$.
- `native` (GARCH family's $G$, `vix_implied`, `*_cal`): $\hat v = E_t[v]$, so $\hat y = \sqrt{E_t[y^2]}$
  is the RMS vol, high as a mean by about $100\operatorname{Var}_t(v)/(8\hat v^{3/2})$ (second-order Taylor).
  For $y \sim N(16, 4^2)$: $E[v] = 0.0272$, $\operatorname{Var}(v) = 1.6896\times10^{-4}$, approximation
  0.471 against the exact $\sqrt{16^2+4^2} - 16 = 0.492$.

Comparing a `plugin` and a `native` forecast under the other kind's loss includes this Jensen term.
The reports state it rather than correct it: correcting needs a forecast of
$\operatorname{Var}_t(v)$ that `vix_implied` and `*_cal` do not have.

**Lognormal correction.** If $\ln Y \sim N(\mu, s^2)$, then using the normal MGF
$E[e^{X}] = e^{\mu + s^2/2}$:

$$
E[Y] = e^{\mu + s^2/2}, \qquad \operatorname{median}(Y) = e^{\mu}.
$$

A log-VIX model with $\mu = \ln 20 = 2.995732$ and $s = 0.25$ has median 20 but mean
$20e^{0.03125} = 20 \times 1.031743 = 20.635$. A seeded Monte Carlo (10⁶ draws) gives mean
20.6408 (standard error 0.0052) and median 20.0048. ADR-0001 §4.3 reports $e^{\mu+s^2/2}$.

**Which loss rewards which forecast.** A loss ranks forecasts correctly only for the statistic it
is *consistent* for: the one whose true value minimises the expected loss (Gneiting, 2011). For a
right-skewed $y$ the three level losses pull apart:

| Loss | Consistent for | On this lognormal ($e^{\mu} = 20$, $s = 0.25$) | ADR-0001 §4.3 role |
|---|---|---|---|
| Squared error (RMSE) | the mean | $e^{\mu+s^2/2} = 20.635$ | **primary** for `plugin` |
| QLIKE | the mean of $v$ | (variance space) | **primary** for `native`; a `plugin` $(\hat y/100)^2$ is low by $\operatorname{Var}_t(y)/10^4$, so QLIKE does not rank it as a mean |
| Absolute error (MAE) | the median | $e^{\mu} = 20$ | reported, flagged |
| Absolute percentage error (MAPE) | the "median of order −1": the median of the density reweighted by $1/y$ | $e^{\mu - s^2} = 18.788$ | reported, flagged |

Reweighting a lognormal density by $1/y$ shifts $\mu$ to $\mu - s^2$, hence $e^{\mu-s^2}$. Measured
on $10^6$ draws (command below; the ADR's `tools/probes/mape_check.py` uses $2\times10^6$), the three
sample minimisers are 20.641, 20.005 and 18.792, and
MAPE scores the true mean 20.635 at 0.2118 but the low forecast 18.788 at 0.1976. A better MAE or
MAPE can come from a forecast biased low, so they are reported (the brief asks) but flagged, and
the DM tests use only squared error and QLIKE. For real data all four are published only as ratios
to `rw`'s value, never in vol points, and only for rows that stay unchanged when the VIX path is
rescaled with public SPX fixed: `vix_implied` and the uncalibrated GARCH family on `vix_level` stay
local (ADR-0001 D2, publication set P; [03 §8](03-reproducibility-and-data-contracts.md)).

## 5. Why "GARCH on the VIX level" is a category error

GARCH models $r_t = \mu + \varepsilon_t$, $\varepsilon_t = \sigma_t z_t$, $z_t$ i.i.d. $(0,1)$, with
$E_{t-1}[\varepsilon_t] = 0$. The object is the **conditional variance of an unpredictable
innovation**. VIX is already a conditional-volatility forecast: a persistent, positive *level*.
Feed it in as "$r_t$" with a constant mean, and $\varepsilon_t = V_t - \mu$ is mostly yesterday's
deviation, violating $E_{t-1}[\varepsilon_t] = 0$. The variance equation reads
$(V_{t-1}-\mu)^2$ as "shock size", so a calm year at VIX 12 looks like high variance. You would be
modelling the squared distance of VIX from a constant: mean dynamics mislabelled as clustering.

**Measured.** Simulate a *homoskedastic* AR(1) level, $V_t = 20 + 0.98(V_{t-1}-20) + e_t$,
$e_t \sim N(0,1)$, 5,000 points, seeds 0–9. `arch_model(V, mean='Constant')` GARCH(1,1) finds
$\alpha \in [0.923, 0.976]$ and $\alpha+\beta \in [0.946, 0.976]$: strong "ARCH effects" that do not
exist. With `mean='AR', lags=1`, $\alpha \in [0, 0.0315]$ and the implied variance
$\omega/(1-\alpha-\beta) \in [0.983, 1.029]$, the true value 1.

The legitimate uses: GARCH on **SPX returns** (`garch`, `gjr`, `egarch`), and ARIMA–GARCH on
**$\Delta\ln V$** (`arima_garch_log`): ARIMA for the mean of log-VIX changes, GARCH for their "vol of vol".

## Reproduce

```bash
uv run python - <<'EOF'
import math, numpy as np
print(100*math.sqrt(12*(20e-4+16e-4)), math.sqrt(252/251.5))            # 20.78, 1.000994
T1,T2,N1,N2,N30,N365=34484/525600,44954/525600,34484,44954,43200,525600 # Cboe App. 3
tot=T1*0.019233906*(N2-N30)/(N2-N1)+T2*0.019423884*(N30-N1)/(N2-N1)
print(100*math.sqrt(tot*N365/N30))                                       # 13.9278
w,a,b,s1=0.02,0.10,0.88,2.25; p=a+b; vb=w/(1-p)
S=21*vb+(s1-vb)*(1-p**21)/(1-p); print(S, 100*math.sqrt(12*S/1e4))      # 42.609, 22.61
print(0.0625-12*S/1e4, 100*math.sqrt(0.05), 20*math.exp(0.25**2/2))     # 0.011369, 22.36, 20.635
print(100*1.6896e-4/(8*0.0272**1.5), math.sqrt(16**2+4**2)-16)           # 0.471, 0.492
Y=np.exp(np.random.default_rng(0).normal(math.log(20),0.25,10**6)); print(Y.mean())  # 20.6408
from scipy.optimize import minimize_scalar as m                          # squared, absolute, percentage error:
print([round(float(m(lambda c:np.mean(L(Y-c)/D),bounds=(15,25),method='bounded').x),3) for L,D in ((np.square,1),(np.abs,1),(np.abs,Y))])  # 20.641 20.005 18.792
EOF
```

## Assumptions

Continuous prices, dense strikes, no jumps; per-session variance accrual, uncorrelated returns;
a correct GARCH with $\phi < 1$; Gaussian log-errors; VRP terms in one window and unit.

## Where it breaks

- **Windows differ**: 15–22 sessions behind VIX (§2), always 21 behind $RV_{21}$.
- **Autocorrelation** breaks $\sqrt{n}$:
  $\operatorname{Var}(R) = n\sigma^2[1 + 2\sum_{k=1}^{n-1}(1-k/n)\rho_k]$.
- **Jumps** separate VIX² from quadratic variation (§1).
- **Fat-tailed log-errors** make $e^{s^2/2}$ wrong (Duan's 1983 smearing is the usual fallback);
  ignoring parameter uncertainty makes $s^2$ too small.

## How this repo implements it

- Targets, units and `v = (y/100)²`: ADR-0001 §4.2 ([ADR](../adr/0001-python-core-versioned-json-app.md)).
- $\hat y = 100\sqrt{\hat v}$, `var_method` plug-in vs native, the lognormal back-transform, the
  loss-consistency table (RMSE and QLIKE primary), and the reported Jensen term: §4.3.
- $G_t(a)$, the 21/252 rationale and calendar figures, the $P$/$Q$ tracks and `*_cal`: §4.4.
- Storage: `target_value`, `forecast(yhat, v_hat, var_method)` in [data-model.md](../architecture/data-model.md) §4.
- **Code:** none yet (targets M1, GARCH M3). Only the target ids and units exist, in
  `tools/export_app_data.py:77-80` (`M0_TARGETS`).

## Self-check

1. A GARCH model gives $\sum_{k=1}^{21}\sigma^2_{t+k|t} = 42.609$ percent². Convert it to vol points, and
   say why it should not equal VIX even if the model is perfect.
2. An ARIMA on $\ln V$ forecasts $\mu = \ln 18$, $s = 0.3$ at $h = 21$. What does the repo report,
   and why is $(\hat y/100)^2$ not that model's forecast of $E[v]$?
3. Why is GARCH on the VIX level a category error, and what are the two legitimate alternatives?

<details>
<summary>Answers</summary>

1. $12 \times 42.609/10^4 = 0.051131$; $100\sqrt{0.051131} = 22.61$ (an RMS vol). VIX² is a
   $Q$-expectation: $(VIX/100)^2 - G = VRP + (E^P[v]-G)$. Model error is judged on the $RV_{21}$
   track; on the VIX track the gap shows as a Mincer–Zarnowitz $\alpha \ne 0$ or $\beta \ne 1$.
2. $18e^{0.045} = 18.83$, the lognormal mean ($e^{\mu} = 18$ is the median). $(\hat y/100)^2$
   misses $\operatorname{Var}_t(y)/10^4$, because $E[y^2] = E[y]^2 + \operatorname{Var}(y)$.
3. GARCH models the conditional variance of an unpredictable innovation; VIX's deviations from a
   constant are predictable mean dynamics (a homoskedastic AR(1) gave $\alpha \approx 0.92$–$0.98$).
   Use GARCH on SPX returns, or ARIMA–GARCH on $\Delta\ln V$.

</details>
