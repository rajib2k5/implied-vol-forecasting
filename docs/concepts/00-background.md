# 00 · Background: what the literature knows about forecasting implied volatility

*Compiled 2026-10-28 (M0); revised after [review C](../validation/M0-review-C-background.md). Claims cite author–year; references are at the end. **[unverified]** = not checked against a primary source. "M1 measures" = to be measured, not asserted. Units: [01](01-forecast-targets-and-annualisation.md). Protocol: [ADR-0001](../adr/0001-python-core-versioned-json-app.md).*

---

## 1. Implied volatility and what VIX measures

**Black–Scholes implied volatility** is the $\sigma$ that makes the Black–Scholes price equal the observed option price (Black & Scholes 1973). Each strike and expiry gives a different value (the smile).

**Model-free implied variance** removes the model:
- **Britten-Jones & Neuberger (2000)** derive a volatility forecast that needs only current option prices; **Jiang & Tian (2005)** extend it to jumps and find, on SPX options, that it subsumes Black–Scholes IV and past realised volatility.
- **Variance-swap replication theory (Demeterfi et al. 1999).** A variance swap can be replicated closely by a static strip of out-of-the-money (OTM) calls and puts weighted by $1/K^2$. Cboe's 2003 VIX methodology applies this strip to SPX options and cites the 1999 Goldman Sachs note (Cboe 2026a).

**VIX: history.** The 1993 original measured 30-day at-the-money implied volatility from S&P 100 (OEX) options (Cboe 2026a). Whaley (1993) argued for derivatives on such an index; Whaley (2000) explains the investor-fear-gauge nickname. On 22 Sep 2003 Cboe and Goldman Sachs switched to the SPX strip; values start in January 1990 (Cboe 2026a). The redefined VIX approximates the 30-day S&P 500 variance swap rate, the risk-neutral expected return variance (Carr & Wu 2009).

**VIX: the formula (Cboe 2026a, 2026b).** For each of two expiries $i$ bracketing 30 days:

$$\sigma_i^2=\frac{2}{T_i}\sum_j \frac{\Delta K_j}{K_j^2}\,e^{R_iT_i}\,Q(K_j)\;-\;\frac{1}{T_i}\left(\frac{F_i}{K_{0,i}}-1\right)^2$$

- $Q(K_j)$: mid-quote of the OTM option at $K_j$ (put and call at $K_0$), from AM-settled SPX and PM-settled SPXW options on Cboe Options. A zero bid, or since 10 Feb 2025 a zero ask, excludes an option (Cboe 2026b, App. 1).
- $\Delta K_j$: generally half the gap between neighbouring strikes; $F$: option-implied forward; $K_0$: first strike at or below $F$; $R$: cubic-spline Treasury CMT rate; $T$: minutes / 525,600.

The two variances are interpolated to 30 days, annualised and quoted in vol points:

$$\text{VIX}=100\sqrt{\left[T_1\sigma_1^2\,\frac{N_{T_2}-N_{30}}{N_{T_2}-N_{T_1}}+T_2\sigma_2^2\,\frac{N_{30}-N_{T_1}}{N_{T_2}-N_{T_1}}\right]\frac{N_{365}}{N_{30}}}$$

$N_{30}=43{,}200$, $N_{365}=525{,}600$ minutes. Cboe's v5.0 Mathematics Methodology (2026b) writes $M$ ($M_{T_1}$, $M_{CM}$, $M_{365}$) where 01 writes $N$.

**Consequence.** $(\text{VIX}/100)^2$ is an annualised risk-neutral (Q) expected variance over the next 30 **calendar** days (Carr & Wu 2009), not a real-world forecast of realised volatility.

**Term structure.** Cboe applies the method at 9 days (VIX9D), 30 days, 3 months (VIX3M), 6 months (VIX6M) and 1 year (VIX1Y) (Cboe n.d.). VIX3M's ticker changed from VXV on 18 Sep 2017 (Cboe VIX3M page, Wayback capture 15 Oct 2017). Cboe's `VIX3M_History.csv` starts 2009-09-18 (read 2026-10-28), so term-structure regimes are null before then unless M1 adds a VXV source (ADR-0001 §4.8).

| Shape | Definition | Reading |
|---|---|---|
| Contango | $\text{VIX} < \text{VIX3M}$ | Risk-neutral variance rises with horizon; in futures, a **risk-neutral** expected VIX increase (Simon & Campasano 2014) |
| Backwardation | $\text{VIX} > \text{VIX3M}$ | The reverse |

- **Futures basis.** In 2006–2011 the VIX futures basis did **not** forecast spot VIX changes but did forecast futures price changes: a risk premium, not a real-world expectation (Simon & Campasano 2014). How often each shape occurs: M1 measures.
- **VIX futures launch: 26 Mar 2004.** CFE's archived trade files for the May and June 2004 contracts start on 3/26/2004 (CFE 2004), as does Carr & Wu's (2004) working paper. Cboe 2026a's date of 24 March is the outlier.

## 2. Stylised facts

| Fact | Meaning | Evidence | M1 measures |
|---|---|---|---|
| Volatility clustering | Large moves follow large moves | Cont (2001), fact 6; Engle (1982) | ACF of $r_t^2$, $\lvert r_t\rvert$ (SPX) |
| Persistence / long memory | Slow ACF decay of $\lvert r\rvert$; long-range dependence in VIX | Cont, fact 8; Corsi (2009); Fernandes, Medeiros & Scharth (2014) | ACF of log-VIX; AR(1) $\phi$, half-life $\ln 0.5/\ln\phi$ |
| Mean reversion | IV returns to a long-run level | Simon & Campasano (2014) | Long-run mean; reversion speed |
| Leverage / asymmetry | Negative shocks raise volatility, positive shocks lower it | Black (1976); Cont, fact 9; Nelson (1991); Glosten, Jagannathan & Runkle (1993); Fernandes et al. (2014) | corr($r_t$, $\Delta\log\text{VIX}_t$); GJR $\gamma$ |
| Heavy tails | Remain after a GARCH filter | Cont, facts 2 and 7 | Kurtosis of standardised residuals |
| Vol-of-vol | IV changes are heteroskedastic | No dedicated source retrieved; motivates ARIMA-GARCH | Rolling sd of $\Delta\log\text{VIX}$; ARCH-LM |
| Spikes | e.g. Aug 2024, Apr 2025 | Events, not a stylised fact | Spike count; post-spike decay |

Per-event statistics, rolling series and level-unit numbers stay local. Real results are committed only as ADR-0001 D2's publication set P (rounded, unitless whole-period numbers), and EDA prose about real data is limited to D2's qualitative statements E1–E7.

### 2.1 IV versus realised volatility: the variance risk premium

The two standard definitions have opposite signs.

| Paper | Definition | Finding |
|---|---|---|
| Carr & Wu (2009) | $RV_{t,T}-SW_{t,T}$ (realised minus synthetic swap rate) | Strongly **negative** on average for the S&P 500, S&P 100 and DJIA; Nasdaq-100 t-statistics lower but still strongly significant; stocks vary widely. CAPM beta and Fama–French factors explain only a small portion |
| Bollerslev, Tauchen & Zhou (2009) | $IV^2-RV$ | Predicts aggregate returns, most strongly at a quarterly horizon |
| Bekaert & Hoerova (2014) | $\text{VIX}^2$ = conditional variance + premium | Premium predicts returns; conditional variance predicts economic activity |

**IV as a forecast of RV: biased but informative.** The classic test is $RV_{t,t+h}=a+b\,IV_t+u$.
- **Christensen & Prabhala (1998).** On S&P 100 options, IV beats past volatility and in some specifications subsumes it. Earlier findings of bias reflected shorter, overlapping samples and a regime shift around the 1987 crash.
- **Poon & Granger (2003)** survey 93 papers; their companion summary ranks option-implied volatility most accurate, then tentatively historical volatility, ARCH/GARCH, stochastic volatility (Poon & Granger 2005).
- **Why $b\neq1$.** The premium makes IV exceed expected RV. Under long memory the regression is a fractional cointegration, so OLS slopes are inconsistent and biased downward, while long-run unbiasedness is supported (Bandi & Perron 2006).
- **Noisy yardsticks.** Against squared daily returns, good forecasts can look poor (Andersen & Bollerslev 1998).

## 3. Forecasting IV itself

| Paper | Data | Finding | Lesson |
|---|---|---|---|
| Harvey & Whaley (1992) | S&P 100 option IV | IV changes predictable; daily trading earns no abnormal profit after costs | Statistical ≠ economic |
| Konstantinidi, Skiadopoulos & Tzagkaraki (2008) | US and European IV indices | Statistically predictable; no abnormal profits in CBOE volatility futures | Same |
| Fernandes, Medeiros & Scharth (2014) | Daily VIX | Long memory; parametric and semiparametric HAR; linearity not rejected; **pure HAR very hard to beat** given VIX's persistence | HAR is the baseline |
| Vrontos, Galakis & Vrontos (2021) | VIX *direction* | ML beats econometric models statistically and in trading | Sign, not level |

- **Konstantinidi et al. (2008)**, author's revised manuscript: one of their models (economic variables, AR/VAR, PCA, ARIMA, ARFIMA) beat the **random walk** in 28% of cases, 41% (22/54) for European and 18% for US indices. The US fraction is printed as 15/72 (= 20.8%), so one figure is a typo.
- **Fernandes et al.'s 2007 working paper differs from the article:** ARMA/ARFIMA win at short horizons and lose at long ones, where START trees win. The 2014 article stresses HAR with covariates: S&P return (−), volume (+), small negative term spread.

**Why the random walk is hard to beat.** If $y_t=\log\text{VIX}_t$ is AR(1) with $\phi\approx1$, the predictable part of the next change, $(1-\phi)(\mu-y_t)$, is small next to the shock. **Rule:** every model is reported against the random walk, the default DM comparator.

## 4. The models we build

| Model | Equation (sketch) | Fitted on | Source | Milestone |
|---|---|---|---|---|
| Random walk | $\hat y_{t+h}=y_t$ | log-VIX | Benchmark | M2 |
| Historical mean | $\hat y_{t+h}=\bar y_{\text{train}}$ | log-VIX | Benchmark | M2 |
| ARIMA(p,d,q) | $\phi(L)(1-L)^d y_t=\theta(L)\varepsilon_t$ | log-VIX | Box & Jenkins (1970) | M2 |
| HAR | $y_{t+h}=\beta_0+\beta_d y_t+\beta_w\bar y_{t-4:t}+\beta_m\bar y_{t-21:t}+\varepsilon$ | log-VIX | Corsi (2009) | M2 |
| GARCH(1,1) | $\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2$ | SPX returns | Engle (1982); Bollerslev (1986) | M3 |
| GJR (modern form) | $+\gamma\,\varepsilon_{t-1}^2\,\mathbb{I}[\varepsilon_{t-1}<0]$ | SPX returns | Glosten, Jagannathan & Runkle (1993), reparameterised | M3 |
| EGARCH | $\ln\sigma_t^2=\omega+\alpha(\lvert e_{t-1}\rvert-\sqrt{2/\pi})+\gamma e_{t-1}+\beta\ln\sigma_{t-1}^2$ | SPX returns | Nelson (1991); `arch` 8.0.0 form | M3 |
| ARIMA-GARCH | ARMA mean + GARCH variance | $\Delta\log\text{VIX}$ | As above | M3 |
| LSTM | Gated recurrent network | log-VIX lags (+ features) | Hochreiter & Schmidhuber (1997) | M4 |

- **HAR.** Corsi (2009): an additive cascade of daily, weekly (5-day, eq. 4) and monthly (22-day) components (eq. 8). No true long memory, yet it reproduces long memory and fat tails and forecasts well.
- **GJR's own eq. 9** puts the indicator on **positive** shocks, $g_1\varepsilon^2+g_2\varepsilon^2\,\mathbb{I}[\varepsilon>0]$ with $g_2<0$. Setting $\alpha=g_1+g_2$, $\gamma=-g_2$ gives the modern negative-shock form; the economics is identical.
- **Does anything beat a GARCH(1,1)?** Among 330 ARCH-type models, Hansen & Lunde (2005) found no evidence that GARCH(1,1) was outperformed on DM–\$ rates, but it was clearly inferior to leverage models on IBM returns. For SPX, "asymmetric beats GARCH" is an M3 hypothesis.

**Multi-step GARCH variance** (ADR-0001 §4.4), with $\bar\sigma^2=\omega/(1-\alpha-\beta)$:

$$\hat\sigma^2_{t+k|t}=\bar\sigma^2+(\alpha+\beta)^{k-1}\left(\hat\sigma^2_{t+1|t}-\bar\sigma^2\right),\qquad G_t(a)=\frac{252}{21}\sum_{k=a+1}^{a+21}\hat\sigma^2_{t+k|t}$$

$G_t(a)$ is a real-world (P) annualised variance for the 21 sessions after $t+a$. The `rv21` track compares $G_t(0)$ with realised variance; the `vix_level` track compares $100\sqrt{G_t(h)}$ with $\text{VIX}_{t+h}$, whose window it matches. The premium (§2.1) predicts raw GARCH below VIX on average.

### 4.1 Why "GARCH on the IV level" is a category error

**What GARCH models:** the conditional variance of an *innovation*, $r_t=\mu+\varepsilon_t$, $\varepsilon_t=\sigma_t z_t$, $\sigma_t^2=\operatorname{Var}(\varepsilon_t\mid\mathcal F_{t-1})$. Engle (1982) defines ARCH processes as mean-zero and serially uncorrelated, with variance depending on the past. Setting $r_t:=\text{VIX}_t$ breaks this three ways:
1. **Wrong mean equation.** VIX is persistent, so $\varepsilon_t=\text{VIX}_t-\mu$ is strongly autocorrelated. $\varepsilon_t^2$ then autocorrelates because the *level* persists, not because of clustering, and $\hat\alpha+\hat\beta$ absorbs that persistence.
2. **Wrong object.** $\sigma^2_{t+h|t}$ becomes $\operatorname{Var}(\text{VIX}_{t+h}\mid\mathcal F_t)$, a vol-of-vol in VIX points²; the point forecast of VIX is just $\mu$.
3. **Wrong units.** Reading $\sigma_t$ as an IV forecast compares the volatility of volatility with volatility.

**Legitimate uses:** (a) GARCH, GJR, EGARCH on SPX returns, giving $G_t(\cdot)$; (b) ARIMA-GARCH on $\Delta\log\text{VIX}_t$: ARMA forecasts the path, GARCH the vol-of-vol that sets the forecast variance. Exponentiating a log forecast gives the median, so log-space models report the lognormal mean $\exp(m+s^2/2)$ (ADR-0001 §4.3).

### 4.2 LSTM and the evidence on ML

| Paper | Target | Finding |
|---|---|---|
| Bucci (2020) | Realised volatility | LSTM and NARX beat econometric models; long-range dependence *seems* to help in a highly volatile period |
| Christensen, Siggaard & Veliyev (2023) | RV, DJIA stocks | Lightly tuned ML beats the HAR family, even on HAR lags alone; gains grow with horizon, attributed to higher persistence in the ML models |
| Vrontos et al. (2021) | VIX direction | ML beats econometric models |

**Caveats.** This evidence is mostly about **realised** volatility, not daily IV levels, and pure HAR is hard to beat for VIX (Fernandes et al. 2014); M4 presumes nothing. **Leakage controls:** identical walk-forward folds, scalers fitted on training data only, early stopping inside the training window, several seeds.

## 5. Evaluation

### 5.1 Loss functions

A loss ranks forecasts correctly only for the statistic it is consistent for (Gneiting 2011). ADR-0001 §4.3:

| Loss | Consistent for | Role |
|---|---|---|
| RMSE (level) | The **mean** of $y$ | **Primary** for plug-in forecasts; a DM loss |
| QLIKE, $v/\hat v-\ln(v/\hat v)-1$ | The **mean** of variance $v$ | **Primary** for variance-native forecasts (GARCH family, VIX-implied); a DM loss |
| MAE | The **median** of $y$ | Reported because the brief asks; can reward low forecasts |
| MAPE, $\lvert y-\hat y\rvert/y$ | The median of order −1, below the median (Gneiting 2011) | Reported because the brief asks; rewards low forecasts more strongly |

**Patton (2011): noisy proxies.** With a conditionally unbiased noisy proxy for latent variance, only MSE (on variance) and QLIKE, of nine common losses, rank forecasts as the true variance would. Both sit in one homogeneous family ($b=0$: MSE; $b=-2$: QLIKE); QLIKE is the only robust loss depending only on the ratio. "Robust" does **not** mean robust to outliers. QLIKE had the most power among robust losses in simulations (Patton & Sheppard 2009).
- **Patton's MAE-prop, $\lvert\hat\sigma^2/h-1\rvert$, divides by the forecast $h$**: it is not the project's MAPE. Measured here (2,000,000 draws, proxy $\sigma^2z^2$): its minimiser is $2.36\sigma^2$, rewarding **over**-forecasts. The project's MAPE divides by the actual and rewards **under**-forecasts.
- **For us.** Patton's argument bites on the noisy `rv21` target, hence QLIKE on variance there. RMSE in vol points targets the mean of the realised quantity itself. VIX targets are observed almost without noise.
- **MAPE arithmetic.** A 1-point miss costs $1/12=8.3\%$ at VIX 12 but $1/30=3.3\%$ at VIX 30: calm days weigh 2.5× more.

### 5.2 Diebold–Mariano (ADR-0001 §4.8)

$$d_t=L(e_{1t})-L(e_{2t}),\qquad \text{DM}=\frac{\bar d}{\sqrt{\widehat{\text{LRV}}/n}},\qquad \widehat{\text{LRV}}=\hat\gamma_0+2\sum_{j=1}^{h-1}\hat\gamma_j$$

- **The test (Diebold & Mariano 1995).** Loss need not be quadratic or symmetric; errors may be non-Gaussian, serially and contemporaneously correlated. Diebold (2015) frames it as regressing $d_t$ on a constant with HAC errors.
- **Truncation at $h-1$ is an assumption.** Optimal $h$-step *errors* are MA($h-1$) (Harvey, Leybourne & Newbold 1997, p. 281), but that need not carry over to $d_t$ for nonlinear losses. When relative accuracy depends on a persistent volatility state, the normal case here, $d_t$ stays autocorrelated beyond $h-1$; under such a null the ADR measured size 0.1258 at $n=583$ (nominal 0.05). Remedy: open item O6, decided in M5.
- **Estimator.** The rectangular (equal-weight) LRV above, DM's original, times the HLN factor $\sqrt{(n+1-2h+h(h-1)/n)/n}$ with $t_{n-1}$ p-values (HLN 1997, eq. 9). If it is ≤ 0, fall back to Bartlett at lag $h-1$ and flag the row. Bartlett is not the default: at lag $h-1$ it under-estimates the LRV of MA($h-1$) differentials (ADR-measured size 0.09–0.12).
- **Do not use `statsmodels.tsa.stattools.diebold_mariano_test` (0.15.0) here.** It uses Bartlett (Newey–West) weights at any `lags` (default `max(h-1, ceil(n**(1/3)))`), i.e. the fallback. The rectangular estimate is `OLS(d, 1).fit(cov_type="HAC", cov_kwds={"maxlags": h-1, "kernel": "uniform", "use_correction": False})`, HLN applied by hand. Measured here on a simulated MA(20) differential ($n=583$): uniform-kernel t = −2.852070 = hand-coded rectangular; the same call without `kernel` gives −3.475135 = hand-coded Bartlett.
- **Caveat (Diebold 2015).** DM compares *forecasts*; for comparing *models* he argues full-sample procedures are simpler and more compelling, with pseudo-out-of-sample analysis still useful for historical episodes.

### 5.3 Mincer–Zarnowitz and the model confidence set

$$y_{t+h}=a+b\,\hat y_{t+h|t}+u_{t+h},\qquad H_0:\ (a,b)=(0,1)$$

- **Mincer & Zarnowitz (1969).** A forecast is **unbiased** if $E(\hat y)=E(y)$ (p. 8) and **efficient** if the slope is 1, i.e. the error is uncorrelated with the forecast (p. 9). $a=0$ alone is not unbiasedness: with $b\neq1$ it still allows $E(y)\neq E(\hat y)$. The joint Wald test of $(a,b)=(0,1)$ tests both. MSE splits into mean, slope and residual components.
- **Here** the Wald test uses Newey–West at lag $h-1$, with the same Bartlett bias; open item O5.
- **$R^2$ as a ranking.** MZ $R^2$ is robust to proxy noise; log-regression $R^2$ is not (Patton & Sheppard 2009, citing Meddahi and Hansen & Lunde).
- **Model confidence set (aside).** Contains the best model with given confidence; uninformative data give a large set (Hansen, Lunde & Nason 2011). Optional in M5.

## 6. Trading on volatility forecasts

**VIX is not tradable**; the proxies are built on futures.
- **Index.** The S&P 500 VIX Short-Term Futures Index rolls daily from the first- to the second-month future, targeting a one-month weighted-average maturity (ProShares n.d.).
- **Roll cost (my approximation).** With an unchanged curve, a long constant-maturity position loses roughly $(F_2-F_1)/F_1$ per $(\tau_2-\tau_1)$ days in contango and gains in backwardation.
- **Whaley (2013).** VIX ETPs launched in January 2009; the popular ones are "virtually guaranteed to lose money over time"; short-term-futures ETPs had lost nearly USD 4 billion by 2013.
- **Simon & Campasano (2014).** Shorting futures in contango and buying in backwardation, when the daily roll exceeded ±0.10 points, hedged with mini-S&P futures, was profitable after costs.

| Date | Event | Source |
|---|---|---|
| 5 Feb 2018 | XIV intraday indicative value ≤ 20% of prior close: acceleration event (announced 6 Feb; valuation date 15 Feb; acceleration/payment date 21 Feb) | Credit Suisse (2018) |
| Close of 27 Feb 2018 | SVXY −1x → −0.5x; UVXY 2x → 1.5x; SVXY returns before 28 Feb 2018 reflect −1x | ProShares (2018a, n.d.) |
| 18 Sep 2018 | SVXY 1-for-4 reverse split | ProShares (2018b) |
| 30 Jan 2019 | Original VXX (Series A) matured | Barclays (2026) |
| 2 May 2019 | Series B: VXXB became VXX | MIAX (2019); Barclays (2026) |
| 23 Apr 2021, 7 Mar 2023, 24 Jul 2024 | VXX 1-for-4 reverse splits | Barclays (2023, 2026); OCC (2021, 2024) |
| 11 Apr 2024 | SVXY 2-for-1 split | ProShares (2024) |

Series A VXX had 1-for-4 splits on 9 Nov 2010, 5 Oct 2012, 8 Nov 2013, 9 Aug 2016 and 23 Aug 2017 (Barclays 2018). yfinance 1.7.0 also shows SVXY 2-for-1 splits on 2012-10-05, 2014-01-24 and 2017-07-17 (read 2026-10-28; vendor data only).

**M6 backtest pitfalls:**
1. **Split adjustment** rewrites past prices; check returns around every split above.
2. **Which VXX.** Pre-2019 VXX was a different note; M1 checks what yfinance splices.
3. **SVXY's Feb-2018 leverage break:** split the sample or rebuild −0.5x from index returns.
4. **Daily reset.** A multi-day SVXY return is not −0.5 × the index's (compounding).
5. **Survivorship.** XIV is absent.

**Always-short-vol is a strong naive benchmark.** The variance premium (Carr & Wu 2009) and the futures risk premium (Simon & Campasano 2014) make it pay on average; XIV shows its tail. A forecast strategy must beat it risk-adjusted.

**Sharpe-ratio pitfalls.**
- **Annualisation.** Under serial correlation, $\sqrt{q}$ scaling is wrong; in Lo's (2002) example a hedge fund's annual Sharpe is overstated by up to 65%.
- **PSR** (Bailey & López de Prado 2012) adjusts for skew and fat tails, which matter for short vol.
- **DSR** (Bailey & López de Prado 2014): the PSR at the expected maximum Sharpe across $N$ **independent** trials. **Log every variant**; counting correlated variants as independent overstates $N$, erring conservative.
- **Reuse:** `~/dev/sharpe-observatory/scripts/metrics.py:97,122,147` (MIT).

$$\text{PSR}(SR^*)=\Phi\!\left(\frac{(\widehat{SR}-SR^*)\sqrt{n-1}}{\sqrt{1-\hat\gamma_3\widehat{SR}+\frac{\hat\gamma_4-1}{4}\widehat{SR}^2}}\right)$$

Per-period $\widehat{SR}$, $SR^*$; $\hat\gamma_3$ skewness; $\hat\gamma_4$ Pearson kurtosis (normal = 3).

## 7. Paper → what we take → where it is used

| Paper | What we take | Milestone / module (planned) |
|---|---|---|
| Cboe (2026a,b); Demeterfi et al. (1999); Carr & Wu (2009) | What VIX is; units; variance-swap link | M1 `ivf.data`, `ivf.features` |
| Cboe term structure; Cont (2001); Black (1976) | VIX9D…VIX1Y, contango flag; stylised facts | M1 `ivf.features`, EDA |
| Bollerslev, Tauchen & Zhou (2009); Bekaert & Hoerova (2014) | VRP definition and sign | M1 IV–RV spread |
| Christensen & Prabhala (1998); Poon & Granger (2003, 2005); Bandi & Perron (2006) | IV as a forecast of RV; why $b\neq1$ | M3, M5 MZ |
| Harvey & Whaley (1992); Konstantinidi et al. (2008) | Statistical ≠ economic predictability | M5, M6 |
| Fernandes et al. (2014); Corsi (2009); Box & Jenkins (1970) | HAR baseline; ARIMA on log-VIX | M2 `ivf.models` |
| Engle (1982); Bollerslev (1986); GJR (1993); Nelson (1991); Hansen & Lunde (2005) | Variance models on returns; GARCH(1,1) as benchmark | M3 `ivf.models` (`arch` 8.0.0) |
| Hochreiter & Schmidhuber (1997); Bucci (2020); Christensen et al. (2023); Vrontos et al. (2021) | LSTM and its expected value | M4 `ivf.models` |
| Gneiting (2011); Patton (2011); Patton & Sheppard (2009); Andersen & Bollerslev (1998) | Primary losses; QLIKE; proxy noise | M5 `ivf.evaluation` |
| Diebold & Mariano (1995); Harvey et al. (1997); Diebold (2015); Mincer & Zarnowitz (1969); Hansen, Lunde & Nason (2011) | DM (rectangular + HLN, Bartlett fallback); MZ joint test; MCS | M5 `ivf.evaluation` |
| Whaley (2013); Simon & Campasano (2014); Barclays; ProShares; OCC; MIAX; Credit Suisse | ETP mechanics and history | M6 `ivf.backtest` |
| Lo (2002); Bailey & López de Prado (2012, 2014) | Sharpe inference, PSR, DSR | M6 `ivf.backtest` |

## 8. What we will NOT claim

- **That a forecast is tradable.** VIX is not investable; VXX/SVXY track futures. Statistical wins are not profits (Harvey & Whaley 1992; Konstantinidi et al. 2008).
- **Accurate RV targets.** RV from daily squared returns is a noisy proxy (Andersen & Bollerslev 1998; Patton 2011); without intraday data, RV comparisons have low power.
- **Synchronous closes.** VIX is disseminated until 16:15 ET (Cboe 2026a). **[unverified: which timestamp the CBOE CSV close uses, hence whether it is synchronous with SPX and ETP closes; M1 checks for look-ahead.]**
- **Regime conclusions from the holdout.** ADR-0001 proposes a fixed holdout start of 2024-07-01, about 28 months (§4.6, open item O1); a trailing 24 months would start 2024-10-28 and exclude Aug-2024. With two spikes, regime results are case studies, not statistics.
- **Survivorship-free ETP history.** XIV and Series-A VXX are missing or spliced; SVXY's leverage changed.
- **That ML results on RV transfer to daily IV** (§4.2).
- **That DM p-values settle model choice.** DM compares forecasts (Diebold 2015); its size under state-dependent losses is open (O6).
- **Exact costs.** The cost model is an assumption, sensitivity-checked.
- **Any number not measured here.** Figures above are other authors' results, except the two probes marked "measured here".

---

## References

- Andersen, T. G. & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review* 39(4), 885–905. doi:10.2307/2527343. Abstract checked via NBER w6023: https://www.nber.org/papers/w6023
- Bailey, D. H. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2), 3–44. doi:10.21314/JOR.2012.255
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management* 40(5), 94–107. doi:10.3905/jpm.2014.40.5.094
- Bandi, F. M. & Perron, B. (2006). Long memory and the relation between implied and realized volatility. *Journal of Financial Econometrics* 4(4), 636–670. doi:10.1093/jjfinec/nbl003
- Barclays Bank PLC (2018). Pricing supplement (Form 424B2), iPath S&P 500 VIX Short-Term Futures ETNs, 4 Jan 2018. https://www.sec.gov/Archives/edgar/data/312070/000110465918000716/a18-2032_7424b2.htm
- Barclays Bank PLC (2023). Pricing supplement (Form 424B2), iPath Series B S&P 500 VIX Short-Term Futures ETNs, 7 Mar 2023. https://www.sec.gov/Archives/edgar/data/312070/000148105723001590/form424b2.htm
- Barclays Bank PLC (2026). Pricing supplement (Form 424B2), iPath Series B S&P 500 VIX Short-Term Futures ETNs, 13 Feb 2026. https://www.sec.gov/Archives/edgar/data/312070/000191870426004069/form424b2.htm
- Bekaert, G. & Hoerova, M. (2014). The VIX, the variance premium and stock market volatility. *Journal of Econometrics* 183(2), 181–192. doi:10.1016/j.jeconom.2014.05.008 (abstract via NBER w18995)
- Black, F. (1976). Studies of stock price volatility changes. *Proc. 1976 Meetings of the ASA, Business and Economic Statistics Section*, 177–181. **[unverified: bibliographic details from secondary listings; original not retrieved]**
- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy* 81(3), 637–654. doi:10.1086/260062
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics* 31(3), 307–327. doi:10.1016/0304-4076(86)90063-1
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies* 22(11), 4463–4492. doi:10.1093/rfs/hhp008
- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Britten-Jones, M. & Neuberger, A. (2000). Option prices, implied price processes, and stochastic volatility. *Journal of Finance* 55(2), 839–866. doi:10.1111/0022-1082.00228
- Bucci, A. (2020). Realized volatility forecasting with neural networks. *Journal of Financial Econometrics* 18(3), 502–531. doi:10.1093/jjfinec/nbaa008
- Carr, P. & Wu, L. (2004). Variance risk premia. Working paper, version of 27 May 2004. https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf (despite the file name, this is the 2004 working paper)
- Carr, P. & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies* 22(3), 1311–1341. doi:10.1093/rfs/hhn038
- Cboe (2026a). *Cboe Volatility Index Methodology*, v6.0, last revised 26 Feb 2026. https://cdn.cboe.com/resources/indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf
- Cboe (2026b). *Cboe Volatility Index Mathematics Methodology*, v5.0, last revised 26 Feb 2026. https://cdn.cboe.com/resources/indices/governance/Cboe_Volatility_Index_Mathematics_Methodology.pdf
- Cboe (n.d.). VIX term structure. https://www.cboe.com/tradable_products/vix/term_structure/ ; VIX3M product page, Wayback capture 15 Oct 2017: https://web.archive.org/web/20171015065020/http://www.cboe.com:80/products/vix-index-volatility/volatility-on-stock-indexes/cboe-3-month-volatility-index-vxv ; `VIX3M_History.csv`: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv (all accessed 2026-10-28)
- CFE (2004). VX futures trade data, May 2004 contract (`CFE_K04_VX.csv`), Wayback capture 3 Feb 2007: http://web.archive.org/web/20070203014242/http://cfe.cboe.com:80/Publish/ScheduledTask/MktData/datahouse/CFE_K04_VX.csv
- Christensen, B. J. & Prabhala, N. R. (1998). The relation between implied and realized volatility. *Journal of Financial Economics* 50(2), 125–150. doi:10.1016/S0304-405X(98)00034-8
- Christensen, K., Siggaard, M. & Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics* 21(5), 1680–1727. doi:10.1093/jjfinec/nbac020
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance* 1(2), 223–236. doi:10.1080/713665670
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7(2), 174–196. doi:10.1093/jjfinec/nbp001
- Credit Suisse AG (2018). Credit Suisse AG announces the event acceleration of its XIV ETNs, 6 Feb 2018. Form 6-K, Exhibit 99.1: https://www.sec.gov/Archives/edgar/data/1053092/000095010318001572/dp86358_ex9901.htm
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). A guide to volatility and variance swaps. *Journal of Derivatives* 6(4), 9–32. doi:10.3905/jod.1999.319129 (Goldman Sachs research note, March 1999: "More than you ever wanted to know about volatility swaps")
- Diebold, F. X. (2015). Comparing predictive accuracy, twenty years later: a personal perspective on the use and abuse of Diebold–Mariano tests. *Journal of Business & Economic Statistics* 33(1), 1. doi:10.1080/07350015.2014.983236 (text via NBER w18391)
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13(3), 253–263. doi:10.1080/07350015.1995.10524599
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica* 50(4), 987–1008. doi:10.2307/1912773
- Fernandes, M., Medeiros, M. C. & Scharth, M. (2007). Modeling and predicting the CBOE market volatility index. PUC-Rio Texto para Discussão 548, Wayback capture 9 Dec 2007: http://web.archive.org/web/20071209030353/http://www.econ.puc-rio.br/PDF/td548.pdf (the current RePEc handle serves a later revision)
- Fernandes, M., Medeiros, M. C. & Scharth, M. (2014). Modeling and predicting the CBOE market volatility index. *Journal of Banking & Finance* 40, 1–10. doi:10.1016/j.jbankfin.2013.11.004
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance* 48(5), 1779–1801. doi:10.1111/j.1540-6261.1993.tb05128.x
- Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association* 106(494), 746–762. doi:10.1198/jasa.2011.r10138
- Hansen, P. R. & Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)? *Journal of Applied Econometrics* 20(7), 873–889. doi:10.1002/jae.800
- Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2), 453–497. doi:10.3982/ECTA5771
- Harvey, C. R. & Whaley, R. E. (1992). Market volatility prediction and the efficiency of the S&P 100 index option market. *Journal of Financial Economics* 31(1), 43–73. doi:10.1016/0304-405X(92)90011-L. Abstract: https://people.duke.edu/~charvey/papers/p16.html
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13(2), 281–291. doi:10.1016/S0169-2070(96)00719-4
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9(8), 1735–1780. doi:10.1162/neco.1997.9.8.1735
- Jiang, G. J. & Tian, Y. S. (2005). The model-free implied volatility and its information content. *Review of Financial Studies* 18(4), 1305–1342. doi:10.1093/rfs/hhi027
- Konstantinidi, E., Skiadopoulos, G. & Tzagkaraki, E. (2008). Can the evolution of implied volatility be forecasted? Evidence from European and US implied volatility indices. *Journal of Banking & Finance* 32(11), 2401–2411. doi:10.1016/j.jbankfin.2008.02.003. Author's revised manuscript (draft 18 Dec 2007), University of Exeter via figshare: https://ndownloader.figshare.com/files/56687882
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36–52. doi:10.2469/faj.v58.n4.2453
- MIAX (2019). Corporate action alert: iPath Series B S&P 500 VIX Short-Term Futures ETN, symbol VXXB → VXX effective 2 May 2019 (posted 1 May 2019). https://www.miaxglobal.com/alerts/2019/05/01/miax-options-miax-pearl-and-miax-emerald-corporate-action-alert-ipath-series-b-sp
- Mincer, J. A. & Zarnowitz, V. (1969). The evaluation of economic forecasts. In J. Mincer (ed.), *Economic Forecasts and Expectations*, NBER, 3–46. https://www.nber.org/chapters/c1214
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach. *Econometrica* 59(2), 347–370. doi:10.2307/2938260
- OCC (2021, 2024). Information Memos #48572 (14 Apr 2021) and #54913 (22 Jul 2024): VXX 1-for-4 reverse splits, hosted by MIAX. https://www.miaxglobal.com/sites/default/files/alert-files/VXX_RevSplit_48572.pdf ; https://miaxglobal.com/sites/default/files/alert-files/VXX_Reverse_Split_54913.pdf
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1), 246–256. doi:10.1016/j.jeconom.2010.03.034. Equations checked in the 2006 working paper (UTS QFRC RP 175).
- Patton, A. J. & Sheppard, K. (2009). Evaluating volatility and correlation forecasts. In T. G. Andersen et al. (eds), *Handbook of Financial Time Series*, Springer, 801–838. doi:10.1007/978-3-540-71297-8_36 (read as Oxford-Man working paper OMI02/07, 2007)
- Poon, S.-H. & Granger, C. W. J. (2003). Forecasting volatility in financial markets: a review. *Journal of Economic Literature* 41(2), 478–539. doi:10.1257/jel.41.2.478
- Poon, S.-H. & Granger, C. (2005). Practical issues in forecasting volatility. *Financial Analysts Journal* 61(1), 45–56. doi:10.2469/faj.v61.n1.2683
- ProShares Trust II (2018a). Form 8-K, Exhibit 99.1, press release of 26 Feb 2018 (SVXY/UVXY objective change). https://www.sec.gov/Archives/edgar/data/0001415311/000119312518059052/d503117dex991.htm
- ProShares Trust II (2018b). Form 8-K, Exhibit 99.1, press release of 29 Aug 2018 (share splits; SVXY 1-for-4 effective 18 Sep 2018). https://www.sec.gov/Archives/edgar/data/0001415311/000119312518262021/d615180dex991.htm
- ProShares Trust II (2024). Form 8-K filed 20 Mar 2024 (share splits; SVXY 2-for-1 effective 11 Apr 2024). https://www.sec.gov/Archives/edgar/data/1415311/000119312524072839/d795086d8k.htm
- ProShares (n.d.). SVXY fund page; "About S&P 500 VIX futures indexes" FAQ. https://www.proshares.com/our-etfs/strategic/svxy ; https://proshares.com/faqs/about-sp-500-vix-futures-indexes (accessed 2026-10-28)
- Simon, D. P. & Campasano, J. (2014). The VIX futures basis: evidence and trading strategies. *Journal of Derivatives* 21(3), 54–69. doi:10.3905/jod.2014.21.3.054
- Vrontos, S. D., Galakis, J. & Vrontos, I. D. (2021). Implied volatility directional forecasting: a machine learning approach. *Quantitative Finance* 21(10), 1687–1706. doi:10.1080/14697688.2021.1905869
- Whaley, R. E. (1993). Derivatives on market volatility: hedging tools long overdue. *Journal of Derivatives* 1(1), 71–84. doi:10.3905/jod.1993.407868
- Whaley, R. E. (2000). The investor fear gauge. *Journal of Portfolio Management* 26(3), 12–17. doi:10.3905/jpm.2000.319728
- Whaley, R. E. (2013). Trading volatility: at what cost? *Journal of Portfolio Management* 40(1), 95–108. doi:10.3905/jpm.2013.40.1.095
