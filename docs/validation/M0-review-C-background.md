# M0 review C: literature brief `docs/concepts/00-background.md` (evidence-gated citation and fact review)

| | |
|---|---|
| Target | `docs/concepts/00-background.md` sha256 `a12daf40…b528f`; compared against `docs/adr/0001-python-core-versioned-json-app.md` sha256 `c855b112…6741e` (untracked tree, 2026-10-28). The ADR was edited by someone else during the review (now `14495e46…`). The passages claim 6 relies on were re-checked in the new version and are unchanged: 4.3 primary losses; 4.6 "exclude Aug-2024"; 4.8 rectangular, assumption and 0.1258; O1; O6. |
| Environment | repo `.venv`: Python 3.12.13, statsmodels 0.15.0 (pinned `==0.15.0` in `pyproject.toml`), numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, arch 8.0.0, yfinance 1.7.0 |
| Mode | Tier 2 (dual channel) for every claim about a paper, an instrument or a library; Tier 1 for structure. **Channel A** = three separate subagents restricted to primary sources and retrieval (papers in econometrics; papers on IV, VIX and ML; issuer and exchange documents). **Channel B** = the reviewing context: a second registry (OpenAlex against Crossref), market and exchange data (yfinance, Cboe CSVs, archived CFE trade files, Wayback captures), and computation. |
| Independence caveat | The reviewer read review B's ledger before registering this one; that ledger contains Channel-A findings on Corsi, HLN, VXX, SVXY and XIV. Channel A's reports also reached the reviewer before the Channel B probes ran. Probe designs were frozen first (`claims-ledger.md`, 14:40Z). Rows whose probe was added later are marked **ADDED**. Agreement here is therefore **dual-confirmed (fallback mode)**, not independent confirmation. |
| Default | Every material claim started as wrong. |
| Scratch | `<session scratchpad>/reviewC/`: `claims-ledger.md`, `channel-a-{econometrics,ivpapers,instruments}.md`, `probes/` (`refs.py`, `authors_rev.py`, `dm_probe.py`, `market.py`, `psr_mc.py`, `patton_probe.py`, `oa_abs.py`), and downloads in `dl-*`. None of it is in the repo. |

## Verdict: FAIL

| # | Claim | Verdict | Worst severity |
|---|---|---|---|
| 1 | Every reference exists with the stated metadata | **REFUTED in part.** All 41 DOI'd references exist and match. Three non-DOI references are mis-attributed or mis-titled. | HIGH (per the brief's rule; low practical impact) |
| 2 | Each paper is described correctly | **REFUTED in part.** 6 attributions or glosses are wrong; most samples survive. | MED |
| 3 | Instrument facts | **REFUTED in part.** VXX's 2023 split is missing; the futures launch date is resolvable to 26 Mar 2004 and its citation is wrong. | MED |
| 4 | statsmodels 0.15.0 `diebold_mariano_test`, `harvey_adj`, default lags | **SURVIVES**, but the doc omits that the kernel is **Bartlett**, the estimator the ADR rejected | MED-HIGH (as omission, see claim 6) |
| 5 | `[unverified]` tags are warranted and not restated as fact | **REFUTED.** 9 of 11 tags are resolvable now. The errors sit in *untagged* sentences. | MED |
| 6 | Doc agrees with ADR-0001 as amended | **REFUTED** in three places: holdout length, DM long-run variance and O6, primary losses | MED-HIGH |
| 7 | Copyright | **SURVIVES** | LOW notes |

## Evidence table

A = primary-source channel (document, page or equation, short quote). B = registry, data or computation channel (command and verbatim output). Probe records with their falsifiers are in `probes/`.

### Claim 1: references

| ID | Claim | A | B | Gate | Sev |
|---|---|---|---|---|---|
| R-DOI | 41 DOI'd entries: authors, year, title, journal, volume, issue, pages | Crossref `/works/{doi}` per entry (`refs.py`) | OpenAlex `/works/doi:` per entry. All flags are artefacts: HTML in titles, `&amp;`, online-first years, missing last pages. Authors checked in both directions (`authors_rev.py`). **Falsifier:** year+1 and issue+1 flagged by both registries; an added co-author was missed by the first comparator, so the reverse check was ADDED, and it flags it. | ACCEPTED | — |
| R-DOI.n1 | Titles complete | Crossref: DSR 2014's subtitle "Correcting for Selection Bias, Backtest Overfitting, and Non-Normality" and Diebold 2015's subtitle "A Personal Perspective on the Use and Abuse of Diebold–Mariano Tests" are both dropped | same in OpenAlex | REFUTED (truncation) | LOW |
| R-DOI.n2 | Engle 1982 pages 987–1007 | Econometric Society page gives 987–1008 (Channel A) | Crossref and OpenAlex give the first page only | UNRESOLVED | LOW |
| R-DOI.n3 | Corsi 2009 | JFEc 7(2) 174–196 | Crossref issued 2008-11-07 (online first); the volume-7 issue is 2009 | ACCEPTED | — |
| **R-INV** | "Bloomberg via Investing.com (2019)" | Investing.com JSON-LD author is the organisation Seeking Alpha; canonical source is seekingalpha.com/news/3427129 | Wayback capture 2022-07-06: "…set to die this week By Seeking Alpha"; Bloomberg appears only as "Bloomberg reports" in the body | **REJECTED: mis-attributed byline** | **HIGH** under the brief's rule. Practical impact is low: the 30 Jan 2019 date it supports is correct per Barclays. |
| **R-MIAX** | "MIAX (2021, 2024)" | Documents #48572 (14 Apr 2021) and #54913 (22 Jul 2024) are OCC Information Memos hosted on miaxglobal.com | `pdfinfo` Author: **OCC** on both; text carries OCC's adjustment disclaimer | **REJECTED: mis-attributed author** (MIAX 2019 is a genuine MIAX alert) | **HIGH** under the brief's rule; the dates they support are right |
| R-CS | "Credit Suisse (2018). Credit Suisse AG announces event acceleration of its XIV ETNs" | 6-K Ex. 99.1 title differs | The linked page's `<title>`: "…Announces Event Acceleration Of Its Daily Inverse VIX Short Term ETNs". The claim that the credit-suisse.com URL redirects to UBS cannot be checked, because no URL is given. | REJECTED (wrong title) | MED |
| R-PS | Patton & Sheppard: "[published-version details unverified]" | Channel A: Crossref record | Crossref `10.1007/978-3-540-71297-8_36`: *Handbook of Financial Time Series*, Springer 2009, pp. 801–838 | Tag unwarranted (see claim 5) | LOW |
| R-URL | 15 non-DOI URLs resolve | Channel A opened each | HTTP 200 for 14; Investing.com 403 to curl, but Wayback 200 | ACCEPTED | — |
| R-FMSWP | RePEc `rio:texdis:548` is the 2007 working paper the doc describes | The original Aug-2007 PDF (Wayback) has the ARMA/ARFIMA short-versus-long result | The current PDF at that handle is a **revised** version, with oil price and a term spread that "has no long-run impact" (`fms_td548.txt` lines 42–44) | REFUTED (link serves a different version) | LOW |
| R-Black | Black 1976 tag | No registry record | OpenAlex search finds nothing | Tag warranted | — |

### Claim 2: paper descriptions (sampled set as requested, plus everything Channel A reached)

| ID | Claim | A | B | Gate | Sev |
|---|---|---|---|---|---|
| P01 | Corsi: additive cascade; no true long memory but reproduces long memory and fat tails; 1/5/22 | Abstract p.174; weekly 5-day eq.(4) p.177; HAR eq.(8) p.181; monthly 22 days p.188 | OpenAlex abstract has cascade, long memory, fat tails, forecast. Review B's earlier retrieval also found eq.(8) p.181 with 5/22. | ACCEPTED (fallback) | — |
| P02 | Patton 2011: nine losses; only MSE and QLIKE robust; b=0 MSE, b=−2 QLIKE; QLIKE the only robust loss depending only on the ratio | p.250; eq.(24) p.252; Prop. 2(ii) p.251 | `patton_probe.py`, proxy σ²z²: argmin h/σ² = MSE 0.9981, QLIKE 0.9981, MSE-SD 0.6348, MAE 0.4533, MAE-prop 2.3652. eq.(24) at b=0 is 0.32 = (σ̂²−h)²/2; at b→−2 it is 0.252900 = QLIKE. **Falsifier:** b→−1 gives 0.2812. | ACCEPTED | — |
| P02f | Table row "MAPE-type ("MAE-prop") \|σ̂²/h−1\|" | Patton eq.(13) divides by the **forecast**; his footnote 8 calls it MAPE | The project's MAPE divides by the **actual** (doc §5.1 bullet; ADR 4.3). The minimiser for /forecast is **2.37σ²** (over-forecasts win); for /actual it runs to the lower bound (under-forecasts win). | REFUTED as applied: the row reads as covering the project's MAPE, but that loss has the opposite bias | LOW-MED |
| P03 | Patton & Sheppard: QLIKE most powerful among robust losses; MZ R² robust, log-regression R² not | pdf p.15; power holds within the homogeneous robust family, in 3 of 4 panels | n/a (A only; claim is about their simulations) | ACCEPTED (scope caveat) | LOW |
| P04 | DM 1995: non-quadratic, asymmetric loss; non-Gaussian, nonzero-mean, serially and contemporaneously correlated errors | Abstract | OpenAlex abstract contains all six terms | ACCEPTED | — |
| P05 | HLN: factor `√((T+1−2h+h(h−1)/T)/T)`, t(T−1) | eq.(9) p.283 | `dm_probe.py`: the library's ratio equals the formula to 8 dp (0.94994737 at T=50, h=3). **Falsifier:** using h in place of 2h gives 0.98102. | ACCEPTED | — |
| **P06** | MZ 1969: "a=0 means unbiased, b=1 is what they call efficient", MSE split into three parts | p.8: unbiased means E(P)=E(A). p.9: efficient means σ²(u)=σ²(v), i.e. β=1. Decomposition eq.(5a) p.11 | Counterexample: A = 0·1 + 0.5·P + v with E(P)=20 gives α=0 but E(A)=10 ≠ E(P), so the forecast is biased | **REJECTED** for "a=0 means unbiased"; the rest ACCEPTED | MED (an interview-level definition) |
| P07 | Hansen & Lunde: 330 models, out of sample, DM–$ and IBM | Published abstract | OpenAlex abstract: 330; DM–$; IBM via "a new data set of realized variance"; "no evidence that a GARCH(1,1) is outperformed"; "clearly inferior" to leverage models | ACCEPTED | LOW: "Nothing beat" turns *no evidence* into evidence of absence; §7's "leverage helps for equities" generalises from one stock |
| P08 | Christensen & Prabhala | Publisher abstract | OpenAlex abstract confirms everything, and also credits "**longer time series** and nonoverlapping data"; the doc keeps only overlapping data | ACCEPTED | LOW (omission) |
| P09a | Carr & Wu: VRP = RV − SW; strongly negative for S&P 500, S&P 100 and DJIA | RFS p.1321 `(RV − SW)`; p.1312 | OpenAlex: "the difference between the realized variance and this synthetic variance swap rate" | ACCEPTED | — |
| P09c | "Also negative, but smaller, for the Nasdaq-100 and most stocks" | 2009 RFS never says "smaller": NDX t-statistics are "lower, but remain strongly significant"; stocks show "large cross-sectional variation". "Smaller absolute magnitude" is 2004 working-paper wording. | `cw_rfs2009.txt` lines 62–66 and 644; WP lines 127 and 1339 | REFUTED for the cited 2009 version | LOW |
| P09e | "the Fama–French factors do not explain it" | p.1325: they "can only explain a small portion" | — | REFUTED (overstated) | LOW |
| **P09g** | "26 March (Carr & Wu 2009)" | Not in the 2009 RFS. "On March 26, 2004, the CBOE launched…" is in the 2004 WP "Variance Risk Premia" p.1. | `cw_wp.txt:85-86` has it; `cw_rfs2009.txt` has no "CFE" or "Futures Exchange" | **REJECTED (wrong source)** | MED |
| P10 | BTZ: VRP = implied − realised; strongest at the quarterly horizon | eq.(22) p.4475; p.4463 | OpenAlex: "particularly strong at the intermediate quarterly return horizon" | ACCEPTED | — |
| P11 | Bekaert & Hoerova | RePEc and NBER abstracts | (OpenAlex has no abstract) | ACCEPTED (A only) | — |
| P12 | FMS 2014: long memory; HAR; linearity not rejected; pure HAR hard to beat; covariate signs | 2014 abstract; NN-HAR in the revised WP p.9 | — | ACCEPTED | LOW: the §3 row merges the 2007 WP (where START, not HAR, won at long horizons: `fms_td548_2007.txt:46-47`) with the 2014 paper |
| P13 | KST: 28% (18% US, 41% EU) beat the RW; model list; no futures profits | Original accepted manuscript p.12, p.16, pp.14–15 | Vrontos et al. 2021 p.(vgv.txt 153-157) repeats 28/41/18. **Arithmetic:** the original prints "18% (15/72)", but 15/72 = 20.8%, and (22+15)/126 = 29.4%, not 28%. 13/72 = 18.1% and 35/126 = 27.8% would be consistent, so "15/72" is probably a typo in the original. | ACCEPTED (the doc's percentages match the original) | LOW (see "Found elsewhere") |
| P14 | GJR: `+γ ε² I[ε<0]` with "Source: GJR (1993)" | Published eq.(9) p.1787 and Table III: indicator = 1 if η_{t−1} **> 0**, g2 negative | The 1990 Minneapolis Fed WP 505 copy has the same "1 if … > 0" (`gjr_wp505.txt:2223`). The doc's form is the equivalent reparameterisation α′=α+γ, γ′=−γ. | REFUTED as GJR's own equation; the economics (positive shocks lower vol) is ACCEPTED | LOW (labelled "sketch") |
| P15 | EGARCH "`arch` documentation form" | Nelson abstract | `inspect.getdoc(arch.univariate.EGARCH)` in arch 8.0.0 gives exactly the doc's form with p=o=q=1 | ACCEPTED | — |
| P16/P17 | Engle 1982 ARCH definition; Bollerslev 1986 GARCH(1,1) | Abstract; Bollerslev eq.(8) p.311 | OpenAlex Engle abstract: "mean zero, serially uncorrelated … nonconstant variances conditional on the past" | ACCEPTED | — |
| P18 | LSTM = gated recurrent network | Input and output gates only; no forget gate in 1997 (0 hits for "forget") | OpenAlex: "multiplicative gate units" | ACCEPTED (the doc does not claim a forget gate) | — |
| P19 | PSR formula with √(n−1) and (γ̂₄−1)/4, Pearson kurtosis | eq.(11) p.9 | `psr_mc.py`, 20,000 × n=1000, skew −1.414, kurtosis 6, SR 0.4: Var(SR̂)(n−1) = **1.7549** (SE 0.0175) vs doc 1.7657. **Falsifier:** (γ₄−3)/4 gives 1.6857 (4.0 SE away) | ACCEPTED | — |
| P20 | DSR = PSR at E[max SR] over N trials | p.9 | `metrics.py:147` is exactly `probabilistic_sharpe_ratio(sr, expected_max_sr, …)` | ACCEPTED | LOW: N means *independent* trials, so logging correlated variants overstates it (conservative) |
| P21 | Lo: overstated "by as much as 65%" | CFA abstract | OpenAlex: "65 percent", serial correlation, hedge fund | ACCEPTED | — |
| P22 | Bucci: RNNs beat econometric models, "including in a high-volatility period" | IRIS abstract | OpenAlex: LSTM/NARX "**seems** to improve" accuracy in a highly volatile period | ACCEPTED; the hedge is dropped | LOW |
| P23 | CSV 2023: DJIA, minimal tuning, HAR lags only, gains grow with horizon "attributed to persistence" | Accepted-manuscript abstract | OpenAlex: "higher persistence **in the ML models**" | ACCEPTED; the referent is ambiguous in the doc | LOW |
| P24 | Vrontos: VIX direction; ML better statistically and economically | Open-access version | OpenAlex: direction, VIX, statistical, economic | ACCEPTED | — |
| P25 | Poon & Granger 93 papers; 2005 ranking | AEA abstract; FAJ abstract "possible ranking" | OpenAlex PG05: implied, historical, SV, ranking | ACCEPTED | — |
| P26 | Bandi & Perron | Published abstract plus 2003 WP for the slope bias | OpenAlex abstract has fractional, cointegration, inconsistent, bias, long-run unbiasedness | ACCEPTED | — |
| P27 | Harvey & Whaley 1992 | Duke abstract | — | ACCEPTED (A only) | — |
| P28 | Simon & Campasano: 2006–2011; the basis forecasts futures changes, not spot; hedged strategy profitable after costs | Publisher abstract plus 2013 manuscript | (The OpenAlex record holds an unrelated ChIP-seq abstract, so B is unusable) | ACCEPTED (A only) | LOW: the ±0.10 roll threshold is omitted |
| P28d | §1 table: contango means "(for futures, an expected VIX increase: Simon & Campasano 2014)" | The manuscript states it only as the **risk-neutral**, model-implied reading (`sc_may2013.txt:106`); their evidence is that futures converge toward spot | Internal check: doc line 47 states S&C's contrary P-measure finding | REFUTED as worded: it needs "risk-neutral", otherwise it reverses S&C | LOW-MED |
| P29 | Whaley 2013: launched Jan 2009; quote verbatim; nearly $4 bn | Archived publisher abstract | OpenAlex: "Launched in January 2009", "virtually guaranteed to lose money over time" (not "through time"), "nearly $4 billion" | ACCEPTED | — |
| P30 | Cont fact numbers 2, 6, 7, 8, 9 | p.224, all match | — | ACCEPTED (A only) | — |
| P31 | Jiang & Tian; Britten-Jones & Neuberger; DDKZ replication | RePEc abstracts; DDKZ note p.9 | OpenAlex JT: jumps, subsumes, realized | ACCEPTED | — |
| P31e | "That portfolio is what VIX prices" under Demeterfi et al. (1999) | 1999 note: zero occurrences of "VIX" | Date logic: 1999 predates the 22 Sep 2003 redefinition; OpenAlex DDKZ abstract has no "VIX" | REFUTED (attribution; Carr & Wu RFS p.1313 supports the link) | LOW |
| P32 | Diebold 2015 | NBER w18391 | OpenAlex: forecasts, models, pseudo-out-of-sample | ACCEPTED | LOW: "needs care" understates his position |
| P33 | Whaley 2000 fear gauge; Andersen & Bollerslev; MCS | Abstracts | OpenAlex: "called the investor fear gauge"; MCS: confidence, best, uninformative | ACCEPTED. Whaley 1993 "on such an index": title only, UNVERIFIABLE | LOW |

### Claim 3: instrument facts

| ID | Claim | A | B | Gate | Sev |
|---|---|---|---|---|---|
| I01 | VIX: 30-day; SPX AM + SPXW PM; Cboe quotes; ΔK; K₀; cubic-spline CMT; T in minutes/525,600; N₃₀ = 43,200 | Cboe 2026a v6.0 and 2026b v5.0, both revised 26 Feb 2026 | 30·1440 = 43,200 and 365·1440 = 525,600; review B reproduced App.3's 13.93 | ACCEPTED | — |
| I01d | "only non-zero bids are used" | 2026b App.1: since 10 Feb 2025, zero bid **or zero ask** excluded | `cboe_b.txt:268` "bid price or ask price equal to zero" | REFUTED (incomplete) | LOW |
| I01l | "Cboe's v6.0 document writes M where 01 writes N" | v6.0 pages rendered: no M symbol | `grep -c 𝑀`: v6.0 **0**, v5.0 **32** (M_T1, M_CM, M_365) | **REJECTED**: it is the v5.0 Mathematics Methodology, which writes M_CM, not M_30 | LOW |
| I02 | 1993 OEX ATM; 22 Sep 2003 with Goldman Sachs; data from Jan 1990 | 2026a §1, §5 | Cboe `VIX_History.csv` first row `01/02/1990` | ACCEPTED | — |
| I03 | VIX9D…VIX1Y; VXV → VIX3M on 18 Sep 2017 | Cboe VIX3M page, Wayback 2017-10-15 | Wayback captures of the same URL: **2017-09-13 shows "VXV" only**; 2017-10-15 states the change. This brackets 18 Sep. | ACCEPTED, so the tag is unwarranted | — |
| **I04** | "Sources disagree: 24 March (Cboe 2026a) vs 26 March (Carr & Wu 2009)" | 2026a line 52: "On March 24, 2004, Cboe introduced…" (true, but wrong). CFE launch-day release (26 Mar 2004) and the 2024 anniversary release both give 26 March. | **CFE's own archived trade files** `CFE_K04_VX.csv` and `CFE_M04_VX.csv` (Wayback 2006): first `Trade Date` is **3/26/2004** for both contracts | **REJECTED as framed.** It is resolvable to 26 Mar 2004; Cboe 2026a is the outlier; the C&W 2009 attribution is wrong (P09g) | MED |
| I05a-c | Series A matured 30 Jan 2019; VXXB → VXX 2 May 2019; 1-for-4 splits 23 Apr 2021 and 24 Jul 2024 | Barclays pricing supplements (4 Jan 2018, 13 Feb 2026); MIAX alert; OCC memos | `yf.Ticker('VXX').splits` → `[('2021-04-23', 0.25), ('2023-03-07', 0.25), ('2024-07-24', 0.25)]` | ACCEPTED | — |
| **I05d** | Split table lists only 2021 and 2024; "a possible 2023 split [unverified]" | Barclays supplement 7 Mar 2023: 1-for-4 effective 7 Mar 2023. Series A: 9 Nov 2010, 5 Oct 2012, 8 Nov 2013, 9 Aug 2016, 23 Aug 2017. | yfinance (above) shows **2023-03-07, 0.25** | **REJECTED**: the table is incomplete, and M6 pitfall 1 ("check returns around the split dates") would miss one of the three Series-B splits | MED |
| I06 | SVXY −1x → −0.5x after the close of 27 Feb 2018; UVXY 2x → 1.5x | ProShares 8-K Ex. 99.1: "effective as of close of business on February 27, 2018" | `market.py`: daily SVXY/VXX return ratio −1.08, −1.01, −1.19, −1.16 (22–27 Feb), then −0.31, −0.43, −0.46 (28 Feb–2 Mar). Slope on VXX: SVXY −0.835 → −0.430; UVXY 2.076 → 1.301. The break date is unambiguous; the magnitudes are attenuated by VXX tracking. | ACCEPTED | — |
| I07 | XIV: 5 Feb IIV ≤ 20% of prior close; announced 6 Feb; valuation 15 Feb | CS 6-K Ex. 99.1 | The doc's own cited mondovisione page says the same (20%, February 5, February 15) | ACCEPTED | — |
| I07d | "A search snippet of another copy gives 21 Feb [unverified]" | 21 Feb 2018 is the **acceleration (payment) date** | The **cited** page itself: "acceleration date … expected to be February 21, 2018, … three business days after the accelerated valuation date" | REJECTED: it is neither another copy nor a conflicting date | LOW |
| I08 | Short-Term Futures Index "created 2009"; daily roll; one-month weighted maturity | Roll and maturity confirmed. S&P DJI methodology (Jul 2024): launch **23 Dec 2008**; ProShares FAQ says 2009 | not probed | Roll ACCEPTED; "created 2009" UNRESOLVED (A refutes; B none) | LOW |
| I09 | VIX disseminated to 16:15 ET; Cboe treats 16:00 ET as the close | 2026a: RTH "Between 09:31 a.m. ET and 4:15 p.m. ET". The 4:00 p.m. "close of trading" sentence defines **SPXW expiry**. | `cboe_a.txt:286, 350` | 16:15 ACCEPTED; the 16:00 framing is misleading | LOW |

### Claim 4: statsmodels 0.15.0

| ID | Claim | A (installed source) | B (`dm_probe.py`, run in the repo `.venv`) | Gate |
|---|---|---|---|---|
| S1 | `diebold_mariano_test(..., harvey_adj=, horizon=)` exists | `statsmodels/tsa/stattools/_stattools.py:2756-2766`; exported from `statsmodels.tsa.stattools` | Import and call, rc=0 | ACCEPTED |
| S2 | HLN factor and t(T−1) p-values | Lines 2972–2977: `adj_factor = np.sqrt((t + 1 - (2 * horizon) + (horizon * (horizon - 1) / t)) / t)`; `stats.t.cdf(..., df=t - 1)` | Ratio of adjusted to unadjusted stat = formula to 8 dp at four (T, h) pairs. p_adj = t(T−1) value (0.0570106), not normal (0.0512742). **Falsifier:** the wrong factor differs (0.98102). | ACCEPTED |
| S3 | Default `lags = max(h−1, ceil(T**(1/3)))` | Line 2946: `int(max(horizon - 1, np.ceil(t ** (1 / 3))))` | T=50, h=3 → **4** (h−1 = 2); T=8, h=1 → 2; T=583, h=21 → 20 | ACCEPTED |
| **S4 ADDED** | Which kernel? (the doc is silent) | Line 2968: `OLS(d, ones).fit(cov_type="HAC", cov_kwds={"maxlags": lags})`, which is **Newey-West / Bartlett** | Library stat = Bartlett by hand to 6 dp at every (T, h); rectangular differs: **6.4020 vs 5.5597** (T=1000, h=21), **6.6018 vs 5.9919** (T=583, h=21); explicit `lags=h−1` is still Bartlett | **Finding:** pinning `lags=h−1` does **not** give ADR 4.8's rectangular estimator. HLN's factor was derived for the rectangular one (HLN eqs. 5–9). |

### Claim 5: `[unverified]` tags

| Tag | Resolvable now? | Source |
|---|---|---|
| U1 VXV → VIX3M 18 Sep 2017 | **Yes** | Cboe VIX3M page (Wayback 2017-10-15), bracketed by the 2017-09-13 capture |
| U2 KST 28/18/41 | **Yes** | Original accepted manuscript p.12 |
| U3 Corsi 1/5/22 | **Yes** | Corsi 2009 eq.(4) p.177, eq.(8) p.181, p.188 |
| U4 MA(h−1) "no source retrieved" | **Yes** | HLN 1997 p.281 (`hln1997.txt:43`); DM 1995 §1.1 |
| U5 HLN "original not retrieved" | **Yes** | HLN eq.(9) p.283. The ADR already removed its own tag in M0 (review B C2.4), so the doc is stale against the ADR. |
| U6 XIV 21 Feb | **Yes** | It is the acceleration date, in the cited page |
| U7 VXX Series A maturity [secondary] | **Yes** | Barclays pricing supplements on EDGAR |
| U8 earlier and 2023 VXX splits | **Yes** | Barclays supplements; yfinance |
| U9 which time the CSV close uses | No; tag warranted | No Cboe source says |
| U10 Black 1976 bibliography | No; tag warranted | No registry record |
| U11 Patton & Sheppard published version | **Yes** | Crossref 10.1007/978-3-540-71297-8_36 |

- **U12 (no tagged fact restated as fact elsewhere): SURVIVES.** Every tagged fact that is restated, such as the HAR windows in the §4 equation, HLN in §7, and "Pre-2019 VXX was a different note", turned out to be true.
- **The inverse failure is the finding.** The doc's errors sit in sentences that carry **no** tag: P06, P09g, P31e, P14, I01l, I04 and A-holdout.

### Claim 6: agreement with ADR-0001 (as amended in M0)

| ID | Doc | ADR | Gate | Sev |
|---|---|---|---|---|
| A1 | Targets: "our 21-day targets overlap"; no horizon list | 4.2: `vix_level` h = 1/5/21; `rv21` h = 21 | SURVIVES (no contradiction) | — |
| **A2** | §5.1:164 "**RMSE**/MAE/MAPE on the VIX level are reported because the brief asks"; nowhere says MAE/MAPE reward low forecasts. §5.1:162 "RV targets: use QLIKE and MSE on variance". | 4.3: RMSE (plugin) and QLIKE (native) are **primary** and the DM losses. Only MAE and MAPE are "reported because the brief asks", flagged "not consistent for the mean". RMSE for `rv21` plugin forecasts is in vol points. | **REJECTED** | MED |
| **A3** | §5.2:172: "Optimal h-step errors are MA(h−1) … so the long-run variance must be HAC with at least h−1 lags". §5.2:176: statsmodels' lags differ, "so M5 must pin and state the choice". No mention of the rectangular estimator, the Bartlett fallback, or O6. | 4.8: the vanishing autocorrelation beyond h−1 is an "**Assumption, not a consequence**". Rectangular LRV at h−1 + HLN + Bartlett fallback; Bartlett rejected (size 0.09–0.12). O6: size **0.1258** at n=583 under a state-dependent null. | **REJECTED.** The doc repeats the premise the ADR amended, and its statsmodels advice yields the rejected Bartlett estimator (S4). | **MED-HIGH** |
| A4 | D2: nothing to commit; no VXX/SVXY price quoted; hypothetical VIX 12/30 only | R1–R4 | SURVIVES | LOW: the M1 EDA list includes "decay half-life after each spike", "rolling sd" and kurtosis. Per-event statistics are extremes under R3 and rolling sd is a series, so the doc should say they stay local. |
| **A5** | §8:256 "The **24-month** holdout contains two spikes"; §2:61 "(both in our holdout)" | 4.6: "the last 24 months would start on 2024-10-28 and **exclude Aug-2024**". Proposed 2024-07-01, about 28 months; O1 open. | **REJECTED** | MED |
| A6 | §4: M3 compares V̂_{t,21} with (VIX/100)²·**30/365** | 4.4: G = (252/21)·Σσ² against (V/100)², i.e. V̂ against (V/100)²·21/252. The `vix_level` track uses G_t(h) against V_{t+h}. | REFUTED (scale differs by 365·21/(30·252) = **1.0139**; the wrong window for `vix_level`) | LOW |
| A7 | §4.1(b): "The choice between median and mean must be stated for each model" | 4.3: plugin forecasts **always** report the lognormal mean exp(μ+s²/2) | REFUTED (stale: the ADR has decided) | LOW |

### Claim 7: copyright

| ID | Check | Result | Gate |
|---|---|---|---|
| C1 | No quote over 15 words | 7 quoted spans with lengths 2, 3, 5, 1, 1, 1, 7 words; only one is from a source | SURVIVES |
| C2 | At most one quote per source | Whaley (2013) only | SURVIVES |
| C3 | Quotes verbatim | "virtually guaranteed to lose money over time": verbatim in the archived publisher abstract (A) and the OpenAlex abstract (B) | SURVIVES |
| C4 | No close paraphrase | Longest shared word n-gram with any abstract is 8 tokens, an index name ("the S&P 500 VIX Short-Term Futures"); otherwise ≤ 6. **Falsifier:** an injected 12-gram is detected as 12. Same-structure restatements of single factual clauses: Lo 2002 (65%) and FMS ("hard to beat … persistent"). | SURVIVES (LOW notes) |

### Structure (Tier 1)

- **T1 citations: SURVIVES.** All 50 references are cited in the text. Every author–year citation maps to an entry; script hits that were dates are excluded.
- **T2 links: SURVIVES.** `01-forecast-targets-and-annualisation.md` exists.
- **T3 reuse: SURVIVES.** `~/dev/sharpe-observatory/scripts/metrics.py:97, 122, 147` are `probabilistic_sharpe_ratio`, `expected_max_sharpe` and `deflated_sharpe_ratio`. LICENSE is MIT. The code's PSR matches the doc's formula.

## Found elsewhere (not asked)

1. **statsmodels' DM test uses Bartlett weights** (S4). Anyone following §5.2 will call it with `lags=h−1` and get the estimator ADR 4.8 rejected, with measured size 0.09–0.12. M5 must implement the rectangular estimator. **Measured (ADDED):** `OLS(d, ones).fit(cov_type="HAC", cov_kwds={"maxlags": h-1, "kernel": "uniform"})` exists in 0.15.0 (`sandwich_covariance.py:569`). On an MA(20) toy at n=583 it gives t = 2.026794, equal to the rectangular estimate by hand (2.026794). The default gives 2.388589, equal to Bartlett by hand. HLN's factor must then be applied by hand.
2. **SVXY also split during the backtest sample.** yfinance gives SVXY 1-for-4 on **2018-09-18** and 2-for-1 on **2024-04-11**. §6's split pitfall names VXX only. This is B only: Channel A did not check it against ProShares.
3. **Cboe's own methodology (2026a) has the wrong VIX-futures launch date.** It says 24 Mar; CFE's trade data starts 26 Mar 2004.
4. **KST (2008) is internally inconsistent.** "18% (15/72)" is 20.8%. Anyone recomputing from the fractions gets 21% and 29%, not 18% and 28%.
5. **Cboe `VIX3M_History.csv` starts 09/18/2009** (measured 2026-10-28), not 2007. This bears on ADR 4.8: `term_structure` regimes are null before VIX3M's first date. That makes the null period most of the dev years 2000–2009, not a short gap, if the Cboe CSV is the source.
6. **The FMS RePEc handle now serves a revised paper.** Its findings differ from both the 2007 WP and the 2014 article (term spread "no long-run impact").

## Not validated

- The full texts of the paywalled papers (Bandi & Perron, Diebold 2015, Bekaert & Hoerova). Abstracts and working papers were used.
- Simon & Campasano: B was unavailable, because the OpenAlex record is corrupt.
- Whaley 1993's content.
- The S&P DJI launch date: A only.
- VXX splits after Feb 2026.

Channel A's sources for GJR, HLN and Corsi are publisher PDFs on course servers, not author-hosted copies.

## Single most important fix

**§5.2 (DM).** Replace "errors are MA(h−1), so HAC with at least h−1 lags" and the "pin the lags" advice with the ADR's position:
- the rectangular estimator truncated at h−1, plus HLN, with the Bartlett fallback;
- the vanishing-autocorrelation premise stated as an assumption, with O6's 0.1258 size;
- the fact that `statsmodels.tsa.stattools.diebold_mariano_test` is Newey-West (Bartlett) at any `lags`, so it cannot implement the ADR as-is.

## Delete

**The "Bloomberg via Investing.com (2019)" reference.** Its byline is Seeking Alpha, it is a secondary news item, and the fact it supports (Series A matured 30 Jan 2019) is in Barclays' pricing supplements on EDGAR. Replace it with the Barclays document and drop the `[secondary]` tag.
