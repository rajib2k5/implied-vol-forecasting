# M3 review A: GARCH done right

Adversarial review of the eight claims of the first M3 report, 2026-10-29. The reviewer worked in a
scratch copy of the repo (`<session scratchpad>/m3-review-a/repo`) with its own probes
(`<session scratchpad>/m3-review-a/probe1.py`, `probe2.py`) and mutants (`mutate.py`). Synthetic data
only; every number below is publishable. The fixes are in
[the M3 walkthrough](../milestones/03-garch-done-right.md#what-review-a-found).

## The reviewer's pre-registered ledger (verbatim)

```
# Pre-registered ledger (mode: single-context sequential fallback). Every claim starts REJECTED.
C1 poison after cutoff -> 6 fits + forecast at origin c identical. Probe: run_rv21 on synthetic, garbage (x1e3 and NaN) after c; compare fit params_json and yhat at origin c. Falsifier: mutant fit(inputs.upto(c+21)).
C2 har_rv tau<=c-21 binds; features of pair tau use data<=tau. Probe: refit with training_origins(n)+1 -> params differ; design row tau from r[:tau+1] == from full r. Falsifier: lstsq with tau extended.
C3 G == E_t[v] in rv21 units; const 1%^2 -> 15.874508; ŷ=100 sqrt(G) scored; QLIKE on v_hat. Probe: compare build.rv_trailing vs process.py rv21; aggregate(ones)->; read metric QLIKE code.
C4 filter_variance == arch variances & one-step to 1e-9 (3x2); GARCH/GJR path == arch analytic. Probe: fit synthetic, compare. Docs: arch source recursions, egarch sqrt(2/pi).
C5 EGARCH product == brute force; t-MGF diverges. Probe: enumerate 3-point z for k<=4; numeric integral of t-density*exp(s|z|) to L grows unbounded; mean(z^2) of residuals vs 1.
C6 truncation: forecast at t identical with inputs.upto(t) vs upto(last), 6 models + arima_garch_log.
C7 holdout x100 changes nothing (run test + mutant pre_holdout_inputs '<' -> '<=').
C8 synthetic QLIKE numbers; arima_garch d=1 vs d=0 AR(1)+const GARCH on 100 ln VIX at h=1,5,21 RMSE.
X  AIC param counts (arch num_params), same-rows expected list, docs claims, pytest count, 3 new mutants.
```

## Probe output (probe1.py, re-run by the builder against the reviewed copy; everything it printed)

```
n folds 97 n sessions 8689
C1 fold 0 poison=huge: fits identical=True n_models=6 forecast@c identical=True
C1 fold 0 poison=nan: fits identical=True n_models=6 forecast@c identical=True
C1 fold 48 poison=huge: fits identical=True n_models=6 forecast@c identical=True
C1 fold 48 poison=nan: fits identical=True n_models=6 forecast@c identical=True
C2 train_last_origin == c-21: True | last target == c: True
C2 adding tau=c-20 changes beta: max rel diff 0.0004879028180658773
C2 design rows tau whose value changes when data after tau is removed: 0
C6 rv21 truncation: 24 model-origins compared, max rel diff 0.00e+00
C6 arima_garch_log truncation max rel diff 0.00e+00
C4 garch  normal in-sample max rel 7.8e-16 one-step rel 2.2e-16 21-step max rel 6.7e-16 nu=nan
C4 garch  t      in-sample max rel 6.7e-16 one-step rel 1.1e-16 21-step max rel 1.1e-15 nu=6.43
C4 gjr    normal in-sample max rel 6.7e-16 one-step rel 1.1e-16 21-step max rel 8.9e-16 nu=nan
C4 gjr    t      in-sample max rel 7.8e-16 one-step rel 1.1e-16 21-step max rel 7.8e-16 nu=6.50
C4 egarch normal in-sample max rel 1.3e-15 one-step rel 0.0e+00 nu=nan
C5 egarch normal: mean(z^2) of training std resid = 1.0007
C4 egarch t      in-sample max rel 8.9e-16 one-step rel 0.0e+00 nu=6.26
C5 egarch t: mean(z^2) of training std resid = 0.9896
C5 product vs brute-force max rel diff 3.3306690738754696e-15
C5 t(6) partial E[exp(0.1|z|)] over [0,10] = 1.08
C5 t(6) partial E[exp(0.1|z|)] over [0,50] = 1.08
C5 t(6) partial E[exp(0.1|z|)] over [0,100] = 1.08
C5 t(6) partial E[exp(0.1|z|)] over [0,200] = 1.08
C5 t(6) partial E[exp(0.1|z|)] over [0,400] = 211.5
```

## The reviewer's extra mutants (mutants.log, verbatim)

```
SURVIVED rc=0 | EGARCH M from raw (unstandardised) residuals | ['46 passed in 7.18s']
SURVIVED rc=0 | ARIMA-GARCH AR residual misaligned by one (p>0 silently refused) | ['46 passed in 6.89s']
SURVIVED rc=0 | forecastable needs 21 returns not 22 | ['46 passed in 7.10s']
KILLED rc=1 | choose by loglik (no parameter penalty) | ['FAILED tests/test_garch.py::test_choose_keeps_the_lowest_aic_among_usable_fits_and_a_tie_keeps_normal', '1 failed, 15 passed in 2.20s']
```

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| 1. Poisoning after the cutoff changes no fit and no forecast at c | not refuted | C1, two folds, huge and NaN poison |
| 2. `har_rv` uses τ ≤ c − 21 and the restriction binds | not refuted | C2 |
| 3. `G = E_t[v]` in rv21 units | not refuted | no refutation reported |
| 4. Our recursion equals `arch`'s (1e-9) | not refuted | C4: ≤ 1.3e-15 |
| 5. EGARCH product formula; Student-t MGF diverges, "a simulated mean is noise" | **partly refuted** | The product formula holds (3.3e-15). The divergence needs \|z\| of a few hundred (1.08 up to 200, 211.5 up to 400), so a simulated mean is a stable truncated value, not noise. The training residuals had mean(z²) = 0.9896, not 1, and were not rescaled |
| 6. Forecasts at t use data ≤ t, every rv21 model and `arima_garch_log` | not refuted | C6 |
| 7. The holdout is unread | not refuted | no refutation reported |
| 8. The synthetic numbers, and the `arima_garch_log` design | **design refuted** | d = 1 is worse than the random walk at every horizon; d = 0 is better (re-measurement below) |
| Mutants | **3 silent survivors** | the first three above |

## Coordinator's re-measurement (probe2.py, synthetic, RMSE at h = 1/5/21)

Shipped d = 1 mean 1.1871/2.6158/5.0857; rw 1.1857/2.5992/4.9454; d = 0 (AR(p) + constant + GARCH(1,1)
on 100·ln VIX) 1.1798/2.5373/4.4975. Same mistake as M2's `arima_log`.
