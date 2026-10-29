# 15. PSR, DSR and counting trials honestly

*Milestone 6. Code: `src/ivf/backtest/sharpe.py` (copied from sharpe-observatory, MIT),
`src/ivf/backtest/metrics.py`, `src/ivf/backtest/strategy.py`. Walkthrough:
[milestone 6](../milestones/06-backtest.md).*

## A Sharpe ratio is an estimate

The Sharpe ratio of a backtest is `mean / standard deviation` of its daily returns (times √252 to
annualise; cash earns 0 here, so the returns are already excess returns). It comes from one finite
sample, so it has sampling error. Two things make that error larger than the textbook formula says:

- **Short samples.** The usable VXX/SVXY window is about 6.3 years on dev and 2.3 on the holdout.
- **Non-normal returns.** Short volatility earns small gains most days and loses a lot on a few
  (negative skew, fat tails). Those few days decide the Sharpe ratio.

## PSR: how sure are we that the true Sharpe is above a bar?

The **Probabilistic Sharpe Ratio** (Bailey & López de Prado 2012) is the probability that the true
per-day Sharpe ratio exceeds a benchmark `SR*`, given the estimate `SR`, the number of days `T`, the
skewness `g3` and the Pearson kurtosis `g4` (3 for a normal distribution):

```
PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )
```

The denominator is the standard error of `SR`, made larger by negative skew and fat tails. With
`g3 = 0` and `g4 = 3` it is `sqrt(1 + SR^2/2)`, the normal-returns case. Milestone 6 reports `psr0`,
PSR against `SR* = 0`: "is this better than holding cash?"

Use the per-day `SR` in this formula, never the annualised one (`sharpe.py` docstring).

## DSR: the bar rises with the number of tries

Try N rules that all have no edge, keep the best, and its Sharpe ratio will look good by luck alone.
The expected best of N such estimates is about (Bailey & López de Prado 2014, eq. 2; the False
Strategy Theorem of Bailey, Borwein, López de Prado & Zhu 2014):

```
E[max SR] = sqrt(V) * ( (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N e)) )
```

with `V` the variance of the trials' Sharpe estimates, `gamma ≈ 0.5772` (Euler-Mascheroni) and `Z^-1`
the normal quantile. The **Deflated Sharpe Ratio** is PSR with that as the bar:
`DSR = PSR(E[max SR])`. A DSR near 1 says the result is unlikely to be the luckiest of N tries; near 0
says it is what luck alone would give.

Checked against the authors' PDF of Bailey & López de Prado (2014) on 2026-10-29: eq. (2), V is the
variance **across** the trials' estimated SRs, N the number of **independent** trials, `g3` and `g4`
of the **selected** strategy, and their Snippet 1 computes `E[max SR]` as the code does. The
typeset eq. (2) shows `sqrt(T - 1)` and `(g4 - 1)/4` with Pearson kurtosis (checked by M6 review A;
the first check's text extraction had lost those glyphs).

## Two choices made here (defaults taken 2026-10-29, owner can override)

1. **V is the larger of the cross-trial variance and `1/T`.** Our trials are band variants of one
   rule, so their Sharpe ratios are close together and the cross-trial variance is tiny, which would
   hardly deflate anything. `1/T` is the variance of a per-day SR estimate when the true SR is 0 and
   returns are normal. Taking the larger is the conservative side (sharpe-observatory's README makes
   the same choice). The paper's version is reported too, as `dsr_paper_var`.
2. **N is the raw count of trials, not the paper's "implied independent trials".** The paper's
   Appendix 3 (eq. 9) shrinks N when trials are correlated. Using the raw count overstates
   `E[max SR]`, which the paper itself notes: again the conservative side.

## Counting trials

A trial is any rule run on dev and looked at. Re-running the same rule is not a new trial; changing
anything that could have been chosen by looking (a band, a model, a signal definition) is.

- Every pre-registered forecast strategy in `configs/backtest.yaml` is a trial: four band values of
  the primary rule (0, 1, 2, 4 vol points) and the VIX-level rule. **N = 5** after the first real run.
- Each run appends one line per (strategy, cost level) to the append-only trial log under the data
  root, keyed by the strategy id and its parameters, the chosen model included. `n_trials` counts
  distinct keys, so a re-run adds nothing and a new band or a different model adds one.
- A rule is never deleted. It is retired (`retired: true`) and stays counted in the log.
- Benchmarks (always long VXX, always long SVXY, cash) are not trials: nothing is chosen among them.
- Cost levels are not trials: they change the cost, not the positions, and the default cost was
  fixed in advance. Each cost level gets its own DSR, computed with that level's Sharpe ratios.
- The holdout adds no trials. It reports the primary's PSR and the DSR with dev's N.

What the log cannot see: a rule someone tried in a notebook and never ran through the command. The
discipline is to run every variant through `python -m ivf.backtest run`, and to say so in the
milestone record.

## Reading the numbers

- A PSR of 0.95 is the usual bar. Below it, the sample cannot tell the strategy from cash. That is
  low power, not proof that there is no edge.
- With N = 5 the deflation is mild. It grows quickly at first and then slowly: `E[max SR]` grows
  roughly like `sqrt(2 ln N)`.
- PR/FAQ hypothesis H4 says that after costs no forecast-driven rule will have a DSR distinguishable
  from zero. The dev run reports it locally; the holdout look settles it.
