# 14. VIX ETPs: roll yield, leverage breaks and tradable segments

*Milestone 6. Code: `src/ivf/backtest/segments.py`, `configs/instruments.yaml`. Walkthrough:
[milestone 6](../milestones/06-backtest.md). Sources: the references of
[00-background](00-background.md), section 6, unless stated.*

## You cannot buy the VIX

The VIX is a number computed from S&P 500 option prices. Nothing pays its daily change. What you can
trade are products built on **VIX futures**:

- **VXX** (iPath Series B S&P 500 VIX Short-Term Futures ETN) tracks the S&P 500 VIX Short-Term
  Futures Index. Each day the index sells a little of the first-month future and buys the second
  month, so that it always holds about one month to expiry (ProShares n.d.).
- **SVXY** (ProShares Short VIX Short-Term Futures ETF) aims at a fixed multiple of that index's
  **daily** return. The multiple has been **-0.5x** since the close of 2018-02-27; before that it was
  -1x (ProShares 2018a, the issuer's press release of 2018-02-26, read on 2026-10-29).

So a forecast of the VIX is not a forecast of these prices. Two things sit between them.

## Roll yield

Futures for later months usually cost more than nearer ones (contango). The index keeps selling the
cheaper near future and buying the dearer next one. If the curve does not move, the future it holds
drifts down toward the spot VIX as it ages, so a long position loses roughly the slope of the curve
per month (background note, section 6, "my approximation"). In backwardation (after a spike) the
sign flips. That is why Whaley (2013) calls the long products "virtually guaranteed to lose money
over time", and why a short-volatility product earns that drift most of the time and gives it back,
and more, in a spike.

What this means for the rule. A forecast that the VIX will rise is only worth trading if the rise is
larger than what the futures curve already prices. The futures basis predicts futures returns, not
spot VIX changes (Simon & Campasano 2014). Milestone 6's primary rule therefore compares a
**realised** volatility forecast with the VIX (the variance risk premium), not a VIX forecast with
today's VIX. The VIX-level version is kept as one more trial.

## Breaks in the price history

A ticker's history can join together things that are not one product:

| Break | What happened | How it is handled |
|---|---|---|
| SVXY -1x to -0.5x | The daily target was halved at the close of 2018-02-27 | Two segments; only -0.5x is usable |
| VXX series | Series A matured 2019-01-30; Series B (VXXB) took the VXX ticker on 2019-05-02 (Barclays 2026, MIAX 2019) | Yahoo's VXX starts 2018-01-25 and is Series B throughout; the fetched closes show no jump on either date |
| Reverse splits | VXX 1:4 on 2021-04-23, 2023-03-07, 2024-07-24; SVXY 1:4 on 2018-09-18, 2:1 on 2024-04-11 | The closes are split-adjusted; checked on each date |
| Daily reset | -0.5x of each day's return is not -0.5x of a month's | Returns are always daily |

## Tradable segments

A **segment** is a span of closes with one stable exposure (`valid_from` to `valid_to`, the first and
last close). A daily return uses two closes, so it belongs to a segment only when **both** are inside.
Two segments can share their boundary close, and then every return belongs to exactly one of them:

```
close 2018-02-26 --(return dated 02-27)--> close 2018-02-27 --(return dated 02-28)--> close 2018-02-28
        -1x segment ends at this close  ^^^^^^^^^^^^^^^^^  -0.5x segment starts at this close
```

The backtest runs only on returns inside a **usable** segment of both VXX and SVXY, and the window
must be unbroken. SVXY's -0.5x segment starts at the close of 2018-02-27; VXX's usable segment starts
at 2018-08-01, the first close after its last **stale print** (below), so the dev window is the
returns dated **2018-08-02** to 2024-06-28, the last session before the holdout.

## Stale prints

A new, thinly traded product can print a "bar" with no trade in it: open, high, low and close all
equal, or a volume of 0. Its close is then a carried-forward quote, and the next day's return
contains two days of moves. Yahoo's VXX (then the young Series B, ticker VXXB) has a cluster of such
days in its first half-year (found by M6 review A; counts and dates in the milestone record). The rule
(added after the first dev run, default taken 2026-10-29, owner can override): **a usable segment
holds no stale print** on dev data, so VXX's starts after the last one. SVXY had none.

## Checking the dates against the data

A press release says what was intended; the prices say what happened. `check_segments` reads the
fetched closes (before `holdout_start` on the dev path) and fails the run when:

1. a close is missing inside a usable segment;
2. on a listed split date the adjusted close moves by half the split's size or more in log terms (an
   unadjusted 1:4 split would move by ln 4 ≈ 1.39);
3. on a usable segment, the OLS slope of SVXY's daily return on VXX's is more than 0.15 from
   -0.5 / +1.
4. on dev sessions, a usable segment holds a stale print.

The slope check is weak evidence of the exact break date: one wrong day in a thousand barely moves
a slope. The issuer's statement is the evidence for the date; the slope shows that the usable
segment really is -0.5x. On the unusable -1x segment the slope is only reported: its short overlap
with VXX includes the February 2018 collapse, when SVXY's close-to-close moves stopped tracking -1x
VXX's (the measured value stays local). The open end of the -0.5x segment rests on the issuer's
fund page (ProShares n.d.), not on holdout data.

## What can still go wrong

- **Yahoo's adjusted closes are a vintage.** A restatement would move them; the raw snapshot and its
  hash are kept, as for every source (ADR 0001 D3).
- **Series B's inception date** was not verified against the issuer. Yahoo starts at 2018-01-25,
  which is before the usable window, so it does not change the window.
- **Fees and tracking error** are inside the product's price, so they are in the returns. VXX's
  tracking of its index, and SVXY's of -0.5x, are not exact (the measured slope is close to, not at,
  -0.5).
