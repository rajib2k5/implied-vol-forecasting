# 16. Scale equivariance, metamorphic tests and what we may publish

*Milestone 7 (polish). Code: `tools/licensing/p_tie_lstm.py`, `tools/licensing/check_raw_files.py`,
`src/ivf/app_results.py`. Ledger: `docs/validation/O10-lstm-equivariance.md`. Walkthrough:
[milestone 7](../milestones/07-polish.md). Earlier: [03, reproducibility and data contracts](03-reproducibility-and-data-contracts.md)
(the publication set P) and [05, persistence](05-persistence-half-life-and-unit-roots.md) (the half-life
under a rescaling).*

## The question

The VIX is a vendor's series, and we may not publish it or any material part of it (ADR 0001 D2). We may
publish results that are made *from* it, as long as they do not let a reader rebuild the series. The
rule we use for a number is: **it must not change if the whole VIX path V is replaced by c·V**, for any
constant c > 0, with the S&P 500 returns held fixed (they are public). A number that does not move cannot
contain the scale of V, so it cannot tell a reader whether V was 15 or 30.

## Why a ratio to a benchmark does not move

Let a forecaster be **scale-equivariant** in V: if the whole VIX path is multiplied by c, its forecast of
a VIX level is multiplied by c as well. The target is the VIX itself, so it scales by c too. Each error
`e_t = y_t - ŷ_t` becomes `c·e_t`. Then:

- RMSE and MAE become `c` times larger. They are in vol points, and a vol point scales with V.
- The random walk `ŷ_t = V_t` is equivariant, so its RMSE scales by `c` too.
- The **ratio** RMSE(model) / RMSE(rw) is `(c·a) / (c·b) = a / b`: unchanged.
- MAPE uses `|e| / y`, so `c` cancels: unchanged. QLIKE uses `y / ŷ`, which is unchanged when both scale.

A made-up check (four closes, three one-step forecasts), computed by hand and by a short script:

| c | RMSE of `rw` | RMSE of the model | ratio | MAPE |
|---|---:|---:|---:|---:|
| 1 | 2.1602 | 1.3229 | 0.6124 | 0.0509 |
| 1.5 | 3.2404 | 1.9843 | 0.6124 | 0.0509 |
| 0.5 | 1.0801 | 0.6614 | 0.6124 | 0.0509 |

(V = 20, 22, 21, 24; model forecasts 21, 21.5, 22 for the last three. `rw` errors are 2, -1, 3, so
`rw`'s RMSE is `sqrt(14/3)` = 2.1602.) The levels move with `c`. The ratio and the MAPE do not.

A forecaster of a target that does **not** scale (the realised volatility `rv21` is built from S&P 500
returns only) must be **invariant** rather than equivariant: its forecast must not change at all when V is
rescaled, even though V is one of its inputs. A model that feeds `ln V` through a scaler fitted on the
training window is invariant, because `ln(cV) = ln c + ln V` and the scaler subtracts the training mean,
which absorbs `ln c`.

## The metamorphic relation as a test oracle

We cannot check a neural network against a known right answer. But we know a relation between two
runs. A **metamorphic test** runs the code on an input and on a transformed input and checks that the
outputs relate as the theory says: here, V to cV, and the evaluation entries must be identical.

`tools/licensing/p_tie_lstm.py` does this for the real `LstmLog` and `LstmRv`. It uses simulated paths only
and compares 16 entries (RMSE ratio, QLIKE ratio, MAPE ratio and MAPE, on four panels) at 6 significant
figures. On 2026-11-09 it printed 0 of 16 entries changed for c = 0.8 and for c = 1.25, with a largest
difference of 4.4e-16, which is float rounding.

Two limits of this oracle:

1. It is **necessary, not sufficient**. It tests a few values of c, one simulated seed and a cheap
   configuration. It shows that no *scale* leaks through the model. It does not show that nothing
   about the series leaks (concept 03 lists the residual: the ratios between days).
2. An oracle that cannot say "no" proves nothing. So the probe has falsifiers.

## Falsifiers: break it on purpose

A test is only evidence if it fails when the property is false. The probe patches the features to add one
input that depends on the level of V:

| Falsifier | Added input | Why it breaks equivariance | Changed at 6 s.f. (c = 0.8, 1.25) | Changed at 3 s.f. (c = 0.8, 1.25) |
|---|---|---|---:|---:|
| `raw_level` | V / 100, not z-scored | a level the scaler does not remove: the network can read `c` from it | 16 of 16 at both | 2 and 1 of 16 |
| `flag20` | 1 if V > 20 | a threshold in vol points: the flag flips for some days when V moves to 0.8·V | 16 of 16 at both | 15 and 12 of 16 |

(Printed by the probe on 2026-11-09.) The 3 s.f. column is the precision at which results are published:
the `raw_level` leak is small enough that most entries still agree at 3 s.f. That is why the probe judges
at 6 s.f. and not at the published precision. A check at the published precision would have let that
falsifier through.

## Where equivariance breaks

Any of these in a model's inputs or outputs ties the number to the scale of V:

- **Absolute losses.** RMSE, MAE and bias in vol points scale by `c`. Only their ratios to `rw` are
  candidates for P.
- **A threshold.** A flag such as `V > 20`, or a regime cut (the VIX-level regimes are cut at fixed
  vol-point edges), changes which days fall on which side when V is rescaled. This is also why regime
  cells and ribbons are kept out of any real file (milestone 7, decision P3).
- **A clip.** A floor or cap in absolute units (a minimum variance in vol points, say) binds on different
  days after a rescaling.
- **A level input that is not standardised**, or standardised on a window that includes the future.
- **`iv_rv_spread`** (`src/ivf/features/build.py:105`): `(V/100)^2 - (rv21/100)^2`. Under V to cV the first
  term scales by `c^2` and the second, built from S&P 500 returns, does not. The spread has no
  fixed relation to its old value, so a model that takes it as an input is not equivariant.
- **Scale-tied rows.** The uncalibrated GARCH family on the VIX level and `vix_implied` on `rv21`
  compare a quantity with a vendor level; ADR 0001 keeps them local (O8).

## The link to ADR 0001 D2 and O10

D2 admits a model's rows to the publication set P only if every number in the row is unchanged under V to
cV (concept 03 states the rule, and the measurement of the P entries). O10 asked whether `lstm_log` is
such a model. M4 left it open because its inputs (ln V and its averages; returns, ln V, ln rv5 and ln rv21)
look equivariant but nobody had run it. On 2026-11-09 the probe above was run on the real code, the
falsifiers fired, and O10 was decided: `lstm_log` is equivariant, so its unitless rows qualify for P under
the same rounding as the other models.

Not covered by that decision (from the ledger): the production configuration, the frozen-fit path
(`refit_every: 252`), DM and MZ entries, and more than one seed. The argument that they hold
is the same as above (the network sees `ln V`, the scaler removes `ln c`), but it was not measured.

The same discipline runs through the publication work:

- Real regime cells and ribbons never enter a payload (`regimes_allowed`), because the regimes are
  thresholds on V.
- `tools/licensing/check_raw_files.py` stops vendor *files* entering git, by comparing the SHA-256 of
  every tracked file (at HEAD) and every staged file (the index blob) with the hashes of the raw snapshots.
  It matches content, never names, so a renamed copy is caught. A modified copy is not, and the text
  scanner `scan_real_rows.py` is the other half.

## Reading the numbers

- "Equivariant" is a property of code and inputs, shown by a relation plus falsifiers. It is not a proof of
  privacy: D2 says plainly that it argues, and does not prove, that P reveals no material part.
- A 0 of 16 is meaningful only next to the falsifiers' 16 of 16. Report both.
- When you add a feature to a model that is in P, add the same probe for it first.
