"""Regime breakdown of the losses (ADR 0001 4.8, rule L5; milestone 5).

The regimes are M1's ``regime_label`` table (``ivf.features.build.build_regime_labels``), labelled
**at the forecast origin** from data dated at the origin only, with thresholds fixed in
``configs/protocol.yaml`` before any model ran (no sample quantiles):

* ``vix_level``: ``low`` V < 15, ``mid`` 15 <= V < 25, ``high`` V >= 25, V the VIX close at the origin;
* ``term_structure``: ``backwardation`` if VIX > VIX3M at the origin, else ``contango``; no label
  before VIX3M's first close (2009-09-18 on the real file), so those origins are in no
  ``term_structure`` cell.

So a regime is known when the forecast is made: it can condition a trading rule, and splitting the
losses by it uses no information from after the origin. Regimes from different schemes overlap;
they are slices, not a partition. A cell with ``n < min_cell_n`` is written with its ``n`` and null
losses (``suppressed``). Regime cells never leave the machine for real data (ADR 0001 D2, O7).
"""

from __future__ import annotations

import pandas as pd

from ivf.evaluation import losses

SCHEMES = ("vix_level", "term_structure")


def attach(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add ``regime_<scheme>`` columns: the label at each row's ``origin_date`` (None where there is none)."""
    out = frame.copy()
    for scheme in SCHEMES:
        mapping = labels[labels.scheme == scheme].set_index("date")["regime"]
        out[f"regime_{scheme}"] = out["origin_date"].map(mapping)
        out[f"regime_{scheme}"] = out[f"regime_{scheme}"].astype(object).where(out[f"regime_{scheme}"].notna(), None)
    return out


def variants(group: pd.DataFrame) -> dict[str, str]:
    """The forecasts scored for one model: ``mean`` (yhat), ``median`` (yhat_naive, log-space models) and
    ``mean_cal`` (the calibrated variant, ``yhat_cal``), each only if present on every row."""
    out = {"mean": "yhat"}
    for name, column in (("median", "yhat_naive"), ("mean_cal", "yhat_cal")):
        if column in group and group[column].notna().all():
            out[name] = column
    return out


def metric_rows(scored: pd.DataFrame, period: str, min_cell_n: int) -> pd.DataFrame:
    """The ``metric`` table: one row per (model, target, horizon, scheme, regime, forecast).

    ``scheme = all`` is the unconditional row; then one row per label of each scheme in ``SCHEMES``.
    """
    rows = []
    for (model_id, target_id, h), g in scored.groupby(["model_id", "target_id", "horizon"], sort=True):
        cells = [("all", "all", g)]
        for scheme in SCHEMES:
            column = f"regime_{scheme}"
            if column in g:
                labelled = g[g[column].notna()]
                cells += [(scheme, label, cell) for label, cell in labelled.groupby(column, sort=True)]
        for name, column in variants(g).items():
            for scheme, label, cell in cells:
                rows.append({
                    "model_id": model_id, "target_id": target_id, "horizon": int(h), "period": period,
                    "scheme": scheme, "regime": label, "forecast": name,
                    **losses.loss_row(cell["y"].to_numpy(), cell[column].to_numpy(), min_cell_n),
                })
    return pd.DataFrame(rows)
