"""Point-forecast losses (ADR 0001 4.3), exactly as the ADR's loss table defines them.

``y`` is the observed target and ``yhat`` the forecast, both in vol points and both > 0.
QLIKE works in variance space: ``v = (y/100)**2`` and ``v_hat = (yhat/100)**2``.

Which forecast each loss rewards (Gneiting 2011; ADR 0001 4.3). If ``ln y`` is normal with mean
mu and variance s2 given the information at the origin, the forecast that minimises the expected
loss is:

==========  ===========================  =========================================
loss        minimised by                 for a lognormal y
==========  ===========================  =========================================
MAPE        median of order -1           ``exp(mu - s2)``
MAE         median                       ``exp(mu)``: the naive back-transform
RMSE        mean of y                    ``exp(mu + s2/2)``: the corrected one
QLIKE       mean of v = E[y**2]/10**4    ``exp(mu + s2)``, as a vol (``100*sqrt(E v)``)
==========  ===========================  =========================================

So the four losses, in that order, reward ever higher forecasts. The project evaluates the
conditional mean (``yhat``, the lognormal correction) and treats RMSE as primary for it; MAE and
MAPE are reported because the brief asks and are flagged as not consistent for the mean; QLIKE
of a plug-in ``(yhat/100)**2`` is biased low by ``Var_t(y)/10**4`` (ADR 0001 4.3).
"""

from __future__ import annotations

import numpy as np


def _pair(y, yhat) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("y and yhat must be non-empty 1-d arrays of the same length")
    if not (np.isfinite(y).all() and np.isfinite(yhat).all()):
        raise ValueError("a loss is computed on scored rows only: no NaN or inf")
    return y, yhat


def bias(y, yhat) -> float:
    """mean(y - yhat): positive means the model under-forecasts. A local diagnostic only (D2)."""
    y, yhat = _pair(y, yhat)
    return float(np.mean(y - yhat))


def rmse(y, yhat) -> float:
    """sqrt(mean((y - yhat)**2)), vol points. Consistent for the mean of y."""
    y, yhat = _pair(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    """mean(|y - yhat|), vol points. Consistent for the median, not the mean."""
    y, yhat = _pair(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat) -> float:
    """mean(|y - yhat| / y), a fraction. Needs y > 0; rewards forecasts below the median."""
    y, yhat = _pair(y, yhat)
    if (y <= 0).any():
        raise ValueError("MAPE needs y > 0")
    return float(np.mean(np.abs(y - yhat) / y))


def qlike(y, yhat) -> float:
    """mean(v/v_hat - ln(v/v_hat) - 1) with v = (y/100)**2 (Patton 2011, eq. 24 at b = -2).

    >= 0, and 0 only when every v_hat equals v. Needs v, v_hat > 0.
    """
    y, yhat = _pair(y, yhat)
    if (y <= 0).any() or (yhat <= 0).any():
        raise ValueError("QLIKE needs y > 0 and yhat > 0")
    ratio = (y / 100) ** 2 / (yhat / 100) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


LOSSES = {"bias": bias, "rmse": rmse, "mae": mae, "mape": mape, "qlike": qlike}


def loss_row(y, yhat, min_cell_n: int) -> dict:
    """``n``, ``suppressed`` and every loss; the losses are None when ``n < min_cell_n`` (ADR 0001 4.8)."""
    n = len(y)
    if n < min_cell_n:
        return {"n": n, "suppressed": True, **{name: None for name in LOSSES}}
    return {"n": n, "suppressed": False, **{name: fn(y, yhat) for name, fn in LOSSES.items()}}
