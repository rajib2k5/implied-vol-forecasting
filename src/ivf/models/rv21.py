"""The ``rv21`` track of milestone 3: what the GARCH family is compared with (ADR 0001 4.2, 4.4).

**The target.** ``rv21`` at date d is ``rv_trailing(r)[d] = 100 sqrt((252/21) sum_{j=0}^{20} r_{d-j}**2)``,
r the decimal ``^GSPC`` log return, no mean removed. It is a **window** value: it is observed at d,
the window's last session, and it is stored on that row (``target_series``). A forecast from origin t
has horizon 21 and is scored against the value at ``t + 21``, whose window is exactly the 21
sessions ``t+1..t+21``.

**Rule L2 for a window target (O-M2-7, settled in M3).** M2's L2 held "because each value sits on
the row of the date it is observed". A window target breaks that if it is stored on its *origin's*
row, as M1's ``forward_rv21`` is: the training pair at origin ``tau = c`` would then carry the whole
window ``c+1..c+21`` that the fold's first forecast must predict. Here:

* every input is a **daily** series observed on its own row (returns, VIX), and the runner gives a
  fit the inputs up to the cutoff c and nothing later (``Rv21Inputs.upto``);
* a supervised model may pair origin tau with its target only if the target's window ends at or
  before c: ``tau <= c - 21`` (``training_origins``). The model records its last training origin
  (``Fit.train_last_origin``) and the runner asserts ``train_last_origin + 21 <= cutoff``;
* features at origin tau use returns up to tau only (trailing windows, rule L1).

**The models.** ``rw`` (``yhat = rv21`` of the window ending at t, ADR 0001 D2's benchmark),
``har_rv`` (HAR on realised variance), ``vix_implied`` (``v_hat = (V_t/100)**2``: VIX as a forecast of
realised variance), and the GARCH family (``ivf.models.garch.GarchRv``). Every rv21 model has the
interface ``fit(inputs, previous) -> Fit`` and ``forecast(fit, inputs, origins) -> Forecast`` at the
single horizon 21.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ivf.features import build
from ivf.models.baselines import Fit, Forecast

#: The one rv21 horizon (ADR 0001 4.2): the target window is exactly the 21 sessions after the origin.
H = build.RV_WINDOW
TARGET_ID = "rv21"
SYMBOL = "^GSPC"


@dataclass(frozen=True)
class Rv21Inputs:
    """Daily series on the master calendar, each value on the row of the date it is observed."""

    r: np.ndarray  # decimal log return of ^GSPC, r[i] = ln(S_i / S_{i-1})
    vix: np.ndarray  # VIX close

    def __post_init__(self):
        if len(self.r) != len(self.vix):
            raise ValueError("r and vix must be aligned on the same sessions")

    def __len__(self) -> int:
        return len(self.r)

    def upto(self, i: int) -> Rv21Inputs:
        """The inputs dated <= session i: what a fit at cutoff i, or a forecast at origin i, may read."""
        return Rv21Inputs(self.r[: i + 1], self.vix[: i + 1])


def target_series(r: np.ndarray) -> np.ndarray:
    """``rv21`` dated at the END of its window (vol points): ``build.rv_trailing``, the target_value table's value."""
    return build.rv_trailing(np.asarray(r, dtype=float))


def training_origins(n_obs: int, horizon: int = H) -> np.ndarray:
    """Origins tau whose target window ends at or before the last session ``c = n_obs - 1``: ``tau <= c - horizon``.

    Worked number: inputs up to c = 100 (n_obs = 101) allow origins 0..79; the pair at tau = 79 has
    the window 80..100, which ends at c. tau = 80 would need the return at 101.
    """
    return np.arange(max(0, n_obs - horizon))


def forecastable(inputs: Rv21Inputs) -> np.ndarray:
    """True where the 22 returns t-21..t (HAR's monthly window) and VIX at t are present: a rule on the data, not the model."""
    return np.isfinite(build.trailing(inputs.r, build.HAR_MONTH, "sum")) & np.isfinite(inputs.vix)


def _native(v_hat: np.ndarray, floored: np.ndarray | None = None) -> Forecast:
    v_hat = np.asarray(v_hat, dtype=float)
    y = 100 * np.sqrt(v_hat)
    return Forecast(yhat=y, yhat_naive=y.copy(), v_hat=v_hat, floored=floored)


class RandomWalkRv:
    """``yhat = rv21`` of the window ending at the origin (ADR 0001 D2: the rv21 benchmark). Nothing to fit."""

    model_id = "rw"
    fit_space = "level"
    var_method = "plugin"

    def fit(self, inputs: Rv21Inputs, previous: Fit | None) -> Fit:
        return Fit(params={}, hyperparams={}, n_train=0, train_first=None, train_last_target=None)

    def forecast(self, fit: Fit, inputs: Rv21Inputs, origins: np.ndarray) -> Forecast:
        y = target_series(inputs.r)[origins]
        return Forecast(yhat=y, yhat_naive=y.copy())


class VixImplied:
    """``v_hat = (V_t/100)**2``: VIX at t as a forecast of the realised variance of the next 21 sessions (ADR 0001 4.4)."""

    model_id = "vix_implied"
    fit_space = "level"
    var_method = "native"

    def fit(self, inputs: Rv21Inputs, previous: Fit | None) -> Fit:
        return Fit(params={}, hyperparams={}, n_train=0, train_first=None, train_last_target=None)

    def forecast(self, fit: Fit, inputs: Rv21Inputs, origins: np.ndarray) -> Forecast:
        return _native((inputs.vix[origins] / 100) ** 2)


def har_rv_design(r: np.ndarray) -> np.ndarray:
    """Rows ``[1, vd_t, vw_t, vm_t]``: annualised decimal variances over 1, 5 and 22 sessions ending at t.

    ``vd = 252 r_t**2``, ``vw = 252 mean(r_{t-4..t}**2)``, ``vm = 252 mean(r_{t-21..t}**2)`` (Corsi 2009's
    clocks with M2's windows). Trailing only (L1); NaN where a window has a gap (G2).
    """
    r2 = np.asarray(r, dtype=float) ** 2
    year = build.SESSIONS_PER_YEAR
    return np.column_stack([np.ones(len(r2)), year * r2, year * build.trailing(r2, build.HAR_WEEK, "mean"), year * build.trailing(r2, build.HAR_MONTH, "mean")])


class HarRv:
    """HAR in variance space, direct at h = 21: ``v_{tau+21} = b0 + b_d vd + b_w vw + b_m vm + e`` by OLS.

    ``v`` is the rv21 target in variance, ``(rv21/100)**2``, so the fit is a forecast of ``E_t[v]``
    (``var_method = native``), the quantity QLIKE ranks. Pairs use ``training_origins`` only (L2 for a
    window target). An OLS forecast can be <= 0; it is raised to the **smallest training target**
    (scale-free: it scales with the data, ADR 0001 D2), and the count is reported (``floored``).
    """

    model_id = "har_rv"
    fit_space = "variance"
    var_method = "native"
    COEFS = ("b0", "b_d", "b_w", "b_m")

    def fit(self, inputs: Rv21Inputs, previous: Fit | None) -> Fit:
        r = inputs.r
        design = har_rv_design(r)
        v = (target_series(r) / 100) ** 2
        tau = training_origins(len(r))
        rows, target = design[tau], v[tau + H]
        keep = np.isfinite(rows).all(axis=1) & np.isfinite(target)
        n = int(keep.sum())
        if n <= len(self.COEFS):
            raise ValueError(f"only {n} complete HAR-RV pairs")
        beta, _, rank, _ = np.linalg.lstsq(rows[keep], target[keep], rcond=None)
        if rank < len(self.COEFS):
            raise ValueError(f"HAR-RV design is rank {rank}")
        used = tau[keep]
        params: dict[str, Any] = {**dict(zip(self.COEFS, map(float, beta))), "floor": float(target[keep].min()), "n": n}
        return Fit(
            params=params,
            hyperparams={"day": 1, "week": build.HAR_WEEK, "month": build.HAR_MONTH, "estimator": "OLS", "space": "variance", "horizon": H},
            n_train=n,
            train_first=int(used[0]) - (build.HAR_MONTH - 1),
            train_last_target=int(used[-1]) + H,
            train_last_origin=int(used[-1]),
        )

    def forecast(self, fit: Fit, inputs: Rv21Inputs, origins: np.ndarray) -> Forecast:
        beta = np.array([fit.params[c] for c in self.COEFS])
        v_hat = har_rv_design(inputs.r)[origins] @ beta
        floored = v_hat < fit.params["floor"]
        return _native(np.where(floored, fit.params["floor"], v_hat), floored)


def build_rv21_models(model_ids=None, dists=None) -> list:
    """The rv21 models in registry order: rw, har_rv, vix_implied, garch, gjr, egarch."""
    from ivf.models import garch

    kw = {} if dists is None else {"dists": tuple(dists)}
    every = {m.model_id: m for m in (RandomWalkRv(), HarRv(), VixImplied(), *(garch.GarchRv(k, **kw) for k in garch.KINDS))}
    wanted = list(every) if model_ids is None else [m for m in model_ids if m in every]
    return [every[m] for m in wanted]
