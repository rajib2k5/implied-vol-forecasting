"""Seeded synthetic market data with known truth (ADR 0001 D2 "Synthetic fixture").

Two jobs:

1. **Known answers for the EDA.** Each process has hand-set, round parameters,
   so the statistics the EDA estimates have closed forms to compare with:

   * ln VIX is an AR(1): ``x_t = mu + phi*(x_{t-1} - mu) + sigma*e_t``. Its
     ACF at lag k is ``phi**k`` and its half-life is ``ln(0.5)/ln(phi)``.
   * S&P 500 log returns are a GARCH(1,1) with normal shocks:
     ``r_t = sqrt(h_t)*z_t``, ``h_t = omega + alpha*r_{t-1}**2 + beta*h_{t-1}``.
     The ACF of r**2 has a closed form (``garch11_r2_acf``).
   * The other tenors are fixed functions of ln VIX, so the contango share
     has a closed form (``contango_probability``).

2. **The real code path in CI.** ``write_synthetic_raw`` writes Cboe-shaped
   CSVs and the yfinance-shaped CSV into ``ROOT/synthetic/raw/`` with the same
   storage and provenance code that ``fetch`` uses. ``build_interim``,
   ``check``, the features and the EDA then run on them unchanged.

The parameters are not fitted to real data. VIX and the S&P 500 are drawn
**independently** here, so the gap between implied and realised variance is a
difference of two unconditional levels, not a risk premium that moves with the
state. ADR 0001's later fixture (a two-factor term structure driving returns,
VXX/SVXY by roll yield) is for the app's model views and is not built in M1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ivf import repo_guard
from ivf.data import calendar, fetch, raw
from ivf.protocol import Protocol

#: When each synthetic index "starts", so the pipeline meets indices with short
#: histories the way it does on real data. These are the first dates in Cboe's
#: own files (checked on the files in M1); VIX starts at train_start.
FIRST_DATES = {
    "VIX1Y": date(2007, 1, 3),
    "VIX6M": date(2008, 1, 2),
    "VIX3M": date(2009, 9, 18),
    "VIX9D": date(2011, 1, 4),
    "VVIX": date(2006, 3, 6),
}


@dataclass(frozen=True)
class Tenor:
    """ln V_tenor = ln V_30 + offset + (slope - 1) * (ln V_30 - mu): a fixed map of ln VIX.

    ``offset`` is the log gap to VIX at VIX's long-run log level; ``slope`` is
    how strongly the tenor follows VIX (below 1: longer tenors move less).
    """

    symbol: str
    offset: float
    slope: float


@dataclass(frozen=True)
class SyntheticParams:
    seed: int = 20261028
    # ln VIX: AR(1). phi = 0.98 gives a half-life of ln(0.5)/ln(0.98) = 34.31 sessions.
    phi: float = 0.98
    mu: float = math.log(18.0)  # long-run mean of ln VIX: a median VIX of 18
    # sigma = 0.30*sqrt(1 - 0.98**2) = 0.30*0.19900 = 0.059699, so the stationary
    # sd of ln VIX is 0.30.
    sigma: float = 0.30 * math.sqrt(1 - 0.98**2)
    # Returns: GARCH(1,1). omega/(1 - alpha - beta) = 0.16**2/252 = 1.01587e-4 per
    # session, i.e. 16% annualised volatility.
    omega: float = 0.02 * 0.16**2 / 252
    alpha: float = 0.08
    beta: float = 0.90
    spx_start: float = 350.0
    # VVIX: the true vol of ln VIX is constant here, 100*sigma*sqrt(252) = 94.77, so
    # VVIX is that times small iid lognormal noise.
    vvix_noise: float = 0.05
    tenors: tuple[Tenor, ...] = field(
        default_factory=lambda: (
            Tenor("VIX9D", -0.06, 1.25),
            Tenor("VIX", 0.0, 1.0),
            Tenor("VIX3M", 0.08, 0.75),
            Tenor("VIX6M", 0.13, 0.60),
            Tenor("VIX1Y", 0.17, 0.50),
        )
    )

    @property
    def stationary_sd(self) -> float:
        return self.sigma / math.sqrt(1 - self.phi**2)

    @property
    def half_life(self) -> float:
        return math.log(0.5) / math.log(self.phi)

    @property
    def daily_variance(self) -> float:
        return self.omega / (1 - self.alpha - self.beta)


def simulate_log_ar1(n: int, phi: float, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """n draws of a stationary Gaussian AR(1); the first is drawn from the stationary law."""
    if not -1 < phi < 1:
        raise ValueError("phi must be inside (-1, 1) for a stationary start")
    shocks = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = mu + sigma / math.sqrt(1 - phi**2) * shocks[0]
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * shocks[t]
    return x


def simulate_garch11(
    n: int, omega: float, alpha: float, beta: float, rng: np.random.Generator, burn: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """``(r, h)``: n GARCH(1,1) returns with normal shocks and their conditional variances.

    Starts at the unconditional variance and discards ``burn`` draws, so the
    sample is close to the stationary law from its first value.
    """
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for a finite unconditional variance")
    z = rng.standard_normal(n + burn)
    h = np.empty(n + burn)
    r = np.empty(n + burn)
    h[0] = omega / (1 - alpha - beta)
    r[0] = math.sqrt(h[0]) * z[0]
    for t in range(1, n + burn):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = math.sqrt(h[t]) * z[t]
    return r[burn:], h[burn:]


def garch11_r2_acf(alpha: float, beta: float, lags: int) -> np.ndarray:
    """The ACF of r**2 for a GARCH(1,1) with normal shocks, at lags 1..``lags``.

    rho_1 = alpha*(1 - alpha*beta - beta**2) / (1 - 2*alpha*beta - beta**2) and
    rho_k = rho_1*(alpha + beta)**(k - 1) (Bollerslev 1988; He and Terasvirta
    1999). It needs a finite fourth moment: 3*alpha**2 + 2*alpha*beta + beta**2 < 1.

    Worked number: alpha = 0.10, beta = 0.80 gives
    rho_1 = 0.10*(1 - 0.08 - 0.64)/(1 - 0.16 - 0.64) = 0.10*0.28/0.20 = 0.14.
    """
    if 3 * alpha**2 + 2 * alpha * beta + beta**2 >= 1:
        raise ValueError("the fourth moment is infinite, so the ACF of r**2 is undefined")
    rho1 = alpha * (1 - alpha * beta - beta**2) / (1 - 2 * alpha * beta - beta**2)
    return rho1 * (alpha + beta) ** np.arange(lags)


def contango_probability(params: SyntheticParams, long_symbol: str = "VIX3M") -> float:
    """P(VIX <= VIX_long) under the stationary law of ln VIX.

    ln(VIX/VIX_long) = -offset + (1 - slope)*(x - mu), so VIX <= VIX_long exactly
    when x - mu <= offset/(1 - slope). With VIX3M (0.08, 0.75) the threshold is
    0.08/0.25 = 0.32 and the probability is Phi(0.32/0.30) = Phi(1.0667) = 0.8569.
    """
    tenor = next(t for t in params.tenors if t.symbol == long_symbol)
    return float(norm.cdf(tenor.offset / (1 - tenor.slope) / params.stationary_sd))


def synthetic_market(protocol: Protocol, params: SyntheticParams | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(iv, price)`` frames in the interim tables' shape, one row per session of 𝒯 up to ``data_end``.

    Every index has a value on every session from its first date: there are no
    gaps unless a test makes one.
    """
    params = params or SyntheticParams()
    sessions = calendar.xnys_sessions(protocol.train_start, protocol.require_data_end())
    n = len(sessions)
    rng = np.random.default_rng(params.seed)
    x = simulate_log_ar1(n, params.phi, params.mu, params.sigma, rng)
    r, _ = simulate_garch11(n, params.omega, params.alpha, params.beta, rng)
    vvix = 100 * params.sigma * math.sqrt(252) * np.exp(params.vvix_noise * rng.standard_normal(n))
    iv_rows = []
    for tenor in params.tenors:
        level = np.exp(x + tenor.offset + (tenor.slope - 1) * (x - params.mu))
        first = FIRST_DATES.get(tenor.symbol, protocol.train_start)
        iv_rows += [
            {"symbol": tenor.symbol, "date": d, "open": v, "high": v, "low": v, "close": v}
            for d, v in zip(sessions, level)
            if d >= first
        ]
    iv_rows += [
        {"symbol": "VVIX", "date": d, "open": None, "high": None, "low": None, "close": v}
        for d, v in zip(sessions, vvix)
        if d >= FIRST_DATES["VVIX"]
    ]
    # The first return is not used (its previous close is before train_start).
    close = params.spx_start * np.exp(np.cumsum(np.concatenate([[0.0], r[1:]])))
    previous = np.concatenate([[params.spx_start], close[:-1]])
    price = pd.DataFrame(
        {
            "ticker": "^GSPC",
            "date": sessions,
            "open": previous,
            "high": np.maximum(previous, close),
            "low": np.minimum(previous, close),
            "close": close,
            "adj_close": close,
            "volume": pd.array([None] * n, dtype="Int64"),
        }
    )
    return pd.DataFrame(iv_rows), price


def cboe_csv_bytes(frame: pd.DataFrame, symbol: str) -> bytes:
    """A Cboe-shaped CSV: ``DATE,OPEN,HIGH,LOW,CLOSE`` with MM/DD/YYYY dates, or ``DATE,<SYMBOL>`` when close-only."""
    rows = frame[frame["symbol"] == symbol].sort_values("date")
    close_only = rows["open"].isna().all()
    lines = [f"DATE,{symbol}" if close_only else "DATE,OPEN,HIGH,LOW,CLOSE"]
    for row in rows.itertuples(index=False):
        day = row.date.strftime("%m/%d/%Y")
        values = [row.close] if close_only else [row.open, row.high, row.low, row.close]
        lines.append(",".join([day] + [f"{v:.6f}" for v in values]))
    return ("\n".join(lines) + "\n").encode("ascii")


def yfinance_shaped_csv_bytes(price: pd.DataFrame) -> bytes:
    """The bytes ``fetch.yfinance_csv_bytes`` would store, built from a price frame."""
    index = pd.DatetimeIndex(pd.to_datetime(price["date"])).tz_localize(fetch.YF_TIMEZONE)
    frame = pd.DataFrame(
        {
            "Open": price["open"].to_numpy(),
            "High": price["high"].to_numpy(),
            "Low": price["low"].to_numpy(),
            "Close": price["close"].to_numpy(),
            "Adj Close": price["adj_close"].to_numpy(),
            "Volume": price["volume"].astype("float64").to_numpy(),
        },
        index=index,
    )
    return fetch.yfinance_csv_bytes(frame)


def write_synthetic_raw(
    root: Path, protocol: Protocol, params: SyntheticParams | None = None, fetched_at: datetime | None = None
) -> list[raw.FetchRecord]:
    """Write every synthetic file into ``root/raw`` exactly as ``fetch`` stores a download.

    ``fetched_at`` defaults to the day after ``data_end`` at noon UTC, so every
    session up to ``data_end`` counts as settled and repeated runs log the same
    time. Re-running stores nothing new (the bytes are identical) and appends
    one provenance line per symbol with ``changed`` false, as a real re-fetch would.
    A ``root`` inside the repository is refused first, as for real data (amended M1, review A).
    """
    import uuid

    import ivf

    repo_guard.require_data_root_outside_repo(root)
    raw.require_appendable_log(root)  # before storing anything (amended M1, review B)
    params = params or SyntheticParams()
    iv, price = synthetic_market(protocol, params)
    if fetched_at is None:
        end = protocol.require_data_end()
        fetched_at = datetime(end.year, end.month, end.day, 12, tzinfo=timezone.utc) + pd.Timedelta(days=1)
    contents = [(symbol, cboe_csv_bytes(iv, symbol)) for symbol in sorted(iv["symbol"].unique())]
    contents.append(("^GSPC", yfinance_shaped_csv_bytes(price)))
    records = []
    for symbol, content in contents:
        sha256, raw_path, changed = raw.store_snapshot(root, "synthetic", symbol, content)
        record = raw.FetchRecord(
            fetch_id=uuid.uuid4().hex,
            source="synthetic",
            symbol=symbol,
            url=f"synthetic://{symbol}?seed={params.seed}",
            final_url=None,
            fetched_at=raw.format_utc(fetched_at),
            http_status=None,
            bytes=len(content),
            sha256=sha256,
            raw_path=raw_path,
            changed=changed,
            tool_version=f"ivf {ivf.__version__}; ivf.data.synthetic",
        )
        raw.append_provenance(root, record)
        records.append(record)
    return records
