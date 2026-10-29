"""Probabilistic and Deflated Sharpe ratios, copied from sharpe-observatory (milestone 6).

ATTRIBUTION AND LICENCE
    The functions below the line ``# ---- copied ----`` are copied verbatim from
    ``sharpe-observatory/scripts/metrics.py`` at commit 080a044 (Rajib Bahar, MIT licence):

    * lines 36 and 39: ``TRADING_DAYS``, ``EULER_MASCHERONI``
    * lines 42-54:   ``sharpe_ratio``
    * lines 84-94:   ``sample_moments``
    * lines 97-112:  ``probabilistic_sharpe_ratio``
    * lines 122-139: ``expected_max_sharpe``
    * lines 142-144: ``null_sharpe_variance``
    * lines 147-149: ``deflated_sharpe_ratio``

    The MIT licence of that file, reproduced as it requires:

        MIT License

        Copyright (c) 2026 Rajib Bahar

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

CHECKED AGAINST THE PRIMARY SOURCE (2026-10-29)
    Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for Selection
    Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40(5), the
    authors' PDF at davidhbailey.com/dhbpapers/deflated-sharpe.pdf, eq. (2): DSR = Z[(SR^ - SR^_0)
    sqrt(T - 1) / sqrt(1 - g3 SR^ + (g4 - 1)/4 SR^2)], with SR^_0 = sqrt(V[{SR^_n}]) ((1 - gamma)
    Z^-1[1 - 1/N] + gamma Z^-1[1 - 1/(N e)]), V the variance ACROSS the trials' estimated SRs, N the
    number of independent trials, and g3, g4 the skewness and kurtosis of the SELECTED strategy's
    returns. The paper's own Snippet 1 (``getExpMaxSR``) matches ``expected_max_sharpe`` with
    ``mean_sr`` = mu. The typeset eq. (2) shows sqrt(T - 1) and (g4 - 1)/4 with g4 the Pearson
    kurtosis (checked by M6 review A, 2026-10-29; the text extraction of the first check had lost
    those glyphs). With g3 = 0, g4 = 3 the denominator is 1 + SR^2/2, the iid-normal variance of an
    SR estimate.

HOW MILESTONE 6 CALLS THEM (``ivf.backtest.metrics``)
    The paper's null is "every trial has true SR 0", so ``expected_max_sharpe`` is called with
    ``mean_sr = 0``. ``var_sr`` is the LARGER of the cross-trial variance and the null sampling
    variance ``null_sharpe_variance(T)`` (sharpe-observatory's README convention): with a few highly
    correlated trials (band variants of one rule) the cross-trial variance alone is tiny and would
    barely deflate. The paper-only version is kept as ``dsr_paper_var``. N is the raw count of trials
    M, not the paper's "implied independent trials" (Appendix 3, eq. 9), which is smaller for
    correlated trials: using M overstates E[max SR] (the paper says so), the conservative side.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

# ---- copied from sharpe-observatory/scripts/metrics.py @ 080a044 (MIT) ----

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649


def sharpe_ratio(excess: ArrayLike) -> float:
    """Per-period Sharpe ratio: mean(excess) / std(excess, ddof=1).

    ``excess`` is total return minus the per-period risk-free rate. Returns NaN for
    fewer than two observations or zero dispersion.
    """
    x = np.asarray(excess, dtype=float)
    if x.size < 2:
        return math.nan
    sd = x.std(ddof=1)
    if sd == 0.0:
        return math.nan
    return float(x.mean() / sd)


def sample_moments(x: ArrayLike) -> tuple[float, float]:
    """Sample skewness and PEARSON kurtosis (normal = 3) of ``x``.

    Uses the moment estimators of Bailey & López de Prado (2012): gamma3 = m3 / m2^1.5 and
    gamma4 = m4 / m2^2 with central moments m_k = mean((x - mean(x))^k). scipy's
    ``kurtosis`` returns excess kurtosis unless ``fisher=False``; PSR needs Pearson.
    """
    arr = np.asarray(x, dtype=float)
    skew = float(stats.skew(arr, bias=True))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=True))
    return skew, kurt


def probabilistic_sharpe_ratio(sr: float, sr_star: float, n: int, skew: float, kurtosis: float) -> float:
    """Probabilistic Sharpe Ratio, Bailey & López de Prado (2012), eq. for PSR(SR*).

    PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + ((g4 - 1) / 4) * SR^2) )

    ``sr`` and ``sr_star`` are per-period (not annualized); ``n`` is the number of
    observations; ``skew`` is g3; ``kurtosis`` is g4, PEARSON (normal = 3). Returns NaN
    when the variance term is non-positive or any input is NaN.
    """
    if n < 2 or not all(math.isfinite(v) for v in (sr, sr_star, skew, kurtosis)):
        return math.nan
    variance_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr**2
    if variance_term <= 0.0:
        return math.nan
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, mean_sr: float, var_sr: float) -> float:
    """Expected maximum Sharpe ratio across ``n_trials`` independent trials (False Strategy Theorem).

    E[max SR_N] = E[SR] + sqrt(V[SR]) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))

    with g the Euler-Mascheroni constant and Z the standard-normal quantile (Bailey,
    Borwein, López de Prado & Zhu 2014; López de Prado 2020, Machine Learning for Asset
    Managers, ch. 8 and App. B). Inputs are per-period.
    The expression is an extreme-value approximation that diverges at N = 1 because
    Z(0) = -inf; the maximum of a single trial is that trial, so N = 1 returns E[SR].
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return float(mean_sr)
    g = EULER_MASCHERONI
    quantile_mix = (1.0 - g) * stats.norm.ppf(1.0 - 1.0 / n_trials) + g * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_sr + math.sqrt(var_sr) * quantile_mix)


def null_sharpe_variance(n_obs: int, sr: float = 0.0) -> float:
    """Asymptotic variance of a per-period SR estimate for iid normal returns: (1 + SR^2/2) / n."""
    return (1.0 + sr**2 / 2.0) / n_obs


def deflated_sharpe_ratio(sr: float, n: int, skew: float, kurtosis: float, expected_max_sr: float) -> float:
    """Deflated Sharpe Ratio: PSR evaluated at SR* = E[max SR_N] (all per-period)."""
    return probabilistic_sharpe_ratio(sr, expected_max_sr, n, skew, kurtosis)


# ---- end of the copied code ----
