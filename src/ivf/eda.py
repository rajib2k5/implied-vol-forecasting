"""Exploratory statistics for milestone 1. Pure functions: arrays in, numbers out, no I/O.

Each function answers one question from the milestone brief:

(a) ``acf_ljung_box``: do returns cluster in volatility? ACF of r, r**2, |r| with Ljung-Box.
(b) ``acf``: how persistent are VIX and ln VIX?
(c) ``ar1_half_life``: AR(1) on ln VIX, half-life ln(0.5)/ln(phi) with a confidence interval.
(d) ``unit_root_tests``: ADF and KPSS on ln VIX. Their nulls are opposite (see there).
(e) ``iv_rv_spread``: VIX minus the *following* 21-session realised volatility.
(f) ``vol_of_vol``: annualised rolling sd of d ln VIX, and how it moves with the level.
(g) ``term_structure_summary``: how often the curve is in contango, and its slope.

Missing values are NaN (gap policy G2). Statistics over a series use
pairwise-complete data *on the calendar*: a lag-k pair needs both sessions,
k sessions apart on 𝒯, to have a value. Dropping NaNs first and then lagging
would silently pair values across a gap.

Real-data results from these functions are local only (ADR 0001 D2): none of
them is in the publication set P except the AR(1) half-life, and that only per
whole period, at 2 significant figures, from a run on a clean tree.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

SESSIONS_PER_YEAR = 252


def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def acf(x, nlags: int) -> np.ndarray:
    """Sample autocorrelations at lags 0..nlags, pairwise-complete on the calendar.

    rho_k = sum_t (x_t - m)(x_{t-k} - m) / sum_t (x_t - m)**2, with m the mean
    of the non-missing values and each sum over the terms where both values
    exist. With no gaps this is exactly the textbook (and statsmodels) estimator,
    which divides every lag by the same lag-0 sum.

    Worked number: x = 1, 2, 3, 4 has m = 2.5 and deviations -1.5, -0.5, 0.5,
    1.5; lag-0 sum = 2.25 + 0.25 + 0.25 + 2.25 = 5; lag-1 sum = (-0.5)(-1.5) +
    (0.5)(-0.5) + (1.5)(0.5) = 0.75 - 0.25 + 0.75 = 1.25; rho_1 = 1.25/5 = 0.25.
    """
    x = _as_float(x)
    d = x - np.nanmean(x)
    d0 = np.where(np.isnan(d), 0.0, d)
    denominator = np.sum(d0 * d0)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = np.sum(d0[k:] * d0[:-k]) / denominator
    return out


@dataclass(frozen=True)
class LjungBox:
    lags: np.ndarray
    acf: np.ndarray  # at lags 1..max
    q: np.ndarray  # Ljung-Box Q at each lag
    p_value: np.ndarray
    n: int


def acf_ljung_box(x, nlags: int = 20) -> LjungBox:
    """ACF at lags 1..nlags plus the Ljung-Box test of "no autocorrelation up to lag m" for each m.

    Q(m) = n(n+2) * sum_{k=1}^{m} rho_k**2 / (n - k), compared with chi-squared(m).
    n is the number of non-missing values. Under the null (an uncorrelated
    series) Q is small; volatility clustering makes Q for r**2 and |r| huge
    while Q for r stays modest.

    Worked number: n = 100, rho_1 = 0.2 gives Q(1) = 100*102*0.04/99 = 4.1212,
    p = P(chi2_1 > 4.1212) = 0.0423.
    """
    x = _as_float(x)
    n = int(np.sum(~np.isnan(x)))
    rho = acf(x, nlags)[1:]
    k = np.arange(1, nlags + 1)
    q = n * (n + 2) * np.cumsum(rho**2 / (n - k))
    return LjungBox(lags=k, acf=rho, q=q, p_value=stats.chi2.sf(q, k), n=n)


def lag_pairs(x, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(x_t, x_{t-k})`` for every t where both exist on the calendar."""
    x = _as_float(x)
    y, lagged = x[k:], x[:-k]
    keep = ~(np.isnan(y) | np.isnan(lagged))
    return y[keep], lagged[keep]


@dataclass(frozen=True)
class HalfLife:
    phi: float
    phi_se: float
    phi_ci: tuple[float, float]
    half_life: float  # sessions; inf if phi >= 1, nan if phi <= 0
    half_life_ci: tuple[float, float]  # half_life_bound of each end of phi_ci: 0 for an end <= 0, inf for one >= 1
    n: int
    se_kind: str


def half_life_of(phi: float) -> float:
    """ln(0.5)/ln(phi) sessions: how long a deviation from the mean takes to halve.

    Defined for 0 < phi < 1. phi >= 1 never halves (inf); phi <= 0 is not a
    smooth decay (nan). Worked number: phi = 0.98 gives -0.693147/-0.020203 = 34.31.
    """
    if phi >= 1:
        return math.inf
    if phi <= 0:
        return math.nan
    return math.log(0.5) / math.log(phi)


def half_life_bound(phi: float) -> float:
    """The half-life at one END of a confidence interval for phi (amended M1, review B).

    Like ``half_life_of`` inside (0, 1), and its limits outside: an end at or above 1 gives inf
    (the interval reaches a unit root, so no half-life is ruled out above), and an end at or below
    0 gives 0. The second is the convention review B asked for: the first version returned nan,
    so a white-noise series reported a half-life interval of (nan, 0.245). The half-lives the
    interval allows are those of its phi in (0, 1), and as phi falls to 0 the half-life falls to 0
    (ln(0.5)/ln(phi) -> 0), so 0 is the honest lower end: "as short as you like". A negative phi
    is not a smooth decay at all, so an interval wholly at or below 0 gives (0, 0).

    The point estimate keeps ``half_life_of``'s nan for phi <= 0: one number that is not a decay
    has no half-life, while an interval end is a bound on the ones that are.

    Worked numbers: phi = 0.5 gives ln 0.5/ln 0.5 = 1 session; phi = 0.25 gives
    -0.693147/-1.386294 = 0.5; phi = -0.1 and phi = 0 give 0; phi = 1.01 gives inf.
    """
    if phi <= 0:
        return 0.0
    return half_life_of(phi)


def ar1_half_life(x, level: float = 0.95, se_kind: str = "HC1") -> HalfLife:
    """OLS of x_t on (1, x_{t-1}) over complete pairs, and the half-life with a confidence interval.

    The interval for phi is phi_hat +/- z*se. The half-life is a monotone
    (increasing) function of phi on (0, 1), so mapping the two ends of phi's
    interval gives the half-life's interval exactly, with no delta method. An
    upper end at or above 1 gives an infinite upper half-life, which is the
    honest answer: the data cannot rule out a unit root at that level. A lower
    end at or below 0 gives a lower half-life of 0 (``half_life_bound``;
    amended M1, review B).

    ``se_kind`` is ``HC1`` (White's heteroskedasticity-robust SE, the default,
    because the shocks to ln VIX are larger in stressed markets) or ``OLS``.
    OLS phi is biased down in finite samples by about (1 + 3*phi)/n (Kendall
    1954): at phi = 0.98 and n = 9,000, 3.94/9000 = 0.00044, which moves the
    half-life from 34.31 to about 33.6 sessions.
    """
    y, lagged = lag_pairs(x, 1)
    n = len(y)
    if n < 3:
        raise ValueError("need at least 3 complete pairs")
    X = np.column_stack([np.ones(n), lagged])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    if se_kind == "OLS":
        cov = xtx_inv * (resid @ resid) / (n - 2)
    elif se_kind == "HC1":
        meat = (X * resid[:, None] ** 2).T @ X
        cov = xtx_inv @ meat @ xtx_inv * n / (n - 2)
    else:
        raise ValueError(se_kind)
    phi, se = float(beta[1]), float(math.sqrt(cov[1, 1]))
    z = stats.norm.ppf(0.5 + level / 2)
    lo, hi = phi - z * se, phi + z * se
    return HalfLife(
        phi=phi,
        phi_se=se,
        phi_ci=(lo, hi),
        half_life=half_life_of(phi),
        half_life_ci=(half_life_bound(lo), half_life_bound(hi)),
        n=n,
        se_kind=se_kind,
    )


@dataclass(frozen=True)
class UnitRoot:
    adf_stat: float
    adf_p: float
    adf_lags: int
    kpss_stat: float
    kpss_p: float
    kpss_p_bound: str  # "", "<" or ">": statsmodels reports KPSS p only inside [0.01, 0.10]
    n: int  # length of the segment tested
    stationary_at_5pct: bool
    segment: tuple[int, int] = (0, 0)  # [start, stop) positions of the tested segment in the input
    n_nonmissing: int = 0  # all non-missing values in the input, for comparison with n

    ADF_NULL = "ADF null: the series has a unit root (it does NOT mean-revert). Rejecting it is evidence of mean reversion."
    KPSS_NULL = "KPSS null: the series is (level-)stationary. Rejecting it is evidence AGAINST mean reversion."


def longest_gap_free_segment(x) -> tuple[int, int]:
    """``(start, stop)`` of the longest run of consecutive non-missing values; the first one on a tie.

    Worked example: ``[nan, 1, 2, nan, 3, 4, 5, nan]`` has runs [1, 3) and [4, 7); the answer is (4, 7).
    """
    present = ~np.isnan(_as_float(x))
    best, start = (0, 0), None
    for i, ok in enumerate(np.append(present, False)):  # the sentinel closes a run at the end
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def unit_root_tests(x, alpha: float = 0.05) -> UnitRoot:
    """ADF (constant, lag order by AIC) and KPSS (constant, automatic bandwidth) on the longest gap-free segment.

    The nulls are opposite, so the two tests answer different questions:

    * ADF H0: unit root. A small p-value rejects it: the series mean-reverts.
    * KPSS H0: stationary. A small p-value rejects it: it does not.

    "Stationary at 5%" (statement E7) means ADF rejects and KPSS does not.

    Both tests regress on lags and differences of consecutive values, so they
    need an unbroken stretch of the calendar. Gap policy G2 forbids pairing a
    value with "the previous row that exists", so the tests run on the longest
    run of consecutive sessions with a value (``longest_gap_free_segment``),
    and ``n``, ``segment`` and ``n_nonmissing`` say how much that left out.
    (amended M1, review A: the first version dropped the NaNs and so joined the
    neighbours of every gap, which G2 forbids.)
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning

    raw_values = _as_float(x)
    start, stop = longest_gap_free_segment(raw_values)
    values = raw_values[start:stop]
    adf_stat, adf_p, adf_lags, *_ = adfuller(values, regression="c", autolag="AIC", result_object=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")
    bound = ""
    for warning in caught:
        if issubclass(warning.category, InterpolationWarning):
            bound = "<" if "smaller" in str(warning.message) else ">"
    return UnitRoot(
        adf_stat=float(adf_stat),
        adf_p=float(adf_p),
        adf_lags=int(adf_lags),
        kpss_stat=float(kpss_stat),
        kpss_p=float(kpss_p),
        kpss_p_bound=bound,
        n=len(values),
        stationary_at_5pct=bool(adf_p < alpha and kpss_p >= alpha),
        segment=(int(start), int(stop)),
        n_nonmissing=int(np.sum(~np.isnan(raw_values))),
    )


@dataclass(frozen=True)
class Spread:
    n: int
    mean_vol_points: float
    median_vol_points: float
    mean_variance: float
    share_positive: float  # share of origins with implied variance above realised


def iv_rv_spread(vix, rv21_forward, mask=None) -> Spread:
    """VIX at t minus the realised volatility of the 21 sessions after t.

    Vol points: V_t - RV_{t->t+21}. Variance units (annualised, decimal):
    (V_t/100)**2 - (RV/100)**2. A positive mean in variance units is the
    ex-post variance risk premium (statement E5). ``rv21_forward`` is
    ``features.forward(rv21, 21)``: it uses data after t, which is fine for a
    description of the past and forbidden as a model input (rule L1).

    Worked number: V = 20 and RV = 15 give 5 vol points and
    0.04 - 0.0225 = 0.0175 in variance.
    """
    v, rv = _as_float(vix), _as_float(rv21_forward)
    keep = ~(np.isnan(v) | np.isnan(rv))
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    gap = v[keep] - rv[keep]
    var_gap = (v[keep] / 100) ** 2 - (rv[keep] / 100) ** 2
    n = int(keep.sum())
    if n == 0:
        return Spread(0, math.nan, math.nan, math.nan, math.nan)
    return Spread(
        n=n,
        mean_vol_points=float(gap.mean()),
        median_vol_points=float(np.median(gap)),
        mean_variance=float(var_gap.mean()),
        share_positive=float((var_gap > 0).mean()),
    )


#: E6's bootstrap (amended M1, review A; blocks amended M1, review B): resamples, and the seed that
#: makes the p-value reproducible.
E6_BOOTSTRAP = 999
E6_SEED = 20261029

#: E6 answers "yes" only when the fitted AR(1) phi of ln V is at most this (M1 final check).
#: Its size was measured to hold only up to here. Skewed GARCH(0.03, 0.965) shocks, no level
#: effect, 200 paths of 9,000 sessions, B = 999 (tools/probes/e6_size_power.py): rejects at 5%
#: on 0.0% at phi 0.98, 2.0% at 0.985, 15.0% at 0.99 and 44.5% at 0.995 (lognormal innovations
#: at 0.995: 34.5%). The block (mean about 245, near the rule's cap) is shorter than the level's
#: memory there. Pre-registered rule: the largest of 0.975, 0.98 and 0.985 at which every null in
#: the probe rejects on at most 8% and says "yes" on at most 5%.
E6_PHI_MAX = 0.985
#: ...and only on at least this many level/next-shock pairs (M1 final check, the fix round's own
#: finding). Below it the test is oversized under persistent skewed clustering even inside the phi
#: cutoff: skewed GARCH(0.03, 0.965), phi 0.98, B = 99, "yes" on 39/100 paths at n = 1,000 and
#: 24/100 at 2,000. Rule (default decision, 2026-10-29): the smallest of 4,000, 6,000 and 8,000 at
#: which "yes" is at most 10 of 200 fresh paths (B = 999) for skewed GARCH(0.03, 0.965) at phi
#: 0.98 and 0.985 and for iid skewed shocks (walkthrough, section 7.3, has the table). Set to
#: 6,000, not the rule's 4,000: 4,000 passed only on the boundary (10 of 200 at phi 0.98, on 3,999
#: pairs), so the next size, with a clear margin (1, 0 and 3 of 200), was taken.
E6_MIN_N = 6000
E6_REASON_SHORT = f"sample below the size-checked length (n < {E6_MIN_N} pairs)"
E6_REASON_PERSISTENT = f"persistence above the size-checked range (fitted AR(1) phi above {E6_PHI_MAX})"
E6_REASON_NO_P = "no bootstrap p-value"
E6_REASON_NOT_REJECTED = "not rejected at 5%; the test has little power when shocks cluster, so this is not a no"
_E6_CHUNK = 128  # bootstrap paths simulated at once, to bound memory (128 x 9,000 floats = 9 MB)


@dataclass(frozen=True)
class VolOfVol:
    vov_annualised: np.ndarray  # rolling sd of d ln VIX, times sqrt(252)
    mean_vov: float
    spearman_level_vs_next_abs_resid: float  # V_t against |AR(1) shock to ln V at t+1|: the E6 measure
    e6_p_bootstrap: float  # two-sided bootstrap p-value of the E6 measure (nan if not computed)
    e6_n_boot: int  # bootstrap paths behind it (0: not computed)
    e6_block: int  # circular-block length of that bootstrap (e6_block_length; 0: not computed)
    e6_phi: float  # the fitted AR(1) phi of ln V that E6's verdict is gated on (E6_PHI_MAX)
    e6_verdict: str  # "yes" or "not shown" (e6_verdict); never "no"
    e6_reason: str  # why "not shown" ("" for "yes")
    spearman_level_vs_next_abs_dlog: float  # V_t against |ln V_{t+1} - ln V_t|
    spearman_level_vs_next_abs_dpoints: float  # V_t against |V_{t+1} - V_t|
    spearman_vix_vs_vvix: float  # nan without VVIX
    n: int


def _spearman(a, b) -> tuple[float, float]:
    a, b = _as_float(a), _as_float(b)
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return math.nan, math.nan  # a constant side has no ranks to correlate
    result = stats.spearmanr(a[keep], b[keep])
    return float(result.statistic), float(result.pvalue)


def _ols_rows(lagged: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS of each row of ``y`` on (1, the same row of ``lagged``): ``(intercept, slope)``, one per row."""
    lm, ym = lagged.mean(axis=-1, keepdims=True), y.mean(axis=-1, keepdims=True)
    slope = ((lagged - lm) * (y - ym)).sum(axis=-1, keepdims=True) / ((lagged - lm) ** 2).sum(axis=-1, keepdims=True)
    return ym - slope * lm, slope


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman's rho of each row of ``a`` with the same row of ``b``: Pearson on average ranks, as scipy does."""
    ra, rb = stats.rankdata(a, axis=-1), stats.rankdata(b, axis=-1)
    ra -= ra.mean(axis=-1, keepdims=True)
    rb -= rb.mean(axis=-1, keepdims=True)
    return (ra * rb).sum(axis=-1) / np.sqrt((ra**2).sum(axis=-1) * (rb**2).sum(axis=-1))


def ar1_shocks(log_v) -> tuple[float, float, np.ndarray]:
    """``(c, phi, shock)``: the AR(1) of ``log_v`` by OLS on calendar pairs, and its residuals.

    ``shock[t] = log_v[t] - c - phi*log_v[t-1]`` where both values exist, NaN
    elsewhere (gap policy G2: no pair across a gap). The fit is the one
    ``ar1_half_life`` makes. Worked number: ``0, 1, 1.5, 1.75`` lies exactly on
    ``x_t = 1 + 0.5*x_{t-1}`` (1 + 0.5*0 = 1, 1 + 0.5*1 = 1.5, 1 + 0.5*1.5 =
    1.75), so c = 1, phi = 0.5 and every shock is 0.
    """
    x = _as_float(log_v)
    y, lagged = x[1:], x[:-1]
    keep = ~(np.isnan(y) | np.isnan(lagged))
    c, phi = _ols_rows(lagged[keep], y[keep])
    c, phi = float(c[0]), float(phi[0])
    shock = np.full(x.shape, np.nan)
    shock[1:][keep] = y[keep] - c - phi * lagged[keep]
    return c, phi, shock


def two_sided_bootstrap_p(observed: float, draws: np.ndarray) -> float:
    """Equal-tailed two-sided p-value of ``observed`` against bootstrap ``draws``, with the +1 rule.

    ``min(1, 2*min((1 + #{d >= observed}) / (B + 1), (1 + #{d <= observed}) / (B + 1)))``. The +1
    counts the observed sample as one of the draws, so p is never 0 and its smallest value is
    2/(B + 1). Worked number: draws -0.3, -0.1, 0.1, 0.2 and observed 0.15: one draw is >= (1 + 1)/5
    = 0.4, three are <= (1 + 3)/5 = 0.8, p = 2*0.4 = 0.8.
    """
    draws = np.asarray(draws, dtype=float)
    above = (1 + np.sum(draws >= observed)) / (len(draws) + 1)
    below = (1 + np.sum(draws <= observed)) / (len(draws) + 1)
    return float(min(1.0, 2 * min(above, below)))


def e6_block_length(shocks) -> int:
    """The block length of E6's bootstrap: Politis and White's rule applied to |shock| (amended M1, review B).

    The automatic block length of Politis & White (2004), with the correction of Patton, Politis &
    White (2009), for the circular block bootstrap, as ``arch.bootstrap.optimal_block_length``
    computes it (its ``circular`` column), applied to the absolute fitted shocks, rounded, and at
    least 1. NaNs (the shocks of gap sessions) are dropped first.

    Why |shock| and not the shock. The shocks of an AR(1) fit are nearly uncorrelated whatever
    their volatility does, so the rule applied to them picks L = 1 even under strong clustering.
    Clustering lives in the *size* of the shocks, and the size is what E6 is about. The rule grows
    L with the autocorrelation of |shock|: iid shocks get L = 1, which is the review-A iid bootstrap
    exactly; GARCH-like shocks get long blocks (arch caps L at min(3*sqrt(n), n/3), 284 at n =
    9,000). The walkthrough (section 7.3) lists the L it chose on each simulated null.
    """
    from arch.bootstrap import optimal_block_length

    size = np.abs(_as_float(shocks))
    size = size[~np.isnan(size)]
    if len(size) < 3 or np.ptp(size) == 0:
        return 1  # constant sizes: no dependence to keep, and the rule would divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        length = float(optimal_block_length(size)["circular"].iloc[0])
    return max(1, int(round(length))) if math.isfinite(length) else 1


def _circular_blocks(rng: np.random.Generator, series: np.ndarray, rows: int, length: int, block: int) -> np.ndarray:
    """``rows`` resamples of ``series``, each ``length`` long: blocks of ``block`` consecutive values, wrapping at the end.

    Each block starts at a uniformly drawn position and runs on, wrapping from the last value to
    the first (Politis & Romano's circular block bootstrap). ``block = 1`` draws single values with
    replacement. Worked example: series 10, 11, 12, 13, block 3, starts 3 and 1 give 13, 10, 11 |
    11, 12, 13, cut to the length asked for.
    """
    n_blocks = -(-length // block)
    starts = rng.integers(len(series), size=(rows, n_blocks))
    index = (starts[:, :, None] + np.arange(block)) % len(series)
    return series[index.reshape(rows, n_blocks * block)[:, :length]]


def e6_level_vs_next_shock(
    vix, n_boot: int = E6_BOOTSTRAP, seed: int = E6_SEED, block: int | None = None
) -> tuple[float, float, int]:
    """Statement E6: ``(rho, p, n_pairs)``, the rank correlation of V_t with the size of the next AR(1) shock.

    ``rho`` = Spearman(V_t, |ln V_{t+1} - c - phi*ln V_t|) over calendar pairs, with c and phi
    fitted on the same sample (``ar1_shocks``).

    **Its p-value is a circular block bootstrap of the fitted shocks that refits the AR(1) on every
    resample** (amended M1, review A for the refit; review B for the blocks). Each of ``n_boot``
    paths starts at the first observed ln V and follows ``x*_t = c + phi*x*_{t-1} + e*_t``, where
    ``e*`` is the fitted shock series cut into blocks of ``block`` consecutive shocks
    (``e6_block_length`` when ``block`` is None) and put back together in random order
    (``_circular_blocks``). The path takes the same gaps as the data and goes through the same fit
    and the same rho. ``p`` is the equal-tailed two-sided position of the observed rho among them
    (``two_sided_bootstrap_p``).

    What the null is. Inside a block, the shocks keep their order, so whatever their sizes do
    together over a few dozen sessions (volatility clustering, and big up-moves followed by high
    volatility) is kept, and so is its effect on the level the path reaches. Between blocks the order
    is random, so the level a path is at when a block starts has nothing to do with that block's
    shock sizes. A small p therefore says: the level goes with the size of the next shock more than
    the shocks' own short-range dependence explains.

    Why blocks (review B). The review-A version drew single shocks (iid). Under GARCH(1,1) shocks
    with no level effect at all it rejected on 47% of 300 paths (alpha 0.10, beta 0.88) and 19%
    (0.05, 0.90) at a nominal 5%. Both rho's series are persistent there (the level through phi, the
    shock size through the clustering), so rho spreads far wider than iid draws allow. With skewed
    GARCH shocks the level and the next shock size even correlate in the population (+0.13 and
    +0.25, n = 10^6), because a big up-move raises both. A wild bootstrap fails too: it keeps |e_t|
    but flips signs, so it loses the skew (26% size, Rademacher, on iid exponential shocks). The
    walkthrough (section 7.3) has the table, its commands, and the power.

    Why not scipy's Spearman p-value (review A). It assumes independent pairs and known shocks. The
    error in phi_hat moves every residual by (phi - phi_hat)*ln V_t, which grows with the level,
    and with skewed shocks that rejected on 30% of 300 stationary paths. Refitting inside every
    resample reproduces that error; fitting on one half and testing on the other did not.

    ``n_boot = 0`` skips the bootstrap (``p`` is nan). So does a fit with ``|phi| >= 1``, where the
    stationary null cannot be simulated.
    """
    v = _as_float(vix)
    log_v = np.log(v)
    c, phi, shock = ar1_shocks(log_v)
    pairs = ~np.isnan(shock[1:])  # pair (t, t+1) exists on the calendar
    level, size = log_v[:-1][pairs], np.abs(shock[1:][pairs])
    n_pairs = int(pairs.sum())
    if n_pairs < 3 or np.ptp(level) == 0 or np.ptp(size) == 0:
        return math.nan, math.nan, n_pairs
    rho = float(_spearman_rows(level, size))
    if n_boot <= 0 or not -1 < phi < 1:
        return rho, math.nan, n_pairs
    rng = np.random.default_rng(seed)
    # The fitted shocks in calendar order. A gap's missing shock is left out, so a block can run
    # across a gap; it is a sequence of shocks, not of levels, so nothing is paired across the gap.
    residuals = shock[~np.isnan(shock)]
    residuals = residuals - residuals.mean()  # zero already, up to rounding: OLS with an intercept
    block = e6_block_length(residuals) if block is None else int(block)
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}")
    first = int(np.flatnonzero(~np.isnan(log_v))[0])
    missing = np.isnan(log_v[first:])
    draws = np.empty(n_boot)
    for lo in range(0, n_boot, _E6_CHUNK):
        rows = min(_E6_CHUNK, n_boot - lo)
        e = _circular_blocks(rng, residuals, rows, len(missing) - 1, block)
        # x*_t = phi*x*_{t-1} + (c + e*_t), started at the first observed value: a linear filter.
        path = np.empty((rows, len(missing)))
        path[:, 0] = log_v[first]
        path[:, 1:] = signal.lfilter([1.0], [1.0, -phi], c + e, axis=1, zi=np.full((rows, 1), phi * log_v[first]))[0]
        path[:, missing] = np.nan  # the data's gaps, so the same calendar pairs exist
        y, lagged = path[:, 1:][:, pairs[first:]], path[:, :-1][:, pairs[first:]]
        c_b, phi_b = _ols_rows(lagged, y)  # refit: the step the Spearman p-value leaves out
        draws[lo : lo + rows] = _spearman_rows(lagged, np.abs(y - c_b - phi_b * lagged))
    return rho, two_sided_bootstrap_p(rho, draws), n_pairs


def e6_verdict(rho: float, p: float, phi: float, n_pairs: int) -> tuple[str, str]:
    """Statement E6's verdict, ``(word, reason)``: "yes" or "not shown", never "no" (M1 final check).

    "yes" needs ``rho > 0``, ``p < 0.05``, ``n_pairs >= E6_MIN_N`` AND ``phi <= E6_PHI_MAX``: the
    test answers only where its size was measured to hold. Shorter samples are oversized under
    persistent skewed clustering (``E6_MIN_N``). Above the cutoff, under persistent skewed clustering with no level
    effect, it said "yes" on 15% (phi 0.99) to 44.5% (0.995) of simulated paths (``E6_PHI_MAX``).
    The word is "not shown" with a reason otherwise; ``rho`` and ``p`` are still reported beside
    it. Failing to reject is "not shown", not "no": the block bootstrap has little power when
    shocks cluster (18% against a strong level effect on top of GARCH; walkthrough, section 7.3).
    A ``phi`` of nan counts as above the cutoff.
    """
    if not (math.isfinite(rho) and math.isfinite(p)):
        return "not shown", E6_REASON_NO_P
    if n_pairs < E6_MIN_N:
        return "not shown", E6_REASON_SHORT
    if not phi <= E6_PHI_MAX:
        return "not shown", E6_REASON_PERSISTENT
    if rho > 0 and p < 0.05:
        return "yes", ""
    return "not shown", E6_REASON_NOT_REJECTED


def vol_of_vol(vix, window: int = 21, vvix=None, *, e6_boot: int = E6_BOOTSTRAP, e6_seed: int = E6_SEED) -> VolOfVol:
    """Vol-of-vol, and whether it rises with the level (statement E6).

    ``vov_annualised[t]`` = sd(d ln V over the ``window`` sessions to t) * sqrt(252),
    trailing only. Worked number: a daily sd of d ln V of 0.06 is
    0.06*sqrt(252) = 0.06*15.8745 = 0.9525 annualised (95% a year).

    "Rises with the level" (E6) is measured as the rank correlation of the
    level V_t with the size of the NEXT AR(1) shock to ln V,
    |ln V_{t+1} - c - phi*ln V_t|, with c and phi fitted on the sample, and
    tested by the circular block bootstrap of ``e6_level_vs_next_shock``
    (amended M1, review B). Two simpler measures mislead, and the tests show
    both:

    * the same-day move |d ln V_t|: a level is high *because* of a big up-move,
      so with skewed shocks (VIX spikes up and decays down) the two correlate
      even when the shock size never depends on the level;
    * the next raw move |d ln V_{t+1}|: it contains the mean-reversion pull
      (phi - 1)(ln V_t - mu), which is larger far from the mean, so with skewed
      shocks it also correlates with the level (synthetic, exponential shocks:
      rank correlation about +0.11 over 5 seeds, against about 0 for the shock).

    With iid shocks the E6 measure is centred near zero whatever their shape,
    but its spread is wider than independent pairs give once phi is
    estimated, and wider still when the shocks cluster; with skewed clustered
    shocks it is not even centred at zero. That is why its p-value comes from
    a bootstrap that refits phi and keeps runs of consecutive shocks together.
    In vol points (|d V_{t+1}|) the correlation is positive by construction,
    since the same percentage move is more points at a higher level.
    """
    from ivf.features.build import log_returns, trailing

    v = _as_float(vix)
    dlog = log_returns(v)
    dpoints = np.full(v.shape, np.nan)
    dpoints[1:] = v[1:] - v[:-1]
    vov = trailing(dlog, window, "std") * math.sqrt(SESSIONS_PER_YEAR)
    _, phi, shocks = ar1_shocks(np.log(v))
    finite = shocks[~np.isnan(shocks)]
    block = e6_block_length(finite) if e6_boot > 0 and len(finite) > 2 else 0
    rho_resid, p_resid, n_pairs = e6_level_vs_next_shock(v, n_boot=e6_boot, seed=e6_seed, block=block or None)
    rho_log, _ = _spearman(v[:-1], np.abs(dlog[1:]))
    rho_points, _ = _spearman(v[:-1], np.abs(dpoints[1:]))
    rho_vvix = _spearman(v, vvix)[0] if vvix is not None else math.nan
    verdict, reason = e6_verdict(rho_resid, p_resid, phi, n_pairs)
    return VolOfVol(
        vov_annualised=vov,
        mean_vov=float(np.nanmean(vov)),
        spearman_level_vs_next_abs_resid=rho_resid,
        e6_p_bootstrap=p_resid,
        e6_n_boot=e6_boot if not math.isnan(p_resid) else 0,
        e6_block=block if not math.isnan(p_resid) else 0,
        e6_phi=float(phi),
        e6_verdict=verdict,
        e6_reason=reason,
        spearman_level_vs_next_abs_dlog=rho_log,
        spearman_level_vs_next_abs_dpoints=rho_points,
        spearman_vix_vs_vvix=rho_vvix,
        n=int(np.sum(~np.isnan(vov))),
    )


@dataclass(frozen=True)
class TermStructure:
    n: int
    contango_share: float  # VIX <= VIX3M (the regime rule: backwardation iff VIX > VIX3M)
    mean_log_slope: float  # mean of ln(VIX3M/VIX)
    n_full_curve: int
    upward_full_curve_share: float  # every tenor <= the next longer one, where all five exist


def term_structure_summary(levels: dict[str, np.ndarray], order=("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"), mask=None) -> TermStructure:
    """Contango share and slope (statement E4).

    Worked number: VIX = 15, 20, 30 and VIX3M = 18, 20, 25 give ratios 0.833,
    1.0, 1.2: contango (VIX <= VIX3M) on 2 of 3 sessions, share 0.6667.
    """
    vix, vix3m = _as_float(levels["VIX"]), _as_float(levels["VIX3M"])
    keep = ~(np.isnan(vix) | np.isnan(vix3m))
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    curve = np.vstack([_as_float(levels[s]) for s in order])
    full = ~np.isnan(curve).any(axis=0)
    if mask is not None:
        full &= np.asarray(mask, dtype=bool)
    upward = np.all(curve[:-1, full] <= curve[1:, full], axis=0)
    return TermStructure(
        n=int(keep.sum()),
        contango_share=float((vix[keep] <= vix3m[keep]).mean()) if keep.any() else math.nan,
        mean_log_slope=float(np.log(vix3m[keep] / vix[keep]).mean()) if keep.any() else math.nan,
        n_full_curve=int(full.sum()),
        upward_full_curve_share=float(upward.mean()) if full.any() else math.nan,
    )
