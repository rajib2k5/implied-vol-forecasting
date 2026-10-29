"""Tests for ivf.eda: hand-worked values, independent implementations, and recovery of known truth.

Recovery tolerances come from 20 seeds (scratch script, 2026-10-29): each is at
least 4 standard deviations of the estimator across seeds, so a pass is not luck
and a broken estimator still fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as sm_acf

from ivf import eda
from ivf.data import synthetic
from ivf.features import build

P = synthetic.SyntheticParams()


def approx(x, rel=1e-9, abs=1e-12):
    return pytest.approx(x, rel=rel, abs=abs, nan_ok=True)


# --- (a) ACF and Ljung-Box -------------------------------------------------------------


def test_acf_of_1_2_3_4_is_0_25_at_lag_1():
    # m = 2.5; deviations -1.5, -0.5, 0.5, 1.5; lag-0 sum 5; lag-1 sum 0.75 - 0.25 + 0.75 = 1.25.
    assert eda.acf([1, 2, 3, 4], 1) == approx([1.0, 0.25])


def test_acf_agrees_with_statsmodels_on_a_series_without_gaps():
    x = np.random.default_rng(7).standard_normal(500).cumsum()
    assert eda.acf(x, 10) == approx(sm_acf(x, nlags=10, adjusted=False, fft=False), rel=1e-10)


def test_acf_never_pairs_values_across_a_gap():
    # x = 1, 2, nan, 4, 5: m = 3, deviations -2, -1, ., 1, 2, lag-0 sum 10.
    # Lag-1 pairs that exist on the calendar: (2,1) -> (-1)(-2) = 2 and (5,4) -> (2)(1) = 2: rho_1 = 4/10 = 0.4.
    # Dropping the NaN first would pair 4 with 2 and give 3/10 = 0.3.
    assert eda.acf([1, 2, np.nan, 4, 5], 1)[1] == approx(0.4)


def test_ljung_box_on_1_2_3_4_is_q_0_5():
    # n = 4, rho_1 = 0.25: Q(1) = 4*6*0.0625/3 = 0.5; P(chi2_1 > 0.5) = 0.4795001.
    lb = eda.acf_ljung_box([1, 2, 3, 4], 1)
    assert lb.q == approx([0.5])
    assert lb.p_value == approx([0.47950012218695337])


def test_ljung_box_agrees_with_statsmodels():
    x = np.random.default_rng(11).standard_normal(800)
    ours = eda.acf_ljung_box(x, 12)
    theirs = acorr_ljungbox(x, lags=12)
    assert ours.q == approx(theirs["lb_stat"].to_numpy(), rel=1e-10)
    assert ours.p_value == approx(theirs["lb_pvalue"].to_numpy(), rel=1e-8)


def test_garch_squared_returns_cluster_with_the_closed_form_acf_while_returns_do_not():
    # alpha = 0.10, beta = 0.80: rho_1(r^2) = 0.10*0.28/0.20 = 0.14 and rho_5 = 0.14*0.9**4 = 0.091854.
    # Across 20 seeds at n = 200,000 the sample rho_1 had sd 0.0054 and rho_5 sd 0.0047.
    assert synthetic.garch11_r2_acf(0.10, 0.80, 2) == approx([0.14, 0.126])
    r, _ = synthetic.simulate_garch11(200_000, 1e-6, 0.10, 0.80, np.random.default_rng(3))
    a = eda.acf(r**2, 5)
    assert a[1] == pytest.approx(0.14, abs=0.025)
    assert a[5] == pytest.approx(0.091854, abs=0.025)
    squared, plain = eda.acf_ljung_box(r**2, 20), eda.acf_ljung_box(r, 20)
    assert squared.p_value[-1] < 1e-100  # clustering: overwhelming
    assert squared.q[-1] > 100 * plain.q[-1]


def test_the_closed_form_needs_a_finite_fourth_moment():
    # 3*0.2**2 + 2*0.2*0.79 + 0.79**2 = 0.12 + 0.316 + 0.6241 = 1.0601 >= 1
    with pytest.raises(ValueError, match="fourth moment"):
        synthetic.garch11_r2_acf(0.2, 0.79, 3)


# --- (b, c) persistence and the AR(1) half-life --------------------------------------------


@pytest.mark.parametrize("phi, expected", [(0.98, 34.309618491520645), (0.5, 1.0), (0.25, 0.5)])
def test_half_life_is_ln_half_over_ln_phi(phi, expected):
    # ln 0.5 / ln 0.98 = -0.693147/-0.020203 = 34.3096; ln 0.5/ln 0.5 = 1; ln 0.5/ln 0.25 = 0.5.
    assert eda.half_life_of(phi) == approx(expected)


def test_half_life_outside_zero_one_is_infinite_or_undefined():
    assert eda.half_life_of(1.0) == math.inf and eda.half_life_of(1.02) == math.inf
    assert math.isnan(eda.half_life_of(0.0)) and math.isnan(eda.half_life_of(-0.3))


@pytest.mark.parametrize("phi, bound", [(0.5, 1.0), (0.25, 0.5), (0.0, 0.0), (-0.1, 0.0), (1.0, math.inf), (1.01, math.inf)])
def test_a_confidence_interval_end_maps_to_a_half_life_bound_even_outside_zero_one(phi, bound):
    # Amended M1, review B: an interval end at or below 0 gives 0 (the limit of ln 0.5/ln phi as
    # phi falls to 0), not nan. Inside (0, 1) it is half_life_of: ln 0.5/ln 0.5 = 1, ln 0.5/ln 0.25 = 0.5.
    assert eda.half_life_bound(phi) == approx(bound)


def test_ar1_on_an_exact_halving_series_finds_phi_one_half():
    # 8, 4, 2, 1, 0.5: every step halves, so OLS fits x_t = 0 + 0.5*x_{t-1} exactly.
    hl = eda.ar1_half_life([8, 4, 2, 1, 0.5])
    assert (hl.phi, hl.half_life, hl.n) == (approx(0.5), approx(1.0), 4)


def test_ar1_forms_pairs_on_the_calendar_not_across_a_gap():
    # 16, 8, 4, gap, 1, 0.5, 0.25: the pairs are (8, 16), (4, 8), (0.5, 1), (0.25, 0.5), all on
    # x_t = 0.5*x_{t-1}. Bridging the gap would add the pair (1, 4) and pull phi off 0.5.
    hl = eda.ar1_half_life([16, 8, 4, np.nan, 1, 0.5, 0.25])
    assert hl.n == 4 and hl.phi == approx(0.5)


def test_ar1_recovers_phi_and_the_interval_covers_it():
    # phi = 0.98, n = 20,000: across 20 seeds phi_hat had sd 0.00124 and the half-life sd 2.1 sessions.
    x = synthetic.simulate_log_ar1(20_000, 0.98, P.mu, P.sigma, np.random.default_rng(5))
    hl = eda.ar1_half_life(x)
    assert hl.phi == pytest.approx(0.98, abs=0.005)
    assert hl.phi_ci[0] <= 0.98 <= hl.phi_ci[1]
    assert hl.half_life == pytest.approx(34.31, abs=9)
    lo, hi = hl.half_life_ci
    assert lo < hl.half_life < hi
    assert (lo, hi) == approx((eda.half_life_of(hl.phi_ci[0]), eda.half_life_of(hl.phi_ci[1])))


def test_hc1_and_ols_errors_agree_on_homoskedastic_data_and_differ_on_heteroskedastic():
    rng = np.random.default_rng(9)
    x = synthetic.simulate_log_ar1(5_000, 0.9, 0.0, 1.0, rng)
    ols, hc1 = eda.ar1_half_life(x, se_kind="OLS"), eda.ar1_half_life(x, se_kind="HC1")
    assert hc1.phi_se == pytest.approx(ols.phi_se, rel=0.1)
    # Shock size grows with |x_{t-1}|: OLS understates the uncertainty of phi (seeds 0-4: HC1/OLS = 11-14).
    y = np.zeros(5_000)
    for t in range(1, 5_000):
        y[t] = 0.5 * y[t - 1] + (0.2 + abs(y[t - 1])) * rng.standard_normal()
    assert eda.ar1_half_life(y, se_kind="HC1").phi_se > 3 * eda.ar1_half_life(y, se_kind="OLS").phi_se


def test_hc1_standard_error_and_95_percent_interval_by_hand():
    # x = 1, 2, 1, 3, 2: pairs (x_{t-1}, x_t) = (1,2), (2,1), (1,3), (3,2); n = 4, k = 2 regressors.
    # mean x_{t-1} = 7/4, mean x_t = 2; Sxx = 9/16 + 1/16 + 9/16 + 25/16 = 11/4; Sxy = -1.
    # phi = -1/(11/4) = -4/11; c = 2 + (4/11)(7/4) = 29/11. Residuals -3/11, -10/11, 8/11, 5/11.
    # HC0 var(phi) = sum (x - xbar)^2 e^2 / Sxx^2 = (9*9 + 1*100 + 9*64 + 25*25)/(16*121) / (121/16)
    #              = (1382/1936)/(121/16) = 0.0943925.
    # HC1 multiplies by n/(n - k) = 4/2: 0.1887849, se = 0.4344939 (HC0 would give 0.3072336).
    # 95% interval: phi -/+ 1.959964*se = (-1.2152287, 0.4879560); a 90% one would use 1.644854.
    hl = eda.ar1_half_life([1, 2, 1, 3, 2])
    assert hl.phi == approx(-4 / 11)
    assert hl.phi_se == approx(0.4344938653915675, rel=1e-12)
    assert hl.phi_ci == approx((-1.2152286913074302, 0.48795596403470287), rel=1e-12)
    # Half-life (amended M1, review B): phi < 0 is no decay, so the point estimate is nan; the
    # interval's lower end -1.2152 <= 0 maps to 0, its upper end to ln 0.5/ln 0.4879560 =
    # -0.6931472/-0.7175301 = 0.9660182 sessions. The first version gave (nan, 0.966).
    assert math.isnan(hl.half_life)
    assert hl.half_life_ci == approx((0.0, 0.9660182426208885), rel=1e-12)


def test_persistence_acf_of_an_ar1_decays_like_phi_to_the_k():
    x = synthetic.simulate_log_ar1(200_000, 0.9, 0.0, 1.0, np.random.default_rng(2))
    assert eda.acf(x, 3)[1:] == pytest.approx([0.9, 0.81, 0.729], abs=0.01)


# --- (d) unit-root tests ------------------------------------------------------------------


def test_a_random_walk_is_not_judged_stationary():
    # ADF H0 (unit root) is not rejected, KPSS H0 (stationary) is rejected. 20 of 20 seeds agreed.
    walk = np.cumsum(np.random.default_rng(1).standard_normal(2_000))
    ur = eda.unit_root_tests(walk)
    assert ur.adf_p > 0.05
    assert ur.kpss_p < 0.05
    assert ur.stationary_at_5pct is False


def test_a_stationary_ar1_is_judged_stationary():
    # phi = 0.5, n = 5,000: ADF rejected on 100 of 100 seeds, KPSS on 7 of 100 (its size).
    x = synthetic.simulate_log_ar1(5_000, 0.5, 0.0, 1.0, np.random.default_rng(1000))
    ur = eda.unit_root_tests(x)
    assert ur.adf_p < 0.05 and ur.kpss_p >= 0.05
    assert ur.stationary_at_5pct is True
    assert ur.kpss_p_bound == ">"  # statsmodels clips KPSS p at 0.10 and warns; we keep the bound


def test_when_kpss_rejects_the_verdict_is_not_stationary_even_if_adf_rejects():
    # A fast AR(0.5) whose mean shifts by +1 halfway: ADF rejects a unit root (the series reverts
    # quickly) and KPSS rejects stationarity around ONE mean (both rejected on 10 of 10 seeds).
    # "Stationary at 5%" needs ADF to reject AND KPSS not to, so the verdict is no.
    x = synthetic.simulate_log_ar1(2_000, 0.5, 0.0, 1.0, np.random.default_rng(0))
    x[1_000:] += 1.0
    ur = eda.unit_root_tests(x)
    assert ur.adf_p < 0.05 and ur.kpss_p < 0.05
    assert ur.stationary_at_5pct is False


def test_kpss_tests_level_stationarity_not_trend_stationarity():
    # A linear trend plus noise is trend-stationary but not level-stationary. KPSS with
    # regression="c" (E7's null: stationary around one level) rejects it; "ct" would not (statistic
    # 0.0513, p > 0.10, measured with statsmodels 0.15.0). Our statistic must be statsmodels' "c" one.
    from statsmodels.tsa.stattools import kpss

    x = np.arange(400) * 0.01 + np.random.default_rng(0).standard_normal(400)
    ur = eda.unit_root_tests(x)
    with pytest.warns(Warning):
        expected = kpss(x, regression="c", nlags="auto")[0]
    assert ur.kpss_stat == approx(expected, rel=1e-12) and ur.kpss_stat == approx(3.2138177067996554, rel=1e-9)
    assert ur.kpss_p == 0.01 and ur.kpss_p_bound == "<" and ur.stationary_at_5pct is False


def test_unit_root_tests_use_the_longest_gap_free_segment_and_never_join_across_a_gap():
    # Gap policy G2: ADF and KPSS lag and difference consecutive values, so they run on the longest
    # run of sessions with no gap. Here: 300 values, gap at 99 and 250 -> runs [0, 99), [100, 250)
    # and [251, 300); the longest is [100, 250), 150 values. Review A: the first version dropped the
    # NaNs and tested 298 values joined across both gaps.
    x = synthetic.simulate_log_ar1(300, 0.5, 0.0, 1.0, np.random.default_rng(4))
    x[[99, 250]] = np.nan
    assert eda.longest_gap_free_segment(x) == (100, 250)
    assert eda.longest_gap_free_segment([np.nan, 1, 2, np.nan, 3, 4, 5, np.nan]) == (4, 7)
    ur = eda.unit_root_tests(x)
    assert (ur.n, ur.segment, ur.n_nonmissing) == (150, (100, 250), 298)
    same = eda.unit_root_tests(x[100:250])
    assert (ur.adf_stat, ur.kpss_stat) == (same.adf_stat, same.kpss_stat)


def test_unit_root_nulls_are_stated_the_right_way_round():
    assert "unit root" in eda.UnitRoot.ADF_NULL and "stationary" in eda.UnitRoot.KPSS_NULL


# --- (e) the IV-RV spread -------------------------------------------------------------------


def test_spread_by_hand():
    # V = 20, 10 and the next-21-session RV = 15, 12: vol gaps 5 and -2 (mean 1.5, median 1.5);
    # variance gaps 0.04 - 0.0225 = 0.0175 and 0.01 - 0.0144 = -0.0044 (mean 0.00655); half positive.
    s = eda.iv_rv_spread([20.0, 10.0, 30.0], [15.0, 12.0, np.nan])
    assert (s.n, s.mean_vol_points, s.median_vol_points) == (2, approx(1.5), approx(1.5))
    assert s.mean_variance == approx(0.00655)
    assert s.share_positive == 0.5


def test_spread_respects_the_period_mask():
    s = eda.iv_rv_spread([20.0, 10.0], [15.0, 12.0], mask=[True, False])
    assert (s.n, s.mean_variance) == (1, approx(0.0175))


def test_spread_recovers_the_known_variance_gap():
    # E[(V/100)^2] = exp(2 mu + 2 s^2)/1e4 = 324*exp(0.18)/1e4 = 0.038790; E[rv21^2/1e4] = 252*omega/(1-a-b)
    # = 0.0256. Truth 0.013190. Across 20 seeds at n = 200,000 the estimate had sd 0.00064.
    rng = np.random.default_rng(4)
    x = synthetic.simulate_log_ar1(200_000, P.phi, P.mu, P.sigma, rng)
    r, _ = synthetic.simulate_garch11(200_000, P.omega, P.alpha, P.beta, rng)
    s = eda.iv_rv_spread(np.exp(x), build.forward(build.rv_trailing(r), 21))
    truth = math.exp(2 * P.mu + 2 * P.stationary_sd**2) / 1e4 - 252 * P.daily_variance
    assert truth == approx(0.0131898425651, rel=1e-9)
    assert s.mean_variance == pytest.approx(truth, abs=0.003)


# --- (f) vol-of-vol ----------------------------------------------------------------------------


def test_vov_is_annualised_with_sqrt_252():
    # The alternating +-0.1 series has a 21-session sd of 0.10235326 (test_features), times
    # sqrt(252) = 15.8745079: 1.6248077.
    steps = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(21)])
    v = np.exp(np.concatenate([[3.0], 3.0 + np.cumsum(steps)]))
    assert eda.vol_of_vol(v).vov_annualised[21] == approx(1.6248077, rel=1e-7)


def test_vov_recovers_the_vol_of_log_vix_and_finds_no_level_effect_when_shocks_are_iid():
    # Truth: sd(d ln V) = sigma*sqrt(2/(1+phi)) = 0.059699*1.005038 = 0.060000, times sqrt(252) = 0.952470.
    # A 21-session sd is biased low (about 0.8% here, measured over 20 seeds: mean 0.9446, sd 0.0016).
    x = synthetic.simulate_log_ar1(200_000, P.phi, P.mu, P.sigma, np.random.default_rng(8))
    vov = eda.vol_of_vol(np.exp(x), e6_boot=0)  # no bootstrap: this test is about the estimates
    assert vov.mean_vov == pytest.approx(0.9524704719832526, rel=0.025)
    # With iid shocks the size of the next shock does not depend on the level (5 seeds: |rho| <= 0.0044).
    assert abs(vov.spearman_level_vs_next_abs_resid) < 0.012
    # ...but in vol points it does, by construction: a 6% move is more points at a higher level.
    assert vov.spearman_level_vs_next_abs_dpoints > 0.3


def skewed_ar1(seed: int, n: int = 200_000) -> np.ndarray:
    """ln V with iid but skewed shocks (exponential minus 1: big up-jumps, small down-moves), like VIX spikes."""
    shocks = np.random.default_rng(seed).exponential(1.0, n) - 1.0
    x = np.empty(n)
    x[0] = P.mu
    for t in range(1, n):
        x[t] = P.mu + P.phi * (x[t - 1] - P.mu) + P.sigma * shocks[t]
    return x


def test_e6_measure_ignores_skewed_shocks_that_fool_the_simpler_measures():
    # The shock size never depends on the level here, so the E6 measure must be near zero (the
    # verdict would be "not shown"; E6 never says "no"). Measured over 5 seeds:
    # the next AR(1) shock gave |rho| <= 0.0095, the next raw move +0.105 to +0.114 (mean-reversion
    # pull), the same-day move +0.163 to +0.172 (a high level is made by a big up-move).
    from scipy.stats import spearmanr

    x = skewed_ar1(0)
    v = np.exp(x)
    vov = eda.vol_of_vol(v, e6_boot=0)
    assert abs(vov.spearman_level_vs_next_abs_resid) < 0.02
    assert vov.spearman_level_vs_next_abs_dlog > 0.08
    assert spearmanr(v[1:], np.abs(np.diff(x))).statistic > 0.12


def test_vov_sees_a_level_effect_when_there_is_one():
    rng = np.random.default_rng(12)
    x = np.zeros(50_000)
    x[0] = P.mu
    for t in range(1, 50_000):  # shocks twice as large above the mean
        scale = 2.0 if x[t - 1] > P.mu else 1.0
        x[t] = P.mu + P.phi * (x[t - 1] - P.mu) + scale * P.sigma * rng.standard_normal()
    vov = eda.vol_of_vol(np.exp(x), e6_boot=199)
    assert vov.spearman_level_vs_next_abs_resid > 0.1
    # Every one of the 199 bootstrap draws lies below the observed rho, so p takes its smallest
    # possible value: 2*(1 + 0)/(199 + 1) = 0.01.
    assert vov.e6_p_bootstrap == approx(0.01) and vov.e6_n_boot == 199


MIN_N = 6000  # eda.E6_MIN_N as chosen in the M1 final check (walkthrough, section 7.3)


def test_e6_verdict_answers_only_up_to_the_persistence_cutoff():
    # M1 final check: above E6_PHI_MAX the test's size was not shown to hold (15% to 44.5% false
    # "yes" under persistent skewed GARCH), so a significant positive rho there is "not shown".
    n = eda.E6_MIN_N
    assert eda.E6_PHI_MAX == 0.985
    assert eda.e6_verdict(0.2, 0.01, 0.985, n) == ("yes", "")  # at the cutoff: answered
    assert eda.e6_verdict(0.2, 0.01, 0.9851, n) == ("not shown", eda.E6_REASON_PERSISTENT)
    assert eda.e6_verdict(0.2, 0.01, math.nan, n) == ("not shown", eda.E6_REASON_PERSISTENT)
    assert eda.e6_verdict(0.2, 0.20, 0.98, n) == ("not shown", eda.E6_REASON_NOT_REJECTED)
    assert eda.e6_verdict(-0.2, 0.01, 0.98, n) == ("not shown", eda.E6_REASON_NOT_REJECTED)  # never "no"
    assert eda.e6_verdict(0.2, math.nan, 0.98, n) == ("not shown", eda.E6_REASON_NO_P)


def test_e6_verdict_answers_only_from_the_minimum_sample_length():
    # M1 final check: below E6_MIN_N pairs the size failed even inside the phi cutoff (skewed
    # GARCH(0.03, 0.965), phi 0.98: "yes" on 24/100 paths at n = 2,000).
    assert eda.E6_MIN_N == MIN_N
    assert eda.e6_verdict(0.2, 0.01, 0.98, MIN_N) == ("yes", "")  # at the minimum: answered
    assert eda.e6_verdict(0.2, 0.01, 0.98, MIN_N - 1) == ("not shown", eda.E6_REASON_SHORT)
    assert str(MIN_N) in eda.E6_REASON_SHORT


def test_e6_is_not_shown_on_a_short_series_even_with_a_real_level_effect():
    # phi = 0.98 (inside the cutoff) and shocks twice as large above the mean, 2,000 sessions:
    # rho > 0 and p < 0.05, the old "yes"; the sample is short, so "not shown".
    rng = np.random.default_rng(4)
    x = np.empty(2000)
    x[0] = P.mu
    for t in range(1, 2000):
        scale = 2.0 if x[t - 1] > P.mu else 1.0
        x[t] = P.mu + 0.98 * (x[t - 1] - P.mu) + scale * P.sigma * rng.standard_normal()
    vov = eda.vol_of_vol(np.exp(x), e6_boot=199)
    assert vov.spearman_level_vs_next_abs_resid > 0 and vov.e6_p_bootstrap < 0.05  # the old "yes"
    assert vov.e6_phi <= eda.E6_PHI_MAX
    assert (vov.e6_verdict, vov.e6_reason) == ("not shown", eda.E6_REASON_SHORT)


def test_e6_is_not_shown_on_a_persistent_series_even_with_a_real_level_effect():
    # phi = 0.995 and shocks twice as large above the mean, a sample longer than E6_MIN_N: rho > 0
    # and p < 0.05, which was "yes" before the cutoff. phi_hat is above E6_PHI_MAX, so the verdict
    # is "not shown", with rho and p still reported.
    rng = np.random.default_rng(3)
    x = np.empty(MIN_N + 1000)
    x[0] = P.mu
    for t in range(1, len(x)):
        scale = 2.0 if x[t - 1] > P.mu else 1.0
        x[t] = P.mu + 0.995 * (x[t - 1] - P.mu) + scale * P.sigma * rng.standard_normal()
    vov = eda.vol_of_vol(np.exp(x), e6_boot=199)
    assert vov.spearman_level_vs_next_abs_resid > 0 and vov.e6_p_bootstrap < 0.05  # the old "yes"
    assert vov.e6_phi > eda.E6_PHI_MAX
    assert (vov.e6_verdict, vov.e6_reason) == ("not shown", eda.E6_REASON_PERSISTENT)


def skewed_garch_ar1(seed: int, n: int, phi: float, a: float = 0.03, b: float = 0.965) -> np.ndarray:
    """ln V with skewed GARCH(1,1) shocks (Exp(1) - 1 innovations), unconditional sd 0.06, no level effect."""
    rng = np.random.default_rng(seed)
    z = rng.exponential(size=n) - 1.0
    w, s2, e = 0.06**2 * (1 - a - b), 0.06**2, np.empty(n)
    for t in range(n):
        e[t] = math.sqrt(s2) * z[t]
        s2 = w + a * e[t] ** 2 + b * s2
    return P.mu + signal.lfilter([1.0], [1.0, -phi], e)


def test_e6_verdict_keeps_its_size_under_persistent_skewed_garch_through_the_cutoff():
    # The reviewer's failing null at suite size: skewed GARCH(0.03, 0.965), phi = 0.995, no level
    # effect, 100 paths of 2,000 sessions, B = 99 (about 4 s). Measured: rho > 0 and p < 0.05 on
    # 55/100 (the test without the cutoff), the verdict "yes" on 2/100. Fails above 10 verdicts, or
    # if fewer than 30 paths are significant (then the scenario no longer tests the cutoff).
    # The verdict is asked as if the sample were long enough (n_pairs = E6_MIN_N), to test the phi
    # gate alone: at n = 2,000 the test is oversized even at phi 0.98 (24/100), which is what
    # E6_MIN_N is for (test_e6_verdict_answers_only_from_the_minimum_sample_length).
    significant = verdicts = 0
    for seed in range(100):
        v = np.exp(skewed_garch_ar1(700_000 + seed, 2000, 0.995))
        _, phi, _ = eda.ar1_shocks(np.log(v))
        rho, p, _ = eda.e6_level_vs_next_shock(v, n_boot=99)
        significant += rho > 0 and p < 0.05
        verdicts += eda.e6_verdict(rho, p, phi, eda.E6_MIN_N)[0] == "yes"  # the phi gate alone
    assert significant >= 30, significant
    assert verdicts <= 10, verdicts


# --- (f) E6's test (amended M1, review A) -------------------------------------------------------


def test_ar1_shocks_on_an_exact_line_are_zero_and_need_the_intercept():
    # 0, 1, 1.5, 1.75 lies on x_t = 1 + 0.5*x_{t-1}: 1 + 0.5*0 = 1, 1 + 0.5*1 = 1.5, 1 + 0.5*1.5 = 1.75.
    # So c = 1, phi = 0.5 and every shock is 0. A shock without the intercept would be 1 each.
    c, phi, shock = eda.ar1_shocks([0.0, 1.0, 1.5, 1.75])
    assert (c, phi) == (approx(1.0), approx(0.5))
    assert np.isnan(shock[0]) and shock[1:] == approx([0.0, 0.0, 0.0], abs=1e-12)


def test_ar1_shocks_form_pairs_on_the_calendar_not_across_a_gap():
    # 0, 1, 1.5, gap, 3, 2.5: the calendar pairs (0,1), (1,1.5) and (3,2.5) all lie on
    # x_t = 1 + 0.5*x_{t-1} (1 + 0.5*3 = 2.5). Bridging the gap would add the pair (1.5, 3), which is
    # off the line (1 + 0.75 = 1.75), and pull c and phi away from 1 and 0.5.
    c, phi, shock = eda.ar1_shocks([0.0, 1.0, 1.5, np.nan, 3.0, 2.5])
    assert (c, phi) == (approx(1.0), approx(0.5))
    assert np.isnan(shock[[0, 3, 4]]).all() and shock[[1, 2, 5]] == approx([0.0, 0.0, 0.0], abs=1e-12)


def test_two_sided_bootstrap_p_by_hand():
    # draws -0.3, -0.1, 0.1, 0.2; observed 0.15: (1 + 1)/5 = 0.4 at or above, (1 + 3)/5 = 0.8 at or
    # below, so p = 2*0.4 = 0.8. Observed 0.5 is beyond every draw: 2*(1 + 0)/5 = 0.4.
    draws = np.array([-0.3, -0.1, 0.1, 0.2])
    assert eda.two_sided_bootstrap_p(0.15, draws) == approx(0.8)
    assert eda.two_sided_bootstrap_p(0.5, draws) == approx(0.4)
    assert eda.two_sided_bootstrap_p(0.0, draws) == approx(1.0)  # 2*min(3/5, 3/5) = 1.2, capped at 1


def test_e6_bootstrap_is_reproducible_and_keeps_the_calendar_pairs():
    x = skewed_ar1(3, 1_200)
    x[[100, 700]] = np.nan  # two gaps: 1,199 pairs minus 2 per gap
    a = eda.e6_level_vs_next_shock(np.exp(x), n_boot=99, seed=7)
    assert a == eda.e6_level_vs_next_shock(np.exp(x), n_boot=99, seed=7)
    assert a[2] == 1_199 - 4
    assert math.isnan(eda.e6_level_vs_next_shock(np.exp(x), n_boot=0)[1])  # no bootstrap, no p-value


def test_e6_size_under_skewed_iid_shocks_stays_near_5_percent():
    # Review A: with iid exponential shocks (no level effect at all) scipy's Spearman p-value on the
    # fitted shocks rejected at 5% on 30% of paths, because phi_hat's error enters every shock as
    # (phi - phi_hat)*ln V_t. The bootstrap refits phi on each resampled path, so its size is right.
    # Here: 100 fixed paths of 1,000 sessions, 99 resamples each (about 1.2 s). Measured 2026-10-29:
    # bootstrap 5/100, Spearman 30/100. The bound 12/100 is far from both. Larger runs (n = 9,000,
    # 300 paths, 999 resamples) are in the walkthrough, section 7.3.
    from scipy.stats import spearmanr

    rejected = spearman_rejected = 0
    for seed in range(100):
        x = skewed_ar1(seed, 1_000)
        rho, p, _ = eda.e6_level_vs_next_shock(np.exp(x), n_boot=99, seed=1)
        rejected += p < 0.05
        _, _, shock = eda.ar1_shocks(x)
        spearman_rejected += spearmanr(x[:-1], np.abs(shock[1:])).pvalue < 0.05
    assert rejected <= 12
    assert spearman_rejected >= 20  # the design this replaced fails on the same paths


# --- (f) E6 under volatility clustering (amended M1, review B) ------------------------------------


def garch_ar1(seed: int, n: int = 1_000, alpha: float = 0.10, beta: float = 0.88) -> np.ndarray:
    """ln V = AR(1) (phi 0.98) driven by symmetric GARCH(1,1) shocks: clustered, with NO level effect.

    The innovations are normal, so flipping the sign of every shock maps the path to its mirror
    image and leaves every shock size alone: the population rank correlation of the level with the
    next shock size is 0 by symmetry.
    """
    rng = np.random.default_rng(seed)
    z, shocks, variance = rng.standard_normal(n), np.empty(n), 1.0
    for t in range(n):
        shocks[t] = math.sqrt(variance) * z[t]
        variance = (1 - alpha - beta) + alpha * shocks[t] ** 2 + beta * variance
    x = np.empty(n)
    x[0] = P.mu
    for t in range(1, n):
        x[t] = P.mu + P.phi * (x[t - 1] - P.mu) + P.sigma * shocks[t]
    return x


def test_circular_blocks_by_hand():
    # Series 10, 11, 12, 13 and blocks of 3. Seed 4's first two draws below 4 are 2 and 3 (checked on
    # the generator itself), so the blocks are 12, 13, 10 and 13, 10, 11 (each wraps from the end to
    # the start); cut to 5 values: 12, 13, 10, 13, 10.
    assert np.random.default_rng(4).integers(4, size=(1, 2)).tolist() == [[2, 3]]
    out = eda._circular_blocks(np.random.default_rng(4), np.array([10.0, 11.0, 12.0, 13.0]), 1, 5, 3)
    assert out.tolist() == [[12.0, 13.0, 10.0, 13.0, 10.0]]


def test_e6_block_length_is_1_for_iid_shocks_and_long_for_clustered_ones():
    # The Politis-White rule on |shock|: iid shocks have no dependence to keep (measured over 300
    # paths of 9,000: mean 1.6, at most 4); GARCH(0.10, 0.88) shocks need long blocks (mean 204,
    # at least 140). The rule on the shocks themselves would see no dependence in either.
    iid = np.random.default_rng(1).standard_normal(9_000)
    assert eda.e6_block_length(iid) <= 4
    _, _, clustered = eda.ar1_shocks(garch_ar1(2, 9_000))
    assert eda.e6_block_length(clustered) >= 100
    assert eda.e6_block_length(np.full(50, 0.3)) == 1  # constant sizes: nothing to estimate, no division by 0


def test_e6_size_under_garch_shocks_with_no_level_effect_stays_near_5_percent():
    # Review B: the review-A bootstrap drew shocks one at a time (iid). With GARCH shocks and no
    # level effect it rejected at 5% on 47% of 300 paths of 9,000 (walkthrough, section 7.3),
    # because the level and the shock size are both persistent and rho spreads far wider than iid
    # draws allow. The circular block bootstrap keeps runs of shocks together.
    # Here: 100 fixed paths of 1,000 sessions, 99 resamples each. Measured 2026-10-29: shipped 1/100,
    # block=1 (the review-A design, on the same paths and seed) 44/100. Bounds 12 and 25 are far from both.
    rejected = review_a_rejected = 0
    for seed in range(100):
        v = np.exp(garch_ar1(seed))
        _, p, _ = eda.e6_level_vs_next_shock(v, n_boot=99, seed=1)
        _, p_iid, _ = eda.e6_level_vs_next_shock(v, n_boot=99, seed=1, block=1)
        rejected += p < 0.05
        review_a_rejected += p_iid < 0.05
    assert rejected <= 12
    assert review_a_rejected >= 25  # the design this replaced fails on the same paths


def test_e6_block_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        eda.e6_level_vs_next_shock(np.exp(skewed_ar1(1, 300)), n_boot=9, block=0)


# --- (g) term structure ----------------------------------------------------------------------


def test_term_structure_by_hand():
    # VIX 15, 20, 30 against VIX3M 18, 20, 25: contango (VIX <= VIX3M) on 2 of 3 sessions.
    # mean ln(VIX3M/VIX) = (ln 1.2 + 0 + ln(25/30))/3 = (0.182322 + 0 - 0.182322)/3 = 0.
    levels = {
        "VIX9D": np.array([14.0, 21.0, 35.0]),
        "VIX": np.array([15.0, 20.0, 30.0]),
        "VIX3M": np.array([18.0, 20.0, 25.0]),
        "VIX6M": np.array([19.0, 21.0, 24.0]),
        "VIX1Y": np.array([20.0, np.nan, 23.0]),
    }
    ts = eda.term_structure_summary(levels)
    assert (ts.n, ts.contango_share) == (3, approx(2 / 3))
    assert ts.mean_log_slope == approx(0.0, abs=1e-15)
    # Full curve only where all five exist (sessions 1 and 3): upward on 1 only.
    assert (ts.n_full_curve, ts.upward_full_curve_share) == (2, 0.5)


def test_contango_share_recovers_the_closed_form():
    # P(contango) = Phi(0.08/0.25/0.30) = Phi(1.0667) = 0.856939; 20 seeds at n = 200,000: sd 0.0064.
    x = synthetic.simulate_log_ar1(200_000, P.phi, P.mu, P.sigma, np.random.default_rng(6))
    vix = np.exp(x)
    vix3m = np.exp(x + 0.08 + (0.75 - 1) * (x - P.mu))
    ts = eda.term_structure_summary({"VIX9D": vix, "VIX": vix, "VIX3M": vix3m, "VIX6M": vix3m, "VIX1Y": vix3m})
    assert synthetic.contango_probability(P) == approx(0.8569388078044908)
    assert ts.contango_share == pytest.approx(0.8569388, abs=0.03)
