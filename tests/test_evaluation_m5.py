"""Milestone 5: Diebold-Mariano, Holm, Mincer-Zarnowitz, regimes, calibration and the report step.

Hand-worked values carry their arithmetic in the comments. No network; synthetic data only.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ivf.evaluation import calibration, dm, losses, mz, regimes, report

# --- Diebold-Mariano: the pieces by hand -------------------------------------------------------


def test_hln_factor_by_hand():
    # sqrt((n + 1 - 2h + h(h-1)/n) / n): n = 10, h = 2 -> (10 + 1 - 4 + 2/10)/10 = 7.2/10 = 0.72
    assert dm.hln_factor(10, 2) == pytest.approx(math.sqrt(0.72), abs=1e-15)
    # h = 1: (n + 1 - 2 + 0)/n = (n - 1)/n; n = 4 -> 0.75
    assert dm.hln_factor(4, 1) == pytest.approx(math.sqrt(0.75), abs=1e-15)
    # n = 4, h = 2: (4 + 1 - 4 + 2/4)/4 = 1.5/4 = 0.375
    assert dm.hln_factor(4, 2) == pytest.approx(math.sqrt(0.375), abs=1e-15)


def test_dm_rectangular_by_hand():
    # d = [2, 0, 1, 3]: mean 1.5, deviations [0.5, -1.5, -0.5, 1.5]
    # gamma_0 = (0.25 + 2.25 + 0.25 + 2.25)/4 = 1.25
    # gamma_1 = (0.5*-1.5 + -1.5*-0.5 + -0.5*1.5)/4 = (-0.75 + 0.75 - 0.75)/4 = -0.1875
    # h = 2, rectangular: LRV = 1.25 + 2*(-0.1875) = 0.875 > 0
    # DM = 1.5 / sqrt(0.875/4) = 1.5 / 0.4677072 = 3.2071349; times HLN sqrt(0.375) = 1.9639610
    r = dm.dm_test([2.0, 0.0, 1.0, 3.0], 2, "rectangular")
    assert r.lrv == pytest.approx(0.875, abs=1e-15)
    assert r.lrv_method == "rectangular" and r.max_lag == 1 and r.n == 4 and r.mean_d == 1.5
    assert r.dm_stat == pytest.approx(1.5 / math.sqrt(0.875 / 4) * math.sqrt(0.375), abs=1e-12)
    assert r.dm_stat == pytest.approx(1.9639610, abs=1e-7)
    assert r.p_value == pytest.approx(2 * stats.t.sf(1.9639610121, 3), abs=1e-9)  # t with n - 1 = 3 df


def test_dm_rectangular_falls_back_to_bartlett_when_the_estimate_is_not_positive():
    # d = [1, -1, 2, 0]: mean 0.5, deviations [0.5, -1.5, 1.5, -0.5]
    # gamma_0 = (0.25 + 2.25 + 2.25 + 0.25)/4 = 1.25; gamma_1 = (-0.75 - 2.25 - 0.75)/4 = -0.9375
    # rectangular: 1.25 - 1.875 = -0.625 <= 0 -> Bartlett at lag h-1 = 1, w_1 = 1 - 1/2 = 0.5:
    # 1.25 + 2*0.5*(-0.9375) = 0.3125; DM = 0.5/sqrt(0.3125/4) = 1.7888544; * sqrt(0.375) = 1.0954451
    r = dm.dm_test([1.0, -1.0, 2.0, 0.0], 2, "rectangular")
    assert r.lrv_method == "bartlett_fallback" and r.lrv == pytest.approx(0.3125, abs=1e-15)
    assert r.dm_stat == pytest.approx(1.0954451, abs=1e-7)
    nw = dm.dm_test([1.0, -1.0, 2.0, 0.0], 2, "bartlett")  # Newey-West at h-1 is the same number
    assert nw.lrv_method == "bartlett" and nw.dm_stat == pytest.approx(r.dm_stat, abs=1e-15)


def test_fft_autocovariances_equal_the_direct_sums():
    x = np.random.default_rng(3).standard_normal((4, 300))
    fft = dm.autocovariances(x, 40)
    direct = np.array([dm.autocovariances(row, 40) for row in x])
    assert np.allclose(fft, direct, atol=1e-12)


def test_andrews_bandwidth_by_hand():
    # rho = 0.5: alpha(1) = 4*0.25 / ((0.5)^2 (1.5)^2) = 1/0.5625 = 1.7777778
    # n = 1000: (1777.7778)^(1/3) = 12.114137; * 1.1447 = 13.867053
    assert dm.andrews_bandwidth_from_rho(0.5, 1000) == pytest.approx(13.867053, abs=1e-6)
    # an exact AR(1) slope: d = [1, 2, 1, 2, 1]: mean 1.4, deviations [-.4, .6, -.4, .6, -.4]
    # slope = sum(x_t x_{t-1}) / sum(x_{t-1}^2) = (4 * -0.24) / (0.16 + 0.36 + 0.16 + 0.36) = -0.96/1.04
    assert dm.ar1_coefficient([1, 2, 1, 2, 1]) == pytest.approx(-0.96 / 1.04, abs=1e-12)


def test_fixed_b_null_is_seeded_and_the_p_value_is_monotone():
    # Amended M5, review A: the first version asserted agreement with a Kiefer-Vogelsang cubic quoted from
    # memory (2.2606); that comparison is not verified and was removed. What is pinned is the shipped,
    # seeded null: 20,000 draws, n = 1,000, seed 20261028, 95% quantile of |t| = 2.24576 (measured), the
    # same on a fresh simulation (the cache cleared).
    null = dm.fixed_b_null(dm.FIXED_B)
    assert np.quantile(null, 0.95) == pytest.approx(2.24576, abs=5e-6)
    dm.fixed_b_null.cache_clear()
    assert np.array_equal(dm.fixed_b_null(dm.FIXED_B), null)
    assert dm.fixed_b_lags(583) == 59 and dm.fixed_b_lags(6090) == 609  # ceil(0.1 n)
    assert dm.fixed_b_p_value(0.0) == 1.0
    assert dm.fixed_b_p_value(1.0) > dm.fixed_b_p_value(2.0) > dm.fixed_b_p_value(3.0)
    assert dm.fixed_b_p_value(1e6) == 1 / (1 + null.size)  # beyond every draw: (1 + 0)/(1 + R), never 0
    assert math.isnan(dm.fixed_b_p_value(math.nan))  # amended M5, review A: was 1/(1 + R)
    assert dm.fixed_b_p_value(-2.0) == dm.fixed_b_p_value(2.0)  # two-sided


def test_dm_fixed_b_statistic_by_hand():
    # Amended M5, review A: the primary statistic had no hand-worked test (HLN applied to it passed all tests).
    # d = [2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 1]: n = 11, sum 11, mean 1
    # deviations x = [1, -1, 0, 1, -1, 0, 1, -1, 0, 0, 0]
    # gamma_0 = (1+1+0+1+1+0+1+1+0+0+0)/11 = 6/11
    # gamma_1 = (x2x1 + ... + x11x10)/11 = (-1 + 0 + 0 - 1 + 0 + 0 - 1 + 0 + 0 + 0)/11 = -3/11
    # fixed-b: M = ceil(0.1 * 11) = 2 -> Bartlett lags j < 2, w_1 = 1 - 1/2 = 0.5
    # LRV = 6/11 + 2 * 0.5 * (-3/11) = 3/11; t = 1 / sqrt((3/11)/11) = 11/sqrt(3) = 6.3508530
    # No HLN factor (it belongs to the lag-(h-1) estimators): at h = 2 it would give 6.35 * 0.9258 instead.
    d = np.array([2.0, 0, 1, 2, 0, 1, 2, 0, 1, 1, 1])
    r = dm.dm_test(d, 2, "fixed_b")
    assert (r.max_lag, r.bandwidth, r.lrv_method) == (1, 2.0, "fixed_b")
    assert r.lrv == pytest.approx(3 / 11, abs=1e-15)
    assert r.dm_stat == pytest.approx(11 / math.sqrt(3), abs=1e-12) and r.dm_stat == pytest.approx(6.3508530, abs=1e-7)
    assert r.p_value == dm.fixed_b_p_value(11 / math.sqrt(3))  # the same simulated null, (1 + #exceed)/(1 + R)
    neg = dm.dm_test(-d, 2, "fixed_b")
    assert neg.dm_stat == pytest.approx(-r.dm_stat, abs=1e-12) and neg.p_value == r.p_value


def test_the_simulated_ma_errors_are_sums_of_h_consecutive_shocks():
    # design "ma", h = 3: d_t = eps_t + eps_{t+1} + eps_{t+2}, from the same generator state
    d = dm.simulate_differentials("ma", 10, 3, 2, np.random.default_rng(9))
    eps = np.random.default_rng(9).standard_normal((2, 12))
    assert np.allclose(d, eps[:, 0:10] + eps[:, 1:11] + eps[:, 2:12], atol=1e-12)


def test_vectorised_rejections_agree_with_dm_test_row_by_row():
    d = dm.simulate_differentials("state", 300, 21, 40, np.random.default_rng(5))
    for method in dm.METHODS:
        one_by_one = np.mean([dm.dm_test(row, 21, method).p_value < 0.05 for row in d])
        assert dm.rejection_rates(d, 21, (method,))[method] == pytest.approx(one_by_one, abs=1 / 40 + 1e-12)


def test_dm_size_at_h21_with_overlapping_errors():
    """The size check of milestone 5 (section 7.1 of the walkthrough runs it with more samples).

    Two equally good forecasts with overlapping 21-step errors, n = 6,090 (the dev sample at h = 21):
    the primary (fixed-b) test rejects near 5%, Newey-West at lag h-1 does not. Under the
    state-dependent null the M0 rectangular estimator over-rejects and the fixed-b one does not.
    """
    rng = np.random.default_rng(20261028)
    eq = dm.rejection_rates(dm.simulate_differentials("sq_error", 6090, 21, 1000, rng), 21)
    assert 0.03 <= eq[dm.PRIMARY_METHOD] <= 0.075, eq
    assert eq["bartlett"] > 0.075, eq  # the test can tell a biased estimator apart
    state = dm.rejection_rates(dm.simulate_differentials("state", 6090, 21, 1000, rng), 21)
    assert state[dm.PRIMARY_METHOD] <= 0.08 < state["rectangular"], state


def test_holm_by_hand_and_against_statsmodels():
    # p = [0.01, 0.04, 0.03, 0.005], m = 4. Ascending: 0.005*4 = 0.02, 0.01*3 = 0.03,
    # 0.03*2 = 0.06, 0.04*1 = 0.04 -> running max 0.06. Back in order: [0.03, 0.06, 0.06, 0.02].
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(dm.holm(p), [0.03, 0.06, 0.06, 0.02], atol=1e-15)
    assert np.allclose(dm.holm(p), multipletests(p, method="holm")[1], atol=1e-15)
    out = dm.holm([0.02, np.nan, 0.5])  # NaN is not in the family: m = 2 -> [0.04, nan, 0.5]
    assert out[0] == pytest.approx(0.04) and math.isnan(out[1]) and out[2] == pytest.approx(0.5)


def test_loss_series_average_to_the_metric_losses():
    rng = np.random.default_rng(1)
    y, yhat = 15 + 10 * rng.random(50), 15 + 10 * rng.random(50)
    assert dm.loss_series(y, yhat, "sq_error").mean() == pytest.approx(losses.rmse(y, yhat) ** 2, rel=1e-13)
    assert dm.loss_series(y, yhat, "qlike").mean() == pytest.approx(losses.qlike(y, yhat), rel=1e-13)


# --- Mincer-Zarnowitz --------------------------------------------------------------------------

X4 = np.array([1.0, 2.0, 3.0, 4.0])
E4 = np.array([1.0, -1.0, -1.0, 1.0])  # sum 0 and sum(e * x) = 1 - 2 - 3 + 4 = 0: orthogonal to (1, x)


def test_mz_exact_linear_case_white_covariance_by_hand():
    # y = x + e with e orthogonal to (1, x): OLS gives alpha = 0, beta = 1, residuals = e exactly.
    # h = 1, so no lags: S = (1/4) sum g g', g_t = (e_t, e_t x_t), e_t^2 = 1 -> S = Q = X'X/4 = [[1, 2.5], [2.5, 7.5]]
    # V = Q^-1 S Q^-1 / 4 = Q^-1 / 4; det Q = 7.5 - 6.25 = 1.25, Q^-1 = [[6, -2], [-2, 0.8]] -> V = [[1.5, -.5], [-.5, .2]]
    r = mz.mz_regression(X4 + E4, X4, 1, "hansen_hodrick")
    assert (r.alpha, r.beta) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert r.alpha_se == pytest.approx(math.sqrt(1.5), abs=1e-12) and r.beta_se == pytest.approx(math.sqrt(0.2), abs=1e-12)
    assert r.wald_stat == pytest.approx(0.0, abs=1e-20) and r.wald_p == pytest.approx(1.0)
    # shift y by 1: alpha = 1, same V; W = (V^-1)_11 = (4Q)_11 = 4; p = exp(-4/2) for chi-squared(2)
    s = mz.mz_regression(X4 + E4 + 1, X4, 1, "hansen_hodrick")
    assert s.alpha == pytest.approx(1.0, abs=1e-12) and s.wald_stat == pytest.approx(4.0, abs=1e-10)
    assert s.wald_p == pytest.approx(math.exp(-2), abs=1e-12)


def test_mz_hansen_hodrick_lag_one_by_hand():
    # h = 2, one rectangular lag. g = (1,1), (-1,-2), (-1,-3), (1,4).
    # Gamma_1 = (1/4)[g2 g1' + g3 g2' + g4 g3'] = (1/4)[[-1,-1],[-2,-2]] + [[1,2],[3,6]] + [[-1,-3],[-4,-12]]
    #         = (1/4)[[-1,-2],[-3,-8]];  S = [[1, 2.5],[2.5, 7.5]] + Gamma_1 + Gamma_1' = [[0.5, 1.25],[1.25, 3.5]]
    # Q^-1 S Q^-1 = [[2, -0.6], [-0.6, 0.24]] -> V = that / 4 = [[0.5, -0.15], [-0.15, 0.06]]
    r = mz.mz_regression(X4 + E4 + 1, X4, 2, "hansen_hodrick")
    assert r.cov_method == "hansen_hodrick"
    assert r.alpha_se == pytest.approx(math.sqrt(0.5), abs=1e-12) and r.beta_se == pytest.approx(math.sqrt(0.06), abs=1e-12)
    # W = (V^-1)_11 = V_22 / det V = 0.06 / (0.5*0.06 - 0.15^2) = 0.06/0.0075 = 8; p = exp(-4)
    assert r.wald_stat == pytest.approx(8.0, abs=1e-9) and r.wald_p == pytest.approx(math.exp(-4), abs=1e-12)


def test_mz_covariances_equal_statsmodels_hac():
    rng = np.random.default_rng(11)
    x = 20 + np.cumsum(rng.standard_normal(400)) * 0.3
    y = 1 + 0.9 * x + np.convolve(rng.standard_normal(404), np.ones(5), "valid")
    for method, kernel in (("hansen_hodrick", "uniform"), ("newey_west", "bartlett")):
        ours = mz.mz_regression(y, x, 5, method)
        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "kernel": kernel, "use_correction": False})
        assert ours.cov_method == method
        assert (ours.alpha, ours.beta) == pytest.approx(tuple(fit.params), rel=1e-12)
        assert (ours.alpha_se, ours.beta_se) == pytest.approx(tuple(fit.bse), rel=1e-10)


def test_mz_hansen_hodrick_falls_back_when_not_positive_definite():
    # Found by search: on these six points the rectangular lag-2 covariance has a non-positive eigenvalue.
    x, y = np.arange(1.0, 7.0), np.array([-1.0, 0.0, 1.0, 6.0, 6.0, 8.0])
    r = mz.mz_regression(y, x, 3, "hansen_hodrick")
    nw = mz.mz_regression(y, x, 3, "newey_west")
    assert r.cov_method == "bartlett_fallback" and nw.cov_method == "newey_west"
    assert (r.alpha_se, r.beta_se, r.wald_stat) == (nw.alpha_se, nw.beta_se, nw.wald_stat)


def test_mz_size_with_a_rational_forecast_at_h21():
    # A rational forecast of a persistent level (alpha = 0, beta = 1 true), n = 6,090. Hansen-Hodrick is
    # near 5%; Newey-West at lag h-1 is not (the walkthrough, 7.2, runs 1,000 samples).
    rates = mz.simulate_mz(6090, 21, 200, np.random.default_rng(20261028), methods=("hansen_hodrick", "newey_west"))
    assert mz.PRIMARY_METHOD == "hansen_hodrick" and rates[mz.PRIMARY_METHOD] <= 0.09 < rates["newey_west"], rates


# --- calibration --------------------------------------------------------------------------------

D = [date(2020, 1, 1) + timedelta(days=i) for i in range(40)]


def _cal_frame(y_poison=None):
    # Two folds, h = 1. Fold A (cutoff D[10]): origins D[10], D[11]. Fold B (cutoff D[12]): origins D[12], D[13].
    rows = [(D[10], D[11], D[10], 10.0, 12.0), (D[11], D[12], D[10], 20.0, 18.0),
            (D[12], D[13], D[12], 30.0, 27.0), (D[13], D[14], D[12], 40.0, 44.0)]
    frame = pd.DataFrame(rows, columns=["origin_date", "target_date", "cutoff_date", "yhat", "y"])
    frame["v_hat"] = (frame["yhat"] / 100) ** 2
    if y_poison is not None:
        frame.loc[frame.target_date > D[12], "y"] = y_poison
    return frame.assign(model_id="m", target_id="vix_level", horizon=1)


def test_calibration_multiplier_by_hand():
    # At cutoff D[10] no target is dated <= D[10]: 0 pairs -> warm-up, k = 1.
    # At cutoff D[12] the pairs with target <= D[12] are (10, 12) and (20, 18):
    # k = (10*12 + 20*18) / (10^2 + 20^2) = (120 + 360) / 500 = 0.96
    k = calibration.multipliers(_cal_frame(), min_pairs=2)
    assert list(k["n_pairs"]) == [0, 2] and list(k["warm_up"]) == [True, False]
    assert list(k["k"]) == pytest.approx([1.0, 0.96], abs=1e-15)
    out, _ = calibration.calibrate(_cal_frame(), min_pairs=2)
    assert list(out["yhat_cal"]) == pytest.approx([10.0, 20.0, 28.8, 38.4], abs=1e-12)  # 0.96 * 30, 0.96 * 40
    assert np.allclose(out["yhat_cal"], 100 * np.sqrt(out["v_hat_cal"]), rtol=1e-14)
    assert list(out["yhat"]) == [10.0, 20.0, 30.0, 40.0]  # the model's own forecast is untouched


def test_calibration_uses_no_target_dated_after_its_cutoff():
    clean, _ = calibration.calibrate(_cal_frame(), min_pairs=2)
    poisoned, _ = calibration.calibrate(_cal_frame(y_poison=1e6), min_pairs=2)  # targets after D[12] only
    assert list(poisoned["yhat_cal"]) == list(clean["yhat_cal"])


def test_calibration_verdict_follows_rule_6():
    # Amended M5, review A. Rule 6: "helps" needs RMSE lower AND QLIKE lower AND Holm-adjusted p < 0.05.
    # a: all three -> True.  b: QLIKE higher -> False.  c: raw p 0.01 but Holm p 0.2 -> False.  d: RMSE higher -> False.
    cases = {"a": (2.0, 1.9, 0.10, 0.09, 0.01, 0.01), "b": (2.0, 1.9, 0.10, 0.11, 0.01, 0.01),
             "c": (2.0, 1.9, 0.10, 0.09, 0.01, 0.20), "d": (2.0, 2.1, 0.10, 0.09, 0.01, 0.01)}
    metric, tests = [], []
    for m, (rr, rc, qr, qc, p, ph) in cases.items():
        for forecast, rmse, ql in (("mean", rr, qr), ("mean_cal", rc, qc)):
            metric.append({"model_id": m, "target_id": "vix_level", "horizon": 21, "scheme": "all", "forecast": forecast,
                           "rmse": rmse, "qlike": ql})
        tests.append({"family": "calibration", "primary": True, "loss": "sq_error", "model_a": m, "model_b": m + "+cal",
                      "target_id": "vix_level", "horizon": 21, "mean_d": 0.1, "p_value": p, "p_holm": ph})
        tests.append({**tests[-1], "loss": "qlike", "p_value": 0.9, "p_holm": 0.9})  # only sq_error counts
    out = report.calibration_summary(pd.DataFrame(metric), pd.DataFrame(tests)).set_index("model_id")
    assert out["helps"].to_dict() == {"a": True, "b": False, "c": False, "d": False}


def test_holdout_check_by_hand():
    # Pre-registered in M5 (review A): the dev-best model by RMSE (and QLIKE on rv21) must beat rw on the holdout.
    def metric(rows):
        return pd.DataFrame([{"model_id": m, "target_id": t, "horizon": h, "scheme": "all", "forecast": f, "rmse": r, "qlike": q}
                             for m, t, h, f, r, q in rows])
    dev = metric([("har_log", "vix_level", 21, "mean", 4.5, 0.10), ("rw", "vix_level", 21, "mean", 4.9, 0.12),
                  ("har_log", "vix_level", 21, "mean_cal", 4.0, 0.05),  # a variant is never the "dev best":
                  ("rw", "vix_level", 21, "mean_cal", 3.0, 0.05),       # not even rw's calibrated forecast
                  ("garch", "rv21", 21, "mean", 4.0, 0.14), ("har_rv", "rv21", 21, "mean", 4.1, 0.13),
                  ("rw", "rv21", 21, "mean", 4.6, 0.20),
                  ("rw", "vix_level", 1, "mean", 1.1, 0.01), ("har_log", "vix_level", 1, "mean", 1.2, 0.01)])
    hold = metric([("har_log", "vix_level", 21, "mean", 6.0, 0.2), ("rw", "vix_level", 21, "mean", 5.0, 0.2),
                   ("garch", "rv21", 21, "mean", 5.0, 0.30), ("har_rv", "rv21", 21, "mean", 9.0, 0.10),
                   ("rw", "rv21", 21, "mean", 6.0, 0.25),
                   ("rw", "vix_level", 1, "mean", 1.0, 0.01), ("har_log", "vix_level", 1, "mean", 1.0, 0.01)])
    out = report.holdout_check(dev, hold).set_index(["target_id", "horizon", "loss"])
    # vix_level 21: dev best har_log (4.5 < 4.9), holdout 6.0 > rw's 5.0 -> did not hold up
    assert out.loc[("vix_level", 21, "rmse"), "dev_best"] == "har_log" and out.loc[("vix_level", 21, "rmse"), "held_up"] is False
    assert out.loc[("vix_level", 21, "rmse"), ["dev_best_value", "dev_rw_value"]].tolist() == [4.5, 4.9]  # mean rows only
    # rv21 RMSE: dev best garch (4.0), holdout 5.0 < 6.0 -> held up; rv21 QLIKE: dev best har_rv (0.13), 0.10 < 0.25
    assert (out.loc[("rv21", 21, "rmse"), "dev_best"], out.loc[("rv21", 21, "rmse"), "held_up"]) == ("garch", True)
    assert (out.loc[("rv21", 21, "qlike"), "dev_best"], out.loc[("rv21", 21, "qlike"), "held_up"]) == ("har_rv", True)
    # vix_level 1: rw is the dev best: nothing to confirm
    assert out.loc[("vix_level", 1, "rmse"), "dev_best"] == "rw" and out.loc[("vix_level", 1, "rmse"), "held_up"] is None
    assert ("vix_level", 21, "qlike") not in out.index  # QLIKE is checked on rv21 only


def test_calibration_waits_for_min_pairs():
    k = calibration.multipliers(_cal_frame(), min_pairs=3)
    assert list(k["k"]) == [1.0, 1.0] and list(k["warm_up"]) == [True, True]


# --- regimes and the report's tables -------------------------------------------------------------


def _world(n_dev=30, n_hold=10, hs_index=40):
    """Two models on vix_level h = 1 over consecutive days; holdout_start = D2[hs_index]."""
    days = [date(2021, 1, 1) + timedelta(days=i) for i in range(n_dev + n_hold + 20)]
    hs = days[hs_index]
    rng = np.random.default_rng(7)
    origins = days[hs_index - n_dev : hs_index + n_hold]
    rows = []
    for i, o in enumerate(origins):
        cutoff = origins[(i // 5) * 5]
        for model, noise in (("rw", 1.0), ("har_log", 0.5), ("hist_mean", 2.0)):
            yhat = 20 + noise * rng.standard_normal()
            rows.append({"model_id": model, "target_id": "vix_level", "horizon": 1, "origin_date": o,
                         "target_date": days[days.index(o) + 1], "cutoff_date": cutoff, "yhat": yhat,
                         "v_hat": (yhat / 100) ** 2, "yhat_naive": np.nan})
    targets = pd.DataFrame({"target_id": "vix_level", "date": days, "y": 20 + rng.standard_normal(len(days))})
    targets["v"] = (targets["y"] / 100) ** 2
    labels = pd.DataFrame({"date": days, "scheme": "vix_level", "regime": ["low" if i % 3 else "high" for i in range(len(days))],
                           "driver": 0.0})
    return pd.DataFrame(rows), targets, labels, hs


def test_regime_cells_label_at_the_origin_and_suppress_small_cells():
    forecasts, targets, labels, hs = _world()
    scored = regimes.attach(report.scored_rows(report.attach_targets(forecasts, targets), hs, "dev"), labels)
    # 30 dev origins at days 10..39; the origin of day 39 targets day 40 = holdout_start: not scored -> 29.
    # "high" where the day index is a multiple of 3: days 12, 15, ..., 36 -> 9 origins; "low" 20.
    assert scored.groupby("model_id").size().to_dict() == {"har_log": 29, "hist_mean": 29, "rw": 29}
    by_date = targets.set_index("date")["y"]
    assert np.array_equal(scored["y"].to_numpy(), scored["target_date"].map(by_date).to_numpy())  # y at the TARGET date
    metric = regimes.metric_rows(scored, "dev", min_cell_n=10)
    rw = metric[(metric.model_id == "rw") & (metric.forecast == "mean")].set_index(["scheme", "regime"])
    assert rw.loc[("all", "all"), "n"] == 29 and rw.loc[("vix_level", "low"), "n"] == 20
    assert rw.loc[("vix_level", "high"), "n"] == 9 and bool(rw.loc[("vix_level", "high"), "suppressed"])
    assert math.isnan(rw.loc[("vix_level", "high"), "rmse"])  # suppressed: null, with its n kept
    cell = scored[(scored.model_id == "rw") & (scored.regime_vix_level == "low")]
    assert rw.loc[("vix_level", "low"), "rmse"] == pytest.approx(losses.rmse(cell["y"], cell["yhat"]), rel=1e-15)


def test_evaluate_dev_and_the_synthetic_holdout_path():
    forecasts, targets, labels, hs = _world()
    dev = report.evaluate(forecasts, targets, labels, hs, "dev", min_cell_n=1, min_pairs=5)
    hold = report.evaluate(forecasts, targets, labels, hs, "holdout", min_cell_n=1, min_pairs=5)
    assert dev["_scored"].origin_date.max() < hs and dev["_scored"].target_date.max() < hs
    assert hold["_scored"].origin_date.min() >= hs and len(hold["_scored"]) == 3 * 10
    for tables, period in ((dev, "dev"), (hold, "holdout")):
        assert set(tables["metric"].period) == {period} and set(tables["dm_test"].period) == {period}
        assert {"mean", "mean_cal"} <= set(tables["metric"].forecast)  # the variant is extra, never a replacement
    d = dev["dm_test"]
    # three pairs + three (model, model+cal) rows, per loss, per method
    assert len(d) == (3 + 3) * len(dm.LOSSES) * len(dm.METHODS) and d.primary.sum() == 6 * len(dm.LOSSES)
    for _, family in d.groupby(["loss", "method", "family"]):  # Holm within each family of three tests
        assert len(family) == 3
        assert np.allclose(family["p_holm"], dm.holm(family["p_value"].to_numpy()), equal_nan=True)
    assert not d.descriptive_only.any() and hold["dm_test"].descriptive_only.all() and hold["mz_regression"].descriptive_only.all()
    hc = hold["holdout_check"]  # the confirmatory check exists on the holdout path only, chosen on dev rows
    assert "holdout_check" not in dev and list(hc.loss) == ["rmse"]
    dev_mean = dev["metric"][(dev["metric"].scheme == "all") & (dev["metric"].forecast == "mean")].set_index("model_id")["rmse"]
    hold_mean = hold["metric"][(hold["metric"].scheme == "all") & (hold["metric"].forecast == "mean")].set_index("model_id")["rmse"]
    assert hc.dev_best.iloc[0] == dev_mean.idxmin()
    assert hc.held_up.iloc[0] == (None if dev_mean.idxmin() == "rw" else bool(hold_mean[dev_mean.idxmin()] < hold_mean["rw"]))
    pair = d[(d.family == "pairs") & (d.loss == "sq_error") & (d.method == "rectangular") & (d.model_b == "rw")].iloc[0]
    s = dev["_scored"].sort_values("origin_date")
    la = dm.loss_series(s[s.model_id == "har_log"].y, s[s.model_id == "har_log"].yhat, "sq_error")
    lb = dm.loss_series(s[s.model_id == "rw"].y, s[s.model_id == "rw"].yhat, "sq_error")
    assert (pair.model_a, pair.model_b) == ("har_log", "rw")  # model_a < model_b, d = L_a - L_b
    assert pair.mean_d == pytest.approx(float(np.mean(la - lb)), rel=1e-12)
    assert pair.lower_loss == ("har_log" if np.mean(la) < np.mean(lb) else "rw")  # positive mean_d: model_a is worse
    assert pair.dm_stat == pytest.approx(dm.dm_test(la - lb, 1, "rectangular").dm_stat, rel=1e-12)


def test_report_refuses_the_real_holdout_before_reading_anything(tmp_path, monkeypatch, capsys):
    from ivf.data import build_interim

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("read_interim must not be called")

    monkeypatch.setattr(build_interim, "read_interim", boom)
    assert report.main(["report", "--period", "holdout"]) == report.EXIT_USAGE
    assert "refusing --period holdout on real data" in capsys.readouterr().err
    assert not (tmp_path / "implied-vol-forecasting" / "processed").exists()


def test_report_command_on_the_synthetic_pipeline(tmp_path, monkeypatch, capsys):
    from ivf.data import cli as data_cli
    from ivf.models import cli as models_cli

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    for step in ("fetch", "build-interim", "check"):
        assert data_cli.main(["--synthetic", step]) == 0
    assert models_cli.main(["--synthetic", "baselines", "--symbols", "VIX", "--models", "rw", "har_log"]) == 0
    capsys.readouterr()
    from ivf.features import build

    read = {"targets": [], "labels": []}
    real_targets, real_labels = build.build_target_values, build.build_regime_labels
    monkeypatch.setattr(build, "build_target_values", lambda dates, iv, price: read["targets"].append(max(dates)) or real_targets(dates, iv, price))
    monkeypatch.setattr(build, "build_regime_labels", lambda dates, iv, edges: read["labels"].append(max(dates)) or real_labels(dates, iv, edges))
    assert report.main(["--synthetic", "report"]) == 0
    # dev: targets AND regime labels built from sessions before holdout_start only (amended M5, review A: labels)
    assert read["targets"] and read["labels"] and max(read["targets"] + read["labels"]) < date(2024, 7, 1)
    out = capsys.readouterr().out
    assert "sessions ending 2024-06-28" in out and "no later value enters an output" in out and "holdout_looks 0" in out
    assert "vix_level 1: 6110" in out  # every dev origin whose target is before holdout_start (M4's count)
    processed = tmp_path / "implied-vol-forecasting" / "synthetic" / "processed"
    (folder,) = list(processed.glob("run_id=*/evaluation/period=dev"))
    meta = json.loads((folder / "report_meta.json").read_text())
    assert meta["holdout_looks"] == 0 and meta["period"] == "dev" and meta["dm_primary"] == dm.PRIMARY_METHOD
    d = pd.read_parquet(folder / "dm_test.parquet")
    assert len(d) == 3 * (1 + 2) * len(dm.LOSSES) * len(dm.METHODS)  # 3 horizons x (1 pair + 2 calibration) x losses x methods
    metric = pd.read_parquet(folder / "metric.parquet")
    assert {"all", "vix_level", "term_structure"} == set(metric.scheme)
    z = pd.read_parquet(folder / "mz_regression.parquet")
    assert set(z.space) == {"level"} and len(z) == 2 * 2 * 3 * len(mz.COV_METHODS)  # (rw, har_log) x (mean, mean_cal) x 3 h
    # Poison every interim value dated on or after holdout_start (x100): every dev output is unchanged
    # (amended M5, review A: covers targets, regime labels and anything else that might read them).
    from ivf.data import build_interim

    clean = {f.name: pd.read_parquet(f) for f in folder.glob("*.parquet")}
    real_read = build_interim.read_interim

    def poisoned(root):
        iv, price, rep_ = real_read(root)
        for frame in (iv, price):
            late = pd.to_datetime(frame["date"]).dt.date >= date(2024, 7, 1)
            cols = [c for c in frame.columns if pd.api.types.is_float_dtype(frame[c])]
            assert late.sum() > 0 and cols
            frame.loc[late, cols] = frame.loc[late, cols] * 100
        return iv, price, rep_

    monkeypatch.setattr(build_interim, "read_interim", poisoned)
    assert report.main(["--synthetic", "report"]) == 0
    for name, frame in clean.items():
        assert pd.read_parquet(folder / name).equals(frame), name
    monkeypatch.setattr(build_interim, "read_interim", real_read)
    capsys.readouterr()
    # The synthetic holdout path runs the same code; the model steps make no holdout forecast, so it stops there.
    assert report.main(["--synthetic", "report", "--period", "holdout"]) == report.EXIT_INPUT
    assert "no scored rows" in capsys.readouterr().err
