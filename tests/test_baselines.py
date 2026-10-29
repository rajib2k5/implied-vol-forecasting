"""ivf.models: the milestone 2 baselines and the model registry, on hand-worked numbers."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ivf.models import baselines, registry
from ivf.models.baselines import Arima, Fit, Har, HistoricalMean, RandomWalk
from ivf.protocol import ProtocolError

H = (1, 5, 21)


# --- the back-transform ----------------------------------------------------------------


def test_lognormal_forecast_reports_the_mean_as_yhat_and_the_median_as_naive():
    # mu = ln 20, s2 = 0.04 (s = 0.2):
    #   median exp(mu)         = 20
    #   mean   exp(mu + s2/2)  = 20 * exp(0.02) = 20.404026800535114
    #   80%    20 * exp(-/+ 1.2815515655446004 * 0.2) = 15.47803559320025 .. 25.8430727589053
    fc = baselines.lognormal_forecast(np.array([math.log(20)]), np.array([0.04]))
    assert fc.yhat_naive[0] == pytest.approx(20.0, rel=1e-15)
    assert fc.yhat[0] == pytest.approx(20.404026800535114, rel=1e-15)
    assert fc.pi80_lo[0] == pytest.approx(15.47803559320025, rel=1e-14)
    assert fc.pi80_hi[0] == pytest.approx(25.8430727589053, rel=1e-14)


# --- random walk and historical means -------------------------------------------------------


def test_random_walk_forecasts_the_close_at_the_origin_for_every_horizon():
    y = np.array([10.0, 11.0, 12.0])
    model = RandomWalk()
    out = model.forecast(model.fit(y, H, None), y, np.array([1, 2]), H)
    for h in H:
        assert out[h].yhat.tolist() == [11.0, 12.0] and out[h].yhat_naive.tolist() == [11.0, 12.0]


def test_expanding_mean_uses_every_close_up_to_the_cutoff_and_skips_gaps():
    # y = [10, 12, 14, nan, 20]: (10 + 12 + 14 + 20) / 4 = 56 / 4 = 14, from 4 closes.
    y = np.array([10.0, 12.0, 14.0, np.nan, 20.0])
    fit = HistoricalMean("hist_mean", None).fit(y, H, None)
    assert fit.params == {"mean": 14.0}
    assert (fit.n_train, fit.train_first, fit.train_last_target) == (4, 0, 4)
    out = HistoricalMean("hist_mean", None).forecast(fit, y, np.array([4]), H)
    assert all(out[h].yhat.tolist() == [14.0] for h in H)


def test_window_mean_uses_the_last_w_sessions_of_the_calendar():
    # Window 3 is the last 3 sessions [14, nan, 20]: (14 + 20) / 2 = 17, from 2 closes, first at index 2.
    # A window counted in closes instead of sessions would take [12, 14, 20] = 15.333.
    y = np.array([10.0, 12.0, 14.0, np.nan, 20.0])
    fit = HistoricalMean("hist_mean_3", 3).fit(y, H, None)
    assert fit.params == {"mean": 17.0} and (fit.n_train, fit.train_first) == (2, 2)
    # Window 2 is [nan, 20] = 20.
    assert HistoricalMean("hist_mean_2", 2).fit(y, H, None).params == {"mean": 20.0}


# --- HAR ------------------------------------------------------------------------------------


def exact_har_series(h: int, coefs, n: int = 100, seed: int = 0) -> np.ndarray:
    """ln y with x_{t+h} = b0 + b_d x_t + b_w mean(x_{t-4..t}) + b_m mean(x_{t-21..t}) exactly, for t >= 21.

    The first 21 + h values are random; every later one is computed from values dated at least h
    sessions earlier, so the relation holds for every complete HAR row. OLS must recover the
    coefficients exactly (to rounding) and a residual variance of zero.
    """
    b0, bd, bw, bm = coefs
    x = np.empty(n)
    x[: 21 + h] = 3.0 + 0.3 * np.random.default_rng(seed).standard_normal(21 + h)
    for t in range(21, n - h):
        x[t + h] = b0 + bd * x[t] + bw * x[t - 4 : t + 1].mean() + bm * x[t - 21 : t + 1].mean()
    return x


@pytest.mark.parametrize("h", [1, 5, 21])
def test_har_recovers_an_exact_h_step_relation(h):
    # Slopes sum to 0.6 + 0.3 + 0.12 = 1.02, so the series keeps moving and the design stays full rank.
    coefs = (0.1, 0.6, 0.3, 0.12)
    x = exact_har_series(h, coefs)
    fit = Har().fit(np.exp(x), (h,), None)
    p = fit.params[f"h{h}"]
    assert [p["b0"], p["b_d"], p["b_w"], p["b_m"]] == pytest.approx(list(coefs), abs=1e-7)
    assert p["s2"] < 1e-12
    # Pairs (tau, tau + h) for tau = 21 .. 99 - h: n = 100 - h - 21; the last target is index 99.
    assert p["n"] == 100 - h - 21 and fit.train_last_target == 99 and fit.train_first == 0


def test_har_s2_is_the_rss_of_its_own_forecasts_over_n_minus_4_per_horizon():
    # Fit h = 1 and h = 5 together on a noisy AR(1), then forecast every training origin with the
    # forecast path (a different code path from the fit's residuals). For each h:
    #   pairs: tau = 21 .. 299 - h, so n = 300 - h - 21 (h=1: 278, h=5: 274)
    #   s2 = sum((x_{tau+h} - mu_tau)^2) / (n - 4)
    # Dividing by n, or forecasting h = 5 with the h = 1 slopes, breaks the equality.
    y = simulate(0.9, 300, 4)
    x = np.log(y)
    fit = Har().fit(y, (1, 5), None)
    for h in (1, 5):
        origins = np.arange(21, 300 - h)
        fc = Har().forecast(fit, y, origins, (1, 5))[h]
        rss = float(np.sum((x[origins + h] - fc.mu_log) ** 2))
        n = 300 - h - 21
        assert fit.params[f"h{h}"]["n"] == n
        assert fit.params[f"h{h}"]["s2"] == pytest.approx(rss / (n - 4), rel=1e-10)
        assert fc.s2_log[0] == fit.params[f"h{h}"]["s2"]


def test_har_forecast_on_hand_worked_regressors():
    # x = 3.0 on 21 sessions, then 3.5 at t = 21:
    #   x_t = 3.5, xw = (4*3 + 3.5)/5 = 3.1, xm = (21*3 + 3.5)/22 = 66.5/22 = 3.0227272727
    #   mu = 0.1 + 0.5*3.5 + 0.3*3.1 + 0.15*3.0227272727 = 3.2334090909090913
    #   yhat = exp(mu + 0.01/2) = 25.493132212989515; naive exp(mu) = 25.36598468563353
    x = np.r_[np.full(21, 3.0), 3.5]
    fit = Fit(params={"h1": {"b0": 0.1, "b_d": 0.5, "b_w": 0.3, "b_m": 0.15, "s2": 0.01, "n": 99}},
              hyperparams={}, n_train=99, train_first=0, train_last_target=21)
    fc = Har().forecast(fit, np.exp(x), np.array([21]), (1,))[1]
    assert fc.mu_log[0] == pytest.approx(3.2334090909090913, rel=1e-14)
    assert fc.yhat[0] == pytest.approx(25.493132212989515, rel=1e-13)
    assert fc.yhat_naive[0] == pytest.approx(25.36598468563353, rel=1e-13)
    assert fc.s2_log[0] == 0.01


def test_har_design_is_nan_until_the_monthly_window_is_full():
    design = baselines.har_design(np.arange(30, dtype=float))
    assert np.isnan(design[20, 3]) and not np.isnan(design[21]).any()
    # At t = 21: x = 21, week mean of 17..21 = 19, month mean of 0..21 = 10.5
    assert design[21].tolist() == [1.0, 21.0, 19.0, 10.5]


# --- ARIMA ----------------------------------------------------------------------------------


def ar1_fit() -> Fit:
    # AR(1) with a constant: x_t = 0.3 + 0.9 x_{t-1} + e, Var e = 0.01; its mean is 0.3/(1 - 0.9) = 3.
    return Fit(params={"order": [1, 0, 0], "trend": "c", "names": ["intercept", "ar.L1", "sigma2"], "values": [0.3, 0.9, 0.01]},
               hyperparams={}, n_train=40, train_first=0, train_last_target=39)


def test_arima_forecast_matches_the_ar1_closed_form_and_ignores_data_after_the_origin():
    # Origin t = 30 with x_t = 3.5; the values after it (2.0) must not move the forecast.
    #   mu_h = 3 + 0.9^h * 0.5:  h=1 3.45; h=5 3 + 0.59049*0.5 = 3.295245; h=21 3 + 0.10941898913*0.5 = 3.0547094945657562
    #   s2_h = 0.01 * (1 - 0.9^(2h)) / (1 - 0.81):  h=1 0.01; h=5 0.0342800821; h=21 0.05200144656933884
    x = np.full(40, 3.0)
    x[30], x[31:] = 3.5, 2.0
    out = Arima().forecast(ar1_fit(), np.exp(x), np.array([30]), H)
    expected = {1: (3.45, 0.01), 5: (3.295245, 0.0342800821), 21: (3.0547094945657562, 0.05200144656933884)}
    for h, (mu, s2) in expected.items():
        assert out[h].mu_log[0] == pytest.approx(mu, rel=1e-9)
        assert out[h].s2_log[0] == pytest.approx(s2, rel=1e-9)
        assert out[h].yhat[0] == pytest.approx(math.exp(mu + s2 / 2), rel=1e-9)


def simulate(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 3.0
    for i in range(1, n):
        x[i] = 3.0 * (1 - phi) + phi * x[i - 1] + 0.05 * rng.standard_normal() if phi < 1 else x[i - 1] + 0.05 * rng.standard_normal()
    return np.exp(x)


def test_arima_default_d_rule_is_d0_even_on_a_random_walk():
    # Amended M2, review A: d = 0 by default, an ARMA with a constant on ln y. KPSS is not run.
    walk = Arima(max_p=1, max_q=1).fit(simulate(1.0, 1500, 1), H, None)
    assert walk.params["order"][1] == 0 and walk.params["trend"] == "c"
    assert walk.hyperparams["d_rule"] == "d0" and walk.hyperparams["kpss_p"] is None
    with pytest.raises(ValueError, match="d_rule"):
        Arima(d_rule="adf")


def test_arima_kpss_option_takes_d_from_kpss_and_p_q_from_the_smallest_aic():
    # d_rule "kpss": a random walk makes KPSS reject level stationarity, so d = 1 and no constant.
    walk = Arima(max_p=1, max_q=1, d_rule="kpss").fit(simulate(1.0, 1500, 1), H, None)
    assert walk.params["order"][1] == 1 and walk.params["trend"] == "n" and walk.hyperparams["kpss_p"] < 0.05
    # A stationary AR(1) with phi = 0.5: KPSS does not reject, so d = 0 with a constant.
    ar = Arima(max_p=1, max_q=1, d_rule="kpss").fit(simulate(0.5, 1500, 2), H, None)
    assert ar.params["order"][1] == 0 and ar.params["trend"] == "c" and ar.hyperparams["kpss_p"] >= 0.05
    # The chosen order is the smallest AIC among the converged candidates, and every candidate is recorded.
    candidates = ar.params["arima_candidates"]
    assert [c["order"] for c in candidates] == [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
    best = min((c for c in candidates if c["converged"]), key=lambda c: c["aic"])
    assert ar.params["order"] == best["order"] and ar.aic == best["aic"] and ar.fallback == "none"


def test_kpss_runs_on_the_longest_gap_free_run():
    # Gap policy G2 (amended M2, review A: no test gave this path a gap). x = a random walk of 300,
    # one missing value, then a stationary AR(1) (phi = 0.5) of 500. The longest run is the AR(1):
    # n = 500 and KPSS does not reject (p clipped at 0.10). On the random walk alone p is 0.01, and
    # statsmodels refuses the whole series (NaN).
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.standard_normal(300))
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.5 * ar[i - 1] + rng.standard_normal()
    p, n = baselines.kpss_p_value(np.r_[walk, np.nan, ar])
    assert (p, n) == (0.1, 500)
    assert baselines.kpss_p_value(walk) == (0.01, 300)


def test_arima_treats_a_missing_close_as_missing_not_as_the_previous_value():
    # The Kalman filter's missing-value path (G2/G3). AR(1) mean 3, phi 0.9, sigma2 0.01, with
    # x_29 = 3.5 and x_30 missing. At origin 30 the state is the 1-step prediction from 29, so the
    # h-step forecast is the (h+1)-step one from x_29:
    #   mu_h = 3 + 0.9^(h+1)*0.5:  h=1 3 + 0.81*0.5 = 3.405; h=5 3 + 0.531441*0.5 = 3.2657205;
    #                              h=21 3 + 0.0984770902*0.5 = 3.049238545109181
    #   s2_h = 0.01*(1 - 0.9^(2(h+1)))/0.19:  h=1 0.01*0.3439/0.19 = 0.0181; h=5 0.037766866501;
    #                                          h=21 0.05212117172116446
    # Filling x_30 with 3.5 (the previous value) would give the h-step numbers from 3.5 instead.
    x = np.full(40, 3.0)
    x[29], x[30] = 3.5, np.nan
    out = Arima().forecast(ar1_fit(), np.exp(x), np.array([30]), H)
    expected = {1: (3.405, 0.0181), 5: (3.2657205, 0.037766866501), 21: (3.049238545109181, 0.05212117172116446)}
    for h, (mu, s2) in expected.items():
        assert out[h].mu_log[0] == pytest.approx(mu, rel=1e-9)
        assert out[h].s2_log[0] == pytest.approx(s2, rel=1e-9)


def test_arima_fits_through_missing_training_closes():
    # Core gaps in burn-in (G3): five missing closes in the training data are skipped by the filter,
    # not filled; the fit converges and counts the 595 closes it used.
    y = simulate(0.5, 600, 5)
    y[[50, 51, 200, 333, 400]] = np.nan
    fit = Arima(max_p=1, max_q=0).fit(y, H, None)
    assert fit.fallback == "none" and fit.converged and fit.n_train == 595 and fit.train_first == 0


def test_arima_first_fit_falls_back_to_a_closed_form_random_walk(monkeypatch):
    # Every candidate fails. x alternates 3.1, 2.9, ...: each step is +/-0.2, so the random walk's
    # variance is mean(dx^2) = 0.04.
    def fail(self, x, order, start):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(Arima, "_fit_one", fail)
    y = np.exp(3.0 + 0.1 * (-1.0) ** np.arange(60))
    fit = Arima().fit(y, H, None)
    assert fit.fallback == "rw_log" and fit.converged is False
    assert fit.params["order"] == [0, 1, 0] and fit.params["values"] == pytest.approx([0.04], rel=1e-12)
    assert all(c["error"] == "LinAlgError" for c in fit.params["arima_candidates"])


def test_arima_keeps_the_previous_fold_when_no_candidate_converges(monkeypatch):
    class Result:
        aic, llf, params, mle_retvals = -1.0, 1.0, np.array([0.1]), {"converged": False}

        class model:
            param_names = ["sigma2"]

    monkeypatch.setattr(Arima, "_fit_one", lambda self, x, order, start: (Result(), False))
    previous = ar1_fit()
    fit = Arima().fit(np.exp(3.0 + 0.1 * (-1.0) ** np.arange(60)), H, previous)
    assert fit.fallback == "previous_fold"
    assert fit.params["order"] == [1, 0, 0] and fit.params["values"] == [0.3, 0.9, 0.01]


# --- the registry ---------------------------------------------------------------------------


def test_the_committed_registry_names_the_five_baselines_and_their_settings():
    defs, settings = registry.load_models()
    assert [d.model_id for d in defs] == ["rw", "hist_mean", "hist_mean_252", "arima_log", "har_log"]
    assert {d.model_id for d in defs if d.fit_space == "log_level"} == {"arima_log", "har_log"}
    assert settings.symbols[0] == "VIX" and settings.hist_window == 252 and (settings.max_p, settings.max_q, settings.d_rule) == (2, 2, "d0")
    assert [m.model_id for m in baselines.build_models(settings)] == [d.model_id for d in defs]
    assert registry.target_id_for("VIX") == "vix_level" and registry.target_id_for("VIX3M") == "vix3m_level"


def test_the_registry_refuses_an_unknown_key(tmp_path):
    from ivf import config

    text = (config.configs_dir() / "models.yaml").read_text().replace("hist_window: 252", "hist_window: 252\n  hist_windw: 5")
    (tmp_path / "models.yaml").write_text(text)
    with pytest.raises(ProtocolError, match="unknown keys"):
        registry.load_models(tmp_path / "models.yaml")


def test_fit_params_are_json_ready():
    json.dumps(Har().fit(simulate(0.5, 200, 3), H, None).params, allow_nan=False)
