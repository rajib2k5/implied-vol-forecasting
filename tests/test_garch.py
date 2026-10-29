"""ivf.models.garch: the GARCH-family recursions, the 21-session variance aggregation, the innovation rule,
ARIMA-GARCH on changes of ln VIX, and the category-error demonstration (milestone 3).

Hand-worked values carry their arithmetic in comments; where arch is the oracle (its own recursion and
analytic forecast), the test says so. Synthetic data only, no network.
"""

from __future__ import annotations

import itertools
import math
from datetime import date

import numpy as np
import pytest

from ivf.data import synthetic
from ivf.features import build
from ivf.models import garch
from ivf.models.baselines import Fit

pytestmark = pytest.mark.usefixtures("no_network")


def returns(n: int, seed: int, df: float | None = None) -> np.ndarray:
    """100 x a GARCH(1,1) return series with the synthetic market's parameters (percent); t(df) shocks if df."""
    p = synthetic.SyntheticParams()
    rng = np.random.default_rng(seed)
    if df is None:
        r, _ = synthetic.simulate_garch11(n + 1, p.omega, p.alpha, p.beta, rng)
        return 100 * r[1:]
    z = rng.standard_t(df, n) / math.sqrt(df / (df - 2))  # unit variance
    h, out = p.omega / (1 - p.alpha - p.beta), np.empty(n)
    for i in range(n):
        out[i] = math.sqrt(h) * z[i]
        h = p.omega + p.alpha * out[i] ** 2 + p.beta * h
    return 100 * out


# --- the recursion and the backcast ---------------------------------------------------------------


def test_backcast_is_arch_s_weighted_mean_of_the_first_75_squares():
    from arch.univariate import GARCH

    # e = (1, 3): weights (1, 0.94)/1.94, so (1*1 + 0.94*9)/1.94 = 9.46/1.94 = 4.876289.
    assert garch.backcast(np.array([1.0, 3.0])) == pytest.approx(4.876289, abs=1e-6)
    e = returns(300, 1)
    assert garch.backcast(e) == pytest.approx(GARCH().backcast(e), rel=1e-14)  # arch as the oracle
    assert garch.backcast(e) == garch.backcast(np.r_[e[:75], 50 * e[75:]])  # only the first 75 values count


def test_garch_and_gjr_recursions_by_hand():
    # GARCH omega 0.1, alpha 0.1, beta 0.8, backcast 1, e = (2, -1):
    #   s2[0] = 0.1 + 0.1*1 + 0.8*1   = 1.0   (lagged terms are the backcast)
    #   s2[1] = 0.1 + 0.1*4 + 0.8*1.0 = 1.3
    #   s2[2] = 0.1 + 0.1*1 + 0.8*1.3 = 1.24  (the one-step forecast past the sample)
    g = garch.filter_variance("garch", {"omega": 0.1, "alpha": 0.1, "beta": 0.8}, np.array([2.0, -1.0]), 1.0)
    np.testing.assert_allclose(g, [1.0, 1.3, 1.24], rtol=1e-14)
    # GJR adds gamma 0.2 on negative shocks; at i = 0 arch uses gamma * backcast / 2:
    #   s2[0] = 0.1 + 0.1*1 + 0.2*0.5 + 0.8*1     = 1.1
    #   s2[1] = 0.1 + 0.1*4 + 0     + 0.8*1.1     = 1.38   (e = 2 > 0: no gamma term)
    #   s2[2] = 0.1 + 0.1*1 + 0.2*1 + 0.8*1.38    = 1.504  (e = -1 < 0)
    j = garch.filter_variance("gjr", {"omega": 0.1, "alpha": 0.1, "gamma": 0.2, "beta": 0.8}, np.array([2.0, -1.0]), 1.0)
    np.testing.assert_allclose(j, [1.1, 1.38, 1.504], rtol=1e-14)


@pytest.mark.parametrize("kind", garch.KINDS)
@pytest.mark.parametrize("dist", garch.DISTS)
def test_our_recursion_reproduces_arch_in_sample_and_its_one_step_forecast(kind, dist):
    from arch.univariate import ZeroMean

    e = returns(1500, 7)
    cand, resid = garch.fit_candidate(kind, e, dist, garch.backcast(e))
    assert cand.ok and cand.filter_max_rel_diff < 1e-9
    model = ZeroMean(e, rescale=False)
    model.volatility, model.distribution = garch._vol_process(kind), garch._distribution(dist)
    res = model.fit(disp="off", backcast=garch.backcast(e))
    ours = garch.filter_variance(kind, cand.vol, resid, garch.backcast(e))
    np.testing.assert_allclose(ours[:-1], res.conditional_volatility**2, rtol=1e-9)
    # The value past the sample is arch's own one-step forecast from the last observation.
    assert ours[-1] == pytest.approx(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0], rel=1e-9)


# --- the multi-step path and the aggregation to rv21 ---------------------------------------------


def test_garch_21_session_aggregate_by_hand():
    # omega 0.02, alpha 0.08, beta 0.90: long-run variance 0.02/(1 - 0.98) = 1 (percent^2), and
    # sigma2_{t+k} = 1 + 0.98**(k-1) * (s1 - 1). With s1 = 2:
    #   sum_{k=1}^{21} = 21 + (1 - 0.98**21)/0.02 = 21 + (1 - 0.654256)/0.02 = 21 + 17.287209 = 38.287209
    #   G = (252/21) * 38.287209 / 100**2 = 12 * 0.0038287209 = 0.045944651;  yhat = 100*sqrt(G) = 21.434703
    path = garch.variance_path("garch", {"omega": 0.02, "alpha": 0.08, "beta": 0.90}, np.array([2.0]))
    assert path.shape == (1, 21) and path[0, 0] == 2.0 and path[0, 1] == pytest.approx(0.02 + 0.98 * 2, rel=1e-14)
    assert path.sum() == pytest.approx(38.287209, abs=1e-6)
    g = garch.aggregate(path)
    assert g[0] == pytest.approx(0.045944651, abs=1e-9) and 100 * math.sqrt(g[0]) == pytest.approx(21.434703, abs=1e-6)


def test_a_constant_1_percent_variance_aggregates_to_rv21_of_twenty_one_1_percent_returns():
    # The units check against the target's own definition (ADR 0001 4.2): 21 returns of 0.01 have
    # rv21 = 100*sqrt(12 * 21 * 0.0001) = 15.874508; a constant sigma2 of 1 percent^2 a day must give
    # G = 12 * 21 * 1 / 10**4 = 0.0252, the same number in variance.
    rv = build.rv_trailing(np.full(21, 0.01))[-1]
    g = garch.aggregate(garch.variance_path("garch", {"omega": 1.0, "alpha": 0.0, "beta": 0.0}, np.array([1.0])))
    assert g[0] == pytest.approx(0.0252, rel=1e-14) and 100 * math.sqrt(g[0]) == pytest.approx(rv, rel=1e-14) == pytest.approx(15.874508, abs=1e-6)


def test_gjr_persistence_uses_half_of_gamma():
    # alpha 0.05, gamma 0.10, beta 0.85: alpha + gamma/2 + beta = 0.95, so sigma2_{t+2} = 0.1 + 0.95*2 = 2.0
    p = {"omega": 0.1, "alpha": 0.05, "gamma": 0.10, "beta": 0.85}
    assert garch.persistence("gjr", p) == pytest.approx(0.95)
    assert garch.variance_path("gjr", p, np.array([2.0]))[0, 1] == pytest.approx(2.0, rel=1e-14)


@pytest.mark.parametrize("kind", ["garch", "gjr"])
def test_the_analytic_path_equals_arch_s_analytic_forecast(kind):
    from arch.univariate import ZeroMean

    e = returns(1500, 11)
    model = ZeroMean(e, rescale=False)
    model.volatility = garch._vol_process(kind)
    res = model.fit(disp="off", backcast=garch.backcast(e))
    vol = dict(zip(garch.VOL_NAMES[kind], res.params))
    s1 = garch.filter_variance(kind, vol, e, garch.backcast(e))[-1:]
    theirs = res.forecast(horizon=21, reindex=False).variance.iloc[-1].to_numpy()
    np.testing.assert_allclose(garch.variance_path(kind, vol, s1)[0], theirs, rtol=1e-9)


def test_egarch_mgf_by_hand():
    # z = (-1, 1), alpha 0.1, gamma -0.05, beta 0.5: g(z) = 0.1*(1 - 0.797885) -/+ 0.05, so at c = 1
    # M = exp(0.0202115) * (exp(0.05) + exp(-0.05))/2 = 1.020417 * cosh(0.05) = 1.020417 * 1.001250 = 1.021693.
    mgf = garch.egarch_mgf({"alpha": 0.1, "gamma": -0.05, "beta": 0.5}, np.array([-1.0, 1.0]))
    assert len(mgf) == 20 and mgf[0] == pytest.approx(1.021693, abs=1e-6)
    # At c = beta = 0.5 the same with half the exponent: exp(0.0101058) * cosh(0.025) = 1.010157 * 1.000313 = 1.010473.
    assert mgf[1] == pytest.approx(1.010473, abs=1e-6)


def test_egarch_expected_variance_equals_brute_force_enumeration_of_every_shock_path():
    # Under a two-point z = -1 or +1 (each 1/2), E_t[sigma2_{t+k}] can be enumerated over the 2**(k-1)
    # shock paths. The closed form exp(omega*sum beta**i + beta**(k-1) ln s1) * prod M(beta**i) must match.
    p = {"omega": -0.05, "alpha": 0.2, "gamma": -0.1, "beta": 0.9}
    z = np.array([-1.0, 1.0])
    s1 = 1.7
    path = garch.variance_path("egarch", p, np.array([s1]), 21, garch.egarch_mgf(p, z))[0]
    for k in range(1, 6):
        total = 0.0
        for shocks in itertools.product(z, repeat=k - 1):
            ln = math.log(s1)
            for s in shocks:
                ln = p["omega"] + p["alpha"] * (abs(s) - garch.SQRT2_OV_PI) + p["gamma"] * s + p["beta"] * ln
            total += math.exp(ln)
        assert path[k - 1] == pytest.approx(total / 2 ** (k - 1), rel=1e-12)


# --- the innovation rule ------------------------------------------------------------------------


def _cand(dist, aic, ok=True, p=0):
    return garch.Candidate(p, dist, ok, aic, None, [], [], {}, ok, "" if ok else "x", None)


def test_choose_keeps_the_lowest_aic_among_usable_fits_and_a_tie_keeps_normal():
    assert garch.choose([_cand("normal", 100.0), _cand("t", 99.0)]).dist == "t"
    assert garch.choose([_cand("normal", 99.0), _cand("t", 100.0)]).dist == "normal"
    assert garch.choose([_cand("normal", 100.0), _cand("t", 100.0)]).dist == "normal"  # tie: the first
    assert garch.choose([_cand("normal", 50.0, ok=False), _cand("t", 100.0)]).dist == "t"  # unusable skipped
    assert garch.choose([_cand("normal", 50.0, ok=False)]) is None


def test_fat_tailed_returns_choose_student_t():
    from ivf.models.rv21 import Rv21Inputs

    e = returns(3000, 5, df=4.0)
    fit = garch.GarchRv("garch").fit(Rv21Inputs(e / 100, np.full(len(e), 20.0)), None)
    aics = fit.hyperparams["aic_by_dist"]
    assert fit.params["dist"] == "t" and aics["t"] < aics["normal"] and fit.aic == aics["t"]
    assert 3 < fit.params["nu"] < 6


def test_a_non_stationary_fit_is_not_usable(monkeypatch):
    # Force arch's estimate to alpha + beta = 1.0: the candidate must be refused, not used.
    monkeypatch.setattr(garch, "persistence", lambda kind, p: 1.0)
    e = returns(800, 3)
    cand, _ = garch.fit_candidate("garch", e, "normal", garch.backcast(e))
    assert not cand.ok and cand.why == "not stationary"


def test_garch_rv_fit_fails_when_no_candidate_is_usable(monkeypatch):
    from ivf.models.rv21 import Rv21Inputs

    monkeypatch.setattr(garch, "choose", lambda cands: None)
    with pytest.raises(ValueError, match="no candidate"):
        garch.GarchRv("gjr").fit(Rv21Inputs(returns(500, 2) / 100, np.full(500, 20.0)), None)


# --- ARIMA-GARCH on ln VIX: d = 0 (default; amended M3, review A) and the d = 1 option ---------------


def test_ar_stationary_by_hand():
    # Companion eigenvalues: (1.2, -0.3) -> l**2 - 1.2 l + 0.3 = 0 -> 0.845, 0.355 (stationary);
    # (0.5, 0.6) -> l**2 - 0.5 l - 0.6 = 0 -> 1.064 (not); (1.0,) is a unit root (not).
    assert garch.ar_stationary([0.5]) and garch.ar_stationary([1.2, -0.3]) and garch.ar_stationary([])
    assert not garch.ar_stationary([1.0]) and not garch.ar_stationary([0.5, 0.6])


def test_psi_weights_by_hand():
    # AR(1) phi 0.5: psi = 1, 0.5, 0.25. AR(2) (0.5, 0.2): psi_2 = 0.5*0.5 + 0.2 = 0.45.
    np.testing.assert_allclose(garch.psi_weights([0.5], 3), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(garch.psi_weights([0.5, 0.2], 3), [1.0, 0.5, 0.45])


def test_d0_forecast_by_hand():
    # d = 0: x = 100 ln y, x_t = c + 0.5 x_{t-1} + eps with c = 100; GARCH omega 0.1, alpha 0, beta 0.5,
    # backcast 1. y = 20, 21, 20, 22: 4 values, 1 held back, so the recursion runs over 3 residuals:
    #   s2 = 0.1 + 0.5*1 = 0.6, then 0.4, 0.3, and 0.25 past the sample.
    # Origin t = 3 (x_t = 100 ln 22 = 309.104245): x-hat_{t+1} = 100 + 0.5*309.104245 = 254.552123,
    #   x-hat_{t+2} = 100 + 0.5*254.552123 = 227.276061, so mu_1 = 2.545521 and mu_2 = 2.272761.
    #   sigma2_{t+1} = 0.25, sigma2_{t+2} = 0.1 + 0.5*0.25 = 0.225;
    #   s2_2 = (psi_1**2 * 0.25 + psi_0**2 * 0.225)/100**2 = (0.0625 + 0.225)/10**4 = 2.875e-5.
    # Origin t = 2 (x_t = 100 ln 20 = 299.573227): mu_1 = (100 + 149.786614)/100 = 2.497866, s2_1 = 0.3/10**4.
    fit = Fit(params={"p": 1, "const": 100.0, "phi": [0.5], "omega": 0.1, "alpha": 0.0, "beta": 0.5, "backcast": 1.0},
              hyperparams={"hold_back": 1, "d": 0}, n_train=3, train_first=0, train_last_target=3)
    out = garch.ArimaGarch(max_p=1).forecast(fit, np.array([20.0, 21.0, 20.0, 22.0]), np.array([2, 3]), (1, 2))
    assert out[1].mu_log[1] == pytest.approx(2.545521, abs=1e-6) and out[2].mu_log[1] == pytest.approx(2.272761, abs=1e-6)
    assert out[1].s2_log[1] == pytest.approx(2.5e-5, rel=1e-12) and out[2].s2_log[1] == pytest.approx(2.875e-5, rel=1e-12)
    assert out[1].mu_log[0] == pytest.approx(2.497866, abs=1e-6) and out[1].s2_log[0] == pytest.approx(3e-5, rel=1e-12)
    assert out[2].yhat[1] == pytest.approx(math.exp(2.272761 + 2.875e-5 / 2), rel=1e-6)
    assert out[2].yhat_naive[1] == pytest.approx(math.exp(2.272761), rel=1e-6)  # the median is stored too


def test_psi_cumulative_by_hand():
    # AR(1) phi 0.5: psi = 1, 0.5, 0.25 -> Psi = 1, 1.5, 1.75. AR(2) (0.5, 0.2): psi_2 = 0.5*0.5 + 0.2 = 0.45 -> Psi_2 = 1.95.
    np.testing.assert_allclose(garch.psi_cumulative([0.5], 3), [1.0, 1.5, 1.75])
    np.testing.assert_allclose(garch.psi_cumulative([0.5, 0.2], 3), [1.0, 1.5, 1.95])
    # p = 0 (a random walk in ln y): psi = 1, 0, 0 -> Psi = 1, 1, 1, each later shock counts once.
    np.testing.assert_allclose(garch.psi_cumulative([], 3), [1.0, 1.0, 1.0])


def test_d1_option_forecast_by_hand():
    # The d = 1 option. p = 1, phi = 0.5, and alpha = beta = 0, so sigma2 = omega = 0.1 (percent^2) at every step.
    # y = 20, 21, 20, 22; origin t = 3. The last change is dx_t = 100 ln(22/20) = 9.531018, so
    #   dx-hat_{t+1} = 4.765509, dx-hat_{t+2} = 2.382754, and mu_2 = ln 22 + (4.765509 + 2.382754)/100
    #   = 3.091042 + 0.071483 = 3.162525.
    #   s2_2 = (Psi_1**2 + Psi_0**2) * 0.1 / 100**2 = (2.25 + 1) * 1e-5 = 3.25e-5;  s2_1 = 1e-5.
    fit = Fit(params={"p": 1, "phi": [0.5], "omega": 0.1, "alpha": 0.0, "beta": 0.0, "backcast": 0.1},
              hyperparams={"hold_back": 1, "d": 1}, n_train=2, train_first=0, train_last_target=3)
    out = garch.ArimaGarch(max_p=1).forecast(fit, np.array([20.0, 21.0, 20.0, 22.0]), np.array([3]), (1, 2))
    assert out[2].mu_log[0] == pytest.approx(3.162525, abs=1e-6) and out[2].s2_log[0] == pytest.approx(3.25e-5, rel=1e-12)
    assert out[1].mu_log[0] == pytest.approx(math.log(22) + 0.04765509, abs=1e-8) and out[1].s2_log[0] == pytest.approx(1e-5, rel=1e-12)
    assert out[2].yhat[0] == pytest.approx(math.exp(3.162525 + 3.25e-5 / 2), rel=1e-6)


def test_d1_option_variance_weights_a_moving_garch_path_by_hand():
    # The d = 1 option. p = 1, phi = 0.5; GARCH omega 0.1, alpha 0, beta 0.5, backcast 1. y = 20, 21, 20, 22 gives 3
    # changes; with 1 held back the recursion runs over 2 residuals:
    #   s2[0] = 0.1 + 0.5*1   = 0.6;  s2[1] = 0.1 + 0.5*0.6 = 0.4;  s2[2] = 0.1 + 0.5*0.4 = 0.3
    # Origin t = 3 (3 changes up to t): sigma2_{t+1} = s2[2] = 0.3, sigma2_{t+2} = 0.1 + 0.5*0.3 = 0.25, so
    #   s2_2 = (Psi_1**2 * 0.3 + Psi_0**2 * 0.25)/100**2 = (2.25*0.3 + 0.25)/10**4 = 9.25e-5
    #   (the Psi in the wrong order would give (0.3 + 2.25*0.25)/10**4 = 8.625e-5).
    # Origin t = 2 (2 changes up to t): sigma2_{t+1} = s2[1] = 0.4, so s2_1 = 4e-5 (using s2[2] would look ahead).
    fit = Fit(params={"p": 1, "phi": [0.5], "omega": 0.1, "alpha": 0.0, "beta": 0.5, "backcast": 1.0},
              hyperparams={"hold_back": 1, "d": 1}, n_train=2, train_first=0, train_last_target=3)
    out = garch.ArimaGarch(max_p=1).forecast(fit, np.array([20.0, 21.0, 20.0, 22.0]), np.array([2, 3]), (1, 2))
    assert out[2].s2_log[1] == pytest.approx(9.25e-5, rel=1e-12)
    assert out[1].s2_log[0] == pytest.approx(4e-5, rel=1e-12) and out[1].s2_log[1] == pytest.approx(3e-5, rel=1e-12)


@pytest.mark.parametrize("d", [0, 1])
def test_arima_garch_forecast_at_t_does_not_depend_on_closes_after_t(d):
    # The truncation test (ADR 0001 4.7) for arima_garch_log: parameters fixed at the cutoff, the
    # forecast at t from closes up to t equals the one made with closes up to the fold's last origin.
    p = synthetic.SyntheticParams()
    y = np.exp(synthetic.simulate_log_ar1(1300, p.phi, p.mu, p.sigma, np.random.default_rng(12)))
    model = garch.ArimaGarch(max_p=2, dists=("normal",), d=d)
    c = 1200
    fit = model.fit(y[: c + 1], (1, 21), None)
    origins = np.array([c, c + 1, c + 40, c + 99])
    full = model.forecast(fit, y[: c + 100], origins, (1, 21))
    for i, t in enumerate(origins):
        part = model.forecast(fit, y[: t + 1], np.array([t]), (1, 21))
        for h in (1, 21):
            assert part[h].mu_log[0] == full[h].mu_log[i] and part[h].s2_log[0] == pytest.approx(full[h].s2_log[i], rel=1e-13)


def test_d0_matches_arch_s_own_mean_and_variance_at_every_horizon():
    # arch as the oracle: its ARX forecast with the same fixed parameters gives the mean of 100 ln y at
    # h and the variance of its h-step error (psi-weighted GARCH path); ours must equal both.
    from arch.univariate import ARX

    p = synthetic.SyntheticParams()
    x = synthetic.simulate_log_ar1(1200, p.phi, p.mu, p.sigma, np.random.default_rng(4))
    y = np.exp(x)
    model = garch.ArimaGarch(max_p=2, dists=("normal",))
    fit = model.fit(y, (1, 5, 21), None)
    assert fit.params["p"] >= 1 and fit.hyperparams["d"] == 0
    out = model.forecast(fit, y, np.array([len(y) - 1]), (1, 5, 21))
    ref = ARX(100 * x, lags=fit.params["p"], constant=True, hold_back=2, rescale=False)
    ref.volatility = garch._vol_process("garch")
    res = ref.fix([fit.params["const"], *fit.params["phi"], fit.params["omega"], fit.params["alpha"], fit.params["beta"]])
    fc = res.forecast(horizon=21, reindex=False)
    for h in (1, 5, 21):
        assert out[h].mu_log[0] == pytest.approx(fc.mean.iloc[-1, h - 1] / 100, abs=1e-12)
        assert out[h].s2_log[0] == pytest.approx(fc.variance.iloc[-1, h - 1] / 100**2, rel=1e-9)


def test_d1_option_matches_arch_s_own_mean_and_one_step_variance_at_the_sample_end():
    from arch.univariate import ARX

    p = synthetic.SyntheticParams()
    x = synthetic.simulate_log_ar1(1200, p.phi, p.mu, p.sigma, np.random.default_rng(4))
    y = np.exp(x)
    model = garch.ArimaGarch(max_p=2, dists=("normal",), d=1)
    fit = model.fit(y, (1, 5), None)
    out = model.forecast(fit, y, np.array([len(y) - 1]), (1, 5))
    dx = 100 * np.diff(x)
    k = fit.params["p"]
    if k:
        ref = ARX(dx, lags=k, constant=False, hold_back=2, rescale=False)
    else:
        from arch.univariate import ZeroMean

        ref = ZeroMean(dx, hold_back=2, rescale=False)
    ref.volatility = garch._vol_process("garch")
    res = ref.fix(fit.params["phi"] + [fit.params["omega"], fit.params["alpha"], fit.params["beta"]])
    fc = res.forecast(horizon=5, reindex=False)
    # arch's mean forecast of the changes, cumulated, is our mean of ln y; its h = 1 variance is ours.
    assert out[5].mu_log[0] == pytest.approx(x[-1] + fc.mean.iloc[-1].sum() / 100, abs=1e-10)
    assert out[1].s2_log[0] == pytest.approx(fc.variance.iloc[-1, 0] / 100**2, rel=1e-6)


def test_arima_garch_aic_compares_orders_on_the_same_sample():
    p = synthetic.SyntheticParams()
    y = np.exp(synthetic.simulate_log_ar1(900, p.phi, p.mu, p.sigma, np.random.default_rng(8)))
    fit = garch.ArimaGarch(max_p=2, dists=("normal",)).fit(y, (1,), None)
    cands = fit.params["candidates"]
    assert [c["p"] for c in cands] == [0, 1, 2]
    assert fit.n_train == 900 - 2  # d = 0: 900 values of ln y, the first 2 held back for every order
    assert garch.ArimaGarch(max_p=2, dists=("normal",), d=1).fit(y, (1,), None).n_train == 899 - 2  # d = 1: 899 changes
    best = min((c for c in cands if c["ok"]), key=lambda c: c["aic"])
    assert fit.params["p"] == best["p"] and fit.aic == best["aic"]


# --- concept note 10: GARCH on the level -------------------------------------------------------------


def test_garch_on_the_vix_level_shows_the_category_error_on_synthetic_data():
    # Synthetic ln VIX is an AR(1) with CONSTANT innovation variance: there is no volatility clustering
    # in it at all. GARCH on the level still "finds" it, because the level's own persistence is in the
    # squared deviations; its standardised residuals keep that persistence; and its mean forecast is
    # the historical mean. On the log changes, the right space, alpha is about 0 and the residuals are white.
    from ivf import protocol

    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, _ = synthetic.synthetic_market(proto)
    demo = garch.garch_on_the_level_demo(iv.loc[iv["symbol"] == "VIX", "close"].to_numpy()[:4000])
    assert demo["level_alpha_plus_beta"] > 0.9 and demo["level_std_resid_acf1"] > 0.7
    assert demo["level_one_step_rmse"] > 3 * demo["rw_one_step_rmse"]
    assert demo["dlog_alpha"] < 0.03 and abs(demo["dlog_std_resid_acf1"]) < 0.05


def test_the_committed_garch_settings_and_their_strict_reading(tmp_path):
    from ivf import config
    from ivf.models import registry
    from ivf.protocol import ProtocolError

    assert registry.load_garch_settings() == registry.GarchSettings(dists=("normal", "t"), max_p=2, d=0)
    defs, _ = registry.load_models()
    assert registry.models_for(defs, "rv21", 3) == ["rw", "har_rv", "vix_implied", "garch", "gjr", "egarch"]
    assert registry.models_for(defs, "vix_level", 3)[-1] == "arima_garch_log" and "arima_garch_log" not in registry.models_for(defs, "vix_level", 2)
    text = (config.configs_dir() / "models.yaml").read_text()
    for bad, match in ((text.replace("dists: [normal, t]", "dists: [normal, t]\n  dist: t"), "unknown keys"),
                       (text.replace("dists: [normal, t]", "dists: [normal, skewt]"), "garch.dists"),
                       (text.replace("  d: 0\n", "  d: 2\n"), "garch.d")):
        (tmp_path / "models.yaml").write_text(bad)
        with pytest.raises(ProtocolError, match=match):
            registry.load_garch_settings(tmp_path / "models.yaml")



# --- missing values: skipped, never filled (amended M3, first real run) -------------------------------


def test_a_missing_return_is_skipped_in_the_fit_and_in_the_forecast():
    from ivf.models.rv21 import Rv21Inputs

    e = returns(1200, 21)
    r = np.r_[np.nan, e / 100]  # the first return is always missing (no previous close)
    gappy = r.copy()
    gappy[300] = np.nan
    dropped = np.delete(r, 300)
    vix = np.full(len(r), 20.0)
    model = garch.GarchRv("garch", dists=("normal",))
    a = model.fit(Rv21Inputs(gappy, vix), None)
    b = model.fit(Rv21Inputs(dropped, vix[:-1]), None)
    assert a.n_train == b.n_train == 1200 - 1 and a.params["omega"] == b.params["omega"]
    # The forecast at the last origin uses the same shocks in the same order either way.
    fa = model.forecast(a, Rv21Inputs(gappy, vix), np.array([len(r) - 1]))
    fb = model.forecast(b, Rv21Inputs(dropped, vix[:-1]), np.array([len(dropped) - 1]))
    assert fa.v_hat[0] == pytest.approx(fb.v_hat[0], rel=1e-12)


@pytest.mark.parametrize("d", [0, 1])
def test_arima_garch_fits_through_a_missing_close_shortly_before_the_cutoff(d):
    # The first real run's failure mode (synthetic numbers): an isolated missing close shortly before the
    # cutoff. d = 0 skips that one value of ln y; d = 1 skips the two changes that touch it.
    p = synthetic.SyntheticParams()
    x = synthetic.simulate_log_ar1(1500, p.phi, p.mu, p.sigma, np.random.default_rng(6))
    y = np.exp(x)
    y[len(y) - 1 - 51] = np.nan
    model = garch.ArimaGarch(max_p=2, dists=("normal",), d=d)
    fit = model.fit(y, (1,), None)
    series = 100 * (np.log(y) if d == 0 else np.diff(np.log(y)))
    kept = series[np.isfinite(series)]
    assert fit.n_train == ((1500 - 1) if d == 0 else (1499 - 2)) - 2  # 2 held back
    bc = garch.backcast(garch.ols_residuals(kept, 0, d == 0, 2))
    cand, _ = garch.fit_candidate("garch", kept, "normal", bc, p=0, constant=d == 0, hold_back=2)
    assert fit.params["candidates"][0]["aic"] == pytest.approx(cand.aic, rel=1e-12)
    out = model.forecast(fit, y, np.array([len(y) - 1]), (1, 21))
    assert np.isfinite(out[21].yhat).all()


# --- review A: silent refusals, the EGARCH residual scale ----------------------------------------------


def test_every_candidate_is_usable_on_clean_synthetic_data():
    # The FILTER_RTOL and stationarity checks refuse a candidate silently (the fit falls back to another
    # one). On clean data none may be refused, so a mis-specified residual (review A's surviving mutant)
    # shows up as a failure here, and in the fit table's candidates_refused column.
    from ivf.evaluation.walkforward import Fold, _fit_row
    from ivf.models.rv21 import Rv21Inputs

    e = returns(2000, 31)
    inputs = Rv21Inputs(np.r_[np.nan, e / 100], np.full(len(e) + 1, 20.0))
    # 100 ln y: an AR(1) (c 6, phi 0.98) whose shocks are GARCH(1,1) (alpha 0.08, beta 0.90): with
    # clustering in the shocks, a wrongly computed residual moves the variance recursion and is refused.
    shocks = returns(2000, 32) * 3
    x = np.empty(2000)
    x[0] = 300.0
    for i in range(1, 2000):
        x[i] = 6.0 + 0.98 * x[i - 1] + shocks[i]
    y = np.exp(x / 100)
    fits = [garch.GarchRv(k).fit(inputs, None) for k in garch.KINDS]
    fits += [garch.ArimaGarch(max_p=2, d=d).fit(y, (1,), None) for d in (0, 1)]
    for fit in fits:
        assert all(c["ok"] for c in fit.params["candidates"]), [c["why"] for c in fit.params["candidates"]]
        assert _fit_row("m", "t", Fold(0, 0, 0, "dev"), fit, [date(2000, 1, 3)] * 2001)["candidates_refused"] == 0
    assert len(fits[-2].params["candidates"]) == 6  # 3 orders x 2 distributions


@pytest.mark.parametrize("dist", garch.DISTS)
def test_egarch_mgf_uses_the_standardised_residuals_rescaled_to_unit_rms(dist):
    # z from arch's own standardised residuals (the oracle), rescaled so that mean(z**2) = 1, then
    # M(beta**i) = mean(exp(beta**i * g(z))). Raw percent residuals, or no rescaling, give other numbers.
    from arch.univariate import ZeroMean

    from ivf.models.rv21 import Rv21Inputs

    e = returns(1500, 33, df=6.0 if dist == "t" else None)
    fit = garch.GarchRv("egarch", dists=(dist,)).fit(Rv21Inputs(e / 100, np.full(len(e), 20.0)), None)
    model = ZeroMean(e, rescale=False)
    model.volatility, model.distribution = garch._vol_process("egarch"), garch._distribution(dist)
    res = model.fix([fit.params[k] for k in ("omega", "alpha", "gamma", "beta")] + ([fit.params["nu"]] if dist == "t" else []))
    z = np.asarray(res.std_resid, dtype=float)
    assert fit.params["z_rms"] == pytest.approx(math.sqrt(np.mean(z**2)), rel=1e-6)
    z = z / math.sqrt(np.mean(z**2))
    p = {k: fit.params[k] for k in ("alpha", "gamma", "beta")}
    np.testing.assert_allclose(fit.params["mgf"], garch.egarch_mgf(p, z), rtol=1e-6)


def test_ols_residuals_by_hand():
    # data = 1, 3, 5, 9 with 1 held back: y = 3, 5, 9.
    # Constant only: mean 17/3, residuals -8/3, -2/3, 10/3.
    # One lag, no constant: b = (1*3 + 3*5 + 5*9)/(1 + 9 + 25) = 63/35 = 1.8; residuals 3 - 1.8 = 1.2,
    # 5 - 5.4 = -0.4, 9 - 9 = 0.
    data = np.array([1.0, 3.0, 5.0, 9.0])
    np.testing.assert_allclose(garch.ols_residuals(data, 0, True, 1), [-8 / 3, -2 / 3, 10 / 3], atol=1e-12)
    np.testing.assert_allclose(garch.ols_residuals(data, 1, False, 1), [1.2, -0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(garch.ols_residuals(data, 0, False, 1), [3.0, 5.0, 9.0])


def test_a_non_stationary_ar_candidate_is_refused(monkeypatch):
    # With a constant (d = 0) the AR must be stationary, or the mean forecast never reverts.
    monkeypatch.setattr(garch, "ar_stationary", lambda phi: False)
    p = synthetic.SyntheticParams()
    z = 100 * synthetic.simulate_log_ar1(600, p.phi, p.mu, p.sigma, np.random.default_rng(9))
    cand, _ = garch.fit_candidate("garch", z, "normal", garch.backcast(garch.ols_residuals(z, 1, True, 2)), p=1, constant=True, hold_back=2)
    assert not cand.ok and cand.why == "AR not stationary"
    zero, _ = garch.fit_candidate("garch", z - z.mean(), "normal", 1.0, p=1, constant=False, hold_back=2)
    assert zero.why != "AR not stationary"  # the d = 1 option (no constant) is not subject to it

