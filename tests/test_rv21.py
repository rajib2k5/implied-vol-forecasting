"""The rv21 track of milestone 3: target alignment, rule L2 for a window target (O-M2-7), the holdout,
the truncation test, same rows, the three rv21 baselines, and the ``garch`` command.

The L2 tests are the point of this file. rv21 at origin t is a WINDOW value observed at t + 21, so a fit
at cutoff c may pair only origins tau <= c - 21 with their targets, and may read no return after c.
Synthetic data only.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ivf import protocol
from ivf.data import calendar, synthetic
from ivf.evaluation import walkforward
from ivf.features import build
from ivf.models import garch, rv21
from ivf.models.baselines import Fit

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(scope="module")
def world():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, price = synthetic.synthetic_market(proto)
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, proto.data_end)
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    return {
        "proto": proto, "iv": iv, "price": price, "cal": cal, "calendar_dates": calendar_dates,
        "dev": [f for f in folds if f.period == "dev"], "dates": dates, "inputs": inputs_from(dates, iv_pre, price_pre),
    }


def inputs_from(dates, iv, price) -> rv21.Rv21Inputs:
    return rv21.Rv21Inputs(build.log_returns(build.align(dates, price, "ticker", "^GSPC")), build.align(dates, iv, "symbol", "VIX"))


def every_model():
    return rv21.build_rv21_models()


def run(world, inputs, folds, models=None):
    return walkforward.run_rv21(inputs, world["dates"][: len(inputs)], world["calendar_dates"], folds, models or every_model(),
                                world["proto"].holdout_start, 504)


FORECAST_COLS = ["yhat", "v_hat", "var_method", "fold_id", "target_date"]


# --- the target and the training origins ---------------------------------------------------------


def test_training_origins_end_21_sessions_before_the_cutoff():
    # Inputs up to c = 100 (101 values): origins 0..79. The pair at tau = 79 has the window 80..100.
    tau = rv21.training_origins(101)
    assert tau[0] == 0 and tau[-1] == 79 and len(tau) == 80
    assert len(rv21.training_origins(21)) == 0 and list(rv21.training_origins(22)) == [0]


def test_the_target_of_origin_t_is_the_window_t_plus_1_to_t_plus_21():
    # Returns of 0.01 on t+1..t+21 and 0.05 everywhere else: rv21 at t + 21 must be exactly the
    # 1%-a-day value 100*sqrt(12 * 21 * 0.0001) = 15.874508, and one session earlier it must not be.
    r = np.full(80, 0.05)
    t = 30
    r[t + 1 : t + 22] = 0.01
    y = rv21.target_series(r)
    assert y[t + 21] == pytest.approx(15.874508, abs=1e-6)
    assert y[t + 20] > 16 and y[t + 22] > 16


# --- rule L2 for the window target ------------------------------------------------------------------


def test_har_rv_trains_only_on_origins_whose_window_ends_by_the_cutoff(world):
    # The origin-truncation test: the fit at cutoff c equals OLS on pairs built by hand from the FULL
    # series, restricted to tau <= c - 21; its last training origin is c - 21 and its last target c.
    inputs = world["inputs"]
    c = world["dev"][-1].cutoff_idx
    fit = rv21.HarRv().fit(inputs.upto(c), None)
    assert fit.train_last_origin == c - 21 and fit.train_last_target == c
    r = inputs.r
    design = rv21.har_rv_design(r)

    def ols_on_pairs_up_to(last_tau):
        taus = np.arange(last_tau + 1)
        target = np.array([12 * np.sum(r[t + 1 : t + 22] ** 2) for t in taus])  # (252/21) x the window t+1..t+21
        keep = np.isfinite(design[taus]).all(axis=1) & np.isfinite(target)
        return np.linalg.lstsq(design[taus][keep], target[keep], rcond=None)[0]

    np.testing.assert_allclose([fit.params[k] for k in rv21.HarRv.COEFS], ols_on_pairs_up_to(c - 21), rtol=1e-9)
    # One more origin (tau = c - 20, whose window needs the return at c + 1) changes the fit, so the
    # restriction is binding, not vacuous.
    assert not np.allclose(ols_on_pairs_up_to(c - 21), ols_on_pairs_up_to(c - 20), rtol=1e-12)


def test_poisoned_returns_after_the_cutoff_change_no_fit_and_no_forecast_at_the_cutoff(world):
    # Every return and VIX close after the cutoff is replaced by garbage. A fit that read any of it
    # (an input, or a target window ending after c) would change; so would the forecast at origin c.
    fold = world["dev"][-1]
    c = fold.cutoff_idx
    clean = world["inputs"]
    r, v = clean.r.copy(), clean.vix.copy()
    rng = np.random.default_rng(0)
    r[c + 1 :] = rng.normal(0, 0.2, len(r) - c - 1)  # 20% a day
    v[c + 1 :] = 300.0
    a = run(world, clean, [fold])
    b = run(world, rv21.Rv21Inputs(r, v), [fold])
    assert a.fits["params_json"].tolist() == b.fits["params_json"].tolist()
    assert set(a.fits.model_id) == {"rw", "har_rv", "vix_implied", "garch", "gjr", "egarch"}
    first = world["dates"][c]
    fa = a.forecasts[a.forecasts.origin_date == first].set_index("model_id")[FORECAST_COLS]
    fb = b.forecasts[b.forecasts.origin_date == first].set_index("model_id")[FORECAST_COLS]
    assert len(fa) == 6
    pd.testing.assert_frame_equal(fa, fb)
    # ... while later origins of the fold do move: the poison is visible to forecasts made after c.
    later = world["dates"][c + 5]
    assert not np.allclose(a.forecasts.loc[a.forecasts.origin_date == later, "yhat"], b.forecasts.loc[b.forecasts.origin_date == later, "yhat"])


def test_the_runner_refuses_a_fit_whose_last_target_window_ends_after_the_cutoff(world):
    # A model that pairs origin c - 20 (window ending at c + 1) must be refused, even if it claims a
    # last target on the cutoff.
    class Leaky(rv21.HarRv):
        def fit(self, inputs, previous):
            fit = super().fit(inputs, previous)
            fit.train_last_origin += 1
            return fit

    with pytest.raises(AssertionError, match="window target"):
        run(world, world["inputs"], world["dev"][-1:], [Leaky()])


def test_every_rv21_fit_ends_at_its_cutoff(world):
    fits = run(world, world["inputs"], world["dev"][-2:]).fits
    used = fits.dropna(subset=["train_last_target_date"])
    assert set(used.model_id) == {"har_rv", "garch", "gjr", "egarch"}
    assert (used["train_last_target_date"] == used["cutoff_date"]).all()  # GARCH: the return at c; HAR: the window ending at c


def test_truncation_rv21_forecasts_at_t_do_not_depend_on_data_after_t(world):
    # ADR 0001 4.7 for the rv21 track: a run on inputs truncated at t gives the same forecasts at t.
    folds = world["dev"][-2:]
    full = run(world, world["inputs"], folds).forecasts.set_index(["model_id", "origin_date"])
    c = folds[1].cutoff_idx
    for t in (c - 1, c, c + 1, c + 30):
        part = run(world, world["inputs"].upto(t), folds).forecasts
        at_t = part[part.origin_date == world["dates"][t]].set_index(["model_id", "origin_date"])
        assert len(at_t) == 6
        pd.testing.assert_frame_equal(at_t[FORECAST_COLS], full.loc[at_t.index, FORECAST_COLS], rtol=1e-12)


# --- the holdout ------------------------------------------------------------------------------------------


def test_poisoning_every_holdout_row_changes_nothing_in_the_rv21_track(world):
    hs = world["proto"].holdout_start
    price, iv = world["price"].copy(), world["iv"].copy()
    price.loc[price["date"] >= hs, "close"] *= 100
    iv.loc[iv["date"] >= hs, "close"] *= 100
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(world["cal"], iv, price, hs)
    assert dates == world["dates"] and price_pre["date"].max() < hs
    models = [rv21.RandomWalkRv(), rv21.HarRv(), rv21.VixImplied(), garch.GarchRv("garch")]
    a = run(world, world["inputs"], world["dev"][-1:], models).forecasts
    b = run(world, inputs_from(dates, iv_pre, price_pre), world["dev"][-1:], models).forecasts
    pd.testing.assert_frame_equal(a, b)
    # The last dev fold's last 21 origins target the holdout: produced, no target, never scored.
    rw = a[a.model_id == "rw"]
    assert len(rw) == 63 and rw["y"].isna().sum() == 21 and len(walkforward.scored(rw, hs)) == 42
    assert (rw.loc[rw["y"].isna(), "target_date"] >= hs).all()


def test_run_rv21_refuses_holdout_sessions_and_holdout_folds(world):
    from ivf.data.process import HoldoutRowRefused

    holdout = walkforward.fold_schedule(world["calendar_dates"], world["proto"].dev_start, world["proto"].holdout_start,
                                        world["proto"].fold_len, world["proto"].data_end)[-1]
    with pytest.raises(HoldoutRowRefused):
        run(world, world["inputs"], [holdout], [rv21.RandomWalkRv()])
    dates = world["calendar_dates"][: len(world["dates"]) + 1]
    longer = rv21.Rv21Inputs(np.r_[world["inputs"].r, 0.0], np.r_[world["inputs"].vix, 20.0])
    with pytest.raises(HoldoutRowRefused):
        walkforward.run_rv21(longer, dates, world["calendar_dates"], world["dev"][-1:], [rv21.RandomWalkRv()], world["proto"].holdout_start, 504)


# --- same rows, the baselines and the scored values -----------------------------------------------------


def test_every_rv21_model_forecasts_the_same_origins_and_a_vix_gap_drops_the_origin_for_all(world):
    fold = world["dev"][-1]
    vix = world["inputs"].vix.copy()
    vix[fold.cutoff_idx + 3] = np.nan  # no VIX close at one origin: no model may forecast it
    out = run(world, rv21.Rv21Inputs(world["inputs"].r, vix), [fold]).forecasts
    walkforward.check_same_rows(out, ["rw", "har_rv", "vix_implied", "garch", "gjr", "egarch"])
    assert out.groupby("model_id").size().eq(62).all()
    assert world["dates"][fold.cutoff_idx + 3] not in set(out.origin_date)
    with pytest.raises(AssertionError, match="no row"):
        walkforward.check_same_rows(out[out.model_id != "egarch"], ["rw", "har_rv", "vix_implied", "garch", "gjr", "egarch"])


def test_rw_vix_implied_and_garch_rows_carry_the_right_numbers(world):
    fold = world["dev"][-1]
    out = run(world, world["inputs"], [fold], [rv21.RandomWalkRv(), rv21.VixImplied(), garch.GarchRv("garch")]).forecasts
    c = fold.cutoff_idx
    at_c = out[out.origin_date == world["dates"][c]].set_index("model_id")
    rv_back = build.rv_trailing(world["inputs"].r)[c]
    assert at_c.loc["rw", "yhat"] == pytest.approx(rv_back, rel=1e-14) and at_c.loc["rw", "var_method"] == "plugin"
    # vix_implied: yhat = V_t, v_hat = (V_t/100)**2 exactly (e.g. 20 -> 0.04), native.
    v = world["inputs"].vix[c]
    assert at_c.loc["vix_implied", "yhat"] == pytest.approx(v, rel=1e-14) and at_c.loc["vix_implied", "v_hat"] == pytest.approx((v / 100) ** 2, rel=1e-14)
    assert at_c.loc["garch", "var_method"] == "native" and at_c.loc["vix_implied", "var_method"] == "native"
    assert np.allclose(out["yhat"], 100 * np.sqrt(out["v_hat"]), rtol=1e-12)
    # The scored value of origin c is rv21 at c + 21 (its window c+1..c+21).
    assert at_c.loc["garch", "y"] == pytest.approx(build.rv_trailing(world["inputs"].r)[c + 21], rel=1e-14)
    assert at_c.loc["garch", "target_date"] == world["dates"][c + 21]


def test_garch_forecast_at_t_is_the_aggregate_of_its_own_recursion(world):
    # Recompute the forecast at one origin from the stored parameters: s1 from the recursion through
    # returns up to t, the 21-step path, G = 12 * sum / 10**4.
    fold = world["dev"][-1]
    model = garch.GarchRv("gjr")
    run_ = run(world, world["inputs"], [fold], [model])
    fit_row = run_.fits.iloc[0]
    params = json.loads(fit_row.params_json)
    t = fold.cutoff_idx + 10
    first = world["dates"].index(fit_row.train_first_date)
    e = 100 * world["inputs"].r[first : t + 1]
    s1 = garch.filter_variance("gjr", params, e, params["backcast"])[-1]
    path = garch.variance_path("gjr", params, np.array([s1]))
    expected = 100 * math.sqrt(12 * path.sum() / 1e4)
    got = run_.forecasts.loc[run_.forecasts.origin_date == world["dates"][t], "yhat"].iloc[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_an_origin_needs_the_22_returns_t_minus_21_to_t_and_vix_at_t():
    # Review A: pin the window. A missing return at session 10 blocks every origin whose window
    # t-21..t contains it: t = 10..31. t = 32 (window 11..32) is forecastable; a 21-return window
    # would wrongly allow t = 31 (window 11..31).
    r, vix = np.full(60, 0.01), np.full(60, 20.0)
    r[0] = r[10] = np.nan
    vix[45] = np.nan
    ok = rv21.forecastable(rv21.Rv21Inputs(r, vix))
    assert not ok[31] and ok[32] and not ok[10] and not ok[21]
    assert not ok[45] and ok[44] and ok[46]
    assert list(np.flatnonzero(ok)) == [*range(32, 45), *range(46, 60)]


def test_har_rv_design_by_hand():
    # r = 0 on sessions 0..21 except r_17 = 0.03 and r_21 = 0.02. At t = 21:
    #   vd = 252 * 0.02**2 = 0.1008
    #   vw = 252 * (0.0009 + 0.0004)/5  = 0.06552    (sessions 17..21; a 4-session window would miss r_17: 0.0252)
    #   vm = 252 * (0.0009 + 0.0004)/22 = 0.0148909  (sessions 0..21)
    r = np.zeros(22)
    r[17], r[21] = 0.03, 0.02
    row = rv21.har_rv_design(r)[21]
    np.testing.assert_allclose(row, [1.0, 0.1008, 0.06552, 252 * 0.0013 / 22], rtol=1e-12)
    assert np.isnan(rv21.har_rv_design(r)[20, 3])  # the monthly window needs 22 returns


def test_har_rv_floors_a_non_positive_forecast_at_the_smallest_training_target():
    # b0 = -1 and every slope 0: the OLS forecast is -1 < floor 0.002, so v_hat = 0.002 (yhat = 100*sqrt(0.002) = 4.472136).
    fit = Fit(params={"b0": -1.0, "b_d": 0.0, "b_w": 0.0, "b_m": 0.0, "floor": 0.002, "n": 10}, hyperparams={}, n_train=10,
              train_first=0, train_last_target=30, train_last_origin=9)
    out = rv21.HarRv().forecast(fit, rv21.Rv21Inputs(np.full(40, 0.01), np.full(40, 20.0)), np.array([30, 35]))
    assert out.floored.all() and np.allclose(out.v_hat, 0.002) and out.yhat[0] == pytest.approx(4.472136, abs=1e-6)


def test_metric_qlike_for_rv21_is_on_variance():
    # y = 20 and yhat = 10 vol points: v/v_hat = 0.04/0.01 = 4, QLIKE = 4 - ln 4 - 1 = 1.613706.
    rows = pd.DataFrame({"model_id": "garch", "target_id": "rv21", "horizon": 21, "y": [20.0] * 30, "yhat": [10.0] * 30, "yhat_naive": [10.0] * 30,
                         "target_date": [date(2020, 1, 2)] * 30, "origin_date": [date(2019, 12, 1)] * 30})
    m = walkforward.metric_table(rows, date(2024, 7, 1), 30, set())
    assert m["qlike"].iloc[0] == pytest.approx(4 - math.log(4) - 1, abs=1e-12)


# --- the garch command, end to end on synthetic files --------------------------------------------------


def test_garch_command_merges_into_the_baselines_tables(tmp_path, monkeypatch, capsys):
    from ivf.data import cli as data_cli
    from ivf.models import cli as models_cli

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    for step in ("fetch", "build-interim", "check"):
        assert data_cli.main(["--synthetic", step]) == 0
    assert models_cli.main(["--synthetic", "baselines", "--symbols", "VIX3M", "--models", "rw"]) == 0
    capsys.readouterr()
    rc = models_cli.main(["--synthetic", "garch", "--models", "rw", "har_rv", "vix_implied"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "97 dev folds" in out and "no row on or after it is read" in out and "0 with train_last_target_date after the cutoff" in out
    root = tmp_path / "implied-vol-forecasting" / "synthetic" / "processed"
    folder = root / f"run_id={pd.read_parquet(root / 'runs.parquet').run_id.iloc[0]}"
    forecast = pd.read_parquet(folder / "forecast.parquet")
    counts = forecast.groupby(["target_id", "model_id"]).size().to_dict()
    # rv21: 97 folds x 63 origins (G3: none dropped) x 1 horizon per model; VIX: x 3 horizons for rw;
    # vix3m_level rows from the baselines run are kept (merged, not overwritten).
    assert counts[("rv21", "rw")] == counts[("rv21", "har_rv")] == counts[("rv21", "vix_implied")] == 97 * 63
    assert counts[("vix_level", "rw")] == 97 * 63 * 3 and counts[("vix3m_level", "rw")] > 0
    assert forecast["origin_date"].max() < date(2024, 7, 1)
    rv = forecast[forecast.target_id == "rv21"]
    assert set(rv.horizon) == {21} and set(rv.loc[rv.model_id == "vix_implied", "var_method"]) == {"native"}
    fit = pd.read_parquet(folder / "fit.parquet")
    assert "candidates_refused" in fit.columns and fit["candidates_refused"].isna().all()  # these models have no candidates
    metric = pd.read_parquet(folder / "metric.parquet")
    assert set(metric.loc[metric.target_id == "rv21", "model_id"]) == {"rw", "har_rv", "vix_implied"}
    assert models_cli.main(["--synthetic", "garch", "--models", "nope"]) == 3


def test_rv21_min_train_boundary_504_returns_fit_503_do_not(world):
    # A fold is fitted when at least 504 returns are dated <= its cutoff (a rule on the data).
    fold = world["dev"][0]
    c = fold.cutoff_idx
    for n_finite, fitted in ((504, 1), (503, 0)):
        r = world["inputs"].r.copy()
        r[: c + 1 - n_finite] = np.nan
        out = run(world, rv21.Rv21Inputs(r, world["inputs"].vix), [fold], [rv21.RandomWalkRv(), rv21.HarRv()])
        assert (out.folds_fitted, out.folds_skipped_min_train) == (fitted, 1 - fitted)
