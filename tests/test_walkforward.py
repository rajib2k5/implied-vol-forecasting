"""ivf.evaluation.walkforward and the ``python -m ivf.models baselines`` step (ADR 0001 4.5-4.7).

The look-ahead rules each have a test here: the fold schedule is dates only (4.5), the holdout is
never read (4.6), training pairs end at the cutoff (L2), forecasts use data <= the origin (L1, the
truncation test of 4.7), and every model forecasts at the same origins (4.5, "same rows").
Synthetic data only.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ivf import protocol
from ivf.data import calendar, synthetic
from ivf.data.process import HoldoutRowRefused
from ivf.evaluation import walkforward
from ivf.features import build
from ivf.models.baselines import Arima, Har, HistoricalMean, RandomWalk

pytestmark = pytest.mark.usefixtures("no_network")
H = (1, 5, 21)


def small_models():
    # ARIMA with a 2 x 2 grid keeps the test fast; the code path is the same.
    return [RandomWalk(), HistoricalMean("hist_mean", None), HistoricalMean("hist_mean_252", 252), Arima(max_p=1, max_q=1), Har()]


@pytest.fixture(scope="module")
def world():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, price = synthetic.synthetic_market(proto)
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, proto.data_end)
    dates, _, iv_pre, _ = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    return {
        "proto": proto, "iv": iv, "price": price, "cal": cal, "calendar_dates": calendar_dates, "folds": folds,
        "dev": [f for f in folds if f.period == "dev"], "dates": dates, "y": build.align(dates, iv_pre, "symbol", "VIX"),
    }


def run(world, y, folds, models=None, symbol="VIX", min_train=504):
    return walkforward.run_symbol(
        symbol, f"{symbol.lower()}_level", y, world["dates"][: len(y)] if len(y) < len(world["dates"]) else world["dates"],
        world["calendar_dates"], folds, models or small_models(), H, world["proto"].holdout_start, min_train,
    )


# --- folds: dates only ----------------------------------------------------------------------


def test_fold_schedule_on_the_xnys_calendar(world):
    # ADR 0001 4.5: 97 dev folds, the first cutoff on 2000-03-15, anchored on holdout_start, so
    # the last dev fold ends on the session before 2024-07-01 and the first holdout fold starts on it.
    dev, folds, d = world["dev"], world["folds"], world["calendar_dates"]
    assert len(dev) == 97 and d[dev[0].cutoff_idx] == date(2000, 3, 15)
    assert d[dev[-1].last_origin_idx] == date(2024, 6, 28)
    first_holdout = folds[len(dev)]
    assert first_holdout.period == "holdout" and d[first_holdout.cutoff_idx] == date(2024, 7, 1)
    assert [f.fold_id for f in folds] == list(range(len(folds)))
    assert all(f.last_origin_idx - f.cutoff_idx + 1 == 63 for f in dev)
    assert d[folds[-1].last_origin_idx] == world["proto"].data_end  # the last holdout fold is partial


def test_fold_table_and_calendar_fold_ids(world):
    table = walkforward.fold_table(world["folds"], world["calendar_dates"])
    assert list(table.columns) == ["fold_id", "cutoff_date", "last_origin", "period", "n_sessions"]
    cal = walkforward.calendar_with_folds(world["cal"], world["folds"])
    first = world["dev"][0].cutoff_idx
    assert cal["fold_id"].iloc[:first].isna().all() and cal["fold_id"].iloc[first] == 0
    assert cal["fold_id"].iloc[first + 63] == 1 and cal.loc[cal["date"] > world["proto"].data_end, "fold_id"].isna().all()


# --- the holdout is never read --------------------------------------------------------------


def test_run_symbol_refuses_a_holdout_session_or_a_holdout_fold(world):
    holdout = [f for f in world["folds"] if f.period == "holdout"]
    with pytest.raises(HoldoutRowRefused):
        run(world, world["y"], holdout[:1], [RandomWalk()])
    dates = world["calendar_dates"][: len(world["dates"]) + 1]  # one session too many: holdout_start itself
    with pytest.raises(HoldoutRowRefused):
        walkforward.run_symbol("VIX", "vix_level", np.r_[world["y"], 20.0], dates, world["calendar_dates"], world["dev"][-1:],
                               [RandomWalk()], H, world["proto"].holdout_start, 504)


def test_poisoning_every_holdout_row_changes_nothing(world):
    # Multiply every close on or after holdout_start by 100. If any of it were read (an input, a
    # fit, a target), a forecast or a scored target would change.
    hs = world["proto"].holdout_start
    iv = world["iv"].copy()
    iv.loc[iv["date"] >= hs, "close"] *= 100
    dates, _, iv_pre, _ = walkforward.pre_holdout_inputs(world["cal"], iv, world["price"], hs)
    assert dates == world["dates"] and iv_pre["date"].max() < hs
    poisoned = build.align(dates, iv_pre, "symbol", "VIX")
    models = [RandomWalk(), HistoricalMean("hist_mean", None), Har(), Arima(max_p=1, max_q=0)]
    a = run(world, world["y"], world["dev"][-1:], models).forecasts
    b = run(world, poisoned, world["dev"][-1:], models).forecasts
    pd.testing.assert_frame_equal(a, b)
    # The last dev fold's last h origins target the holdout: produced, with no target, never scored.
    rw = a[(a.model_id == "rw")]
    for h in H:
        rows = rw[rw.horizon == h]
        assert len(rows) == 63 and rows["y"].isna().sum() == h
        assert len(walkforward.scored(rows, hs)) == 63 - h
        assert (rows.loc[rows["y"].isna(), "target_date"] >= hs).all()


# --- L1 / L2 / the truncation test ----------------------------------------------------------


def test_truncation_test_forecasts_at_t_do_not_depend_on_data_after_t(world):
    # ADR 0001 4.7: a run on data truncated at t must give the same forecasts at t. Origins: a
    # cutoff, the session after it, mid-fold, and the last origin of a fold.
    folds = world["dev"][-2:]
    full = run(world, world["y"], folds).forecasts.set_index(["model_id", "origin_date", "horizon"])
    c = folds[1].cutoff_idx
    for t in (c, c + 1, c + 30, c - 1):
        part = run(world, world["y"][: t + 1], folds).forecasts
        at_t = part[part.origin_date == world["dates"][t]].set_index(["model_id", "origin_date", "horizon"])
        assert len(at_t) == 5 * 3
        cols = ["yhat", "yhat_naive", "mu_log", "s2_log", "fold_id"]
        pd.testing.assert_frame_equal(at_t[cols], full.loc[at_t.index, cols], rtol=1e-12)


def test_every_fit_ends_at_its_cutoff_rule_l2(world):
    fits = run(world, world["y"], world["dev"][-2:]).fits
    used = fits.dropna(subset=["train_last_target_date"])
    assert set(used.model_id) == {"hist_mean", "hist_mean_252", "arima_log", "har_log"}
    assert (used["train_last_target_date"] <= used["cutoff_date"]).all()
    # HAR's last pair is (cutoff - h, cutoff): its last target is the cutoff itself.
    har = fits[fits.model_id == "har_log"]
    assert (har["train_last_target_date"] == har["cutoff_date"]).all()


def test_a_fit_does_not_change_when_data_after_its_cutoff_changes(world):
    fold = world["dev"][-1]
    y = world["y"].copy()
    y[fold.cutoff_idx + 1 :] = 500.0  # garbage after the cutoff
    a = run(world, world["y"], [fold], [Har(), HistoricalMean("hist_mean_252", 252)]).fits
    b = run(world, y, [fold], [Har(), HistoricalMean("hist_mean_252", 252)]).fits
    assert a["params_json"].tolist() == b["params_json"].tolist()


# --- same rows, gaps, minimum training size, fallback ---------------------------------------


def test_same_rows_check_refuses_a_missing_origin():
    frame = pd.DataFrame({"model_id": ["a", "a", "b"], "target_id": "t", "origin_date": [1, 2, 1], "horizon": 1})
    with pytest.raises(AssertionError, match="same-rows"):
        walkforward.check_same_rows(frame)
    with pytest.raises(AssertionError, match="twice"):
        walkforward.check_same_rows(pd.concat([frame, frame]))


def test_same_rows_check_refuses_a_model_missing_a_whole_horizon():
    # Amended M2, review A: model b has its h = 5 rows but none at h = 1; the first version passed it.
    frame = pd.DataFrame(
        {"model_id": ["a", "a", "b"], "target_id": "t", "origin_date": [1, 1, 1], "horizon": [1, 5, 5]}
    )
    with pytest.raises(AssertionError, match=r"h=1: models \['b'\] have no row"):
        walkforward.check_same_rows(frame)
    # And a model the runner was given that produced nothing at all.
    with pytest.raises(AssertionError, match=r"models \['c'\] have no row"):
        walkforward.check_same_rows(frame[frame.horizon == 5], ["a", "b", "c"])


def test_min_train_sessions_boundary_504_fits_503_does_not(world):
    # The last dev fold's cutoff c. Closes only on c-503..c: 504 of them, the minimum, so the fold
    # is fitted. Closes only on c-502..c: 503, so it is skipped (no forecast from any model).
    c = world["dev"][-1].cutoff_idx
    for first, fitted in ((c - 503, 1), (c - 502, 0)):
        y = world["y"].copy()
        y[:first] = np.nan
        out = run(world, y, world["dev"][-1:], [RandomWalk()])
        assert (out.folds_fitted, out.folds_skipped_min_train) == (fitted, 1 - fitted)


def test_a_missing_forecast_is_an_error_not_a_dropped_row(world):
    class Broken(RandomWalk):
        model_id = "broken"

        def forecast(self, fit, y, origins, horizons):
            out = super().forecast(fit, y, origins, horizons)
            out[5].yhat[3] = np.nan
            return out

    with pytest.raises(ValueError, match="missing row is an error"):
        run(world, world["y"], world["dev"][-1:], [RandomWalk(), Broken()])


def test_gaps_and_late_starts_are_rules_on_the_data_not_on_the_model(world):
    # A symbol whose first close is 600 sessions before the last fold's cutoff, with one gap.
    # min_train 504: the second-to-last fold has 600 - 63 = 537 closes <= its cutoff (fitted), the
    # one before it 474 (skipped), and so on back. One gap at cutoff + 10 makes origins
    # cutoff+10 .. cutoff+31 not forecastable (22 sessions: the HAR month t-21..t), for every model.
    y = world["y"].copy()
    c = world["dev"][-1].cutoff_idx
    y[: c - 600 + 1] = np.nan
    y[c + 10] = np.nan
    out = run(world, y, world["dev"], [RandomWalk(), Har()])
    assert out.folds_fitted == 2 and out.folds_skipped_min_train == 95
    assert out.origins_not_forecastable == 22
    last = out.forecasts[(out.forecasts.fold_id == world["dev"][-1].fold_id) & (out.forecasts.horizon == 1)]
    assert last.groupby("model_id").size().to_dict() == {"har_log": 63 - 22, "rw": 63 - 22}


def test_a_failed_fit_keeps_the_previous_fold_and_a_failed_first_fit_is_an_error(world):
    class Flaky(HistoricalMean):
        calls = 0

        def fit(self, y, horizons, previous):
            Flaky.calls += 1
            if Flaky.calls == 2:
                raise ValueError("forced")
            return super().fit(y, horizons, previous)

    fits = run(world, world["y"], world["dev"][-2:], [Flaky("flaky", None)]).fits
    assert fits["fallback"].tolist() == ["none", "previous_fold"]
    assert fits["params_json"].iloc[0] == fits["params_json"].iloc[1]  # the first fold's mean, kept
    Flaky.calls = 1  # the next call (the first fold) fails
    with pytest.raises(RuntimeError, match="no previous fold"):
        run(world, world["y"], world["dev"][-2:], [Flaky("flaky", None)])


# --- scoring --------------------------------------------------------------------------------


def test_metric_table_scores_dev_rows_and_the_median_only_for_log_models(world):
    out = run(world, world["y"], world["dev"][-1:], [RandomWalk(), Har()]).forecasts
    metric = walkforward.metric_table(out, world["proto"].holdout_start, 20, {"har_log"})
    assert set(zip(metric.model_id, metric.forecast)) == {("rw", "mean"), ("har_log", "mean"), ("har_log", "median")}
    assert (metric["period"] == "dev").all() and dict(zip(metric[metric.model_id == "rw"].horizon, metric[metric.model_id == "rw"].n)) == {1: 62, 5: 58, 21: 42}
    # The rw rmse by hand from the stored rows.
    rows = walkforward.scored(out[(out.model_id == "rw") & (out.horizon == 5)], world["proto"].holdout_start)
    expected = float(np.sqrt(np.mean((rows["y"] - rows["yhat"]) ** 2)))
    assert metric[(metric.model_id == "rw") & (metric.horizon == 5)]["rmse"].iloc[0] == pytest.approx(expected, rel=1e-15)


# --- the command line, end to end on synthetic files ------------------------------------------


def test_baselines_command_on_the_synthetic_pipeline(tmp_path, monkeypatch, capsys):
    from ivf.data import cli as data_cli
    from ivf.models import cli as models_cli

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    for step in ("fetch", "build-interim", "check"):
        assert data_cli.main(["--synthetic", step]) == 0
    capsys.readouterr()
    rc = models_cli.main(["--synthetic", "baselines", "--symbols", "VIX", "VIX3M", "--models", "rw", "hist_mean", "har_log"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "97 dev folds" in out and "no row on or after it is read" in out
    root = tmp_path / "implied-vol-forecasting" / "synthetic" / "processed"
    runs = pd.read_parquet(root / "runs.parquet")
    folder = root / f"run_id={runs.run_id.iloc[0]}"
    tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in ("fold", "fit", "forecast", "metric", "calendar", "run_input")}
    assert all(t.columns[0] == "run_id" for t in tables.values())
    forecast = tables["forecast"]
    # VIX: 97 folds x 63 origins x 3 horizons x 3 models = 54,999 rows. VIX3M starts 2009-09-18, so
    # its first fitted fold is the first with 504 closes <= the cutoff; it has fewer rows.
    assert (forecast.target_id == "vix_level").sum() == 97 * 63 * 3 * 3
    assert 0 < (forecast.target_id == "vix3m_level").sum() < 97 * 63 * 3 * 3
    assert forecast["origin_date"].max() < date(2024, 7, 1) and len(tables["fold"]) == len(set(tables["calendar"].fold_id.dropna()))
    assert np.allclose(forecast["yhat"], 100 * np.sqrt(forecast["v_hat"]), rtol=1e-12)  # yhat = 100*sqrt(v_hat) on every row
    fit = tables["fit"].dropna(subset=["train_last_target_date"])
    assert (fit["train_last_target_date"] <= fit["cutoff_date"]).all()
    assert "y" not in forecast.columns  # targets live in target_value, not in the forecast store
    assert set(tables["metric"].period) == {"dev"}
