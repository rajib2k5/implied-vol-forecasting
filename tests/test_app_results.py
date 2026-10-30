"""Tests for ivf.app_results: the payloads of the app's result views, from a hand-made store.

Expected values are worked out by hand from ``tests/app_store_fixture.py`` (the arithmetic is in
the comments), never recomputed with the code under test.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app_store_fixture import HOLDOUT, SESSIONS, tiny_store
from ivf import app_results as ar


@pytest.fixture
def store():
    return tiny_store()


# --- model_race ---------------------------------------------------------------


def test_race_shows_the_last_race_sessions_dev_origins_of_each_panel(store):
    payload = ar.model_race_payload(store)
    # Origins available: h=1 k=0..28, h=5 k=0..24, h=21 k=0..8; the last 3 are 26-28, 22-24 and 6-8.
    # The calendar starts at the earliest shown origin (k=6) and ends at holdout_start:
    # sessions 6..29 (24 dates) + 2024-07-01 = 25 dates, so holdout_start_idx = 24.
    assert payload["calendar"][0] == SESSIONS[6].isoformat()
    assert payload["calendar"][-1] == "2024-07-01"
    assert len(payload["calendar"]) == 25 and payload["holdout_start_idx"] == 24
    got = [(p["target_id"], p["horizon"], p["origin_idx0"], p["n"]) for p in payload["panels"]]
    assert got == [("vix_level", 1, 20, 3), ("vix_level", 5, 16, 3), ("vix_level", 21, 0, 3), ("rv21", 21, 0, 3)]


def test_race_actuals_and_forecasts_are_the_stores_values_at_the_right_rows(store):
    h1 = ar.model_race_payload(store)["panels"][0]
    # Actual at target k+1 for origins 26..28: 20 + 27/8, 20 + 28/8, 20 + 29/8.
    assert h1["actual"] == [23.375, 23.5, 23.625]
    # Registry order, milestone <= 4, not retired: rw then har_log (old_model retired, future_model is M5).
    assert [s["model_id"] for s in h1["series"]] == ["rw", "har_log"]
    # har_log = y at the origin + 0.5: 20 + 26/8 + 0.5 = 23.75, then 23.875, 24.0.
    assert h1["series"][1]["yhat"] == [23.75, 23.875, 24.0]
    rv = ar.model_race_payload(store)["panels"][3]
    assert [s["model_id"] for s in rv["series"]] == ["rw", "garch"]


def test_race_mz_rows_are_the_primary_level_mean_rows_rounded(store):
    mz = ar.model_race_payload(store)["panels"][0]["mz"]
    # Only hansen_hodrick / level / mean survive; alpha 0.123456 -> 0.1235, beta 0.987654 -> 0.9877.
    assert mz == [
        {"model_id": "har_log", "period": "dev", "n": 100, "alpha": 0.1235, "beta": 0.9877, "wald_p": 0.42},
        {"model_id": "rw", "period": "dev", "n": 100, "alpha": 0.1235, "beta": 0.9877, "wald_p": 0.42},
    ]


def test_race_refuses_a_holdout_row_even_if_the_store_hands_it_one(store):
    scored = store.scored
    late = scored.iloc[[0]].assign(origin_date=HOLDOUT, target_date=HOLDOUT)
    with pytest.raises(ValueError, match="on or after holdout_start"):
        ar.model_race_payload(tiny_store(scored=pd.concat([scored, late], ignore_index=True)))


def test_race_refuses_a_target_that_is_not_h_sessions_after_its_origin(store):
    scored = store.scored.copy()
    first_h1 = scored.index[(scored.target_id == "vix_level") & (scored.horizon == 1) & (scored.origin_date == SESSIONS[28])][0]
    scored.loc[first_h1, "target_date"] = SESSIONS[27]  # one session early
    with pytest.raises(ValueError, match="not 1 sessions after its origin"):
        ar.model_race_payload(tiny_store(scored=scored))


def test_race_panel_without_rows_names_the_missing_panel(store):
    scored = store.scored[store.scored.target_id != "rv21"]
    with pytest.raises(ar.StoreUnavailable, match="rv21 h=21"):
        ar.model_race_payload(tiny_store(scored=scored))


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        (lambda p: p["calendar"].reverse(), "not strictly increasing"),
        (lambda p: p["panels"][0].update(origin_idx0=23), "runs past the calendar|reaches holdout_start"),
        (lambda p: p["panels"][0].update(origin_idx0=21), "reaches holdout_start"),
        (lambda p: p["panels"][0]["actual"].pop(), "arrays of lengths"),
    ],
)
def test_check_model_race_catches_each_broken_invariant(store, change, complaint):
    payload = ar.model_race_payload(store)
    ar.check_model_race(payload)  # the unchanged payload passes
    change(payload)
    with pytest.raises(ValueError, match=complaint):
        ar.check_model_race(payload)


# --- evaluation ---------------------------------------------------------------


def test_losses_are_whole_period_mean_rows_with_ratios_to_rw(store):
    losses = ar.evaluation_payload(store)["losses"]
    assert [(x["target_id"], x["horizon"], x["model_id"]) for x in losses] == [
        ("vix_level", 1, "rw"), ("vix_level", 1, "har_log"), ("vix_level", 5, "rw"), ("vix_level", 5, "har_log"),
        ("vix_level", 21, "rw"), ("vix_level", 21, "har_log"), ("rv21", 21, "rw"), ("rv21", 21, "garch"),
    ]
    har = losses[1]
    # rmse 2.0 / rw 4.0 = 0.5; qlike 0.05 / 0.2 = 0.25; the regime (99) and median (77) rows are not used.
    assert (har["rmse"], har["rmse_ratio_rw"], har["qlike_ratio_rw"]) == (2.0, 0.5, 0.25)
    # bias 0.123456 -> 4 decimals 0.1235; mape 0.0512345 -> 5 decimals 0.05123; mae = rmse / 2 = 1.0.
    assert (har["bias"], har["mape"], har["mae"]) == (0.1235, 0.05123, 1.0)
    garch = losses[7]
    assert (garch["rmse_ratio_rw"], garch["qlike_ratio_rw"]) == (0.75, 0.5)  # 3/4 and 0.1/0.2


def test_dm_keeps_only_the_primary_fixed_b_pairs(store):
    payload = ar.evaluation_payload(store)
    # 3 horizons x 2 losses primary rows; rectangular, calibration-family and regime rows are dropped.
    assert len(payload["dm"]) == 6
    assert sorted((d["horizon"], d["loss"], d["p_holm"]) for d in payload["dm"]) == [
        (1, "qlike", 0.009), (1, "sq_error", 0.009), (5, "qlike", 0.009), (5, "sq_error", 0.009), (21, "qlike", 0.06), (21, "sq_error", 0.009)]
    assert (payload["dm_method"], payload["dm_fixed_b"], payload["mz_method"]) == ("fixed_b", 0.1, "hansen_hodrick")


def test_mz_and_calibration_summary(store):
    payload = ar.evaluation_payload(store)
    assert [(m["target_id"], m["model_id"], m["wald_p"]) for m in payload["mz"]] == [
        ("rv21", "garch", 0.42), ("vix_level", "har_log", 0.42), ("vix_level", "rw", 0.42)]
    assert payload["calibration"] == {"cells": 3, "helps": 1}  # helps = [False, True, False]


def test_check_evaluation_rejects_a_pair_whose_lower_loss_is_neither_model(store):
    payload = ar.evaluation_payload(store)
    ar.check_evaluation(payload)
    payload["dm"][0]["lower_loss"] = "garch"
    with pytest.raises(ValueError, match="neither model"):
        ar.check_evaluation(payload)
    payload["dm"][0].update(lower_loss="har_log", model_b="garch")  # garch has no vix_level loss row
    with pytest.raises(ValueError, match="no loss row"):
        ar.check_evaluation(payload)


# --- equity_curves -------------------------------------------------------------


def test_equity_shows_the_primary_and_the_three_benchmarks_at_the_default_cost(store):
    payload = ar.equity_curves_payload(store)
    assert [s["strategy_id"] for s in payload["strategies"]] == ["fvix_rv21_b2", "bench_long_vxx", "bench_long_svxy", "bench_cash"]
    primary = payload["strategies"][0]
    assert (primary["model_id"], primary["horizon"], primary["kind"]) == ("garch", 21, "forecast")
    assert payload["strategies"][1]["model_id"] is None and payload["strategies"][1]["horizon"] is None
    # The 5 bps rows (the 0 bps rows carry equity + 0.01); drawdown 0.99 / 1.1 - 1 = -0.1.
    assert primary["equity"] == [1.0, 1.1, 0.99]
    assert primary["drawdown"] == [0.0, 0.0, -0.1]
    assert payload["dates"] == [d.isoformat() for d in SESSIONS[-3:]] and payload["cost_bps"] == 5.0
    assert payload["holdout_start_idx"] is None


def test_summary_is_the_default_cost_and_cost_sensitivity_every_level(store):
    payload = ar.equity_curves_payload(store)
    assert [r["strategy_id"] for r in payload["summary"]] == ["fvix_rv21_b2", "bench_long_vxx", "bench_long_svxy", "bench_cash"]
    # fvix_rv21_b2 is strategy index 0 in the fixture: sharpe 0.5*0 - 5/10 = -0.5 at 5 bps.
    assert payload["summary"][0]["sharpe_ann"] == -0.5
    assert [(r["strategy_id"], r["cost_bps"]) for r in payload["cost_sensitivity"]][:2] == [("fvix_rv21_b2", 0.0), ("fvix_rv21_b2", 5.0)]
    assert len(payload["cost_sensitivity"]) == 8  # 4 strategies x 2 cost levels; fvix_rv21_b0 is not shown
    assert payload["cost_sensitivity"][0]["sharpe_ann"] == 0.0  # 0.5*0 - 0/10


def test_check_equity_curves_rejects_a_short_curve(store):
    payload = ar.equity_curves_payload(store)
    ar.check_equity_curves(payload)
    payload["strategies"][2]["equity"].pop()
    with pytest.raises(ValueError, match="bench_long_svxy"):
        ar.check_equity_curves(payload)


# --- eda ----------------------------------------------------------------------


def test_eda_band_is_1_96_over_root_n_and_ln_vix_drops_lag_0(store):
    payload = ar.eda_payload(store)
    by_id = {s["series_id"]: s for s in payload["acf"]}
    assert list(by_id) == ["r", "abs_r", "r2", "log_vix"]
    assert by_id["abs_r"]["band95"] == 0.0196  # 1.959964 / sqrt(10000) = 0.01959964
    assert by_id["log_vix"]["band95"] == 0.098  # 1.959964 / sqrt(400) = 0.0979982
    assert by_id["log_vix"]["acf"] == [0.98, 0.96] and by_id["log_vix"]["ljung_box_p_last"] is None
    assert by_id["r2"]["acf"] == [0.25, 0.1235] and by_id["r2"]["ljung_box_p_last"] == 0.0  # 1e-9 at 4 decimals
    assert by_id["r"]["ljung_box_p_last"] == 0.45  # the last lag's p
    assert payload["nlags"] == 2


def test_eda_half_life_keeps_an_unbounded_end_as_null(store):
    half = {h["sample"]: h for h in ar.eda_payload(store)["half_life"]}
    assert half["pre_holdout"]["half_life_sessions"] == 34.31  # 34.30961 at 2 decimals
    assert half["dev"]["half_life_ci95"] == [68.97, None]


def test_rounding_turns_non_finite_values_into_null():
    assert ar.r(float("nan"), 3) is None and ar.r(float("inf"), 3) is None and ar.r(None, 3) is None
    assert ar.r(1.23456, 3) == 1.235


# --- choosing the store run ----------------------------------------------------


def write_meta(folder, report_run, backtest_run, period="dev"):
    for sub, name, run in (("evaluation", "report_meta.json", report_run), ("backtest", "backtest_meta.json", backtest_run)):
        path = folder / sub / f"period={period}" / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"source_run_id": run, "period": period}))


def test_choose_store_run_takes_the_newest_run_with_both_dev_outputs_of_its_own(tmp_path):
    runs = pd.DataFrame({"run_id": ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"],
                         "started_at": pd.to_datetime(["2026-10-01", "2026-10-03", "2026-10-02"], utc=True)})
    runs.to_parquet(tmp_path / "runs.parquet")
    write_meta(tmp_path / "run_id=aaaaaaaaaaaa", "aaaaaaaaaaaa", "aaaaaaaaaaaa")
    write_meta(tmp_path / "run_id=bbbbbbbbbbbb", "bbbbbbbbbbbb", "aaaaaaaaaaaa")  # newest, but its backtest used another run
    write_meta(tmp_path / "run_id=cccccccccccc", "cccccccccccc", "cccccccccccc")
    assert ar.choose_store_run(tmp_path) == "cccccccccccc"


def test_choose_store_run_without_a_complete_run_names_the_steps(tmp_path):
    with pytest.raises(ar.StoreUnavailable, match="tools/build_synthetic_store.py"):
        ar.choose_store_run(tmp_path)


def test_loss_ratios_keep_6_decimals_so_near_ties_stay_apart(store):
    metric = store.metric.copy()
    pick = (metric.model_id == "har_log") & (metric.target_id == "vix_level") & (metric.horizon == 1) & (metric.scheme == "all") & (metric.forecast == "mean")
    metric.loc[pick, "rmse"] = 3.9801  # / rw 4.0 = 0.995025, which 4 decimals would round to 0.995
    har = next(x for x in ar.evaluation_payload(tiny_store(metric=metric))["losses"] if x["model_id"] == "har_log" and x["horizon"] == 1)
    assert har["rmse_ratio_rw"] == 0.995025
    assert har["rmse"] == 3.9801  # the loss itself keeps its 4 decimals
