"""Milestone 4: the LSTM (``lstm_log``) on the vix_level and rv21 tracks.

What these tests pin: the features are trailing (L1), the training and validation slices end by the cutoff
and are separated by H sessions (L2 and the validation gap), the scaler sees the training slice only,
the back-transform is M2's lognormal mean, the seeds combine as stated, the refit schedule, determinism
(same inputs, same bytes, in this process or a worker), the holdout, and the ``lstm`` command.

Tiny networks (hidden 2-4, 1-3 epochs) keep the file fast. Synthetic data only.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from ivf import protocol
from ivf.data import calendar, synthetic
from ivf.evaluation import walkforward
from ivf.features import build
from ivf.models import baselines, lstm, registry, rv21

pytestmark = pytest.mark.usefixtures("no_network")


ROOT = 20261028  # protocol.seed (configs/protocol.yaml and the synthetic protocol)
DAYS = [date(2000, 1, 3) + timedelta(days=i) for i in range(2000)]  # dates for the bare-array tests


def tiny(**kw) -> lstm.LstmSettings:
    base = dict(members=1, window=22, hidden=4, batch=64, lr=1e-3, max_epochs=1, patience=5, val_sessions=50, refit_every=252)
    return lstm.LstmSettings(**{**base, **kw})


def ar_series(n: int = 700, seed: int = 0) -> np.ndarray:
    """A VIX-like level: 20 exp(x), x an AR(1) with phi 0.98."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + 0.05 * rng.standard_normal()
    return 20 * np.exp(x)


def rv_inputs(n: int = 700, seed: int = 1) -> rv21.Rv21Inputs:
    rng = np.random.default_rng(seed)
    return rv21.Rv21Inputs(rng.normal(0, 0.01, n), ar_series(n, seed + 1))


@pytest.fixture(scope="module")
def world():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, price = synthetic.synthetic_market(proto)
    cal = calendar.master_calendar(proto)
    calendar_dates = list(cal["date"])
    folds = walkforward.fold_schedule(calendar_dates, proto.dev_start, proto.holdout_start, proto.fold_len, proto.data_end)
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    return {
        "proto": proto, "iv": iv, "price": price, "cal": cal, "calendar_dates": calendar_dates, "folds": folds,
        "dev": [f for f in folds if f.period == "dev"], "dates": dates, "vix": build.align(dates, iv_pre, "symbol", "VIX"),
        "inputs": rv21.Rv21Inputs(build.log_returns(build.align(dates, price_pre, "ticker", "^GSPC")), build.align(dates, iv_pre, "symbol", "VIX")),
    }


def run_vix(world, y, folds, models):
    return walkforward.run_symbol("VIX", "vix_level", y, world["dates"][: len(y)], world["calendar_dates"], folds, models,
                                  world["proto"].horizons, world["proto"].holdout_start, 504)


def run_rv(world, inputs, folds, models):
    return walkforward.run_rv21(inputs, world["dates"][: len(inputs)], world["calendar_dates"], folds, models, world["proto"].holdout_start, 504)


COLS = ["yhat", "yhat_naive", "mu_log", "s2_log", "n_members", "fold_id", "target_date"]


# --- features, targets and the split, by hand ----------------------------------------------------------------


def test_lagged_by_hand():
    out = lstm.lagged(np.array([1.0, 2.0, 3.0]), 2)
    np.testing.assert_array_equal(out, [[np.nan, 1], [1, 2], [2, 3]])


def test_vix_features_by_hand():
    # y = exp(t), so x = ln y = t. At t = 25: the sequence is x_4..x_25; xw = mean(21..25) = 23;
    # xm = mean(4..25) = (4 + 25)/2 = 14.5; the anchor is x_25 = 25.
    y = np.exp(np.arange(30.0))
    seq, static, anchor = lstm.vix_features(y, 22)
    np.testing.assert_allclose(seq[25, :, 0], np.arange(4.0, 26.0))
    np.testing.assert_allclose(static[25], [25, 23, 14.5])
    assert anchor[25] == pytest.approx(25)
    assert np.isnan(seq[20]).any() and np.isnan(static[20, 2])  # t = 20 has no x_{-1}: not usable


def test_rv_features_by_hand():
    # A constant 1% return: rv5 = rv21 = 100 sqrt(252 x 0.0001) = 100 x 0.158745 = 15.8745; the sequence is 1.0 (percent).
    r = np.full(40, 0.01)
    seq, static, anchor = lstm.rv_features(rv21.Rv21Inputs(r, np.full(40, 20.0)), 22)
    rv = 100 * math.sqrt(252 * 0.0001)
    np.testing.assert_allclose(seq[30, :, 0], 1.0)
    np.testing.assert_allclose(static[30], [math.log(20), math.log(rv), math.log(rv)])
    assert anchor[30] == pytest.approx(math.log(rv))
    assert np.isnan(anchor[19]) and np.isfinite(anchor[20])  # rv21 needs the 21 returns 0..20


def test_log_targets_by_hand():
    # level 10, 20, 40, 80 and the anchor ln level: every one-step log change is ln 2, every two-step ln 4.
    level = np.array([10.0, 20.0, 40.0, 80.0])
    out = lstm.log_targets(level, np.log(level), (1, 2))
    np.testing.assert_allclose(out[:2], [[math.log(2), math.log(4)]] * 2)
    assert out[2, 0] == pytest.approx(math.log(2)) and np.isnan(out[2, 1]) and np.isnan(out[3]).all()


def test_split_by_hand():
    # c = 100, H = 21, 10 validation sessions: last usable origin 79 (target dated 100 = c); validation 70..79;
    # training 0..49 (its last target, 49 + 21 = 70, is dated at the first validation origin: gap 21 >= every h).
    train, val = lstm.split(100, 21, 10)
    assert (train[0], train[-1], len(train)) == (0, 49, 50)
    assert (val[0], val[-1], len(val)) == (70, 79, 10)
    assert val[0] - train[-1] == 21 and val[-1] + 21 == 100
    with pytest.raises(ValueError, match="no training origin"):
        lstm.split(40, 21, 10)  # tau_max 19, v0 10, last training origin -11


def test_member_seeds_follow_adr_0001_d5():
    # key32("har_log") is the first 4 bytes of sha256("har_log"), big-endian; ADR 0001 D5 measured the first draw of
    # SeedSequence(20261028, spawn_key=(key32("har_log"), 20240701, 1)) as 0.45037008942388645 in two processes.
    assert lstm.key32("har_log") == 3421073489
    sequence = np.random.SeedSequence(ROOT, spawn_key=(lstm.key32("har_log"), 20240701, 1))
    assert np.random.default_rng(sequence).random() == 0.45037008942388645
    seed = lstm.member_seed(ROOT, "har_log", date(2024, 7, 1), 1)
    assert seed == int(np.random.SeedSequence(ROOT, spawn_key=(3421073489, 20240701, 1)).generate_state(1, dtype=np.uint32)[0])
    others = {lstm.member_seed(ROOT, "har_log", date(2024, 7, 1), 2), lstm.member_seed(ROOT, "har_log", date(2024, 7, 2), 1),
              lstm.member_seed(ROOT, "lstm_log", date(2024, 7, 1), 1), lstm.member_seed(ROOT + 1, "har_log", date(2024, 7, 1), 1)}
    assert seed not in others and len(others) == 4 and 0 <= seed < 2**32


# --- the scaler, the validation gap and early stopping (behaviour) -------------------------------------------


def test_the_scaler_is_fitted_on_the_training_slice_only():
    y = ar_series()
    c = len(y) - 1
    train, val = lstm.split(c, 21, 50)  # training 0..608, validation 629..678
    fit = lstm.LstmLog(tiny(), ROOT, DAYS).fit(y, (1, 5, 21), None)
    sc = fit.params["scaler"]
    x = np.log(y)
    used = train[train >= 21]  # origins 0..20 lack x_{t-21}
    # x_t (static column 0) over the training origins, and the h = 21 log change over the same origins:
    assert sc["static_mean"][0] == pytest.approx(x[used].mean(), rel=1e-12)
    assert sc["target_mean"][2] == pytest.approx((x[used + 21] - x[used]).mean(), rel=1e-12)
    assert fit.n_train == len(used) and fit.params["last_train_origin"] == 608 and fit.params["first_val_origin"] == 629


def test_values_only_validation_reads_change_neither_the_scaler_nor_the_weights_and_the_last_training_target_does():
    # Training origins end at 608 = v0 - 21 and read y up to v0 = 629 (the last training target). y[630..c] is read
    # by the validation slice only. One epoch, so the validation loss chooses nothing: the weights see training only.
    y = ar_series()
    c = len(y) - 1
    _, val = lstm.split(c, 21, 50)
    v0 = val[0]
    base = lstm.LstmLog(tiny(), ROOT, DAYS).fit(y, (1, 5, 21), None)
    late = y.copy()
    late[v0 + 1 :] *= 3.0
    poisoned = lstm.LstmLog(tiny(), ROOT, DAYS).fit(late, (1, 5, 21), None)
    assert poisoned.params["scaler"] == base.params["scaler"]
    assert poisoned.params["members"][0]["weights_sha256"] == base.params["members"][0]["weights_sha256"]
    assert poisoned.params["s2"] != base.params["s2"]  # ... while validation did see the poison
    edge = y.copy()
    edge[v0] *= 3.0  # the target of training origin 608 at h = 21
    moved = lstm.LstmLog(tiny(), ROOT, DAYS).fit(edge, (1, 5, 21), None)
    assert moved.params["members"][0]["weights_sha256"] != base.params["members"][0]["weights_sha256"]


def test_rv21_training_never_reads_a_return_after_the_first_validation_origin():
    # rv21 at tau + 21 is the window tau+1..tau+21: the last training origin v0 - 21 reads returns up to v0.
    inputs = rv_inputs()
    c = len(inputs) - 1
    _, val = lstm.split(c, 21, 50)
    v0 = val[0]
    base = lstm.LstmRv(tiny(), ROOT, DAYS).fit(inputs, None)
    r = inputs.r.copy()
    r[v0 + 1 :] *= 5.0
    poisoned = lstm.LstmRv(tiny(), ROOT, DAYS).fit(rv21.Rv21Inputs(r, inputs.vix), None)
    assert poisoned.params["members"][0]["weights_sha256"] == base.params["members"][0]["weights_sha256"]
    assert poisoned.params["scaler"] == base.params["scaler"]
    r = inputs.r.copy()
    r[v0] *= 5.0
    moved = lstm.LstmRv(tiny(), ROOT, DAYS).fit(rv21.Rv21Inputs(r, inputs.vix), None)
    assert moved.params["members"][0]["weights_sha256"] != base.params["members"][0]["weights_sha256"]
    assert base.train_last_origin == c - 21 and base.train_last_target == c


def test_early_stopping_keeps_the_best_epoch_and_stops_after_patience():
    y = ar_series()
    s = tiny(max_epochs=12, patience=2, lr=0.05)
    model = lstm.LstmLog(s, ROOT, DAYS)
    fit = model.fit(y, (1, 5, 21), None)
    m = fit.params["members"][0]
    assert m["epochs_run"] == m["best_epoch"] + 2 or (m["epochs_run"] == 12 and m["best_epoch"] > 10)
    # The kept weights are the best epoch's: their validation MSE (standardised) is the recorded minimum.
    net = model._nets[fit.params["net"]]
    sc = net["scaler"]
    _, val = lstm.split(len(y) - 1, 21, 50)
    seq, static, anchor = lstm.vix_features(y, 22)
    target = lstm.log_targets(y, anchor, (1, 5, 21))
    out = lstm.predict(net["states"][0], net["shape"], *sc.inputs(seq[val], static[val]))
    mse = float(np.mean((out - (target[val] - sc.target_mean) / sc.target_sd) ** 2))
    assert mse == pytest.approx(m["val_mse"], rel=1e-5)


# --- output, back-transform and seeds ------------------------------------------------------------------------


def test_back_transform_and_seed_combination():
    # mu = the mean of the members' mu; s2 = the validation MSE of that mean; yhat = exp(mu + s2/2) (scored),
    # yhat_naive = exp(mu); each member's yhat = exp(mu_m + s2_m/2).
    y = ar_series()
    model = lstm.LstmLog(tiny(members=3), ROOT, DAYS)
    fit = model.fit(y[:650], (1, 5, 21), None)
    origins = np.arange(649 - 5, 650)
    out = model.forecast(fit, y[:650], origins, (1, 5, 21))
    seq, static, anchor = lstm.vix_features(y[:650], 22)
    mu_m = model._mu(fit, seq, static, anchor, origins)
    for j, h in enumerate((1, 5, 21)):
        fc, s2 = out[h], fit.params["s2"][j]
        np.testing.assert_allclose(fc.mu_log, mu_m[:, :, j].mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(fc.yhat, np.exp(fc.mu_log + s2 / 2), rtol=1e-12)
        np.testing.assert_allclose(fc.yhat_naive, np.exp(fc.mu_log), rtol=1e-12)
        own = np.array([row[j] for row in fit.params["s2_members"]])[:, None]
        np.testing.assert_allclose(fc.members, np.exp(mu_m[:, :, j] + own / 2), rtol=1e-12)
    # s2 is the validation MSE of the ensemble mean (in log points), re-derived through the forecast path:
    _, val = lstm.split(649, 21, 50)
    target = lstm.log_targets(y[:650], anchor, (1, 5, 21))
    mean_mu = model._mu(fit, seq, static, anchor, val).mean(axis=0) - anchor[val][:, None]
    np.testing.assert_allclose(fit.params["s2"], ((target[val] - mean_mu) ** 2).mean(axis=0), rtol=1e-10)
    assert len({m["weights_sha256"] for m in fit.params["members"]}) == 3  # three different networks


def test_determinism_same_inputs_same_bytes_and_the_thread_count_is_restored():
    y = ar_series()
    torch.set_num_threads(3)
    a, b = lstm.LstmLog(tiny(max_epochs=2), ROOT, DAYS), lstm.LstmLog(tiny(max_epochs=2), ROOT, DAYS)
    fa, fb = a.fit(y, (1, 5, 21), None), b.fit(y, (1, 5, 21), None)
    assert torch.get_num_threads() == 3 and not torch.are_deterministic_algorithms_enabled()
    assert fa.params == fb.params
    origins = np.arange(600, 650)
    for h in (1, 5, 21):
        assert a.forecast(fa, y, origins, (1, 5, 21))[h].yhat.tobytes() == b.forecast(fb, y, origins, (1, 5, 21))[h].yhat.tobytes()
    other = lstm.LstmLog(tiny(max_epochs=2), ROOT + 1, DAYS).fit(y, (1, 5, 21), None)  # another protocol.seed
    assert other.params["members"][0]["weights_sha256"] != fa.params["members"][0]["weights_sha256"]


def test_a_worker_pool_gives_the_same_bytes_as_this_process():
    inputs = rv_inputs()
    here = lstm.LstmRv(tiny(members=2), ROOT, DAYS)
    pool = lstm.LstmRv(tiny(members=2), ROOT, DAYS, workers=2)
    try:
        fa, fb = here.fit(inputs, None), pool.fit(inputs, None)
    finally:
        pool.close()
    assert fa.params == fb.params
    origins = np.arange(650, 700)
    assert here.forecast(fa, inputs, origins).yhat.tobytes() == pool.forecast(fb, inputs, origins).yhat.tobytes()


# --- the runner: L2, poisoned future, truncation, refit schedule, same rows, holdout ---------------------------


def test_poisoned_closes_after_the_cutoff_change_no_fit_and_no_forecast_at_the_cutoff(world):
    fold = world["dev"][0]
    c = fold.cutoff_idx
    y = world["vix"].copy()
    y[c + 1 :] = 300.0
    a = run_vix(world, world["vix"], [fold], [lstm.LstmLog(tiny(), ROOT, world["dates"])])
    b = run_vix(world, y, [fold], [lstm.LstmLog(tiny(), ROOT, world["dates"])])
    assert a.fits["params_json"].tolist() == b.fits["params_json"].tolist()
    first = world["dates"][c]
    pd.testing.assert_frame_equal(a.forecasts.loc[a.forecasts.origin_date == first, COLS].reset_index(drop=True),
                                  b.forecasts.loc[b.forecasts.origin_date == first, COLS].reset_index(drop=True))
    later = world["dates"][c + 5]
    assert not np.allclose(a.forecasts.loc[a.forecasts.origin_date == later, "yhat"], b.forecasts.loc[b.forecasts.origin_date == later, "yhat"])
    assert (a.fits["train_last_target_date"] == a.fits["cutoff_date"]).all()  # the last validation target is dated c


def test_truncation_forecasts_at_t_do_not_depend_on_data_after_t(world):
    # ADR 0001 4.7: a run on data truncated at t gives the same forecast at t, bit for bit (predict is row by row).
    # refit_every 126: fold 0 is trained, fold 1 applies fold 0's network FROZEN, so t = c - 1 checks a trained fold
    # and t = c .. c + 30 a frozen one (amended M4, review A: refit_every was 63, so no frozen fold was ever truncated).
    folds = world["dev"][:2]
    c = folds[1].cutoff_idx
    run = run_vix(world, world["vix"], folds, [lstm.LstmLog(tiny(refit_every=126), ROOT, world["dates"])])
    assert [json.loads(p)["refit"] for p in run.fits["params_json"]] == [True, False]
    full = run.forecasts.set_index(["origin_date", "horizon"])
    for t in (c - 1, c, c + 1, c + 30):
        part = run_vix(world, world["vix"][: t + 1], folds, [lstm.LstmLog(tiny(refit_every=126), ROOT, world["dates"])]).forecasts
        at_t = part[part.origin_date == world["dates"][t]].set_index(["origin_date", "horizon"])
        assert len(at_t) == 3
        pd.testing.assert_frame_equal(at_t[COLS], full.loc[at_t.index, COLS], rtol=0, atol=0)


def test_rv21_truncation_and_the_window_target_rule(world):
    # Fold 0 trained, fold 1 frozen (refit_every 126; amended M4, review A): truncation at t in each, bit for bit.
    folds = world["dev"][:2]
    c0, c1 = folds[0].cutoff_idx, folds[1].cutoff_idx
    full = run_rv(world, world["inputs"], folds, [lstm.LstmRv(tiny(refit_every=126), ROOT, world["dates"])])
    assert [json.loads(p)["refit"] for p in full.fits["params_json"]] == [True, False]
    for t in (c0 + 10, c1, c1 + 10):
        part = run_rv(world, world["inputs"].upto(t), folds, [lstm.LstmRv(tiny(refit_every=126), ROOT, world["dates"])]).forecasts
        at_t = part[part.origin_date == world["dates"][t]].reset_index(drop=True)
        assert len(at_t) == 1
        pd.testing.assert_frame_equal(at_t[COLS], full.forecasts.loc[full.forecasts.origin_date == world["dates"][t], COLS].reset_index(drop=True),
                                      rtol=0, atol=0)
    fit = full.fits.iloc[0]
    assert fit["train_last_target_date"] == fit["cutoff_date"] and json.loads(fit["params_json"])["trained_cutoff"] == c0


def test_refit_schedule_freezes_the_network_between_training_cutoffs(world):
    # refit_every 126 with 63-session folds: train at folds 0 and 2, apply fold 0's network at fold 1.
    folds = world["dev"][:3]
    model = lstm.LstmLog(tiny(refit_every=126), ROOT, world["dates"])
    out = run_vix(world, world["vix"], folds, [model, baselines.RandomWalk()])
    fits = out.fits[out.fits.model_id == "lstm_log"]
    params = [json.loads(p) for p in fits["params_json"]]
    assert [p["refit"] for p in params] == [True, False, True]
    assert [p["trained_cutoff"] for p in params] == [folds[0].cutoff_idx, folds[0].cutoff_idx, folds[2].cutoff_idx]
    assert params[1]["members"] == params[0]["members"] and params[1]["net"] == params[0]["net"]
    assert fits["train_last_target_date"].tolist()[1] == world["dates"][folds[0].cutoff_idx]
    # The frozen network forecasts every origin of fold 1, the same rows as rw.
    fc = out.forecasts
    assert len(fc[(fc.model_id == "lstm_log") & (fc.fold_id == folds[1].fold_id)]) == 63 * 3
    walkforward.check_same_rows(fc, ["lstm_log", "rw"])


def test_a_frozen_fold_applies_the_trained_network_scaler_and_s2_exactly(world):
    # Review A's P6 (amended M4): fold 1's mu and s2 equal fold 0's networks, its STORED scaler (params_json) and its
    # s2 applied by hand, exactly. A frozen fold that refitted the scaler, or s2, at its own cutoff would differ.
    folds = world["dev"][:2]
    model = lstm.LstmLog(tiny(members=2, refit_every=126), ROOT, world["dates"])
    out = run_vix(world, world["vix"], folds, [model])
    p0, p1 = (json.loads(p) for p in out.fits.drop_duplicates("fold_id")["params_json"])
    assert (p0["refit"], p1["refit"]) == (True, False)
    scaler = lstm.Scaler(*[np.array(p0["scaler"][k]) for k in ("seq_mean", "seq_sd", "static_mean", "static_sd", "target_mean", "target_sd")])
    net = model._nets[p0["net"]]
    seq, static, anchor = lstm.vix_features(world["vix"], 22)
    o = np.arange(folds[1].cutoff_idx, folds[1].last_origin_idx + 1)
    x_seq, x_static = scaler.inputs(seq[o], static[o])
    out_m = np.stack([lstm.predict(state, net["shape"], x_seq, x_static) for state in net["states"]])
    mu = (out_m * scaler.target_sd + scaler.target_mean + anchor[o][None, :, None]).mean(axis=0)
    fc = out.forecasts[out.forecasts.fold_id == folds[1].fold_id]
    assert len(fc) == 63 * 3
    for j, h in enumerate((1, 5, 21)):
        g = fc[fc.horizon == h]
        assert list(g["origin_date"]) == [world["dates"][i] for i in o]
        np.testing.assert_array_equal(g["mu_log"].to_numpy(), mu[:, j])
        np.testing.assert_array_equal(g["s2_log"].to_numpy(), np.full(len(o), p0["s2"][j]))


def test_one_fit_row_per_member_and_n_members_on_every_forecast(world):
    out = run_rv(world, world["inputs"], world["dev"][:1], [lstm.LstmRv(tiny(members=2), ROOT, world["dates"]), rv21.RandomWalkRv()])
    fits = out.fits.set_index(["model_id", "member"])
    cutoff = world["dates"][world["dev"][0].cutoff_idx]
    assert sorted(fits.loc["lstm_log"].index) == [1, 2]
    assert [fits.loc[("lstm_log", m), "seed"] for m in (1, 2)] == [lstm.member_seed(ROOT, "lstm_log", cutoff, m) for m in (1, 2)]
    assert pd.isna(fits.loc[("rw", 0), "seed"])
    fc = out.forecasts
    assert set(fc.loc[fc.model_id == "lstm_log", "n_members"]) == {2} and set(fc.loc[fc.model_id == "rw", "n_members"]) == {1}


def test_poisoning_every_holdout_row_changes_nothing(world):
    hs = world["proto"].holdout_start
    price, iv = world["price"].copy(), world["iv"].copy()
    price.loc[price["date"] >= hs, "close"] *= 100
    iv.loc[iv["date"] >= hs, "close"] *= 100
    dates, _, iv_pre, price_pre = walkforward.pre_holdout_inputs(world["cal"], iv, price, hs)
    assert dates == world["dates"]
    fold = world["dev"][-1:]
    a = run_vix(world, world["vix"], fold, [lstm.LstmLog(tiny(), ROOT, world["dates"])]).forecasts
    b = run_vix(world, build.align(dates, iv_pre, "symbol", "VIX"), fold, [lstm.LstmLog(tiny(), ROOT, world["dates"])]).forecasts
    pd.testing.assert_frame_equal(a.drop(columns="member_yhat"), b.drop(columns="member_yhat"))
    inputs = rv21.Rv21Inputs(build.log_returns(build.align(dates, price_pre, "ticker", "^GSPC")), build.align(dates, iv_pre, "symbol", "VIX"))
    ra = run_rv(world, world["inputs"], fold, [lstm.LstmRv(tiny(), ROOT, world["dates"])]).forecasts
    rb = run_rv(world, inputs, fold, [lstm.LstmRv(tiny(), ROOT, world["dates"])]).forecasts
    pd.testing.assert_frame_equal(ra.drop(columns="member_yhat"), rb.drop(columns="member_yhat"))
    assert ra["origin_date"].max() < hs and ra["y"].isna().sum() == 21  # the last 21 origins target the holdout: not scored


# --- the registry and the command ----------------------------------------------------------------------------


def test_the_committed_lstm_settings_and_their_strict_reading(tmp_path):
    s = registry.load_lstm_settings()
    assert (s.members, s.window, s.hidden, s.val_sessions, s.refit_every) == (5, 22, 32, 252, 252)
    defs, _ = registry.load_models()
    assert registry.models_for(defs, "rv21", 4)[-1] == "lstm_log" and "lstm_log" not in registry.models_for(defs, "rv21", 3)
    text = (registry.config.configs_dir() / registry.MODELS_FILE).read_text()
    for old, new in (("window: 22", "window: 23"), ("members: 5", "members: 0"), ("hidden: 32", "hidden: 32\n  extra: 1")):
        bad = tmp_path / "models.yaml"
        bad.write_text(text.replace(old, new))
        with pytest.raises(protocol.ProtocolError):
            registry.load_lstm_settings(bad)


def test_lstm_command_refuses_without_the_other_models_then_merges_on_the_same_rows(tmp_path, monkeypatch, capsys):
    from ivf.data import cli as data_cli
    from ivf.models import cli as models_cli

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    for step in ("fetch", "build-interim", "check"):
        assert data_cli.main(["--synthetic", step]) == 0
    # Tiny settings, trained once (refit_every far beyond the dev period) so the command runs in seconds.
    monkeypatch.setattr(registry, "load_lstm_settings", lambda path=None: tiny(members=2, hidden=2, refit_every=10**6))
    assert models_cli.main(["--synthetic", "lstm", "--workers", "0"]) == 3  # no garch run yet in this code state
    assert models_cli.main(["--synthetic", "garch", "--models", "rw", "har_rv"]) == 0
    capsys.readouterr()
    # The full expected list (arima_log, ... vix_implied) is not in the run: the same-rows check refuses.
    assert models_cli.main(["--synthetic", "lstm", "--workers", "0"]) == 1
    assert "same-rows rule" in capsys.readouterr().err
    real = registry.models_for
    monkeypatch.setattr(registry, "models_for", lambda defs, t, m: [x for x in real(defs, t, m) if x in ("rw", "har_rv", "lstm_log")])
    rc = models_cli.main(["--synthetic", "lstm", "--workers", "0"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "no row on or after it is read" in out and "0 with train_last_target_date after the cutoff" in out
    assert "(1 trained, 96 frozen)" in out
    root = tmp_path / "implied-vol-forecasting" / "synthetic" / "processed"
    folder = root / f"run_id={pd.read_parquet(root / 'runs.parquet').run_id.iloc[0]}"
    forecast = pd.read_parquet(folder / "forecast.parquet")
    counts = forecast.groupby(["target_id", "model_id"]).size().to_dict()
    assert counts[("vix_level", "lstm_log")] == counts[("vix_level", "rw")] == 97 * 63 * 3
    assert counts[("rv21", "lstm_log")] == counts[("rv21", "har_rv")] == 97 * 63
    assert forecast["origin_date"].max() < date(2024, 7, 1) and set(forecast.loc[forecast.model_id == "lstm_log", "n_members"]) == {2}
    members = pd.read_parquet(folder / "forecast_member.parquet")
    assert len(members) == 2 * (97 * 63 * 4) and set(members["member"]) == {1, 2}
    fit = pd.read_parquet(folder / "fit.parquet")
    lfit = fit[fit.model_id == "lstm_log"]
    first = lfit["cutoff_date"].min()  # trained once, at the first cutoff; both tracks share model id, cutoff and seeds
    assert len(lfit) == 2 * 97 * 2 and set(lfit["member"]) == {1, 2}
    assert set(lfit["seed"]) == {lstm.member_seed(ROOT, "lstm_log", first, m) for m in (1, 2)}
    metric = pd.read_parquet(folder / "metric.parquet")
    assert {"rw", "har_rv", "lstm_log"} <= set(metric.loc[metric.target_id == "rv21", "model_id"])
    assert set(metric.loc[(metric.model_id == "lstm_log") & (metric.target_id == "vix_level"), "forecast"]) == {"mean", "median"}
