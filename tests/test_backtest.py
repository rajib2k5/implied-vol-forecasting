"""Milestone 6: the VXX/SVXY backtest (ivf.backtest). Hand-worked P&L, rule L7 timing, costs, the dead
band, PSR/DSR against hand values, segment boundaries, and the holdout (poison and the synthetic look)."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from ivf import config, protocol
from ivf.backtest import cli, engine, metrics, segments as seg, sharpe, strategy
from ivf.backtest import data as bdata
from ivf.data import calendar, synthetic

PHI = NormalDist().cdf
ZINV = NormalDist().inv_cdf
GAMMA = 0.5772156649


def _sessions(n, start=date(2020, 1, 6)):
    return [start + timedelta(days=i) for i in range(n)]


def _returns(s, vxx, svxy):
    prices = pd.DataFrame([{"ticker": t, "date": d, "adj_close": p} for t, path in (("VXX", vxx), ("SVXY", svxy))
                           for d, p in zip(s, path)])
    return pd.DataFrame({t: seg.returns_on_sessions(prices, t, s) for t in ("VXX", "SVXY")})


# --- the engine: timing and P&L by hand -----------------------------------------------------------------

S7 = _sessions(7)
#            s0     s1     s2     s3     s4     s5      s6
VXX = [100.0, 100.0, 110.0, 99.0, 99.0, 108.9, 108.9]   # returns s1..s6: 0, +10%, -10%, 0, +10%, 0
SVXY = [50.0, 50.0, 45.0, 49.5, 49.5, 44.55, 49.005]    # returns s1..s6: 0, -10%, +10%, 0, -10%, +10%
SIGNAL = pd.Series([1.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0], index=S7)  # signal at the close of each origin


def test_pnl_by_hand_with_next_close_fills():
    pos = engine.positions(SIGNAL, S7, S7[2:], lag=2)
    out = engine.daily(pos, _returns(S7, VXX, SVXY), cost_bps=10.0)
    # The row dated d holds the signal of origin d-2 (filled at the close of d-1), cost 10 bp per side:
    # s2 <- s0 (+1): VXX r=+0.10, entry from cash (turnover 1, cost .001): (1-.001)(1.10)-1 = .0989
    # s3 <- s1 (+1): VXX r=-0.10, no change:                                 -0.10
    # s4 <- s2 (-1): SVXY r=0, switch (sell VXX, buy SVXY: turnover 2, .002): (1-.002)(1.0)-1 = -.002
    # s5 <- s3 (0):  cash, exit SVXY (turnover 1, .001):                      -.001
    # s6 <- s4 (-1): SVXY r=+0.10, entry (turnover 1, .001): (1-.001)(1.10)-1 = .0989
    assert list(out.origin_date) == S7[:5]
    assert list(out.signal) == [1, 1, -1, 0, -1]
    assert np.allclose(out.turnover, [1, 0, 2, 1, 1])
    assert np.allclose(out.net_ret, [0.0989, -0.10, -0.002, -0.001, 0.0989], atol=1e-12)
    # equity 1.0989, x0.9 = 0.98901, x0.998 = 0.98703198, x0.999 = 0.98604494802, x1.0989 = 1.083564793279...
    assert np.allclose(out.equity, [1.0989, 0.98901, 0.98703198, 0.98604494802, 0.98604494802 * 1.0989], atol=1e-12)
    # drawdown from the running peak (at least 1): s3 0.98901/1.0989-1 = -0.1
    assert out.drawdown.iloc[0] == 0.0 and out.drawdown.iloc[1] == pytest.approx(-0.1, abs=1e-12)
    assert out.drawdown.iloc[4] == pytest.approx(0.98604494802 * 1.0989 / 1.0989 - 1, abs=1e-12)


def test_a_same_close_or_same_day_fill_gives_a_different_book():
    rets = _returns(S7, VXX, SVXY)
    l7 = engine.daily(engine.positions(SIGNAL, S7, S7[2:], lag=2), rets, 10.0).net_ret.to_numpy()
    for lag in (0, 1):  # 0: the return of the signal's own day; 1: filled at the close the signal used
        other = engine.daily(engine.positions(SIGNAL, S7, S7[2:], lag=lag), rets, 10.0).net_ret.to_numpy()
        assert not np.allclose(l7, other)
    assert strategy.load_config().lag == 2  # the pre-registered config is rule L7


def test_config_refuses_a_lag_below_two(tmp_path):
    text = (config.configs_dir() / "backtest.yaml").read_text().replace("lag_sessions: 2", "lag_sessions: 1")
    (tmp_path / "b.yaml").write_text(text)
    with pytest.raises(protocol.ProtocolError, match="lag_sessions"):
        strategy.load_config(tmp_path / "b.yaml")


def test_costs_are_charged_on_every_change_of_position():
    rng = np.random.default_rng(3)
    s = _sessions(60)
    sig = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=60), index=s)
    rets = _returns(s, 100 * np.cumprod(1 + 0.01 * rng.standard_normal(60)), 50 * np.cumprod(1 + 0.01 * rng.standard_normal(60)))
    pos = engine.positions(sig, s, s[2:], lag=2)
    free, paid = engine.daily(pos, rets, 0.0), engine.daily(pos, rets, 20.0)
    w = np.vstack([np.zeros(2), pos[["w_vxx", "w_svxy"]].to_numpy()])
    sides = np.abs(np.diff(w, axis=0)).sum(axis=1)   # 1 for an entry or exit, 2 for a switch
    changed = sides > 0
    assert changed.sum() > 10 and (sides == 2).any()
    assert np.allclose(paid.cost, sides * 20.0 / 1e4)
    assert np.array_equal(paid.cost.to_numpy() > 0, changed)  # charged exactly where the position changes
    # net = (1 - c)(1 + gross) - 1 on change days; identical to the free book elsewhere
    assert np.allclose(paid.net_ret, (1 - sides * 0.002) * (1 + free.gross_ret) - 1, atol=1e-15)


def test_dead_band_is_cash_inside_and_at_the_edge():
    idx = _sessions(7)
    vix = pd.Series(20.0, index=idx)
    f = pd.Series([22.0, 22.0 + 1e-9, 18.0, 18.0 - 1e-9, 20.0, np.nan, 25.0], index=idx)
    out = strategy.band_signal(f, vix, band=2.0)
    # d = f - 20: 2 (at +band: cash), >2 (long VXX), -2 (cash), <-2 (long SVXY), 0, missing, 5
    assert out.iloc[:5].tolist() == [0.0, 1.0, 0.0, -1.0, 0.0] and math.isnan(out.iloc[5]) and out.iloc[6] == 1.0
    assert strategy.band_signal(f, vix, band=0.0).iloc[:5].tolist() == [1.0, 1.0, -1.0, -1.0, 0.0]


def test_a_missing_signal_stops_the_engine():
    sig = SIGNAL.copy()
    sig.iloc[2] = np.nan
    with pytest.raises(ValueError, match="no signal at origin"):
        engine.positions(sig, S7, S7[2:], lag=2)


def test_a_missing_return_while_holding_stops_the_engine():
    rets = _returns(S7, VXX, SVXY)
    rets.loc[S7[3], "VXX"] = np.nan  # the row dated s3 holds VXX (origin s1, +1)
    with pytest.raises(ValueError, match="no return on"):
        engine.daily(engine.positions(SIGNAL, S7, S7[2:], lag=2), rets, 10.0)
    rets.loc[S7[3], "VXX"], rets.loc[S7[3], "SVXY"] = 0.0, np.nan  # a missing return of the asset NOT held is harmless
    assert len(engine.daily(engine.positions(SIGNAL, S7, S7[2:], lag=2), rets, 10.0)) == 5


def test_the_signal_compares_the_forecast_with_the_vix_of_its_own_origin():
    cfg = strategy.load_config()
    days = _sessions(6)
    vix = pd.Series([15.0, 25.0, 15.0, 25.0, 15.0, 25.0], index=days)  # alternating, so a one-day shift flips the sign
    f = pd.DataFrame({"model_id": "m", "target_id": "rv21", "horizon": 21, "origin_date": days, "yhat": 20.0})
    sig = strategy.signals(cfg, {"rv21": "m", "vix_level": "x"}, f, vix, days)["fvix_rv21_b2"]
    # d_t = 20 - VIX_t: +5 -> long VXX, -5 -> long SVXY, on the origin's own VIX close
    assert sig.tolist() == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]


# --- PSR and DSR against hand values -------------------------------------------------------------------


def test_psr_by_hand_on_a_four_day_series():
    r = np.array([0.01, 0.01, 0.01, -0.01])
    # mean .005; deviations .005 x3, -.015; m2 = .0003/4 = 7.5e-5; m3 = (3*1.25e-7 - 3.375e-6)/4 = -7.5e-7
    # skew = m3/m2^1.5 = -2/sqrt(3); m4 = (3*6.25e-10 + 5.0625e-8)/4 = 1.3125e-8; Pearson kurt = m4/m2^2 = 7/3
    # SR = .005 / sqrt(.0003/3) = 0.5; PSR(0) = Phi(0.5*sqrt(3) / sqrt(1 + (2/sqrt 3)(0.5) + (7/3 - 1)/4 * 0.25))
    skew, kurt = sharpe.sample_moments(r)
    assert skew == pytest.approx(-2 / math.sqrt(3), rel=1e-12) and kurt == pytest.approx(7 / 3, rel=1e-12)
    hand = PHI(0.5 * math.sqrt(3) / math.sqrt(1 + 1 / math.sqrt(3) + (4 / 3) / 4 * 0.25))
    day = pd.DataFrame({"net_ret": r, "equity": np.cumprod(1 + r), "drawdown": 0.0, "turnover": 0.0, "signal": 1})
    s = metrics.summary(day)
    assert s["sr_daily"] == pytest.approx(0.5, rel=1e-12) and s["psr0"] == pytest.approx(hand, rel=1e-12)
    assert sharpe.probabilistic_sharpe_ratio(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(PHI(1 / math.sqrt(1.005)), rel=1e-12)


def test_dsr_by_hand():
    # N = 3 trials with per-day SRs .01, .03, .05: cross-trial variance (ddof 1) = .0004; T = 100: 1/T = .01 wins
    # E[max SR] = sqrt(.01) * ((1 - g) Z(1 - 1/3) + g Z(1 - 1/(3e))), g = Euler-Mascheroni; null mean 0
    e_max = 0.1 * ((1 - GAMMA) * ZINV(2 / 3) + GAMMA * ZINV(1 - 1 / (3 * math.e)))
    d = metrics.deflate(0.05, 100, 0.0, 3.0, [0.01, 0.03, 0.05], 3)
    assert d["var_sr_cross"] == pytest.approx(0.0004, rel=1e-12) and d["var_sr"] == pytest.approx(0.01, rel=1e-12)
    assert d["e_max_sr"] == pytest.approx(e_max, rel=1e-9)
    # DSR = PSR at SR* = E[max]: Phi((.05 - E) sqrt(99) / sqrt(1 + 3 * .05^2 / ... )), with g3 = 0, g4 = 3
    assert d["dsr"] == pytest.approx(PHI((0.05 - e_max) * math.sqrt(99) / math.sqrt(1 + 0.5 * 0.05**2)), rel=1e-9)
    # the paper's version uses the cross-trial variance alone: sqrt(.0004) = .02
    e_paper = 0.02 * ((1 - GAMMA) * ZINV(2 / 3) + GAMMA * ZINV(1 - 1 / (3 * math.e)))
    assert d["dsr_paper_var"] == pytest.approx(PHI((0.05 - e_paper) * math.sqrt(99) / math.sqrt(1 + 0.5 * 0.05**2)), rel=1e-9)
    assert metrics.deflate(0.05, 100, 0.0, 3.0, [0.05], 1)["dsr"] == pytest.approx(PHI(0.05 * math.sqrt(99) / math.sqrt(1.00125)), rel=1e-9)


def test_trials_are_counted_by_distinct_rule_and_model():
    log = [{"period": "dev", "strategy_id": "a", "params": "p1", "cost_bps": 5.0, "sr_daily": 0.1},
           {"period": "dev", "strategy_id": "a", "params": "p1", "cost_bps": 5.0, "sr_daily": 0.2},   # a re-run: no new trial
           {"period": "dev", "strategy_id": "a", "params": "p2", "cost_bps": 5.0, "sr_daily": 0.3},   # another model: a trial
           {"period": "dev", "strategy_id": "b", "params": "p1", "cost_bps": 5.0, "sr_daily": None},
           {"period": "holdout", "strategy_id": "c", "params": "p1", "cost_bps": 5.0, "sr_daily": 0.0}]
    assert strategy.count_trials(log, "dev") == 3
    assert sorted(strategy.trial_srs(log, "dev", 5.0), key=lambda v: (math.isnan(v), v))[:2] == [0.2, 0.3]


# --- segments ------------------------------------------------------------------------------------------


def test_segment_boundaries_from_the_config():
    segs = seg.load_segments()
    sessions = calendar.xnys_sessions(date(2018, 1, 2), date(2018, 9, 28))
    # the return dated 2018-02-27 (closes 02-26 -> 02-27) is -1x; the one dated 02-28 (02-27 -> 02-28) is -0.5x
    assert seg.segment_of_return(segs, "SVXY", date(2018, 2, 26), date(2018, 2, 27)).leverage == -1.0
    assert seg.segment_of_return(segs, "SVXY", date(2018, 2, 27), date(2018, 2, 28)).leverage == -0.5
    usable = seg.usable_return_dates(segs, sessions)
    # amended M6, review A: VXX's usable segment starts at its first close after the last stale print (2018-08-01)
    assert usable[0] == date(2018, 8, 2) and usable == [d for d in sessions if d >= date(2018, 8, 2)]
    assert seg.segment_of_return(segs, "VXX", date(2018, 7, 31), date(2018, 8, 1)) is None  # a stale close: outside


def _synthetic_world():
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, _ = synthetic.synthetic_market(proto)
    sessions = [d for d in calendar.master_calendar(proto)["date"] if d <= proto.data_end]
    return proto, iv, bdata.synthetic_tradables(iv), sessions


def test_segment_checks_catch_a_gap_an_unadjusted_split_and_a_wrong_exposure():
    proto, iv, prices, sessions = _synthetic_world()
    segs, until = seg.load_segments(), date(2024, 6, 28)
    splits = [{"ticker": "SVXY", "date": "2018-09-18", "ratio": 0.25}]
    assert seg.check_segments(prices, segs, sessions, until, splits, 0.15)[0] == []
    broken = prices.copy()
    early = (broken.ticker == "SVXY") & (broken.date < date(2018, 9, 18))
    broken.loc[early, "adj_close"] *= 0.25  # an unadjusted 1:4 reverse split: x4 on the split day
    assert any("not adjusted" in v for v in seg.check_segments(broken, segs, sessions, until, splits, 0.15)[0])
    gap = prices[~((prices.ticker == "VXX") & (prices.date == date(2019, 5, 1)))]
    assert any("missing closes" in v for v in seg.check_segments(gap, segs, sessions, until, [], 0.15)[0])
    flat = prices.copy()
    row = (flat.ticker == "VXX") & (flat.date == date(2019, 5, 1))
    flat.loc[row, ["open", "high", "low"]] = flat.loc[row, "close"].iloc[0]  # O = H = L = C: a stale print in dev
    assert any("stale prints" in v for v in seg.check_segments(flat, segs, sessions, until, [], 0.15, until)[0])
    zero = prices.copy()
    zero["volume"] = zero["volume"].astype("float64")
    zero.loc[(zero.ticker == "SVXY") & (zero.date == date(2020, 2, 3)), "volume"] = 0.0  # volume 0: stale
    assert any("stale prints" in v for v in seg.check_segments(zero, segs, sessions, until, [], 0.15, until)[0])
    # after stale_until (the holdout) a stale print is counted, not a violation
    v, rows = seg.check_segments(flat, segs, sessions, until, [], 0.15, date(2019, 4, 30))
    assert v == [] and next(r for r in rows if r["ticker"] == "VXX" and r["usable"])["stale"] == 1
    wrong = [s if s.leverage != -0.5 else seg.Segment(s.ticker, s.valid_from, s.valid_to, -1.0, s.usable, s.note) for s in segs]
    assert any("exposure" in v for v in seg.check_segments(prices, tuple(wrong), sessions, until, [], 0.15)[0])


def test_the_engine_never_leaves_the_usable_segment():
    proto, iv, prices, sessions = _synthetic_world()
    cfg, segs = strategy.load_config(), seg.load_segments()
    sig = {"x": pd.Series(-1.0, index=sessions)}
    dev = strategy.run_period(cfg, segs, sessions, prices, sig, "dev", proto.holdout_start, proto.data_end)
    assert dev.date.min() == date(2018, 8, 2) and dev.date.max() < proto.holdout_start
    hold = strategy.run_period(cfg, segs, sessions, prices, sig, "holdout", proto.holdout_start, proto.data_end)
    assert hold.date.min() >= proto.holdout_start and hold.date.max() <= proto.data_end
    cut = tuple(s if s.ticker != "VXX" else seg.Segment("VXX", s.valid_from, date(2020, 1, 2), 1.0, True, "") for s in segs)
    cut += (seg.Segment("VXX", date(2020, 3, 2), None, 1.0, True, ""),)  # a hole: 2020-01-02 .. 03-02 unusable
    with pytest.raises(ValueError, match="not one unbroken segment"):
        strategy.run_period(cfg, cut, sessions, prices, sig, "dev", proto.holdout_start, proto.data_end)


def test_select_model_is_the_dev_best_with_exclusions():
    m = pd.DataFrame({"model_id": ["vix_implied", "rw", "har_rv", "garch", "rw"], "target_id": ["rv21"] * 4 + ["vix_level"],
                      "horizon": 21, "period": "dev", "scheme": "all", "forecast": "mean", "qlike": [0.1, 0.3, 0.2, 0.2, 9.0]})
    assert strategy.select_model(m, "rv21", 21, "qlike", ["vix_implied"]) == "garch"  # tie -> first alphabetically
    assert strategy.select_model(m, "rv21", 21, "qlike", []) == "vix_implied"


# --- the command: holdout poisoning, the one look end to end, the preflight, the refusals ---------------


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A synthetic data root with a real (small) model run, M5's dev report, the ETPs and a dev backtest:
    the state the look starts from. Tests copy it, so each owns its own root and look record."""
    import shutil

    from ivf.data import cli as data_cli
    from ivf.evaluation import report as m5
    from ivf.models import cli as models_cli

    base = tmp_path_factory.mktemp("store")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VOL_DATA_DIR", str(base))
        for step in ("fetch", "build-interim", "check"):
            assert data_cli.main(["--synthetic", step]) == 0
        assert models_cli.main(["--synthetic", "garch", "--models", "rw", "har_rv", "har_log"]) == 0  # rv21 and vix_level tracks
        assert m5.main(["--synthetic", "report"]) == 0
        assert cli.main(["--synthetic", "prepare"]) == 0
        pre_dev = tmp_path_factory.mktemp("pre_dev")
        shutil.copytree(base, pre_dev, dirs_exist_ok=True)  # before any dev backtest: for the preflight test
        assert cli.main(["--synthetic", "run"]) == 0
    return base, pre_dev


def _use(store_root, tmp_path, monkeypatch):
    import shutil

    shutil.copytree(store_root, tmp_path, dirs_exist_ok=True)
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    layout = config.data_layout(synthetic=True)
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    (folder,) = list(layout.processed.glob("run_id=*"))
    return layout, proto, folder


def test_dev_run_reads_nothing_from_the_holdout(store, tmp_path, monkeypatch, capsys):
    from ivf.data import build_interim

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    hs = proto.holdout_start
    # A store holding forecast rows at holdout origins (defence in depth: the look writes them elsewhere)
    fc = pd.read_parquet(folder / "forecast.parquet")
    extra = fc[fc.origin_date == fc.origin_date.max()].assign(origin_date=date(2024, 7, 2), yhat=1.0)
    pd.concat([fc, extra], ignore_index=True).to_parquet(folder / "forecast.parquet")
    seen = {}
    real_signals = strategy.signals

    def spy(cfg, chosen, forecasts, vix, sessions):  # what the signal step is handed (review A, R4)
        seen["forecast"], seen["vix"] = forecasts.origin_date.max(), max(vix.index)
        return real_signals(cfg, chosen, forecasts, vix, sessions)

    monkeypatch.setattr(strategy, "signals", spy)
    capsys.readouterr()
    assert cli.main(["--synthetic", "run"]) == 0
    out = capsys.readouterr().out
    assert "returns 2018-08-02 .. 2024-06-28" in out and "holdout_looks 0" in out
    assert f"dev: {len(extra)} forecast rows with an origin on or after {hs} ignored" in out
    assert seen["forecast"] < hs and seen["vix"] < hs
    dev = folder / "backtest" / "period=dev"
    clean = {f.name: pd.read_parquet(f) for f in dev.glob("*.parquet")}
    assert clean["backtest_daily.parquet"].date.max() < hs
    log = (layout.root / "backtest" / "trial_log.jsonl").read_text().splitlines()
    assert len(log) == 2 * 5 * 4 and strategy.count_trials([json.loads(x) for x in log], "dev") == 5  # two runs, 5 trials
    # Poison every value dated on or after holdout_start (x100): VIX and SPX, ETP prices, forecasts at those origins.
    real_read, real_trad = build_interim.read_interim, bdata.read_tradable_table

    def poisoned(root):
        a, b, rep = real_read(root)
        for frame in (a, b):
            frame.loc[frame["date"] >= hs, ["close"]] *= 100
        return a, b, rep

    def poisoned_trad(root):
        t = real_trad(root)
        t.loc[t["date"] >= hs, ["adj_close", "close"]] *= 100
        return t

    real_parquet = pd.read_parquet

    def poisoned_parquet(path, *a, **k):
        frame = real_parquet(path, *a, **k)
        if str(path).endswith("forecast.parquet") and "yhat" in frame:
            frame.loc[frame.origin_date >= hs, "yhat"] *= 100
        return frame

    monkeypatch.setattr(build_interim, "read_interim", poisoned)
    monkeypatch.setattr(bdata, "read_tradable_table", poisoned_trad)
    monkeypatch.setattr(pd, "read_parquet", poisoned_parquet)
    assert cli.main(["--synthetic", "run"]) == 0
    for name, frame in clean.items():
        assert real_parquet(dev / name).equals(frame), name


def test_the_one_look_end_to_end_on_synthetic_data(store, tmp_path, monkeypatch, capsys):
    from ivf.models import holdout as holdout_mod

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    hs = proto.holdout_start
    log_path = layout.root / "backtest" / "trial_log.jsonl"
    n_log = len(log_path.read_text().splitlines())
    dev_forecast = pd.read_parquet(folder / "forecast.parquet")
    capsys.readouterr()
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == 0
    out = capsys.readouterr().out
    record = json.loads((layout.root / cli.LOOK_RECORD).read_text())
    assert record["status"] == "complete" and record["holdout_looks"] == 0 and "spent" in record["note"]
    # holdout-fold forecasts for every model in the dev store, holdout origins only; the dev table untouched
    held = pd.read_parquet(folder / holdout_mod.FOLDER / "forecast.parquet")
    assert held.origin_date.min() >= hs and held.cutoff_date.min() >= hs
    assert set(zip(held.model_id, held.target_id)) == set(zip(dev_forecast.model_id, dev_forecast.target_id))
    assert pd.read_parquet(folder / "forecast.parquet").equals(dev_forecast)
    hc = pd.read_parquet(folder / "evaluation" / "period=holdout" / "holdout_check.parquet")  # M5's confirmatory check
    dm_hold = pd.read_parquet(folder / "evaluation" / "period=holdout" / "dm_test.parquet")
    assert len(hc) == 5 and dm_hold.descriptive_only.all()  # rv21: rmse, qlike; vix_level h = 1, 5, 21: rmse
    meta = json.loads((folder / "backtest" / "period=holdout" / "backtest_meta.json").read_text())
    assert meta["first_return"] >= str(hs) and meta["n_trials"] == 5 and "held_up" in meta["holdout_verdict"]
    report_meta = json.loads((folder / "evaluation" / "period=holdout" / "report_meta.json").read_text())
    for m in (meta, report_meta):  # review B: the caveat (empty on synthetic data) and both protocol hashes
        assert m["caveat"] == "" and m["protocol_hash"] == proto.protocol_hash == m["run_protocol_hash"]
    # the chain starts at the FIRST dev fold: every stored dev forecast row is recomputed and equal (review B)
    assert f"dev reproduction: {len(dev_forecast)} recomputed dev-fold forecasts equal the stored ones" in out
    assert "preflight: segment check to 2026-10-26: pass" in out
    assert "synthetic holdout path, DM and MZ descriptive only" in out
    assert "look complete" in out and len(log_path.read_text().splitlines()) == n_log  # no new trials
    capsys.readouterr()
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_USAGE  # never twice
    assert "the one look was already taken" in capsys.readouterr().err


def test_a_crash_mid_look_leaves_a_record_that_says_so(store, tmp_path, monkeypatch, capsys):
    from ivf.evaluation import report as m5

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    monkeypatch.setattr(m5, "run_report", lambda *a, **k: 3)  # the M5 stage fails after the forecasts were written
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_FAILED
    record = json.loads((layout.root / cli.LOOK_RECORD).read_text())
    assert record["status"].startswith("failed at M5 holdout report") and (folder / "period=holdout" / "forecast.parquet").exists()
    assert "the look is spent" in capsys.readouterr().err
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_USAGE  # the look is spent


def _vxx_x4(layout):
    """Review B's case: VXX's closes x4 from 2024-07-24, an unadjusted 1:4 reverse split in the holdout."""
    t = bdata.read_tradable_table(layout.root)
    late = (t.ticker == "VXX") & (t.date >= date(2024, 7, 24))
    t.loc[late, ["open", "high", "low", "close", "adj_close"]] *= 4
    return t


@pytest.mark.parametrize("break_it", ["looks", "tradable_gap", "model", "no_dev_backtest", "holdout_folder", "unadjusted_split"])
def test_a_failed_preflight_writes_no_record(store, tmp_path, monkeypatch, capsys, break_it):
    from ivf.models import holdout as holdout_mod

    layout, proto, folder = _use(store[1] if break_it == "no_dev_backtest" else store[0], tmp_path, monkeypatch)
    if break_it == "looks":
        real = strategy.load_config

        def wrong():
            cfg = real()
            cfg.document["holdout_looks_before_look"] = {"real": 1, "synthetic": 1}
            return cfg
        monkeypatch.setattr(strategy, "load_config", wrong)
    elif break_it == "tradable_gap":
        t = bdata.read_tradable_table(layout.root)
        t = t[~((t.ticker == "SVXY") & (t.date == date(2025, 3, 3)))]
        real_trad = bdata.read_tradable_table
        monkeypatch.setattr(bdata, "read_tradable_table", lambda root: t if root == layout.root else real_trad(root))
    elif break_it == "model":
        monkeypatch.setattr(holdout_mod, "build_track_models", lambda t, ids: (_ for _ in ()).throw(ValueError(f"{t}: no spec")))
    elif break_it == "holdout_folder":
        (folder / holdout_mod.FOLDER).mkdir()
    elif break_it == "unadjusted_split":
        t = _vxx_x4(layout)
        monkeypatch.setattr(bdata, "read_tradable_table", lambda root: t)
    capsys.readouterr()
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    if break_it == "tradable_gap":
        assert "preflight: segment check: SVXY segment from 2018-02-27: 1 missing closes, first 2025-03-03" in err
    if break_it == "unadjusted_split":  # named, never measured (no value printed)
        assert "preflight: segment check: VXX split 2024-07-24 (ratio 0.25): not adjusted" in err
        assert "outside the tolerance of the stated leverage (exposure)" in err and "log terms" not in err
    assert "preflight failed; no look record was written or changed, the look is NOT spent" in err
    assert not (layout.root / cli.LOOK_RECORD).exists()
    assert not (folder / holdout_mod.FOLDER / "forecast.parquet").exists()


def test_the_real_look_needs_the_flag_a_clean_pushed_tree_and_happens_once(tmp_path, monkeypatch, capsys):
    from ivf import contract
    from ivf.data import build_interim

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))

    def boom(root):
        raise build_interim.InterimError("stop: no data in this test")

    monkeypatch.setattr(build_interim, "read_interim", boom)
    root = tmp_path / "implied-vol-forecasting"
    assert cli.main(["run", "--period", "holdout"]) == cli.EXIT_USAGE
    assert "refusing --period holdout on real data" in capsys.readouterr().err and not root.exists()
    assert cli.main(["run", "--one-look"]) == cli.EXIT_USAGE  # --one-look is for the holdout only
    monkeypatch.setattr(contract, "git_state", lambda *a: contract.GitState("abc", True))
    assert cli.main(["run", "--period", "holdout", "--one-look"]) == cli.EXIT_USAGE  # doomed: preflight, no record
    err = capsys.readouterr().err
    assert "the working tree is not clean" in err and "no data in this test" in err and not (root / cli.LOOK_RECORD).exists()
    monkeypatch.setattr(contract, "git_state", lambda *a: contract.GitState("abc", False))
    monkeypatch.setattr(cli, "is_pushed", lambda commit: False)
    assert cli.main(["run", "--period", "holdout", "--one-look"]) == cli.EXIT_USAGE
    assert "is not on any remote branch" in capsys.readouterr().err and not (root / cli.LOOK_RECORD).exists()
    (root / "backtest").mkdir(parents=True)
    (root / cli.LOOK_RECORD).write_text("{}")
    assert cli.main(["run", "--period", "holdout", "--one-look"]) == cli.EXIT_USAGE
    assert "the one look was already taken" in capsys.readouterr().err


def test_the_lstm_continues_from_its_last_dev_retrain():
    from ivf.models import holdout as holdout_mod

    fit = pd.DataFrame({"model_id": ["lstm_log"] * 3 + ["rw"], "target_id": ["rv21"] * 4,
                        "cutoff_date": [date(2024, 1, 2), date(2024, 4, 1), date(2024, 4, 1), date(2024, 4, 1)],
                        "params_json": ['{"trained_cutoff": 100}', '{"trained_cutoff": 163}', '{"trained_cutoff": 163}', "{}"]})
    assert holdout_mod.lstm_start_cutoff(fit, "rv21") == 163  # the last dev fold's network was trained at index 163
    with pytest.raises(ValueError, match="no dev fit rows"):
        holdout_mod.lstm_start_cutoff(fit, "vix_level")


def test_the_record_is_create_only_even_after_a_passed_preflight(store, tmp_path, monkeypatch, capsys):
    # A second look starting between the preflight and the record (a race): the create-only write stops it.
    from ivf.models import holdout as holdout_mod

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    real_preflight = cli.preflight

    def racing(layout_, **kw):
        problems, ctx = real_preflight(layout_, **kw)
        (layout_.root / cli.LOOK_RECORD).write_text('{"status": "another look"}')
        return problems, ctx

    monkeypatch.setattr(cli, "preflight", racing)
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_USAGE
    assert "the one look was already taken" in capsys.readouterr().err
    assert json.loads((layout.root / cli.LOOK_RECORD).read_text()) == {"status": "another look"}
    assert not (folder / holdout_mod.FOLDER).exists()



def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*") if p.is_file())


def test_preflight_only_writes_nothing_and_is_not_a_look(store, tmp_path, monkeypatch, capsys):
    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    before = _tree(tmp_path)
    capsys.readouterr()
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--preflight-only"]) == 0
    out = capsys.readouterr().out
    assert "preflight: segment check to 2026-10-26: pass" in out and "preflight ok (nothing written; this is not a look)" in out
    t = _vxx_x4(layout)
    monkeypatch.setattr(bdata, "read_tradable_table", lambda root: t)
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--preflight-only"]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert "preflight FAILED" in out and "not adjusted" in out and "log terms" not in out
    assert _tree(tmp_path) == before  # nothing written, nothing touched


def test_a_failed_look_resumes_at_the_failed_stage(store, tmp_path, monkeypatch, capsys):
    from ivf import contract
    from ivf.evaluation import report as m5
    from ivf.models import holdout as holdout_mod

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    real_report = m5.run_report
    monkeypatch.setattr(m5, "run_report", lambda *a, **k: 3)  # a fault injected at the M5 stage
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_FAILED
    held = folder / holdout_mod.FOLDER / "forecast.parquet"
    stamp = held.stat().st_mtime_ns
    monkeypatch.setattr(m5, "run_report", real_report)
    real_git = contract.git_state
    monkeypatch.setattr(contract, "git_state", lambda *a: contract.GitState("other-commit", False))
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--resume"]) == cli.EXIT_USAGE  # another commit: refused
    assert "the commit differs" in capsys.readouterr().err
    monkeypatch.setattr(contract, "git_state", real_git)
    real_trad = bdata.read_tradable_table
    monkeypatch.setattr(bdata, "read_tradable_table", lambda root: real_trad(root).assign(raw_sha256="0" * 64))
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--resume"]) == cli.EXIT_USAGE  # other inputs: refused
    assert "the input snapshots differ" in capsys.readouterr().err
    monkeypatch.setattr(bdata, "read_tradable_table", real_trad)
    monkeypatch.setattr(holdout_mod, "run_holdout_forecasts", lambda *a, **k: pytest.fail("the completed stage re-ran"))
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--resume"]) == 0
    record = json.loads((layout.root / cli.LOOK_RECORD).read_text())
    assert record["status"] == "complete" and held.stat().st_mtime_ns == stamp
    assert [a["from_stage"] for a in record["attempts"]] == ["holdout forecasts", "M5 holdout report"]
    assert record["attempts"][0]["last_status"].startswith("failed at M5 holdout report")
    assert (folder / "backtest" / "period=holdout" / "backtest_meta.json").exists()
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--resume"]) == cli.EXIT_USAGE  # complete: nothing to resume


def test_a_holdout_fit_that_falls_back_fails_the_stage_loudly(store, tmp_path, monkeypatch, capsys):
    from ivf.models import holdout as holdout_mod
    from ivf.models import rv21

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    hs_idx = list(calendar.master_calendar(proto)["date"]).index(proto.holdout_start)
    real_fit = rv21.HarRv.fit

    def dies_in_the_holdout(self, inputs, previous):  # e.g. a worker that died: the runner keeps the previous fold
        if len(inputs) > hs_idx:
            raise RuntimeError("worker died")
        return real_fit(self, inputs, previous)

    monkeypatch.setattr(rv21.HarRv, "fit", dies_in_the_holdout)
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_FAILED
    record = json.loads((layout.root / cli.LOOK_RECORD).read_text())
    assert record["status"].startswith("failed at holdout forecasts: HoldoutStageError: a holdout fit fell back")
    assert "har_rv/rv21 previous_fold" in record["status"] and not (folder / holdout_mod.FOLDER / "forecast.parquet").exists()
    monkeypatch.setattr(rv21.HarRv, "fit", real_fit)
    assert cli.main(["--synthetic", "run", "--period", "holdout", "--resume"]) == 0  # the stage re-runs and completes


def test_the_holdout_chain_must_reproduce_the_stored_dev_forecasts():
    from ivf.models import holdout as holdout_mod

    stored = pd.DataFrame({"model_id": "m", "target_id": "rv21", "horizon": 21, "origin_date": _sessions(3), "fold_id": 1,
                           "yhat": [20.0, 21.0, 22.0]})
    assert holdout_mod.check_reproduces_dev(stored.copy(), stored) == 3
    off = stored.copy()
    off.loc[1, "yhat"] += 1e-9
    with pytest.raises(holdout_mod.HoldoutStageError, match="do not reproduce"):
        holdout_mod.check_reproduces_dev(off, stored)
    with pytest.raises(holdout_mod.HoldoutStageError, match="1 rows unmatched"):
        holdout_mod.check_reproduces_dev(stored.iloc[:2], stored)


def test_the_lstm_warmup_starts_at_the_fold_of_its_last_retrain():
    from ivf.evaluation.walkforward import Fold
    from ivf.models import holdout as holdout_mod

    folds = [Fold(k, c, c + 62, "dev") for k, c in enumerate((100, 163, 226, 289))]
    assert [f.cutoff_idx for f in holdout_mod.lstm_warmup(folds, 163)] == [163, 226, 289]
    with pytest.raises(holdout_mod.HoldoutStageError, match="no dev fold"):
        holdout_mod.lstm_warmup(folds, 170)


def test_a_dev_run_may_differ_from_the_protocol_only_in_holdout_looks(store, tmp_path, monkeypatch, capsys):
    from ivf import contract

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    run_id = folder.name.split("=", 1)[1]
    runs_path = layout.processed / "runs.parquet"
    runs = pd.read_parquet(runs_path)
    assert cli.run_protocol(layout, run_id, proto) == (proto.protocol_hash, None)
    earlier = contract.canonical_json_sha256({**proto.document, "holdout_looks": 0})
    later = protocol.Protocol(**{**proto.__dict__, "document": {**proto.document, "holdout_looks": 1}})
    assert cli.run_protocol(layout, run_id, later) == (earlier, None)  # only holdout_looks moved: accepted
    runs.loc[runs.run_id == run_id, "protocol_hash"] = contract.canonical_json_sha256({**proto.document, "fold_len": 21})
    runs.to_parquet(runs_path)
    assert "in more than holdout_looks" in cli.run_protocol(layout, run_id, proto)[1]
    capsys.readouterr()
    assert cli.main(["--synthetic", "run"]) == cli.EXIT_INPUT
    assert "in more than holdout_looks" in capsys.readouterr().err


def test_a_look_whose_chain_does_not_reproduce_dev_stops_before_any_holdout_forecast(store, tmp_path, monkeypatch, capsys):
    from ivf.models import holdout as holdout_mod

    layout, proto, folder = _use(store[0], tmp_path, monkeypatch)
    fc = pd.read_parquet(folder / "forecast.parquet")
    fc.loc[fc.index[len(fc) // 2], "yhat"] += 1e-9  # the stored dev forecasts no longer match the code
    fc.to_parquet(folder / "forecast.parquet")
    assert cli.main(["--synthetic", "run", "--period", "holdout"]) == cli.EXIT_FAILED
    record = json.loads((layout.root / cli.LOOK_RECORD).read_text())
    assert record["status"].startswith("failed at holdout forecasts: HoldoutStageError: the recomputed dev folds do not reproduce")
    assert not (folder / holdout_mod.FOLDER / "forecast.parquet").exists()
