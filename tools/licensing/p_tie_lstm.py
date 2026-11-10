"""Licensing probe for ADR 0001 O10: is the REAL M4 ``lstm_log`` (src/ivf/models/lstm.py) V-scale-equivariant?
SIMULATED DATA ONLY: every series is drawn from the seeded generator p_set.sim(); no file is read and no
real market value is embedded. Run from the repo root: uv run python tools/licensing/p_tie_lstm.py

Question: replace the VIX level V by c*V (c = 0.8, 1.25; S&P 500 returns unchanged) and refit/forecast with
``lstm.LstmLog`` (VIX track, h = 1, 5, 21) and ``lstm.LstmRv`` (rv21 track, h = 21, ln VIX is a static input).
Do the evaluation entries (rmse ratio vs rw, qlike ratio vs rw, MAPE ratio and absolute MAPE) stay unchanged
up to float rounding? Also compared: raw forecasts / c (VIX track) and raw forecasts (rv21 track).

Cheap config (stated, not the production one): FOLDS training cutoffs 21 sessions apart, each a full refit,
MEMBERS seeds, small hidden size and max_epochs, threads = lstm.THREADS = 1 (set by lstm.torch_state).

Falsifiers (the probe must see a change when the model is NOT equivariant):
  raw_level : an extra static input equal to V / 100, bypassing the train-only z-scoring (unstandardised level)
  flag20    : an extra static input 1[V > 20] (a fixed absolute threshold; still z-scored)

float64: --float64 runs the same code with torch's default dtype and the module's float32 casts replaced by
float64 (a monkeypatch of lstm.np; the repo's production path is float32 only).
"""
import argparse
import time
import types
from datetime import date, timedelta

import numpy as np
import torch

import p_set as S
from ivf.models import lstm, rv21

DAYS = [date(2000, 1, 3) + timedelta(days=i) for i in range(4000)]
ROOT = 20261028
HS = (1, 5, 21)


def settings(a):
    return lstm.LstmSettings(members=a.members, window=22, hidden=a.hidden, batch=64, lr=1e-3, max_epochs=a.epochs,
                             patience=5, val_sessions=100, refit_every=1)


# --- falsifier mutants: patch the feature builders; undone by the caller -------------------------------------


class Mutant:
    """Context manager adding one static column to both tracks' features (and, for raw_level, skipping the scaler)."""

    def __init__(self, kind):
        self.kind = kind

    def column(self, v):
        v = np.asarray(v, dtype=float)
        return v / 100 if self.kind == "raw_level" else (v > 20).astype(float)

    def __enter__(self):
        if self.kind == "none":
            return self
        self.saved = (lstm.vix_features, lstm.rv_features, lstm.Scaler.fit, lstm.Scaler.inputs)
        vf, rf, fit = self.saved[0], self.saved[1], self.saved[2].__func__
        col = self.column

        def vix_features(y, window):
            seq, static, anchor = vf(y, window)
            return seq, np.column_stack([static, col(y)]), anchor

        def rv_features(inputs, window):
            seq, static, anchor = rf(inputs, window)
            return seq, np.column_stack([static, col(inputs.vix)]), anchor

        raw = self.kind == "raw_level"

        def scaler_fit(cls, seq, static, target):
            sc = fit(cls, seq, static, target)
            if raw:                                    # last static column passes through unstandardised
                sc.static_mean[-1], sc.static_sd[-1] = 0.0, 1.0
            return sc

        lstm.vix_features, lstm.rv_features = vix_features, rv_features
        lstm.Scaler.fit = classmethod(scaler_fit)
        return self

    def __exit__(self, *exc):
        if self.kind != "none":
            lstm.vix_features, lstm.rv_features, lstm.Scaler.fit, lstm.Scaler.inputs = self.saved


class F64:
    """Context manager: float64 everywhere in lstm.py (inputs, targets, weights)."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        if self.on:
            self.saved = (lstm.np, torch.get_default_dtype())
            proxy = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
            proxy.float32 = np.float64
            lstm.np = proxy
            torch.set_default_dtype(torch.float64)
        return self

    def __exit__(self, *exc):
        if self.on:
            lstm.np, dt = self.saved
            torch.set_default_dtype(dt)


# --- the two tracks through the real model classes ----------------------------------------------------------


def cutoffs(n_train, folds):
    return [n_train + 21 * k for k in range(folds)]


def vix_track(V, n_train, a):
    """Raw forecasts per h at the origins of every fold, via LstmLog.fit / .forecast (as walkforward.run_symbol)."""
    model = lstm.LstmLog(settings(a), ROOT, DAYS)
    out = {h: [] for h in HS}
    origins = []
    prev = None
    for c in cutoffs(n_train, a.folds):
        fit = model.fit(V[: c + 1], HS, prev)
        o = np.arange(c, c + 21)
        fc = model.forecast(fit, V[: o[-1] + 1], o, HS)
        for h in HS:
            out[h].append(fc[h].yhat)
        origins.append(o)
        prev = fit
    return np.concatenate(origins), {h: np.concatenate(v) for h, v in out.items()}


def rv_track(V, r, n_train, a):
    model = lstm.LstmRv(settings(a), ROOT, DAYS)
    inputs = rv21.Rv21Inputs(r, V)
    out, origins, prev = [], [], None
    for c in cutoffs(n_train, a.folds):
        fit = model.fit(inputs.upto(c), prev)
        o = np.arange(c, c + 21)
        out.append(model.forecast(fit, inputs.upto(o[-1]), o).yhat)
        origins.append(o)
        prev = fit
    return np.concatenate(origins), np.concatenate(out)


def entries(y, f, rw):
    """Unrounded evaluation entries of forecast f against the random walk rw: p_tie.losses' definitions."""
    def loss(g):
        v, vh = (y / 100) ** 2, (g / 100) ** 2
        return np.sqrt(np.mean((y - g) ** 2)), np.mean(v / vh - np.log(v / vh) - 1), np.mean(np.abs(y - g) / y)
    (a1, a2, a3), (b1, b2, b3) = loss(f), loss(rw)
    return {"rmse_ratio_rw": a1 / b1, "qlike_ratio_rw": a2 / b2, "mape_ratio_rw": a3 / b3, "mape": a3}


def run_all(V, r, n_train, a):
    """{key: (entries dict, raw forecast array)} for vix h=1,5,21 and rv21 h=21."""
    o, F = vix_track(V, n_train, a)
    res = {}
    for h in HS:
        res[("vix", h)] = (entries(V[o + h], F[h], V[o]), F[h])
    RV = rv21.target_series(r)
    o2, f2 = rv_track(V, r, n_train, a)
    res[("rv21", 21)] = (entries(RV[o2 + 21], f2, RV[o2]), f2)
    return res


def sig(x, k):
    return float(f"{x:.{k}g}")


def compare(base, other, c):
    """Print one row per target x horizon: max abs entry diff, #entries differing at 6 and at 3 s.f., raw forecast diff."""
    names = ["rmse_ratio_rw", "qlike_ratio_rw", "mape_ratio_rw", "mape"]
    n6 = n3 = tot = 0
    for key in base:
        e0, f0 = base[key]
        e1, f1 = other[key]
        d = {n: abs(e0[n] - e1[n]) for n in names}
        k6 = sum(sig(e0[n], 6) != sig(e1[n], 6) for n in names)
        k3 = sum(sig(e0[n], 3) != sig(e1[n], 3) for n in names)
        g = f1 / c if key[0] == "vix" else f1                     # VIX track scales with c; rv21 track must not move
        fd, fr = np.abs(g - f0).max(), (np.abs(g - f0) / np.abs(f0)).max()
        n6, n3, tot = n6 + k6, n3 + k3, tot + len(names)
        print(f"    {key[0]:4s} h={key[1]:2d} max|d entry|={max(d.values()):.3e}  differ@6sf={k6}/4  @3sf={k3}/4  "
              f"raw fcst{' /c' if key[0] == 'vix' else '   '} max abs diff={fd:.3e} max rel={fr:.3e}")
    return n6, n3, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--members", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    V, r, _, n_train, _ = S.sim(a.seed)
    print(f"config: sim seed {a.seed}, n_train {n_train}, folds {a.folds} (cutoffs {cutoffs(n_train, a.folds)}, 21 origins each), "
          f"members {a.members}, hidden {a.hidden}, max_epochs {a.epochs}, val_sessions 100, threads {lstm.THREADS}, "
          f"dtype {'float64 (patched)' if a.float64 else 'float32 (production)'}")
    t0 = time.perf_counter()
    summary = {}
    for kind in ("none", "raw_level", "flag20"):
        label = "REAL MODEL" if kind == "none" else f"FALSIFIER {kind}"
        print(f"=== {label}")
        with Mutant(kind), F64(a.float64):
            base = run_all(V, r, n_train, a)
            for key, (e, _) in base.items():
                print(f"  baseline {key[0]:4s} h={key[1]:2d} " + " ".join(f"{k}={v:.6f}" for k, v in e.items()))
            for c in (0.8, 1.25):
                print(f"  c={c}")
                summary[(kind, c)] = compare(base, run_all(V * c, r, n_train, a), c)
        print(f"  elapsed {time.perf_counter() - t0:.0f}s")
    print("=== summary: entries differing / entries compared (VIX h=1,5,21 + rv21 h=21, 4 entries each)")
    for (kind, c), (n6, n3, tot) in summary.items():
        print(f"  {kind:10s} c={c:<5}: {n6}/{tot} differ at 6 s.f., {n3}/{tot} differ at 3 s.f.")
    print(f"total runtime {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
