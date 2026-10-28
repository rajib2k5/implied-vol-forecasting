"""Licensing probe for ADR 0001 D2 (amended M0, review E). SIMULATED DATA ONLY: every series is drawn
from a seeded generator in p_set.sim(); no file is read and no real market value is embedded.
Run from the repo root: uv run python tools/licensing/p_tie_seeds.py

Written by review E; committed unchanged except paths.

ADDED: E1.c2 estimator errors over 10 seeds (simulated only). Same estimators as p_tie.py."""
import math, numpy as np
import p_tie as T
from p_tie import S
rows = []
for seed in range(1, 11):
    V, r, vxx, n_train, n_o = S.sim(seed)
    o = np.arange(n_train, n_train + n_o)
    P0, y, F = T.rv21_track(V, r, n_train, n_o)
    Lrw = T.losses(y, F["rw"])
    MAE = P0[("rv21", "rel_mae", "vix_implied")] * Lrw["ae"].mean()
    QL = P0[("rv21", "rel_qlike", "vix_implied")] * Lrw["ql"].mean()
    RMSE = P0[("rv21", "rel_rmse", "vix_implied")] * math.sqrt(Lrw["se"].mean())
    tm = V[o].mean(); trm = math.sqrt(((y - V[o]) ** 2).mean())
    e1 = 100 * ((y.mean() + MAE) / tm - 1)
    cs = np.linspace(0.5, 2.5, 8001)
    ql_c = np.array([T.losses(y, c * F["garch_like"])["ql"].mean() for c in cs])
    c = cs[np.argmin(np.abs(ql_c - QL))]
    e2 = 100 * ((c * F["garch_like"]).mean() / tm - 1)
    rows.append((seed, 100 * (RMSE / trm - 1), e1, e2, 100 * (V[o] >= y).mean()))
    print(f"seed {seed:2d}: RMSE(V - rv21) recovered err {rows[-1][1]:+.3f}% | mean V est1 {e1:+6.2f}% | est2 (QLIKE) {e2:+6.2f}% | V>=y on {rows[-1][4]:.1f}%")
a = np.array(rows)
print(f"median |err|: RMSE {np.median(np.abs(a[:,1])):.3f}%  est1 {np.median(np.abs(a[:,2])):.2f}%  est2 {np.median(np.abs(a[:,3])):.2f}%; max |err| est2 {np.abs(a[:,3]).max():.2f}%")
