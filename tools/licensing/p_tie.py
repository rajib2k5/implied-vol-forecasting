"""Licensing probe for ADR 0001 D2 (amended M0, review E). SIMULATED DATA ONLY: every series is drawn
from a seeded generator in p_set.sim(); no file is read and no real market value is embedded.
Run from the repo root: uv run python tools/licensing/p_tie.py

Written by review E (docs/validation/M0-review-E-final.md, E1.c2-c4); committed unchanged except paths.

E1.c2 / c3 / c4: the models in the registry that are NOT plain V-equivariant.

Simulated data only (p_set.sim, seed 20261028). Nothing here is real data.

c2: the rv21 track. Registry: rw (y-hat = rv21_back(t)), vix_implied (v-hat = (V_t/100)^2), garch-like (EWMA of SPX
    returns). Attacker holds the SPX return series (public: FRED SP500 / Yahoo) and the committed P rows for rv21.
c3: *_cal: v-hat = a_h + b_h * G_t by OLS on training pairs. Rescale V (SPX fixed) and SPX (V fixed).
c4: a tiny neural forecaster with train-only z-scoring (float32, full-batch GD): V-equivariant up to float rounding?
"""
import math
import numpy as np
import p_set as S

SIG, DP = S.SIG, S.DP


def ewma_var(r, lam=0.94):
    ew = np.zeros(len(r))
    for t in range(1, len(r)):
        ew[t] = lam * ew[t - 1] + (1 - lam) * r[t] ** 2
    return 252 * ew                                            # annualised decimal variance


def rv21(r):
    c = np.concatenate([[0.0], np.cumsum(r ** 2)])
    out = np.full(len(r), np.nan)
    for d in range(21, len(r)):
        out[d] = 100 * math.sqrt(252 / 21 * (c[d + 1] - c[d - 20]))
    return out


def losses(y, f):
    v, vh = (y / 100) ** 2, (f / 100) ** 2
    return {"se": (y - f) ** 2, "ae": np.abs(y - f), "ape": np.abs(y - f) / y, "ql": v / vh - np.log(v / vh) - 1}


def ratio_rows(y, F, h, tag):
    L = {m: losses(y, f) for m, f in F.items()}
    P = {}
    for m in F:
        if m == "rw":
            continue
        P[(tag, "rel_rmse", m)] = SIG(math.sqrt(L[m]["se"].mean() / L["rw"]["se"].mean()), 3)
        P[(tag, "rel_mae", m)] = SIG(L[m]["ae"].mean() / L["rw"]["ae"].mean(), 3)
        P[(tag, "rel_mape", m)] = SIG(L[m]["ape"].mean() / L["rw"]["ape"].mean(), 3)
        P[(tag, "rel_qlike", m)] = SIG(L[m]["ql"].mean() / L["rw"]["ql"].mean(), 3)
        for loss, k in (("sq", "se"), ("ql", "ql")):
            s, p = S.dm(L[m][k], L["rw"][k], h)
            P[(tag, "dm_stat", m, loss)] = DP(s, 2); P[(tag, "dm_p", m, loss)] = DP(p, 2)
        P[(tag, "mz", m)] = S.mz_pass(y, F[m], h)
    return P


def rv21_track(V, r, n_train, n_o, with_vix_implied=True):
    h = 21
    o = np.arange(n_train, n_train + n_o)
    RV = rv21(r)
    y = RV[o + h]
    F = {"rw": RV[o], "garch_like": 100 * np.sqrt(ewma_var(r)[o])}
    if with_vix_implied:
        F["vix_implied"] = V[o]
    return ratio_rows(y, F, h, "rv21"), y, F


def cal_track(V, r, n_train, n_o, h, fixed_a=None):
    """garch_cal: v = a + b*G by OLS on training pairs (tau + h <= n_train - 1); forecasts at origins."""
    G = ewma_var(r)
    v = (V / 100) ** 2
    idx = np.arange(25, n_train - h)
    if fixed_a is None:
        X = np.column_stack([np.ones(len(idx)), G[idx]])
        a, b = np.linalg.lstsq(X, v[idx + h], rcond=None)[0]
    else:
        a = fixed_a
        b = np.linalg.lstsq(G[idx][:, None], v[idx + h] - a, rcond=None)[0][0]
    o = np.arange(n_train, n_train + n_o)
    vh = np.maximum(a + b * G[o], 1e-8)
    F = {"rw": V[o], "garch_cal": 100 * np.sqrt(vh)}
    return ratio_rows(V[o + h], F, h, f"cal_h{h}")


def nn_track(V, r, n_train, n_o, h, extra="none", dtype=np.float32, steps=400):
    """Tiny 1-hidden-layer tanh net on z-scored inputs (train-only scaler), z-scored ln V target, full-batch GD."""
    lnV = np.log(V)
    w5 = np.convolve(lnV, np.ones(5) / 5, "full")[: len(V)]
    w22 = np.convolve(lnV, np.ones(22) / 22, "full")[: len(V)]
    cols = [lnV, w5, w22]
    RV = rv21(r)
    if extra == "rv21_back":
        cols.append(RV)                                         # SPX-only feature: unchanged when V is rescaled
    if extra == "iv_rv_spread":
        cols.append((V / 100) ** 2 - (RV / 100) ** 2)           # mixes VIX and SPX scales
    Xall = np.column_stack(cols)
    tr = np.arange(25, n_train - h)
    mu, sd = Xall[tr].mean(0), Xall[tr].std(0)
    ty = lnV[tr + h]; tmu, tsd = ty.mean(), ty.std()
    Z = ((Xall - mu) / sd).astype(dtype); zy = ((ty - tmu) / tsd).astype(dtype)
    g = np.random.default_rng(7)
    W1 = (g.standard_normal((Z.shape[1], 8)) * 0.3).astype(dtype); b1 = np.zeros(8, dtype)
    W2 = (g.standard_normal(8) * 0.3).astype(dtype); b2 = dtype(0)
    lr = dtype(0.05)
    Xt = Z[tr]
    for _ in range(steps):
        Hh = np.tanh(Xt @ W1 + b1); pred = Hh @ W2 + b2; err = pred - zy
        gW2 = Hh.T @ err / len(err); gb2 = err.mean()
        dH = np.outer(err, W2) * (1 - Hh ** 2)
        gW1 = Xt.T @ dH / len(err); gb1 = dH.mean(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    resid = (np.tanh(Xt @ W1 + b1) @ W2 + b2 - zy) * tsd
    s2 = float(np.var(resid))
    o = np.arange(n_train, n_train + n_o)
    m = (np.tanh(Z[o] @ W1 + b1) @ W2 + b2).astype(np.float64) * tsd + tmu
    F = {"rw": V[o], "nn": np.exp(m + s2 / 2)}
    return ratio_rows(V[o + h], F, h, f"nn_{extra}_h{h}")


def diff(P, Q):
    return [k for k in P if P[k] != Q.get(k)]


if __name__ == "__main__":
    V, r, vxx, n_train, n_o = S.sim(20261028)
    o = np.arange(n_train, n_train + n_o)

    print("=== c2: rv21 track. Which entries move under V -> cV (SPX fixed)?")
    P0, y, F = rv21_track(V, r, n_train, n_o)
    for c in (0.8, 1.25):
        Pc, _, _ = rv21_track(V * c, r, n_train, n_o)
        d = diff(P0, Pc)
        print(f"  c={c}: {len(d)} of {len(P0)} rv21 entries differ; models: {sorted({k[2] for k in d})}")
    Pn, _, _ = rv21_track(V, r, n_train, n_o, with_vix_implied=False)
    Pnc, _, _ = rv21_track(V * 1.25, r, n_train, n_o, with_vix_implied=False)
    print(f"  FALSIFIER (vix_implied removed) c=1.25: {len(diff(Pn, Pnc))} of {len(Pn)} differ")

    print("=== c2 attack: attacker holds SPX returns r (public) + committed rv21 rows of P")
    Lrw = losses(y, F["rw"])                                    # computable from SPX alone
    MAE = P0[("rv21", "rel_mae", "vix_implied")] * Lrw["ae"].mean()
    RMSE = P0[("rv21", "rel_rmse", "vix_implied")] * math.sqrt(Lrw["se"].mean())
    QL = P0[("rv21", "rel_qlike", "vix_implied")] * Lrw["ql"].mean()
    true_mean, true_rmse = V[o].mean(), math.sqrt(((y - V[o]) ** 2).mean())
    print(f"  recovered absolute RMSE(vix_implied) = {RMSE:.4f} vol pts (true {true_rmse:.4f}); MAE = {MAE:.4f} (true {np.abs(y - V[o]).mean():.4f})")
    est1 = y.mean() + MAE                                        # exact if V >= y on every origin
    share = (V[o] >= y).mean()
    print(f"  estimator 1, mean(y) + MAE: holdout mean V = {est1:.3f}  (true {true_mean:.3f}, error {100*(est1/true_mean-1):+.2f}%; V >= y on {100*share:.1f}% of origins)")
    # estimator 2: the attacker's own up-to-scale shape from SPX (EWMA vol), scale c solved from the published MAE
    shape = F["garch_like"]
    cs = np.linspace(0.5, 2.5, 20001)
    mae_c = np.array([np.abs(y - c * shape).mean() for c in cs])
    ql_c = np.array([losses(y, c * shape)["ql"].mean() for c in cs])
    c_mae, c_ql = cs[np.argmin(np.abs(mae_c - MAE))], cs[np.argmin(np.abs(ql_c - QL))]
    for name, c in (("MAE", c_mae), ("QLIKE", c_ql)):
        Vhat = c * shape
        print(f"  estimator 2 ({name} solve), V ~ c*EWMA vol: c={c:.4f}  mean {Vhat.mean():.3f} (error {100*(Vhat.mean()/true_mean-1):+.2f}%), "
              f"daily RMS error {math.sqrt(((Vhat - V[o])**2).mean()):.3f} vol pts, median abs daily error {np.median(np.abs(Vhat - V[o])):.3f}")
    print(f"  (for scale: sd of V over the holdout {V[o].std():.3f}; sd of daily change {np.diff(V[o]).std():.3f})")

    print("=== c3: garch_cal (OLS with intercept) under V -> cV (SPX fixed) and r -> s r (V fixed)")
    for h in (1, 5, 21):
        base = cal_track(V, r, n_train, n_o, h)
        dv = sum(len(diff(base, cal_track(V * c, r, n_train, n_o, h))) for c in (0.8, 1.25, 2.0))
        ds = sum(len(diff(base, cal_track(V, r * s, n_train, n_o, h))) for s in (0.5, 1.5))
        fb = cal_track(V, r, n_train, n_o, h, fixed_a=0.01)
        dfx = len(diff(fb, cal_track(V * 1.25, r, n_train, n_o, h, fixed_a=0.01)))
        print(f"  h={h}: {len(base)} entries; changed over 3 V-scales: {dv}; over 2 SPX-scales: {ds}; FALSIFIER fixed a=0.01, c=1.25: {dfx} changed")

    print("=== c4: tiny neural net, train-only z-scaling, float32 (V -> cV, SPX fixed)")
    for extra in ("none", "rv21_back", "iv_rv_spread"):
        tot = cnt = 0
        for h in (1, 5, 21):
            base = nn_track(V, r, n_train, n_o, h, extra)
            for c in (0.8, 1.25, 2.0):
                d = diff(base, nn_track(V * c, r, n_train, n_o, h, extra)); cnt += len(d); tot += len(base)
        print(f"  inputs {extra:13s}: {cnt} of {tot} entry-comparisons changed (3 horizons x 3 scales)")
    tot = cnt = 0
    for h in (1, 5, 21):
        base = nn_track(V, r, n_train, n_o, h, "none", dtype=np.float64)
        for c in (0.8, 1.25, 2.0):
            d = diff(base, nn_track(V * c, r, n_train, n_o, h, "none", dtype=np.float64)); cnt += len(d); tot += len(base)
    print(f"  inputs none, float64: {cnt} of {tot} changed")
