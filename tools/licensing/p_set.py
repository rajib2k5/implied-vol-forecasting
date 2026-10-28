"""Licensing probe for ADR 0001 D2 (amended M0, review E). SIMULATED DATA ONLY: every series is drawn
from a seeded generator in p_set.sim(); no file is read and no real market value is embedded.
Run from the repo root: uv run python tools/licensing/p_set.py

Computes exactly the publication set P (ADR D2) on a simulated VIX-like path, then tests what P
reveals. Published rows are the scale-equivariant models only; the SPX-driven model `ewma_spx`
stands in for a scale-tied row (uncalibrated GARCH on vix_level) and is added only to show why such
rows are excluded.

P per period and horizon: RMSE/QLIKE/MAE/MAPE ratios to the random walk (3 s.f.), DM HLN statistic and
p-value vs RW for sq_error and qlike (2 dp), MZ beta=1 and joint (alpha=0, beta=1) pass/fail at 5%
(booleans), counts n; per strategy Sharpe, PSR, DSR (2 dp), n_trials, turnover (2 s.f.); AR(1) half-life
of ln V (2 s.f.).
"""
import math
import numpy as np
from scipy import stats

SIG = lambda x, k: float(f"{x:.{k}g}")
DP = lambda x, k: round(float(x), k)
H, HMAX = (1, 5, 21), 21


def sim(seed, n_train=2500, n_o=562):
    g = np.random.default_rng(seed)
    N = n_train + n_o + HMAX + 1
    mu, phi, sig = math.log(18.0), 0.98, 0.075
    x = np.empty(N); x[0] = mu + 0.3
    for t in range(1, N):
        x[t] = mu + phi * (x[t - 1] - mu) + sig * g.standard_normal()
    V = np.round(np.exp(x), 2)                                   # closes at 2 dp
    r = g.standard_normal(N) * 0.01 * (V / 18.0)                  # SPX-like returns, vol tied to V's state
    vxx = g.standard_normal(N) * 0.04 - 0.002                     # VXX-like simple returns
    return V, r, vxx, n_train, n_o


def fit_direct(lnV, n_train, h, extra):
    """OLS of ln V_{t+h} on [1, ln V_t, extra...] using training pairs with target <= n_train-1 (label rule)."""
    idx = np.arange(25, n_train - h)
    X = np.column_stack([np.ones(len(idx)), lnV[idx]] + [e[idx] for e in extra])
    b, *_ = np.linalg.lstsq(X, lnV[idx + h], rcond=None)
    s2 = np.var(lnV[idx + h] - X @ b, ddof=X.shape[1])
    return b, s2


def forecasts(V, r, n_train, n_o, h, scale_free_only=True):
    lnV = np.log(V)
    w5 = np.convolve(lnV, np.ones(5) / 5, "full")[: len(V)]       # trailing means (t-4..t)
    w22 = np.convolve(lnV, np.ones(22) / 22, "full")[: len(V)]
    o = np.arange(n_train, n_train + n_o)                           # origins O_p
    F = {"rw": V[o], "hist_mean": np.full(n_o, V[:n_train].mean())}
    b, s2 = fit_direct(lnV, n_train, h, [])
    F["ar1_log"] = np.exp(b[0] + b[1] * lnV[o] + s2 / 2)
    b, s2 = fit_direct(lnV, n_train, h, [w5, w22])
    F["har_log"] = np.exp(b[0] + b[1] * lnV[o] + b[2] * w5[o] + b[3] * w22[o] + s2 / 2)
    if not scale_free_only:                                         # forecast from SPX returns only: not V-equivariant
        ew = np.zeros(len(r)); lam = 0.94
        for t in range(1, len(r)):
            ew[t] = lam * ew[t - 1] + (1 - lam) * r[t] ** 2
        F["ewma_spx"] = 100 * np.sqrt(252 * ew[o])
    return F, V[o + h]


def lrv_rect(d, h):
    dc = d - d.mean(); n = len(d)
    g = [dc[j:] @ dc[: n - j] / n for j in range(h)]
    lrv = g[0] + 2 * sum(g[1:])
    if lrv <= 0:
        lrv = g[0] + 2 * sum((1 - j / h) * g[j] for j in range(1, h))
    return lrv


def dm(la, lb, h):
    d = la - lb; n = len(d)
    stat = d.mean() / math.sqrt(lrv_rect(d, h) / n) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), n - 1))


def mz_pass(y, yh, h):
    if np.std(yh) < 1e-12:
        return None, None                                            # constant forecast: MZ undefined
    n = len(y); X = np.column_stack([np.ones(n), yh])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); u = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); S = (X * u[:, None]).T @ (X * u[:, None])
    for j in range(1, h):                                           # Newey-West, lag h-1
        G = (X[j:] * u[j:, None]).T @ (X[:-j] * u[:-j, None]); S += (1 - j / h) * (G + G.T)
    C = XtXi @ S @ XtXi
    p_beta = 2 * (1 - stats.norm.cdf(abs((b[1] - 1) / math.sqrt(C[1, 1]))))
    dlt = b - np.array([0.0, 1.0]); w = dlt @ np.linalg.solve(C, dlt)
    return bool(p_beta >= 0.05), bool(1 - stats.chi2.cdf(w, 2) >= 0.05)


def strategy(F, V, vxx, n_train, n_o):
    o = np.arange(n_train, n_train + n_o)
    sig = np.sign(F["ar1_log"] - V[o])                               # long vol if forecast above spot
    ret = sig[:-2] * vxx[o[2:]]                                      # executed at close of t+1 (L7)
    sr = ret.mean() / ret.std(ddof=1)
    sk = stats.skew(ret, bias=True); ku = stats.kurtosis(ret, fisher=False, bias=True)
    psr = stats.norm.cdf(sr * math.sqrt(len(ret) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    emax = math.sqrt(1 / len(ret)) * ((1 - 0.5772) * stats.norm.ppf(1 - 1 / 5) + 0.5772 * stats.norm.ppf(1 - 1 / (5 * math.e)))
    dsr = stats.norm.cdf((sr - emax) * math.sqrt(len(ret) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    turn = np.abs(np.diff(sig)).sum() / len(sig) * 252
    return {"sharpe_ann": DP(sr * math.sqrt(252), 2), "psr0": DP(psr, 2), "dsr": DP(dsr, 2), "n_trials": 5, "turnover_ann": SIG(turn, 2)}


def publication_set(V, r, vxx, n_train, n_o, include_scale_tied=False):
    scale_free_only = not include_scale_tied
    P = {}
    for h in H:
        F, y = forecasts(V, r, n_train, n_o, h, scale_free_only)
        v = (y / 100) ** 2
        L = {m: {"se": (y - f) ** 2, "ae": np.abs(y - f), "ape": np.abs(y - f) / y,
                 "ql": v / (f / 100) ** 2 - np.log(v / (f / 100) ** 2) - 1} for m, f in F.items()}
        for m in F:
            P[("n", h, m)] = len(y)
            if m != "rw":
                P[("rel_rmse", h, m)] = SIG(math.sqrt(L[m]["se"].mean() / L["rw"]["se"].mean()), 3)
                P[("rel_qlike", h, m)] = SIG(L[m]["ql"].mean() / L["rw"]["ql"].mean(), 3)
                P[("rel_mae", h, m)] = SIG(L[m]["ae"].mean() / L["rw"]["ae"].mean(), 3)
                P[("rel_mape", h, m)] = SIG(L[m]["ape"].mean() / L["rw"]["ape"].mean(), 3)
                for loss, k in (("sq_error", "se"), ("qlike", "ql")):
                    s, p = dm(L[m][k], L["rw"][k], h)
                    P[("dm_stat", h, m, loss)] = DP(s, 2); P[("dm_p", h, m, loss)] = DP(p, 2)
            P[("mz_beta_pass", h, m)], P[("mz_joint_pass", h, m)] = mz_pass(y, F[m], h)
    o = np.arange(n_train, n_train + n_o); lv = np.log(V[o])
    phi = np.polyfit(lv[:-1], lv[1:], 1)[0]
    P[("half_life",)] = SIG(math.log(0.5) / math.log(phi), 2)
    for k, val in strategy(forecasts(V, r, n_train, n_o, 1, True)[0], V, vxx, n_train, n_o).items():
        P[("strategy", k)] = val
    return P


def diff(P, Q):
    return [k for k in P if P[k] != Q.get(k)]


if __name__ == "__main__":
    seed = 20261028
    V, r, vxx, n_train, n_o = sim(seed)
    P = publication_set(V, r, vxx, n_train, n_o)
    print(f"seed={seed}: |P| = {len(P)} published numbers for one period ({n_o} origins, {n_o + HMAX} closes in play)")
    print("sample:", {str(k): P[k] for k in list(P)[:6]})

    # (a) inputs the shift-telescoping solve (review D p1) and the R3 contrasts (p2) need
    need = {"eda mean of V": "eda_mean", "eda sd of V": "eda_sd", "MZ beta": "mz_beta", "MZ R^2": "mz_r2",
            "RMSE (absolute)": "rmse", "QLIKE_rw (absolute)": "qlike", "ann_vol": "ann_vol", "mean_d": "mean_d",
            "bias (review B A1)": "bias", "regime cells (B A2-A4)": "regime", "ann_return": "ann_return", "MZ alpha": "mz_alpha"}
    present = {name for name, key in need.items() if any(k[0] == key for k in P)}
    print(f"(a) attack inputs present in P: {sorted(present) or 'none'}  (review D p1 needs eda mean/sd, MZ beta, R^2, RMSE; p2 needs absolute RMSEs, QLIKE_rw, ann_vol; review B needs bias, regime cells)")
    print("    P key classes:", sorted({k[0] if k[0] != "strategy" else "strategy." + k[1] for k in P}))

    # (b) scale indistinguishability: V -> cV, returns -> c*returns. Identical P => no function of P can return the level.
    for c in (0.8, 1.25, 2.0):
        Pc = publication_set(V * c, r, vxx * c, n_train, n_o)
        print(f"(b) scale c={c}: P entries that differ (published rows: V-equivariant models + strategy) = {len(diff(P, Pc))} of {len(P)}")
    Pt = publication_set(V, r, vxx, n_train, n_o, include_scale_tied=True)
    dd = diff(Pt, publication_set(V * 1.25, r, vxx, n_train, n_o, include_scale_tied=True))
    print(f"(b') with the scale-tied SPX-driven model added (not in P), c=1.25: {len(dd)} of {len(Pt)} differ, all for model 'ewma_spx'? {all('ewma_spx' in k for k in dd)}")

    # (c) single-day indistinguishability: perturb one close by +/-delta; identical P => that day is not determinable within delta
    base = publication_set(V, r, vxx, n_train, n_o)
    days = np.arange(n_train - 21, n_train + n_o + HMAX)   # inputs of every origin and target in the period
    for delta in (0.25, 1.0):
        same, ends = 0, {}
        for d in days:
            ok = True
            for s in (+1, -1):
                W = V.copy(); W[d] += s * delta
                if diff(base, publication_set(W, r, vxx, n_train, n_o)):
                    ok = False; break
            same += ok
            if d in (n_train, n_train + n_o, n_train + n_o - 1 + HMAX):
                ends[int(d)] = ok
        print(f"(c) delta={delta} vol pts: P identical under +/-delta on {same}/{len(days)} single days "
              f"({100*same/len(days):.1f}%); endpoint days (first origin, day after last origin, last target): {ends}")
