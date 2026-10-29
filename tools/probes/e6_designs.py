"""E6 bootstrap DESIGNS compared on the same paths (M1 review B fix round; synthetic data only, no file read).

    uv run python tools/probes/e6_designs.py SCHEME SCENARIO PATHS SEED0 [SESSIONS] [B]
    # walkthrough 7.3: every SCHEME in iid rad mam mbb sb, every SCENARIO below, 300 paths, seeds 201000 + 1000*i

SCHEME (how the fitted AR(1) shocks are resampled; the AR(1) is refitted on every path, as shipped):
  iid  draws with replacement, one at a time (review A's design)
  rad  wild bootstrap, e* = e_hat * eta with Rademacher eta (+1 or -1)
  mam  wild bootstrap with Mammen's two-point eta (keeps the third moment)
  mbb  circular blocks of L = Politis-White on |shock| (arch's "circular" column), a Python loop; the
       shipped code (ivf.eda.e6_level_vs_next_shock) is the same design, vectorised
  sb   stationary bootstrap, geometric block lengths with mean Politis-White on |shock| ("stationary")
  cbb  the vectorised circular blocks; mbbN or cbbN fixes L = N
SCENARIO: normal, skew (iid exponential minus 1), garchmild (0.05, 0.90), garch (0.10, 0.88),
  skewgarchmild, skewgarch (the same, exponential innovations), skewgarch995, gjr, skew995, skewgaps,
  level0.5, level0.15, levelgarch0.5. tools/probes/e6_size_power.py describes them; this file keeps
  its own copy so that the design table can be re-run exactly as measured.
It prints the share of paths with two-sided p < 0.05, the verdict "yes" count (rho > 0 and p < 0.05),
rho's mean and sd, and the mean block length.
"""
import math, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import signal
from multiprocessing import Pool
warnings.simplefilter("ignore")

PHI, MU = 0.98, math.log(18)


def garch_shocks(rng, n, a, b, skewed):
    z = (rng.exponential(size=n) - 1.0) if skewed else rng.standard_normal(n)
    w = 0.06**2 * (1 - a - b)
    e = np.empty(n); s2 = 0.06**2
    for t in range(n):
        e[t] = math.sqrt(s2) * z[t]
        s2 = w + a * e[t] ** 2 + b * s2
    return e


def path(scen, seed, n):
    rng = np.random.default_rng(seed)
    if scen == "skew":
        return MU + signal.lfilter([1.0], [1.0, -PHI], 0.06 * (rng.exponential(size=n) - 1.0))
    if scen == "normal":
        return MU + signal.lfilter([1.0], [1.0, -PHI], 0.06 * rng.standard_normal(n))
    if scen == "skew995":
        return MU + signal.lfilter([1.0], [1.0, -0.995], 0.03 * (rng.exponential(size=n) - 1.0))
    if scen in ("garchmild", "garch", "skewgarch", "skewgarchmild"):
        a, b = (0.05, 0.90) if scen.endswith("mild") else (0.10, 0.88)
        return MU + signal.lfilter([1.0], [1.0, -PHI], garch_shocks(rng, n, a, b, scen.startswith("skew")))
    if scen == "skewgarch995":  # very persistent clustering: alpha + beta = 0.995 (vol half-life 138 sessions)
        return MU + signal.lfilter([1.0], [1.0, -PHI], garch_shocks(rng, n, 0.03, 0.965, True))
    if scen == "gjr":  # symmetric normal z, but vol rises more after UP moves (inverse leverage, like VIX)
        z = rng.standard_normal(n); a, g, b = 0.02, 0.12, 0.88
        w = 0.06**2 * (1 - a - g / 2 - b)
        e = np.empty(n); s2 = 0.06**2
        for t in range(n):
            e[t] = math.sqrt(s2) * z[t]
            s2 = w + (a + g * (e[t] > 0)) * e[t] ** 2 + b * s2
        return MU + signal.lfilter([1.0], [1.0, -PHI], e)
    if scen.startswith("levelgarch"):  # level effect on top of symmetric GARCH(0.05, 0.90) clustering
        k = float(scen[10:])
        z = rng.standard_normal(n); a, b = 0.05, 0.90
        x = np.empty(n); x[0] = MU; h = 1.0
        for t in range(1, n):
            u = math.sqrt(h) * z[t]
            x[t] = MU + PHI * (x[t - 1] - MU) + 0.06 * min(math.exp(k * (x[t - 1] - MU)), 3.0) * u
            h = (1 - a - b) + a * u * u + b * h
        return x
    if scen.startswith("level"):
        k = float(scen[5:])
        z = rng.standard_normal(n)
        x = np.empty(n); x[0] = MU
        for t in range(1, n):
            x[t] = MU + PHI * (x[t - 1] - MU) + 0.06 * min(math.exp(k * (x[t - 1] - MU)), 3.0) * z[t]
        return x
    if scen == "skewgaps":
        x = MU + signal.lfilter([1.0], [1.0, -PHI], 0.06 * (rng.exponential(size=n) - 1.0))
        x[rng.choice(n, 60, replace=False)] = np.nan
        return x
    raise SystemExit(scen)


def shocks_for(scheme, rng, rows, resid_full, residuals, L):
    """Bootstrap shocks of shape (rows, m) where m = len(resid_full)."""
    m = len(resid_full)
    if scheme == "iid":
        return rng.choice(residuals, size=(rows, m), replace=True)
    if scheme in ("rad", "mam"):
        base = resid_full.copy()
        holes = np.isnan(base)
        e = np.broadcast_to(base, (rows, m)).copy()
        if holes.any():
            e[:, holes] = rng.choice(residuals, size=(rows, int(holes.sum())), replace=True)
        if scheme == "rad":
            eta = rng.choice(np.array([-1.0, 1.0]), size=(rows, m))
        else:
            s5 = math.sqrt(5)
            lo, hi = -(s5 - 1) / 2, (s5 + 1) / 2
            p_lo = (s5 + 1) / (2 * s5)
            eta = np.where(rng.random((rows, m)) < p_lo, lo, hi)
        return e * eta
    if scheme == "cbb":
        nr = len(residuals)
        nblocks = -(-m // L)
        starts = rng.integers(nr, size=(rows, nblocks))
        idx = ((starts[:, :, None] + np.arange(L)) % nr).reshape(rows, nblocks * L)[:, :m]
        return residuals[idx]
    if scheme.startswith("mbb") or scheme == "sb":
        r = residuals
        nr = len(r)
        out = np.empty((rows, m))
        ext = np.concatenate([r, r])  # circular
        for i in range(rows):
            pos = 0
            while pos < m:
                if scheme == "sb":
                    ln = int(rng.geometric(1.0 / L))
                else:
                    ln = L
                s = int(rng.integers(nr))
                take = min(ln, m - pos, nr)
                out[i, pos:pos + take] = ext[s:s + take]
                pos += take
        return out
    raise SystemExit(scheme)


def e6(vix, scheme, n_boot=999, seed=20261029, Lfixed=None):
    from ivf import eda
    v = np.asarray(vix, float)
    log_v = np.log(v)
    c, phi, shock = eda.ar1_shocks(log_v)
    pairs = ~np.isnan(shock[1:])
    level, size = log_v[:-1][pairs], np.abs(shock[1:][pairs])
    rho = float(eda._spearman_rows(level, size))
    rng = np.random.default_rng(seed)
    residuals = shock[~np.isnan(shock)]
    residuals = residuals - residuals.mean()
    first = int(np.flatnonzero(~np.isnan(log_v))[0])
    missing = np.isnan(log_v[first:])
    resid_full = shock[first + 1:] - np.nanmean(shock)
    L = None
    if scheme.startswith("mbb") or scheme in ("sb", "cbb"):
        if Lfixed:
            L = Lfixed
        else:
            from arch.bootstrap import optimal_block_length
            ob = optimal_block_length(np.abs(residuals))
            L = float(ob["stationary" if scheme == "sb" else "circular"].iloc[0])
            L = max(1, int(round(L))) if scheme != "sb" else max(1.0, L)
    if scheme == "cbb" and Lfixed:
        L = Lfixed
    draws = np.empty(n_boot)
    CH = 128
    for lo in range(0, n_boot, CH):
        rows = min(CH, n_boot - lo)
        e = shocks_for(scheme, rng, rows, resid_full, residuals, L)
        path_ = np.empty((rows, len(missing)))
        path_[:, 0] = log_v[first]
        path_[:, 1:] = signal.lfilter([1.0], [1.0, -phi], c + e, axis=1, zi=np.full((rows, 1), phi * log_v[first]))[0]
        path_[:, missing] = np.nan
        y, lagged = path_[:, 1:][:, pairs[first:]], path_[:, :-1][:, pairs[first:]]
        c_b, phi_b = eda._ols_rows(lagged, y)
        draws[lo:lo + rows] = eda._spearman_rows(lagged, np.abs(y - c_b - phi_b * lagged))
    return rho, eda.two_sided_bootstrap_p(rho, draws), (L if L is not None else 0)


def one(args):
    scheme, scen, seed, n, b = args
    Lfixed = int(scheme[3:]) if scheme[:3] in ("mbb", "cbb") and len(scheme) > 3 else None
    x = path(scen, seed, n)
    return e6(np.exp(x), scheme[:3] if Lfixed else scheme, n_boot=b, Lfixed=Lfixed)


if __name__ == "__main__":
    scheme, scen, ns, s0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 9000
    b = int(sys.argv[6]) if len(sys.argv) > 6 else 999
    t0 = time.time()
    with Pool(18) as pool:
        res = np.array(pool.map(one, [(scheme, scen, s, n, b) for s in range(s0, s0 + ns)]))
    rho, p, L = res[:, 0], res[:, 1], res[:, 2]
    rej = int(np.sum(p < 0.05)); yes = int(np.sum((rho > 0) & (p < 0.05)))
    print(f"{scheme:7s} {scen:13s} n={n} B={b} seeds={s0}..{s0+ns-1}: p<0.05 {rej}/{ns} = {rej/ns:.3f}; "
          f"rho>0&p<0.05 {yes}/{ns}; rho mean {np.nanmean(rho):+.4f} sd {np.nanstd(rho):.4f}; "
          f"L mean {np.mean(L):.1f}; {time.time()-t0:.0f}s", flush=True)
