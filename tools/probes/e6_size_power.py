"""Size and power of statement E6's test, ivf.eda.e6_level_vs_next_shock (amended M1, review B).

    uv run python tools/probes/e6_size_power.py [paths] [sessions] [scenario ...]   # walkthrough: 300 9000

Simulated data only (no file is read). ln V is an AR(1) with phi = 0.98, mu = ln 18, shock sd 0.06.
Each scenario runs the SHIPPED test (circular block bootstrap, block length by
``eda.e6_block_length``) and the review-A test (the same code with ``block=1``: iid draws) on the
same paths, and prints how often each rejects at 5% (two-sided p < 0.05), how often the E6 verdict
says "yes" (rho > 0 and p < 0.05), rho's mean and sd, and the mean block length the rule chose.

Nulls (the shock size does not depend on the level, so a 5% test should reject on about 5% of
paths or fewer; the walkthrough's acceptance bound is 8%):
  normal, skew            iid N(0,1) and iid Exp(1) - 1 shocks
  garchmild, garch        symmetric GARCH(1,1) shocks, (alpha, beta) = (0.05, 0.90) and (0.10, 0.88)
  skewgarchmild, skewgarch  the same GARCH recursions driven by Exp(1) - 1 innovations; here the
                          level and the next shock size correlate in the population (a big up-move
                          raises both), with no level effect in the volatility equation
  skewgarch995            skewed GARCH with alpha + beta = 0.995 (volatility half-life 138 sessions)
  gjr                     normal innovations, volatility rising more after UP moves (0.02, 0.12, 0.88)
  skew995, skewgaps       iid skewed shocks with phi = 0.995; and with 60 sessions missing
  skew98both, skew985both, skew99both, skew995both
                          skewed GARCH(0.03, 0.965) shocks with phi = 0.98, 0.985, 0.99, 0.995: the
                          clustering AND the level persistent (M1 final check). The last two fail:
                          the block (mean about 245, near the rule's cap) is shorter than the
                          level's memory, so E6 answers only when phi_hat <= eda.E6_PHI_MAX
  lnorm995both            the same with standardised lognormal(0, 1) innovations, phi = 0.995
Alternatives (the shock sd is 0.06*min(exp(k*(x_{t-1} - mu)), 3), so it rises with the level):
  level0.5, level0.15     power
  levelgarch0.5           the same level effect on top of GARCH(0.05, 0.90) clustering

E6_PROBE_SEED_OFFSET=K adds K to every seed (fresh paths). About 1 minute per scenario for 300 paths of 9,000 sessions (999 resamples, 18 processes): 13 m 47 s for all 13.
"""

import math
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

from ivf import eda  # noqa: E402

warnings.simplefilter("ignore")
PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
PHI, MU, SD = 0.98, math.log(18), 0.06
NULLS_B = ["normal", "skew", "garchmild", "garch", "skewgarchmild", "skewgarch", "skewgarch995", "gjr", "skew995", "skewgaps"]
NULLS_FINAL = ["skew98both", "skew985both", "skew99both", "skew995both", "lnorm995both"]  # M1 final check
NULLS = NULLS_B + NULLS_FINAL
ALTERNATIVES = ["level0.5", "level0.15", "levelgarch0.5"]
SCENARIOS = sys.argv[3:] or NULLS + ALTERNATIVES
# Review B's seeds are kept (its table stays reproducible); the final-check rows get fresh ones.
SEED0 = {name: 300_000 + 1_000 * i for i, name in enumerate(NULLS_B + ALTERNATIVES)}
SEED0 |= {name: 400_000 + 1_000 * i for i, name in enumerate(NULLS_FINAL)}
CUTOFFS = (0.975, 0.98, 0.985)  # the candidate persistence cutoffs the final check chose among
# Fresh seeds for a re-run (the E6_MIN_N choice used 1e6, 2e6 and 3e6 for n = 4,000, 6,000, 8,000).
SEED_OFFSET = int(os.environ.get("E6_PROBE_SEED_OFFSET", "0"))
SEED0 = {name: seed + SEED_OFFSET for name, seed in SEED0.items()}


def lognormal_innovations(rng):
    """Standardised lognormal(0, 1): mean 0, sd 1, a heavier right skew than Exp(1) - 1."""
    m, var = math.exp(0.5), (math.e - 1.0) * math.e
    return (rng.lognormal(0.0, 1.0, size=N) - m) / math.sqrt(var)


def garch(rng, a, b, skewed, gamma=0.0, z=None):
    """GARCH(1,1) (GJR if gamma > 0: + gamma*e^2 after an UP move) shocks with unconditional sd SD."""
    if z is None:
        z = (rng.exponential(size=N) - 1.0) if skewed else rng.standard_normal(N)
    w = SD**2 * (1 - a - gamma / 2 - b)
    e, s2 = np.empty(N), SD**2
    for t in range(N):
        e[t] = math.sqrt(s2) * z[t]
        s2 = w + (a + gamma * (e[t] > 0)) * e[t] ** 2 + b * s2
    return e


def path(name, seed):
    rng = np.random.default_rng(seed)
    ar = lambda e, phi=PHI: MU + lfilter([1.0], [1.0, -phi], e)  # noqa: E731
    if name == "normal":
        return ar(SD * rng.standard_normal(N))
    if name == "skew":
        return ar(SD * (rng.exponential(size=N) - 1.0))
    if name == "skew995":
        return ar(SD / 2 * (rng.exponential(size=N) - 1.0), 0.995)
    if name == "skewgaps":
        x = ar(SD * (rng.exponential(size=N) - 1.0))
        x[rng.choice(N, 60, replace=False)] = np.nan
        return x
    if name in ("garchmild", "garch", "skewgarchmild", "skewgarch"):
        a, b = (0.05, 0.90) if name.endswith("mild") else (0.10, 0.88)
        return ar(garch(rng, a, b, name.startswith("skew")))
    if name == "skewgarch995":
        return ar(garch(rng, 0.03, 0.965, True))
    if name in ("skew98both", "skew985both", "skew99both", "skew995both"):
        return ar(garch(rng, 0.03, 0.965, True), {"98": 0.98, "985": 0.985, "99": 0.99, "995": 0.995}[name[4:-4]])
    if name == "lnorm995both":
        return ar(garch(rng, 0.03, 0.965, True, z=lognormal_innovations(rng)), 0.995)
    if name == "gjr":
        return ar(garch(rng, 0.02, 0.88, False, gamma=0.12))
    if name.startswith("level"):
        k = float(name.removeprefix("levelgarch").removeprefix("level"))
        clustered = name.startswith("levelgarch")
        z = rng.standard_normal(N)
        x, h = np.empty(N), 1.0
        x[0] = MU
        for t in range(1, N):
            u = math.sqrt(h) * z[t]
            x[t] = MU + PHI * (x[t - 1] - MU) + SD * min(math.exp(k * (x[t - 1] - MU)), 3.0) * u
            if clustered:
                h = 0.05 + 0.05 * u * u + 0.90 * h
        return x
    raise SystemExit(f"unknown scenario {name}")


def one(job):
    name, seed = job
    v = np.exp(path(name, seed))
    _, phi_hat, shocks = eda.ar1_shocks(np.log(v))
    block = eda.e6_block_length(shocks)
    rho, p_shipped, _ = eda.e6_level_vs_next_shock(v, block=block)
    _, p_iid, _ = eda.e6_level_vs_next_shock(v, block=1)
    return rho, p_shipped, p_iid, block, phi_hat


if __name__ == "__main__":
    print(f"E6 size/power: {PATHS} paths of {N} sessions per scenario, B = {eda.E6_BOOTSTRAP}, seed {eda.E6_SEED}")
    with Pool(min(18, os.cpu_count() or 1)) as pool:
        for name in SCENARIOS:
            t0 = time.time()
            seeds = range(SEED0[name], SEED0[name] + PATHS)
            out = np.array(pool.map(one, [(name, s) for s in seeds]))
            rho, p_new, p_old, block, phi_hat = out.T
            kind = "H0" if name in NULLS else "H1"

            def rate(p):
                return f"{int(np.sum(p < 0.05)):3}/{PATHS} = {np.mean(p < 0.05):.3f} (yes {int(np.sum((rho > 0) & (p < 0.05)))})"

            print(
                f"{name:13} {kind} seeds {seeds.start}..{seeds.stop - 1}: shipped p<0.05 {rate(p_new)}; "
                f"review-A iid {rate(p_old)}; rho mean {rho.mean():+.4f} sd {rho.std():.4f}; "
                f"block mean {block.mean():.1f} [{int(block.min())}, {int(block.max())}]; "
                f"phi_hat mean {phi_hat.mean():.4f} [{phi_hat.min():.4f}, {phi_hat.max():.4f}]; "
                "yes with phi_hat <= cutoff " + ", ".join(
                    f"{c}: {int(np.sum((rho > 0) & (p_new < 0.05) & (phi_hat <= c)))}" for c in CUTOFFS
                ) + f"; {time.time() - t0:.0f}s",
                flush=True,
            )
