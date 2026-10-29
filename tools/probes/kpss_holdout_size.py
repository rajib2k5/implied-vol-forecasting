"""How often ADF, KPSS and statement E7 call a stationary AR(1) of holdout size non-stationary (M1 review A).

    uv run python tools/probes/kpss_holdout_size.py

Simulated data only. 100 paths of 583 sessions (the holdout to 2026-10-26) with phi = 0.98.
Measured 2026-10-29: KPSS rejects 61/100, ADF rejects 44/100, E7 says "not stationary" on 77/100.
"""
import warnings, numpy as np
from ivf import eda
warnings.simplefilter("ignore")
def ar1(phi, n, rng):
    e = rng.standard_normal(n); x = np.empty(n); x[0] = e[0] / np.sqrt(1 - phi**2)
    for t in range(1, n): x[t] = phi * x[t-1] + e[t]
    return x
k = a = no = 0
for s in range(100):
    u = eda.unit_root_tests(ar1(0.98, 583, np.random.default_rng(s)))
    k += u.kpss_p < 0.05; a += u.adf_p < 0.05; no += not u.stationary_at_5pct
print(f"phi=0.98 n=583 seeds 0-99: KPSS rejects {k}/100, ADF rejects {a}/100, E7 verdict 'not stationary' {no}/100")
