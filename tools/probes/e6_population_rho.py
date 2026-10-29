"""E6's rho in the population (n = 10^6) when the shock size never depends on the level (M1 review B fix round).

    uv run python tools/probes/e6_population_rho.py      # walkthrough 7.3, about 4 s with 16 processes

Simulated data only. For each scenario of tools/probes/e6_designs.py with no level effect, 8 paths of
10^6 sessions (seeds 700000..700007), E6's rho with no bootstrap: its mean and sd over the 8 paths.
Symmetric shocks give 0 by symmetry; skewed clustered shocks do not, which is why no test of
"rho = 0" can have the right size on them.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_designs import path  # noqa: E402

from ivf import eda  # noqa: E402

SCENARIOS = ["normal", "skew", "garchmild", "garch", "skewgarchmild", "skewgarch"]


def one(job):
    scenario, seed = job
    return eda.e6_level_vs_next_shock(np.exp(path(scenario, seed, 1_000_000)), n_boot=0)[0]


if __name__ == "__main__":
    with Pool(min(16, os.cpu_count() or 1)) as pool:
        for scenario in SCENARIOS:
            rho = np.array(pool.map(one, [(scenario, 700_000 + s) for s in range(8)]))
            print(f"{scenario:14} n=1e6 x8: rho mean {rho.mean():+.4f} sd {rho.std():.4f}", flush=True)
