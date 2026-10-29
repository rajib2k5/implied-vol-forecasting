"""Size of the Diebold-Mariano and Mincer-Zarnowitz tests at h = 21 (milestone 5; ADR 0001 O5, O6).

    uv run python tools/probes/dm_mz_size.py [dm_reps] [mz_reps]     # walkthrough: 2000 1000

Simulated data only (no file is read); seed 20261028. Prints, for each null and sample size, the
share of simulated samples on which each estimator rejects at a nominal 5%:

* DM (``ivf.evaluation.dm.simulate_differentials``): ``ma`` (linear MA(20) differential),
  ``sq_error`` (two equally good forecasts with overlapping errors, correlation 0.8) and ``state``
  (model A's accuracy depends on a persistent AR(1) state, phi = 0.98), at n = 583 (the holdout's
  size) and 6,090 (the dev size at h = 21).
* MZ (``ivf.evaluation.mz.simulate_mz``): a rational forecast of an AR(1) level (phi = 0.98), so
  alpha = 0 and beta = 1 hold, at the same two sizes.

The Monte Carlo standard error of a rate near 0.05 is about sqrt(0.05 * 0.95 / reps).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ivf.evaluation import dm, mz

SEED = 20261028
H = 21
SIZES = (583, 6090)


def main(argv: list[str]) -> int:
    dm_reps = int(argv[0]) if argv else 2000
    mz_reps = int(argv[1]) if len(argv) > 1 else 1000
    rng = np.random.default_rng(SEED)
    start = time.perf_counter()
    null = dm.fixed_b_null()
    print(f"fixed-b null (b = {dm.FIXED_B}, {null.size} draws, n = {dm.FIXED_B_SIM_N}): 95% quantile of |t| "
          f"{np.quantile(null, 0.95):.4f} (comparison with Kiefer-Vogelsang's tables not verified)")
    print(f"DM, h = {H}, nominal 0.05, {dm_reps} samples per cell (MC SE {np.sqrt(0.0475 / dm_reps):.4f})")
    print(f"{'design':10} {'n':>5} " + " ".join(f"{m:>12}" for m in dm.METHODS))
    for design in ("ma", "sq_error", "state"):
        for n in SIZES:
            rates = {m: 0.0 for m in dm.METHODS}
            done = 0
            while done < dm_reps:  # in chunks, to bound memory
                k = min(500, dm_reps - done)
                for m, r in dm.rejection_rates(dm.simulate_differentials(design, n, H, k, rng), H).items():
                    rates[m] += r * k
                done += k
            print(f"{design:10} {n:>5} " + " ".join(f"{rates[m] / dm_reps:>12.4f}" for m in dm.METHODS))
    print(f"MZ joint Wald, h = {H}, nominal 0.05, {mz_reps} samples per cell (MC SE {np.sqrt(0.0475 / mz_reps):.4f})")
    print(f"{'n':>5} " + " ".join(f"{m:>15}" for m in mz.COV_METHODS))
    for n in SIZES:
        rates = mz.simulate_mz(n, H, mz_reps, rng)
        print(f"{n:>5} " + " ".join(f"{rates[m]:>15.4f}" for m in mz.COV_METHODS))
    print(f"{time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
