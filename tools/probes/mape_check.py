"""Which point forecast minimises each loss, for y ~ lognormal(ln 20, 0.25^2)? (C1.9 check)"""
import numpy as np
mu, s = np.log(20.0), 0.25
rng = np.random.default_rng(20261028)
y = np.exp(mu + s * rng.standard_normal(2_000_000))
grid = np.linspace(17.0, 23.0, 6001)
def argmin(loss):
    vals = [loss(x) for x in grid]
    return grid[int(np.argmin(vals))]
mean_a, med_a, mape_a = np.exp(mu + s**2 / 2), np.exp(mu), np.exp(mu - s**2)  # analytic: mean, median, median of y-weighted-by-1/y
print(f"analytic  mean {mean_a:.4f}  median {med_a:.4f}  MAPE-optimal exp(mu-s^2) {mape_a:.4f}")
print(f"grid MC   SE-argmin {argmin(lambda x: np.mean((y-x)**2)):.3f}  AE-argmin {argmin(lambda x: np.mean(np.abs(y-x))):.3f}  APE-argmin {argmin(lambda x: np.mean(np.abs(y-x)/y)):.3f}")
for name, x in [("mean", mean_a), ("median", med_a), ("MAPE-opt", mape_a)]:
    print(f"forecast={name:8s} {x:.3f}: RMSE {np.sqrt(np.mean((y-x)**2)):.4f}  MAE {np.mean(np.abs(y-x)):.4f}  MAPE {np.mean(np.abs(y-x)/y):.5f}")
