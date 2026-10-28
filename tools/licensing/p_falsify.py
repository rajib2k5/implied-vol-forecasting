"""Licensing probe for ADR 0001 D2 (amended M0, review E). SIMULATED DATA ONLY: every series is drawn
from a seeded generator in p_set.sim(); no file is read and no real market value is embedded.
Run from the repo root: uv run python tools/licensing/p_falsify.py

Falsifier for probe (b): the scale test must flag level-unit statistics (the old R3 set) if they were published."""
import numpy as np
import p_set as S
V, r, vxx, n_train, n_o = S.sim(20261028)
def old_r3(V, vxx):
    F, y = S.forecasts(V, r, n_train, n_o, 1, True); o = V[n_train:n_train + n_o]
    e = y - F["rw"]; X = np.column_stack([np.ones(n_o), F["rw"]]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    ret = vxx[n_train:n_train + n_o]
    return {"eda_sd": S.SIG(o.std(ddof=1), 4), "eda_mean": S.SIG(o.mean(), 4), "rmse_rw": S.SIG(np.sqrt((e**2).mean()), 4),
            "mz_beta": S.SIG(b[1], 4), "ann_vol": S.SIG(ret.std(ddof=1) * np.sqrt(252), 4)}
a, b = old_r3(V, vxx), old_r3(V * 1.25, vxx * 1.25)
print("old-R3 statistic under c=1.25 ->", {k: (a[k], b[k], "CHANGED" if a[k] != b[k] else "same") for k in a})
