"""Licensing probe for ADR 0001 D2 (amended M0, review E). SIMULATED DATA ONLY: every series is drawn
from a seeded generator in p_set.sim(); no file is read and no real market value is embedded.
Run from the repo root: uv run python tools/licensing/p_sens.py

Which P entry classes flip at their stated rounding when one close moves by +/-0.25 vol points?"""
import collections, numpy as np
import p_set as S
V, r, vxx, n_train, n_o = S.sim(20261028)
base = S.publication_set(V, r, vxx, n_train, n_o)
days = np.arange(n_train - 21, n_train + n_o + S.HMAX)
cnt = collections.Counter(); per_day = []
for d in days:
    flipped = set()
    for s in (+1, -1):
        W = V.copy(); W[d] += s * 0.25
        flipped |= {k[0] if k[0] != "strategy" else "strategy." + k[1] for k in S.diff(base, S.publication_set(W, r, vxx, n_train, n_o))}
    cnt.update(flipped); per_day.append(len(flipped))
print("days:", len(days), "| class -> number of days on which >=1 entry of that class flipped:")
for k, v in cnt.most_common(): print(f"  {k:14s} {v:4d} ({100*v/len(days):.1f}%)")
