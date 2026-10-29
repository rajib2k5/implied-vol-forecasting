"""Data access (milestone 1). Run it as ``uv run python -m ivf.data <command>`` (see ``cli``).

* ``raw``: the content-addressed raw store and its append-only provenance log (ADR 0001 D3).
* ``fetch``: Cboe CSVs and yfinance prices into the raw store; ``network_get`` is the only
  function that touches the network.
* ``build_interim``: raw snapshots to typed, de-duplicated interim parquet.
* ``calendar``: the master calendar 𝒯 from the XNYS sessions (ADR 0001 D4.1).
* ``check``: data_check, the gap policy and the settled ``data_end``.
* ``process``: features, targets, regimes and the local EDA report under ``processed/``.
* ``synthetic``: a seeded generator with known truth that writes the same file shapes, so CI
  runs the real pipeline without licensed data (ADR 0001 D2).
"""
