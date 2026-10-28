"""Data access (milestone 1).

Will hold the code that fetches Cboe, FRED and Yahoo Finance files into the
raw store as SHA-256-named snapshots with an append-only provenance log, the
step that turns raw files into typed interim parquet tables, and the seeded
synthetic data generator that lets CI run the real pipeline without any
licensed data (ADR 0001 D2 and D3). Nothing is implemented here yet.
"""
