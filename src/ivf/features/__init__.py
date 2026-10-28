"""Feature engineering (milestone 1).

Will hold the transformations from interim tables to model inputs: log VIX,
trailing realised volatility, the HAR regressors (daily, weekly and monthly
averages) and the other trailing-window features. Every feature may use only
data dated at or before the forecast origin (ADR 0001 rule L1). Nothing is
implemented here yet.
"""
