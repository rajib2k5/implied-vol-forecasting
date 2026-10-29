"""Feature engineering (milestone 1): ``build`` turns interim tables into the feature,
target_value and regime_label tables on the master calendar.

Every feature uses trailing windows only, so it holds data dated at or before
its session (ADR 0001 rule L1). Targets are the one forward-looking series.
"""
