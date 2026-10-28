"""Forecasting models (milestones 2 to 4).

Milestone 2 adds the baselines: random walk, historical mean, ARIMA on log VIX
and HAR. Milestone 3 adds the GARCH family (GARCH, GJR, EGARCH) fitted on
index returns, plus their calibrated versions. Milestone 4 adds the LSTM. All
models share one interface and one walk-forward protocol so their forecasts
are comparable (ADR 0001 D4). Nothing is implemented here yet.
"""
