"""Evaluation protocol and scoring (milestones 2 and 5).

Milestone 2 adds the shared trading calendar and the walk-forward fold generator that every model
uses (``walkforward``), plus the loss functions (``losses``: RMSE, MAE, MAPE, QLIKE). Milestone 5 adds
the Diebold-Mariano tests with Holm's correction (``dm``), the Mincer-Zarnowitz regressions (``mz``),
the regime breakdown (``regimes``), the walk-forward bias calibration (``calibration``) and the
``python -m ivf.evaluation report`` step that writes them (``report``) (ADR 0001 D4, amended M5).
"""
