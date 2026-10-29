"""Trading backtest (milestone 6): VXX/SVXY driven by the frozen forecasts.

* ``segments``: the tradable segments of ``configs/instruments.yaml`` and the checks that the fetched
  data support them (no gap, splits adjusted, exposure matching the stated leverage);
* ``engine``: signal at origin t, fill at the close of t+1, first return t+1 -> t+2 (ADR 0001 rule L7),
  costs per side on every change of weight;
* ``strategy``: the pre-registered rules of ``configs/backtest.yaml``, the dev-only model choice, the
  trial log and one period's backtest;
* ``metrics`` and ``sharpe``: return, volatility, Sharpe, Sortino, Calmar, drawdown, turnover, hit rate,
  PSR and DSR (``sharpe`` is copied from sharpe-observatory, MIT, with attribution);
* ``data``: the tradable price table and the synthetic VXX/SVXY; ``cli``: ``python -m ivf.backtest``.
"""
