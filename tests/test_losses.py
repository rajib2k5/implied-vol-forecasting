"""ivf.evaluation.losses: the ADR 0001 4.3 loss table, on hand-worked numbers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ivf.evaluation import losses

# The worked example used throughout: two scored rows.
#   y    = [10, 20]   (vol points)
#   yhat = [12, 15]
#   e = y - yhat = [-2, 5]
Y = [10.0, 20.0]
YHAT = [12.0, 15.0]


def test_bias_is_the_mean_of_y_minus_yhat():
    # (-2 + 5) / 2 = 1.5: positive, so the model under-forecasts on average.
    assert losses.bias(Y, YHAT) == pytest.approx(1.5, abs=1e-15)


def test_rmse():
    # sqrt(((-2)^2 + 5^2) / 2) = sqrt((4 + 25) / 2) = sqrt(14.5) = 3.8078865529319543
    assert losses.rmse(Y, YHAT) == pytest.approx(3.8078865529319543, rel=1e-15)


def test_mae():
    # (|-2| + |5|) / 2 = 7 / 2 = 3.5
    assert losses.mae(Y, YHAT) == pytest.approx(3.5, abs=1e-15)


def test_mape_divides_by_the_observed_y():
    # (2/10 + 5/20) / 2 = (0.2 + 0.25) / 2 = 0.225. Dividing by yhat instead would give
    # (2/12 + 5/15) / 2 = 0.25, so the denominator is pinned.
    assert losses.mape(Y, YHAT) == pytest.approx(0.225, rel=1e-14)


def test_qlike_is_pattons_normalised_form_in_variance_space():
    # v = (y/100)^2 = [0.01, 0.04]; v_hat = (yhat/100)^2 = [0.0144, 0.0225]
    # r = v / v_hat = [25/36, 16/9] = [0.694444, 1.777778]
    # r - ln r - 1: 0.694444 + 0.364643 - 1 = 0.0590875580; 1.777778 - 0.575364 - 1 = 0.2024136329
    # mean = 0.13075059545328482
    # In vol space (r = y/yhat) the answer would be 0.0338, and with v_hat/v swapped 0.112.
    assert losses.qlike(Y, YHAT) == pytest.approx(0.13075059545328482, rel=1e-13)


def test_qlike_is_zero_only_at_a_perfect_forecast():
    assert losses.qlike(Y, Y) == 0.0
    assert losses.qlike(Y, [10.0, 20.000001]) > 0


@pytest.mark.parametrize("fn", [losses.mape, losses.qlike])
def test_losses_that_divide_need_positive_values(fn):
    with pytest.raises(ValueError):
        fn([0.0, 20.0], YHAT)


def test_qlike_refuses_a_zero_forecast():
    with pytest.raises(ValueError):
        losses.qlike(Y, [0.0, 15.0])


@pytest.mark.parametrize("bad", [[np.nan, 20.0], [math.inf, 20.0]])
def test_a_loss_refuses_unscored_rows(bad):
    with pytest.raises(ValueError, match="scored rows only"):
        losses.rmse(bad, YHAT)


def test_loss_row_suppresses_a_cell_below_min_cell_n():
    # n = 2 < 3: every loss is None and n is kept (ADR 0001 4.8).
    row = losses.loss_row(np.array(Y), np.array(YHAT), min_cell_n=3)
    assert row == {"n": 2, "suppressed": True, "bias": None, "rmse": None, "mae": None, "mape": None, "qlike": None}
    row = losses.loss_row(np.array(Y), np.array(YHAT), min_cell_n=2)
    assert row["suppressed"] is False and row["n"] == 2
    assert row["rmse"] == pytest.approx(3.8078865529319543, rel=1e-15) and row["qlike"] == pytest.approx(0.13075059545328482, rel=1e-13)


def test_each_loss_rewards_the_forecast_the_docstring_says():
    # A lognormal sample: ln y ~ N(mu, s^2) with mu = ln 20, s = 0.25, drawn by quantiles so the
    # sample is exact to discretisation, not to Monte Carlo noise. The minimisers (losses module
    # docstring) are exp(mu - s^2) = 18.788, exp(mu) = 20, exp(mu + s^2/2) = 20.635 and
    # exp(mu + s^2) = 21.300; each loss, evaluated on a grid, must be lowest at its own one.
    from scipy.stats import norm

    mu, s = math.log(20), 0.25
    y = np.exp(mu + s * norm.ppf((np.arange(20_000) + 0.5) / 20_000))
    grid = np.linspace(18.0, 22.0, 401)  # steps of 0.01

    def argmin(fn):
        return grid[int(np.argmin([fn(y, np.full(y.size, g)) for g in grid]))]

    assert argmin(losses.mape) == pytest.approx(math.exp(mu - s**2), abs=0.011)
    assert argmin(losses.mae) == pytest.approx(math.exp(mu), abs=0.011)
    assert argmin(losses.rmse) == pytest.approx(math.exp(mu + s**2 / 2), abs=0.011)
    assert argmin(losses.qlike) == pytest.approx(math.exp(mu + s**2), abs=0.011)
