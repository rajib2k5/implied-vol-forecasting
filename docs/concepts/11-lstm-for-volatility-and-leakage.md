# 11 · An LSTM for volatility, and the ways it can see the future

Milestone 4 adds one neural model, `lstm_log`, to a project whose other models are a few lines of
algebra. This note says in plain words what the network does, why it is set up the way it is, and the
four places where a neural forecaster leaks future information that the linear models cannot.
Background: [02 · Walk-forward evaluation](02-walk-forward-evaluation-and-look-ahead.md) (rules L1 to
L4), [08 · HAR](08-har-and-heterogeneous-memory.md), [09 · The GARCH family](09-garch-family-and-variance-aggregation.md).
Code: [`src/ivf/models/lstm.py`](../../src/ivf/models/lstm.py); notebook 04; walkthrough
[milestone 4](../milestones/04-lstm.md).

## 1. What an LSTM is, here

- A **recurrent network** reads a sequence one step at a time and keeps a small vector of state (the
  "hidden state", 32 numbers here) that it updates at each step. An **LSTM** (long short-term memory,
  Hochreiter & Schmidhuber 1997, in the references of concept 00) is a recurrent network whose update
  has gates, learned switches that decide how much of the old state to keep and how much of the new
  input to let in. The gates are what let it carry information across many steps.
- **Our sequence is short and fixed:** the 22 sessions `t-21..t` (one HAR "month"). On the VIX track
  each step is `ln VIX`; on the `rv21` track each step is that day's S&P 500 return.
- **HAR sits beside it.** The output layer sees the LSTM's last hidden state *and* HAR's regressors at
  t (today's value, the 5-session and the 22-session averages). With the LSTM part switched off, the
  network is a linear HAR-type regression. So the LSTM can only add to HAR, and HAR is the model it
  has to beat (concept 08).
- **It predicts a change, in logs:** `ln y_{t+h} − ln y_t` (VIX track) or `ln rv21_{t+21} − ln rv21_t`.
  Volatility is persistent, so "no change" is a strong default; starting from it is easier than
  learning the level from scratch.
- **Several horizons at once.** On the VIX track one network has three outputs, h = 1, 5 and 21: a
  *direct* forecast for each (ADR 0001 4.5), not a one-step model iterated.

The literature the project cites expects modest gains: Bucci (2020) and Christensen, Siggaard &
Veliyev (2023) report machine-learning models beating HAR-family models on *realised* volatility, the
latter even on HAR's own lags (concept 00, section 4.2). Whether that carries over to the VIX level,
already a forecast made by the market, is the question this milestone measures. It does not assume it.

## 2. From a log forecast to a vol-point forecast

The network gives `mu`, a forecast of the **mean of `ln y`**. As for `har_log` (milestone 2), the
forecast that RMSE rewards is the mean of `y` itself, and if `ln y` is roughly normal with variance
`s2`, that is `exp(mu + s2/2)`, a little above `exp(mu)` (the median). `s2` has to come from
somewhere: the network has no variance output, so we use its **mean squared error on the validation
slice** (section 3), data the weights were not trained on. That slice also picked the training epoch,
so `s2` is slightly optimistic. Both `exp(mu + s2/2)` (scored) and `exp(mu)` (stored as
`yhat_naive`) are kept, so the metric table shows what the correction is worth.

## 3. Four ways a neural forecaster sees the future

The linear models of M2 and M3 leak only if their *inputs* or *training pairs* do. A network trained
by gradient descent has two more doors: the numbers used to scale its inputs, and the data used to
decide when to stop training.

| Door | The leak | The rule | How the code shuts it |
|---|---|---|---|
| **Inputs** | A window that reaches past t (a centred average, a value from tomorrow) | L1: trailing only | Every feature at t uses sessions `≤ t`; a forecast at t is the same whether or not data after t exists (the truncation test, **bit for bit**) |
| **Training pairs** | Pairing origin τ with a target dated after the cutoff c | L2: target dated `≤ c` | Origins `τ ≤ c − H` only. For `rv21` the target is a 21-session *window*, so this is the window rule of M3 (O-M2-7): the window `τ+1..τ+21` must end by c |
| **The scaler** | Standardising with a mean and spread that include the validation rows (or, worse, the forecast period) | L3: scalers fitted on the rows used for gradient updates only | The scaler is fitted on the training slice alone, at every training cutoff |
| **Early stopping** | The epoch is chosen by a loss whose targets overlap the training targets, so validation looks better than it is and training runs too long | L4: the last 252 usable origins are the validation block; an h-session gap separates training from it | Training origins end `H` sessions before the first validation origin `v0`, so the last training target is dated `v0`; no validation target was trained on |

The last row is the one people miss. With a 21-session target, the training pair at `v0 − 1` would
have the target window `v0..v0+20`, and the validation pair at `v0` the window `v0+1..v0+21`: twenty
of their twenty-one returns are the same. Training on the first and validating on the second tests the
network on data it has nearly seen. The 21-session gap removes that overlap; it is what the walk-forward
literature calls a *purge* (the term is used in concept 02; its original source is not in this repo's
references, *to verify*).

The tests make each door a behaviour, not a reading of the code: poisoning every value that only the
validation slice reads leaves the scaler and the weights unchanged to the last bit, while poisoning the
single last training target changes the weights (`tests/test_lstm.py`). That holds at one epoch,
where nothing is chosen; with early stopping, the validation values legitimately pick which epoch is
kept, so they can change the kept weights without any leak (amended M4, review A).

## 4. Seeds and determinism

A network's starting weights and the order of its training batches are random. Two seeds can give
forecasts that differ by more than two *models* do, so one seed proves little. Here every training
cutoff trains five networks (members), and the forecast is the mean of their `mu`. Each member's own
forecast is stored (`forecast_member`), so the spread across seeds can be compared with the gap
between models.

Each member's seed comes from the protocol's one root seed, the model id, the cutoff date and the member
number (ADR 0001 D5), never from a position in a list, so reordering anything does not change it.
Every training runs on one CPU thread with deterministic algorithms, so the same inputs give the same
weights bit for bit, whether the member runs in the main process or in a worker. On another machine or
thread count, the low-order bits can differ (ADR 0002): conclusions are checked across seeds, not by
matching hashes across machines.

## 5. The refit schedule

Retraining five networks on two tracks at every quarterly cutoff (97 cutoffs) was measured at about
100 seconds of single-thread training per cutoff, roughly 2.8 hours in all on one core (derived from per-member times)
(milestone 4, section 7). The default retrains every fourth cutoff (every 252 sessions) and applies the
trained network, unchanged, in between. This **handicaps** the LSTM (its weights are up to a year
old), and **leaks nothing**: a frozen network was trained on data up to an earlier cutoff, and its
inputs at each origin are still data `≤ t`. It still forecasts every row the other models forecast.
ADR 0001 O2 proposed this fallback before any time was measured.

## 6. Where to look

- Code: `split`, `Scaler`, `train_member`, `predict`, `LstmLog`, `LstmRv` in
  [`src/ivf/models/lstm.py`](../../src/ivf/models/lstm.py); the runner and the member rows in
  [`src/ivf/evaluation/walkforward.py`](../../src/ivf/evaluation/walkforward.py).
- Tests: `tests/test_lstm.py` (the split by hand, the scaler on the training slice only, the poisoned
  validation-only values, early stopping, the back-transform and seed mean, determinism in and out of
  a worker, the truncation and poisoned-future tests, the refit schedule, the holdout, the command).
- Sources (all in the references of concept 00): Hochreiter & Schmidhuber (1997); Bucci (2020);
  Christensen, Siggaard & Veliyev (2023).

## Self-check

1. At cutoff c = 1,000 with H = 21 and 252 validation sessions, what are the first validation origin
   and the last training origin? Which target dates does training use?
2. Why is the scaler fitted on the training slice and not on training plus validation?
3. A member trained at cutoff c is applied, frozen, at the next fold's origins. Which data does a
   forecast at one of those origins read, and why is that not a leak?
4. Two seeds' forecasts differ by 3%. The LSTM beats HAR by 1% in RMSE. What do you conclude, and what
   would you check next?
