"""Milestone 4: an LSTM on log volatility, ``lstm_log``, on both tracks of the walk-forward runner.

One model id, two tracks, like ``rw``: on ``vix_level`` (VIX, h = 1, 5, 21; ``LstmLog``, called by
``walkforward.run_symbol``) and on ``rv21`` (h = 21; ``LstmRv``, called by ``walkforward.run_rv21``).
Both are trained and scored on the same folds and the same origins as every M2/M3 model.

**What the network sees at origin t (rule L1, trailing only).** A sequence of the ``window`` = 22
sessions ``t-21..t`` (exactly the window the runner's "forecastable" rule guarantees, so the LSTM can
forecast every row the other models do) and a few static values dated t that go straight to the
output layer beside the LSTM's last hidden state:

* ``vix_level``: the sequence is ``x_s = ln VIX_s``; the static values are HAR's regressors
  ``[x_t, mean(x_{t-4..t}), mean(x_{t-21..t})]`` (``har_log``'s design). VIX's own information set only,
  the same as ``har_log`` and ``arima_log``: the comparison is about the model, not the inputs
  (decision D3 of ``docs/milestones/04-lstm.md``).
* ``rv21``: the sequence is the daily return in percent, ``100 r_s``; the static values are
  ``[ln VIX_t, ln rv5_t, ln rv21_t]`` with ``rv_k`` the trailing realised vol over k sessions ending at t.

Because the head sees HAR's regressors directly, the network nests a linear HAR-type model; the LSTM
can only add to it.

**What it is trained on (rule L2, and the validation gap).** At cutoff c with longest horizon H
(21 on both tracks), an origin tau is usable only if every target is dated <= c: ``tau <= c - H``
(for ``rv21`` that is O-M2-7's ``tau <= c - 21``: the target window ``tau+1..tau+21`` ends by c). The
last ``val_sessions`` of those origins are the **validation** slice ``v0..c-H``; the **training**
slice is ``tau <= v0 - H``. So the last training target (dated ``v0``) is not after the first
validation origin, and no validation target was trained on (``split``). Early stopping keeps the
weights of the epoch with the lowest validation loss.

**Scaling.** Every input channel, static column and target column is standardised with the mean and
standard deviation of the **training slice only**, refitted at every training cutoff (``Scaler``).

**Output and back-transform (consistent with M2).** The network predicts the log change
``ln y_{t+h} - a_t`` (``a_t`` is ``x_t`` on ``vix_level`` and ``ln rv21_t`` on ``rv21``), trained on
squared error, so ``mu = a_t + prediction`` is a forecast of the mean of ``ln y_{t+h}``. ``s2`` is the
mean squared error of ``mu`` on the validation slice (out of sample for the weights, but the slice
also chose the epoch, so ``s2`` is optimistic by that selection). Then, exactly as ``har_log``:
``yhat = exp(mu + s2/2)`` (the lognormal mean, the scored forecast, ``var_method = plugin``) and
``yhat_naive = exp(mu)`` (the median).

**Seeds (ADR 0001 D5).** ``members`` networks per training cutoff, identical but for the seed (initial
weights and batch order). Member m at the training cutoff dated d gets
``SeedSequence(protocol.seed, spawn_key=(key32("lstm_log"), yyyymmdd(d), m))`` (``member_seed``), so a
seed depends on the model, the cutoff date and the member, never on a position in a list. The forecast's ``mu`` is the mean of the members' ``mu``; ``s2`` is the validation
error of that mean. Each member's own ``yhat`` is kept (``Forecast.members``) for the
``forecast_member`` table, which shows the seed spread.

**Determinism.** Every training and prediction runs under ``torch.set_num_threads(THREADS)`` (1),
``torch.use_deterministic_algorithms(True)`` and ``torch.manual_seed(seed)``, with batches shuffled by
a CPU ``torch.Generator`` seeded with the same seed. The same inputs give bitwise identical weights
(``weights_sha256`` in ``params``), in this process or in a worker process (``workers``), because every
member runs on one thread wherever it runs.

**Refit schedule (O-M4-1).** A full retrain needs ``refit_every`` sessions since the last training
cutoff; at the folds in between, the network trained at that cutoff (with its scaler and ``s2``) is
applied unchanged, ``params.refit = false``. It still forecasts every origin of every fold.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import torch

from ivf.features import build
from ivf.models import rv21
from ivf.models.baselines import Fit, Forecast, lognormal_forecast

#: Threads per training or prediction, in this process or in a worker (ADR 0002 decision 2).
THREADS = 1
DEVICE = "cpu"


@dataclass(frozen=True)
class LstmSettings:
    """The ``lstm`` block of ``configs/models.yaml``. Set before any LSTM forecast was scored (rule L6)."""

    members: int
    window: int
    hidden: int
    batch: int
    lr: float
    max_epochs: int
    patience: int
    val_sessions: int
    refit_every: int  # sessions since the last training cutoff before a full retrain

    def hyperparams(self) -> dict:
        return {**asdict(self), "torch": torch.__version__, "device": DEVICE, "threads": THREADS}


def key32(text: str) -> int:
    """ADR 0001 D5: the first 4 bytes of ``sha256(text)``, unsigned big-endian (never Python's salted ``hash``)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def member_seed(root_seed: int, model_id: str, cutoff: date, member: int) -> int:
    """The torch seed of one member: the first 32-bit word of ADR 0001 D5's ``SeedSequence`` for it."""
    sequence = np.random.SeedSequence(root_seed, spawn_key=(key32(model_id), int(cutoff.strftime("%Y%m%d")), member))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


# --- features and targets (trailing only) -----------------------------------------------------------


def lagged(x: np.ndarray, window: int) -> np.ndarray:
    """Row t holds ``x[t-window+1 .. t]`` (oldest first); NaN where the window starts before the array.

    Worked number: x = [1, 2, 3], window 2 gives [[nan, 1], [1, 2], [2, 3]].
    """
    x = np.asarray(x, dtype=float)
    padded = np.concatenate([np.full(window - 1, np.nan), x])
    return np.lib.stride_tricks.sliding_window_view(padded, window).copy()


def vix_features(y: np.ndarray, window: int):
    """``(seq (n, window, 1), static (n, 3), anchor (n,))`` for the ``vix_level`` track, from closes y."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(np.asarray(y, dtype=float))
    static = np.column_stack([x, build.trailing(x, build.HAR_WEEK, "mean"), build.trailing(x, build.HAR_MONTH, "mean")])
    return lagged(x, window)[:, :, None], static, x


def rv_features(inputs: rv21.Rv21Inputs, window: int):
    """``(seq, static, anchor)`` for the ``rv21`` track: percent returns; ``[ln VIX, ln rv5, ln rv21]``; ``ln rv21``."""
    r = np.asarray(inputs.r, dtype=float)
    rv5 = 100 * np.sqrt(build.SESSIONS_PER_YEAR * build.trailing(r**2, build.HAR_WEEK, "mean"))
    with np.errstate(divide="ignore", invalid="ignore"):
        anchor = np.log(rv21.target_series(r))
        static = np.column_stack([np.log(np.asarray(inputs.vix, dtype=float)), np.log(rv5), anchor])
    return (100 * lagged(r, window))[:, :, None], static, anchor


def log_targets(level: np.ndarray, anchor: np.ndarray, horizons) -> np.ndarray:
    """``D[t, j] = ln level[t + h_j] - anchor[t]``; NaN where ``t + h_j`` is past the array."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_level = np.log(np.asarray(level, dtype=float))
    n = len(log_level)
    out = np.full((n, len(horizons)), np.nan)
    for j, h in enumerate(horizons):
        out[: n - h, j] = log_level[h:] - anchor[: n - h]
    return out


def split(c: int, horizon: int, val_sessions: int) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation origins at cutoff c when targets reach ``horizon`` sessions ahead.

    Worked number: c = 100, horizon 21, val_sessions 10. The last usable origin is 100 - 21 = 79 (its
    target is dated 100 = c). Validation is 70..79; training is 0..49, since 49 + 21 = 70 is the first
    validation origin: the last training target is dated at v0, and the gap 70 - 49 = 21 >= every h.
    """
    tau_max = c - horizon
    v0 = tau_max - val_sessions + 1
    last_train = v0 - horizon
    if last_train < 0:
        raise ValueError(f"cutoff {c} leaves no training origin before the validation slice")
    return np.arange(0, last_train + 1), np.arange(v0, tau_max + 1)


@dataclass(frozen=True)
class Scaler:
    """Means and standard deviations of the training slice only (one per channel, static column, target)."""

    seq_mean: np.ndarray
    seq_sd: np.ndarray
    static_mean: np.ndarray
    static_sd: np.ndarray
    target_mean: np.ndarray
    target_sd: np.ndarray

    @classmethod
    def fit(cls, seq: np.ndarray, static: np.ndarray, target: np.ndarray) -> Scaler:
        flat = seq.reshape(-1, seq.shape[-1])
        parts = [(flat.mean(0), flat.std(0)), (static.mean(0), static.std(0)), (target.mean(0), target.std(0))]
        if any((sd <= 0).any() or not np.isfinite(sd).all() for _, sd in parts):
            raise ValueError("a training column is constant or not finite; it cannot be standardised")
        return cls(*[v for pair in parts for v in pair])

    def inputs(self, seq: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((seq - self.seq_mean) / self.seq_sd).astype(np.float32), ((static - self.static_mean) / self.static_sd).astype(np.float32)

    def to_json(self) -> dict:
        return {k: [float(v) for v in getattr(self, k)] for k in self.__dataclass_fields__}


# --- the network, one member's training, and prediction -------------------------------------------------


class Net(torch.nn.Module):
    """``LSTM(n_seq -> hidden)`` over the window; ``Linear(hidden + n_static -> n_out)`` on its last state and the statics."""

    def __init__(self, n_seq: int, n_static: int, hidden: int, n_out: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_seq, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden + n_static, n_out)

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return self.head(torch.cat([out[:, -1, :], static], dim=1))


@contextlib.contextmanager
def torch_state():
    """THREADS threads and deterministic algorithms for the block; the caller's settings are restored after."""
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def weights_sha256(state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name]).tobytes())
    return digest.hexdigest()


def _shape(arrays: dict, hidden: int) -> tuple[int, int, int, int]:
    return arrays["seq_tr"].shape[-1], arrays["static_tr"].shape[-1], hidden, arrays["target_tr"].shape[-1]


def train_member(arrays: dict[str, np.ndarray], seed: int, settings: LstmSettings) -> dict:
    """Train one member on standardised float32 arrays; keep the epoch with the lowest validation MSE.

    A top-level function so a worker process can run it. Returns the best weights as numpy arrays.
    """
    started = time.perf_counter()
    with torch_state():
        torch.manual_seed(seed)
        order = torch.Generator().manual_seed(seed)
        net = Net(*_shape(arrays, settings.hidden))
        opt = torch.optim.Adam(net.parameters(), lr=settings.lr)
        seq, static, target = (torch.from_numpy(arrays[k]) for k in ("seq_tr", "static_tr", "target_tr"))
        v_seq, v_static, v_target = (torch.from_numpy(arrays[k]) for k in ("seq_va", "static_va", "target_va"))
        n = len(target)
        best, best_epoch, bad, epochs = math.inf, 0, 0, 0
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        for epoch in range(1, settings.max_epochs + 1):
            net.train()
            perm = torch.randperm(n, generator=order)
            for i in range(0, n, settings.batch):
                idx = perm[i : i + settings.batch]
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(net(seq[idx], static[idx]), target[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(torch.nn.functional.mse_loss(net(v_seq, v_static), v_target))
            epochs = epoch
            if val < best:
                best, best_epoch, bad = val, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= settings.patience:
                    break
    if not math.isfinite(best):
        raise ValueError(f"seed {seed}: the validation loss was never finite")
    state = {k: v.numpy().copy() for k, v in best_state.items()}
    return {
        "seed": int(seed), "epochs_run": epochs, "best_epoch": best_epoch, "val_mse": best,
        "weights_sha256": weights_sha256(state), "seconds": time.perf_counter() - started, "state": state,
    }


def predict(state: dict[str, np.ndarray], shape: tuple[int, int, int, int], seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Standardised outputs ``(n, n_out)`` of one member, as float64, **one row at a time**.

    Row by row on purpose: in a batch, the matrix kernels' blocking depends on the batch size, which moved
    outputs by up to ~1e-8 log points (measured; ``docs/milestones/04-lstm.md``). One row per call makes a
    forecast at origin t a bitwise function of its own inputs, so the truncation test can demand equality.
    """
    with torch_state():
        net = Net(*shape)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        net.eval()
        seq_t, static_t = torch.from_numpy(seq), torch.from_numpy(static)
        with torch.no_grad():
            rows = [net(seq_t[i : i + 1], static_t[i : i + 1]) for i in range(len(seq_t))]
        return torch.cat(rows).numpy().astype(float) if rows else np.empty((0, shape[3]))


# --- the two models ----------------------------------------------------------------------------------------


class _Lstm:
    model_id = "lstm_log"
    fit_space = "log_level"
    var_method = "plugin"

    def __init__(self, settings: LstmSettings, root_seed: int, dates: list[date], workers: int = 0):
        self.settings = settings
        self.root_seed = root_seed  # protocol.seed
        self.dates = dates  # the run's sessions: a training cutoff's date keys its members' seeds
        self.workers = workers  # 0 or 1: train the members in this process; more: a spawn pool (same bytes)
        self._nets: dict[int, dict] = {}
        self._pool: ProcessPoolExecutor | None = None
        self.train_seconds = 0.0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _train_members(self, arrays: dict, c: int) -> list[dict]:
        seeds = [member_seed(self.root_seed, self.model_id, self.dates[c], m) for m in range(1, self.settings.members + 1)]
        if self.workers > 1:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(min(self.workers, len(seeds)), mp_context=multiprocessing.get_context("spawn"))
            return list(self._pool.map(train_member, [arrays] * len(seeds), seeds, [self.settings] * len(seeds)))
        return [train_member(arrays, seed, self.settings) for seed in seeds]

    def _frozen(self, previous: Fit | None, c: int) -> Fit | None:
        """The previous network, unchanged, while fewer than ``refit_every`` sessions have passed since it was trained."""
        if previous is None or c - previous.params["trained_cutoff"] >= self.settings.refit_every:
            return None
        return Fit(
            params={**previous.params, "refit": False}, hyperparams=previous.hyperparams, n_train=previous.n_train,
            train_first=previous.train_first, train_last_target=previous.train_last_target, converged=previous.converged,
            train_last_origin=previous.train_last_origin, members=previous.members,
        )

    def _fit(self, c: int, seq, static, anchor, target, horizon: int, extra: dict) -> Fit:
        s = self.settings
        train, val = split(c, horizon, s.val_sessions)

        def complete(idx):
            ok = np.isfinite(seq[idx]).all(axis=(1, 2)) & np.isfinite(static[idx]).all(axis=1) & np.isfinite(target[idx]).all(axis=1)
            return idx[ok]

        train, val = complete(train), complete(val)
        if len(train) < s.batch or len(val) < 2:
            raise ValueError(f"cutoff {c}: {len(train)} training and {len(val)} validation origins")
        scaler = Scaler.fit(seq[train], static[train], target[train])
        arrays = {}
        for name, idx in (("tr", train), ("va", val)):
            arrays[f"seq_{name}"], arrays[f"static_{name}"] = scaler.inputs(seq[idx], static[idx])
            arrays[f"target_{name}"] = ((target[idx] - scaler.target_mean) / scaler.target_sd).astype(np.float32)
        started = time.perf_counter()
        members = self._train_members(arrays, c)
        self.train_seconds += time.perf_counter() - started
        shape = _shape(arrays, s.hidden)
        # Validation errors of each member's mu and of the members' mean mu, in log points.
        pred = np.stack([predict(m["state"], shape, arrays["seq_va"], arrays["static_va"]) * scaler.target_sd + scaler.target_mean for m in members])
        s2_members = ((target[val][None] - pred) ** 2).mean(axis=1)
        s2 = ((target[val] - pred.mean(axis=0)) ** 2).mean(axis=0)
        key = len(self._nets)
        self._nets[key] = {"states": [m.pop("state") for m in members], "scaler": scaler, "shape": shape}
        lookback = max(s.window, build.HAR_MONTH)
        params = {
            **extra, "net": key, "trained_cutoff": int(c), "refit": True, "n_val": int(len(val)),
            "first_val_origin": int(val[0]), "last_train_origin": int(train[-1]),
            "s2": [float(v) for v in s2], "s2_members": [[float(v) for v in row] for row in s2_members],
            "scaler": scaler.to_json(), "members": [{k: v for k, v in m.items() if k != "seconds"} for m in members],
        }
        return Fit(
            params=params, hyperparams=s.hyperparams(), n_train=int(len(train)), train_first=max(0, int(train[0]) - lookback + 1),
            train_last_target=int(val[-1]) + horizon, train_last_origin=int(val[-1]),
            members=[{"seed": m["seed"], "epochs_run": m["epochs_run"], "seconds": m["seconds"]} for m in members],
        )

    def _mu(self, fit: Fit, seq, static, anchor, origins) -> np.ndarray:
        """Each member's ``mu``, shape ``(K, len(origins), n_out)``."""
        net = self._nets[fit.params["net"]]
        scaler = net["scaler"]
        x_seq, x_static = scaler.inputs(seq[origins], static[origins])
        out = np.stack([predict(state, net["shape"], x_seq, x_static) for state in net["states"]])
        return out * scaler.target_sd + scaler.target_mean + anchor[origins][None, :, None]

    @staticmethod
    def _forecast(mu_members: np.ndarray, j: int, fit: Fit) -> Forecast:
        s2 = fit.params["s2"][j]
        fc = lognormal_forecast(mu_members[:, :, j].mean(axis=0), np.full(mu_members.shape[1], s2))
        own = np.array([row[j] for row in fit.params["s2_members"]])[:, None]
        fc.members = np.exp(mu_members[:, :, j] + own / 2)
        return fc


class LstmLog(_Lstm):
    """``lstm_log`` on ``vix_level``: one network per training cutoff and seed with one output per horizon (direct)."""

    def fit(self, y: np.ndarray, horizons, previous: Fit | None) -> Fit:
        c = len(y) - 1
        frozen = self._frozen(previous, c)
        if frozen is not None:
            return frozen
        seq, static, anchor = vix_features(y, self.settings.window)
        target = log_targets(y, anchor, horizons)
        return self._fit(c, seq, static, anchor, target, max(horizons), {"horizons": list(horizons)})

    def forecast(self, fit: Fit, y: np.ndarray, origins: np.ndarray, horizons) -> dict[int, Forecast]:
        if list(horizons) != fit.params["horizons"]:
            raise ValueError("forecast horizons differ from the fitted ones")
        seq, static, anchor = vix_features(y, self.settings.window)
        mu = self._mu(fit, seq, static, anchor, origins)
        return {h: self._forecast(mu, j, fit) for j, h in enumerate(horizons)}


class LstmRv(_Lstm):
    """``lstm_log`` on ``rv21`` (h = 21): ``ln rv21_{t+21} - ln rv21_t`` from returns and VIX up to t."""

    def fit(self, inputs: rv21.Rv21Inputs, previous: Fit | None) -> Fit:
        c = len(inputs) - 1
        frozen = self._frozen(previous, c)
        if frozen is not None:
            return frozen
        seq, static, anchor = rv_features(inputs, self.settings.window)
        target = log_targets(rv21.target_series(inputs.r), anchor, (rv21.H,))
        return self._fit(c, seq, static, anchor, target, rv21.H, {"horizons": [rv21.H]})

    def forecast(self, fit: Fit, inputs: rv21.Rv21Inputs, origins: np.ndarray) -> Forecast:
        seq, static, anchor = rv_features(inputs, self.settings.window)
        return self._forecast(self._mu(fit, seq, static, anchor, origins), 0, fit)
