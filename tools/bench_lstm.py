"""Tiny-LSTM training benchmark sized like the implied-vol-forecasting problem.

Problem shape (from the project brief): ~5000 daily samples, look-back window of 22
trading days, a handful of features (log-IV, HAR lags, RV...), hidden 32-64, batch 64.
At the default settings that is 63 optimizer steps per epoch (4,000 training samples at
batch 64; the other 1,000 samples are validation) on a model with ~10-20k parameters.

Prints one JSON line per run so the driver can tabulate results.

Usage:
  uv run python tools/bench_lstm.py --device cpu --threads 10 --hidden 64 --epochs 6
  uv run python tools/bench_lstm.py --device cuda --hidden 64 --epochs 6 [--deterministic]
"""
import argparse
import hashlib
import json
import os
import time

t_import0 = time.perf_counter()
import numpy as np
import torch
t_import = time.perf_counter() - t_import0


def make_data(n: int, seq: int, feats: int, seed: int):
    """Synthetic AR(1) 'log-vol' series turned into (window -> next value) pairs.

    Content does not matter for timing; it just has to be learnable so the loss moves.
    """
    rng = np.random.default_rng(seed)
    T = n + seq + 1
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.97 * x[t - 1] + 0.1 * rng.standard_normal()
    base = np.stack([np.roll(x, k) for k in range(feats)], axis=1)  # lagged copies as "features"
    X = np.stack([base[i:i + seq] for i in range(n)]).astype(np.float32)
    y = x[seq:seq + n].astype(np.float32)[:, None]
    return torch.from_numpy(X), torch.from_numpy(y)


class Net(torch.nn.Module):
    def __init__(self, feats, hidden):
        super().__init__()
        self.lstm = torch.nn.LSTM(feats, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_one(device: str, hidden: int, epochs: int, seed: int, n: int = 5000, seq: int = 22,
              feats: int = 6, batch: int = 64, threads: int | None = None):
    if threads:
        torch.set_num_threads(threads)
    t_init0 = time.perf_counter()
    if device == "cuda":
        torch.zeros(1, device="cuda")  # force CUDA context creation
        torch.cuda.synchronize()
    t_init = time.perf_counter() - t_init0

    X, y = make_data(n, seq, feats, seed=123)
    n_val = n // 5
    Xtr, ytr = X[:-n_val].to(device), y[:-n_val].to(device)
    Xva, yva = X[-n_val:].to(device), y[-n_val:].to(device)

    torch.manual_seed(seed)
    model = Net(feats, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)  # shuffling on CPU: same order on every device
    epoch_times, val_losses = [], []
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(Xtr.shape[0], generator=gen).to(device)
        for i in range(0, Xtr.shape[0], batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(model(Xva), yva).item()  # .item() syncs the GPU
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        val_losses.append(vl)

    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return {
        "device": device, "threads": torch.get_num_threads(), "hidden": hidden, "epochs": epochs,
        "seed": seed, "steps_per_epoch": -(-Xtr.shape[0] // batch),
        "n_params": sum(p.numel() for p in model.parameters()),
        "import_s": round(t_import, 3), "cuda_init_s": round(t_init, 3),
        "first_epoch_s": round(epoch_times[0], 4),
        "steady_epoch_median_s": round(float(np.median(epoch_times[1:])), 4) if epochs > 1 else None,
        "total_train_s": round(sum(epoch_times), 3),
        "final_val_mse": val_losses[-1], "param_sha256": h.hexdigest()[:16],
        "deterministic_algos": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="train this many models in-process")
    a = ap.parse_args()
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        for r in range(a.repeat):
            print(json.dumps(train_one(a.device, a.hidden, a.epochs, a.seed + r, threads=a.threads)), flush=True)
    except Exception as e:  # report, don't hide, e.g. determinism errors
        print(json.dumps({"device": a.device, "error": f"{type(e).__name__}: {e}"[:400]}), flush=True)
