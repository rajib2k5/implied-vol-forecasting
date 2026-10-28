# ADR 0002: Compute and numeric stack

## Status

Proposed, 2026-10-27. It becomes Accepted when `pyproject.toml` and `uv.lock` land with
the pins below.

Every number in this ADR was measured on this machine on 2026-10-27, in throwaway uv
projects outside the repo. The commands and their output are under **Evidence**. Numbers
that are worked out from measurements rather than measured directly are labelled
*derived*.

## Context

**Machine.**
- aarch64 (Linux 7.0.0-1019-nvidia).
- 20 CPU cores of two kinds: 10 Cortex-X925 at 3.9 GHz and 10 Cortex-A725 at 2.8 GHz.
- NVIDIA GB10 GPU, compute capability 12.1 (Blackwell), 48 SMs, 130.7 GB of memory shared
  with the CPU.
- Driver 580.178.04. `nvidia-smi` reports "CUDA Version: 13.0".
- uv 0.11.25.
- The system Python's torch is CPU-only.

**Workload.** The project fits ARIMA, HAR and GARCH-family models (statsmodels, arch,
pmdarima) and one deep model, an LSTM. The LSTM problem is small:
- about 5,000 daily samples;
- a 22-day look-back window;
- a handful of features;
- hidden size 32 to 64, batch size 64.

That is roughly 63 optimizer steps per epoch on a model with 5k to 18k parameters. The
walk-forward design multiplies this into many separate small fits: folds × horizons ×
seeds.

**Open questions.**
1. Does a CUDA build of torch run on this aarch64 + Blackwell box at all?
2. If it does, is it worth having for this problem size?
3. Does the classical stack work on Python 3.12, numpy 2 and aarch64? pmdarima is the
   risky one.
4. How should versions be pinned so results can be reproduced?

## Decision

1. **Use CPU-only torch, `torch==2.14.1`, from the PyTorch CPU index.** Do not take it
   from PyPI's default index (see the surprise below). The code stays device-agnostic.
   Moving to the GPU only means swapping the index and sources blocks for the ones under
   Alternative A or B:
   - Variant A was installed on this box and passed every check.
   - Variant B was locked and installed, and reports CUDA as available.
2. **Set torch's thread count explicitly in the LSTM code. Never rely on the default.**
   Torch's default here is 20 threads, and across runs that was 6.7 to 20 times slower than
   8 threads (tables below).
3. **Keep pmdarima.** `pmdarima==2.1.1` works on py3.12, numpy 2.5.4, pandas 3.1.0,
   statsmodels 0.15.0 and aarch64, using a prebuilt wheel with no compiler step. The
   statsmodels AIC grid-search fallback was also run, and it picks the same orders.
   *(amended M0, review A)* "Works" does not mean "recovers the true order". On simulated
   series (3 seeds each, 1,000 points) `auto_arima` returned the generating order for
   AR(1), MA(1) and white noise every time. For AR(2) it never returned (2,0,0): it picked
   (1,0,2) twice and (1,0,3) once. On the one AR(2) series in Evidence §6 the statsmodels
   AIC grid also picked (1,0,2). Evidence §8 has the numbers.
4. **Pin every direct dependency exactly (`==`) and commit `uv.lock`.** The lock pins
   136 third-party packages (137 entries, counting the project itself), each with its
   exact wheel URL and sha256.
5. **Use Python 3.12 managed by uv.** uv downloaded cpython-3.12.13 for this check.

Ready-to-paste block, tested as a set. It locked, installed, and passed the smoke suite
plus pytest:

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy==2.5.4",
  "pandas==3.1.0",
  "scipy==1.18.1",
  "statsmodels==0.15.0",
  "arch==8.0.0",
  "pmdarima==2.1.1",
  "torch==2.14.1",
  "pyarrow==26.0.0",
  "yfinance==1.7.0",
  "requests==2.34.2",
  "matplotlib==3.11.2",
  "pytest==9.2.0",
  "nbformat==5.11.1",
  "nbconvert==7.17.2",
  "ipykernel==7.4.0",
  "jupyterlab==4.6.4",
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Notes on the block:
- **`explicit = true`.** uv only uses the PyTorch index for packages that name it in
  `[tool.uv.sources]`, which here is just torch. Everything else, numpy included, comes
  from PyPI. Without this flag the PyTorch index (which also hosts copies of numpy and
  others) could supply packages it was never meant to.
- **The torch pin has no `+cpu` suffix.** Under PEP 440, `==2.14.1` matches
  `2.14.1+cpu`, `2.14.1+cu130` and the rest. The **index** picks the build, and `uv.lock`
  records which build was picked. The CPU lock has two torch entries:
  - `2.14.1` for macOS (`macosx_14_0_arm64`);
  - `2.14.1+cpu` for Linux (x86_64, aarch64) and Windows.
- **`requires-python`.** `">=3.12,<3.13"` and `"==3.12.*"` produced the same 136-package
  set (checked by re-locking).
- **Pins that are not listed directly.** scikit-learn 1.9.1, joblib 1.6.0, patsy 1.0.3,
  Cython 3.3.0 and setuptools 84.2.1 come in through pmdarima and statsmodels and are
  pinned by the lock. If `ivf` imports scikit-learn itself (for a scaler, say), add it to
  the list. Do not rely on a transitive dependency.

## Why: the explanations to give in an interview

### Why the GPU barely matters for a daily-frequency LSTM

A GPU is fast when there is a lot of arithmetic per kernel launch. One optimizer step of
this LSTM involves about 0.15 GFLOP. That figure is *derived*: about 2 × 4h(F+h) FLOPs per
sample per time step, times 22 steps, times a batch of 64, times about 3 for forward plus
backward. A modern GPU does that much arithmetic in microseconds. What it cannot avoid is a
fixed cost for each kernel launch and each sync, paid again for every small kernel in the
step. The kernel count was not measured. *Derived:* the measured 0.10 s epoch divided by
63 steps is about 1.6 ms per step, which is far more than the arithmetic needs.

So the GPU is **latency-bound** at this size. The measurement shows it clearly: GPU epoch
time stays flat at about 0.08 to 0.13 s from 5k to 1M parameters. CPU epoch time grows
with the model:

| hidden | params | GPU s/epoch | CPU (8 threads) s/epoch | GPU speed-up |
|---:|---:|---:|---:|---:|
| 32 | 5,153 | 0.081 | 0.079 | 1.0x (parity) |
| 64 | 18,497 | 0.114 | 0.145 | 1.3x |
| 128 | 69,761 | 0.094 | 0.350 | 3.7x |
| 256 | 270,593 | 0.128 | 0.978 | 7.6x |
| 512 | 1,065,473 | 0.112 | 2.790 | 25x |

Our models sit at hidden 32 to 64, right at the crossover. A walk-forward study is many
small fits, so what counts is throughput. Here are 40 fits at hidden 64:

| how the box is used | wall time | per model |
|---|---:|---:|
| GPU, 1 process, fits one after another (the "natural" loop) | 50.7 s | 1.27 s |
| CPU, pool of 20 processes × 1 thread | 33.0 s | 0.83 s |
| GPU, 5 processes in parallel | 14.7 s | 0.37 s |

So the GPU's best case is about **2.0 to 2.3 times** the CPU's best case: 2.0x in the
20-model run, 2.3x in the 40-model run. It gets there only when several GPU processes run
at once. A plain sequential GPU loop was *slower* than a CPU process pool.

*Derived:* a plausible final run is 24 refits × 3 horizons × 5 seeds × 50 epochs, about
18,000 model-epochs. At the measured rates that takes about 25 min on a CPU pool and about
11 min on a GPU pool. Neither number changes what the project can do.

Two other effects are as large as, or larger than, the choice of device:
- **Thread count.** Torch's default of 20 threads was 6.7 to 20 times slower than 8
  threads.
- **Parallelism.** Running fits in parallel instead of one after another gave 1.7x on the
  CPU (20 models: 1×8 threads vs 20×1) and 2.6 to 3.6x on the GPU.

**The interview answer:** "At daily frequency the network is so small that a GPU step is
dominated by launch overhead, not arithmetic. I measured parity at hidden 32 and at most
2.3x at hidden 64, and only with parallel GPU processes. The CPU thread setting was a
lever at least as big as the device. The GPU would pay off from about hidden 128 upward,
where I measured 3.7x and more."

### What cu13x vs cu12x means for a Blackwell GPU

- **`nvidia-smi`'s "CUDA Version: 13.0"** is the newest CUDA runtime the *driver*
  supports natively. It is not an installed toolkit.
- **PyTorch wheels bring their own CUDA runtime.** With `+cu130` torch, `libcudart.so.13`,
  `libcublas.so.13` and `libcudnn.so.9` were loaded from `site-packages/nvidia/...`. Only
  `libcuda.so.580.178.04` (the driver) came from the system, even though a system CUDA
  13.0 toolkit exists in `/usr/local/cuda`.
- **The suffix `cuXYZ`** is the CUDA version the wheel was built with. It decides two
  things:
  - which GPU architectures have compiled kernels ("cubins") in the wheel;
  - which drivers can run it.
- **GB10 is compute capability 12.1 (`sm_121`).** No wheel ships `sm_121` kernels. The
  CUDA 12.8+ builds ship `sm_120`. A cubin built for 12.0 runs on 12.1 (same major
  version, higher minor). PyTorch's own warning text states this rule, for example "9.0
  which supports hardware CC >=9.0,<10.0".

Measured on this box:

| index | torch | CUDA | cuDNN | arch list | outcome on GB10 |
|---|---|---|---|---|---|
| cu126 | 2.14.1 | 12.6 | 9.10 | sm_80, sm_90 | **fails**: `CUDA error: no kernel image is available for execution on the device` |
| cu128 | 2.11.0 (newest on that index) | 12.8 | 9.19 | sm_80, 90, 100, 120 | all GPU ops OK |
| cu130 | 2.14.1 | 13.0 | 9.24 | sm_80, 90, 100, 110, 120 | all GPU ops OK |
| cu132 | 2.14.1 | 13.2 | 9.24 | sm_80, 90, 100, 110, 120 | all GPU ops OK. CUDA 13.2 is above the driver's 13.0 and runs through minor-version compatibility. |

The cu126 failure is the lesson. **`torch.cuda.is_available()` returned True,
host-to-GPU copies worked, and even a matmul worked.** cuBLAS is NVIDIA's own library and
ships kernels for newer GPUs. But `x + 1` and `torch.zeros(..., device="cuda")`, which are
PyTorch's *own* kernels, failed. The only real check is to run a real model forward and
backward on the device.

**Rule:**
- On Blackwell, use a cu128-or-newer build.
- Prefer the build whose CUDA version matches what the driver reports (cu130 here). That
  avoids relying on minor-version compatibility.
- cu12x builds also carry a larger footprint (cu128: 3.6 GB of wheels) and, on that index,
  an older torch.

### Why exact `==` pins plus `uv.lock`

- **A version string does not identify a binary.**
  - On this box, `torch==2.14.1` resolved to five different aarch64 wheel files across
    four build variants: `+cpu`, `+cu126`, `+cu130`, `+cu132`, and PyPI's own `+cu130`.
    The PyPI file has a different sha256 from the cu130-index file.
  - The build changes the numbers. Same torch version, same CPU, same seed, same 1
    thread: the `+cpu` and `+cu130` builds trained to different weights (param hash
    `43e08e03…` vs `ae7f83fe…`).
  - `uv.lock` records the exact wheel URL and sha256 for every one of the 136 packages,
    so `uv sync --frozen` rebuilds the same environment.
- **`==` in `pyproject.toml` makes upgrades deliberate and readable.** An unpinned resolve
  today gives **pandas 3.1.0**. The user's previous repo (`kaggle-optiver-close`) was
  pinned to pandas 2.2.3. pandas 3 changes defaults; for example, strings now come back
  with dtype `str`, as the parquet round-trip showed. Several deprecations in this stack
  are already counting down:
  - statsmodels 0.15 warns that `adfuller` and `kpss` change their return type "in release
    0.16 or after July 2027";
  - yfinance 1.7.0 raises `Pandas4Warning`s about `inplace=`.

  Pins mean these changes land when we choose, in a commit we can bisect.
- **The two jobs are different.** The lock gives reproducibility (the same bytes). The
  `==` pins give intent: a reviewer can see the direct dependencies, and
  `uv lock --upgrade` cannot quietly move them.

## Alternatives considered (rejected)

**A. CUDA torch (cu130) everywhere.** This is the same block with
`url = ".../whl/cu130"` and `index = "pytorch-cu130"`. It was locked, installed, and
passed every check (14/14 smoke checks, pmdarima, GPU probe). It was rejected for four
reasons:
- **The gain is small at this size.** 1.0x at hidden 32, 1.3 to 1.5x single-process at
  hidden 64, and 2.0 to 2.3x only with 4 or more concurrent GPU processes.
- **The cu130 index has no macOS wheel for torch.** The lock lists only manylinux aarch64,
  manylinux x86_64 and win_amd64, so `uv sync` on a reader's Mac would fail. This is a
  public repo.
- **It is about 8 times heavier.** The download is about 3.4 GB against about 0.4 GB, and
  the venv is 5.9 GB against 1.5 GB.
- **It ties the project to the NVIDIA driver version.** That is one more thing a learner
  has to debug.

*What would make it win:*
- hidden 128 or more, stacked LSTMs or attention models (GPU measured 3.7x at hidden 128
  and 7.6x at 256);
- or an LSTM search large enough that the CPU pool takes more than a few hours;
- or intraday data.

**B. CUDA on Linux, CPU elsewhere (platform-split sources).** Verified: it locks, gives
`2.14.1+cu130` with CUDA available on this box, and the lock holds macOS and Windows CPU
wheels. To switch, replace the index and sources blocks with:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux'" },
  { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },
]
```

Rejected for now. Every Linux machine without a GPU (CI runners, readers) would still
download the 3.4 GB CUDA set, and the measured gain does not change any project outcome.
This is the variant to adopt if the conditions under A become true.

**C. Torch from PyPI's default index, with no index configuration.** This is the
surprise. On Linux aarch64, PyPI's `torch==2.14.1` is **the CUDA 13.0 build**:
- once installed it reports `2.14.1+cu130`;
- it passed the GPU probe;
- it pulls in 3,163 MB in total.

Its PyPI metadata adds `cuda-toolkit==13.0.3` and the `nvidia-*-cu13` wheels under
`platform_system == "Linux"`, so this applies to every Linux platform. Only aarch64 was
installed. "Take the default CPU wheel" therefore installs CUDA. Rejected because
the CUDA variant would be implicit, and the default could move to a CUDA version newer
than the driver at the next torch bump.

**D. cu128, cu132 or cu126 instead of cu130.**
- cu126 fails on this GPU (see the table above).
- cu128 works, but caps torch at 2.11.0 and is heavier (542.6 MB torch wheel, 3.6 GB in
  total).
- cu132 works only through minor-version compatibility, with no measured benefit.
- cu129 (newest torch 2.13.0) was not tested.

If the GPU is adopted, use cu130.

**E. Drop pmdarima and use only the statsmodels AIC grid.** Not needed now: pmdarima
works and picked the same orders as the grid on all three test processes. Keep the grid
search as the documented fallback. pmdarima's latest release (2.1.1, uploaded 2025-11-17)
is older than the statsmodels 0.15.0 it calls internally (uploaded 2026-08-27). That makes
it the dependency most exposed to a statsmodels change.

**F. pandas 2.x.** Not tested. pandas 3.1.0 passed every check, and a new codebase has no
pandas-2 code to protect. It would win only if a dependency broke under pandas 3.

## Consequences

**LSTM code (milestone 4):**
- Accept a `device` argument that defaults to `"cpu"`. No module-level `.cuda()` calls.
- Call `torch.set_num_threads(n)` explicitly and record `n` in the forecast metadata.
  Measured settings that work:
  - a single fit: 8 threads (fastest of 1/2/4/6/8/10/20);
  - many fits: a process pool of 20 workers × 1 thread or 10 × 2 (0.83 s and 0.88 s per
    model for 40 fits);
  - never 20 threads per process.
- Seed `torch.manual_seed` and the numpy Generator, and do the shuffling with a seeded
  CPU `torch.Generator`. Call `torch.use_deterministic_algorithms(True)` as a guard: it
  raises if a future op has no deterministic implementation. Its overhead could not be
  told apart from run-to-run noise here.
- Record `torch.__version__` (which includes the build, e.g. `2.14.1+cpu`), the device
  and the thread count with every LSTM forecast.

**Reproducibility:**
- What it means here: for the benchmark model, the same lock, the same machine and the
  same thread count give **bitwise identical** weights. Measured: 3 fresh processes per
  configuration gave the same hash every time, on both CPU and GPU.
- Changing the thread count changes the low-order bits:
  - 1 and 2 threads agree, 4 differs, 6 and 8 agree;
  - CPU and GPU final validation MSE differ by about 4e-7 (0.0097592 vs 0.0097588).
- So tests compare LSTM outputs with tolerances, and conclusions are checked across
  several seeds, not by matching hashes across machines.

**GARCH code (milestone 3):**
- `arch` EGARCH has **no analytic forecast beyond 1 step**: `ValueError: Analytic
  forecasts not available for horizon > 1`. 21-day EGARCH variance needs
  `method="simulation"` or `method="bootstrap"`.
- With `method="simulation"`, **`random_state=` is ignored**. Two calls with
  `RandomState(0)` gave 0.999475 and 0.999363. Pass `rng=` with a seeded Generator
  instead; that gave identical results. `method="bootstrap"` does respect `random_state`.
- *(amended M0, review A)* **Always pass `method` in lower case.** arch 8.0.0 accepts
  `method="Simulation"` (and `"SIMULATION"`) without complaint and then runs the
  **bootstrap**, not the simulation. `forecast` checks `method.lower()` against the three
  known names, but chooses the code path by comparing the raw string with `"simulation"`
  (`arch/univariate/volatility.py`, lines 755 to 766 in 8.0.0), so a capital letter falls
  through to the bootstrap branch. Nothing is raised, and `rng=` is then ignored. Measured
  (Evidence §8): `"Simulation"` with `random_state=RandomState(0)` gave exactly the
  bootstrap's number, and two forecasts with a freshly seeded `rng=` did not repeat. A test
  that two such forecasts agree to the last digit therefore catches a capitalised spelling.

**ARIMA code (milestone 2)** *(amended M0, review A)*:
- `n_periods` must be a Python `int`. `model.predict(n_periods=np.int64(5))` raises
  `TypeError: n_periods must be an int`. A horizon that comes out of a numpy array or a
  pandas column has to go through `int(h)` first.
- Do not write a test that expects `auto_arima` to find the generating order of an AR(2)
  or higher process (see Decision 3). Test its search on AR(1), MA(1) and white noise, and
  test everything else with a fixed `order=`.

**Footprint:**
- About 0.4 GB download (243 MB of packages plus the 159 MB torch wheel) and a 1.5 GB
  venv.
- CI on x86_64 Linux would get the 196 MB `+cpu` x86_64 wheel. That comes from the lock;
  CI itself was not run.

**Pins to revisit deliberately:**
- statsmodels 0.16 (the `adfuller`/`kpss` return type changes);
- pandas 4 (yfinance's `inplace=` use);
- any pmdarima release.

**Relation to ADR 0001** (its rows "Python", "Seeds" and "Fixture platform"):
- **Python row.** Consistent: `==` pins, a committed lock, and CI running
  `uv sync --locked`. With CPU torch, a CI run downloads about 0.4 GB rather than about
  3.4 GB.
- **"Real LSTM runs may use the GPU".** Under this ADR that holds only if Alternative A or
  B is adopted. `fit.device` stays useful either way.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`.** Harmless, and irrelevant on CPU. torch
  2.14.1+cu130 did not need it (Evidence §5).
- **The fixture's `OMP_NUM_THREADS=1`.** Consistent with the finding that thread count
  changes low-order bits.
- **The open M3 item "EGARCH multi-step needs simulation, which must be seeded".** Answered
  here: analytic forecasts beyond one step raise, and the simulation must be seeded with
  `rng=`, not `random_state=`.

**Known harmless warnings:**
- ipykernel 7.4 prints "Kernel is running over TCP without encryption" when nbconvert
  executes a notebook.
- statsmodels may emit `ConvergenceWarning` on some ARIMA grid points.

**Update (2026-10-28, *derived*, not measured): refit count.**
- The run estimate under "Why the GPU barely matters" (18,000 model-epochs, about 25 min
  on a CPU pool and about 11 min on a GPU pool) assumed 24 refits. ADR 0001 4.5 refits every
  63 sessions from 2000, which gives 97 dev folds plus 10 holdout folds, about 107 refits.
- Scaling the same estimate linearly: 107 × 3 horizons × 5 seeds × 50 epochs = 80,250
  model-epochs, 4.46 times as many. That is about 111 min on a CPU pool (4.46 × 25) and
  about 49 min on a GPU pool (4.46 × 11). It ignores that the training set grows with each
  fold (slower) and that early stopping ends fits before 50 epochs (faster).
- The decision does not change: CPU torch stays. The CPU figure is below the "more than a
  few hours" trigger under Alternative A but not far below it, so milestone 4 should
  measure real fit times before relying on it.

## Evidence

Throwaway uv projects with Python 3.12.13 (uv-managed). The other probe scripts were in the
session scratchpad and are not kept. The benchmark script is kept as `tools/bench_lstm.py`;
the CPU timings can be re-run with
`uv run python tools/bench_lstm.py --device cpu --threads 8 --hidden 32 --epochs 6`
(the GPU rows need the CUDA torch variants of Alternative A or B, not the locked CPU build).
The benchmark is described here so it can be re-created:

- **Data.** A synthetic AR(1) series with φ = 0.97, turned into 22-step windows with 6
  lagged features: 5,000 samples, of which 4,000 train (63 steps per epoch at batch 64)
  and 1,000 validation.
- **Model.** `nn.LSTM(6, h, batch_first=True)` followed by a `Linear(h, 1)` on the last
  step. Adam with lr 1e-3, MSE loss. A validation pass every epoch.
- **Data placement.** Data is moved to the device once, before training starts.
- **Timing.** `perf_counter` with `torch.cuda.synchronize()` before each read.
- **Reported figures.** "Steady epoch" is the median over epochs 2 to N.
- **Noise.** The machine was shared with other sessions. The load average ran from 1.3 to
  23 during the runs, partly from these benchmarks themselves.
  - The 20-thread rows are noisy: 0.65 to 2.9 s/epoch across sessions and hidden sizes.
  - CPU rows with 1 to 8 threads repeated within about 10%.
  - GPU rows spread more, for example 0.079 to 0.120 s/epoch at hidden 32.

### 1. Which torch wheels exist for aarch64 cp312

```
$ curl -s https://download.pytorch.org/whl/<idx>/torch/ | grep -o 'torch-...cp312...aarch64.whl' | sort -V
cu130: ... 2.13.0, 2.14.0, 2.14.1     cu129: ... newest 2.13.0     cu128: ... newest 2.11.0
cu132: 2.12.0 ... 2.14.1             cu126: ... 2.14.1            cpu: ... 2.14.1
cu134: (no aarch64 cp312 wheels)
$ curl -s https://pypi.org/pypi/torch/2.14.1/json   # Linux requires_dist:
cuda-toolkit[cublas,cudart,cufft,...]==13.0.3; platform_system == "Linux"
nvidia-cudnn-cu13==9.24.0.43; platform_system == "Linux"      (aarch64 wheel: 454.0 MB)
```

### 2. Lock, install and probe each build

`pyproject.toml` contained `[[tool.uv.index]] name="pytorch-<idx>" url=".../whl/<idx>"
explicit=true` and `[tool.uv.sources] torch = { index = "pytorch-<idx>" }`. Then:

```
$ uv lock --managed-python && uv sync --frozen --managed-python
cu130: Resolved 31 packages; torch==2.14.1+cu130
$ (sum of wheel sizes in uv.lock) + curl -sIL <torch wheel> | grep content-length
cu130: torch 454,001,391 B + 2709.2 MB deps (cudnn 651.0, cublas 542.8, triton 226.5, ...) ; .venv 5.1G
cu128: torch 542,576,164 B + 3081.9 MB deps ; .venv 5.8G
cpu:   torch 159,242,183 B + 26.1 MB deps   ; .venv 700M
pypi (no index config): 3163.2 MB, 30 packages ; .venv 5.1G ; torch reports 2.14.1+cu130
```

GPU probe output: real ops, compared against the CPU.

```
cu130  torch 2.14.1+cu130  torch.version.cuda 13.0  cudnn 92400  cuda.is_available True
       device_name NVIDIA GB10  capability (12, 1)
       arch_list ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']  total_memory_GB 130.7 SMs 48
       matmul_f64 max|gpu-cpu| 1.85e-13   matmul_f32 max|gpu-cpu| 9.9e-05 (allow_tf32 False)
       lstm loss |gpu-cpu| 2.6e-06   lstm grad max|gpu-cpu| 8.5e-07   ALL_GPU_OPS_OK True
cu132  torch 2.14.1+cu132  cuda 13.2  same arch list   ALL_GPU_OPS_OK True
cu128  torch 2.11.0+cu128  cuda 12.8  cudnn 91900  arch ['sm_80','sm_90','sm_100','sm_120']  ALL_GPU_OPS_OK True
pypi   torch 2.14.1+cu130  (same as cu130)  ALL_GPU_OPS_OK True
cu126  torch 2.14.1+cu126  cuda 12.6  arch ['sm_80','sm_90']
       UserWarning: NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation.
       matmul OK;  a.cuda()+1 -> AcceleratorError: CUDA error: no kernel image is available for execution on the device
       torch.zeros(3, device='cuda') -> same error ;  LSTM FAILED ;  ALL_GPU_OPS_OK False
```

Where the CUDA libraries load from (cu130 env, read from `/proc/self/maps` after an LSTM
forward):

```
.venv/.../site-packages/nvidia/cu13/lib/libcudart.so.13
.venv/.../site-packages/nvidia/cu13/lib/libcublas.so.13
.venv/.../site-packages/nvidia/cudnn/lib/libcudnn.so.9
/usr/lib/aarch64-linux-gnu/libcuda.so.580.178.04
```

### 3. Single-fit timing

10 epochs, 3 fresh processes per row; median steady-epoch seconds, with [min, max].

cu130 build (GPU, and CPU through the same wheel):

| config | hidden 32 | hidden 64 |
|---|---|---|
| GPU | 0.113 [0.079, 0.120] | 0.103 [0.093, 0.108] |
| CPU 1 thread | 0.286 | 0.693 |
| CPU 2 | 0.156 | 0.365 |
| CPU 4 | 0.107 | 0.237 |
| CPU 6 | 0.144 | 0.210 |
| **CPU 8** | **0.080** | **0.152** |
| CPU 10 | 0.103 | 0.278 |
| CPU 20 (torch default) | 0.754 | 1.144 |

`+cpu` build, same matrix: hidden 32 / hidden 64 at 8 threads = 0.076 / 0.166, at 20
threads = 0.688 / 1.106. That is the same performance profile as the CUDA wheel's CPU
path.

Other single-fit costs:
- **GPU fixed costs:** `import torch` about 1.6 s; CUDA context creation 0.28 s; first
  epoch 0.32 to 0.35 s against about 0.10 s steady (kernel and cuDNN warm-up).
- **Thread anomaly, not explained:** 10 threads pinned with `taskset` to the ten fast X925
  cores ran at 1.15 to 1.38 s/epoch. That is slower than 10 threads unpinned (0.30) and
  than 10 threads pinned to the slow A725 cores (0.48 to 0.53). So a simple "slow cores
  are stragglers" explanation does not fit. The consistent pattern is that fine-grained
  parallelism on tiny tensors degrades sharply once the thread count approaches the
  number of cores on a shared machine. Single-thread speed: X925 0.68 s/epoch, A725 1.38.

The hidden-size scaling table in the "Why" section came from 6 epochs × 3 runs per row in
the cu130 env.

### 4. Throughput: many fits

Hidden 64, 10 epochs per model. End-to-end wall time, including each process's Python
start, import and CUDA initialisation.

A dash means the plan was not run at that size.

| plan | 20 models | 40 models |
|---|---:|---:|
| GPU 1 process | 26.8 s | 50.7 s |
| GPU 2 processes | 15.2 s | – |
| GPU 4 processes | 10.7 s (10.9 on repeat) | – |
| GPU 5 processes | 10.2 s | 14.7 s |
| GPU 8 processes | – | 14.1 s |
| GPU 10 processes | 11.0 s | – |
| CPU 1 process × 8 threads | 34.4 s | – |
| CPU 2 processes × 4 threads | 28.9 s | – |
| CPU 4 processes × 4 threads | – | 51.8 s |
| CPU 5 processes × 2 threads | 22.0 s | 42.8 s |
| CPU 10 processes × 1 thread | 26.9 s | 48.4 s |
| CPU 10 processes × 2 threads | – | 35.2 s |
| CPU 20 processes × 1 thread | 20.4 s (21.0 on repeat) | 33.0 s |

At hidden 32 with 20 models: GPU 4 processes 10.7 s, CPU 5 × 2 12.2 s, CPU 20 × 1 13.5 s.

### 5. Determinism

hidden 64, 10 epochs; SHA-256 of the trained parameters (first 16 hex characters):

```
GPU default flags, run 1 / run 2                        41fd8dd1cddeadcf / 41fd8dd1cddeadcf
GPU use_deterministic_algorithms(True)+cudnn.deterministic,
    without CUBLAS_WORKSPACE_CONFIG                      41fd8dd1cddeadcf   (no error raised)
    with CUBLAS_WORKSPACE_CONFIG=:4096:8 (x2)            41fd8dd1cddeadcf
CPU 8 threads default / deterministic                    2aa9880a24b2614b / 2aa9880a24b2614b
CPU 1 / 2 / 4 threads (cu130 wheel)                      ae7f83fe0726ee8b / ae7f83fe0726ee8b / ac04936fe50d71c6
CPU 1 thread, +cpu wheel                                 43e08e03a940dd2c
final val MSE: CPU 0.009759165, GPU 0.009758761
```

Under `use_deterministic_algorithms(True)` with no `CUBLAS_WORKSPACE_CONFIG`, `mm`, `bmm`,
`index_add_` and `cumsum` all ran on CUDA without raising in torch 2.14.1+cu130. PyTorch's
documentation for earlier releases says cuBLAS ops raised without that variable; no older
build was tested here.

### 6. Numeric stack: real calls, simulated data with known truth

The unpinned resolve picked the versions in the block above, and the pinned set re-locked
to the identical versions. The checks ran in three environments: unpinned, pinned with
`+cpu` torch, and pinned with `+cu130` torch. Result in each: `SUMMARY {'PASS': 14,
'FAIL': 0}`.

```
numpy linalg.solve residual 2.53e-14
pandas rolling + pyarrow parquet round-trip: string column dtype 'str' (pandas 3 default)
scipy brentq(v^2-2) 1.414213562373, minimize -> [3, -1]
statsmodels ARIMA(1,0,0) on AR(1) phi=0.7: phi_hat 0.7004; ADF p 4e-29; KPSS p 0.100
statsmodels OLS with HAC: params [0.1001 0.399 0.3006 0.1993] (true .1 .4 .3 .2)
arch GARCH(1,1) on simulated (omega .05, alpha .08, beta .90): .0496 / .0777 / .9024;
    21-step analytic variance h1 1.706 -> h21 1.966 (unconditional 2.5)
arch GJR (o=1): gamma .0027 (true 0)
arch EGARCH(1,1,1): fits; analytic h=21 raises ValueError; method="simulation" works
pmdarima auto_arima on AR(1): order (1, 0, 0); ndiffs(random walk) = 1
matplotlib Agg savefig OK; nbformat + nbconvert executed a cell via ipykernel ('2.5.4 3.1.0 2')
requests HEAD cdn.cboe.com VIX_History.csv: [307] -> 200 at cdn-api.cboe.com
yfinance SPY 5d: 5 rows, MultiIndex columns
pytest -q on a 2-test probe: 2 passed;  jupyter lab --version: 4.6.4
```

pmdarima against the fallback, 1,000 simulated points each:

```
AR(1) phi=.7    auto_arima stepwise (1,0,0) | auto_arima full grid (1,0,0) | statsmodels AIC grid (1,0,0)
AR(2) .5,.3     auto_arima stepwise (1,0,2) | auto_arima full grid (1,0,2) | statsmodels AIC grid (1,0,2)
MA(1) theta=.6  auto_arima stepwise (0,0,1) | auto_arima full grid (0,0,1) | statsmodels AIC grid (0,0,1)
pm.arima.ARIMA((1,0,0)).fit(500 obs).update(20 obs).predict(5, return_conf_int=True): OK, nobs 520
```

For the same order, AICs differed by up to about 2.7 between pmdarima stepwise, pmdarima
full grid and statsmodels. Examples:
- MA(1): 2858.43 vs 2860.24;
- AR(1): pmdarima full grid 2724.88 vs stepwise 2722.15.

Different intercept or estimation choices are the likely reason; this was not isolated.
Compare AICs only within one fitting routine.

### 7. Lock comparisons

- The pinned `+cpu` and `+cu130` locks have identical versions for every package that is
  not torch, NVIDIA, CUDA or triton.
- Packages with no macOS wheel: CPU lock, only `pywinpty` (Windows-only). The cu130 lock
  has no macOS wheel for `torch`, nor for the Linux-only `nvidia-*`, `cuda-bindings` and
  `triton` packages.

### 8. Amendment probes (M0, review A, measured 2026-10-28)

Both ran in this repo's locked environment (`uv run python <script>`): arch 8.0.0, pmdarima
2.1.1, numpy 2.5.4, pandas 3.1.0, statsmodels 0.15.0. The scripts were scratchpad files and
are not kept; the arch one is short enough to give in full.

```python
import numpy as np
from arch import arch_model
y = np.random.default_rng(0).standard_normal(2000)      # fixed data
res = arch_model(y, mean="Constant", vol="EGARCH", p=1, o=1, q=1).fit(disp="off")
def h21(method, seed=0):
    f = res.forecast(horizon=21, method=method, simulations=500,
                     random_state=np.random.RandomState(seed), reindex=False)
    return f.variance.to_numpy()[-1, -1]
for m in ("simulation", "Simulation", "SIMULATION", "bootstrap"):
    a, b = h21(m), h21(m); print(f"method={m!r} RandomState(0) twice -> {a:.6f} {b:.6f} identical={a == b}")
print("Simulation == bootstrap (same seed):", h21("Simulation") == h21("bootstrap"))
# and with rng=: g = default_rng(0); res.forecast(..., method=m, rng=lambda size: g.standard_normal(size))
```

```
method='simulation'   RandomState(0) twice -> 1.017233 1.017547  identical=False
method='Simulation'   RandomState(0) twice -> 1.017399 1.017399  identical=True
method='SIMULATION'   RandomState(0) twice -> 1.017399 1.017399  identical=True
method='bootstrap'    RandomState(0) twice -> 1.017399 1.017399  identical=True
Simulation == bootstrap (same seed): True
simulation == bootstrap (same seed): False
method='simulation'   fresh seeded rng= twice -> 1.017799 1.017799  identical=True
method='Simulation'   fresh seeded rng= twice -> 1.016106 1.017429  identical=False
```

The first line of each pair is the lower-case simulation, which ignores `random_state=` (as
Consequences says). The capitalised spellings repeat under `random_state=` and equal the
bootstrap exactly, which is what the bootstrap does and the simulation does not.

pmdarima `auto_arima(stepwise=True, seasonal=False, information_criterion="aic")` on
simulated ARMA data, 1,000 points, seeds 0, 1 and 2, noise from `default_rng(seed)`:

```
AR(1) phi=.7    (1, 0, 0)  (1, 0, 0)  (1, 0, 0)
AR(2) .5,.3     (1, 0, 2)  (1, 0, 3)  (1, 0, 2)
MA(1) theta=.6  (0, 0, 1)  (0, 0, 1)  (0, 0, 1)
white noise     (0, 0, 0)  (0, 0, 0)  (0, 0, 0)
ndiffs(random walk) = 1   ndiffs(white noise) = 0
predict(n_periods=np.int64(5)) -> TypeError: n_periods must be an int
```

Why AIC prefers (1,0,2) to the true (2,0,0) was not investigated.

## Not verified

- Install on macOS or x86_64. The lock contains the wheels; nothing was run there.
- Cold-cache download time. The uv cache already held many of these wheels.
- Long training runs: 50+ epochs with early stopping, real VIX data, real features.
- cu129, and every torch version other than those named above.
- pmdarima against any statsmodels other than 0.15.0.

## Defects found while getting here

1. **The brief's premise was wrong.** "Torch should come from the default index CPU
   wheel" does not hold on Linux. PyPI's default `torch==2.14.1` is the CUDA 13.0 build
   (3.16 GB). It went unnoticed because "the system torch is CPU-only" was true for
   *that* install and was generalised to PyPI.
2. **My own smoke check had an ineffective seed.** It passed `random_state=` to an EGARCH
   `method="simulation"` forecast, which ignores it. It showed up because the "seeded"
   h=21 variance came out different in three environments (2.125, 2.107, 2.142) while
   every other check was identical. The fix is `rng=` (see Consequences).
3. **"A matmul runs" is not a GPU check.** On cu126 a matmul passed while PyTorch's own
   kernels had no image for this GPU. The probe ran an LSTM forward and backward as well,
   and that is what caught it.
