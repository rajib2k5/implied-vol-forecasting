"""Build the synthetic dev store that the app export reads, with one command (polish review).

    uv run python tools/build_synthetic_store.py            # every step, in order
    uv run python tools/build_synthetic_store.py --dry-run  # print the steps, run nothing

Runs, with ``--synthetic`` and in this order, every step of milestones 1 to 6 on the seeded
synthetic market under ``$VOL_DATA_DIR/implied-vol-forecasting/synthetic/`` (default
``~/data/vol-projects``): the data pipeline and EDA, the baselines, the GARCH family, the LSTM,
the M5 evaluation report and the M6 backtest (prepare, then the dev run). Nothing here reads or
writes the holdout: every step runs its development-period path.

Then ``uv run python tools/export_app_data.py --mode synthetic`` writes the app's JSON from it.
Run it at a clean commit if the JSON is to be committed: the store's run id hashes the commit.

Each step prints its wall time. The first step that fails stops the chain, and the exit code is
that step's (1 if it died without one). Run time is dominated by the model steps; the polish
review measured the baselines step alone at 1,318 s, and the whole chain has not been timed end to
end here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Callable, Sequence

STEPS: tuple[tuple[str, ...], ...] = (
    ("ivf.data", "fetch"),
    ("ivf.data", "build-interim"),
    ("ivf.data", "check"),
    ("ivf.data", "eda"),
    ("ivf.models", "baselines"),
    ("ivf.models", "garch"),
    ("ivf.models", "lstm"),
    ("ivf.evaluation", "report"),
    ("ivf.backtest", "prepare"),
    ("ivf.backtest", "run"),
)


def command(step: Sequence[str]) -> list[str]:
    """``python -m <module> --synthetic <subcommand>``: ``--synthetic`` goes before the subcommand."""
    module, *rest = step
    return [sys.executable, "-m", module, "--synthetic", *rest]


def main(argv: list[str] | None = None, run: Callable[[list[str]], int] | None = None, out=None) -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic dev store, every step in order.")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and run nothing")
    args = parser.parse_args(argv)
    out = out or sys.stdout
    run = run or (lambda cmd: subprocess.run(cmd, check=False).returncode)
    total = time.perf_counter()
    for i, step in enumerate(STEPS, 1):
        cmd = command(step)
        shown = " ".join(["python", *cmd[1:]])
        if args.dry_run:
            print(f"{i:2d}. {shown}", file=out)
            continue
        print(f"[{i}/{len(STEPS)}] {shown}", file=out, flush=True)
        started = time.perf_counter()
        code = run(cmd)
        print(f"[{i}/{len(STEPS)}] exit {code} in {time.perf_counter() - started:.1f} s", file=out, flush=True)
        if code != 0:
            print(f"error: step {i} ({shown}) failed; the steps after it did not run", file=sys.stderr)
            return code if isinstance(code, int) and code > 0 else 1
    if not args.dry_run:
        print(f"ok: synthetic store built in {time.perf_counter() - total:.1f} s; now run "
              "`uv run python tools/export_app_data.py --mode synthetic`", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
