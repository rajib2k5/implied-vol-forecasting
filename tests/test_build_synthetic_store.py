"""Tests for tools/build_synthetic_store.py: the one command that rebuilds the synthetic dev store.

No step is run for real here (the chain takes well over 20 minutes); a stand-in runner records the
commands, which is what the tool controls: which steps, in which order, with --synthetic placed
before the subcommand, and that the first failure stops the chain.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_synthetic_store as bss  # noqa: E402

EXPECTED = [
    "ivf.data --synthetic fetch", "ivf.data --synthetic build-interim", "ivf.data --synthetic check", "ivf.data --synthetic eda",
    "ivf.models --synthetic baselines", "ivf.models --synthetic garch", "ivf.models --synthetic lstm",
    "ivf.evaluation --synthetic report", "ivf.backtest --synthetic prepare", "ivf.backtest --synthetic run",
]


def test_runs_every_milestone_step_in_order_with_synthetic_before_the_subcommand():
    seen = []
    out = io.StringIO()
    assert bss.main([], run=lambda cmd: seen.append(cmd) or 0, out=out) == 0
    assert [cmd[0] for cmd in seen] == [sys.executable] * 10
    assert [" ".join(cmd[2:]) for cmd in seen] == EXPECTED  # cmd[1] is "-m"
    assert "ok: synthetic store built" in out.getvalue()


def test_the_first_failing_step_stops_the_chain_with_its_exit_code(capsys):
    seen = []

    def run(cmd):
        seen.append(cmd)
        return 3 if cmd[-1] == "garch" else 0

    assert bss.main([], run=run, out=io.StringIO()) == 3
    assert [cmd[-1] for cmd in seen] == ["fetch", "build-interim", "check", "eda", "baselines", "garch"]
    assert "step 6 (python -m ivf.models --synthetic garch) failed" in capsys.readouterr().err


def test_dry_run_prints_the_steps_and_runs_nothing():
    out = io.StringIO()
    assert bss.main(["--dry-run"], run=lambda cmd: 1 / 0, out=out) == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == " 1. python -m ivf.data --synthetic fetch" and len(lines) == 10
