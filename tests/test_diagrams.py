"""Tests for tools/diagrams/: the rendered diagrams in docs/architecture/out/ match their specs.

The vendored diagram kit (tools/diagrams/, see VENDORED.md) renders the YAML specs in
docs/architecture/specs/ into docs/architecture/out/. `--check` regenerates every output in a
temp directory and compares it byte for byte with out/: exit 0 when everything is current, 1
when an output is stale, 2 on a spec error.

`--no-png` is used because the kit verified PNG bytes on aarch64 only (kit README,
"Determinism"). The other outputs (.mmd, .svg, .drawio, .lucid.xml) are still compared byte for
byte, and a PNG is a raster of the SVG that was compared. These tests never compare PNGs.

The kit runs in a subprocess with the interpreter that is running pytest, which has to carry the
`diagrams` dependency group (pyyaml, resvg-py). That is the same interpreter
`uv run --group diagrams python tools/diagrams/build.py ...` would use, without a nested uv
call that could re-sync the project's environment in the middle of a test run. When the group is
not installed the tests skip, naming the command that installs it.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "tools" / "diagrams" / "build.py"
SPECS = REPO / "docs" / "architecture" / "specs"
OUT = REPO / "docs" / "architecture" / "out"

GROUP_MODULES = ("yaml", "resvg_py")  # pyyaml, resvg-py: the `diagrams` dependency group
needs_group = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in GROUP_MODULES),
    reason=(
        "the `diagrams` dependency group (pyyaml, resvg-py) is not installed; "
        "run `uv run --group diagrams pytest tests/test_diagrams.py`"
    ),
)


def run_check(specs: Path, out: Path) -> subprocess.CompletedProcess:
    """The kit's --check line, as the kit README gives it, without PNGs."""
    return subprocess.run(
        [sys.executable, str(BUILD), str(specs), "--out", str(out), "--check", "--no-png"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


@needs_group
def test_rendered_diagrams_are_up_to_date_with_their_specs():
    n_specs = len(list(SPECS.glob("*.yaml")))
    assert n_specs > 0, f"no specs found in {SPECS}"

    result = run_check(SPECS, OUT)

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    # The count ties the pass to every spec being checked, not to an empty folder.
    assert result.stdout.startswith(f"ok: {n_specs} spec(s) up to date"), result.stdout


@needs_group
def test_check_notices_a_spec_edited_without_rebuilding(tmp_path):
    # A copy of specs and out where one spec's title is edited and out/ is not rebuilt.
    specs = tmp_path / "specs"
    out = tmp_path / "out"
    shutil.copytree(SPECS, specs)
    shutil.copytree(OUT, out)
    edited = sorted(specs.glob("*.yaml"))[0]
    text = edited.read_text(encoding="utf-8")
    changed = re.sub(r"(?m)^title: (.*)$", r"title: \1 edited", text, count=1)
    assert changed != text, f"no top-level title: line in {edited.name}"
    edited.write_text(changed, encoding="utf-8")

    result = run_check(specs, out)

    assert result.returncode == 1, f"{result.stdout}{result.stderr}"
    assert f"differs  {out / (edited.stem + '.mmd')}" in result.stderr, result.stderr
