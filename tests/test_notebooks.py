"""Tests for tools/build_notebooks.py: notebooks are generated, never hand-edited.

The committed .ipynb files must equal what the specs generate, byte for byte,
and --check must notice when they do not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import build_notebooks  # noqa: E402

M0 = "00_environment_and_contract.ipynb"


@pytest.fixture
def written(tmp_path):
    """A scratch notebooks/ folder holding freshly generated notebooks."""
    build_notebooks.write(tmp_path)
    return tmp_path


def read_notebook(path):
    return nbformat.read(path, as_version=4)


# --- the committed notebooks are in sync -------------------------------------


def test_check_command_passes_on_the_committed_notebooks(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOOLS / "build_notebooks.py"), "--check"],
        cwd=tmp_path,  # prove it does not depend on the working directory
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok: 1 notebook(s)")


def test_committed_bytes_equal_generated_bytes():
    assert (REPO / "notebooks" / M0).read_bytes() == build_notebooks.generate()[M0]


# --- what a generated notebook looks like ------------------------------------


def test_cell_ids_are_derived_from_position():
    # Cell i gets the id "cell-" + 3-digit i, so 0, 1, 2 give cell-000, cell-001, cell-002.
    notebook = nbformat.reads(build_notebooks.generate()[M0].decode(), as_version=4)
    ids = [cell.id for cell in notebook.cells]
    assert ids[:3] == ["cell-000", "cell-001", "cell-002"]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", i) for i in ids)  # nbformat's id rule


def test_generation_is_deterministic():
    assert build_notebooks.generate() == build_notebooks.generate()


def test_outputs_are_stripped():
    notebook = nbformat.reads(build_notebooks.generate()[M0].decode(), as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells, "the M0 notebook must have code cells"
    assert all(c.outputs == [] and c.execution_count is None for c in code_cells)


def test_generated_notebook_is_valid_nbformat():
    nbformat.validate(nbformat.reads(build_notebooks.generate()[M0].decode(), as_version=4))


def test_code_cells_run_top_to_bottom_and_show_the_contract_at_work(capsys):
    # Run the cells in one shared namespace, like a kernel would. Executing for
    # real catches a notebook that was generated fine but does not run.
    notebook = nbformat.reads(build_notebooks.generate()[M0].decode(), as_version=4)
    namespace: dict = {"__name__": "__notebook__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, cell.id, "exec"), namespace)
    out = capsys.readouterr().out
    assert "torch.get_num_threads():" in out
    assert '"data_mode":"synthetic"' in out  # the example envelope was serialised
    assert "'paper' is not one of ['synthetic', 'real']" in out  # the violation was reported
    assert "export missing values as None, not NaN" in out  # NaN was refused
    assert namespace["envelope"]["dataset"] == "example"


# --- --check notices drift (mutation-style) ----------------------------------


def test_check_is_clean_for_fresh_notebooks(written):
    assert build_notebooks.check(written) == []


def test_check_detects_an_edited_cell(written):
    path = written / M0
    path.write_bytes(path.read_bytes().replace(b"import ivf", b"import ivx", 1))
    problems = build_notebooks.check(written)
    assert len(problems) == 1 and problems[0].startswith("stale:")


def test_check_detects_saved_outputs(written):
    # What happens when someone runs the notebook in Jupyter and saves it.
    path = written / M0
    notebook = read_notebook(path)
    code_cell = next(c for c in notebook.cells if c.cell_type == "code")
    code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="hi\n")]
    code_cell.execution_count = 1
    nbformat.write(notebook, path)
    problems = build_notebooks.check(written)
    assert len(problems) == 1 and problems[0].startswith("stale:")


def test_check_detects_a_trailing_whitespace_change(written):
    path = written / M0
    path.write_bytes(path.read_bytes() + b"\n")
    assert [p.split(":")[0] for p in build_notebooks.check(written)] == ["stale"]


def test_check_detects_a_missing_notebook(written):
    (written / M0).unlink()
    problems = build_notebooks.check(written)
    assert len(problems) == 1 and problems[0].startswith("missing:")


def test_check_detects_a_notebook_without_a_spec(written):
    (written / "99_scratch.ipynb").write_bytes((written / M0).read_bytes())
    problems = build_notebooks.check(written)
    assert len(problems) == 1 and problems[0].startswith("unexpected:") and "99_scratch.ipynb" in problems[0]


def test_check_command_exits_nonzero_on_drift(written, monkeypatch, capsys):
    path = written / M0
    path.write_bytes(path.read_bytes().replace(b"import ivf", b"import ivx", 1))
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS_DIR", written)
    assert build_notebooks.main(["--check"]) == 1
    assert "stale:" in capsys.readouterr().err


def test_check_command_exits_zero_when_in_sync(written, monkeypatch, capsys):
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS_DIR", written)
    assert build_notebooks.main(["--check"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_write_then_check_round_trips_in_a_clean_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS_DIR", tmp_path / "notebooks")
    assert build_notebooks.main([]) == 0  # no --check: writes (and creates the folder)
    assert build_notebooks.main(["--check"]) == 0
