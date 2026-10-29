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
M1 = "01_data_and_eda.ipynb"
M2 = "02_baselines.ipynb"
M3 = "03_garch.ipynb"


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
    assert result.stdout.startswith("ok: 4 notebook(s)")  # amended M3: 03_garch.ipynb


@pytest.mark.parametrize("name", [M0, M1, M2, M3])
def test_committed_bytes_equal_generated_bytes(name):
    assert (REPO / "notebooks" / name).read_bytes() == build_notebooks.generate()[name]


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


# --- milestone 1: the data and EDA notebook ---------------------------------------


def test_m1_notebook_runs_top_to_bottom_on_synthetic_data_when_there_is_no_real_data(tmp_path, monkeypatch, capsys, no_network):
    # An empty VOL_DATA_DIR: no interim tables, so the notebook must fall back to synthetic
    # data and say so, then run every EDA cell.
    import matplotlib.pyplot as plt

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    # Another test may already have imported pyplot with a GUI backend (DISPLAY is set on a desktop),
    # and a figure must never open a window from pytest.
    plt.switch_backend("Agg")
    notebook = nbformat.reads(build_notebooks.generate()[M1].decode(), as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.outputs == [] and c.execution_count is None for c in code_cells)
    namespace: dict = {"__name__": "__notebook__"}
    for cell in code_cells:
        exec(compile(cell.source, cell.id, "exec"), namespace)
    out = capsys.readouterr().out
    assert namespace["MODE"] == "synthetic"
    assert "data: SYNTHETIC (seed 20261028)" in out
    assert "gap policy: OK" in out
    assert "stationary at 5% (ADF rejects and KPSS does not):" in out
    assert "contango share" in out
    # No holdout row reaches the EDA (amended M1, review A).
    holdout_start = namespace["PROTO"].holdout_start
    assert max(namespace["DATES"]) < holdout_start
    assert namespace["EDA_IV"]["date"].max() < holdout_start and namespace["EDA_PRICE"]["date"].max() < holdout_start
    assert set(namespace["SAMPLES"]) == {"pre_holdout", "burn_in", "dev"}
    plt.close("all")


def test_m1_notebook_can_be_forced_to_synthetic_data(monkeypatch, capsys, no_network):
    monkeypatch.setenv("IVF_NOTEBOOK_DATA", "synthetic")
    notebook = nbformat.reads(build_notebooks.generate()[M1].decode(), as_version=4)
    namespace: dict = {"__name__": "__notebook__"}
    exec(compile(notebook.cells[1].source, "cell-001", "exec"), namespace)
    assert namespace["MODE"] == "synthetic"


def test_m1_notebook_stops_when_check_reports_violated(monkeypatch, capsys, no_network):
    # Review B: the notebook printed "gap policy: VIOLATED" and went on to compute every statistic.
    # Here the synthetic build report says the snapshots were fetched on data_end's own New York day
    # (15:00 UTC is 11:00 there), so the closes are not settled and the frozen data_end fails.
    monkeypatch.setenv("IVF_NOTEBOOK_DATA", "synthetic")
    notebook = nbformat.reads(build_notebooks.generate()[M1].decode(), as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    namespace: dict = {"__name__": "__notebook__"}
    exec(compile(code_cells[0].source, code_cells[0].id, "exec"), namespace)
    for entry in namespace["REPORT"]["files"]:
        entry["fetched_at"] = "2026-10-26T15:00:00.000000Z"
    with pytest.raises(RuntimeError, match="check reports VIOLATED"):
        exec(compile(code_cells[1].source, code_cells[1].id, "exec"), namespace)
    assert "gap policy: VIOLATED" in capsys.readouterr().out
    assert "FEATURES" not in namespace  # nothing after the check ran


# --- milestone 2: the baselines notebook ------------------------------------------------------


def test_m2_notebook_runs_top_to_bottom_on_synthetic_data(tmp_path, monkeypatch, capsys, no_network):
    # Synthetic data, one fold: the notebook must fit every baseline, print the losses and the
    # known-truth floor, and read nothing on or after holdout_start.
    import matplotlib.pyplot as plt

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IVF_NOTEBOOK_FOLDS", "1")
    plt.switch_backend("Agg")
    notebook = nbformat.reads(build_notebooks.generate()[M2].decode(), as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.outputs == [] and c.execution_count is None for c in code_cells)
    namespace: dict = {"__name__": "__notebook__"}
    for cell in code_cells:
        exec(compile(cell.source, cell.id, "exec"), namespace)
    out = capsys.readouterr().out
    assert namespace["MODE"] == "synthetic" and "data: SYNTHETIC" in out
    assert "97 dev folds; first cutoff 2000-03-15; last dev origin 2024-06-28" in out
    assert out.count("ARIMA(") == 1 and "RMSE of the true conditional mean" in out
    assert set(namespace["FC"].model_id) == {"rw", "hist_mean", "hist_mean_252", "arima_log", "har_log"}
    assert max(namespace["DATES"]) < namespace["PROTO"].holdout_start
    plt.close("all")


# --- milestone 3: the GARCH notebook ------------------------------------------------------------


def test_m3_notebook_runs_top_to_bottom_on_synthetic_data(tmp_path, monkeypatch, capsys, no_network):
    # Synthetic data, one fold: every rv21 model and ARIMA-GARCH fitted, the hand-worked aggregate,
    # the known-truth floor and the category-error demo printed; nothing on or after holdout_start read.
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IVF_NOTEBOOK_FOLDS", "1")
    notebook = nbformat.reads(build_notebooks.generate()[M3].decode(), as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.outputs == [] and c.execution_count is None for c in code_cells)
    namespace: dict = {"__name__": "__notebook__"}
    for cell in code_cells:
        exec(compile(cell.source, cell.id, "exec"), namespace)
    out = capsys.readouterr().out
    assert namespace["MODE"] == "synthetic" and "data: SYNTHETIC" in out
    assert "sum of the 21 daily variances 38.287209" in out and "yhat = 21.4347 vol points" in out
    assert "truth: QLIKE" in out and out.count("AR(") == 1 and "level_alpha_plus_beta" in out
    assert set(namespace["FC"].model_id) == {"rw", "har_rv", "vix_implied", "garch", "gjr", "egarch"}
    assert max(namespace["DATES"]) < namespace["PROTO"].holdout_start

