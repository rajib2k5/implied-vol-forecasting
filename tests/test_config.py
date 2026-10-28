"""Tests for ivf.config: where data lives (ADR 0001 D3).

Expected paths are literals written out by hand, never built with the code
under test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ivf import config

REPO = Path(__file__).resolve().parents[1]


def test_default_is_data_vol_projects_under_home(monkeypatch, tmp_path):
    # With HOME=<tmp> and no VOL_DATA_DIR, "~/data/vol-projects" must become
    # <tmp>/data/vol-projects, and the project root is one folder deeper.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VOL_DATA_DIR", raising=False)
    assert config.vol_data_dir() == tmp_path / "data" / "vol-projects"
    assert config.data_layout().root == tmp_path / "data" / "vol-projects" / "implied-vol-forecasting"


def test_env_var_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path / "elsewhere"))
    assert config.vol_data_dir() == tmp_path / "elsewhere"


def test_empty_env_var_counts_as_unset(monkeypatch, tmp_path):
    # An empty string would otherwise become "." and scatter data into the cwd.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOL_DATA_DIR", "")
    assert config.vol_data_dir() == tmp_path / "data" / "vol-projects"


def test_tilde_in_env_var_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOL_DATA_DIR", "~/my-vol-data")
    assert config.vol_data_dir() == tmp_path / "my-vol-data"


def test_layout_matches_adr_d3(monkeypatch):
    # ADR 0001 D3: ROOT = $VOL_DATA_DIR/implied-vol-forecasting with raw/,
    # interim/, processed/; real app JSON at ROOT/app-data/real/ (D2).
    monkeypatch.setenv("VOL_DATA_DIR", "/data/vol")
    layout = config.data_layout()
    assert layout.root == Path("/data/vol/implied-vol-forecasting")
    assert layout.raw == Path("/data/vol/implied-vol-forecasting/raw")
    assert layout.interim == Path("/data/vol/implied-vol-forecasting/interim")
    assert layout.processed == Path("/data/vol/implied-vol-forecasting/processed")
    assert layout.app_data_real == Path("/data/vol/implied-vol-forecasting/app-data/real")


def test_layout_is_read_at_call_time(monkeypatch):
    # The environment is consulted on every call, so a notebook or test can
    # change VOL_DATA_DIR after importing ivf.
    monkeypatch.setenv("VOL_DATA_DIR", "/one")
    first = config.data_layout().root
    monkeypatch.setenv("VOL_DATA_DIR", "/two")
    second = config.data_layout().root
    assert first == Path("/one/implied-vol-forecasting")
    assert second == Path("/two/implied-vol-forecasting")


def test_computing_paths_creates_nothing(monkeypatch, tmp_path):
    target = tmp_path / "never-created"
    monkeypatch.setenv("VOL_DATA_DIR", str(target))
    config.data_layout()
    config.vol_data_dir()
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_importing_ivf_creates_nothing(tmp_path):
    # Import in a fresh interpreter: modules already imported by pytest would
    # hide an import-time mkdir. cwd and HOME are the empty tmp dir, and
    # VOL_DATA_DIR points inside it, so any stray directory would show up.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "VOL_DATA_DIR": str(tmp_path / "vol")}
    code = (
        "import ivf, ivf.config, ivf.contract, ivf.data, ivf.features, ivf.models, ivf.evaluation, ivf.backtest"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


# --- repo root discovery ---------------------------------------------------


def make_fake_repo(root: Path) -> None:
    (root / "src" / "ivf").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (root / "src" / "ivf" / "__init__.py").write_text("")


def test_find_repo_root_walks_up_from_a_nested_folder(tmp_path):
    make_fake_repo(tmp_path / "proj")
    deep = tmp_path / "proj" / "notebooks" / "scratch"
    deep.mkdir(parents=True)
    assert config.find_repo_root(deep) == (tmp_path / "proj").resolve()


def test_find_repo_root_accepts_a_relative_start_path(tmp_path, monkeypatch):
    # A relative path has no parents above the current directory: Path("notebooks").parents is
    # just Path("."), and Path(".").parents is empty. So the walk only works after the path has
    # been made absolute, and the answer must be an absolute path too.
    make_fake_repo(tmp_path / "proj")
    scratch = tmp_path / "proj" / "notebooks" / "scratch"
    scratch.mkdir(parents=True)
    expected = (tmp_path / "proj").resolve()

    monkeypatch.chdir(scratch)
    assert config.find_repo_root(Path(".")) == expected  # no parents at all as written
    assert config.find_repo_root(Path("..")) == expected  # ".." is not an ancestor name either

    monkeypatch.chdir(tmp_path / "proj")
    assert config.find_repo_root(Path("notebooks")) == expected  # the repo is the cwd, outside the path's parents


def test_find_repo_root_ignores_an_unrelated_pyproject(tmp_path):
    # A pyproject.toml with no src/ivf next to it is some other project.
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "pyproject.toml").write_text("")
    with pytest.raises(config.RepoRootNotFound):
        config.find_repo_root(tmp_path / "other")


def test_find_repo_root_raises_outside_any_repo(tmp_path):
    with pytest.raises(config.RepoRootNotFound):
        config.find_repo_root(tmp_path)


def test_repo_root_is_this_checkout():
    # tests/ sits directly inside the repo, so its parent is the root.
    assert config.repo_root() == REPO


def test_schema_dir_holds_the_contract_schemas():
    assert config.schema_dir() == REPO / "schemas" / "app-data"
    assert (config.schema_dir() / "envelope.schema.json").is_file()
