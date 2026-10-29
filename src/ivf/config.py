"""Where things live on disk (ADR 0001 D3).

Real market data is licensed for personal, non-commercial use, so it must
never be inside the git repository. Everything the pipeline reads or writes
therefore lives under one directory *outside* the repo::

    $VOL_DATA_DIR/implied-vol-forecasting/      <- project root ("ROOT" in the ADR)
        raw/            bytes exactly as downloaded
        interim/        typed, de-duplicated parquet
        processed/      parquet written by the pipeline stages
        app-data/real/  JSON for the app, built from real data
        eda/            local EDA results on real data (never committed, ADR 0001 D2)
        synthetic/      the same layout again, for the seeded synthetic data

``VOL_DATA_DIR`` defaults to ``~/data/vol-projects``, a folder shared by the
sibling volatility projects.

Two rules that every function here follows:

* Nothing creates a directory. These functions only compute paths. Whoever
  writes a file calls ``mkdir`` at that moment. Importing this module on a
  machine with no data folder must be harmless.
* The environment variable is read when a function is *called*, not when the
  module is imported. Tests (and notebooks) can then change it on the fly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Environment variable that moves the whole data tree.
ENV_VAR = "VOL_DATA_DIR"
#: Used when ``VOL_DATA_DIR`` is unset or empty. ``~`` is expanded at call time.
DEFAULT_DATA_DIR = "~/data/vol-projects"
#: Name of this project's folder inside ``VOL_DATA_DIR``.
PROJECT_DIRNAME = "implied-vol-forecasting"

# The repo is recognised by a pyproject.toml that sits next to src/ivf/.
# Checking both avoids stopping at an unrelated pyproject.toml higher up.
_REPO_MARKERS = ("pyproject.toml", "src/ivf/__init__.py")


class RepoRootNotFound(RuntimeError):
    """Raised when ivf is not running from a source checkout (for example, a
    wheel installed into site-packages has no repo around it)."""


@dataclass(frozen=True)
class DataLayout:
    """The directories of ADR 0001 D3, as paths (not necessarily existing)."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    #: JSON for the app built from real data. Outside the repo on purpose:
    #: ``vite build`` copies ``app/public/`` into ``dist/``, so real data
    #: placed there could be deployed by accident (ADR 0001 D2 and O4).
    app_data_real: Path
    #: Local EDA results (milestone 1). Real-data EDA numbers are outside the
    #: publication set P (ADR 0001 D2), so they live here and nowhere else.
    eda: Path


def vol_data_dir() -> Path:
    """``$VOL_DATA_DIR``, or ``~/data/vol-projects``, with ``~`` expanded.

    An empty ``VOL_DATA_DIR=`` counts as unset: an empty string would turn
    into the current directory, which is never what anyone means.
    """
    return Path(os.environ.get(ENV_VAR) or DEFAULT_DATA_DIR).expanduser()


def data_layout(*, synthetic: bool = False) -> DataLayout:
    """Every data directory for this project, derived from ``vol_data_dir()``.

    ``synthetic=True`` gives the same layout under ``ROOT/synthetic/`` (ADR 0001
    D3: "Synthetic runs use the same layout under ROOT/synthetic/"), so the
    synthetic data runs through exactly the code the real data does.
    """
    root = vol_data_dir() / PROJECT_DIRNAME
    if synthetic:
        root = root / "synthetic"
    return DataLayout(
        root=root,
        raw=root / "raw",
        interim=root / "interim",
        processed=root / "processed",
        app_data_real=root / "app-data" / "real",
        eda=root / "eda",
    )


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the first directory that looks like this repo.

    Raises ``RepoRootNotFound`` if there is none.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if all((candidate / marker).is_file() for marker in _REPO_MARKERS):
            return candidate
    raise RepoRootNotFound(f"no directory above {start} contains {' and '.join(_REPO_MARKERS)}")


def repo_root() -> Path:
    """The git checkout this code runs from.

    The search starts at this file, not at the current directory, so the
    answer is the same whether a notebook runs from ``notebooks/`` or a tool
    runs from the repo root.
    """
    return find_repo_root(Path(__file__))


def configs_dir() -> Path:
    """The committed YAML configuration (data-model.md section 1)."""
    return repo_root() / "configs"


def schema_dir() -> Path:
    """The JSON Schemas of the app-data contract (the single source of truth).

    There is deliberately no helper for ``app/public/data/synthetic/`` here:
    ADR 0001 D1 says ``src/ivf`` must not touch anything under ``app/``, so
    only ``tools/export_app_data.py`` knows that location.
    """
    return repo_root() / "schemas" / "app-data"
