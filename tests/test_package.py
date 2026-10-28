"""Tests for the package itself: version and layout."""

from __future__ import annotations

import tomllib
from pathlib import Path

import ivf

REPO = Path(__file__).resolve().parents[1]


def test_version_comes_from_pyproject():
    # pyproject.toml is the one place the version is written. The package
    # reads it back from the installed metadata, so the two cannot drift.
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert ivf.__version__ == pyproject["project"]["version"]
