"""Tests for .gitignore: vendor data and secrets must not be swept in by `git add -A`.

Each path below is asked of git itself (`git check-ignore`), so the test checks
what git does, not what the file says. The paths need not exist.

`--no-index` makes git judge the path as if it were untracked, which is the
situation that matters (a file saved by a notebook, not yet added). The
`core.excludesFile` override keeps one developer's global ignore rules from
deciding the result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Everything here must be ignored.
MUST_BE_IGNORED = [
    # Vendor tables (Cboe/FRED/Yahoo terms: personal, non-commercial use). The old file
    # covered only *.csv.gz, so a plain VIX_History.csv at the root or in a notebook folder
    # was committed by `git add -A`.
    "VIX_History.csv",
    "VIX_HISTORY.CSV",  # git has no case-insensitive switch; the pattern spells out both cases
    "Vix_History.Csv",
    "notebooks/VIX_History.csv",
    "reports/VIX_History.csv",
    "VIXCLS.csv",
    "raw/cboe/VIX/2026-10-27.csv",
    "tests/x.csv",  # only tests/fixtures/ is exempt, not the whole tests/ folder
    "reports/metrics.csv",  # a committed table must be named *.agg.csv (ADR 0001 D2)
    "reports/regimes/high/table.agg.csv",  # regime cells stay local (D2 publication set P)
    "history.csv.gz",
    "bars.parquet",
    "data/raw.bin",  # the repo-root data/ folder (anchored pattern)
    "app/public/data/real/manifest.json",
    "app/public/data/term_structure.json",  # real export outside synthetic/ and real/
    "app/src/data/term-structure.real.json",
    "notebooks/vix.real.json",
    "reports/model_race.real.json",  # outside notebooks/ and app/, so only *.real.json catches it
    # Other data and cache formats
    "VIX_History.tsv",
    "VIX_History.zip",
    "frame.feather",
    "frame.arrow",
    "prices.npy",
    "prices.npz",
    "model.joblib",
    "frame.pkl",
    "frame.pickle",
    "store.h5",
    "store.hdf5",
    "book.xlsx",
    "book.xls",
    "cache.sqlite",
    "cache.sqlite-journal",
    "prices.db",
    "prices.duckdb",
    ".cache/requests/abc",
    "py-yfinance/cookies.txt",  # not *.db, so only the folder pattern catches it
    # JSON dropped where a notebook or a quick export would put it (FRED's API returns JSON)
    "notebooks/fred.json",
    "notebooks/data/deep/vix.json",
    "app/src/data/term-structure.json",  # only *.synthetic.json is committed there
    "app/src/data/vix.json",
    # Secrets
    ".env",
    ".env.local",
    "credentials.json",
    "kaggle.json",
    ".netrc",
    ".npmrc",
    "server.pem",
    "server.key",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    ".pypirc",
    "secrets.toml",
    ".envrc",  # direnv: a common home for FRED_API_KEY=...
    "client_secret.json",
    "client_secret_123.apps.googleusercontent.com.json",
    "token.json",
    ".git-credentials",
    ".aws/credentials",
    "fred_api_key.txt",
    "FRED_API_KEY.txt",  # the character classes make the api_key pattern case-insensitive
    "fred-api-key",  # separator - as well as _
    "my_Api_Key.json",
    "config/apikey.json",
    # Per-machine tool files
    ".claude/settings.local.json",
    "CLAUDE.local.md",
    ".mcp.json",
    ".vscode/settings.json",
    ".idea/workspace.xml",
]

# Everything here is committed on purpose and must NOT be ignored.
MUST_BE_KEPT = [
    # The two CSV exceptions
    "tests/fixtures/x.csv",
    "tests/fixtures/deep/er/y.csv",  # "**/" also matches several folders
    "reports/metrics.agg.csv",
    "tests/fixtures/x.parquet",
    # The committed synthetic fixtures (an unanchored `data/` pattern would hide them)
    "app/src/data/term-structure.synthetic.json",
    "app/public/data/synthetic/manifest.json",
    # Ordinary project files and shared (not local) settings
    ".env.example",
    ".claude/settings.json",
    "README.md",
    "src/ivf/config.py",
    "schemas/app-data/envelope.schema.json",
    "app/package.json",
    "app/tsconfig.json",
    "notebooks/00_environment_and_contract.ipynb",  # .ipynb, not .json: notebooks/**/*.json must not catch it
    "notebooks/01_data_and_eda.ipynb",
    # Milestone 1: the committed protocol and registries (data-model.md section 1)
    "configs/protocol.yaml",
    "configs/protocol.synthetic.yaml",
    "configs/iv_indices.yaml",
    "configs/instruments.yaml",
]


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "-q", "--no-index", path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 0: ignored. Exit 1: not ignored. Anything else (128) means git itself failed.
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_vendor_data_secrets_and_local_tool_files_are_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize("path", MUST_BE_KEPT)
def test_committed_files_are_not_ignored(path):
    assert not is_ignored(path)
