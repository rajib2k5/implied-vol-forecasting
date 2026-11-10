"""Fail if a raw vendor file, or a very large file, is tracked or staged in git (ADR 0001 D2). Local or CI.

    uv run python tools/licensing/check_raw_files.py            # ROOT from $VOL_DATA_DIR
    uv run python tools/licensing/check_raw_files.py --root DIR [--repo DIR]

This gate sits beside ``scan_real_rows.py``. The scanner reads text and looks for vendor *values*;
this gate compares whole files and looks for vendor *files*, so it also catches a renamed copy, a
binary one, and a file too large to read as text. It checks:

1. Raw content. The raw snapshots under ``ROOT/raw/**/*.csv`` are named by the SHA-256 of their bytes
   (``src/ivf/data/raw.py``); the synthetic store under ``ROOT/synthetic/raw/`` is not under ``ROOT/raw``
   and is not vendor data. Every file git holds is hashed with SHA-256 and the hash is compared with the
   hash of each snapshot. The match is by content, never by name, so ``notes.txt`` with a snapshot's
   bytes is flagged. Each snapshot is re-hashed from its bytes, so a file that was renamed in the store
   is still found; an empty snapshot is ignored (the empty file would match every empty file).
2. Size. A file over ``FAIL_BYTES`` (50 MB) fails; over ``WARN_BYTES`` (10 MB) it warns.
3. Ignored raw extensions. A tracked or staged path that ends in a raw data extension (``RAW_SUFFIXES``)
   and that ``.gitignore`` would ignore (``git check-ignore --no-index``) fails. ``.gitignore`` already
   blocks these (every ``*.csv``, ``*.parquet``, ``*.zip``, ... except ``tests/fixtures/`` and
   ``reports/*.agg.csv``), so a match means someone forced the file in with ``git add -f``. Asking git
   keeps the exceptions in one place and makes this check minimal.

"Held by git" is two sets: the committed content of every path in ``git ls-files`` (read as
``HEAD:path``; a path new in the index has none), and the staged content of every path in
``git diff --cached --name-only --diff-filter=ACMR`` (read from the index blob, ``:path``). So a
staged-but-modified file is checked as staged, and an unstaged untracked file is not checked at all
(git would not commit it). Exit 0 and one summary line if all is well; exit 1 and one line per offender
(path, reason, hash) otherwise. With no raw store, the gate says so and still runs the size and
extension checks.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ivf import config

FAIL_BYTES = 50 * 1024 * 1024
WARN_BYTES = 10 * 1024 * 1024
#: Extensions of raw or bulk data that .gitignore blocks (lower case; the path is lower-cased to compare).
RAW_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".parquet",
    ".zip",
    ".xlsx",
    ".xls",
    ".feather",
    ".arrow",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".sqlite",
    ".db",
    ".duckdb",
    ".pkl",
    ".pickle",
    ".joblib",
)


@dataclass(frozen=True)
class Offence:
    path: str
    reason: str
    sha256: str


@dataclass(frozen=True)
class Report:
    offences: list[Offence]
    warnings: list[Offence]
    checked: int  # (path, content) pairs hashed
    oversize: int  # of those, the number over FAIL_BYTES
    raw_snapshots: int  # 0 when there is no raw store


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, input=stdin).stdout


def _names(output: bytes) -> list[str]:
    return [name.decode() for name in output.split(b"\0") if name]


def raw_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of the bytes of each non-empty snapshot under ``root/raw``, mapped to its path (for the report)."""
    found: dict[str, str] = {}
    raw_dir = root / "raw"
    if not raw_dir.is_dir():
        return found
    for path in sorted(raw_dir.rglob("*.csv")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if data:
            found[hashlib.sha256(data).hexdigest()] = str(path.relative_to(raw_dir))
    return found


class _Blobs:
    """``git cat-file --batch``: blob bytes by rev spec such as ``HEAD:a.txt`` or ``:a.txt`` (None if absent)."""

    def __init__(self, repo: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def get(self, spec: str) -> bytes | None:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(spec.encode() + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            return None  # "<spec> missing", or a submodule or tree
        size = int(header[2])
        data = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # the newline after the content
        return data

    def close(self) -> None:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.close()
        self._proc.wait()
        self._proc.stdout.close()


def _ignored_raw_extension(repo: Path, paths: list[str]) -> set[str]:
    candidates = [p for p in paths if p.lower().endswith(RAW_SUFFIXES)]
    if not candidates:
        return set()
    # exit 1 means "none ignored"; --no-index makes git apply the rules to tracked paths too
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=repo,
        capture_output=True,
        input=b"\0".join(p.encode() for p in candidates) + b"\0",
    )
    if proc.returncode not in (0, 1):
        raise subprocess.CalledProcessError(proc.returncode, proc.args, proc.stdout, proc.stderr)
    return set(_names(proc.stdout))


def check(repo: Path, root: Path) -> Report:
    snapshots = raw_hashes(root)
    tracked = _names(_git(repo, "ls-files", "-z"))
    staged = _names(_git(repo, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"))
    offences: list[Offence] = []
    warnings: list[Offence] = []
    checked = oversize = 0
    blobs = _Blobs(repo)
    try:
        for label, paths, prefix in (("tracked", tracked, "HEAD:"), ("staged", staged, ":")):
            for path in paths:
                data = blobs.get(prefix + path)
                if data is None:
                    continue
                checked += 1
                digest = hashlib.sha256(data).hexdigest()
                if digest in snapshots:
                    offences.append(Offence(path, f"{label} file has the bytes of a raw vendor snapshot", digest))
                if len(data) > FAIL_BYTES:
                    oversize += 1
                    offences.append(Offence(path, f"{label} file is {len(data):,} bytes, over the {FAIL_BYTES:,} limit", digest))
                elif len(data) > WARN_BYTES:
                    warnings.append(Offence(path, f"{label} file is {len(data):,} bytes, over {WARN_BYTES:,}", digest))
    finally:
        blobs.close()
    for path in sorted(_ignored_raw_extension(repo, sorted(set(tracked) | set(staged)))):
        offences.append(Offence(path, "tracked or staged raw data extension that .gitignore blocks", "-"))
    return Report(offences, warnings, checked, oversize, len(snapshots))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, help="the data root (default: ROOT from VOL_DATA_DIR)")
    parser.add_argument("--repo", type=Path, help="the checkout to check (default: this one)")
    args = parser.parse_args(argv)
    root = (args.root or config.data_layout().root).expanduser()
    repo = (args.repo or config.repo_root()).expanduser()
    report = check(repo, root)
    if not report.raw_snapshots:
        print(f"note: no raw snapshots under {root / 'raw'}; the raw-content check is skipped, size and extension checks run")
    for warning in report.warnings:
        print(f"warning: {warning.path}: {warning.reason} (sha256 {warning.sha256})")
    for offence in report.offences:
        print(f"{offence.path}: {offence.reason} (sha256 {offence.sha256})")
    if report.offences:
        print(f"raw-file gate FAILED: {len(report.offences)} problem(s) in {report.checked} files checked")
        return 1
    print(f"raw-file gate: 0 raw vendor files tracked or staged, 0 files over 50 MB ({report.checked} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
