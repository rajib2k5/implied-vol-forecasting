"""Tests for tools/licensing/check_raw_files.py on made-up snapshots and scratch git repositories.

The gate compares the bytes of every tracked or staged file with the bytes of the raw vendor
snapshots. Here the "vendor" snapshots are made-up CSV bytes written under ``tmp_path``; no real
vendor file is read. Each test builds a scratch repo, so the real repo's index plays no part.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "licensing"))

import check_raw_files as gate  # noqa: E402

FAKE_VENDOR = b"DATE,OPEN,HIGH,LOW,CLOSE\n2031-03-03,11.11,12.22,10.33,11.44\n2031-03-04,12.55,13.66,11.77,12.88\n"
OTHER_VENDOR = b"DATE,CLOSE\n2031-04-01,20.01\n2031-04-02,20.02\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "README.md").write_text("hello\n")
    git(path, "add", "README.md")
    git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A data root with two fake vendor snapshots, and a synthetic store that is not vendor data."""
    base = tmp_path / "data"
    for symbol, content in (("VIX", FAKE_VENDOR), ("VIX3M", OTHER_VENDOR)):
        folder = base / "raw" / "cboe" / symbol
        folder.mkdir(parents=True)
        (folder / f"{hashlib.sha256(content).hexdigest()}.csv").write_bytes(content)
    synthetic = base / "synthetic" / "raw" / "synthetic" / "VIX"
    synthetic.mkdir(parents=True)
    (synthetic / "whatever.csv").write_bytes(b"DATE,CLOSE\n2031-05-01,1.5\n")
    return base


def paths(report: gate.Report) -> set[str]:
    return {o.path for o in report.offences}


def test_clean_repo_passes(repo: Path, root: Path) -> None:
    report = gate.check(repo, root)
    assert report.offences == [] and report.warnings == []
    assert report.checked == 1 and report.raw_snapshots == 2


def test_staged_renamed_copy_is_caught(repo: Path, root: Path) -> None:
    (repo / "notes.txt").write_bytes(FAKE_VENDOR)
    git(repo, "add", "notes.txt")
    report = gate.check(repo, root)
    assert paths(report) == {"notes.txt"}
    assert report.offences[0].sha256 == hashlib.sha256(FAKE_VENDOR).hexdigest()


def test_staged_copy_is_caught_by_its_staged_bytes_not_the_work_tree(repo: Path, root: Path) -> None:
    (repo / "notes.txt").write_bytes(FAKE_VENDOR)
    git(repo, "add", "notes.txt")
    (repo / "notes.txt").write_bytes(b"innocent now\n")  # the work tree changed after staging
    assert paths(gate.check(repo, root)) == {"notes.txt"}


def test_committed_copy_is_caught(repo: Path, root: Path) -> None:
    (repo / "docs").mkdir()
    (repo / "docs" / "sample.txt").write_bytes(OTHER_VENDOR)
    git(repo, "add", "docs/sample.txt")
    git(repo, "commit", "-q", "-m", "oops")
    assert paths(gate.check(repo, root)) == {"docs/sample.txt"}


def test_modified_copy_is_not_flagged(repo: Path, root: Path) -> None:
    (repo / "notes.txt").write_bytes(FAKE_VENDOR + b"2031-03-05,13.00,14.00,12.00,13.50\n")
    (repo / "head.txt").write_bytes(FAKE_VENDOR[:-1])  # one byte short
    git(repo, "add", "notes.txt", "head.txt")
    assert gate.check(repo, root).offences == []


def test_unstaged_untracked_copy_is_not_flagged(repo: Path, root: Path) -> None:
    (repo / "notes.txt").write_bytes(FAKE_VENDOR)
    report = gate.check(repo, root)
    assert report.offences == [] and report.checked == 1


def test_synthetic_store_is_not_vendor_data(repo: Path, root: Path) -> None:
    (repo / "fake.txt").write_bytes(b"DATE,CLOSE\n2031-05-01,1.5\n")
    git(repo, "add", "fake.txt")
    assert gate.check(repo, root).offences == []


def test_snapshot_stored_under_a_wrong_name_is_still_matched(repo: Path, root: Path) -> None:
    (root / "raw" / "cboe" / "VIX3M" / "renamed-in-store.csv").write_bytes(b"A,B\n1,2\n3,4\n")
    (repo / "x.txt").write_bytes(b"A,B\n1,2\n3,4\n")
    git(repo, "add", "x.txt")
    assert paths(gate.check(repo, root)) == {"x.txt"}


def test_empty_file_is_not_a_match_for_an_empty_snapshot(repo: Path, root: Path) -> None:
    (root / "raw" / "cboe" / "VIX" / "empty.csv").write_bytes(b"")
    (repo / "__init__.py").write_bytes(b"")
    git(repo, "add", "__init__.py")
    assert gate.check(repo, root).offences == []


def test_size_limits(repo: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "FAIL_BYTES", 1000)
    monkeypatch.setattr(gate, "WARN_BYTES", 100)
    (repo / "at_fail.bin").write_bytes(b"\0" * 1000)  # exactly the limit passes
    (repo / "over_fail.bin").write_bytes(b"\0" * 1001)
    (repo / "at_warn.bin").write_bytes(b"\1" * 100)  # exactly the warn limit is quiet
    (repo / "over_warn.bin").write_bytes(b"\2" * 101)
    git(repo, "add", ".")
    report = gate.check(repo, root)
    assert paths(report) == {"over_fail.bin"}
    assert {w.path for w in report.warnings} == {"over_warn.bin"} | {"at_fail.bin"}
    assert report.oversize == 1


def test_real_limits_are_50_and_10_megabytes() -> None:
    assert (gate.FAIL_BYTES, gate.WARN_BYTES) == (50 * 1024 * 1024, 10 * 1024 * 1024)


def test_committed_oversize_file_is_caught(repo: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "FAIL_BYTES", 500)
    (repo / "big.bin").write_bytes(b"\0" * 501)
    git(repo, "add", "big.bin")
    git(repo, "commit", "-q", "-m", "big")
    assert paths(gate.check(repo, root)) == {"big.bin"}


def test_force_added_ignored_extension_is_caught(repo: Path, root: Path) -> None:
    (repo / ".gitignore").write_text("*.csv\n!tests/fixtures/**/*.csv\n")
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "ok.csv").write_text("a,b\n1,2\n")
    (repo / "leak.csv").write_text("DATE,CLOSE\n2031-06-01,9.99\n")
    git(repo, "add", ".gitignore", "tests/fixtures/ok.csv")
    git(repo, "add", "-f", "leak.csv")
    assert paths(gate.check(repo, root)) == {"leak.csv"}


def test_no_raw_store_still_checks_size(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "FAIL_BYTES", 10)
    (repo / "big.bin").write_bytes(b"\0" * 11)
    git(repo, "add", "big.bin")
    report = gate.check(repo, tmp_path / "no-such-root")
    assert report.raw_snapshots == 0 and paths(report) == {"big.bin"}


def test_main_exit_codes_and_output(repo: Path, root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--repo", str(repo), "--root", str(root)]
    assert gate.main(argv) == 0
    assert capsys.readouterr().out.strip() == (
        "raw-file gate: 0 raw vendor files tracked or staged, 0 files over 50 MB (1 files checked)"
    )
    (repo / "copy.dat").write_bytes(FAKE_VENDOR)
    git(repo, "add", "copy.dat")
    assert gate.main(argv) == 1
    out = capsys.readouterr().out
    assert "copy.dat" in out and hashlib.sha256(FAKE_VENDOR).hexdigest() in out and "FAILED" in out


def test_main_without_raw_store_exits_zero_and_says_so(repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.main(["--repo", str(repo), "--root", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    assert "no raw snapshots" in out and "raw-file gate: 0 raw vendor files" in out
