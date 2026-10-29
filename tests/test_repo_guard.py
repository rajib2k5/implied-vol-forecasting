"""The data pipeline refuses a data root inside the repository (ivf.repo_guard; amended M1, review A).

Review A (N1): ``VOL_DATA_DIR=<repo>/localdata`` let fetch, build-interim and eda write raw
snapshots, raw hashes and every EDA number into untracked, unignored files of the checkout. The
export tool had a guard; the pipeline had none. Both now call the same function, which compares
folders by device and inode (export tests: tests/test_export.py), and refuse with exit 3 and a
message that starts "error: refusing to use".
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ivf import config, protocol, repo_guard
from ivf.data import build_interim, check, cli, fetch, process, raw, synthetic

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC = protocol.load_protocol(protocol.protocol_path(synthetic=True))  # read before fake_repo hides configs/


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A stand-in checkout with no .git entry, so only folder identity can give it away."""
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    monkeypatch.setattr(config, "repo_root", lambda: root)
    return root


def refused(capsys) -> str:
    err = capsys.readouterr().err
    assert err.startswith("error: refusing to use "), err
    assert "must stay outside the repository" in err and "ADR 0001 D2" in err
    return err


@pytest.mark.parametrize("command", [["fetch"], ["build-interim"], ["verify"], ["check"], ["check", "--freeze-data-end"], ["eda"]])
@pytest.mark.parametrize("synthetic_flag", [[], ["--synthetic"]])
def test_every_command_refuses_a_data_root_inside_the_repo_with_exit_3(fake_repo, monkeypatch, capsys, command, synthetic_flag):
    monkeypatch.setenv(config.ENV_VAR, str(fake_repo / "data"))
    assert cli.main([*synthetic_flag, *command]) == repo_guard.EXIT_REFUSED == 3
    refused(capsys)
    assert not (fake_repo / "data" / config.PROJECT_DIRNAME).exists()  # nothing was created


def test_a_symlink_from_outside_into_the_repo_is_refused(fake_repo, tmp_path, monkeypatch, capsys):
    link = tmp_path / "innocent-looking"
    link.symlink_to(fake_repo / "data", target_is_directory=True)
    monkeypatch.setenv(config.ENV_VAR, str(link))
    assert cli.main(["--synthetic", "fetch"]) == 3
    assert str(fake_repo / "data") in refused(capsys)  # resolved: the message names the real folder
    assert list((fake_repo / "data").iterdir()) == []


def test_a_second_name_that_resolve_cannot_see_is_refused_by_device_and_inode(fake_repo, tmp_path, monkeypatch):
    # A bind mount or a case variant: resolve() leaves the text alone. Simulated with a symlink and
    # resolve() reduced to "make absolute", so only the (device, inode) comparison can connect them.
    alias = tmp_path / "alias"
    alias.symlink_to(fake_repo, target_is_directory=True)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "resolve", lambda self, strict=False: Path(os.path.abspath(self)))
        with pytest.raises(repo_guard.RepoWriteRefused, match="same folder as one inside it"):
            repo_guard.require_data_root_outside_repo(alias / "data" / config.PROJECT_DIRNAME)


def test_another_checkout_is_refused_by_its_git_entry(fake_repo, tmp_path):
    clone = tmp_path / "other-clone"
    (clone / ".git").mkdir(parents=True)
    with pytest.raises(repo_guard.RepoWriteRefused, match=r"\.git entry"):
        repo_guard.require_data_root_outside_repo(clone / "data" / config.PROJECT_DIRNAME)


def test_the_real_checkout_is_refused_and_nothing_is_written(monkeypatch, capsys):
    target = REPO / "localdata-review-a-probe"
    assert not target.exists()
    monkeypatch.setenv(config.ENV_VAR, str(target))
    assert cli.main(["--synthetic", "fetch"]) == 3
    refused(capsys)
    assert not target.exists()


def test_a_root_outside_the_repo_is_accepted(fake_repo, tmp_path):
    assert repo_guard.require_data_root_outside_repo(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()


def test_the_library_writers_refuse_too_not_only_the_command_line(fake_repo):
    # A notebook or a script can call these directly; each checks before its first write.
    root = fake_repo / "data" / config.PROJECT_DIRNAME
    layout = config.DataLayout(root, root / "raw", root / "interim", root / "processed", root / "app-data" / "real", root / "eda")
    calls = [
        lambda: fetch.fetch_one(root, fetch.Request("cboe", "VIX", "https://x"), max_age=timedelta(0), force=True, network=lambda r: None),
        lambda: synthetic.write_synthetic_raw(root, SYNTHETIC),
        lambda: build_interim.build_interim(root, ["VIX"], ["^GSPC"]),
        lambda: check.run_check(layout, synthetic=True, freeze=True, now=datetime.now(timezone.utc)),
        lambda: process.run_eda(layout, synthetic=True),
    ]
    for call in calls:
        with pytest.raises(repo_guard.RepoWriteRefused):
            call()
    assert list((fake_repo / "data").iterdir()) == []


# --- symlinked subfolders and the write-time check (amended M1, review B) ---------------------------
# Review B: with ROOT outside the repo but ROOT/raw (or ROOT/eda) a symlink into the checkout, fetch
# wrote 8 snapshots and eda wrote eda_report.json into it, rc 0: the guard looked at ROOT only.


def outside_root(tmp_path) -> Path:
    root = tmp_path / "outside" / config.PROJECT_DIRNAME / "synthetic"
    root.mkdir(parents=True)
    return root


@pytest.mark.parametrize("sub, command", [("raw", "fetch"), ("eda", "eda"), ("interim", "build-interim"), ("processed", "eda")])
def test_a_data_subfolder_symlinked_into_the_repo_exits_3_and_writes_nothing(fake_repo, tmp_path, monkeypatch, capsys, sub, command):
    root = outside_root(tmp_path)
    target = fake_repo / "data" / f"{sub}-target"
    target.mkdir()
    (root / sub).symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "outside"))
    assert cli.main(["--synthetic", command]) == 3
    refused(capsys)
    assert list(target.iterdir()) == []  # nothing in the repo
    assert list(root.iterdir()) == [root / sub]  # and nothing anywhere else: refused before the first write


@pytest.mark.parametrize("sub", ["raw", "eda"])
def test_the_real_checkout_behind_a_symlinked_subfolder_is_refused_and_nothing_is_created(tmp_path, monkeypatch, capsys, sub):
    # A dangling link into the real checkout: resolve() follows it to a folder that does not exist yet,
    # whose ancestor is the repo. Nothing may create that folder.
    target = REPO / f"localdata-review-b-probe-{sub}"
    assert not target.exists()
    root = outside_root(tmp_path)
    (root / sub).symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "outside"))
    assert cli.main(["--synthetic", "fetch" if sub == "raw" else "eda"]) == 3
    refused(capsys)
    assert not target.exists()


def test_a_deeper_symlink_is_caught_at_write_time_and_the_command_exits_3(fake_repo, tmp_path, monkeypatch, capsys):
    # ROOT/raw is a real folder outside the repo, so the first check passes; ROOT/raw/synthetic is a
    # link into the repo. Only the check write_file makes when it writes into that folder sees it.
    root = outside_root(tmp_path)
    target = fake_repo / "data" / "deep-target"
    target.mkdir()
    (root / "raw").mkdir()
    (root / "raw" / "synthetic").symlink_to(target, target_is_directory=True)
    repo_guard.require_data_root_outside_repo(root)  # passes: the link is below the folders it checks
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "outside"))
    shutil.copytree(REPO / "configs", fake_repo / "configs")  # the stand-in checkout needs the registries
    assert cli.main(["--synthetic", "fetch"]) == 3
    assert "refusing to use" in capsys.readouterr().err
    assert list(target.iterdir()) == []


def _writers(folder: Path):
    import pandas as pd
    import pyarrow as pa

    record = raw.FetchRecord("0" * 32, "cboe", "VIX", "u", None, "2026-10-29T12:00:00.000000Z", 200, 1, "a" * 64, "raw/cboe/VIX/" + "a" * 64 + ".csv", True, "t")
    return {
        "write_file": lambda: repo_guard.write_file(folder / "x.bin", b"x", outside_repo=True),
        "raw snapshot": lambda: raw.store_snapshot(folder.parent.parent.parent, "cboe", "VIX", b"x\n"),
        "provenance log": lambda: raw.append_provenance(folder.parent, record),
        "interim parquet": lambda: build_interim._write_parquet(pa.table({"a": [1]}), folder / "t.parquet"),
        "processed parquet": lambda: process._write(pd.DataFrame({"a": [1]}), folder / "t.parquet"),
        "real export": lambda: export_write_files({"a.json": b"{}"}, folder),
    }


def export_write_files(files, folder):
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    import export_app_data

    export_app_data.write_files(files, folder)  # outside_repo defaults to True, as --mode real passes


@pytest.mark.parametrize("writer", ["write_file", "raw snapshot", "provenance log", "interim parquet", "processed parquet", "real export"])
def test_every_writer_checks_the_folder_it_writes_into_at_write_time(fake_repo, tmp_path, writer):
    # The folder each writer is handed is a link into the repo; no command-level check runs first.
    target = fake_repo / "data" / "target"
    (target / "raw").mkdir(parents=True)
    root = tmp_path / "root"
    (root / "raw" / "cboe").mkdir(parents=True)
    (root / "raw" / "cboe" / "VIX").symlink_to(target / "raw", target_is_directory=True)
    folder = root / "raw" / "cboe" / "VIX"
    if writer == "provenance log":  # the log lives in raw/ itself: link raw/ instead
        root = tmp_path / "root2"
        root.mkdir()
        (root / "raw").symlink_to(target / "raw", target_is_directory=True)
        folder = root / "raw" / "cboe"
    with pytest.raises(repo_guard.RepoWriteRefused):
        _writers(folder)[writer]()
    assert list((target / "raw").iterdir()) == []


def test_a_folder_swapped_for_a_link_after_the_check_is_caught_before_the_write(fake_repo, tmp_path, monkeypatch):
    # The check passes on a folder outside the repo; then, before the write, the folder becomes a
    # link into the repo (simulated inside makedirs, the step between the check and the open).
    target = fake_repo / "data" / "target"
    target.mkdir()
    folder = tmp_path / "out" / "dir"
    real_makedirs = os.makedirs

    def swap(path, exist_ok=False):
        real_makedirs(Path(path).parent, exist_ok=True)
        Path(path).symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(repo_guard.os, "makedirs", swap)
    with pytest.raises(repo_guard.RepoWriteRefused, match="changed while it was being checked"):
        repo_guard.write_file(folder / "x.bin", b"x", outside_repo=True)
    assert list(target.iterdir()) == []


def test_appending_to_a_log_that_is_a_symlink_is_refused(tmp_path):
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_bytes(b"precious\n")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / raw.PROVENANCE).symlink_to(elsewhere)
    with pytest.raises(repo_guard.RepoWriteRefused, match="is a symlink"):
        repo_guard.write_file(tmp_path / "raw" / raw.PROVENANCE, b"line\n", outside_repo=True, how="append")
    assert elsewhere.read_bytes() == b"precious\n"


def test_appending_to_a_log_hard_linked_into_the_repo_is_refused_and_the_repo_file_is_unchanged(fake_repo, tmp_path):
    # M1 final check: O_NOFOLLOW stops a symlink but not a hard link. raw/provenance.jsonl
    # hard-linked to a file in the checkout got 13 lines appended. The fd is now checked for a
    # second name before anything is written.
    in_repo = fake_repo / "data" / "precious.jsonl"
    in_repo.write_bytes(b"precious\n")
    root = outside_root(tmp_path)
    (root / "raw").mkdir()
    log = root / "raw" / raw.PROVENANCE
    os.link(in_repo, log)
    with pytest.raises(repo_guard.RepoWriteRefused, match="single name"):
        repo_guard.write_file(log, b"line\n", outside_repo=True, how="append")
    assert in_repo.read_bytes() == b"precious\n"
    assert repo_guard.EXIT_REFUSED == 3


def test_fetch_with_a_provenance_log_hard_linked_into_the_repo_exits_3(fake_repo, tmp_path, monkeypatch, capsys):
    # The reviewer's case end to end: the synthetic fetch reaches the append and must stop with exit 3.
    in_repo = fake_repo / "data" / "precious.jsonl"
    in_repo.write_bytes(b"precious\n")
    root = outside_root(tmp_path)
    (root / "raw").mkdir()
    os.link(in_repo, root / "raw" / raw.PROVENANCE)
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "outside"))
    shutil.copytree(REPO / "configs", fake_repo / "configs")  # the stand-in checkout needs the registries
    assert cli.main(["--synthetic", "fetch"]) == 3
    assert "single name" in capsys.readouterr().err
    assert in_repo.read_bytes() == b"precious\n"


@pytest.mark.parametrize("how", ["replace", "new"])
def test_replace_and_new_never_write_through_a_hard_link(fake_repo, tmp_path, how):
    # Both modes write a fresh temporary file and rename or link it into place, so a hard-linked
    # name is swapped out ("replace") or left alone ("new"), and the repo file keeps its bytes.
    in_repo = fake_repo / "data" / "precious.bin"
    in_repo.write_bytes(b"precious\n")
    root = outside_root(tmp_path)
    os.link(in_repo, root / "target.bin")
    written = repo_guard.write_file(root / "target.bin", b"new\n", outside_repo=True, how=how)
    assert written is (how == "replace")
    assert in_repo.read_bytes() == b"precious\n"


def test_write_file_flushes_the_folder_after_the_new_name_appears(tmp_path, monkeypatch):
    # Durability (amended M1, review B): a rename or link is safe from a crash only once the folder
    # holding the name is flushed. Spy on os.fsync: one call must be on the target folder itself, and
    # at that moment the file must already have its final name.
    seen = []
    real_fsync = os.fsync

    def spy(fd):
        info = os.fstat(fd)
        seen.append((stat_is_dir(info), (info.st_dev, info.st_ino), sorted(p.name for p in tmp_path.iterdir())))
        return real_fsync(fd)

    monkeypatch.setattr(repo_guard.os, "fsync", spy)
    for how, name in (("new", "a.csv"), ("replace", "b.json"), ("append", "c.jsonl")):
        seen.clear()
        assert repo_guard.write_file(tmp_path / name, b"x\n", outside_repo=True, how=how)
        folder_syncs = [names for is_dir, ident, names in seen if is_dir and ident == repo_guard.identity(tmp_path)]
        assert folder_syncs and name in folder_syncs[-1], (how, seen)


def stat_is_dir(info) -> bool:
    import stat

    return stat.S_ISDIR(info.st_mode)


def test_a_link_to_a_folder_not_yet_in_the_repo_is_refused_before_anything_is_created(fake_repo, tmp_path):
    # The link dangles: its target would be a new folder of the repo. The check runs before makedirs,
    # so the folder is never created (a check only after creating it would leave it behind).
    target = fake_repo / "data" / "not-yet"
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(repo_guard.RepoWriteRefused, match="refusing to use"):
        repo_guard.write_file(tmp_path / "link" / "deeper" / "x.bin", b"x", outside_repo=True)
    assert not target.exists()


def test_a_write_time_refusal_in_the_real_fetch_loop_exits_3_not_5(fake_repo, tmp_path, monkeypatch, capsys):
    # The fetch loop turns a failed download into exit 5 and carries on with the next symbol. A refusal
    # is not a failed download: it must stop the command with exit 3 before a second symbol is tried.
    shutil.copytree(REPO / "configs", fake_repo / "configs")
    root = tmp_path / "outside" / config.PROJECT_DIRNAME
    target = fake_repo / "data" / "cboe-target"
    target.mkdir()
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "cboe").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path / "outside"))
    calls = []

    def network(request):
        calls.append(request.symbol)
        return fetch.Response(b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,1,1,1,1\n", None, request.url, 200, "fake")

    monkeypatch.setattr(fetch, "network_get", network)
    assert cli.main(["fetch"]) == 3
    assert "refusing to use" in capsys.readouterr().err
    assert len(calls) == 1 and list(target.iterdir()) == []
