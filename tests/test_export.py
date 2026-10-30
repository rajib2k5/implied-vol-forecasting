"""Tests for tools/export_app_data.py (ADR 0001 D1, D2, D5).

Expected values are literals worked out by hand or by a tool outside Python,
never recomputed by the code under test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ivf import contract, repo_guard

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import export_app_data  # noqa: E402
from app_store_fixture import tiny_store, write_store  # noqa: E402

#: The exporter's own store loader, kept before the autouse fixture below replaces it.
REAL_STORE_LOADER = export_app_data.load_synthetic_store
RESULT_FILES = ["eda.json", "equity_curves.json", "evaluation.json", "manifest.json", "model_race.json"]


@pytest.fixture(scope="module")
def disk_store(tmp_path_factory):
    """$VOL_DATA_DIR holding the hand-made store on disk, for tests that run the exporter in a subprocess.

    Polish review: these tests used to read the local store and were skipped without one; now they
    never depend on local data and never skip."""
    vol_dir = tmp_path_factory.mktemp("vol")
    write_store(vol_dir)
    return vol_dir


@pytest.fixture(autouse=True)
def hand_made_store(monkeypatch):
    """In-process exports read the hand-made store of tests/app_store_fixture.py, never local data."""
    monkeypatch.setattr(export_app_data, "load_synthetic_store", lambda: tiny_store())

GIT_A = contract.GitState(commit="1" * 40, dirty=False)
GIT_B = contract.GitState(commit="2" * 40, dirty=True)
TIME_A = datetime(2026, 10, 28, 9, 30, 0, tzinfo=timezone.utc)
TIME_B = datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
COMMITTED = REPO / "app" / "public" / "data" / "synthetic"


def run_cli(*args, cwd=None, env=None):
    return subprocess.run(
        [sys.executable, str(TOOLS / "export_app_data.py"), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_main(capsys, *args):
    """The command line in-process, so it reads the hand-made store; same fields as run_cli's result."""
    code = export_app_data.main(list(args))
    out, err = capsys.readouterr()
    return subprocess.CompletedProcess(args=list(args), returncode=code, stdout=out, stderr=err)


def parsed(blob: bytes):
    return json.loads(blob)


@pytest.fixture
def exported(tmp_path):
    """A scratch folder with a fresh synthetic export written into it."""
    files = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    for name, blob in files.items():
        (tmp_path / name).write_bytes(blob)
    return tmp_path


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A stand-in repository root with no .git entry, so only the folder's identity can give it away."""
    root = tmp_path / "proj"
    (root / "app" / "public" / "data").mkdir(parents=True)
    monkeypatch.setattr(export_app_data.config, "repo_root", lambda: root)
    return root


# --- the manifest -------------------------------------------------------------


def test_synthetic_export_is_the_manifest_and_four_result_files_that_validate():
    files = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    assert sorted(files) == RESULT_FILES
    for name, blob in files.items():
        result = parsed(blob)
        contract.validate_document(result)
        assert (result["dataset"], result["data_mode"], result["schema_version"]) == (name[:-5], "synthetic", "1.1.0")
    document = parsed(files["manifest.json"])
    contract.validate_document(document)  # against manifest.schema.json, $refs resolved locally
    assert document["dataset"] == "manifest"
    assert document["data_mode"] == "synthetic"
    assert document["source_commit"] == "1" * 40
    assert document["generated_at"] == "2026-10-28T09:30:00Z"


def test_manifest_lists_each_result_file_with_its_hash_size_and_class():
    # data-model.md 5.1: a file appears in `files` once it exists, with the sha256 and size of its bytes.
    files = export_app_data.generate("synthetic")
    payload = parsed(files["manifest.json"])["payload"]
    assert [(f["dataset"], f["path"], f["license_class"]) for f in payload["files"]] == [
        ("model_race", "model_race.json", "series"), ("evaluation", "evaluation.json", "aggregate"),
        ("equity_curves", "equity_curves.json", "series"), ("eda", "eda.json", "aggregate")]
    for entry in payload["files"]:
        blob = files[entry["path"]]
        assert (entry["sha256"], entry["bytes"]) == (hashlib.sha256(blob).hexdigest(), len(blob))
    # The display registry: milestone <= 4 and not retired, in registry order (old_model and future_model are left out).
    assert [(m["model_id"], m["targets"]) for m in payload["models"]] == [
        ("rw", ["vix_level", "rv21"]), ("har_log", ["vix_level"]), ("garch", ["rv21"])]
    assert payload["notice"].startswith("Synthetic data (seed 7), not market data.") and "abcdef012345" not in payload["notice"]
    assert "holdout has not been looked at" in payload["notice"]
    assert [t["target_id"] for t in payload["targets"]] == ["vix_level", "rv21"]
    assert [t["horizons"] for t in payload["targets"]] == [[1, 5, 21], [21]]


def test_protocol_hash_matches_a_hash_computed_outside_python():
    # protocol_hash = SHA-256 of the sorted-key compact JSON of the protocol dict:
    #   printf '%s' '{"data_end":"2026-10-26","dev_start":"2000-01-03","fold_len":63,
    #     "holdout_looks":0,"holdout_start":"2024-07-01","horizons":[1,5,21],"min_cell_n":20,
    #     "train_start":"1990-01-02","vix_regime_edges":[15,25]}' | sha256sum
    payload = parsed(export_app_data.generate("synthetic", git=GIT_A)["manifest.json"])["payload"]
    assert payload["protocol"]["protocol_hash"] == "03fe7772aa6b34d6e244fd35d4340887bda0a3474f0d2b11821bea2c19bb47c1"
    assert payload["protocol"]["holdout_start"] == "2024-07-01"  # ADR 0001 4.6 proposal (O1)


def run_id_for(git):
    return parsed(export_app_data.generate("synthetic", git=git)["manifest.json"])["payload"]["run_id"]


def test_run_id_hashes_protocol_commit_dirty_mode_and_inputs():
    # ADR 0001 D3: run_id = first 12 hex characters of SHA-256 of the sorted-key JSON of
    # {protocol_hash, source_commit, source_dirty, data_mode, sorted raw sha256s}.
    # With no input files yet, for GIT_A (commit 1111..., clean) the text is
    #   {"data_mode":"synthetic","protocol_hash":"03fe7772...47c1","raw_sha256s":[],
    #    "source_commit":"1111111111111111111111111111111111111111","source_dirty":false}
    # and the shell's `printf '%s' <text> | sha256sum | cut -c1-12` gives fbe9bc2a1017.
    assert run_id_for(GIT_A) == "fbe9bc2a1017"
    # GIT_B (commit 2222..., dirty): the same recipe gives 807b96c9b5dd.
    assert run_id_for(GIT_B) == "807b96c9b5dd"
    # Same commit as GIT_A but dirty: 31c8d4c13221. So the dirty flag is part of the hash too.
    assert run_id_for(contract.GitState(commit="1" * 40, dirty=True)) == "31c8d4c13221"
    # No commits yet (null id, dirty): 886e7eacc485.
    assert run_id_for(contract.GitState(commit=contract.NO_COMMIT, dirty=True)) == "886e7eacc485"


def test_run_id_and_envelope_use_the_same_git_state():
    document = parsed(export_app_data.generate("synthetic", git=GIT_B)["manifest.json"])
    assert document["source_commit"] == "2" * 40
    assert document["payload"]["run_id"] == "807b96c9b5dd"


def test_content_does_not_depend_on_git_state_or_time():
    # Only the provenance fields may change between exports: the three envelope
    # fields, the manifest's run_id (which hashes two of them) and, in the manifest's
    # file list, each file's sha256 and size (they hash bytes that hold the envelope
    # fields). Everything else must be identical, or the committed fixture would go
    # stale with every commit.
    first = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    second = export_app_data.generate("synthetic", git=GIT_B, generated_at=TIME_B)
    a, b = parsed(first["manifest.json"]), parsed(second["manifest.json"])
    assert a != b
    assert a["payload"]["run_id"] != b["payload"]["run_id"]
    for field in export_app_data.ENVELOPE_PROVENANCE:
        a.pop(field)
        b.pop(field)
    for field in export_app_data.PAYLOAD_PROVENANCE:
        a["payload"].pop(field, None)
        b["payload"].pop(field, None)
    for doc in (a, b):
        doc["payload"]["files"] = [{k: v for k, v in f.items() if k not in ("sha256", "bytes")} for f in doc["payload"]["files"]]
    assert a == b
    for name in (n for n in RESULT_FILES if n != "manifest.json"):
        x, y = parsed(first[name]), parsed(second[name])
        for field in ("generated_at", "source_commit", "source_dirty"):
            x.pop(field)
            y.pop(field)
        assert x == y, name


def test_real_mode_without_the_pipeline_output_is_unavailable_and_says_which_step(monkeypatch, tmp_path):
    # Milestone 1: real mode exists, but needs ROOT/interim/ (python -m ivf.data build-interim).
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    with pytest.raises(export_app_data.ModeUnavailable, match="build-interim"):
        export_app_data.generate("real", git=GIT_A, generated_at=TIME_A)


def test_generate_refuses_to_produce_a_manifest_that_breaks_the_schema(monkeypatch):
    # The schema requires at least one target. An exporter that validates
    # before writing must stop here instead of handing the app a bad file.
    monkeypatch.setattr(export_app_data, "M0_TARGETS", [])
    with pytest.raises(contract.SchemaViolation) as caught:
        export_app_data.generate("synthetic")
    assert [e.json_path for e in caught.value.errors] == ["$.payload.targets"]


# --- writing and --check ------------------------------------------------------


def test_cli_writes_the_files_and_check_passes_on_them(tmp_path, capsys):
    out = tmp_path / "fixture"
    written = run_main(capsys, "--mode", "synthetic", "--out-dir", str(out))
    assert written.returncode == 0, written.stderr
    assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
    contract.validate_document(json.loads((out / "manifest.json").read_bytes()))
    checked = run_main(capsys, "--mode", "synthetic", "--check", "--out-dir", str(out))
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.startswith("ok: 5 file(s)")


def test_check_mode_never_writes(tmp_path, capsys):
    out = tmp_path / "not-created"
    result = run_main(capsys, "--mode", "synthetic", "--check", "--out-dir", str(out))
    assert result.returncode == 1
    assert "missing:" in result.stderr
    assert not out.exists()


def test_cli_subprocess_writes_from_a_store_on_disk_and_check_passes(disk_store, tmp_path):
    # The gate's command line end to end, in a subprocess, against a store on disk.
    env = {**os.environ, "VOL_DATA_DIR": str(disk_store)}
    out = tmp_path / "fixture"
    written = run_cli("--mode", "synthetic", "--out-dir", str(out), cwd=REPO.parent, env=env)
    assert written.returncode == 0, written.stderr
    assert sorted(p.name for p in out.iterdir()) == RESULT_FILES
    checked = run_cli("--mode", "synthetic", "--check", "--out-dir", str(out), cwd=REPO.parent, env=env)
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.startswith("ok: 5 file(s)")


def test_the_committed_fixture_is_synthetic_valid_and_its_manifest_matches_its_files():
    # Without any store: what can be checked of the committed files alone. (That they equal a fresh
    # export is the gate `export_app_data.py --mode synthetic --check`, which needs the local store.)
    names = sorted(p.name for p in COMMITTED.glob("*.json"))
    assert names == RESULT_FILES
    for name in names:
        document = parsed((COMMITTED / name).read_bytes())
        contract.validate_document(document)
        assert document["data_mode"] == "synthetic"
    assert export_app_data._manifest_mismatches(parsed((COMMITTED / "manifest.json").read_bytes()), COMMITTED) == []


def test_check_ignores_the_store_run_id(exported):
    # source_run_id hashes the commit that built the store, so a rebuild at a clean commit changes it.
    for name in ("evaluation.json", "eda.json"):
        path = exported / name
        document = parsed(path.read_bytes())
        document["payload"]["source_run_id"] = "ffffffffffff"
        path.write_bytes(contract.serialise(document))
    problems = export_app_data.check(export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A), exported)
    # Only the manifest's hashes of the two edited files can differ; the content of each file matches.
    assert problems == [f"differs: {exported / 'manifest.json'} lists a sha256 or size for {n} that the file does not have"
                        for n in ("evaluation.json", "eda.json")]
    notice = parsed((exported / "manifest.json").read_bytes())["payload"]["notice"]
    assert "abcdef012345" not in notice  # the store run id is not pinned in the notice


def test_synthetic_mode_without_a_store_is_unavailable_and_names_the_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(export_app_data, "load_synthetic_store", REAL_STORE_LOADER)
    with pytest.raises(export_app_data.ModeUnavailable, match="synthetic dev store.*uv run python tools/build_synthetic_store.py"):
        export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    assert export_app_data.main(["--mode", "synthetic", "--check"]) == export_app_data.EXIT_UNAVAILABLE == 4


def test_check_detects_a_changed_result_value(exported):
    path = exported / "evaluation.json"
    document = parsed(path.read_bytes())
    document["payload"]["losses"][1]["rmse_ratio_rw"] = 0.6  # the fixture's value is 0.5
    path.write_bytes(contract.serialise(document))
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    # The content difference, and the manifest's hash of the file no longer matching its bytes.
    assert problems == [f"differs: {path} at $.payload.losses[1].rmse_ratio_rw: 0.6 vs 0.5",
                        f"differs: {exported / 'manifest.json'} lists a sha256 or size for evaluation.json that the file does not have"]


def test_check_ignores_all_provenance_fields(exported):
    # generated_at, source_commit, source_dirty and run_id all differ from a fresh export.
    path = exported / "manifest.json"
    document = parsed(path.read_bytes())
    document.update(generated_at="2020-01-01T00:00:00Z", source_commit="f" * 40, source_dirty=True)
    document["payload"]["run_id"] = "000000000000"
    path.write_bytes(contract.serialise(document))
    fresh = export_app_data.generate("synthetic", git=GIT_B, generated_at=TIME_B)
    assert export_app_data.check(fresh, exported) == []


def test_check_ignores_a_different_run_id_on_its_own(exported):
    # run_id is derived from source_commit and source_dirty. The fixture written under
    # GIT_A is checked against a fresh export under GIT_B: only run_id, source_commit
    # and source_dirty differ, and none of them may fail the check.
    fresh = export_app_data.generate("synthetic", git=GIT_B, generated_at=TIME_A)
    assert parsed(fresh["manifest.json"])["payload"]["run_id"] != parsed((exported / "manifest.json").read_bytes())["payload"]["run_id"]
    assert export_app_data.check(fresh, exported) == []


def test_check_still_requires_run_id_to_be_12_hex(exported):
    # Ignored in the comparison, but the committed file must still satisfy the schema.
    path = exported / "manifest.json"
    document = parsed(path.read_bytes())
    document["payload"]["run_id"] = "not-a-run-id"
    path.write_bytes(contract.serialise(document))
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    assert len(problems) == 1 and problems[0].startswith("invalid:") and "run_id" in problems[0]


def test_check_detects_a_changed_payload_value_and_says_where(exported):
    path = exported / "manifest.json"
    document = parsed(path.read_bytes())
    document["payload"]["protocol"]["fold_len"] = 21
    path.write_bytes(contract.serialise(document))
    fresh = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    problems = export_app_data.check(fresh, exported)
    assert len(problems) == 1
    assert problems[0].startswith("differs:") and "$.payload.protocol.fold_len: 21 vs 63" in problems[0]


def test_check_detects_an_added_field(exported):
    path = exported / "manifest.json"
    document = parsed(path.read_bytes())
    document["payload"]["files"].append(
        {"dataset": "model_race", "path": "model_race.json", "sha256": "c" * 64, "bytes": 5, "license_class": "series"}
    )
    path.write_bytes(contract.serialise(document))
    problems = export_app_data.check(export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A), exported)
    assert len(problems) == 1 and "$.payload.files has 5 items vs 4" in problems[0]


def test_check_treats_15_and_15_point_0_as_the_same_number(exported):
    path = exported / "manifest.json"
    fresh = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    path.write_bytes(path.read_bytes().replace(b"[15,25]", b"[15.0,25]"))
    assert export_app_data.check(fresh, exported) == []


def test_difference_finder_keeps_json_true_apart_from_the_number_1():
    # In Python True == 1; in JSON `true` and `1` are different values.
    first = export_app_data._first_difference
    assert first({"a": [True]}, {"a": [1]}) == "$.a[0]: True vs 1"
    assert first({"a": [15]}, {"a": [15.0]}) is None
    assert first({"a": [1, 2]}, {"a": [1, 3]}) == "$.a[1]: 2 vs 3"
    assert first({"a": 1}, {"b": 1}) == "$.a exists on one side only"


def test_check_detects_a_missing_file(exported):
    (exported / "manifest.json").unlink()
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    assert len(problems) == 1 and problems[0].startswith("missing:")


def test_check_detects_a_missing_folder(tmp_path):
    problems = export_app_data.check(export_app_data.generate("synthetic"), tmp_path / "no-such-folder")
    assert len(problems) == 5 and all(p.startswith("missing:") for p in problems)


def test_check_detects_a_file_the_exporter_no_longer_produces(exported):
    (exported / "term_structure.json").write_bytes(b"{}")
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    assert len(problems) == 1 and problems[0].startswith("unexpected:") and "term_structure.json" in problems[0]


def test_check_detects_unparseable_json(exported):
    (exported / "manifest.json").write_bytes(b"{not json")
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    assert len(problems) == 1 and problems[0].startswith("invalid:")


def test_check_detects_a_committed_file_that_breaks_the_schema_in_a_provenance_field(exported):
    # source_commit is ignored by the comparison, but it must still be 40 hex.
    path = exported / "manifest.json"
    document = parsed(path.read_bytes())
    document["source_commit"] = "not-a-commit"
    path.write_bytes(contract.serialise(document))
    problems = export_app_data.check(export_app_data.generate("synthetic"), exported)
    assert len(problems) == 1 and problems[0].startswith("invalid:") and "source_commit" in problems[0]


def test_check_command_exits_1_on_drift(exported, capsys):
    path = exported / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b'"fold_len":63', b'"fold_len":64'))
    result = run_main(capsys, "--mode", "synthetic", "--check", "--out-dir", str(exported))
    assert result.returncode == 1
    assert "differs:" in result.stderr


# --- the licensing guard (ADR 0001 D2) ---------------------------------------


@pytest.mark.parametrize(
    "inside",
    [
        REPO,
        REPO / "app" / "public" / "data" / "real",  # the gitignored second layer
        REPO / "app" / "public" / "data" / "synthetic",
        REPO / "reports",  # does not exist yet; the guard must not need it to
        REPO / "tests" / ".." / "app",  # ".." must be normalised first
        # Starts outside the repo and only ".." leads back in. Without resolve() the text
        # begins with <repo parent>/zzz, so it looks outside; resolve() cancels "zzz/..".
        REPO.parent / "zzz" / ".." / REPO.name / "app",
    ],
)
def test_real_mode_refuses_any_path_inside_the_repo(inside):
    with pytest.raises(export_app_data.RepoWriteRefused, match="outside the repository"):
        export_app_data.resolve_output_dir("real", inside)


def test_real_mode_refuses_a_symlink_that_points_into_the_repo(tmp_path):
    link = tmp_path / "innocent-looking"
    link.symlink_to(REPO / "app", target_is_directory=True)
    with pytest.raises(export_app_data.RepoWriteRefused):
        export_app_data.resolve_output_dir("real", link / "data" / "real")


@pytest.mark.parametrize(
    "inside",
    [(), ("app",), ("app", "public", "data", "real")],  # the root itself, a folder that exists, one that does not
    ids=["root", "existing folder", "folder not created yet"],
)
def test_real_mode_refuses_a_second_name_for_the_repo_by_device_and_inode(fake_repo, tmp_path, inside):
    # On a case-insensitive filesystem ".../IVF" and ".../ivf" are one directory, and so are a bind
    # mount and its source; resolve() leaves both spellings alone. A symlink is the same situation
    # one step earlier. This calls the check WITHOUT resolve(), so the alias's text says nothing
    # about the repo and only the (device, inode) pair can connect them.
    alias = tmp_path / "alias"
    alias.symlink_to(fake_repo, target_is_directory=True)
    spelled = alias.joinpath(*inside)
    assert fake_repo not in (spelled, *spelled.parents)  # a text comparison would call this outside
    reason = export_app_data._why_inside_a_checkout(spelled, export_app_data._directories_inside(fake_repo))
    assert reason is not None and "same folder as one inside" in reason


def test_real_mode_refuses_an_alias_that_resolve_does_not_normalise(fake_repo, tmp_path, monkeypatch):
    # The same through resolve_output_dir, with resolve() reduced to "make absolute", which is
    # all it does for a bind mount or a case variant.
    alias = tmp_path / "alias"
    alias.symlink_to(fake_repo, target_is_directory=True)
    with monkeypatch.context() as patched:
        patched.setattr(Path, "resolve", lambda self, strict=False: Path(os.path.abspath(self)))
        with pytest.raises(export_app_data.RepoWriteRefused, match="outside the repository"):
            export_app_data.resolve_output_dir("real", alias / "app" / "public" / "data" / "real")


def test_a_second_name_for_a_folder_below_the_repo_root_is_refused_too(fake_repo, tmp_path):
    # Review D's attack without the mount: only app/public is known (a bind mount of that folder
    # shows up elsewhere as a folder with its identity), and the alias is a symlink that is NOT
    # resolved, so its text mentions neither the repo nor app/public. The root's identity is not in
    # the set, so a check that only compared with the root would accept this.
    public = fake_repo / "app" / "public"
    alias = tmp_path / "alias"
    alias.symlink_to(public, target_is_directory=True)
    only_public = {export_app_data._identity(public)}
    reason = export_app_data._why_inside_a_checkout(alias / "data" / "real", only_public)
    assert reason is not None and "same folder as one inside" in reason
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert export_app_data._why_inside_a_checkout(elsewhere, only_public) is None


def test_directories_inside_lists_the_root_and_every_folder_below_it_but_follows_no_symlink(fake_repo, tmp_path):
    deep = fake_repo / "a" / "b" / "c"
    deep.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (fake_repo / "link-out").symlink_to(outside, target_is_directory=True)  # points out of the repo
    found = export_app_data._directories_inside(fake_repo)
    identity = export_app_data._identity
    assert {identity(fake_repo), identity(fake_repo / "app" / "public" / "data"), identity(deep)} <= found
    # The link's target is not part of the repo (git stores the link, not what it points to),
    # and neither is the folder that holds the repo.
    assert identity(outside) not in found
    assert identity(tmp_path) not in found


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read every folder, so none can be unreadable")
def test_a_folder_inside_the_repo_that_cannot_be_read_is_refused_not_skipped(fake_repo):
    # Skipping it would leave its identity out of the set, and a mount of it would go unnoticed.
    locked = fake_repo / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        with pytest.raises(export_app_data.RepoWriteRefused, match=r"cannot read .*locked"):
            export_app_data._directories_inside(fake_repo)
    finally:
        locked.chmod(0o700)


def walk_that_deletes(victim: Path):
    """An os.walk that removes ``victim`` just after listing it and before the caller can look at it."""
    real_walk = os.walk

    def walk(*args, **kwargs):
        for folder, dirs, files in real_walk(*args, **kwargs):
            if Path(folder) == victim:
                victim.rmdir()
            yield folder, dirs, files

    return walk


def test_a_folder_that_vanishes_during_the_walk_is_refused_not_a_traceback(fake_repo, monkeypatch):
    # Review E: os.walk lists a folder, something deletes it, and the stat that follows raised an
    # uncaught FileNotFoundError. A moved folder looks the same and may hold an identity the walk
    # never sees, so the guard fails closed (see _directories_inside).
    victim = fake_repo / "app" / "public" / "data"
    with monkeypatch.context() as patched:
        patched.setattr(repo_guard.os, "walk", walk_that_deletes(victim))
        with pytest.raises(export_app_data.RepoWriteRefused, match=r"cannot read .*/data .*No such file.*run it again"):
            export_app_data._directories_inside(fake_repo)


def test_cli_real_export_exits_3_with_a_message_when_a_repo_folder_vanishes_mid_walk(fake_repo, tmp_path, monkeypatch, capsys):
    victim = fake_repo / "app" / "public" / "data"
    out = tmp_path / "out"
    with monkeypatch.context() as patched:
        patched.setattr(repo_guard.os, "walk", walk_that_deletes(victim))
        assert export_app_data.main(["--mode", "real", "--out-dir", str(out)]) == export_app_data.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("error: cannot read ")
    assert not out.exists()  # nothing was written


@pytest.fixture(scope="module")
def can_bind_mount(tmp_path_factory):
    """Skip unless this process can bind-mount without being root.

    ``unshare -rm`` gives the command a private mount namespace in which it is
    root, so the mount disappears when the command ends and the host never sees it.
    """
    base = tmp_path_factory.mktemp("bind-probe")
    (base / "source").mkdir()
    (base / "target").mkdir()
    try:
        probe = subprocess.run(
            ["unshare", "-rm", "mount", "--bind", str(base / "source"), str(base / "target")],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        pytest.skip("unshare is not installed, so a real bind mount cannot be made without root")
    if probe.returncode != 0:
        pytest.skip(f"no bind mount without root here (unshare -rm mount --bind failed: {probe.stderr.strip()})")


# Runs inside the mount namespace. The stand-in repo replaces the real one, and a faked milestone 1
# generator replaces generate(), because milestone 0 has no real data to export.
BIND_MOUNT_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import export_app_data as ead
repo, out = Path(sys.argv[2]), Path(sys.argv[3])
ead.config.repo_root = lambda: repo
ead.generate = lambda mode, **kwargs: {"manifest.json": b'{"data_mode":"real"}\\n'}
print(ead.main(["--mode", "real", "--out-dir", str(out)]))
"""


def real_export_through_bind_mount(source: Path, mountpoint: Path, repo: Path, folder: Path) -> int:
    """Bind-mount ``source`` at ``mountpoint`` and run a faked real export into ``folder``. Returns main()'s exit code."""
    result = subprocess.run(
        ["unshare", "-rm", "sh", "-c", 'mount --bind "$1" "$2" && shift 2 && exec "$@"', "sh", str(source), str(mountpoint),
         sys.executable, "-c", BIND_MOUNT_SCRIPT, str(TOOLS), str(repo), str(folder)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip().splitlines()[-1])


def test_real_export_through_a_bind_mount_of_a_repo_subfolder_is_refused(can_bind_mount, fake_repo, tmp_path):
    # Review D's attack: repo/app/public mounted at an outside path. That path is not the repo
    # root and has no .git, so only the identities of the repo's own folders give it away.
    mountpoint = tmp_path / "outside"
    mountpoint.mkdir()
    code = real_export_through_bind_mount(fake_repo / "app" / "public", mountpoint, fake_repo, mountpoint / "data" / "real")
    assert code == export_app_data.EXIT_REFUSED
    assert not (fake_repo / "app" / "public" / "data" / "real").exists()  # nothing landed in the repo


def test_the_same_bind_mount_of_a_folder_outside_the_repo_is_allowed(can_bind_mount, fake_repo, tmp_path):
    # Control for the test above: identical set-up, but the mounted folder is not in the repo.
    # It proves the refusal above comes from the guard and not from the mount or the script.
    source = tmp_path / "elsewhere"
    source.mkdir()
    mountpoint = tmp_path / "outside"
    mountpoint.mkdir()
    code = real_export_through_bind_mount(source, mountpoint, fake_repo, mountpoint / "data" / "real")
    assert code == export_app_data.EXIT_OK
    assert (source / "data" / "real" / "manifest.json").read_bytes() == b'{"data_mode":"real"}\n'


@pytest.mark.parametrize("git_entry", ["folder", "file"])
@pytest.mark.parametrize("below", [(), ("out",), ("out", "not", "created", "yet")])
def test_real_mode_refuses_any_folder_inside_another_git_checkout(fake_repo, tmp_path, git_entry, below):
    # Another clone of this project is not the repo root by inode, but `git add -A` there
    # publishes just the same. A linked work tree has a .git *file* rather than a folder.
    clone = tmp_path / "other-clone"
    clone.mkdir()
    if git_entry == "folder":
        (clone / ".git").mkdir()
    else:
        (clone / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    with pytest.raises(export_app_data.RepoWriteRefused, match=r"\.git entry"):
        export_app_data.resolve_output_dir("real", clone.joinpath(*below))


def test_real_mode_accepts_a_path_outside_the_repo(tmp_path):
    assert export_app_data.resolve_output_dir("real", tmp_path / "out") == (tmp_path / "out").resolve()


def test_a_sibling_folder_sharing_the_repo_name_prefix_is_outside_the_repo(fake_repo):
    # <tmp>/proj-other is not inside <tmp>/proj, although its text starts with the same characters.
    sibling = Path(str(fake_repo) + "-other") / "app-data"
    assert export_app_data.resolve_output_dir("real", sibling) == sibling


def test_real_default_is_the_out_of_repo_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path))
    expected = (tmp_path / "implied-vol-forecasting" / "app-data" / "real").resolve()
    assert export_app_data.resolve_output_dir("real", None) == expected


def test_real_default_is_refused_when_vol_data_dir_points_into_the_repo(monkeypatch):
    # The repo's .gitignore lists /data/, so someone might set VOL_DATA_DIR=<repo>/data.
    monkeypatch.setenv("VOL_DATA_DIR", str(REPO / "data"))
    with pytest.raises(export_app_data.RepoWriteRefused):
        export_app_data.resolve_output_dir("real", None)


def test_synthetic_mode_writes_inside_the_repo_by_design():
    assert export_app_data.resolve_output_dir("synthetic", None) == COMMITTED


def test_cli_real_mode_in_repo_exits_3_and_writes_nothing():
    target = REPO / "app" / "public" / "data" / "real"
    result = run_cli("--mode", "real", "--out-dir", str(target))
    assert result.returncode == 3
    assert "outside the repository" in result.stderr
    assert not target.exists()


def test_cli_real_check_in_repo_is_refused_too():
    result = run_cli("--mode", "real", "--check", "--out-dir", str(REPO / "app"))
    assert result.returncode == 3


def test_cli_real_mode_outside_the_repo_passes_the_guard_but_exits_4_without_data(tmp_path):
    # Exit 4, not 3: the guard let the path through. Nothing is written,
    # because this data folder holds no interim tables yet.
    out = tmp_path / "real-out"
    env = dict(os.environ, VOL_DATA_DIR=str(tmp_path / "empty"))
    result = subprocess.run(
        [sys.executable, str(TOOLS / "export_app_data.py"), "--mode", "real", "--out-dir", str(out)],
        capture_output=True, text=True, env=env, check=False,
    )
    assert result.returncode == 4
    assert "build-interim" in result.stderr
    assert not out.exists()


# --- writing without following links (ADR 0001 D2) ---------------------------


def make_link(kind: str, link: Path, target: Path) -> None:
    """A second name for ``target``: a symlink or a hard link."""
    if kind == "symlink":
        link.symlink_to(target)
    else:
        os.link(target, link)


@pytest.fixture
def precious(tmp_path):
    """A file outside the export folder that no export may touch, and the empty folder to export into."""
    outside = tmp_path / "precious.json"
    outside.write_bytes(b"precious\n")
    out = tmp_path / "out"
    out.mkdir()
    return outside, out


def test_write_files_writes_the_bytes_and_leaves_no_temporary_file(tmp_path):
    (tmp_path / "a.json").write_bytes(b"old\n")
    export_app_data.write_files({"a.json": b"new a\n", "b.json": b"new b\n"}, tmp_path)
    assert (tmp_path / "a.json").read_bytes() == b"new a\n"
    assert (tmp_path / "b.json").read_bytes() == b"new b\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.json"]


def test_new_files_get_the_mode_any_new_file_gets_from_the_umask(tmp_path):
    # 0o666 with the umask's bits cleared: 0o022 clears group and other write, leaving rw-r--r-- = 0o644.
    # (A temporary file made by mkstemp would be 0o600 and stay that way after the rename.)
    old = os.umask(0o022)
    try:
        export_app_data.write_files({"a.json": b"x"}, tmp_path)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "a.json").stat().st_mode) == 0o644


@pytest.mark.parametrize("kind, complaint", [("symlink", "symlink"), ("hardlink", "2 hard links")])
def test_write_files_refuses_a_link_at_the_target_and_leaves_the_linked_file_alone(precious, kind, complaint):
    outside, out = precious
    make_link(kind, out / "a.json", outside)
    with pytest.raises(export_app_data.RepoWriteRefused, match=complaint):
        export_app_data.write_files({"a.json": b"new\n"}, out)
    assert outside.read_bytes() == b"precious\n"
    assert (out / "a.json").read_bytes() == b"precious\n"  # still the link, not a fresh file


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_a_link_that_appears_after_the_check_is_replaced_not_written_through(precious, monkeypatch, kind):
    # The refusal looks at the target a moment before the write. If a link slips in during that
    # gap, os.replace must still swap the NAME and leave the file behind the link as it was.
    # The refusal is switched off here to test that second line of defence on its own.
    monkeypatch.setattr(export_app_data, "_refuse_unsafe_target", lambda path: None)
    outside, out = precious
    make_link(kind, out / "a.json", outside)
    export_app_data.write_files({"a.json": b"new\n"}, out)
    assert outside.read_bytes() == b"precious\n"
    assert (out / "a.json").read_bytes() == b"new\n"
    assert not (out / "a.json").is_symlink()
    assert outside.stat().st_nlink == 1  # the old name is no longer a hard link of anything


def test_a_refusal_happens_before_any_file_is_written(precious):
    outside, out = precious
    (out / "b.json").symlink_to(outside)
    with pytest.raises(export_app_data.RepoWriteRefused):
        export_app_data.write_files({"a.json": b"new a\n", "b.json": b"new b\n"}, out)
    assert not (out / "a.json").exists()  # a.json comes first and would have been written by a check-as-you-go loop


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_bytes(b"old\n")

    def fail(source, destination, **dir_fds):
        raise OSError("disk on fire")

    with monkeypatch.context() as patched:
        patched.setattr(repo_guard.os, "replace", fail)  # write_file does the writing (amended M1, review B)
        with pytest.raises(OSError, match="disk on fire"):
            export_app_data.write_files({"a.json": b"new\n"}, tmp_path)
    assert (tmp_path / "a.json").read_bytes() == b"old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]


def test_a_link_planted_at_the_temporary_name_is_never_written_through(precious, monkeypatch):
    # The temporary name is random, so nobody can plant anything there. If one did guess it
    # (forced here by fixing the random part), the exclusive create must fail rather than follow it.
    outside, out = precious
    monkeypatch.setattr(repo_guard.secrets, "token_hex", lambda nbytes: "fixed")
    (out / ".a.json.fixed.tmp").symlink_to(outside)
    with pytest.raises(FileExistsError):
        export_app_data.write_files({"a.json": b"new\n"}, out)
    assert outside.read_bytes() == b"precious\n"


def test_write_files_refuses_a_directory_at_the_target_and_says_so(tmp_path):
    # A directory has 2 or more links, so it used to be refused as "it has 2 hard links".
    (tmp_path / "a.json").mkdir()
    with pytest.raises(export_app_data.RepoWriteRefused, match="it is a directory"):
        export_app_data.write_files({"a.json": b"x"}, tmp_path)
    assert (tmp_path / "a.json").is_dir()


def test_replacing_a_file_keeps_its_permission_bits(tmp_path):
    # With umask 022 a new file is 0o644. A 0o600 file widened to that on re-export, and 0o664 and
    # 0o755 were narrowed to it; the three modes differ from 0o644 so each case tells.
    modes = {"private.json": 0o600, "group.json": 0o664, "program.json": 0o755}
    for name, mode in modes.items():
        (tmp_path / name).write_bytes(b"old")
        (tmp_path / name).chmod(mode)
    old = os.umask(0o022)
    try:
        export_app_data.write_files({name: b"new" for name in modes}, tmp_path)
    finally:
        os.umask(old)
    assert {name: stat.S_IMODE((tmp_path / name).stat().st_mode) for name in modes} == modes
    assert all((tmp_path / name).read_bytes() == b"new" for name in modes)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores folder permissions, so no folder is read-only to it")
@pytest.mark.parametrize("below", [(), ("new", "folder")], ids=["read-only folder", "new folder under a read-only parent"])
def test_cli_unwritable_output_folder_exits_5_with_a_message_not_a_traceback(tmp_path, below, capsys):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o500)  # read and enter, but not write
    out = parent.joinpath(*below)
    try:
        result = run_main(capsys, "--mode", "synthetic", "--out-dir", str(out))
    finally:
        parent.chmod(0o700)
    assert result.returncode == export_app_data.EXIT_WRITE_FAILED == 5
    assert result.stderr.startswith(f"error: cannot write to {out}") and "Permission denied" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(parent.iterdir()) == []


def test_cli_output_path_that_is_a_file_exits_5(tmp_path, capsys):
    out = tmp_path / "out"
    out.write_bytes(b"i am a file, not a folder")
    result = run_main(capsys, "--mode", "synthetic", "--out-dir", str(out))
    assert result.returncode == 5
    assert result.stderr.startswith("error: cannot write to") and "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_cli_refuses_a_linked_output_file_with_exit_3(precious, kind, capsys):
    outside, out = precious
    make_link(kind, out / "manifest.json", outside)
    result = run_main(capsys, "--mode", "synthetic", "--out-dir", str(out))
    assert result.returncode == 3
    assert result.stderr.startswith("error: refusing to write")
    assert outside.read_bytes() == b"precious\n"


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_a_simulated_m1_real_export_cannot_overwrite_a_repo_file_through_a_link(fake_repo, tmp_path, monkeypatch, kind):
    # The attack from review A. The export folder is outside the repo, so the folder guard passes,
    # but manifest.json in it is a second name for the committed fixture. Milestone 0 has no real
    # generator, so one is faked: the same bytes with data_mode flipped.
    committed = fake_repo / "app" / "public" / "data" / "synthetic" / "manifest.json"
    committed.parent.mkdir(parents=True)
    committed.write_bytes(b'{"data_mode":"synthetic"}\n')
    out = tmp_path / "vol" / "app-data" / "real"
    out.mkdir(parents=True)
    make_link(kind, out / "manifest.json", committed)
    monkeypatch.setattr(export_app_data, "generate", lambda mode, **kwargs: {"manifest.json": b'{"data_mode":"real"}\n'})
    assert export_app_data.main(["--mode", "real", "--out-dir", str(out)]) == export_app_data.EXIT_REFUSED
    assert committed.read_bytes() == b'{"data_mode":"synthetic"}\n'


# --- the documented regenerate command ---------------------------------------


def test_documented_regenerate_command_rewrites_the_existing_fixture_folder(tmp_path, disk_store):
    # `uv run python tools/export_app_data.py --mode synthetic`, run from a repo whose fixture
    # folder already exists (every time after the first). The copy is a minimal stand-in repo:
    # the package, the schemas, the tool and the committed fixture. PYTHONPATH makes the copy's
    # `ivf` win over the editable install, so the command can only touch the copy.
    copy = tmp_path / "copy"
    shutil.copytree(REPO / "src" / "ivf", copy / "src" / "ivf", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "schemas", copy / "schemas")
    shutil.copytree(COMMITTED, copy / "app" / "public" / "data" / "synthetic")
    (copy / "tools").mkdir()
    shutil.copy(TOOLS / "export_app_data.py", copy / "tools" / "export_app_data.py")
    shutil.copy(REPO / "pyproject.toml", copy / "pyproject.toml")
    shutil.copytree(REPO / "configs", copy / "configs")  # the synthetic protocol, models and backtest rules
    env = {**os.environ, "PYTHONPATH": str(copy / "src"), "GIT_CEILING_DIRECTORIES": str(tmp_path), "VOL_DATA_DIR": str(disk_store)}

    def in_copy(*args):
        return subprocess.run(args, cwd=copy, env=env, capture_output=True, text=True, check=False)

    # Safety first: if the copy's package were not the one imported, the command below would
    # rewrite the real fixture. Stop before that.
    root_seen = in_copy(sys.executable, "-c", "from ivf import config; print(config.repo_root())")
    assert root_seen.stdout.strip() == str(copy.resolve()), root_seen.stderr

    fixture = copy / "app" / "public" / "data" / "synthetic" / "manifest.json"
    stale = parsed(fixture.read_bytes())
    stale.update(generated_at="2020-01-01T00:00:00Z", source_commit="f" * 40, source_dirty=False)
    fixture.write_bytes(contract.serialise(stale))

    result = in_copy(sys.executable, "tools/export_app_data.py", "--mode", "synthetic")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("wrote ")

    rewritten = parsed(fixture.read_bytes())
    assert rewritten["generated_at"] != "2020-01-01T00:00:00Z"  # really rewritten, not left alone
    assert rewritten["source_commit"] == contract.NO_COMMIT  # the copy is not a git repo
    assert rewritten["source_dirty"] is True
    assert sorted(p.name for p in fixture.parent.iterdir()) == RESULT_FILES  # no temporary file left behind
    checked = in_copy(sys.executable, "tools/export_app_data.py", "--mode", "synthetic", "--check")
    assert checked.returncode == 0, checked.stderr


# --- a $ref the schemas cannot resolve ---------------------------------------


def test_check_reports_a_schema_it_cannot_resolve_instead_of_crashing(exported, tmp_path, monkeypatch):
    # manifest.schema.json starts from "$ref": "envelope.schema.json". Point it at a file that
    # does not exist and the library raises its own exception type while the document is checked.
    broken = tmp_path / "broken-schemas"
    shutil.copytree(REPO / "schemas" / "app-data", broken)
    manifest_schema = broken / "manifest.schema.json"
    manifest_schema.write_text(manifest_schema.read_text().replace('"$ref": "envelope.schema.json",', '"$ref": "nonesuch.schema.json",', 1))
    fresh = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    monkeypatch.setattr(export_app_data.config, "schema_dir", lambda: broken)
    problems = export_app_data.check(fresh, exported)
    assert len(problems) == 1 and problems[0].startswith("invalid:") and "nonesuch.schema.json" in problems[0]


# --- real mode (milestone 1): term_structure.json from interim tables --------------------------


@pytest.fixture
def real_root(tmp_path, monkeypatch, no_network):
    """A data folder whose ROOT holds interim tables built from synthetic files, used as "real" input.

    The synthetic protocol stands in for configs/protocol.yaml so the dates match the files.
    """
    from ivf import protocol as protocol_mod
    from ivf.data import build_interim, synthetic

    monkeypatch.setenv("VOL_DATA_DIR", str(tmp_path / "vol"))
    synthetic_protocol = protocol_mod.protocol_path(synthetic=True)
    monkeypatch.setattr(protocol_mod, "protocol_path", lambda synthetic=False: synthetic_protocol)
    root = tmp_path / "vol" / "implied-vol-forecasting"
    synthetic.write_synthetic_raw(root, protocol_mod.load_protocol(synthetic_protocol))
    build_interim.build_interim(root, ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX"], ["^GSPC"])
    return root


def test_real_export_is_a_manifest_and_a_term_structure_that_validate(real_root):
    files = export_app_data.generate("real", git=GIT_A, generated_at=TIME_A)
    assert sorted(files) == ["manifest.json", "term_structure.json"]
    manifest, term = parsed(files["manifest.json"]), parsed(files["term_structure.json"])
    for document in (manifest, term):
        contract.validate_document(document)
        assert document["data_mode"] == "real"
    entry = manifest["payload"]["files"][0]
    assert entry["dataset"] == "term_structure" and entry["license_class"] == "series"
    assert entry["bytes"] == len(files["term_structure.json"])
    # sha256 of the exact bytes written next to the manifest (hashlib is fine here: the
    # manifest must agree with the file, whatever the file's content).
    import hashlib

    assert entry["sha256"] == hashlib.sha256(files["term_structure.json"]).hexdigest()
    assert manifest["payload"]["protocol"]["data_end"] == "2026-10-26"
    assert "non-commercial" in manifest["payload"]["notice"]


def test_term_structure_starts_when_every_tenor_exists_and_rounds_as_the_data_model_says(real_root):
    payload = parsed(export_app_data.generate("real", git=GIT_A, generated_at=TIME_A)["term_structure.json"])["payload"]
    assert payload["dates"][0] == "2011-01-04"  # VIX9D, the last tenor to start
    assert payload["dates"][-1] == "2026-10-26"
    assert [t["label"] for t in payload["tenors"]] == ["9D", "30D", "3M", "6M", "1Y"]
    assert [t["tenor_days"] for t in payload["tenors"]] == [9, 30, 93, 184, 366]
    n = len(payload["dates"])
    assert all(len(row) == n for row in payload["levels"]) and len(payload["vix_vix3m"]) == n
    # Index levels at 2 decimals, ratios at 4 (data-model.md section 5).
    assert all(round(v, 2) == v for row in payload["levels"] for v in row)
    assert all(round(v, 4) == v for v in payload["vix_vix3m"])
    # The flag is computed in Python from the unrounded ratio, and only where the ratio exists.
    assert all((flag is None) == (ratio is None) for flag, ratio in zip(payload["backwardation"], payload["vix_vix3m"]))
    assert any(payload["backwardation"]) and not all(payload["backwardation"])
    for flag, ratio in zip(payload["backwardation"], payload["vix_vix3m"]):
        if ratio is not None and abs(ratio - 1) > 1e-4:
            assert flag == (ratio > 1)


def test_real_check_passes_on_its_own_output_and_ignores_the_listed_hash(real_root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    export_app_data.write_files(export_app_data.generate("real", git=GIT_A, generated_at=TIME_A), out)
    # A later export at another time and commit has other bytes, so another sha256 in the manifest:
    # still the same content.
    assert export_app_data.check(export_app_data.generate("real", git=GIT_B, generated_at=TIME_B), out) == []


def test_real_check_catches_a_term_structure_that_does_not_match_its_manifest(real_root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    files = export_app_data.generate("real", git=GIT_A, generated_at=TIME_A)
    export_app_data.write_files(files, out)
    blob = (out / "term_structure.json").read_bytes()
    (out / "term_structure.json").write_bytes(blob.replace(b'"source_dirty":false', b'"source_dirty":true'))
    problems = export_app_data.check(files, out)
    assert len(problems) == 1 and "lists a sha256 or size for term_structure.json" in problems[0]


@pytest.mark.parametrize(
    "change, complaint",
    [
        (lambda p: p["dates"].reverse(), "strictly increasing"),
        (lambda p: p["tenors"].reverse(), "sorted by tenor_days"),
        (lambda p: p["levels"][0].pop(), "levels has"),
        (lambda p: p["backwardation"].append(None), "backwardation has"),
        (lambda p: p["levels"].pop(), "one levels row per tenor"),
    ],
)
def test_term_structure_invariants_the_schema_cannot_express_are_checked(change, complaint):
    payload = {
        "dates": ["2020-01-02", "2020-01-03"],
        "tenors": [{"symbol": "VIX", "tenor_days": 30, "label": "30D"}, {"symbol": "VIX3M", "tenor_days": 93, "label": "3M"}],
        "levels": [[20.0, 21.0], [22.0, 20.5]],
        "vix_vix3m": [0.9091, 1.0244],
        "backwardation": [False, True],
    }
    export_app_data.check_term_structure(payload)  # the unchanged payload passes
    change(payload)
    with pytest.raises(ValueError, match=complaint):
        export_app_data.check_term_structure(payload)


def test_real_export_cli_writes_outside_the_repo_and_never_inside(real_root, tmp_path):
    out = tmp_path / "app-data-real"
    assert export_app_data.main(["--mode", "real", "--out-dir", str(out)]) == export_app_data.EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "term_structure.json"]
    assert export_app_data.main(["--mode", "real", "--out-dir", str(REPO / "app" / "public" / "data" / "real")]) == export_app_data.EXIT_REFUSED
    assert not (REPO / "app" / "public" / "data" / "real").exists()


def test_synthetic_mode_writes_the_result_files_and_never_the_term_structure():
    # Final polish: the manifest plus the four result files; term_structure.json stays real-mode only
    # (the app's ribbon fixture is src/data/term-structure.synthetic.json).
    files = export_app_data.generate("synthetic", git=GIT_A, generated_at=TIME_A)
    assert sorted(files) == RESULT_FILES
    assert [f["dataset"] for f in parsed(files["manifest.json"])["payload"]["files"]] == ["model_race", "evaluation", "equity_curves", "eda"]
