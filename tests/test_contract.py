"""Tests for ivf.contract: the versioned JSON envelope (ADR 0001 D1, D5).

Expected values are literals worked out by hand, with the reasoning in a
comment, never recomputed by the code under test.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ivf import contract

REPO = Path(__file__).resolve().parents[1]
COMMIT = "0123456789abcdef0123456789abcdef01234567"  # 16 + 16 + 8 = 40 hex characters
GIT = contract.GitState(commit=COMMIT, dirty=False)
NOON = datetime(2026, 10, 28, 9, 30, 0, tzinfo=timezone.utc)
ENVELOPE_FIELDS = {"schema_version", "generated_at", "source_commit", "source_dirty", "data_mode", "dataset", "payload"}


def example_envelope(**overrides):
    envelope = contract.build_envelope("example", {"values": [1.5, None]}, "synthetic", git=GIT, generated_at=NOON)
    envelope.update(overrides)
    return envelope


def violation_for(document):
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_envelope(document)
    return caught.value.errors


# --- building the envelope ---------------------------------------------------


def test_envelope_has_exactly_the_seven_fields_with_the_right_values():
    envelope = example_envelope()
    assert set(envelope) == ENVELOPE_FIELDS
    assert envelope == {
        "schema_version": "1.0.0",
        "generated_at": "2026-10-28T09:30:00Z",
        "source_commit": COMMIT,
        "source_dirty": False,
        "data_mode": "synthetic",
        "dataset": "example",
        "payload": {"values": [1.5, None]},
    }
    assert isinstance(envelope["source_dirty"], bool)


def test_generated_at_is_converted_to_utc_and_truncated_to_seconds():
    # 05:30 at UTC-4 is 09:30 UTC (add 4 h). The 999999 microseconds are dropped, not rounded up.
    eastern = timezone(timedelta(hours=-4))
    moment = datetime(2026, 10, 28, 5, 30, 0, 999999, tzinfo=eastern)
    envelope = contract.build_envelope("example", {}, "real", git=GIT, generated_at=moment)
    assert envelope["generated_at"] == "2026-10-28T09:30:00Z"


def test_naive_timestamp_is_refused():
    with pytest.raises(contract.ContractError, match="timezone-aware"):
        contract.build_envelope("example", {}, "synthetic", git=GIT, generated_at=datetime(2026, 10, 28, 9, 30))


@pytest.mark.parametrize("mode", ["Real", "paper", "", "live"])
def test_builder_refuses_a_data_mode_outside_the_closed_enum(mode):
    with pytest.raises(contract.ContractError, match="data_mode"):
        contract.build_envelope("example", {}, mode, git=GIT, generated_at=NOON)


def test_a_built_envelope_passes_the_envelope_schema():
    contract.validate_envelope(example_envelope())


# --- serialising -------------------------------------------------------------


def test_serialise_exact_bytes():
    # Keys sorted ("a" < "b"), compact separators, None -> null, newline at the end.
    assert contract.serialise({"b": [1, None], "a": "x"}) == b'{"a":"x","b":[1,null]}\n'


def test_serialise_does_not_depend_on_insertion_order():
    assert contract.serialise({"a": 1, "b": {"y": 2, "x": 3}}) == contract.serialise({"b": {"x": 3, "y": 2}, "a": 1})


def test_same_input_gives_identical_bytes_every_time():
    assert contract.serialise(example_envelope()) == contract.serialise(example_envelope())


def test_non_ascii_is_escaped_so_bytes_do_not_depend_on_the_platform():
    # U+00E9 (e acute) must be written as the six characters \u00e9.
    assert contract.serialise({"k": "\u00e9"}) == b'{"k":"\\u00e9"}\n'


def test_real_mode_spelling_matches_the_deploy_guard():
    # ADR 0001 D2: `npm run verify` greps dist/ for "data_mode":"real" with no space.
    blob = contract.serialise(contract.build_envelope("example", {}, "real", git=GIT, generated_at=NOON))
    assert b'"data_mode":"real"' in blob


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_rejected(bad):
    # Bare NaN / Infinity are not JSON; a browser's JSON.parse would throw.
    with pytest.raises(contract.ContractError, match="strict JSON"):
        contract.serialise(example_envelope(payload={"values": [1.0, bad]}))


def test_missing_values_are_written_as_null():
    assert contract.serialise({"v": [None]}) == b'{"v":[null]}\n'


def test_canonical_hash_is_independent_of_key_order():
    # Canonical text is {"a":1,"b":[2,3]}. Its SHA-256, computed outside Python with
    #   printf '%s' '{"a":1,"b":[2,3]}' | sha256sum
    expected = "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"
    assert contract.canonical_json_sha256({"b": [2, 3], "a": 1}) == expected
    assert contract.canonical_json_sha256({"a": 1, "b": [2, 3]}) == expected


# --- what may go into a document ---------------------------------------------


def test_string_keys_that_look_like_numbers_sort_as_text():
    # Text order compares character by character: "1" < "9", so "10" comes before "9".
    # With str keys that is the only possible spelling of this JSON value.
    assert contract.serialise({"9": 2, "10": 1}) == b'{"10":1,"9":2}\n'
    assert contract.serialise({"10": 1, "9": 2}) == b'{"10":1,"9":2}\n'


@pytest.mark.parametrize(
    "document",
    [{10: 1}, {1.5: 1}, {True: 1}, {None: 1}, {(1, 2): 1}, {1: "a", "b": 2}, {"b": 2, 1: "a"}],
    ids=["int", "float", "bool", "None", "tuple", "int then str", "str then int"],
)
@pytest.mark.parametrize("canonicalise", [contract.serialise, contract.canonical_json_sha256])
def test_non_string_dict_keys_are_rejected(document, canonicalise):
    # json.dumps(sort_keys=True) sorts {10: 1, 9: 2} numerically, as {"9":2,"10":1}, but the
    # text-keyed {"10": 1, "9": 2} comes out as {"10":1,"9":2}: one JSON value, two hashes.
    # (A mixed int/str dict used to die with a bare TypeError from the sort.)
    with pytest.raises(contract.ContractError, match=r"^\$: dict key .* has type \w+, not str"):
        canonicalise(document)


def test_a_non_string_key_is_found_deep_inside_the_payload_and_located():
    document = {"payload": {"series": [{"ok": 1}, {9: 2, 10: 1}]}}
    with pytest.raises(contract.ContractError, match=r"^\$\.payload\.series\[1\]: dict key 9 has type int"):
        contract.serialise(document)


def test_a_bad_value_after_a_good_one_in_the_same_dict_is_found():
    # Review D's mutant D6 checked only the first key and value of each dict and passed every
    # test, because every test put the bad item first. Here it comes second or third.
    with pytest.raises(contract.ContractError, match=r"^\$\.b: int64 is not a JSON type"):
        contract.serialise({"a": 1, "b": np.int64(2)})
    with pytest.raises(contract.ContractError, match=r"^\$\.c: int64 is not a JSON type"):
        contract.serialise({"a": 1, "b": 2, "c": np.int64(3)})


def test_a_bad_key_in_a_nested_dict_that_is_not_the_first_item_is_found():
    with pytest.raises(contract.ContractError, match=r"^\$\.z: dict key 10 has type int"):
        contract.serialise({"a": 1, "z": {10: 1}})
    with pytest.raises(contract.ContractError, match=r"^\$\.m\.n: dict key 7 has type int"):
        contract.serialise({"a": {"ok": 1}, "m": {"x": 0, "n": {"k": 1, 7: 2}}})


NOT_JSON = [
    np.int64(3),
    np.float32(1.5),
    np.bool_(True),
    np.array([1, 2]),
    pd.NA,
    pd.Timestamp("2026-10-28"),
    {1, 2},
    datetime(2026, 10, 28),
    b"bytes",
]


@pytest.mark.parametrize("value", NOT_JSON, ids=lambda v: type(v).__name__)
@pytest.mark.parametrize("canonicalise", [contract.serialise, contract.canonical_json_sha256])
def test_values_that_are_not_json_types_say_where_they_are_and_how_to_convert(value, canonicalise):
    # The index in the message counts items in the list: [1.0, value] puts the culprit at [1].
    with pytest.raises(contract.ContractError, match=r"^\$\.payload\.values\[1\]: \w+ is not a JSON type.*float\(x\)"):
        canonicalise({"payload": {"values": [1.0, value]}})


def test_numpy_float64_is_a_python_float_so_it_is_accepted_and_its_nan_is_refused():
    # np.float64 subclasses float, unlike np.float32 or np.int64.
    assert contract.serialise({"x": np.float64(1.5)}) == b'{"x":1.5}\n'
    with pytest.raises(contract.ContractError, match="strict JSON"):
        contract.serialise({"x": np.float64("nan")})


def test_tuples_are_written_as_arrays():
    assert contract.serialise({"x": (1, 2)}) == b'{"x":[1,2]}\n'


def test_a_value_that_contains_itself_is_a_contract_error():
    loop: dict = {}
    loop["me"] = loop
    with pytest.raises(contract.ContractError, match="contains itself"):
        contract.serialise(loop)


# --- the envelope schema -----------------------------------------------------


@pytest.mark.parametrize("version", ["1.0.0", "1.12.3", "1.0.100"])
def test_schema_version_accepts_major_1(version):
    contract.validate_envelope(example_envelope(schema_version=version))


@pytest.mark.parametrize(
    "version",
    [
        "2.0.0",  # the pattern pins MAJOR = 1
        "0.9.0",
        "1.0",  # needs three parts
        "1.0.0.0",  # exactly three parts
        "1.00.0",  # no leading zero in MINOR
        "1.0.0-beta",  # no pre-release suffix
        "v1.0.0",
        "",
    ],
)
def test_schema_version_rejects_everything_else(version):
    errors = violation_for(example_envelope(schema_version=version))
    assert [(e.json_path, e.validator) for e in errors] == [("$.schema_version", "pattern")]


@pytest.mark.parametrize("field", sorted(ENVELOPE_FIELDS))
def test_every_envelope_field_is_required(field):
    envelope = example_envelope()
    del envelope[field]
    errors = violation_for(envelope)
    assert [e.validator for e in errors] == ["required"]
    assert f"'{field}' is a required property" in errors[0].message


def test_extra_envelope_field_is_rejected():
    errors = violation_for(example_envelope(note="hello"))
    assert [e.validator for e in errors] == ["additionalProperties"]
    assert "'note'" in errors[0].message


@pytest.mark.parametrize("mode", ["Real", "paper", "", None, True])
def test_data_mode_must_be_synthetic_or_real(mode):
    errors = violation_for(example_envelope(data_mode=mode))
    assert [(e.json_path, e.validator) for e in errors] == [("$.data_mode", "enum")]


@pytest.mark.parametrize("mode", ["synthetic", "real"])
def test_data_mode_accepts_the_two_values(mode):
    contract.validate_envelope(example_envelope(data_mode=mode))


@pytest.mark.parametrize(
    "commit",
    [
        "0123456789abcdef0123456789abcdef0123456",  # 39 characters
        "0123456789abcdef0123456789abcdef012345678",  # 41 characters
        "0123456789ABCDEF0123456789abcdef01234567",  # upper case
        "0123456789abcdef0123456789abcdef0123456g",  # 'g' is not hex
    ],
)
def test_source_commit_must_be_40_lowercase_hex(commit):
    errors = violation_for(example_envelope(source_commit=commit))
    assert [(e.json_path, e.validator) for e in errors] == [("$.source_commit", "pattern")]


def test_the_no_commit_placeholder_is_valid():
    contract.validate_envelope(example_envelope(source_commit=contract.NO_COMMIT, source_dirty=True))
    assert contract.NO_COMMIT == "0" * 40


@pytest.mark.parametrize("dirty", ["false", 0, 1, None])
def test_source_dirty_must_be_a_real_boolean(dirty):
    errors = violation_for(example_envelope(source_dirty=dirty))
    assert [(e.json_path, e.validator) for e in errors] == [("$.source_dirty", "type")]


@pytest.mark.parametrize("stamp", ["2026-10-28T09:30:00Z", "2026-10-28T09:30:00.123Z"])
def test_generated_at_accepts_utc_with_optional_fraction(stamp):
    contract.validate_envelope(example_envelope(generated_at=stamp))


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-10-28T09:30:00",  # no zone
        "2026-10-28T09:30:00+00:00",  # offset spelling, not Z
        "2026-10-28 09:30:00Z",  # space instead of T
        "2026-10-28",
    ],
)
def test_generated_at_rejects_other_spellings(stamp):
    errors = violation_for(example_envelope(generated_at=stamp))
    assert [(e.json_path, e.validator) for e in errors] == [("$.generated_at", "pattern")]


@pytest.mark.parametrize("dataset", ["Manifest", "9lives", "has-dash", ""])
def test_dataset_must_be_a_lowercase_id(dataset):
    errors = violation_for(example_envelope(dataset=dataset))
    assert [(e.json_path, e.validator) for e in errors] == [("$.dataset", "pattern")]


def test_payload_must_be_an_object():
    errors = violation_for(example_envelope(payload=[1, 2]))
    assert [(e.json_path, e.validator) for e in errors] == [("$.payload", "type")]


def test_all_problems_are_reported_together_in_path_order():
    errors = violation_for(example_envelope(data_mode="paper", schema_version="2.0.0"))
    assert [e.json_path for e in errors] == ["$.data_mode", "$.schema_version"]


# --- dataset schemas (manifest as the worked example) ------------------------


def manifest_payload():
    return {
        "run_id": "0123456789ab",
        "notice": "Synthetic data, not market data.",
        "protocol": {
            "protocol_hash": "a" * 64,
            "train_start": "1990-01-02",
            "dev_start": "2000-01-03",
            "holdout_start": "2024-07-01",
            "data_end": "2026-10-26",
            "horizons": [1, 5, 21],
            "fold_len": 63,
            "min_cell_n": 20,
            "holdout_looks": 0,
            "vix_regime_edges": [15, 25],
        },
        "files": [],
        "models": [],
        "targets": [{"target_id": "vix_level", "label": "VIX close", "units": "vol_points", "horizons": [1, 5, 21]}],
    }


def manifest_with(**changes):
    payload = manifest_payload()
    payload.update(changes)
    return contract.build_envelope("manifest", payload, "synthetic", git=GIT, generated_at=NOON)


def file_entry(**changes):
    entry = {"dataset": "term_structure", "path": "term_structure.json", "sha256": "b" * 64, "bytes": 1000, "license_class": "series"}
    entry.update(changes)
    return entry


def violations_of(document):
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(document)
    return [(e.json_path, e.validator) for e in caught.value.errors]


def test_valid_manifest_passes_after_a_serialise_round_trip():
    # Validate what the app would actually read: the parsed bytes.
    contract.validate_document(json.loads(contract.serialise(manifest_with())))


def test_manifest_size_budget_is_2_mib():
    # 2 MiB = 2 * 1024 * 1024 = 2,097,152 bytes: allowed. One more byte: rejected.
    contract.validate_document(manifest_with(files=[file_entry(bytes=2_097_152)]))
    assert violations_of(manifest_with(files=[file_entry(bytes=2_097_153)])) == [("$.payload.files[0].bytes", "maximum")]


def test_manifest_payload_rejects_unknown_fields():
    assert violations_of(manifest_with(extra=1)) == [("$.payload", "additionalProperties")]


def test_manifest_must_not_list_itself():
    assert [p for p, _ in violations_of(manifest_with(files=[file_entry(dataset="manifest")]))] == ["$.payload.files[0].dataset"]


def test_manifest_license_class_is_a_closed_enum():
    assert violations_of(manifest_with(files=[file_entry(license_class="public")])) == [
        ("$.payload.files[0].license_class", "enum")
    ]


def test_dataset_without_a_schema_is_an_error_not_a_pass():
    with pytest.raises(contract.ContractError, match="no schema for dataset 'nonesuch'") as caught:
        contract.validate_document(example_envelope(dataset="nonesuch"))
    assert not isinstance(caught.value, contract.SchemaViolation)


def test_the_envelope_is_not_a_dataset_you_can_name():
    with pytest.raises(contract.ContractError, match="no schema for dataset 'envelope'"):
        contract.validate_document(example_envelope(dataset="envelope"))


def test_malformed_envelope_of_unknown_dataset_reports_the_real_problem():
    document = example_envelope(dataset="nonesuch", data_mode="paper")
    assert violations_of(document) == [("$.data_mode", "enum")]


def test_document_that_is_not_an_object_or_has_no_dataset_gets_the_envelope_errors():
    assert violations_of([]) == [("$", "type")]
    envelope = example_envelope()
    del envelope["dataset"]
    assert violations_of(envelope) == [("$", "required")]


def test_a_ref_that_no_local_file_provides_is_a_contract_error_not_a_crash(tmp_path):
    # $refs are only looked up while a document is checked. The library raises its own
    # exception type then (not a ValueError), which would crash callers that catch ValueError.
    # Nothing is fetched: the registry holds the local files and no retriever.
    broken = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/schemas/broken.schema.json",
        "$ref": "missing.schema.json",
    }
    (tmp_path / "broken.schema.json").write_text(json.dumps(broken))
    with pytest.raises(contract.ContractError, match=r"schema 'broken' has a \$ref that no local schema file provides") as caught:
        contract.validate_document(example_envelope(dataset="broken"), schema_dir=tmp_path)
    assert not isinstance(caught.value, contract.SchemaViolation)
    assert "missing.schema.json" in str(caught.value)


def test_schema_without_an_id_cannot_be_registered():
    with pytest.raises(contract.ContractError, match="no \\$id"):
        contract.build_registry({"broken": {"type": "object"}})


def test_empty_schema_folder_is_an_error(tmp_path):
    with pytest.raises(contract.ContractError, match="no \\*.schema.json files"):
        contract.load_schemas(tmp_path)


def test_validation_can_use_a_different_schema_folder(tmp_path):
    # A copy of the schemas with an extra required envelope field: the same
    # document passes against the real schemas and fails against the copy.
    folder = tmp_path / "schemas"
    folder.mkdir()
    for path in (Path(__file__).resolve().parents[1] / "schemas" / "app-data").glob("*.schema.json"):
        text = path.read_text()
        if path.name == "envelope.schema.json":
            text = text.replace('"required": ["schema_version",', '"required": ["unit_test_field", "schema_version",')
        (folder / path.name).write_text(text)
    contract.validate_envelope(example_envelope())
    with pytest.raises(contract.SchemaViolation, match="unit_test_field"):
        contract.validate_envelope(example_envelope(), schema_dir=folder)


# --- provenance from git -----------------------------------------------------


def git(repo: Path, *args: str) -> None:
    # -c flags: do not depend on the developer's global git config.
    base = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*base, *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Stop git from finding a repository above tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    path = tmp_path / "repo"
    path.mkdir()
    return path


def test_not_a_git_repo_gives_the_placeholder_and_dirty(repo):
    assert contract.git_state(repo) == contract.GitState(commit="0" * 40, dirty=True)


def test_repo_with_no_commits_gives_the_placeholder_and_dirty(repo):
    git(repo, "init", "-b", "main")
    assert contract.git_state(repo) == contract.GitState(commit="0" * 40, dirty=True)


def test_git_program_missing_gives_the_placeholder_and_dirty(repo, tmp_path, monkeypatch):
    git(repo, "init", "-b", "main")
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert contract.git_state(repo) == contract.GitState(commit="0" * 40, dirty=True)


def test_commit_hash_and_clean_flag_after_a_commit(repo):
    git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-m", "first")
    # Read the hash straight from the ref file, a different route from `git rev-parse`.
    expected = (repo / ".git" / "refs" / "heads" / "main").read_text().strip()
    state = contract.git_state(repo)
    assert state == contract.GitState(commit=expected, dirty=False)
    assert re.fullmatch(r"[0-9a-f]{40}", state.commit)


def test_dirty_flag_follows_the_working_tree(repo):
    git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    (repo / ".gitignore").write_text("*.log\n")
    git(repo, "add", "a.txt", ".gitignore")
    git(repo, "commit", "-m", "first")
    assert contract.git_state(repo).dirty is False

    (repo / "debug.log").write_text("ignored\n")  # ignored files do not count
    assert contract.git_state(repo).dirty is False

    (repo / "new.txt").write_text("untracked\n")  # a new source file is a code change
    assert contract.git_state(repo).dirty is True
    git(repo, "add", "new.txt")
    git(repo, "commit", "-m", "second")
    assert contract.git_state(repo).dirty is False

    (repo / "a.txt").write_text("changed\n")  # edited tracked file
    assert contract.git_state(repo).dirty is True


def test_a_failing_git_status_means_dirty(repo):
    # "If status itself fails we cannot show the tree is clean, so say dirty."
    # A damaged index makes `git status` exit 128 with nothing on stdout, while HEAD still
    # resolves. Without the rule, empty output would read as "clean".
    git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-m", "first")
    expected = (repo / ".git" / "refs" / "heads" / "main").read_text().strip()
    assert contract.git_state(repo) == contract.GitState(commit=expected, dirty=False)

    (repo / ".git" / "index").write_bytes(b"garbage")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False)
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    assert (status.returncode, status.stdout, head.returncode) == (128, "", 0)  # the situation the test needs
    assert contract.git_state(repo) == contract.GitState(commit=expected, dirty=True)


def test_a_repo_with_sha256_object_ids_is_refused_not_truncated(repo):
    # The envelope schema holds 40 hex characters; a SHA-256 repo gives 64.
    git(repo, "init", "--object-format=sha256", "-b", "main")
    (repo / "a.txt").write_text("one\n")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-m", "first")
    with pytest.raises(contract.ContractError, match="not a 40-hex SHA-1 commit id"):
        contract.git_state(repo)


def test_default_repo_is_this_project(tmp_path, monkeypatch):
    # git_state() with no argument must describe the project's own checkout, the one
    # config.repo_root() finds, and not whatever directory the process happens to be in.
    # A stand-in project with one commit lets the test pin the exact hash in every state of
    # the real repo, including before its first commit (when the real answer is the placeholder).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    git(project, "init", "-b", "main")
    (project / "a.txt").write_text("one\n")
    git(project, "add", "a.txt")
    git(project, "commit", "-m", "first")
    expected = (project / ".git" / "refs" / "heads" / "main").read_text().strip()

    monkeypatch.setattr(contract.config, "repo_root", lambda: project)
    monkeypatch.chdir(elsewhere)  # not a git repo: asking git about the cwd would give the placeholder
    assert contract.git_state() == contract.GitState(commit=expected, dirty=False)
    (project / "new.txt").write_text("untracked\n")
    assert contract.git_state() == contract.GitState(commit=expected, dirty=True)


def test_this_checkouts_state_agrees_with_git_itself():
    # The real repo. Before its first commit git has no HEAD and the answer is the placeholder;
    # once there is a commit, git_state() must report exactly the hash git reports.
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    state = contract.git_state(REPO)
    if head.returncode == 0:
        assert state.commit == head.stdout.strip()
    else:
        assert state == contract.GitState(commit=contract.NO_COMMIT, dirty=True)

