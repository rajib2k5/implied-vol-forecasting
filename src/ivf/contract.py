"""The Python-to-app JSON contract (ADR 0001 D1, D5; schemas/app-data/).

Python computes, JSON carries, the app draws. Every JSON file the app reads is
an *envelope* around a file-specific ``payload``::

    {"schema_version": "1.0.0", "generated_at": "2026-10-28T09:30:00Z",
     "source_commit": "<40 hex>", "source_dirty": false,
     "data_mode": "synthetic", "dataset": "manifest", "payload": {...}}

This module does three jobs:

1. ``build_envelope``: wrap a payload and stamp it with *where it came from*
   (git commit, dirty flag, time). A number in the app can then be traced back
   to the code that produced it.
2. ``serialise``: turn the envelope into bytes that are identical for
   identical input. Stable bytes make ``--check`` modes and git diffs useful.
3. ``validate_document``: check a document against the JSON Schemas, with all
   ``$ref``s resolved from local files. The schemas are the contract. The
   TypeScript side will check the committed fixture against the same files in a
   vitest contract test (M1); the running app is a tolerant reader and does not
   validate (ADR 0001 D1). Nothing on the TypeScript side validates yet.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from ivf import config

#: Version of the whole contract (one version for all files, ADR 0001 D1).
#: envelope.schema.json pins the major part with a pattern, so bumping the
#: major version means editing that schema on purpose.
SCHEMA_VERSION = "1.1.0"

#: Closed enum (ADR 0001 D1): adding a value is a MAJOR version bump.
DATA_MODES = ("synthetic", "real")

#: The envelope requires 40 hex characters for ``source_commit``. When git
#: cannot give one (no repo, or a repo with no commits yet) we write 40 zeros.
#: That is git's own "null object id", so a reader who knows git recognises
#: it. It can never be a real commit hash, and it is always paired with
#: ``source_dirty = true``.
NO_COMMIT = "0" * 40

_SHA1_HEX = re.compile(r"[0-9a-f]{40}")
_SCHEMA_SUFFIX = ".schema.json"
_ENVELOPE = "envelope"


class ContractError(ValueError):
    """Something cannot be put into, or read from, the contract."""


class SchemaViolation(ContractError):
    """A document does not satisfy its JSON Schema.

    ``errors`` holds the underlying ``jsonschema`` errors, sorted by location,
    so callers (and tests) can look at *which* rule failed and *where*, not
    only that something failed.
    """

    def __init__(self, dataset: str, errors: list[ValidationError]):
        self.errors = errors
        lines = [f"  {e.json_path}: {e.message}" for e in errors]
        super().__init__(f"document violates the '{dataset}' schema:\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# Provenance: which code produced this file?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GitState:
    """``git rev-parse HEAD`` and whether the working tree has changes."""

    commit: str
    dirty: bool


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run git. Return ``None`` if the git program itself is missing."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None


def git_state(repo: Path | None = None) -> GitState:
    """Commit hash and dirty flag of the code in ``repo`` (default: this repo).

    Three situations are handled on purpose:

    * Normal repo: the real 40-hex commit, ``dirty`` true when ``git status``
      lists anything, including untracked files (a new source file changes
      the code even before it is tracked).
    * Not a git repo, or a repo with no commits yet (the state of a brand new
      project): ``git rev-parse --verify HEAD`` fails. We return ``NO_COMMIT``
      and ``dirty=True``. Code that no commit records cannot be reproduced,
      which is exactly what the dirty flag warns about.
    * The ``git`` program is not installed: same answer as above.
    """
    cwd = repo if repo is not None else config.repo_root()
    head = _git(["rev-parse", "--verify", "HEAD"], cwd)
    if head is None or head.returncode != 0:
        return GitState(commit=NO_COMMIT, dirty=True)
    commit = head.stdout.strip()
    if not _SHA1_HEX.fullmatch(commit):
        # For example a repo using SHA-256 object ids (64 hex). The envelope
        # schema cannot hold that, so fail loudly instead of truncating.
        raise ContractError(f"git returned {commit!r}, which is not a 40-hex SHA-1 commit id")
    status = _git(["status", "--porcelain", "--untracked-files=normal"], cwd)
    # If status itself fails we cannot show the tree is clean, so say dirty.
    dirty = status is None or status.returncode != 0 or bool(status.stdout.strip())
    return GitState(commit=commit, dirty=dirty)


# --------------------------------------------------------------------------
# Building and serialising the envelope
# --------------------------------------------------------------------------


def build_envelope(
    dataset: str,
    payload: Mapping[str, Any],
    data_mode: str,
    *,
    git: GitState | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Wrap ``payload`` in the envelope of ADR 0001 D1.

    ``git`` and ``generated_at`` can be passed in so tests (and ``--check``
    modes) do not depend on the clock or on the state of the working tree.
    """
    if data_mode not in DATA_MODES:
        raise ContractError(f"data_mode must be one of {DATA_MODES}, got {data_mode!r}")
    git = git if git is not None else git_state()
    moment = generated_at if generated_at is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        # A naive datetime has no zone, so we cannot know what instant it means.
        raise ContractError("generated_at must be timezone-aware")
    return {
        "schema_version": SCHEMA_VERSION,
        # Whole seconds in UTC: matches the schema pattern and avoids noise.
        "generated_at": moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": git.commit,
        "source_dirty": git.dirty,
        "data_mode": data_mode,
        "dataset": dataset,
        "payload": dict(payload),
    }


def _check_json(value: Any, path: str = "$") -> None:
    """Raise ``ContractError`` unless ``value`` is built only from JSON types.

    ``json.dumps`` is too forgiving about keys and too blunt about values:

    * Keys. It accepts int, float, bool and None keys and turns them into
      text, but sorts them *before* doing so. ``{10: 1, 9: 2}`` gives
      ``{"9":2,"10":1}`` while ``{"10": 1, "9": 2}`` gives ``{"10":1,"9":2}``:
      one JSON value, two byte strings, two hashes. Allowing only ``str`` keys
      leaves a single spelling for every JSON value.
    * Values. numpy and pandas scalars (``np.int64``, ``np.float32``,
      ``np.bool_``, ``pd.NA``, ``pd.Timestamp``) are not Python numbers or
      strings, and ``json.dumps`` fails on them with a bare ``TypeError``.
      Exporters will meet these, so say how to convert instead.

    ``path`` is where ``value`` sits in the document, for the error message.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(
                    f"{path}: dict key {key!r} has type {type(key).__name__}, not str. JSON keys are text, "
                    "and number keys would sort numerically, so one JSON value could get two hashes; "
                    "convert it yourself with str(key)"
                )
            _check_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_json(item, f"{path}[{i}]")
    elif value is not None and not isinstance(value, (str, int, float)):  # bool is an int
        raise ContractError(
            f"{path}: {type(value).__name__} is not a JSON type. Convert it before exporting: "
            "float(x) or int(x) for a number, bool(x) for a flag, x.tolist() for an array, "
            "str(x) for a date or label, None for a missing value"
        )


def _canonical_text(document: Any) -> str:
    """Strict, compact, key-sorted JSON text.

    * ``_check_json``: only ``str`` keys and plain Python values (see there).
    * ``allow_nan=False``: bare ``NaN`` and ``Infinity`` are not valid JSON,
      and a browser's ``JSON.parse`` rejects them. Missing values must be
      turned into ``None`` (JSON ``null``) by the caller (ADR 0001 D5).
    * ``sort_keys=True``: key order must not depend on how a dict was built.
    * ``separators=(",", ":")``: compact, and also the exact spelling the
      deploy guard greps for: ``"data_mode":"real"`` (ADR 0001 D2).
    * ``ensure_ascii`` (the default): escape non-ASCII, so the bytes do not
      depend on the platform's text encoding.
    """
    try:
        _check_json(document)
        return json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except ContractError:
        raise  # _check_json already says what is wrong and where; ContractError is a ValueError, so skip the clause below
    except RecursionError as exc:
        # A dict that contains itself, or nesting deeper than Python's recursion limit.
        raise ContractError("cannot write JSON: the value contains itself or is nested too deeply") from exc
    except ValueError as exc:
        raise ContractError(
            f"cannot write strict JSON ({exc}); export missing values as None, not NaN"
        ) from exc


def serialise(document: Mapping[str, Any]) -> bytes:
    """UTF-8 bytes of ``document``: the same input always gives the same bytes."""
    return (_canonical_text(document) + "\n").encode("utf-8")


def canonical_json_sha256(obj: Any) -> str:
    """SHA-256 hex of the canonical JSON of ``obj`` (ADR 0001 D3 ``protocol_hash``).

    Sorted keys and compact separators mean two configs with the same content
    hash the same, whatever order their keys were written in.
    """
    return hashlib.sha256(_canonical_text(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Validating against schemas/app-data/*.schema.json
# --------------------------------------------------------------------------


def load_schemas(schema_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """All ``*.schema.json`` files, keyed by name (``manifest``, ``envelope``, ...)."""
    directory = schema_dir if schema_dir is not None else config.schema_dir()
    paths = sorted(directory.glob(f"*{_SCHEMA_SUFFIX}"))
    if not paths:
        raise ContractError(f"no *{_SCHEMA_SUFFIX} files in {directory}")
    return {p.name[: -len(_SCHEMA_SUFFIX)]: json.loads(p.read_text(encoding="utf-8")) for p in paths}


def build_registry(schemas: Mapping[str, Mapping[str, Any]]) -> Registry:
    """A ``referencing`` registry holding every schema under its ``$id``.

    A file schema says ``"$ref": "envelope.schema.json"``. That is resolved
    against the file's own ``$id`` (a github.com URL). The registry maps that
    URL to the local file's content, so resolution never touches the network.
    A ``$ref`` to something not in the registry raises an error instead of
    being fetched.
    """
    resources = []
    for name, schema in schemas.items():
        if "$id" not in schema:
            raise ContractError(f"schema '{name}' has no $id, so other schemas cannot $ref it")
        resources.append((schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012)))
    return Registry().with_resources(resources)


def _validate(document: Any, name: str, schemas: Mapping[str, Mapping[str, Any]]) -> None:
    validator = Draft202012Validator(schemas[name], registry=build_registry(schemas))
    try:
        errors = sorted(validator.iter_errors(document), key=lambda e: (e.json_path, e.message))
    except Unresolvable as exc:
        # $refs are looked up lazily, while the document is being checked. The library's error
        # is not a ValueError, so callers that catch ValueError (--check) would crash on it.
        raise ContractError(f"schema '{name}' has a $ref that no local schema file provides: {exc}") from exc
    if errors:
        raise SchemaViolation(name, errors)


def validate_envelope(document: Any, schema_dir: Path | None = None) -> None:
    """Check only the envelope (any ``dataset`` id, any object payload).

    Raises ``SchemaViolation`` on failure.
    """
    _validate(document, _ENVELOPE, load_schemas(schema_dir))


def validate_document(document: Any, schema_dir: Path | None = None) -> None:
    """Check ``document`` against the schema for its own ``dataset``.

    The file schemas ``$ref`` the envelope, so this checks the envelope fields
    *and* the payload. A dataset with no schema file is an error, not a pass:
    quietly validating less would defeat the contract.

    Raises ``SchemaViolation`` for a bad document, ``ContractError`` for an
    unknown dataset.
    """
    schemas = load_schemas(schema_dir)
    dataset = document.get("dataset") if isinstance(document, Mapping) else None
    if isinstance(dataset, str) and dataset != _ENVELOPE and dataset in schemas:
        _validate(document, dataset, schemas)
        return
    # Not a known dataset. Run the envelope check first so a malformed
    # envelope reports its real problem (missing field, bad pattern, ...).
    _validate(document, _ENVELOPE, schemas)
    raise ContractError(f"no schema for dataset {dataset!r}; known: {sorted(set(schemas) - {_ENVELOPE})}")
