"""Write the JSON files the React app reads (ADR 0001 D1, D2, D5).

    uv run python tools/export_app_data.py --mode synthetic          # write the fixture
    uv run python tools/export_app_data.py --mode synthetic --check  # CI: is the fixture current?
    uv run python tools/export_app_data.py --mode real               # ROOT/app-data/real/, never the repo

Synthetic mode produces one file, ``manifest.json``, unchanged since milestone 0.
Real mode (milestone 1) produces ``manifest.json`` and ``term_structure.json``
from the interim tables and the frozen ``configs/protocol.yaml``. The other data
files (model_race, error_surface, equity_curves) arrive with the milestones
that compute them, and each one is added to the manifest's ``files`` list only
once it exists (data-model.md 5.1).

Two modes, two destinations:

* ``synthetic``: made-up data, safe to publish. Written to
  ``app/public/data/synthetic/`` and committed to git.
* ``real``: derived from Cboe/FRED/Yahoo data whose licence forbids
  redistribution. It may only be written OUTSIDE the repository
  (``$VOL_DATA_DIR/implied-vol-forecasting/app-data/real/``). The tool refuses
  any output folder that is inside the repo, so ``git add -A`` or ``vite build``
  can never pick real data up. This is the licensing guard of ADR 0001 D2. It
  compares the folders on disk (device and inode) with every folder inside the
  repo, not the text of the paths, and it never writes through a symlink or
  hard link (see ``write_files``).

Exit codes: 0 success, 1 ``--check`` found a difference, 2 bad command line,
3 refused by the licensing guard, 4 the mode cannot run (real mode: the data
pipeline has not produced its inputs yet; the message says which step to run),
5 the output folder could not be written (permissions, read-only filesystem,
disk full; the message says which).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ivf import config, contract, repo_guard

MANIFEST = "manifest"
TERM_STRUCTURE = "term_structure"

# Provenance: facts about *which run* made a file, not about what is in it. They
# differ on every export, so --check ignores them (ADR 0001 D5, amended M0). The
# fixture is allowed to be stale in these, because --check is what proves the
# *content* still matches the code.
#   * generated_at, source_commit, source_dirty: envelope fields.
#   * run_id: manifest payload field. It is a hash of source_commit and
#     source_dirty (among other things), so it changes whenever they do.
#   * files[].sha256 and files[].bytes in the manifest (milestone 1): they hash
#     the listed file's bytes, and those bytes contain the provenance fields
#     above, so they change on every export. --check compares them with the
#     files on disk instead (``_manifest_mismatches``).
ENVELOPE_PROVENANCE = ("generated_at", "source_commit", "source_dirty")
PAYLOAD_PROVENANCE = ("run_id",)
FILE_ENTRY_PROVENANCE = ("sha256", "bytes")

EXIT_OK, EXIT_CHECK_FAILED, EXIT_UNAVAILABLE, EXIT_WRITE_FAILED = 0, 1, 4, 5
EXIT_REFUSED = repo_guard.EXIT_REFUSED  # 3, shared with python -m ivf.data

# Milestone 0 stand-in for the synthetic protocol. Only the fields the manifest
# schema exposes. Milestone 1 added configs/protocol.synthetic.yaml for the
# synthetic *pipeline* but left the committed fixture on this dict on purpose
# (the brief kept synthetic mode unchanged): switching would change
# protocol_hash and so the committed manifest. The switch is a separate,
# reviewable change. Dates follow ADR 0001 4.6: holdout_start is 2024-07-01
# (O1). data_end is the synthetic placeholder 2026-10-26, which
# configs/protocol.synthetic.yaml repeats; the real data_end is frozen in
# configs/protocol.yaml.
M0_PROTOCOL: dict[str, Any] = {
    "train_start": "1990-01-02",
    "dev_start": "2000-01-03",
    "holdout_start": "2024-07-01",
    "data_end": "2026-10-26",
    "horizons": [1, 5, 21],
    "fold_len": 63,
    "min_cell_n": 20,
    "holdout_looks": 0,
    "vix_regime_edges": [15, 25],
}

# The two forecast targets of ADR 0001 4.2.
M0_TARGETS: list[dict[str, Any]] = [
    {"target_id": "vix_level", "label": "VIX close", "units": "vol_points", "horizons": [1, 5, 21]},
    {"target_id": "rv21", "label": "21-session realised S&P 500 volatility", "units": "vol_points", "horizons": [21]},
]


# Raised for an output folder inside a git checkout (real mode) and for an output file that is a
# symlink or a hard link (any mode). One class, shared with the data pipeline.
RepoWriteRefused = repo_guard.RepoWriteRefused


class ModeUnavailable(Exception):
    """The requested mode cannot run: it is unknown, or (real mode) its inputs do not exist yet."""


# --------------------------------------------------------------------------
# What to write
# --------------------------------------------------------------------------


def synthetic_dir() -> Path:
    """The committed fixture folder. Only this tool may write under ``app/``."""
    return config.repo_root() / "app" / "public" / "data" / "synthetic"


def manifest_payload(git: contract.GitState) -> dict[str, Any]:
    protocol_hash = contract.canonical_json_sha256(M0_PROTOCOL)
    # run_id as in ADR 0001 D3: a hash of the protocol, the code state, the data
    # mode and the input files. M0 has no input files yet, so that list is empty.
    # Because it contains source_commit and source_dirty, run_id is provenance
    # and --check ignores it (see PAYLOAD_PROVENANCE).
    run_id = contract.canonical_json_sha256(
        {
            "protocol_hash": protocol_hash,
            "source_commit": git.commit,
            "source_dirty": git.dirty,
            "data_mode": "synthetic",
            "raw_sha256s": [],
        }
    )[:12]
    return {
        "run_id": run_id,
        "notice": "Synthetic data, not market data. Milestone 0: no data files exist yet.",
        "protocol": {"protocol_hash": protocol_hash, **M0_PROTOCOL},
        # Empty on purpose: a file is listed only once it exists, because each
        # entry needs the real sha256 and byte size of the file.
        "files": [],
        # No model has run yet; the registry fills in from milestone 2.
        "models": [],
        "targets": M0_TARGETS,
    }


def generate(
    mode: str, *, git: contract.GitState | None = None, generated_at: datetime | None = None
) -> dict[str, bytes]:
    """Every file this milestone exports, as ``{file name: bytes}``, validated.

    Nothing is written here, so ``--check`` can reuse it.
    """
    if mode not in contract.DATA_MODES:
        raise ModeUnavailable(f"unknown mode {mode!r}")
    # One git lookup, shared by the envelope and by run_id, so they cannot disagree.
    git = git if git is not None else contract.git_state()
    if mode == "real":
        return generate_real(git=git, generated_at=generated_at)
    envelope = contract.build_envelope(MANIFEST, manifest_payload(git), mode, git=git, generated_at=generated_at)
    blob = contract.serialise(envelope)
    # Validate the bytes we are about to write, parsed back, not the dict we
    # built: that is what the app will actually read.
    contract.validate_document(json.loads(blob))
    return {f"{MANIFEST}.json": blob}


# --------------------------------------------------------------------------
# Real mode (milestone 1): manifest.json and term_structure.json
# --------------------------------------------------------------------------

REAL_NOTICE = (
    "Real data: Cboe index levels (Cboe Global Markets) and S&P 500 prices (Yahoo Finance via yfinance), "
    "for personal, non-commercial use only; not redistributed. Not financial advice."
)
#: Rounding at export (data-model.md section 5): index levels 2 decimals, ratios 4.
LEVEL_DECIMALS = 2
RATIO_DECIMALS = 4
# VIX1Y is 366 days in Cboe's term-index methodology (amended M1, review A).
TENOR_LABELS = {9: "9D", 30: "30D", 93: "3M", 184: "6M", 366: "1Y"}


def _rounded(values, decimals: int) -> list[float | None]:
    """Rounded floats with NaN as None (JSON null): no new number, only rounding (ADR 0001 D1)."""
    return [None if value != value else round(float(value), decimals) for value in values]


def check_term_structure(payload: dict[str, Any]) -> None:
    """The invariants the schema cannot express (data-model.md 5.2). Raises ``ValueError``."""
    dates = payload["dates"]
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError("term_structure: dates are not strictly increasing")
    days = [tenor["tenor_days"] for tenor in payload["tenors"]]
    if days != sorted(days) or len(set(days)) != len(days):
        raise ValueError("term_structure: tenors are not sorted by tenor_days")
    if len(payload["levels"]) != len(payload["tenors"]):
        raise ValueError("term_structure: one levels row per tenor is required")
    for name, column in [*(("levels", row) for row in payload["levels"]), ("vix_vix3m", payload["vix_vix3m"]), ("backwardation", payload["backwardation"])]:
        if len(column) != len(dates):
            raise ValueError(f"term_structure: {name} has {len(column)} values for {len(dates)} dates")


def term_structure_payload(protocol, iv) -> dict[str, Any]:
    """Reshape and round what ``ivf`` computes; compute nothing new (ADR 0001 D1).

    Dates are the sessions of the master calendar from the first session on
    which every term-structure tenor has a close (VIX9D starts last) to
    ``data_end``. Gaps inside that span stay null.
    """
    from ivf import protocol as protocol_mod
    from ivf.data import calendar
    from ivf.features import build

    tenors = protocol_mod.term_structure(protocol_mod.load_iv_indices())
    symbols = [tenor.symbol for tenor in tenors]
    firsts = [iv.loc[iv["symbol"] == symbol, "date"].min() for symbol in symbols]
    if any(first != first or first is None for first in firsts):
        raise ModeUnavailable("the interim table lacks a term-structure index; run `python -m ivf.data build-interim`")
    start = max(firsts)
    data_end = protocol.require_data_end()
    dates = [d for d in calendar.xnys_sessions(start, data_end)]
    panel = build.term_structure_panel(dates, iv, symbols)
    payload = {
        "dates": [d.isoformat() for d in dates],
        "tenors": [
            {"symbol": tenor.symbol, "tenor_days": tenor.tenor_days, "label": TENOR_LABELS.get(tenor.tenor_days, f"{tenor.tenor_days}D")}
            for tenor in tenors
        ],
        "levels": [_rounded(panel["levels"][symbol], LEVEL_DECIMALS) for symbol in symbols],
        "vix_vix3m": _rounded(panel["vix_vix3m"], RATIO_DECIMALS),
        "backwardation": [None if flag != flag else bool(flag) for flag in panel["backwardation"]],
    }
    check_term_structure(payload)
    return payload


def generate_real(*, git: contract.GitState, generated_at: datetime | None = None) -> dict[str, bytes]:
    """``manifest.json`` and ``term_structure.json`` from real data under ``ROOT``.

    Inputs: ``ROOT/interim/*.parquet`` (``python -m ivf.data build-interim``) and
    the frozen ``configs/protocol.yaml``. Without them this raises
    ``ModeUnavailable`` (exit 4) and says which step to run.
    """
    from ivf import protocol as protocol_mod
    from ivf.data import build_interim
    from ivf.data.process import run_id_for

    try:
        protocol = protocol_mod.load_protocol(protocol_mod.protocol_path())
        protocol.require_data_end()
        iv, _price, report = build_interim.read_interim(config.data_layout().root)
    except (protocol_mod.ProtocolError, build_interim.InterimError) as exc:
        raise ModeUnavailable(f"real export needs the milestone 1 pipeline's output: {exc}") from exc
    term = contract.build_envelope(TERM_STRUCTURE, term_structure_payload(protocol, iv), "real", git=git, generated_at=generated_at)
    term_blob = contract.serialise(term)
    contract.validate_document(json.loads(term_blob))
    document = protocol.document
    manifest_payload_real = {
        "run_id": run_id_for(protocol.protocol_hash, git, "real", [entry["raw_sha256"] for entry in report["files"]]),
        "notice": REAL_NOTICE,
        "protocol": {
            "protocol_hash": protocol.protocol_hash,
            **{key: document[key] for key in ("train_start", "dev_start", "holdout_start", "data_end", "horizons", "fold_len", "min_cell_n", "holdout_looks", "vix_regime_edges")},
        },
        "files": [
            {
                "dataset": TERM_STRUCTURE,
                "path": f"{TERM_STRUCTURE}.json",
                "sha256": hashlib.sha256(term_blob).hexdigest(),
                "bytes": len(term_blob),
                "license_class": "series",
            }
        ],
        "models": [],
        "targets": [{key: target[key] for key in ("target_id", "label", "units", "horizons")} for target in document["targets"]],
    }
    manifest = contract.build_envelope(MANIFEST, manifest_payload_real, "real", git=git, generated_at=generated_at)
    manifest_blob = contract.serialise(manifest)
    contract.validate_document(json.loads(manifest_blob))
    return {f"{MANIFEST}.json": manifest_blob, f"{TERM_STRUCTURE}.json": term_blob}


# --------------------------------------------------------------------------
# The licensing guard
# --------------------------------------------------------------------------


# The guard itself lives in ivf.repo_guard, shared with the data pipeline (amended M1, review A).
# These names stay so that callers and tests of this tool keep working unchanged.
_identity = repo_guard.identity
_refuse_unreadable = repo_guard.refuse_unreadable
_directories_inside = repo_guard.directories_inside
_why_inside_a_checkout = repo_guard.why_inside_a_checkout


def resolve_output_dir(mode: str, out_dir: Path | None) -> Path:
    """The folder to write to (or, with --check, compare against).

    For ``real`` mode, raises ``RepoWriteRefused`` if the folder is inside the repo.
    """
    if out_dir is None:
        out_dir = synthetic_dir() if mode == "synthetic" else config.data_layout().app_data_real
    # resolve() follows symlinks and removes "..", so a symlink into the repo or
    # "../implied-vol-forecasting/app" is already seen for what it is. It works
    # for folders that do not exist yet. It cannot help where the filesystem
    # itself has two names for one directory (a bind mount of the repo or of any
    # folder in it), so the check below compares device and inode instead of
    # path text.
    resolved = out_dir.expanduser().resolve()
    repo = config.repo_root().resolve()
    if mode == "real":
        reason = _why_inside_a_checkout(resolved, _directories_inside(repo))
        if reason:
            raise RepoWriteRefused(
                f"refusing to use {resolved}: {reason}. Real-data exports must stay outside the "
                f"repository ({repo}), because Cboe/Yahoo data may not be redistributed (ADR 0001 D2)"
            )
    return resolved


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _refuse_unsafe_target(path: Path) -> None:
    """Refuse a symlink, a directory, or a file with more than one hard link at ``path``.

    A link is a second name for a file that may live elsewhere (inside the
    repo, say). Nobody puts one in an export folder by accident. A directory
    cannot be replaced by a file.
    """
    try:
        info = path.lstat()  # lstat: look at the link itself, not at what it points to
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise RepoWriteRefused(f"refusing to write {path}: it is a symlink")
    if stat.S_ISDIR(info.st_mode):  # checked before st_nlink: a directory always has 2 or more
        raise RepoWriteRefused(f"refusing to write {path}: it is a directory")
    if info.st_nlink > 1:
        raise RepoWriteRefused(f"refusing to write {path}: it has {info.st_nlink} hard links")


def write_files(files: dict[str, bytes], directory: Path, *, outside_repo: bool = True) -> None:
    """Write ``files`` into ``directory`` without writing through a link.

    Every target is checked before the first byte is written, so a refusal
    leaves the folder as it was. Each file is then written by
    ``ivf.repo_guard.write_file``, the one write function the data pipeline
    uses too (amended M1, review B): a temporary file in the same folder (an
    exclusive create, so a link planted at that name is never followed), moved
    over the target with ``os.replace``. Replacing a name never follows it: a
    symlink or hard link planted at the target (after the check above, say) is
    swapped for the new file, and the file it pointed to is left alone.
    ``Path.write_bytes`` would have written into that file.

    ``outside_repo`` (the default, and what ``--mode real`` passes) makes
    ``write_file`` check the folder against the repository *at write time*, and
    write through the folder it checked. ``--mode synthetic`` writes into the
    repo's ``app/public/data/synthetic/`` on purpose and passes False.

    A file that is replaced keeps its permission bits (a 0o600 file stays 0o600);
    a new file gets what any new file gets, 0o666 minus the umask.

    Residual risk: ``ivf.repo_guard``'s module docstring states it.
    """
    targets = {directory / name: blob for name, blob in files.items()}
    for target in targets:
        _refuse_unsafe_target(target)
    for target, blob in targets.items():
        repo_guard.write_file(target, blob, outside_repo=outside_repo)


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def _without_provenance(document: dict[str, Any]) -> dict[str, Any]:
    """The document minus the fields that say which run made it."""
    kept = {k: v for k, v in document.items() if k not in ENVELOPE_PROVENANCE}
    payload = kept.get("payload")
    if isinstance(payload, dict):
        kept["payload"] = {k: v for k, v in payload.items() if k not in PAYLOAD_PROVENANCE}
        if kept.get("dataset") == MANIFEST and isinstance(payload.get("files"), list):
            kept["payload"]["files"] = [
                {k: v for k, v in entry.items() if k not in FILE_ENTRY_PROVENANCE} if isinstance(entry, dict) else entry
                for entry in payload["files"]
            ]
    return kept


def _manifest_mismatches(manifest: dict[str, Any], directory: Path) -> list[str]:
    """Each file the manifest lists must exist beside it with exactly the listed sha256 and size."""
    problems = []
    for entry in manifest.get("payload", {}).get("files", []):
        path = directory / entry["path"]
        if not path.is_file():
            problems.append(f"differs: {directory / 'manifest.json'} lists {entry['path']}, which is missing")
            continue
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"] or len(blob) != entry["bytes"]:
            problems.append(f"differs: {directory / 'manifest.json'} lists a sha256 or size for {entry['path']} that the file does not have")
    return problems


def _same_scalar(a: Any, b: Any) -> bool:
    # In Python True == 1, but in JSON `true` and `1` are different things.
    # Plain numbers compare by value, so 15 and 15.0 agree.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def _first_difference(a: Any, b: Any, path: str = "$") -> str | None:
    """Location of the first difference between two parsed JSON values, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key} exists on one side only"
            found = _first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} has {len(a)} items vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if _same_scalar(a, b) else f"{path}: {a!r} vs {b!r}"


def check(expected: dict[str, bytes], directory: Path) -> list[str]:
    """Compare a fresh export with the files in ``directory``. Empty list = match.

    Exact comparison apart from the provenance fields (``ENVELOPE_PROVENANCE``,
    ``PAYLOAD_PROVENANCE`` and, in a manifest, ``FILE_ENTRY_PROVENANCE``, which
    are checked against the files on disk instead). ADR 0001 D5 also allows a
    difference of one unit in the last exported decimal for floats. Milestone 1
    still compares exactly: its only rounded floats are in the real
    term_structure.json, whose --check is local and reruns on the same interim
    parquet, so the bytes are deterministic. The tolerance is due when a
    committed fixture first carries a rounded float.
    """
    problems: list[str] = []
    for name, blob in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        try:
            committed = json.loads(path.read_bytes())
            contract.validate_document(committed)
        except ValueError as exc:  # JSONDecodeError and the contract's own errors are all ValueErrors
            problems.append(f"invalid: {path}: {exc}")
            continue
        difference = _first_difference(_without_provenance(committed), _without_provenance(json.loads(blob)))
        if difference:
            problems.append(f"differs: {path} at {difference}")
        elif name == f"{MANIFEST}.json":  # same content: now its file list must match the files on disk
            problems += _manifest_mismatches(committed, directory)
    # A file nobody generates any more is stale (for example a dataset that was dropped).
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        if path.name not in expected:
            problems.append(f"unexpected: {path} is not produced by this exporter")
    return problems


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the JSON files the React app reads.")
    parser.add_argument("--mode", required=True, choices=contract.DATA_MODES)
    parser.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    parser.add_argument("--out-dir", type=Path, help="override the default folder (real mode must stay outside the repo)")
    args = parser.parse_args(argv)

    try:
        out_dir = resolve_output_dir(args.mode, args.out_dir)
        files = generate(args.mode)
    except RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ModeUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    if args.check:
        problems = check(files, out_dir)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return EXIT_CHECK_FAILED
        print(f"ok: {len(files)} file(s) in {out_dir} match a fresh {args.mode} export")
        return EXIT_OK

    try:
        write_files(files, out_dir, outside_repo=args.mode == "real")
    except RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:  # read-only folder or filesystem, disk full, a file where the folder should be
        print(f"error: cannot write to {out_dir}: {exc}", file=sys.stderr)
        return EXIT_WRITE_FAILED
    for name, blob in files.items():
        print(f"wrote {out_dir / name} ({len(blob)} bytes)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
