"""Write the JSON files the React app reads (ADR 0001 D1, D2, D5).

    uv run python tools/export_app_data.py --mode synthetic          # write the fixture
    uv run python tools/export_app_data.py --mode synthetic --check  # CI: is the fixture current?

Milestone 0 produces one file, ``manifest.json``. The data files
(term_structure, model_race, error_surface, equity_curves) arrive with the
milestones that compute them, and each one is added to the manifest's ``files``
list only once it exists (data-model.md 5.1).

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
3 refused by the licensing guard, 4 mode not available yet in this milestone,
5 the output folder could not be written (permissions, read-only filesystem,
disk full; the message says which).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ivf import config, contract

MANIFEST = "manifest"

# Provenance: facts about *which run* made a file, not about what is in it. They
# differ on every export, so --check ignores them (ADR 0001 D5, amended M0). The
# fixture is allowed to be stale in these, because --check is what proves the
# *content* still matches the code.
#   * generated_at, source_commit, source_dirty: envelope fields.
#   * run_id: manifest payload field. It is a hash of source_commit and
#     source_dirty (among other things), so it changes whenever they do.
ENVELOPE_PROVENANCE = ("generated_at", "source_commit", "source_dirty")
PAYLOAD_PROVENANCE = ("run_id",)

EXIT_OK, EXIT_CHECK_FAILED, EXIT_REFUSED, EXIT_UNAVAILABLE, EXIT_WRITE_FAILED = 0, 1, 3, 4, 5

# Milestone 0 stand-in for configs/protocol.synthetic.yaml, which does not exist
# yet. Only the fields the manifest schema exposes. M1 replaces this dict with
# the YAML file and computes protocol_hash from the whole file, which changes
# the hash below. Dates follow ADR 0001 4.6: holdout_start is the proposed
# 2024-07-01 (open item O1). data_end is a placeholder: the last complete session
# measured on 2026-10-27. M1 replaces it with the value it freezes.
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


class RepoWriteRefused(Exception):
    """The licensing guard stopped an export.

    Raised for an output folder that is inside a git checkout (real mode), and
    for an output file that is a symlink or a hard link (any mode).
    """


class ModeUnavailable(Exception):
    """The requested mode has no implementation in this milestone."""


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
    if mode != "synthetic":
        raise ModeUnavailable(
            f"{mode!r} export is not available in milestone 0: there is no real-data pipeline until milestone 1"
        )
    # One git lookup, shared by the envelope and by run_id, so they cannot disagree.
    git = git if git is not None else contract.git_state()
    envelope = contract.build_envelope(MANIFEST, manifest_payload(git), mode, git=git, generated_at=generated_at)
    blob = contract.serialise(envelope)
    # Validate the bytes we are about to write, parsed back, not the dict we
    # built: that is what the app will actually read.
    contract.validate_document(json.loads(blob))
    return {f"{MANIFEST}.json": blob}


# --------------------------------------------------------------------------
# The licensing guard
# --------------------------------------------------------------------------


def _identity(path: Path) -> tuple[int, int]:
    """Which directory ``path`` really is on disk: (device, inode).

    Two spellings of one directory (a bind mount at a second path, or a case
    variant such as ``.../IVF`` and ``.../ivf`` on a case-insensitive filesystem)
    have different text but the same pair. Comparing text cannot see that;
    comparing this can.
    """
    info = path.stat()
    return (info.st_dev, info.st_ino)


def _refuse_unreadable(error: OSError) -> None:
    raise RepoWriteRefused(
        f"cannot read {error.filename} inside the repository ({error.strerror}), "
        "so the guard cannot tell whether an output folder is one of the repository's own folders. "
        "If the repository was changing while the guard looked (a folder deleted or moved), run it again"
    )


def _directories_inside(repo: Path) -> set[tuple[int, int]]:
    """The identity of ``repo`` and of every folder below it.

    A bind mount of *any* folder of the repo, not only of its root, shows up
    elsewhere as a folder with one of these identities. Symlinks are not
    followed: a link inside the repo that points outside it does not make its
    target part of the repo (git stores the link, not the target's files).

    node_modules, .venv and .git are walked too. The walk takes about 0.07 s on
    this repo (about 5,900 folders, mostly those three), so skipping them would
    save nothing worth the hole it leaves.

    A folder that cannot be read, or that disappears between being listed and
    being looked at (deleted or moved by something else), is refused rather
    than skipped. Skipping would be harmless for a deleted folder, since nothing
    can be written into it. But a *moved* folder vanishes from its old path in
    just the same way and is still there, possibly somewhere the walk never
    reaches, so its identity would be missing from the set and a mount of it
    would pass. The two cannot be told apart from here, a rerun costs 0.1 s, and
    failing closed is what every other case in this guard does.
    """
    found = set()
    for folder, _, _ in os.walk(repo, followlinks=False, onerror=_refuse_unreadable):
        try:
            found.add(_identity(Path(folder)))
        except OSError as error:  # os.walk already listed this folder, so it was there a moment ago
            _refuse_unreadable(error)
    return found


def _why_inside_a_checkout(directory: Path, repo_dirs: set[tuple[int, int]]) -> str | None:
    """Why ``directory`` counts as part of a git checkout, or ``None`` if it does not.

    ``repo_dirs`` is ``_directories_inside(repo)``. Walks from ``directory`` up
    to ``/`` and looks at every folder that exists (the output folder itself may
    not have been created yet, so the first existing ancestor is where the real
    answer starts). Two things refuse:

    * the folder is the same directory as the repo or one of its folders (same
      device and inode), whatever the path text says;
    * the folder has a ``.git`` entry (a directory, or the file that git
      worktrees and submodules use). That also refuses any *other* clone or
      work tree of this project, which ``repo_dirs`` would not contain.
    """
    for candidate in (directory, *directory.parents):
        if not candidate.exists():
            continue
        if _identity(candidate) in repo_dirs:
            return f"it is in the repository: on disk, {candidate} is the same folder as one inside it"
        if os.path.lexists(candidate / ".git"):
            return f"it is in a git checkout: {candidate} has a .git entry"
    return None


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


def _mode_of(path: Path) -> int | None:
    """The permission bits of the file at ``path``, or ``None`` if there is none."""
    try:
        return stat.S_IMODE(path.lstat().st_mode)
    except FileNotFoundError:
        return None


def write_files(files: dict[str, bytes], directory: Path) -> None:
    """Write ``files`` into the existing ``directory`` without writing through a link.

    Every target is checked before the first byte is written, so a refusal
    leaves the folder as it was. Each file then goes to a temporary name in the
    same folder and is moved over the target with ``os.replace``. Replacing a
    name never follows it: a symlink or hard link planted at the target (after
    the check above, say) is swapped for the new file, and the file it pointed
    to is left alone. ``Path.write_bytes`` would have written into that file.

    A file that is replaced keeps its permission bits (a 0o600 file stays 0o600);
    a new file gets what any new file gets, 0o666 minus the umask.

    Residual risk, accepted: the guard in ``resolve_output_dir`` checks the
    folder, and this function writes into it a moment later. Someone who can
    swap a parent folder for a symlink in between can still redirect the write.
    That needs write access to a parent of the output folder, and anyone with
    that already owns this account, so it is not worth a defence here.
    """
    targets = {directory / name: blob for name, blob in files.items()}
    for target in targets:
        _refuse_unsafe_target(target)
    for target, blob in targets.items():
        # O_EXCL: the temporary name must be new, so it cannot be a planted link either.
        # 0o666 is trimmed by the umask, like write_bytes did, instead of mkstemp's private 0o600.
        existing_mode = _mode_of(target)
        temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
            if existing_mode is not None:
                os.chmod(temporary, existing_mode)  # the rename would otherwise swap in the umask's mode
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def _without_provenance(document: dict[str, Any]) -> dict[str, Any]:
    """The document minus the fields that say which run made it."""
    kept = {k: v for k, v in document.items() if k not in ENVELOPE_PROVENANCE}
    payload = kept.get("payload")
    if isinstance(payload, dict):
        kept["payload"] = {k: v for k, v in payload.items() if k not in PAYLOAD_PROVENANCE}
    return kept


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

    Exact comparison apart from the provenance fields (``ENVELOPE_PROVENANCE``
    and ``PAYLOAD_PROVENANCE``). ADR 0001 D5 also allows a difference of one unit
    in the last exported decimal for floats. That tolerance arrives in milestone 1
    together with the first rounded float field; the milestone 0 manifest has none.
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
        out_dir.mkdir(parents=True, exist_ok=True)
        write_files(files, out_dir)
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
