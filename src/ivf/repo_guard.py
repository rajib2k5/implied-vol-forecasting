"""The licensing guard: real data is never written inside a git checkout (ADR 0001 D2).

One implementation, two callers (amended M1, review A):

* ``tools/export_app_data.py --mode real`` refuses an output folder inside the repo;
* the data pipeline (``python -m ivf.data`` and its library entry points ``fetch_one``,
  ``write_synthetic_raw``, ``build_interim``, ``run_check`` and ``run_eda``) refuses a data
  root (``ROOT``, from ``VOL_DATA_DIR``) inside the repo.

Review A found that only the export tool had this guard: ``VOL_DATA_DIR=<repo>/localdata`` let
``fetch``, ``build-interim`` and ``eda`` write raw snapshots, hashes and every EDA number into
untracked, unignored files of the checkout. ``.gitignore`` covered only ``/data/``.

Why device and inode, not path text. Two spellings of one directory (a symlink, a bind mount at a
second path, a case variant on a case-insensitive filesystem) have different text but the same
``(st_dev, st_ino)``. ``Path.resolve()`` removes symlinks and ``..``, but it cannot see a bind
mount or a case variant, so the guard compares the identity of every existing ancestor of the
target with the identity of every folder inside the repo. A ``.git`` entry on an ancestor also
refuses, which covers any *other* clone or worktree of the project.

**Every write goes through ``write_file``** (amended M1, review B). Review B found that the guard
looked only at ``ROOT``: with ``ROOT`` outside the repo but ``ROOT/raw`` (or ``ROOT/eda``) a symlink
into the checkout, ``fetch`` wrote 8 snapshots and ``eda`` wrote ``eda_report.json`` into it, rc 0.
Now two layers stop that:

1. ``require_data_root_outside_repo`` checks ``ROOT`` *and* every folder the pipeline writes under
   it (``DATA_SUBFOLDERS``), before the first write, so a symlinked ``eda/`` stops ``eda`` before
   ``processed/`` is touched (exit 3, nothing written);
2. ``write_file``, the one function every writer uses (raw snapshots, the provenance log, interim
   and processed parquet, the EDA report, the synthetic store, the real-data export, and, without
   the outside-the-repo rule, the ``data_end`` freeze and the synthetic export), checks the folder
   it writes into *at write time*, then opens it, checks that the folder it holds open is the one
   it checked (same device and inode, same resolved path), and writes through that open handle
   (``dir_fd``). A folder swapped for a symlink after the check is caught there; one swapped after
   the open is not written through, because the writes name the open folder, not a path.

Residual risk (stated, not defended): a folder swapped between ``write_file``'s check and its open
is caught; one *moved into* the repo after the open, while the write runs, is not. That needs
write access to the folder's parent, which already means owning this account. Folders created by
``os.makedirs`` before the write-time check are created only after the first check passed, so at
worst an empty folder appears (the race above). The repo's folder identities are read once per
process per repo (``repo_identities``): a folder under the repo is still refused through its
ancestors (the repo root is always in the set, and has the ``.git`` entry); only a *bind mount* of a
folder created inside the repo after that first read would go unseen.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

#: The exit code both command lines use for a refusal (``tools/export_app_data.py`` EXIT_REFUSED).
EXIT_REFUSED = 3


class RepoWriteRefused(Exception):
    """The licensing guard stopped a write.

    Raised for a folder inside a git checkout, and (by the export tool) for an output file that is
    a symlink or a hard link.
    """


def identity(path: Path) -> tuple[int, int]:
    """Which directory ``path`` really is on disk: ``(device, inode)``."""
    info = path.stat()
    return (info.st_dev, info.st_ino)


def refuse_unreadable(error: OSError) -> None:
    raise RepoWriteRefused(
        f"cannot read {error.filename} inside the repository ({error.strerror}), "
        "so the guard cannot tell whether an output folder is one of the repository's own folders. "
        "If the repository was changing while the guard looked (a folder deleted or moved), run it again"
    )


def directories_inside(repo: Path) -> set[tuple[int, int]]:
    """The identity of ``repo`` and of every folder below it.

    A bind mount of *any* folder of the repo, not only of its root, shows up elsewhere as a folder
    with one of these identities. Symlinks are not followed: a link inside the repo that points
    outside it does not make its target part of the repo (git stores the link, not the target).

    node_modules, .venv and .git are walked too: about 6,100 folders in 0.08 s on this repo
    (measured 2026-10-29), so skipping them would save nothing worth the hole it leaves.

    A folder that cannot be read, or that disappears between being listed and being looked at, is
    refused rather than skipped: a *moved* folder vanishes from its old path in the same way and
    may still be mounted somewhere the walk never reaches. A rerun costs 0.1 s; failing closed is
    what every other case here does.
    """
    found = set()
    for folder, _, _ in os.walk(repo, followlinks=False, onerror=refuse_unreadable):
        try:
            found.add(identity(Path(folder)))
        except OSError as error:  # os.walk listed this folder a moment ago
            refuse_unreadable(error)
    return found


def why_inside_a_checkout(directory: Path, repo_dirs: set[tuple[int, int]]) -> str | None:
    """Why ``directory`` counts as part of a git checkout, or ``None`` if it does not.

    Walks from ``directory`` up to ``/`` and looks at every folder that exists (the folder itself
    may not exist yet). Two things refuse: an ancestor that is the same directory as one inside the
    repo (same device and inode, whatever the text says), and an ancestor with a ``.git`` entry (a
    directory, or the file that worktrees and submodules use).
    """
    for candidate in (directory, *directory.parents):
        if not candidate.exists():
            continue
        if identity(candidate) in repo_dirs:
            return f"it is in the repository: on disk, {candidate} is the same folder as one inside it"
        if os.path.lexists(candidate / ".git"):
            return f"it is in a git checkout: {candidate} has a .git entry"
    return None


_IDENTITIES: dict[Path, set[tuple[int, int]]] = {}


def repo_identities() -> tuple[Path, set[tuple[int, int]]]:
    """``(repo, directories_inside(repo))`` for the checkout this code runs from, read once per process.

    Cached per repo path because ``write_file`` asks before every write (about 0.1 s per walk). See
    the module docstring for the one case a cached set can miss. A failed walk is not cached.
    """
    from ivf import config

    repo = config.repo_root().resolve()
    if repo not in _IDENTITIES:
        _IDENTITIES[repo] = directories_inside(repo)
    return repo, _IDENTITIES[repo]


def _refuse_inside_repo(resolved: Path, what: str = "The data root") -> None:
    from ivf import config

    repo, identities = repo_identities()
    reason = why_inside_a_checkout(resolved, identities)
    if reason:
        raise RepoWriteRefused(
            f"refusing to use {resolved}: {reason}. {what} must stay outside the repository ({repo}), "
            f"because Cboe/Yahoo data may not be redistributed (ADR 0001 D2); set {config.ENV_VAR} to a folder outside it"
        )


#: The folders the pipeline writes under a data root (config.DataLayout), checked before the first write.
DATA_SUBFOLDERS = ("raw", "interim", "processed", "eda", "app-data")


def require_data_root_outside_repo(root: Path) -> Path:
    """Refuse a data root (``ROOT`` or ``ROOT/synthetic``) inside the repo; return it resolved.

    Also refuses the root when any folder the pipeline writes under it (``DATA_SUBFOLDERS``) leads
    into the repo, through a symlink or a second name (amended M1, review B: a symlinked ``raw/``
    or ``eda/`` got past a check of the root alone). Raises ``RepoWriteRefused`` with the same
    wording as the export tool's refusal. Called by every pipeline stage that writes under the
    root, before its first write; ``write_file`` checks again at each write.
    """
    resolved = Path(root).expanduser().resolve()
    _refuse_inside_repo(resolved)
    for name in DATA_SUBFOLDERS:
        _refuse_inside_repo((resolved / name).resolve(), f"The data folder {resolved / name}")
    return resolved


def _open_folder(folder: Path, outside_repo: bool) -> tuple[int, Path]:
    """Check ``folder``, create it if needed, open it, and check that the open folder is the checked one."""
    resolved = Path(folder).expanduser().resolve()
    if outside_repo:
        _refuse_inside_repo(resolved, f"The folder {folder}")  # before anything is created
    os.makedirs(resolved, exist_ok=True)
    handle = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if outside_repo:
            held = os.fstat(handle)
            again = resolved.resolve()
            if again != resolved or identity(again) != (held.st_dev, held.st_ino):
                raise RepoWriteRefused(f"refusing to write into {folder}: it changed while it was being checked")
            _refuse_inside_repo(again, f"The folder {folder}")  # it exists now, so its own identity counts too
    except BaseException:
        os.close(handle)
        raise
    return handle, resolved


def write_file(path: Path, content: bytes, *, outside_repo: bool, how: str = "replace") -> bool:
    """Write ``content`` to ``path``: the one way the pipeline and the export tool write a file (amended M1, review B).

    ``outside_repo=True`` (all data, and anything derived from it) refuses a folder inside the repo,
    at write time (module docstring). ``False`` is for the files that belong in the repo: the
    ``data_end`` freeze in ``configs/`` and the synthetic export.

    ``how``:

    * ``"replace"``: write a temporary file in the same folder (exclusive create, so a planted link
      at that name is never followed), flush it to disk, and rename it over ``path``. A reader sees
      the old file or the new one, never half of one. A replaced file keeps its permission bits; a
      new one gets 0o666 minus the umask.
    * ``"new"``: the same temporary file, then *linked* to ``path``, which fails if the name exists:
      "create only if absent" in one atomic step (the raw store). Returns False if it existed.
    * ``"append"``: append to ``path`` (created if absent), refusing a symlink there, and a file
      with more than one name (a hard link; checked on the open descriptor, M1 final check), and
      flush. ``"replace"`` and ``"new"`` need no such check: they write a freshly created
      temporary file and rename or link it into place, so the bytes never reach an existing inode.

    Every mode flushes the folder too (``fsync`` on it) after a name appears or changes, so a crash
    just after the call cannot lose the new name (amended M1, review B; the raw store had no folder
    fsync). Returns True when the file was written.
    """
    if how not in ("replace", "new", "append"):
        raise ValueError(f"how={how!r}")
    folder, name = Path(path).parent, Path(path).name
    handle, _ = _open_folder(folder, outside_repo)
    try:
        if how == "append":
            try:
                existed = stat.S_ISREG(os.stat(name, dir_fd=handle, follow_symlinks=False).st_mode)
            except FileNotFoundError:
                existed = False
            try:
                fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o666, dir_fd=handle)
            except OSError as exc:
                if os.path.islink(path):
                    raise RepoWriteRefused(f"refusing to append to {path}: it is a symlink") from exc
                raise
            # A hard link is not a symlink: O_NOFOLLOW lets it through, and appending through it
            # writes into whatever other file shares the inode, a file in the checkout included
            # (M1 final check: 13 lines appended to an in-repo file through raw/provenance.jsonl).
            # The open file must be a regular file with this one name, checked on the open fd
            # before a byte is written.
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode) or held.st_nlink != 1:
                os.close(fd)
                raise RepoWriteRefused(
                    f"refusing to append to {path}: it is not a regular file with a single name "
                    f"(links: {held.st_nlink}); a hard link can lead into the repository"
                )
            with os.fdopen(fd, "ab") as out:
                out.write(content)
                out.flush()
                os.fsync(out.fileno())
            if not existed:
                os.fsync(handle)
            return True
        try:
            existing = os.stat(name, dir_fd=handle, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=handle)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(content)
                out.flush()
                if how == "replace" and existing is not None and stat.S_ISREG(existing.st_mode):
                    os.fchmod(out.fileno(), stat.S_IMODE(existing.st_mode))  # the rename would swap in the umask's mode
                os.fsync(out.fileno())
            if how == "new":
                try:
                    os.link(temporary, name, src_dir_fd=handle, dst_dir_fd=handle, follow_symlinks=False)
                except FileExistsError:
                    return False
            else:
                os.replace(temporary, name, src_dir_fd=handle, dst_dir_fd=handle)
            os.fsync(handle)
            return True
        finally:
            try:
                os.unlink(temporary, dir_fd=handle)
            except FileNotFoundError:
                pass
    finally:
        os.close(handle)
