"""Command line for the data pipeline (milestone 1).

    uv run python -m ivf.data fetch [--max-age 1d] [--force] [--symbols VIX ^GSPC]
    uv run python -m ivf.data build-interim [--snapshot cboe/VIX=<sha256> ...]
    uv run python -m ivf.data verify
    uv run python -m ivf.data check [--freeze-data-end]
    uv run python -m ivf.data eda

Add ``--synthetic`` before the command to run the same steps on seeded
synthetic data under ``ROOT/synthetic/`` with ``configs/protocol.synthetic.yaml``.
There, ``fetch`` writes the generator's Cboe- and Yahoo-shaped CSVs into the raw
store instead of downloading, and everything after it is the real code path.

Exit codes: 0 success; 1 a verification or data check failed; 2 bad command
line; 3 missing or malformed input (no data yet, a parse error, an unfrozen
protocol), or the licensing guard refused a data root inside the repository
(the same code and wording as ``tools/export_app_data.py``; amended M1,
review A); 5 a download failed.

Every command runs ``ivf.repo_guard.require_data_root_outside_repo`` on the
data root first, so ``VOL_DATA_DIR`` pointing into the checkout (directly, via
a symlink, or via a second name for one of its folders), or a ``raw/``,
``interim/``, ``processed/``, ``eda/`` or ``app-data/`` under it that leads into
the checkout, stops before anything is read or written. Every file is then
written through ``repo_guard.write_file``, which checks its folder again at
write time; a refusal there also exits 3 (amended M1, review B).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from ivf import config, protocol, repo_guard
from ivf.data import build_interim, fetch, raw

EXIT_OK, EXIT_FAILED, EXIT_INPUT, EXIT_NETWORK = 0, 1, 3, 5


def _registry():
    return protocol.load_iv_indices(), protocol.load_instruments()


def cmd_fetch(args, layout) -> int:
    indices, instruments = _registry()
    if args.synthetic:
        from ivf.data import synthetic

        written = synthetic.write_synthetic_raw(layout.root, protocol.load_protocol(protocol.protocol_path(synthetic=True)))
        for record in written:
            print(f"synthetic {record.symbol:6} {record.bytes} bytes, sha256 {record.sha256[:12]}, changed={record.changed}")
        return EXIT_OK
    try:
        max_age = fetch.parse_max_age(args.max_age)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    requests_ = fetch.requests_for(indices, instruments)
    if args.symbols:
        unknown = set(args.symbols) - {r.symbol for r in requests_}
        if unknown:
            print(f"error: unknown symbols {sorted(unknown)}", file=sys.stderr)
            return 2
        requests_ = [r for r in requests_ if r.symbol in args.symbols]
    failed = 0
    for request in requests_:
        try:
            outcome = fetch.fetch_one(layout.root, request, max_age=max_age, force=args.force)
        except repo_guard.RepoWriteRefused:
            raise  # not a failed download: stop everything, exit 3 (main)
        except Exception as exc:  # one failed symbol must not hide the others; the exit code reports it
            failed += 1
            print(f"FAILED {request.source}/{request.symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"{outcome.action:7} {outcome.source}/{outcome.symbol}: {outcome.message}")
    return EXIT_NETWORK if failed else EXIT_OK


def cmd_build_interim(args, layout) -> int:
    indices, instruments = _registry()
    try:
        pins = build_interim.parse_pins(args.snapshot or [])
        report = build_interim.build_interim(
            layout.root, [i.symbol for i in indices], [i.ticker for i in instruments], pins
        )
    except (build_interim.InterimError, raw.RawStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for entry in report.files:
        print(
            f"{entry['table']:15} {entry['source']}/{entry['symbol']:6} {entry['rows']:6} rows "
            f"{entry['first_date']}..{entry['last_date']} sha256 {entry['raw_sha256'][:12]} "
            f"duplicates dropped {entry['duplicates_dropped']}"
        )
    print(f"wrote {layout.interim}")
    return EXIT_OK


def cmd_verify(args, layout) -> int:
    problems, warnings = raw.verify(layout.root)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for problem in problems:
        print(f"MISMATCH {problem}", file=sys.stderr)
    n_files = len(list((layout.raw).rglob("*.csv"))) if layout.raw.is_dir() else 0
    if problems:
        print(f"verify FAILED: {len(problems)} problem(s) in {layout.raw}", file=sys.stderr)
        return EXIT_FAILED
    print(f"ok: {n_files} raw file(s) in {layout.raw} match their sha256 and the provenance log")
    return EXIT_OK


def cmd_check(args, layout) -> int:
    from ivf.data import check

    return check.run_check(layout, synthetic=args.synthetic, freeze=args.freeze_data_end, now=datetime.now(timezone.utc))


def cmd_eda(args, layout) -> int:
    from ivf.data import process

    return process.run_eda(layout, synthetic=args.synthetic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ivf.data", description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", action="store_true", help="use ROOT/synthetic and the synthetic protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch", help="download into the raw store (ADR 0001 D3)")
    p.add_argument("--max-age", default="1d", help="skip symbols fetched more recently than this (default 1d)")
    p.add_argument("--force", action="store_true", help="fetch even if fresh")
    p.add_argument("--symbols", nargs="+", help="only these symbols")
    p = commands.add_parser("build-interim", help="raw snapshots to interim parquet")
    p.add_argument("--snapshot", action="append", help="pin SOURCE/SYMBOL=SHA256 instead of the latest (repeatable)")
    commands.add_parser("verify", help="re-hash every raw file")
    p = commands.add_parser("check", help="master calendar, data_check and the data_end freeze")
    p.add_argument("--freeze-data-end", action="store_true", help="write data_end into configs/protocol.yaml if unset")
    commands.add_parser("eda", help="features, targets and EDA into processed/ and eda/ (local only)")
    args = parser.parse_args(argv)
    layout = config.data_layout(synthetic=args.synthetic)
    try:
        repo_guard.require_data_root_outside_repo(layout.root)
    except repo_guard.RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return repo_guard.EXIT_REFUSED
    handler = {
        "fetch": cmd_fetch,
        "build-interim": cmd_build_interim,
        "verify": cmd_verify,
        "check": cmd_check,
        "eda": cmd_eda,
    }[args.command]
    try:
        return handler(args, layout)
    except protocol.ProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except repo_guard.RepoWriteRefused as exc:  # a folder that changed after the first check (review B)
        print(f"error: {exc}", file=sys.stderr)
        return repo_guard.EXIT_REFUSED
