"""Command line for the VXX/SVXY backtest (milestone 6).

    uv run python -m ivf.data fetch --symbols VXX SVXY     # network: the two ETPs, with provenance
    uv run python -m ivf.backtest prepare                  # interim/tradable_daily.parquet from the raw store
    uv run python -m ivf.backtest run [--print-tables]     # the dev backtest (default)
    uv run python -m ivf.backtest --synthetic prepare|run  # the same on the synthetic market

``run`` (period ``dev``):

1. checks the protocol's frozen ``data_end`` as every step does, then cuts the VIX, the ETP prices and
   the forecasts to dates before ``holdout_start`` BEFORE anything else reads them (ADR 0001 4.6);
2. checks the tradable segments against the ETP data (``segments.check_segments``): exit 1 on a gap, an
   unadjusted split, a stale print or an exposure that does not match the segment's leverage;
3. picks each track's model from M5's dev ``metric`` table by the pre-registered rule
   (``configs/backtest.yaml``) and builds every strategy's signal;
4. runs every active strategy at every cost level on the return dates of the usable segment;
5. appends the non-benchmark strategies to the trial log, then computes PSR and DSR;
6. writes ``backtest_daily``, ``strategy_summary``, ``segment_check`` and ``backtest_meta.json`` under
   ``processed/run_id=<id>/backtest/period=dev/`` (the run whose forecasts it used). Local only.

The default output is counts, dates and timings; values only with ``--print-tables`` (ADR 0001 D2).

``run --period holdout`` is THE ONE LOOK (ADR 0001 4.6, amended M6, review A). Real data needs
``--one-look``; synthetic data runs the identical path (that is how it is tested). In order:

1. **preflight** (``preflight``): every precondition, BEFORE anything is written. A failure exits 2 and
   leaves no record, so a doomed run cannot spend the look;
2. **record**: ``ROOT/backtest/holdout_look.json`` is created create-only (a second look finds it and
   stops). From here on **the look is spent**, whatever happens next;
3. **holdout forecasts** for every model in the dev store (``ivf.models.holdout``: specifications
   frozen, refits continue at every holdout cutoff);
4. **M5's holdout report** (``report.run_report(..., period="holdout")``: DM and MZ descriptive only, the
   pre-registered ``holdout_check``);
5. **the backtest** on the holdout window with the dev-chosen models, no new trials, and the
   pre-registered ``holdout_rule`` verdict.

The record's ``status`` is updated after each stage: ``complete`` at the end, ``failed at <stage>:
<error>`` if a stage fails. A hard crash leaves the last ``running <stage>``. Either way it says that
the look happened and how far it got.

``--preflight-only`` (real or synthetic, review B) runs the preflight and prints its problems or
"preflight ok"; it writes nothing and is never a look. ``--resume`` (review B; default taken
2026-10-29, owner can override) continues a look whose record says ``failed at <stage>``, from that
stage, reusing what the completed stages stored; it is refused unless the commit, the protocol, the
backtest config and the input snapshots are the look's own, so the resumed look is still the one look
(the same frozen code on the same data); every attempt is appended to the record. A record left at
``running <stage>`` by a hard crash is not resumable by the command: check that no look process is
alive, then set its status to ``failed at <stage>: crashed`` by hand and resume.

Exit codes: 0 success; 1 a data or segment check failed; 2 bad command line, a failed preflight or a
refused look; 3 missing or malformed input, or the licensing guard refused.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from ivf import config, contract, repo_guard
from ivf import protocol as protocol_mod
from ivf.backtest import data as bdata
from ivf.backtest import segments as seg
from ivf.backtest import strategy
from ivf.data import build_interim, calendar, check
from ivf.evaluation import report

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INPUT = 0, 1, 2, 3
LOOK_RECORD = "backtest/holdout_look.json"
REFUSAL = ("refusing --period holdout on real data without --one-look: the one look at the holdout (ADR 0001 4.6) is "
           "taken once, after milestone 6 is reviewed, by the owner.")


def _write(frame: pd.DataFrame, path) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    repo_guard.write_file(path, buffer.getvalue(), outside_repo=True)


def read_log(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose_evaluated_run(processed, run_id: str | None) -> str | None:
    """``run_id`` if given, else the newest run with forecasts AND an M5 dev ``metric`` table."""
    def ok(r):
        folder = processed / f"run_id={r}"
        return (folder / "forecast.parquet").exists() and (folder / "evaluation" / "period=dev" / "metric.parquet").exists()
    if run_id:
        return run_id if ok(run_id) else None
    path = processed / "runs.parquet"
    if not path.exists():
        return None
    for candidate in pd.read_parquet(path).sort_values("started_at", ascending=False)["run_id"]:
        if ok(candidate):
            return candidate
    return None


def is_pushed(commit: str | None) -> bool:
    """True when a remote-tracking branch contains ``commit`` (the tree is then reproducible by others)."""
    if not commit:
        return False
    try:
        done = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=config.repo_root(),
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and bool(done.stdout.strip())


def cmd_prepare(layout, synthetic: bool, out) -> int:
    repo_guard.require_data_root_outside_repo(layout.root)
    tickers = [i.ticker for i in protocol_mod.load_instruments() if i.role == "tradable"]
    if synthetic:
        proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=True))
        for r in bdata.write_synthetic_tradables(layout.root, proto):
            print(f"synthetic {r.symbol}: {r.bytes} bytes, changed={r.changed}", file=out)
    for row in bdata.build_tradable_table(layout.root, tickers):
        print(f"{row['ticker']}: {row['rows']} rows {row['first']} .. {row['last']} ({row['source']}, fetched {row['fetched_at']})", file=out)
    return EXIT_OK


def _load_inputs(layout, proto):
    """``(iv, price, tradables, build_report, problems)``: the interim tables after the frozen ``data_end`` check."""
    try:
        iv, price, build_report = build_interim.read_interim(layout.root)
        tradables = bdata.read_tradable_table(layout.root)
    except build_interim.InterimError as exc:
        return None, None, None, None, [str(exc)]
    result = check.frozen_check(proto, iv, price, build_report)
    return iv, price, tradables, build_report, list(result.policy_violations)


def _backtest(layout, *, proto, cfg, segments, period, chosen_run, iv, tradables, forecasts, synthetic, print_tables,
              out, clock, run_protocol_hash=None, caveat="") -> int:
    """Steps 2-6 of the dev path, or step 5 of the look: segment checks, model choice, signals, engine, tables."""
    hs, data_end = proto.holdout_start, proto.require_data_end()
    sessions = [d for d in calendar.master_calendar(proto)["date"] if d <= data_end]
    read_until = max(d for d in sessions if d < hs) if period == "dev" else data_end
    stale_until = max(d for d in sessions if d < hs)
    violations, seg_rows = seg.check_segments(tradables, segments, sessions, read_until, cfg.splits, cfg.tolerance, stale_until)
    usable = [r for r in seg_rows if r["check"] == "segment" and r["usable"]]
    print("segments: " + "; ".join(f"{r['ticker']} from {r['valid_from']} ({r['sessions']} closes to {read_until}, "
                                   f"{r['missing']} missing, {r['stale']} stale)" for r in usable)
          + f"; {sum(r['check'] == 'split' for r in seg_rows)} splits checked; {len(violations)} violations", file=out)
    if violations:
        for v in violations:
            print(f"error: {v}", file=sys.stderr)
        return EXIT_FAILED
    folder = layout.processed / f"run_id={chosen_run}"
    dev_metric = pd.read_parquet(folder / "evaluation" / "period=dev" / "metric.parquet")
    try:
        chosen = strategy.select_models(cfg, dev_metric)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    print("models (pre-registered rule on M5 dev metric): " + ", ".join(f"{k}: {v}" for k, v in chosen.items())
          + (" (rv21 track: rw, the trailing realised vol, is the dev best)" if chosen.get("rv21") == "rw" else ""), file=out)
    vix = iv[iv["symbol"] == "VIX"].set_index("date")["close"].astype(float)
    try:
        signal_map = strategy.signals(cfg, chosen, forecasts, vix, sessions)
        daily = strategy.run_period(cfg, segments, sessions, tradables, signal_map, period, hs, data_end)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    log_path = layout.root / cfg.trial_log
    log = read_log(log_path)
    git = contract.git_state()
    if period == "dev":
        meta_line = {"run_at": datetime.now(timezone.utc).isoformat(), "commit": git.commit, "dirty": git.dirty,
                     "source_run_id": chosen_run, "data_mode": "synthetic" if synthetic else "real"}
        new = strategy.trial_rows(cfg, chosen, strategy.summarise(daily, cfg, [], "dev"), meta_line)
        repo_guard.write_file(log_path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in new).encode(),
                              outside_repo=True, how="append")
        log += new
    summary = strategy.summarise(daily, cfg, log, "dev")
    destination = folder / "backtest" / f"period={period}"
    tables = {"backtest_daily": daily, "strategy_summary": summary, "segment_check": pd.DataFrame(seg_rows)}
    for name, frame in tables.items():
        _write(frame, destination / f"{name}.parquet")
    dates = sorted(daily["date"].unique())
    default = summary[summary.cost_bps == cfg.cost_bps].set_index("strategy_id")
    meta = {"period": period, "data_mode": "synthetic" if synthetic else "real", "source_run_id": chosen_run,
            "models": chosen, "first_return": str(dates[0]), "last_return": str(dates[-1]), "n_days": len(dates),
            "n_trials": int(summary.n_trials.iloc[0]), "cost_bps_default": cfg.cost_bps, "commit": git.commit,
            "dirty": git.dirty, "holdout_looks": int(proto.document["holdout_looks"]),
            "rows": {k: len(v) for k, v in tables.items()}, "trades": {k: int(v) for k, v in default.n_trades.items()}}
    meta.update(protocol_hash=proto.protocol_hash, run_protocol_hash=run_protocol_hash)
    if period == "holdout":
        meta["holdout_verdict"] = strategy.holdout_verdict(summary, cfg)
        meta["caveat"] = caveat
    repo_guard.write_file(destination / "backtest_meta.json", json.dumps(meta, indent=2, sort_keys=True).encode(), outside_repo=True)
    print(f"period {period}: returns {dates[0]} .. {dates[-1]} ({len(dates)} sessions, origins two sessions earlier, rule L7); "
          f"holdout_looks {meta['holdout_looks']}", file=out)
    print("trades at the default cost (fills, entry from cash included): "
          + ", ".join(f"{k} {v}" for k, v in meta["trades"].items()), file=out)
    print(f"trials (n_trials for DSR): {meta['n_trials']}; trial log {log_path}", file=out)
    if period == "holdout":
        print(f"holdout verdict (pre-registered): held_up = {meta['holdout_verdict']['held_up']}", file=out)
    print("rows: " + ", ".join(f"{k} {v}" for k, v in meta["rows"].items()) + f"; {time.perf_counter() - clock:.1f} s", file=out)
    print(f"tables written to {destination} (local only, never commit real numbers: ADR 0001 D2)", file=out)
    if print_tables:
        cols = ["strategy_id", "cost_bps", "n_days", "ann_return", "ann_vol", "sharpe_ann", "sortino_ann", "calmar",
                "max_drawdown", "turnover_ann", "hit_rate", "n_trades", "psr0", "e_max_sr", "dsr", "dsr_paper_var", "n_trials"]
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            print(summary[cols].round(4).to_string(index=False), file=out)
    return EXIT_OK


def run_protocol(layout, chosen_run: str, proto) -> tuple[str | None, str | None]:
    """``(run_protocol_hash, problem)``. A dev run may have been made under a protocol that differs from the
    current one ONLY in ``holdout_looks`` (review B: it went 0 -> 1 after the dev run, ADR 0001 4.6); any
    other difference is refused. Checked by re-hashing the current protocol with each earlier count."""
    path = layout.processed / "runs.parquet"
    runs = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["run_id", "protocol_hash"])
    row = runs[runs.run_id == chosen_run]
    if row.empty:
        return None, f"run {chosen_run} has no row in {path}"
    run_hash = str(row.protocol_hash.iloc[0])
    looks = int(proto.document["holdout_looks"])
    if any(contract.canonical_json_sha256({**proto.document, "holdout_looks": k}) == run_hash for k in range(looks + 1)):
        return run_hash, None
    return run_hash, (f"run {chosen_run} was made under protocol {run_hash[:12]}, which differs from the current "
                      f"{proto.protocol_hash[:12]} in more than holdout_looks")


def cmd_dev(layout, *, synthetic: bool, run_id: str | None, print_tables: bool, out) -> int:
    clock = time.perf_counter()
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    cfg, segments = strategy.load_config(), seg.load_segments()
    hs = proto.holdout_start
    iv, _, tradables, _, problems = _load_inputs(layout, proto)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return EXIT_INPUT if iv is None else EXIT_FAILED
    iv, tradables = iv[iv["date"] < hs], tradables[tradables["date"] < hs]  # nothing on or after holdout_start past here
    chosen_run = choose_evaluated_run(layout.processed, run_id)
    if chosen_run is None:
        print(f"error: no run with forecasts and an M5 dev metric table under {layout.processed} "
              "(run the models, then `python -m ivf.evaluation report`)", file=sys.stderr)
        return EXIT_INPUT
    run_hash, problem = run_protocol(layout, chosen_run, proto)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_INPUT
    forecasts = pd.read_parquet(layout.processed / f"run_id={chosen_run}" / "forecast.parquet")
    late = int((forecasts.origin_date >= hs).sum())
    forecasts = forecasts[forecasts.origin_date < hs]
    print(f"dev: {late} forecast rows with an origin on or after {hs} ignored", file=out)
    return _backtest(layout, proto=proto, cfg=cfg, segments=segments, period="dev", chosen_run=chosen_run, iv=iv,
                     tradables=tradables, forecasts=forecasts, synthetic=synthetic, print_tables=print_tables, out=out,
                     clock=clock, run_protocol_hash=run_hash)


STAGES = ("holdout forecasts", "M5 holdout report", "holdout backtest")


def _input_hashes(build_report: dict, tradables: pd.DataFrame) -> list[str]:
    """The raw snapshots the look reads: a resume must read the same ones."""
    return sorted({e["raw_sha256"] for e in build_report["files"]} | set(tradables["raw_sha256"].unique()))


def preflight(layout, *, synthetic: bool, run_id: str | None, resume: bool = False) -> tuple[list[str], dict]:
    """Every precondition of the look, checked BEFORE the record is written or changed; ``(problems, context)``.

    Reads dates, keys and configs, and the ETP closes to ``data_end`` for the segment checks (review B:
    a split left unadjusted in the holdout, a gap or a leverage change must stop the look before it is
    spent); problems name the check, never a measured value. With ``resume`` the record must exist,
    say ``failed at <stage>``, and match this commit, protocol, config and input snapshots."""
    from ivf.models import holdout as holdout_mod

    problems: list[str] = []
    ctx: dict = {"notes": []}
    try:
        proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
        cfg, segments = strategy.load_config(), seg.load_segments()
        data_end = proto.require_data_end()
    except protocol_mod.ProtocolError as exc:
        return [f"protocol or config: {exc}"], ctx
    ctx.update(proto=proto, cfg=cfg, segments=segments)
    expected = cfg.document.get("holdout_looks_before_look", {}).get("synthetic" if synthetic else "real")
    looks = int(proto.document["holdout_looks"])
    if looks != expected:
        problems.append(f"protocol holdout_looks is {looks}, configs/backtest.yaml expects {expected} before the look")
    record_path = layout.root / LOOK_RECORD
    git = contract.git_state()
    ctx["git"] = git
    if resume:
        if not record_path.exists():
            problems.append("--resume: there is no look record to resume")
        else:
            record = json.loads(record_path.read_text())
            ctx["record"] = record
            status = str(record.get("status", ""))
            failed = next((st for st in STAGES if status.startswith(f"failed at {st}")), None)
            if failed is None:
                problems.append(f"--resume: the record's status is '{status.split(':')[0]}', not 'failed at <stage>'")
            ctx["resume_stage"] = failed
            if record.get("commit") != git.commit:
                problems.append("--resume: the commit differs from the look's; a resume runs the same frozen code")
            if record.get("protocol_hash") != proto.protocol_hash:
                problems.append("--resume: the protocol differs from the look's")
            if record.get("backtest_config") != cfg.document:
                problems.append("--resume: configs/backtest.yaml differs from the look's")
    elif record_path.exists():
        problems.append(f"{record_path} exists: the one look was already taken")
    if not synthetic:
        if git.dirty or not git.commit:
            problems.append("the working tree is not clean: the look runs from a committed, pushed tree")
        elif not is_pushed(git.commit):
            problems.append(f"commit {git.commit} is not on any remote branch: push it first")
    iv, price, tradables, build_report, input_problems = _load_inputs(layout, proto)
    problems += input_problems
    if iv is None:
        return problems, ctx
    ctx.update(iv=iv, price=price, tradables=tradables, input_hashes=_input_hashes(build_report, tradables))
    if resume and "record" in ctx and ctx["record"].get("input_hashes") != ctx["input_hashes"]:
        problems.append("--resume: the input snapshots differ from the look's")
    sessions = [d for d in calendar.master_calendar(proto)["date"] if d <= data_end]
    usable = set(seg.usable_return_dates(segments, sessions))
    window = [d for d in sessions if d >= proto.holdout_start]
    if not window or any(d not in usable for d in window):
        problems.append("the usable segments do not cover every return date of the holdout window")
    # A missing close in the holdout window is a missing close inside a usable segment: check_segments below
    # reports it (review B fix round: the separate presence check here duplicated it, mutant N4).
    stale_until = max(d for d in sessions if d < proto.holdout_start)
    seg_problems, _ = seg.check_segments(tradables, segments, sessions, data_end, cfg.splits, cfg.tolerance, stale_until,
                                         values=False)
    problems += [f"segment check: {v}" for v in seg_problems]
    ctx["notes"].append(f"segment check to {data_end}: " + ("pass" if not seg_problems else f"FAIL ({len(seg_problems)})"))
    chosen = choose_evaluated_run(layout.processed, run_id)
    if chosen is None:
        problems.append("no run with forecasts and an M5 dev metric table")
        return problems, ctx
    ctx["chosen_run"] = chosen
    run_hash, problem = run_protocol(layout, chosen, proto)
    ctx["run_protocol_hash"] = run_hash
    if problem:
        problems.append(problem)
    folder = layout.processed / f"run_id={chosen}"
    if resume:
        stage = ctx.get("resume_stage")
        if stage and STAGES.index(stage) > 0 and not (folder / holdout_mod.FOLDER / "forecast.parquet").exists():
            problems.append("--resume: the holdout forecasts of the completed stage are missing")
    elif (folder / holdout_mod.FOLDER).exists():
        problems.append(f"{folder / holdout_mod.FOLDER} exists: holdout forecasts were already written")
    dev_forecast = pd.read_parquet(folder / "forecast.parquet", columns=["model_id", "target_id"])
    needed = holdout_mod.needed_models(dev_forecast)
    ctx["needed"] = needed
    try:
        strategy.select_models(cfg, pd.read_parquet(folder / "evaluation" / "period=dev" / "metric.parquet"))
    except ValueError as exc:
        problems.append(f"model choice: {exc}")
    for target_id, ids in needed.items():
        try:
            holdout_mod.build_track_models(target_id, ids)
        except Exception as exc:  # any failure to build a frozen spec stops the look before it starts
            problems.append(f"models: {exc}")
    dev_fit = pd.read_parquet(folder / "fit.parquet") if (folder / "fit.parquet").exists() else None
    ctx["dev_fit"] = dev_fit
    for target_id, ids in needed.items():
        if "lstm_log" in ids:
            try:
                holdout_mod.lstm_start_cutoff(dev_fit if dev_fit is not None else pd.DataFrame(columns=["model_id", "target_id"]), target_id)
            except (ValueError, KeyError) as exc:
                problems.append(f"lstm: {exc}")
    log = read_log(layout.root / cfg.trial_log)
    if not any(r["period"] == "dev" and r.get("source_run_id") == chosen for r in log):
        problems.append(f"no dev backtest of run {chosen} in the trial log: run the dev backtest first")
    return problems, ctx


def _update_record(layout, record: dict, status: str) -> None:
    record["status"] = status
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    record["attempts"][-1]["last_status"] = status
    repo_guard.write_file(layout.root / LOOK_RECORD, json.dumps(record, indent=2, sort_keys=True).encode(), outside_repo=True)


def cmd_look(layout, *, synthetic: bool, run_id: str | None, workers: int, print_tables: bool, out,
             resume: bool = False, preflight_only: bool = False) -> int:
    from ivf.models import holdout as holdout_mod

    clock = time.perf_counter()
    problems, ctx = preflight(layout, synthetic=synthetic, run_id=run_id, resume=resume)
    for note in ctx.get("notes", []):
        print(f"preflight: {note}", file=out)
    if preflight_only:
        for p in problems:
            print(f"preflight: {p}", file=out)
        print("preflight ok (nothing written; this is not a look)" if not problems else
              f"preflight FAILED: {len(problems)} problem(s) (nothing written; this is not a look)", file=out)
        return EXIT_OK if not problems else EXIT_USAGE
    if problems:
        for p in problems:
            print(f"preflight: {p}", file=sys.stderr)
        print("error: preflight failed; no look record was written or changed, the look is NOT spent", file=sys.stderr)
        return EXIT_USAGE
    proto, cfg, git = ctx["proto"], ctx["cfg"], ctx["git"]
    caveat = cfg.document.get("holdout_caveat", "") if not synthetic else ""
    attempt = {"started_at": datetime.now(timezone.utc).isoformat(), "commit": git.commit}
    if resume:
        record = ctx["record"]
        start = STAGES.index(ctx["resume_stage"])
        record["attempts"].append({**attempt, "from_stage": STAGES[start]})
        _update_record(layout, record, f"running {STAGES[start]}")
        print(f"resuming the look at '{STAGES[start]}' (same commit, protocol, config and inputs: still the one look)", file=out)
    else:
        start = 0
        record = {"taken_at": attempt["started_at"], "commit": git.commit, "dirty": git.dirty,
                  "data_mode": "synthetic" if synthetic else "real", "protocol_hash": proto.protocol_hash,
                  "run_protocol_hash": ctx["run_protocol_hash"], "holdout_looks": int(proto.document["holdout_looks"]),
                  "caveat": caveat, "source_run_id": ctx["chosen_run"], "backtest_config": cfg.document,
                  "input_hashes": ctx["input_hashes"], "status": f"running {STAGES[0]}",
                  "attempts": [{**attempt, "from_stage": STAGES[0], "last_status": f"running {STAGES[0]}"}],
                  "note": "the look is spent from the moment this file exists, whatever its status"}
        if not repo_guard.write_file(layout.root / LOOK_RECORD, json.dumps(record, indent=2, sort_keys=True).encode(),
                                     outside_repo=True, how="new"):
            print(f"error: {layout.root / LOOK_RECORD} exists: the one look was already taken", file=sys.stderr)
            return EXIT_USAGE
        print(f"look record written: {layout.root / LOOK_RECORD} (the look is spent from here on)"
              + (f"; caveat: {caveat}" if caveat else ""), file=out)
    folder = layout.processed / f"run_id={ctx['chosen_run']}"
    stage = STAGES[start]
    try:
        for stage in STAGES[start:]:
            if stage != STAGES[start]:
                _update_record(layout, record, f"running {stage}")
            if stage == STAGES[0]:
                holdout_mod.run_holdout_forecasts(folder, proto, ctx["iv"], ctx["price"], ctx["needed"], ctx["dev_fit"], workers, out)
            elif stage == STAGES[1]:
                code = report.run_report(layout, synthetic=synthetic, run_id=ctx["chosen_run"], period="holdout", one_look=True,
                                         out=out, caveat=caveat, run_protocol_hash=ctx["run_protocol_hash"])
                if code != EXIT_OK:
                    raise RuntimeError(f"M5's holdout report exited {code}")
            else:
                forecasts = pd.concat([pd.read_parquet(folder / "forecast.parquet"),
                                       pd.read_parquet(folder / holdout_mod.FOLDER / "forecast.parquet")], ignore_index=True)
                code = _backtest(layout, proto=proto, cfg=cfg, segments=ctx["segments"], period="holdout",
                                 chosen_run=ctx["chosen_run"], iv=ctx["iv"], tradables=ctx["tradables"], forecasts=forecasts,
                                 synthetic=synthetic, print_tables=print_tables, out=out, clock=clock,
                                 run_protocol_hash=ctx["run_protocol_hash"], caveat=caveat)
                if code != EXIT_OK:
                    raise RuntimeError(f"the holdout backtest exited {code}")
    except Exception as exc:
        _update_record(layout, record, f"failed at {stage}: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        print(f"error: the look failed at {stage}; the look record says so and the look is spent. Fix the cause without "
              "changing the code, then resume: run --period holdout --resume (+ --one-look on real data)", file=sys.stderr)
        return EXIT_FAILED
    _update_record(layout, record, "complete")
    print(f"look complete; record {layout.root / LOOK_RECORD}", file=out)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ivf.backtest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", action="store_true", help="use ROOT/synthetic and the synthetic protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="build interim/tradable_daily.parquet (synthetic: write the synthetic ETPs first)")
    r = commands.add_parser("run", help="the backtest for one period (milestone 6)")
    r.add_argument("--period", choices=["dev", "holdout"], default="dev")
    r.add_argument("--run-id", help="the model run whose forecasts and M5 dev metric to use (default: the newest with both)")
    r.add_argument("--one-look", action="store_true", help="real data, --period holdout: take the one look (owner only)")
    r.add_argument("--resume", action="store_true", help="--period holdout: continue a look that failed at a stage")
    r.add_argument("--preflight-only", action="store_true", help="--period holdout: check every precondition, write nothing")
    r.add_argument("--workers", type=int, default=5, help="LSTM processes for the holdout refits")
    r.add_argument("--print-tables", action="store_true", help="print the values (local only)")
    args = parser.parse_args(argv)
    layout = config.data_layout(synthetic=args.synthetic)
    out = sys.stdout
    try:
        repo_guard.require_data_root_outside_repo(layout.root)
        if args.command == "prepare":
            return cmd_prepare(layout, args.synthetic, out)
        if (args.one_look or args.resume or args.preflight_only) and args.period != "holdout":
            print("error: --one-look, --resume and --preflight-only are for --period holdout", file=sys.stderr)
            return EXIT_USAGE
        if args.one_look and args.synthetic:
            print("error: --one-look is for real data; the synthetic look runs without it", file=sys.stderr)
            return EXIT_USAGE
        if args.period == "holdout":
            if not args.synthetic and not args.one_look and not args.preflight_only:
                print(f"error: {REFUSAL}", file=sys.stderr)
                return EXIT_USAGE
            return cmd_look(layout, synthetic=args.synthetic, run_id=args.run_id, workers=args.workers,
                            print_tables=args.print_tables, out=out, resume=args.resume, preflight_only=args.preflight_only)
        return cmd_dev(layout, synthetic=args.synthetic, run_id=args.run_id, print_tables=args.print_tables, out=out)
    except (protocol_mod.ProtocolError, build_interim.InterimError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except repo_guard.RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return repo_guard.EXIT_REFUSED
