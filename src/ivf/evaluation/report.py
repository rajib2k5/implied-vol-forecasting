"""Milestone 5: evaluate every stored model on the dev period (ADR 0001 D4).

    uv run python -m ivf.evaluation report [--run-id ID] [--print-tables]
    uv run python -m ivf.evaluation --synthetic report [--period dev|holdout]

Reads the forecast table of a model run (``ROOT/processed/run_id=<id>/forecast.parquet``; by default
the newest run with forecasts on ``vix_level`` or ``rv21``), the interim tables (re-checked against
the frozen ``data_end`` exactly as the model steps do), and builds, for the period:

* ``metric``: RMSE, MAE, MAPE, QLIKE and bias per (model, target, horizon, scheme, regime, forecast);
  regimes from ``regimes``; the forecasts ``mean`` (scored), ``median`` (log-space models) and
  ``mean_cal`` (``calibration``, a separate variant);
* ``dm_test``: every pair of models per (target, horizon, loss), with each long-run variance method of
  ``dm`` (the primary one flagged) and Holm-adjusted p-values per family; plus each model against its
  own calibrated variant (family ``calibration``);
* ``mz_regression``: Mincer-Zarnowitz per (model, target, horizon, space, forecast) with each HAC method
  of ``mz`` (the primary one flagged); ``rv21`` in level and variance space;
* ``calibration`` (the multiplier per fold) and ``calibration_summary`` (the pre-registered verdict).

Written under ``ROOT/processed/run_id=<id>/evaluation/period=<period>/`` with ``report_meta.json``
(through ``repo_guard.write_file``: never inside the repo). Real numbers are outside the publication set
P (ADR 0001 D2): the default output is counts and timings; ``--print-tables`` prints values locally.

**The holdout (ADR 0001 4.6, policy of 2026-10-29).** On real data this step evaluates ``dev`` only, and
no value dated on or after ``holdout_start`` enters any output: targets and regimes are built from the
sessions before it (``walkforward.pre_holdout_inputs``). The frozen ``data_end`` check reads the full
interim tables, as every step does; that check is the only reader of later rows (amended M5, review A:
the first version said no such row was read). The one look
at the holdout is taken once, at the end of milestone 6, for the forecasts and the backtest together
with every model and rule frozen. ``protocol.holdout_looks`` counts exposures before that look (amended M6,
review A: it is 1, ADR 0001 4.6). ``--period holdout`` is refused on real data before anything is read,
unless the guarded look (``python -m ivf.backtest run --period holdout --one-look``) calls it; with ``--synthetic`` it runs the same code on the synthetic market,
which is how the holdout path is tested. On the holdout, DM and MZ p-values are **descriptive only**
(``descriptive_only``: not size-controlled at n of about 583, ADR 0001 O-M5-5) and the confirmatory check is
``holdout_check``, pre-registered in M5 (amended M5, review A).

Exit codes: 0 success; 1 the data check failed or the forecasts break the same-rows rule; 2 bad command
line, or ``--period holdout`` on real data; 3 missing or malformed input, or the licensing guard refused.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd

from ivf import config, contract, repo_guard
from ivf import protocol as protocol_mod
from ivf.data import build_interim, calendar, check
from ivf.evaluation import calibration, dm, mz, regimes, walkforward
from ivf.features import build
from ivf.models import registry

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INPUT = 0, 1, 2, 3
TARGETS = ("vix_level", "rv21")
#: The DM benchmarks (amended M5): the random walk on each track, and HAR as the strong baseline.
BENCHMARKS = {"vix_level": ("rw", "har_log"), "rv21": ("rw", "har_rv")}
CAL_SUFFIX = "+cal"
HOLDOUT_REFUSAL = ("refusing --period holdout on real data: the one look at the holdout (ADR 0001 4.6) is taken at "
                   "the end of milestone 6, for the forecasts and the backtest together, with every model and rule "
                   "frozen (policy of 2026-10-29); only `python -m ivf.backtest run --period holdout --one-look` "
                   "takes it. Use --synthetic to exercise this path.")


# --------------------------------------------------------------------------
# The tables (pure functions of frames: tested without the command)
# --------------------------------------------------------------------------


def scored_rows(forecasts: pd.DataFrame, holdout_start: date, period: str) -> pd.DataFrame:
    """The rows scored in ``period``: target observed (``y`` not NaN) and

    * ``dev``: origin AND target dated before ``holdout_start`` (the dev purge of ADR 0001 4.6);
    * ``holdout``: origin on or after ``holdout_start``.
    """
    observed = forecasts["y"].notna()
    if period == "dev":
        keep = observed & (forecasts["origin_date"] < holdout_start) & (forecasts["target_date"] < holdout_start)
    elif period == "holdout":
        keep = observed & (forecasts["origin_date"] >= holdout_start)
    else:
        raise ValueError(f"unknown period {period!r}")
    return forecasts[keep]


def attach_targets(forecasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """``y`` and ``v`` at each row's ``target_date`` (NaN where the target is not in ``targets``)."""
    t = targets.rename(columns={"date": "target_date"})[["target_id", "target_date", "y", "v"]]
    out = forecasts.drop(columns=[c for c in ("y", "v") if c in forecasts]).merge(t, on=["target_id", "target_date"], how="left")
    return out


def _with_calibrated(scored: pd.DataFrame) -> pd.DataFrame:
    """Stack each model's calibrated variant under it as model ``<id>+cal`` (for DM and MZ)."""
    cal = scored.assign(model_id=scored["model_id"] + CAL_SUFFIX, yhat=scored["yhat_cal"], v_hat=scored["v_hat_cal"])
    return pd.concat([scored, cal], ignore_index=True)


def _loss_matrix(group: pd.DataFrame, loss: str) -> pd.DataFrame:
    """Losses with one column per model, one row per origin (same rows: an error if any is missing)."""
    values = group.assign(loss=dm.loss_series(group["y"], group["yhat"], loss))
    wide = values.pivot(index="origin_date", columns="model_id", values="loss").sort_index()
    if wide.isna().any().any():
        raise AssertionError("the models were not scored on the same origins (same-rows rule)")
    return wide


def dm_table(scored: pd.DataFrame, period: str) -> pd.DataFrame:
    """Every pair of models (family ``pairs``) and every model against its calibrated variant
    (family ``calibration``), per (target, horizon, loss), each LRV method. ``p_holm`` is Holm within
    (target, horizon, loss, lrv method, family).

    **Sign.** ``d = L_a - L_b`` with ``model_a < model_b`` alphabetically, so a **positive** ``mean_d``
    (and ``dm_stat``) means ``model_a`` has the larger loss: ``model_a`` is worse. ``lower_loss`` names the
    model with the smaller mean loss (amended M5, review A).
    """
    stacked = _with_calibrated(scored)
    rows = []
    for (target_id, h), group in stacked.groupby(["target_id", "horizon"], sort=True):
        models = sorted(m for m in group["model_id"].unique() if not m.endswith(CAL_SUFFIX))
        pairs = [("pairs", a, b) for a, b in combinations(models, 2)]
        pairs += [("calibration", m, m + CAL_SUFFIX) for m in models]
        for loss in dm.LOSSES:
            wide = _loss_matrix(group, loss)
            for family, a, b in pairs:
                d = (wide[a] - wide[b]).to_numpy()
                for method in dm.METHODS:
                    r = dm.dm_test(d, int(h), method)
                    rows.append({
                        "target_id": target_id, "horizon": int(h), "period": period, "scheme": "all", "regime": "all",
                        "loss": loss, "model_a": a, "model_b": b, "family": family, "n": r.n, "mean_d": r.mean_d,
                        "dm_stat": r.dm_stat, "p_value": r.p_value, "nw_lags": int(h) - 1, "method": method,
                        "lrv_method": r.lrv_method, "bandwidth": r.bandwidth, "max_lag": r.max_lag,
                        "primary": method == dm.PRIMARY_METHOD,
                        "lower_loss": a if r.mean_d < 0 else b if r.mean_d > 0 else None,
                        "descriptive_only": period == "holdout",
                    })
    table = pd.DataFrame(rows)
    keys = ["target_id", "horizon", "loss", "method", "family"]
    table["p_holm"] = table.groupby(keys, sort=False)["p_value"].transform(lambda p: pd.Series(dm.holm(p.to_numpy()), index=p.index))
    return table


def mz_table(scored: pd.DataFrame, period: str) -> pd.DataFrame:
    """MZ per (model, target, horizon, space, forecast) and HAC method. ``vix_level`` in level space;
    ``rv21`` in level and variance space (``y`` on ``yhat``; ``v`` on ``v_hat``)."""
    stacked = _with_calibrated(scored)
    rows = []
    for (model_id, target_id, h), g in stacked.groupby(["model_id", "target_id", "horizon"], sort=True):
        g = g.sort_values("origin_date")
        spaces = [("level", g["y"], g["yhat"])]
        if target_id == "rv21":
            spaces.append(("variance", g["v"], g["v_hat"]))
        base, forecast = (model_id[: -len(CAL_SUFFIX)], "mean_cal") if model_id.endswith(CAL_SUFFIX) else (model_id, "mean")
        for space, y, yhat in spaces:
            for method in mz.COV_METHODS:
                r = mz.mz_regression(y.to_numpy(), yhat.to_numpy(), int(h), method)
                rows.append({
                    "model_id": base, "target_id": target_id, "horizon": int(h), "period": period, "space": space,
                    "forecast": forecast, "n": r.n, "alpha": r.alpha, "alpha_se": r.alpha_se, "beta": r.beta,
                    "beta_se": r.beta_se, "r2": r.r2, "wald_stat": r.wald_stat, "wald_p": r.wald_p,
                    "nw_lags": int(h) - 1, "method": method, "cov_method": r.cov_method, "bandwidth": r.bandwidth,
                    "primary": method == mz.PRIMARY_METHOD, "descriptive_only": period == "holdout",
                })
    return pd.DataFrame(rows)


def calibration_summary(metric: pd.DataFrame, dm_tests: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """The pre-registered verdict of ``calibration`` (rule 6), per (model, target, horizon)."""
    allrows = metric[(metric.scheme == "all") & metric.forecast.isin(["mean", "mean_cal"])]
    wide = allrows.pivot_table(index=["model_id", "target_id", "horizon"], columns="forecast", values=["rmse", "qlike"])
    wide.columns = [f"{m}_{'raw' if f == 'mean' else 'cal'}" for m, f in wide.columns]
    test = dm_tests[(dm_tests.family == "calibration") & dm_tests.primary & (dm_tests.loss == "sq_error")]
    test = test.set_index(["model_a", "target_id", "horizon"])[["mean_d", "p_value", "p_holm"]]
    test.index.names = ["model_id", "target_id", "horizon"]
    out = wide.join(test).reset_index()
    out["helps"] = (out.rmse_cal < out.rmse_raw) & (out.qlike_cal < out.qlike_raw) & (out.p_holm < alpha)
    return out


#: The confirmatory holdout check (pre-registered in M5, amended M5 review A; default taken 2026-10-29, owner
#: can override): the losses it compares per target.
HOLDOUT_LOSSES = {"vix_level": ("rmse",), "rv21": ("rmse", "qlike")}


def holdout_check(dev_metric: pd.DataFrame, holdout_metric: pd.DataFrame) -> pd.DataFrame:
    """For each (target, horizon, loss of ``HOLDOUT_LOSSES``): the dev period's best model (lowest loss of
    its scored ``mean`` forecast, ties to the first id alphabetically; never a calibrated or median
    variant) and whether it ALSO has a lower holdout loss than ``rw``. ``held_up`` is that boolean; it is
    None when ``rw`` itself is the dev best (nothing to confirm). Only the dev metric chooses the model,
    so no choice uses holdout results. DM and MZ on the holdout are descriptive only (O-M5-5)."""
    def table(metric):
        m = metric[(metric.scheme == "all") & (metric.forecast == "mean")]
        return m.set_index(["target_id", "horizon", "model_id"])
    dev, hold = table(dev_metric), table(holdout_metric)
    rows = []
    for (target_id, h), g in dev.groupby(level=["target_id", "horizon"], sort=True):
        for loss in HOLDOUT_LOSSES.get(target_id, ("rmse",)):
            ranked = g[loss].droplevel([0, 1]).dropna().sort_index().sort_values(kind="stable")
            best = ranked.index[0]
            hb, hr = hold.loc[(target_id, h, best), loss], hold.loc[(target_id, h, "rw"), loss]
            rows.append({
                "target_id": target_id, "horizon": int(h), "loss": loss, "dev_best": best,
                "dev_best_value": float(ranked.iloc[0]), "dev_rw_value": float(g.loc[(target_id, h, "rw"), loss]),
                "holdout_best_value": float(hb), "holdout_rw_value": float(hr),
                "held_up": None if best == "rw" else bool(hb < hr),
            })
    return pd.DataFrame(rows)


def evaluate(forecasts: pd.DataFrame, targets: pd.DataFrame, labels: pd.DataFrame, holdout_start: date, period: str,
             min_cell_n: int, min_pairs: int = calibration.MIN_PAIRS) -> dict[str, pd.DataFrame]:
    """Every table of the report for ``period`` from a forecast frame (the store's columns), the
    ``target_value`` rows and the ``regime_label`` rows. Calibration runs on every row (each
    multiplier sees only targets dated <= its cutoff); scoring keeps ``period``'s rows."""
    frame = attach_targets(forecasts, targets)
    frame, multipliers = calibration.calibrate(frame, min_pairs)
    scored = regimes.attach(scored_rows(frame, holdout_start, period), labels)
    if scored.empty:
        raise ValueError(f"no scored rows for period {period}")
    for target_id, group in scored.groupby("target_id"):
        walkforward.check_same_rows(group)
    metric = regimes.metric_rows(scored, period, min_cell_n)
    dm_tests = dm_table(scored, period)
    tables = {
        "metric": metric,
        "dm_test": dm_tests,
        "mz_regression": mz_table(scored, period),
        "calibration": multipliers,
        "calibration_summary": calibration_summary(metric, dm_tests),
    }
    if period == "holdout":  # the confirmatory check: the model is chosen on dev rows only
        dev_metric = regimes.metric_rows(scored_rows(frame, holdout_start, "dev"), "dev", min_cell_n)
        tables["holdout_check"] = holdout_check(dev_metric, metric)
    return {**tables, "_scored": scored}


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def _write(frame: pd.DataFrame, path) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    repo_guard.write_file(path, buffer.getvalue(), outside_repo=True)


def choose_run(processed, run_id: str | None) -> str | None:
    """``run_id`` if given, else the newest run (``started_at``) whose forecast table has ``rw`` on a target."""
    if run_id:
        return run_id if (processed / f"run_id={run_id}" / "forecast.parquet").exists() else None
    runs_path = processed / "runs.parquet"
    if not runs_path.exists():
        return None
    runs = pd.read_parquet(runs_path).sort_values("started_at", ascending=False)
    for candidate in runs["run_id"]:
        path = processed / f"run_id={candidate}" / "forecast.parquet"
        if path.exists():
            have = pd.read_parquet(path, columns=["model_id", "target_id"]).drop_duplicates()
            if ((have.model_id == "rw") & have.target_id.isin(TARGETS)).any():
                return candidate
    return None


def run_report(layout, *, synthetic: bool, run_id: str | None = None, period: str = "dev", print_tables: bool = False,
               out=None, one_look: bool = False, caveat: str = "", run_protocol_hash: str | None = None) -> int:
    """``one_look`` (amended M6) is passed only by ``python -m ivf.backtest run --period holdout --one-look``,
    the owner's single look, which guards it (a preflight, then a create-only look record); this command's
    own line has no such flag, so ``--period holdout`` stays refused here on real data. ``caveat`` (the
    look's second-exposure text, review B) and ``run_protocol_hash`` (the protocol the dev run was made
    under) are written into ``report_meta.json``."""
    out = out or sys.stdout
    clock = time.perf_counter()
    repo_guard.require_data_root_outside_repo(layout.root)
    if period == "holdout" and not synthetic and not one_look:
        print(f"error: {HOLDOUT_REFUSAL}", file=sys.stderr)
        return EXIT_USAGE
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    data_end = proto.require_data_end()
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    result = check.frozen_check(proto, iv, price, report)
    if not result.ok:
        for violation in result.policy_violations:
            print(f"error: {violation}", file=sys.stderr)
        return EXIT_FAILED
    chosen = choose_run(layout.processed, run_id)
    if chosen is None:
        print(f"error: no model run with forecasts on {TARGETS} under {layout.processed} (run `python -m ivf.models ...` first)",
              file=sys.stderr)
        return EXIT_INPUT
    folder = layout.processed / f"run_id={chosen}"
    forecasts = pd.read_parquet(folder / "forecast.parquet")
    held = folder / "period=holdout" / "forecast.parquet"  # amended M6: the one look's holdout-fold forecasts
    if period == "holdout" and held.exists():
        forecasts = pd.concat([forecasts, pd.read_parquet(held)], ignore_index=True)
    forecasts = forecasts[forecasts.target_id.isin(TARGETS)].reset_index(drop=True)
    defs, _ = registry.load_models()
    for target_id in TARGETS:
        expected = registry.models_for(defs, target_id, 4)
        have = sorted(set(forecasts.loc[forecasts.target_id == target_id, "model_id"]))
        missing = sorted(set(expected) - set(have))
        print(f"{target_id}: {len(have)} models in run {chosen}" + (f"; not in this run: {missing}" if missing else
                                                                     " (every milestone <= 4 model)"), file=out)
    try:
        for target_id, group in forecasts.groupby("target_id"):
            walkforward.check_same_rows(group)
    except AssertionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    cal = calendar.master_calendar(proto)
    if period == "dev":
        # Nothing on or after holdout_start is read: targets and regimes from the pre-holdout rows only.
        dates, _, iv_read, price_read = walkforward.pre_holdout_inputs(cal, iv, price, proto.holdout_start)
    else:  # synthetic only (refused above on real data)
        dates = [d for d in cal["date"] if d <= data_end]
        iv_read, price_read = iv, price
    targets = build.build_target_values(dates, iv_read, price_read)
    labels = build.build_regime_labels(dates, iv_read, tuple(proto.vix_regime_edges))
    late_origins = int((forecasts.origin_date >= proto.holdout_start).sum())
    tables = evaluate(forecasts, targets, labels, proto.holdout_start, period, proto.document["min_cell_n"])
    scored = tables.pop("_scored")
    destination = folder / "evaluation" / f"period={period}"
    for name, frame in tables.items():
        _write(frame, destination / f"{name}.parquet")
    git = contract.git_state()
    meta = {
        "source_run_id": chosen, "period": period, "data_mode": "synthetic" if synthetic else "real",
        "evaluation_commit": git.commit, "evaluation_dirty": git.dirty,
        "generated_at": datetime.now(timezone.utc).isoformat(), "holdout_start": str(proto.holdout_start),
        "holdout_looks": int(proto.document["holdout_looks"]), "dm_primary": dm.PRIMARY_METHOD, "fixed_b": dm.FIXED_B,
        "caveat": caveat, "protocol_hash": proto.protocol_hash, "run_protocol_hash": run_protocol_hash,
        "mz_primary": mz.PRIMARY_METHOD, "calibration_min_pairs": calibration.MIN_PAIRS,
        "rows": {name: len(frame) for name, frame in tables.items()}, "scored_rows": len(scored),
    }
    repo_guard.write_file(destination / "report_meta.json", json.dumps(meta, indent=2, sort_keys=True).encode(), outside_repo=True)
    counts = scored.groupby(["target_id", "horizon"]).origin_date.nunique()
    print(f"period {period}: holdout_start {proto.holdout_start}; targets and regimes built from {len(dates)} sessions "
          f"ending {dates[-1]}; " + (
              "the frozen data_end check reads the full interim tables, no later value enters an output; "
              f"{late_origins} forecast rows with an origin on or after it ignored; "
              if period == "dev" else f"{meta['data_mode']} holdout path, DM and MZ descriptive only; ")
          + f"holdout_looks {meta['holdout_looks']}" + (f"; caveat: {caveat}" if caveat else ""),
          file=out)
    if "holdout_check" in tables:
        check_rows = tables["holdout_check"]
        print(f"holdout check: {int((check_rows.held_up == True).sum())} of {len(check_rows)} held up "  # noqa: E712
              f"({int(check_rows.held_up.isna().sum())} with rw as the dev best)", file=out)
    print("scored origins per (target, h): " + ", ".join(f"{t} {h}: {n}" for (t, h), n in counts.items()), file=out)
    print("rows: " + ", ".join(f"{k} {v}" for k, v in meta["rows"].items()) + f"; {time.perf_counter() - clock:.1f} s", file=out)
    print(f"tables written to {destination} (local only, never commit real numbers: ADR 0001 D2)", file=out)
    if print_tables:
        _print_tables(tables, out)
    return EXIT_OK


def _print_tables(tables: dict[str, pd.DataFrame], out) -> None:
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        m = tables["metric"]
        m = m[(m.scheme == "all") & (m.forecast == "mean")][["target_id", "horizon", "model_id", "n", "rmse", "mae", "mape", "qlike"]]
        print(m.sort_values(["target_id", "horizon", "rmse"]).to_string(index=False), file=out)
        d = tables["dm_test"]
        bench = d[d.primary & (d.family == "pairs")]
        bench = bench[[any(b in (r.model_a, r.model_b) for b in BENCHMARKS[r.target_id]) for r in bench.itertuples()]]
        print(bench[["target_id", "horizon", "loss", "model_a", "model_b", "mean_d", "dm_stat", "p_value", "p_holm"]].to_string(index=False), file=out)
        z = tables["mz_regression"]
        print(z[z.primary][["target_id", "horizon", "model_id", "forecast", "space", "alpha", "beta", "wald_p", "cov_method"]].to_string(index=False), file=out)
        print(tables["calibration_summary"].to_string(index=False), file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ivf.evaluation", description=__doc__.split("\n\n")[0])
    parser.add_argument("--synthetic", action="store_true", help="use ROOT/synthetic and the synthetic protocol")
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("report", help="metrics, DM, MZ, regimes and calibration for one period (milestone 5)")
    r.add_argument("--run-id", help="the model run to evaluate (default: the newest with forecasts)")
    r.add_argument("--period", choices=["dev", "holdout"], default="dev", help="holdout: --synthetic only")
    r.add_argument("--print-tables", action="store_true", help="print the values (local only)")
    args = parser.parse_args(argv)
    layout = config.data_layout(synthetic=args.synthetic)
    try:
        return run_report(layout, synthetic=args.synthetic, run_id=args.run_id, period=args.period, print_tables=args.print_tables)
    except (protocol_mod.ProtocolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except repo_guard.RepoWriteRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return repo_guard.EXIT_REFUSED
