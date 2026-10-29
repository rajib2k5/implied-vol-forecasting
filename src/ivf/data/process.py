"""The milestone 1 processing stage: interim tables to processed tables and a local EDA report.

    uv run python -m ivf.data eda

Writes, under the data root (``ROOT`` or ``ROOT/synthetic``):

* ``processed/runs.parquet`` (one row per run) and, under
  ``processed/run_id=<id>/``: ``run_input``, ``calendar``, ``data_check``,
  ``feature``, ``target_value``, ``regime_label`` and ``eda_stat``
  (data-model.md section 4);
* ``eda/<run_id>/eda_report.json``: every EDA number, for reading locally.

Nothing here is committed. On real data every number is outside the
publication set P except the AR(1) half-life per period (ADR 0001 D2).

**The EDA never sees the holdout** (amended M1, review A). Every EDA number
is computed on sessions strictly before ``holdout_start``, from interim rows
strictly before it, so no origin and no target date is in the holdout
(``eda_inputs``). Its samples are ``pre_holdout`` (all of it), and the two
calendar periods inside it, ``burn_in`` and ``dev``. The first version also
printed holdout statistics, a look at the holdout before any model is frozen
(ADR 0001 4.6, "one look"), and its "dev" sample silently included burn-in.
The processed ``feature``, ``target_value`` and ``regime_label`` tables still
cover 𝒯 to ``data_end``: they are inputs for milestone 2, not statistics.

Nothing past ``data_end`` is read: the tables are built on the sessions of 𝒯
up to ``data_end``, and ``build.align`` reads interim rows only on those
sessions, so rows from a later fetch are ignored (a test appends some and
checks that nothing changes). ``eda_report`` refuses any session or row at or
after ``holdout_start``, which is never after ``data_end``.
"""

from __future__ import annotations

import io
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ivf
from ivf import contract, eda, repo_guard
from ivf import protocol as protocol_mod
from ivf.data import build_interim, calendar, check
from ivf.features import build


def run_id_for(protocol_hash: str, git: contract.GitState, data_mode: str, raw_sha256s: list[str]) -> str:
    """ADR 0001 D3: the first 12 hex of the SHA-256 of the run's identity."""
    return contract.canonical_json_sha256(
        {
            "protocol_hash": protocol_hash,
            "source_commit": git.commit,
            "source_dirty": git.dirty,
            "data_mode": data_mode,
            "raw_sha256s": sorted(set(raw_sha256s)),
        }
    )[:12]


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Parquet bytes through ``repo_guard.write_file``: the folder is checked at write time (amended M1, review B)."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    repo_guard.write_file(path, buffer.getvalue(), outside_repo=True)


def _finite(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


#: The EDA samples (amended M1, review A). ``pre_holdout`` is every session before holdout_start
#: (burn-in and dev); the other two are the calendar's periods inside it. There is no holdout sample.
EDA_SAMPLES = ("pre_holdout", "burn_in", "dev")


class HoldoutRowRefused(ValueError):
    """The EDA was handed a session or a row on or after ``holdout_start``."""


def eda_inputs(cal: pd.DataFrame, iv: pd.DataFrame, price: pd.DataFrame, holdout_start):
    """``(dates, periods, iv, price)``: what the M1 EDA may see, all strictly before ``holdout_start``.

    ``dates`` and ``periods`` come from the master calendar ``cal``; the frames
    keep only rows dated before ``holdout_start``. A trailing feature at a
    pre-holdout session is the same whether or not later rows exist (rule L1,
    the truncation test), and a forward window that would reach the holdout
    finds no data, so its value is null rather than a holdout number.
    """
    before = cal["date"] < holdout_start
    dates = list(cal.loc[before, "date"])
    periods = list(cal.loc[before, "period"])
    return dates, periods, iv[iv["date"] < holdout_start], price[price["date"] < holdout_start]


def forward_rv21(features: pd.DataFrame) -> np.ndarray:
    """At origin t, the realised volatility of the 21 sessions AFTER t (t+1..t+21): ``rv21_back`` at t+21.

    Ex post, for the EDA's implied-minus-realised spread only (rule L1 forbids it as a feature).
    ``rv21_back`` at d sums the returns of d-20..d, so at d = t+21 the window is t+1..t+21: the
    30 calendar days VIX at t prices, to within ADR 0001 4.4's convention. Trailing RV (window
    t-20..t) or a window one session late (t+2..t+22) give the same *mean* spread on long
    stationary data, which is why the test pins the alignment origin by origin.
    """
    return build.forward(features["rv21_back"].to_numpy(), build.RV_WINDOW)


def eda_report(dates, periods, features: pd.DataFrame, iv: pd.DataFrame, holdout_start, nlags: int = 21) -> dict[str, Any]:
    """Every EDA number of the milestone brief, as a JSON-ready dict. Pure: no I/O.

    ``dates`` and ``periods`` are the sessions and calendar periods from
    ``eda_inputs``; ``features`` is ``build.build_features`` on them. Raises
    ``HoldoutRowRefused`` if any session, period or ``iv`` row is in the holdout.
    """
    late = [d for d in dates if d >= holdout_start] + [d for d in iv["date"] if d >= holdout_start]
    if late or "holdout" in periods:
        raise HoldoutRowRefused(
            f"the M1 EDA computes nothing on or after holdout_start {holdout_start} (ADR 0001 4.6, one look); "
            f"it was given {len(late)} such date(s)"
        )
    if len(periods) != len(dates) or len(features) != len(dates):
        raise ValueError("dates, periods and features must be aligned on the same sessions")
    vix = build.align(dates, iv, "symbol", "VIX")
    log_vix = features["log_vix"].to_numpy()
    r = features["r_spx"].to_numpy()
    rv21_fwd = forward_rv21(features)
    period = np.array(periods, dtype=object)
    samples = {"pre_holdout": np.ones(len(dates), dtype=bool), "burn_in": period == "burn_in", "dev": period == "dev"}

    def box(series) -> dict:
        lb = eda.acf_ljung_box(series, nlags)
        return {
            "n": lb.n,
            "acf": [_finite(a) for a in lb.acf],
            "ljung_box_q": [_finite(q) for q in lb.q],
            "ljung_box_p": [_finite(p) for p in lb.p_value],
        }

    report: dict[str, Any] = {
        "sample": {
            "first": dates[0].isoformat(),
            "last": dates[-1].isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "samples": {name: int(mask.sum()) for name, mask in samples.items()},
            "note": "every number is computed on sessions before holdout_start (pre_holdout = burn_in + dev); "
            "clustering, persistence and vol_of_vol use the whole pre_holdout sample",
        },
        "clustering": {"r": box(r), "r2": box(r**2), "abs_r": box(np.abs(r))},
        "persistence": {
            "vix_acf": [_finite(a) for a in eda.acf(vix, nlags)],
            "log_vix_acf": [_finite(a) for a in eda.acf(log_vix, nlags)],
            "n": int(np.sum(~np.isnan(log_vix))),
        },
        "half_life": {},
        "unit_root": {},
        "iv_rv_spread": {},
        "term_structure": {},
    }
    levels = {s: build.align(dates, iv, "symbol", s) for s in ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")}
    for name, mask in samples.items():
        x = np.where(mask, log_vix, np.nan)
        hl = eda.ar1_half_life(x)
        report["half_life"][name] = {
            "phi": hl.phi,
            "phi_se_hc1": hl.phi_se,
            "phi_ci95": list(hl.phi_ci),
            "half_life_sessions": _finite(hl.half_life),
            "half_life_ci95": [_finite(v) for v in hl.half_life_ci],
            "n": hl.n,
        }
        ur = eda.unit_root_tests(np.where(mask, log_vix, np.nan))
        report["unit_root"][name] = {
            "adf_stat": ur.adf_stat,
            "adf_p": ur.adf_p,
            "adf_lags": ur.adf_lags,
            "kpss_stat": ur.kpss_stat,
            "kpss_p": ur.kpss_p,
            "kpss_p_bound": ur.kpss_p_bound,
            "stationary_at_5pct": ur.stationary_at_5pct,
            "n": ur.n,
            "segment_first": dates[ur.segment[0]].isoformat(),
            "segment_last": dates[ur.segment[1] - 1].isoformat(),
            "n_nonmissing": ur.n_nonmissing,
            "adf_null": eda.UnitRoot.ADF_NULL,
            "kpss_null": eda.UnitRoot.KPSS_NULL,
        }
        sp = eda.iv_rv_spread(vix, rv21_fwd, mask)
        report["iv_rv_spread"][name] = {
            "n": sp.n,
            "mean_vol_points": _finite(sp.mean_vol_points),
            "median_vol_points": _finite(sp.median_vol_points),
            "mean_variance": _finite(sp.mean_variance),
            "share_positive": _finite(sp.share_positive),
        }
        ts = eda.term_structure_summary(levels, mask=mask)
        report["term_structure"][name] = {
            "n": ts.n,
            "contango_share": _finite(ts.contango_share),
            "mean_log_slope_vix3m_over_vix": _finite(ts.mean_log_slope),
            "n_full_curve": ts.n_full_curve,
            "upward_full_curve_share": _finite(ts.upward_full_curve_share),
        }
    vvix = build.align(dates, iv, "symbol", "VVIX") if (iv["symbol"] == "VVIX").any() else None
    vov = eda.vol_of_vol(vix, 21, vvix)
    report["vol_of_vol"] = {
        "n": vov.n,
        "mean_vov_annualised": _finite(vov.mean_vov),
        "spearman_level_vs_next_abs_resid": _finite(vov.spearman_level_vs_next_abs_resid),
        "e6_p_bootstrap": _finite(vov.e6_p_bootstrap),
        "e6_n_boot": vov.e6_n_boot,
        "e6_block": vov.e6_block,  # circular-block length of the bootstrap (amended M1, review B)
        "e6_ar1_phi": _finite(vov.e6_phi),  # the verdict is gated on it (eda.E6_PHI_MAX; M1 final check)
        "e6_verdict": vov.e6_verdict,  # "yes" or "not shown" (eda.e6_verdict)
        "e6_reason": vov.e6_reason,
        "spearman_level_vs_next_abs_dlog": _finite(vov.spearman_level_vs_next_abs_dlog),
        "spearman_level_vs_next_abs_dpoints": _finite(vov.spearman_level_vs_next_abs_dpoints),
        "spearman_vix_vs_vvix": _finite(vov.spearman_vix_vs_vvix),
        "vvix_median": _finite(float(np.nanmedian(vvix))) if vvix is not None else None,
    }
    years = pd.Series([d.year for d in dates])
    slope = np.log(levels["VIX3M"] / levels["VIX"])
    by_year = pd.DataFrame({"year": years, "slope": slope, "contango": np.where(np.isnan(slope), np.nan, slope >= 0)})
    grouped = by_year.dropna().groupby("year")
    report["term_structure"]["by_year"] = [
        {"year": int(y), "n": int(len(g)), "contango_share": float(g.contango.mean()), "mean_log_slope": float(g.slope.mean())}
        for y, g in grouped
    ]
    return report


def eda_stat_rows(report: dict[str, Any]) -> pd.DataFrame:
    """The eda_stat table (data-model.md section 4) for VIX, from the report."""
    rows = []
    for sample, hl in report["half_life"].items():
        rows.append({"symbol": "VIX", "stat": "ar1_phi_log", "sample": sample, "value": hl["phi"], "n": hl["n"]})
        rows.append({"symbol": "VIX", "stat": "half_life_sessions", "sample": sample, "value": hl["half_life_sessions"], "n": hl["n"]})
        ur = report["unit_root"][sample]
        rows.append({"symbol": "VIX", "stat": "adf_p", "sample": sample, "value": ur["adf_p"], "n": ur["n"]})
        rows.append({"symbol": "VIX", "stat": "kpss_p", "sample": sample, "value": ur["kpss_p"], "n": ur["n"]})
        sp = report["iv_rv_spread"][sample]
        rows.append({"symbol": "VIX", "stat": "iv_rv_spread_mean", "sample": sample, "value": sp["mean_variance"], "n": sp["n"]})
    for lag in (1, 5, 21):
        rows.append({"symbol": "VIX", "stat": f"acf_log_lag{lag}", "sample": "pre_holdout", "value": report["persistence"]["log_vix_acf"][lag], "n": report["persistence"]["n"]})
    rows.append({"symbol": "VIX", "stat": "vov21_mean", "sample": "pre_holdout", "value": report["vol_of_vol"]["mean_vov_annualised"], "n": report["vol_of_vol"]["n"]})
    frame = pd.DataFrame(rows, columns=["symbol", "stat", "sample", "value", "n"])
    frame["n"] = frame["n"].astype("int32")
    return frame


def run_eda(layout, *, synthetic: bool) -> int:
    """Processed tables and the local EDA report; exit 1 if ``check`` would fail, 3 without input.

    ``data_check`` runs first with the frozen ``data_end`` re-validated
    (``check.frozen_data_end_violations``) and checked against its freeze
    record (``check.frozen_check``; amended M1, review B), so ``eda`` refuses
    exactly when ``check`` fails. Only sessions up to ``data_end`` are built, and the EDA
    itself sees only sessions before ``holdout_start``.
    """
    repo_guard.require_data_root_outside_repo(layout.root)
    proto = protocol_mod.load_protocol(protocol_mod.protocol_path(synthetic=synthetic))
    data_end = proto.require_data_end()
    try:
        iv, price, report = build_interim.read_interim(layout.root)
    except build_interim.InterimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    result = check.frozen_check(proto, iv, price, report)  # the data and the freeze record (amended M1, review B)
    if not result.ok:
        for violation in result.policy_violations:
            print(f"error: {violation}", file=sys.stderr)
        return 1
    started = datetime.now(timezone.utc)
    git = contract.git_state()
    data_mode = "synthetic" if synthetic else "real"
    raw_shas = [entry["raw_sha256"] for entry in report["files"]]
    run_id = run_id_for(proto.protocol_hash, git, data_mode, raw_shas)
    cal = calendar.master_calendar(proto)
    # Sessions to data_end only. Rows after it (a later fetch) stay in interim, but build.align
    # reads a frame only on the sessions it is given, so they are never read.
    dates = [d for d in cal["date"] if d <= data_end]
    features = build.build_features(dates, iv, price)
    targets = build.build_target_values(dates, iv, price)
    regimes = build.build_regime_labels(dates, iv, proto.vix_regime_edges)
    eda_dates, eda_periods, eda_iv, eda_price = eda_inputs(cal, iv, price, proto.holdout_start)
    eda_out = eda_report(eda_dates, eda_periods, build.build_features(eda_dates, eda_iv, eda_price), eda_iv, proto.holdout_start)
    folder = layout.processed / f"run_id={run_id}"
    tables = {
        "run_input": pd.DataFrame(
            [{"raw_sha256": e["raw_sha256"], "source": e["source"], "symbol": e["symbol"]} for e in report["files"]]
        ),
        "calendar": cal,
        "data_check": result.table,
        "feature": features,
        "target_value": targets,
        "regime_label": regimes,
        "eda_stat": eda_stat_rows(eda_out),
    }
    for name, frame in tables.items():
        _write(frame.assign(run_id=run_id)[["run_id", *frame.columns]], folder / f"{name}.parquet")
    run_row = {
        "run_id": run_id,
        "protocol_hash": proto.protocol_hash,
        "data_mode": data_mode,
        "source_commit": git.commit,
        "source_dirty": git.dirty,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc),
        "ivf_version": ivf.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    runs_path = layout.processed / "runs.parquet"
    new_row = pd.DataFrame([run_row])
    if runs_path.exists():
        # Same inputs, same run_id: a re-run replaces its own row (ADR 0001 D3).
        previous = pd.read_parquet(runs_path)
        runs = pd.concat([previous[previous["run_id"] != run_id], new_row], ignore_index=True)
    else:
        runs = new_row
    _write(runs, runs_path)
    out = layout.eda / run_id / "eda_report.json"
    document = {"run_id": run_id, "data_mode": data_mode, "data_end": data_end.isoformat(), **eda_out}
    repo_guard.write_file(out, (json.dumps(document, indent=1, allow_nan=False) + "\n").encode("utf-8"), outside_repo=True)
    hl = eda_out["half_life"]
    print(f"EDA sample: {eda_out['sample']['first']}..{eda_out['sample']['last']}, before holdout_start {proto.holdout_start}; no holdout row is read")
    print(f"run_id {run_id} ({data_mode}); processed tables in {folder}")
    print(f"EDA report (local only, never commit real numbers: ADR 0001 D2): {out}")
    verdicts = {
        "E1 volatility clustering (Ljung-Box on r^2, lag 21, p < 0.05), pre-holdout": eda_out["clustering"]["r2"]["ljung_box_p"][-1] < 0.05,
        "E3/E7 ln VIX stationary at 5% (ADF rejects, KPSS does not), pre-holdout": eda_out["unit_root"]["pre_holdout"]["stationary_at_5pct"],
        "E4 contango more often than not (VIX <= VIX3M), pre-holdout": eda_out["term_structure"]["pre_holdout"]["contango_share"] > 0.5,
        "E5 implied variance above later realised variance on average, pre-holdout": eda_out["iv_rv_spread"]["pre_holdout"]["mean_variance"] > 0,
    }
    for statement, verdict in verdicts.items():
        print(f"  {statement}: {'yes' if verdict else 'no'}")
    # E6 can only say "yes" or "not shown" (amended M1, review B): its block bootstrap keeps its size
    # when shocks cluster, but then it has little power (18% against a strong level effect on top of
    # GARCH clustering; walkthrough, section 7.3), so failing to reject is no evidence of "no". And
    # it answers only up to the persistence, and from the sample length, where its size was measured
    # (eda.E6_PHI_MAX, eda.E6_MIN_N; M1 final check). The verdict and its reason come from eda.e6_verdict, through the report.
    e6 = eda_out["vol_of_vol"]
    reason = f" ({e6['e6_reason']})" if e6["e6_reason"] else ""
    print(
        "  E6 vol-of-vol rises with the level (Spearman of V_t and the next |AR(1) shock| > 0, block-bootstrap p < 0.05, "
        f"AR(1) phi <= {eda.E6_PHI_MAX}, at least {eda.E6_MIN_N} pairs), pre-holdout: {e6['e6_verdict']}{reason}"
    )
    for sample in ("burn_in", "dev"):
        print(f"  E2 AR(1) half-life of ln VIX, {sample}: about {hl[sample]['half_life_sessions']:.2g} sessions (n = {hl[sample]['n']})")
    return 0
