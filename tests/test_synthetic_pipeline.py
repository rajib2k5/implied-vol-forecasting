"""Tests for ivf.data.synthetic and the whole milestone 1 pipeline run on its files.

The synthetic generator writes the same file shapes as Cboe and yfinance, so
these tests run the real code path end to end: raw store, interim, verify,
check, features and the EDA, then the real-mode export.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ivf import protocol
from ivf.data import build_interim, calendar, process, raw, synthetic
from ivf.features import build

pytestmark = pytest.mark.usefixtures("no_network")


def proto():
    return protocol.load_protocol(protocol.protocol_path(synthetic=True))


def test_the_known_truth_is_what_the_comments_say():
    p = synthetic.SyntheticParams()
    assert p.half_life == pytest.approx(34.309618491520645, rel=1e-12)  # ln 0.5 / ln 0.98
    assert p.stationary_sd == pytest.approx(0.30, rel=1e-12)  # sigma/sqrt(1 - phi^2) by construction
    assert p.daily_variance == pytest.approx(0.16**2 / 252, rel=1e-12)  # 1.01587e-4: 16% a year
    assert synthetic.contango_probability(p) == pytest.approx(0.8569388078044908, rel=1e-12)  # Phi(0.32/0.30)


def test_the_generator_is_deterministic_for_a_seed_and_differs_across_seeds():
    a_iv, a_px = synthetic.synthetic_market(proto())
    b_iv, b_px = synthetic.synthetic_market(proto())
    pd.testing.assert_frame_equal(a_iv, b_iv)
    pd.testing.assert_frame_equal(a_px, b_px)
    c_iv, _ = synthetic.synthetic_market(proto(), synthetic.SyntheticParams(seed=1))
    assert not np.allclose(a_iv["close"].to_numpy()[:100], c_iv["close"].to_numpy()[:100])


def test_tenors_start_on_their_first_dates_and_vvix_is_close_only():
    iv, price = synthetic.synthetic_market(proto())
    first = iv.groupby("symbol")["date"].min().to_dict()
    assert first == {
        "VIX": date(1990, 1, 2), "VIX1Y": date(2007, 1, 3), "VIX6M": date(2008, 1, 2),
        "VIX3M": date(2009, 9, 18), "VIX9D": date(2011, 1, 4), "VVIX": date(2006, 3, 6),
    }
    assert iv.loc[iv.symbol == "VVIX", "open"].isna().all()
    assert price["date"].iloc[-1] == date(2026, 10, 26)


def test_written_files_parse_back_to_the_generated_values(tmp_path):
    iv, price = synthetic.synthetic_market(proto())
    records = synthetic.write_synthetic_raw(tmp_path, proto())
    assert {r.source for r in records} == {"synthetic"}
    assert sorted(r.symbol for r in records) == sorted(["VIX", "VIX9D", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "^GSPC"])
    build_interim.build_interim(tmp_path, ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX"], ["^GSPC"])
    got, got_price, _ = build_interim.read_interim(tmp_path)
    vix = got[got.symbol == "VIX"].reset_index(drop=True)
    expected = iv[iv.symbol == "VIX"].reset_index(drop=True)
    assert list(vix.date) == list(expected.date)
    # Six decimals in the CSV: within 5e-7 of the generated level.
    assert np.max(np.abs(vix.close.to_numpy() - expected.close.to_numpy())) <= 5e-7
    assert np.max(np.abs(got_price.close.to_numpy() - price.close.to_numpy())) <= 5e-7
    assert got.loc[got.symbol == "VVIX", "open"].isna().all()


def test_writing_again_stores_nothing_new_and_logs_unchanged_fetches(tmp_path):
    synthetic.write_synthetic_raw(tmp_path, proto())
    synthetic.write_synthetic_raw(tmp_path, proto())
    records = raw.read_provenance(tmp_path)
    assert [r.changed for r in records] == [True] * 7 + [False] * 7
    assert len(list((tmp_path / "raw").rglob("*.csv"))) == 7
    assert raw.verify(tmp_path) == ([], [])


def run(*args, data_dir):
    env = dict(os.environ, VOL_DATA_DIR=str(data_dir))
    return subprocess.run([sys.executable, "-m", "ivf.data", *args], capture_output=True, text=True, env=env, check=False)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """The whole synthetic pipeline through the command line, once for the tests below."""
    data_dir = tmp_path_factory.mktemp("vol")
    outputs = {step: run("--synthetic", *step.split(), data_dir=data_dir) for step in ("fetch", "build-interim", "verify", "check", "eda")}
    return data_dir / "implied-vol-forecasting" / "synthetic", outputs


def test_every_command_succeeds_on_synthetic_data(pipeline):
    _, outputs = pipeline
    for step, result in outputs.items():
        assert result.returncode == 0, f"{step}: {result.stderr}"
    assert "gap policy: OK" in outputs["check"].stdout
    assert "candidate data_end (last session with both core closes, settled): 2026-10-26" in outputs["check"].stdout
    assert "train_start check: first VIX date 1990-01-02, train_start 1990-01-02" in outputs["check"].stdout
    # The synthetic shocks never depend on the level. E6 says "yes" or "not shown", never "no": its
    # block bootstrap has little power when shocks cluster (amended M1, review B).
    e6 = [line for line in outputs["eda"].stdout.splitlines() if line.lstrip().startswith("E6 ")]
    assert len(e6) == 1 and e6[0].split("pre-holdout: ")[1].startswith("not shown (not rejected at 5%")
    assert any(line.endswith("pre-holdout: yes") for line in outputs["eda"].stdout.splitlines())  # the others keep yes/no


def test_eda_writes_the_processed_tables_with_run_id_first(pipeline):
    root, _ = pipeline
    runs = pd.read_parquet(root / "processed" / "runs.parquet")
    assert len(runs) == 1 and runs.data_mode.iloc[0] == "synthetic"
    run_id = runs.run_id.iloc[0]
    folder = root / "processed" / f"run_id={run_id}"
    names = sorted(p.stem for p in folder.glob("*.parquet"))
    assert names == sorted(["run_input", "calendar", "data_check", "feature", "target_value", "regime_label", "eda_stat"])
    for name in names:
        table = pq.read_table(folder / f"{name}.parquet")
        assert table.schema.names[0] == "run_id"
        assert set(table.column("run_id").to_pylist()) == {run_id}
    stats = pq.read_table(folder / "eda_stat.parquet").to_pandas()
    assert {"half_life_sessions", "ar1_phi_log", "adf_p", "kpss_p", "iv_rv_spread_mean"} <= set(stats.stat)
    # No holdout sample: M1's EDA looks at nothing on or after holdout_start (amended M1, review A).
    assert set(stats["sample"]) == {"pre_holdout", "burn_in", "dev"}


def test_the_eda_on_synthetic_files_recovers_the_known_truth(pipeline):
    # One path of 9,272 sessions, so tolerances are wide: the full-sample phi has sd about
    # sqrt((1 - 0.98**2)/9272) = 0.0021, and the half-life is a steep function of phi near 1.
    root, _ = pipeline
    report = json.loads(next((root / "eda").glob("*/eda_report.json")).read_text())
    assert report["data_mode"] == "synthetic"
    assert report["sample"]["last"] == "2024-06-28" and report["sample"]["holdout_start"] == "2024-07-01"
    full = report["half_life"]["pre_holdout"]
    assert full["phi"] == pytest.approx(0.98, abs=0.01)
    assert full["phi_ci95"][0] < 0.98 < full["phi_ci95"][1]
    assert report["clustering"]["r2"]["ljung_box_p"][-1] < 1e-10  # GARCH returns cluster
    assert report["clustering"]["r2"]["acf"][0] > 5 * abs(report["clustering"]["r"]["acf"][0])
    assert report["unit_root"]["pre_holdout"]["adf_p"] < 0.05  # ln VIX is stationary by construction
    assert report["unit_root"]["pre_holdout"]["n"] == report["unit_root"]["pre_holdout"]["n_nonmissing"]  # no gaps
    assert report["term_structure"]["pre_holdout"]["contango_share"] == pytest.approx(0.8569, abs=0.1)
    assert report["vol_of_vol"]["mean_vov_annualised"] == pytest.approx(0.9525, rel=0.05)
    assert report["iv_rv_spread"]["pre_holdout"]["mean_variance"] > 0
    assert report["vol_of_vol"]["e6_n_boot"] == 999 and 0 < report["vol_of_vol"]["e6_p_bootstrap"] <= 1
    # The synthetic ln VIX has iid normal shocks, so the block rule keeps no runs: blocks of 1 to 4
    # (300 such paths in the walkthrough, section 7.3: mean 1.6, at most 4; amended M1, review B).
    assert 1 <= report["vol_of_vol"]["e6_block"] <= 4
    # The synthetic phi is 0.98, inside the size-checked range (eda.E6_PHI_MAX; M1 final check).
    assert report["vol_of_vol"]["e6_ar1_phi"] <= 0.985 and report["vol_of_vol"]["e6_verdict"] == "not shown"


def test_the_app_eda_core_gives_the_m1_reports_own_numbers(pipeline):
    # ivf.app_results reads eda_core, not the M1 report; both must give the same numbers on the
    # same interim tables (final polish: eda_core was split out of eda_report).
    root, _ = pipeline
    report = json.loads(next((root / "eda").glob("*/eda_report.json")).read_text())
    proto = protocol.load_protocol(protocol.protocol_path(synthetic=True))
    iv, price, _ = build_interim.read_interim(root)
    dates, periods, iv_eda, price_eda = process.eda_inputs(calendar.master_calendar(proto), iv, price, proto.holdout_start)
    core = process.eda_core(dates, periods, build.build_features(dates, iv_eda, price_eda), iv_eda, proto.holdout_start)
    assert json.loads(json.dumps({k: core[k] for k in ("clustering", "persistence", "half_life")})) == {
        k: report[k] for k in ("clustering", "persistence", "half_life")}


def test_the_eda_run_id_changes_with_the_inputs():
    from ivf import contract
    from ivf.data.process import run_id_for

    git = contract.GitState("1" * 40, False)
    a = run_id_for("a" * 64, git, "real", ["b" * 64, "c" * 64])
    assert a == run_id_for("a" * 64, git, "real", ["c" * 64, "b" * 64])  # order does not matter
    assert a != run_id_for("a" * 64, git, "synthetic", ["b" * 64, "c" * 64])
    assert a != run_id_for("a" * 64, contract.GitState("1" * 40, True), "real", ["b" * 64, "c" * 64])
    assert len(a) == 12 and all(ch in "0123456789abcdef" for ch in a)


def test_check_exits_1_when_the_gap_policy_is_broken(tmp_path):
    # A holdout session with no S&P 500 close: build the files, then drop one row from the raw CSV.
    root = tmp_path / "implied-vol-forecasting" / "synthetic"
    synthetic.write_synthetic_raw(root, proto())
    record = next(r for r in raw.read_provenance(root) if r.symbol == "^GSPC")
    lines = (root / record.raw_path).read_text().splitlines(keepends=True)
    gapped = "".join(line for line in lines if not line.startswith("2025-04-08"))
    sha, path, _ = raw.store_snapshot(root, "synthetic", "^GSPC", gapped.encode())
    raw.append_provenance(root, raw.FetchRecord("e" * 32, "synthetic", "^GSPC", "u", None, "2026-10-28T12:00:00.000000Z", None, len(gapped), sha, path, True, "test"))
    assert run("--synthetic", "build-interim", data_dir=tmp_path).returncode == 0
    result = run("--synthetic", "check", data_dir=tmp_path)
    assert result.returncode == 1
    assert "gap policy: VIOLATED" in result.stdout and "2025-04-08" in result.stdout
    assert run("--synthetic", "eda", data_dir=tmp_path).returncode == 1


def test_commands_without_data_exit_3_with_a_hint(tmp_path):
    result = run("--synthetic", "build-interim", data_dir=tmp_path)
    assert result.returncode == 3 and "run `python -m ivf.data fetch` first" in result.stderr
    result = run("--synthetic", "check", data_dir=tmp_path)
    assert result.returncode == 3 and "build-interim" in result.stderr


def test_the_half_life_of_the_written_series_is_finite_and_positive(pipeline):
    root, _ = pipeline
    report = json.loads(next((root / "eda").glob("*/eda_report.json")).read_text())
    for sample in ("pre_holdout", "burn_in", "dev"):
        value = report["half_life"][sample]["half_life_sessions"]
        assert value is not None and 0 < value < math.inf


def test_rerunning_eda_on_the_same_inputs_replaces_its_own_run_row(pipeline):
    # ADR 0001 D3: the same inputs give the same run_id, so a re-run overwrites its own partition.
    root, _ = pipeline
    data_dir = root.parent.parent
    assert run("--synthetic", "eda", data_dir=data_dir).returncode == 0
    runs = pd.read_parquet(root / "processed" / "runs.parquet")
    assert len(runs) == 1
    assert len(list((root / "processed").glob("run_id=*"))) == 1
