"""Tests for tools/results_table.py: the README's synthetic results block.

The data comes from the hand-made store of tests/app_store_fixture.py, exported with the real
exporter; the expected numbers are worked out by hand from that store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app_store_fixture import tiny_store

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import export_app_data  # noqa: E402
import results_table  # noqa: E402


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(export_app_data, "load_synthetic_store", lambda: tiny_store())
    for name, blob in export_app_data.generate("synthetic").items():
        (tmp_path / name).write_bytes(blob)
    return tmp_path


def test_block_prints_the_stores_numbers(data_dir):
    text = results_table.block(data_dir)
    # rmse ratio: rw 4/4 = 1.0, har_log 2/4 = 0.5 (lowest, bold); garch 3/4 = 0.75 on rv21.
    assert "| `rw` | 1.0 | 1.0 | 1.0 | 1.0 |" in text
    assert "| `har_log` | **0.5** | **0.5** | **0.5** |  |" in text
    assert "| `garch` |  |  |  | **0.75** |" in text
    # qlike ratio: har_log 0.05/0.2 = 0.25, garch 0.1/0.2 = 0.5.
    assert "| `har_log` | **0.25** | **0.25** | **0.25** |  |" in text
    # 6 primary pairs: Holm p 0.009 for five, 0.06 for one (raw p 0.003 for all); helps = [False, True, False].
    assert "5 of 6 model pairs differ at Holm-adjusted p < 0.05" in text
    assert "helps in 1 of 3 cells" in text
    assert "34.31 sessions (95% CI 22.76 to 68.97)" in text
    assert "|r| 0.2, r² 0.25, r 0.01 (white-noise band ±0.0196)" in text
    # Summary at 5 bps, fixture strategy index i: ann_return 0.1*i - 0.05, sharpe 0.5*i - 0.5.
    assert "| rv21 band 2 (PRIMARY) | -0.05 | -0.5 | 0.6 | 0.25 | -0.1 | 2 |" in text
    assert "| rv21 band 2 (PRIMARY) | 0.0 | -0.5 |" in text  # Sharpe at 0 and 5 bps
    assert text.startswith(results_table.BEGIN) and text.endswith(results_table.END + "\n")


def test_only_the_lower_of_two_values_that_tie_at_4_decimals_is_bold():
    # VIX h = 1 on the synthetic store: ARIMA 0.995048 and ARIMA-GARCH 0.99503 both read 0.995 at 4 decimals.
    evaluation = {"losses": [
        {"target_id": "vix_level", "horizon": 1, "model_id": "arima_log", "rmse_ratio_rw": 0.995048},
        {"target_id": "vix_level", "horizon": 1, "model_id": "arima_garch_log", "rmse_ratio_rw": 0.99503},
    ]}
    lines = results_table.loss_table(evaluation, "rmse_ratio_rw", ["arima_log", "arima_garch_log"])
    assert lines[2].startswith("| `arima_log` | 0.995048 |")
    assert lines[3].startswith("| `arima_garch_log` | **0.99503** |")


def test_check_passes_on_a_readme_with_the_block_and_fails_on_drift(data_dir, tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n" + results_table.block(data_dir) + "\nmore\n")
    assert results_table.main(["--check", "--data-dir", str(data_dir), "--readme", str(readme)]) == 0
    readme.write_text(readme.read_text().replace("**0.5**", "**0.4**", 1))
    assert results_table.main(["--check", "--data-dir", str(data_dir), "--readme", str(readme)]) == 1
    assert "differs:" in capsys.readouterr().err
    readme.write_text("# no block\n")
    assert results_table.main(["--check", "--data-dir", str(data_dir), "--readme", str(readme)]) == 2


def test_a_file_that_is_not_synthetic_is_refused(data_dir, capsys):
    path = data_dir / "evaluation.json"
    document = json.loads(path.read_text())
    document["data_mode"] = "real"
    path.write_text(json.dumps(document))
    assert results_table.main(["--data-dir", str(data_dir)]) == 2
    assert "not 'synthetic'" in capsys.readouterr().err
    (data_dir / "eda.json").unlink()
    with pytest.raises(results_table.NotPublishable, match="missing"):
        results_table.load("eda", data_dir)


def test_the_committed_readme_block_matches_the_committed_synthetic_data():
    # The gate a verifier re-runs: every README results number comes from this tool.
    assert results_table.main(["--check"]) == 0
