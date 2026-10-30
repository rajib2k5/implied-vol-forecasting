"""Print the README's results tables from the committed SYNTHETIC app data, or check the README.

    uv run python tools/results_table.py            # print the Markdown block
    uv run python tools/results_table.py --check    # exit 1 if README.md's block differs from it

Every number in the block is read from ``app/public/data/synthetic/{evaluation,equity_curves,eda}.json``
(the files ``tools/export_app_data.py --mode synthetic`` writes from the synthetic dev store, and
whose ``--check`` proves they match that store). Nothing is recomputed except counts (how many DM
pairs have a Holm p below 0.05) and the choice of the lowest value in a column; the numbers are
printed exactly as the JSON holds them. The tool refuses any file whose ``data_mode`` is not
``synthetic``: real results are measured locally and are not published under the current data
policy (ADR 0001 D2).

The block sits in README.md between ``<!-- results:begin -->`` and ``<!-- results:end -->``.
Exit codes: 0 ok, 1 the README block differs (``--check``), 2 a file is missing, not synthetic, or
the README has no block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "app" / "public" / "data" / "synthetic"
README = REPO / "README.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
PANELS = (("vix_level", 1), ("vix_level", 5), ("vix_level", 21), ("rv21", 21))
PANEL_LABEL = {("vix_level", 1): "VIX h=1", ("vix_level", 5): "VIX h=5", ("vix_level", 21): "VIX h=21", ("rv21", 21): "rv21 h=21"}
HOLM_ALPHA = 0.05


class NotPublishable(Exception):
    """A file is missing or is not synthetic."""


def load(name: str, data_dir: Path = DATA) -> dict:
    path = data_dir / f"{name}.json"
    if not path.is_file():
        raise NotPublishable(f"missing: {path} (run `uv run python tools/export_app_data.py --mode synthetic`)")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("data_mode") != "synthetic":
        raise NotPublishable(f"refusing {path}: data_mode is {document.get('data_mode')!r}, not 'synthetic' (ADR 0001 D2)")
    return document["payload"]


def cell(value) -> str:
    return "n/a" if value is None else f"{value}"


def loss_table(evaluation: dict, key: str, models: list[str]) -> list[str]:
    """Rows = models, columns = panels; the lowest value of each column in bold."""
    values = {(x["target_id"], x["horizon"], x["model_id"]): x[key] for x in evaluation["losses"]}
    best = {}
    for panel in PANELS:
        column = [values[(*panel, m)] for m in models if values.get((*panel, m)) is not None]
        best[panel] = min(column) if column else None
    lines = ["| Model | " + " | ".join(PANEL_LABEL[p] for p in PANELS) + " |", "|---|" + "---:|" * len(PANELS)]
    for model in models:
        cells = []
        for panel in PANELS:
            v = values.get((*panel, model))
            cells.append("" if v is None else (f"**{v}**" if v == best[panel] else cell(v)))
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    return lines


def block(data_dir: Path = DATA) -> str:
    manifest = load("manifest", data_dir)
    evaluation = load("evaluation", data_dir)
    equity = load("equity_curves", data_dir)
    eda = load("eda", data_dir)
    models = [m["model_id"] for m in manifest["models"]]
    significant = sum(1 for d in evaluation["dm"] if d["p_holm"] is not None and d["p_holm"] < HOLM_ALPHA)
    labels = {s["strategy_id"]: s["label"] for s in equity["strategies"]}
    costs = sorted({r["cost_bps"] for r in equity["cost_sensitivity"]})
    pre = next(h for h in eda["half_life"] if h["sample"] == "pre_holdout")
    acf = {s["series_id"]: s for s in eda["acf"]}
    lines = [
        BEGIN,
        f"Synthetic data only (development period; the backtest runs {equity['dates'][0]} to {equity['dates'][-1]}). "
        "Produced by `uv run python tools/results_table.py`.",
        "",
        "**RMSE relative to the random walk** (below 1 beats it; lowest in bold; ratios carry 6 decimals so near-ties are told apart):",
        "",
        *loss_table(evaluation, "rmse_ratio_rw", models),
        "",
        "**QLIKE relative to the random walk:**",
        "",
        *loss_table(evaluation, "qlike_ratio_rw", models),
        "",
        f"- Diebold-Mariano ({evaluation['dm_method']}, b = {evaluation['dm_fixed_b']}): {significant} of "
        f"{len(evaluation['dm'])} model pairs differ at Holm-adjusted p < {HOLM_ALPHA}.",
        f"- Walk-forward bias calibration helps in {evaluation['calibration']['helps']} of {evaluation['calibration']['cells']} cells.",
        f"- ln VIX half-life (AR(1), sessions before the holdout): {pre['half_life_sessions']} sessions "
        f"(95% CI {cell(pre['half_life_ci95'][0])} to {cell(pre['half_life_ci95'][1])}); ACF at lag 1: "
        f"|r| {acf['abs_r']['acf'][0]}, r² {acf['r2']['acf'][0]}, r {acf['r']['acf'][0]} (white-noise band ±{acf['r']['band95']}).",
        "",
        f"**Backtest at {equity['cost_bps']:g} bps** (growth of 1; Sharpe annualised as daily Sharpe × √252; PSR against "
        "Sharpe 0; DSR deflated for the trials):",
        "",
        "| Strategy | Ann. return | Sharpe (annualised) | PSR | DSR | Max drawdown | Trials |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in equity["summary"]:
        lines.append(f"| {labels[r['strategy_id']]} | {cell(r['ann_return'])} | {cell(r['sharpe_ann'])} | {cell(r['psr0'])} | "
                     f"{cell(r['dsr'])} | {cell(r['max_drawdown'])} | {r['n_trials']} |")
    lines += ["", "**Annualised Sharpe ratio by cost level** (bps per unit of turnover):", "",
              "| Strategy | " + " | ".join(f"{c:g} bps" for c in costs) + " |", "|---|" + "---:|" * len(costs)]
    for sid in labels:
        by_cost = {r["cost_bps"]: r["sharpe_ann"] for r in equity["cost_sensitivity"] if r["strategy_id"] == sid}
        lines.append(f"| {labels[sid]} | " + " | ".join(cell(by_cost.get(c)) for c in costs) + " |")
    lines.append(END)
    return "\n".join(lines) + "\n"


def readme_block(text: str) -> str | None:
    start, stop = text.find(BEGIN), text.find(END)
    if start < 0 or stop < start:
        return None
    return text[start : stop + len(END)] + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print or check the README's synthetic results tables.")
    parser.add_argument("--check", action="store_true", help="compare with README.md instead of printing")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--readme", type=Path, default=README)
    args = parser.parse_args(argv)
    try:
        fresh = block(args.data_dir)
    except NotPublishable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.check:
        sys.stdout.write(fresh)
        return 0
    current = readme_block(args.readme.read_text(encoding="utf-8"))
    if current is None:
        print(f"error: {args.readme} has no {BEGIN} ... {END} block", file=sys.stderr)
        return 2
    if current != fresh:
        print(f"differs: {args.readme}'s results block is not what the committed synthetic data gives; "
              "paste the output of `uv run python tools/results_table.py`", file=sys.stderr)
        return 1
    print(f"ok: {args.readme}'s results block matches the committed synthetic data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
