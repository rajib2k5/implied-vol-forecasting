"""Tests for schemas/app-data/*.schema.json themselves (ADR 0001 D5, "Contract tests").

The schemas are the contract, so they get their own checks: each is a valid
JSON Schema 2020-12 document, each $ref points at something that exists, and
each file schema is wired to the right dataset.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from ivf import config, contract

# ADR 0001 / data-model.md name these six files; schema 1.1.0 (final polish) adds evaluation and eda.
EXPECTED = {"envelope", "manifest", "term_structure", "model_race", "error_surface", "equity_curves", "evaluation", "eda"}
DATASETS = sorted(EXPECTED - {"envelope"})
BASE = "https://github.com/rajib2k5/implied-vol-forecasting/schemas/app-data/"

SCHEMAS = contract.load_schemas()
REGISTRY = contract.build_registry(SCHEMAS)


def iter_refs(node: Any) -> Iterator[str]:
    """Every value of a "$ref" key anywhere in a schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_refs(item)


def test_the_eight_contract_schemas_are_present():
    assert set(SCHEMAS) == EXPECTED
    assert {p.name for p in config.schema_dir().glob("*.schema.json")} == {f"{n}.schema.json" for n in EXPECTED}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_schema_is_a_valid_2020_12_schema(name):
    schema = SCHEMAS[name]
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)  # raises SchemaError if the schema itself is malformed


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_id_is_the_url_that_relative_refs_resolve_against(name):
    # "$ref": "envelope.schema.json" is resolved against the file's own $id,
    # so the $id must end in the file's name.
    assert SCHEMAS[name]["$id"] == f"{BASE}{name}.schema.json"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_ref_resolves_locally(name):
    schema = SCHEMAS[name]
    refs = list(iter_refs(schema))
    assert refs, f"{name} has no $ref at all; the check below would prove nothing"
    resolver = REGISTRY.resolver(base_uri=schema["$id"])
    for ref in refs:
        resolver.lookup(ref)  # raises referencing.exceptions.Unresolvable if the target is missing


@pytest.mark.parametrize("name", DATASETS)
def test_file_schema_extends_the_envelope_and_fixes_its_dataset(name):
    schema = SCHEMAS[name]
    assert schema["$ref"] == "envelope.schema.json"
    assert schema["properties"]["dataset"] == {"const": name}
    assert schema["properties"]["payload"] == {"$ref": "#/$defs/payload"}


def envelope_for(name: str, **changes: Any) -> dict[str, Any]:
    git = contract.GitState(commit="0" * 40, dirty=True)
    envelope = contract.build_envelope(name, {}, "synthetic", git=git)
    envelope.update(changes)
    return envelope


@pytest.mark.parametrize("name", DATASETS)
def test_every_dataset_schema_enforces_the_envelope_rules(name):
    # The payload {} is also wrong for every dataset, but the envelope problems
    # must be reported as well: proof that the file schema layers on the envelope.
    document = envelope_for(name, data_mode="paper", schema_version="2.0.0")
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(document)
    paths = {e.json_path for e in caught.value.errors}
    assert {"$.data_mode", "$.schema_version"} <= paths


@pytest.mark.parametrize("name", DATASETS)
def test_every_dataset_schema_enforces_its_payload_rules(name):
    # The envelope alone accepts {}; each dataset's payload schema must not.
    contract.validate_envelope(envelope_for(name))
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(envelope_for(name))
    assert any(e.json_path.startswith("$.payload") for e in caught.value.errors)


# --- trailing newline: Python's `$` is not ECMA-262's `$` (amended M0) --------
#
# JSON Schema patterns are ECMA-262 regexes, but python-jsonschema runs them with
# Python's `re.search`, where `$` ALSO matches just before a final "\n". So
# "^[0-9a-f]{40}$" accepted 40 hex characters plus "\n" in Python, while the
# app's validator (JavaScript RegExp, as in Ajv) rejects it: the two sides of the
# contract test disagreed. Every pattern now ends in `$(?!\n)`: "end, and no
# newline follows". In both engines that only matches at the true end of the
# string (ADR 0001 D5, "Contract tests").

MANIFEST_PAYLOAD: dict[str, Any] = {
    "run_id": "0123456789ab",
    "notice": "Synthetic data, not market data",
    "protocol": {
        "protocol_hash": "b" * 64,
        "train_start": "1990-01-02",
        "dev_start": "2000-01-03",
        "holdout_start": "2024-07-01",
        "data_end": "2026-10-23",
        "horizons": [1, 5, 21],
        "fold_len": 63,
        "min_cell_n": 20,
        "holdout_looks": 0,
        "vix_regime_edges": [15, 25],
    },
    "files": [
        {
            "dataset": "term_structure",
            "path": "term_structure.json",
            "sha256": "c" * 64,
            "bytes": 1234,
            "license_class": "series",
        }
    ],
    "models": [{"model_id": "rw", "label": "Random walk", "family": "naive", "targets": ["vix_level"]}],
    "targets": [{"target_id": "vix_level", "label": "VIX close", "units": "vol_points", "horizons": [1, 5, 21]}],
}
TERM_STRUCTURE_PAYLOAD: dict[str, Any] = {
    "dates": ["2024-08-05"],
    "tenors": [{"symbol": "VIX", "tenor_days": 30, "label": "30D"}],
    "levels": [[30.00]],  # made up (amended M1, review A: the value here was a real VIX level of that date)
    "vix_vix3m": [1.05],
    "backwardation": [True],
}
EQUITY_CURVES_PAYLOAD: dict[str, Any] = {
    "dates": ["2024-07-01"],
    "holdout_start_idx": 0,
    "cost_bps": 5,
    "strategies": [
        {
            "strategy_id": "rw_sign",
            "label": "Random-walk sign",
            "kind": "forecast",
            "model_id": "rw",  # a string here, so nullableId's pattern applies
            "horizon": 1,
            "equity": [1.0],
            "drawdown": [0.0],
        }
    ],
    "regime_segments": [],
    "summary": [],
    "caveats": [],
}
EVALUATION_PAYLOAD: dict[str, Any] = {
    "period": "dev",
    "source_run_id": "0123456789ab",
    "dm_method": "fixed_b",
    "dm_fixed_b": 0.1,
    "mz_method": "hansen_hodrick",
    "losses": [{"target_id": "vix_level", "horizon": 1, "model_id": "rw", "n": 10, "rmse": 1.0, "mae": 0.8, "bias": 0.0,
                "mape": 0.05, "qlike": 0.01, "rmse_ratio_rw": 1.0, "qlike_ratio_rw": 1.0}],
    "dm": [{"target_id": "vix_level", "horizon": 1, "loss": "qlike", "model_a": "har_log", "model_b": "rw", "n": 10,
            "dm_stat": -2.1, "p_value": 0.04, "p_holm": 0.08, "lower_loss": "har_log"}],
    "mz": [{"target_id": "vix_level", "horizon": 1, "model_id": "rw", "n": 10, "alpha": 0.1, "alpha_se": 0.05, "beta": 0.99,
            "beta_se": 0.01, "r2": 0.9, "wald_p": 0.5}],
    "calibration": {"cells": 28, "helps": 0},
}
EDA_PAYLOAD: dict[str, Any] = {
    "period": "pre_holdout",
    "source_run_id": "0123456789ab",
    "first": "1990-01-02",
    "last": "2024-06-28",
    "holdout_start": "2024-07-01",
    "nlags": 2,
    "acf": [{"series_id": "abs_r", "label": "|r|", "n": 100, "acf": [0.2, 0.1], "band95": 0.196, "ljung_box_p_last": 0.01}],
    "half_life": [{"sample": "dev", "phi": 0.98, "phi_ci95": [0.97, 0.99], "half_life_sessions": 34.31, "half_life_ci95": [22.76, None], "n": 100}],
}
BASE_PAYLOADS = {
    "manifest": MANIFEST_PAYLOAD,
    "term_structure": TERM_STRUCTURE_PAYLOAD,
    "equity_curves": EQUITY_CURVES_PAYLOAD,
    "evaluation": EVALUATION_PAYLOAD,
    "eda": EDA_PAYLOAD,
}

# (dataset, path to a string field, the JSON path jsonschema reports for it)
NEWLINE_CASES = [
    ("manifest", ("schema_version",), "$.schema_version"),
    ("manifest", ("generated_at",), "$.generated_at"),
    ("manifest", ("source_commit",), "$.source_commit"),
    ("manifest", ("payload", "run_id"), "$.payload.run_id"),
    ("manifest", ("payload", "protocol", "protocol_hash"), "$.payload.protocol.protocol_hash"),
    ("manifest", ("payload", "protocol", "train_start"), "$.payload.protocol.train_start"),
    ("manifest", ("payload", "files", 0, "path"), "$.payload.files[0].path"),
    ("manifest", ("payload", "files", 0, "sha256"), "$.payload.files[0].sha256"),
    ("manifest", ("payload", "models", 0, "model_id"), "$.payload.models[0].model_id"),
    ("term_structure", ("payload", "tenors", 0, "symbol"), "$.payload.tenors[0].symbol"),
    ("equity_curves", ("payload", "strategies", 0, "model_id"), "$.payload.strategies[0].model_id"),
    ("evaluation", ("payload", "source_run_id"), "$.payload.source_run_id"),
    ("evaluation", ("payload", "dm", 0, "lower_loss"), "$.payload.dm[0].lower_loss"),
    ("eda", ("payload", "first"), "$.payload.first"),
]


def base_document(dataset: str) -> dict[str, Any]:
    git = contract.GitState(commit="0" * 40, dirty=True)
    return contract.build_envelope(dataset, copy.deepcopy(BASE_PAYLOADS[dataset]), "synthetic", git=git)


def with_trailing_newline(document: dict[str, Any], path: tuple[Any, ...]) -> dict[str, Any]:
    changed = copy.deepcopy(document)
    node: Any = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]] + "\n"
    return changed


def newline_errors(dataset: str, path: tuple[Any, ...]) -> list[Any]:
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(with_trailing_newline(base_document(dataset), path))
    return caught.value.errors


@pytest.mark.parametrize("dataset", sorted(BASE_PAYLOADS))
def test_newline_base_documents_are_valid(dataset):
    # Without this, a "rejected" result below could come from some other defect
    # in the base document and the newline tests would prove nothing.
    contract.validate_document(base_document(dataset))


@pytest.mark.parametrize(("dataset", "path", "json_path"), NEWLINE_CASES, ids=[c[2] for c in NEWLINE_CASES])
def test_newline_after_a_valid_value_is_rejected(dataset, path, json_path):
    # e.g. "0000…0000" (40 hex) is valid; "0000…0000\n" must not be. The ONLY
    # error must be that field's pattern, so nothing else masks the result.
    errors = newline_errors(dataset, path)
    assert [(e.json_path, e.validator) for e in errors] == [(json_path, "pattern")]


def iter_patterns(node: Any) -> Iterator[str]:
    """Every value of a "pattern" keyword anywhere in a schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pattern" and isinstance(value, str):
                yield value
            else:
                yield from iter_patterns(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_patterns(item)


def test_newline_cases_cover_every_pattern_in_the_schemas():
    # Coverage by behaviour, not by reading the regex text: collect the pattern
    # that actually rejected each case above, and require that every distinct
    # pattern in the schemas is among them. A new pattern added without a
    # newline case makes this fail, so it cannot slip in unguarded.
    exercised = {e.schema["pattern"] for d, p, _ in NEWLINE_CASES for e in newline_errors(d, p)}
    declared = {p for schema in SCHEMAS.values() for p in iter_patterns(schema)}
    assert declared, "no patterns found; the comparison below would prove nothing"
    assert exercised == declared


# --- schema 1.1.0 additions (final polish) -------------------------------------

COST_ROW = {"strategy_id": "rw_sign", "period": "dev", "cost_bps": 5.0, "n_days": 1, "ann_return": 0.1, "ann_vol": 0.2,
            "sharpe_ann": 0.5, "psr0": 0.6, "dsr": 0.4, "n_trials": 5, "max_drawdown": -0.1, "turnover_ann": 2.0}


def document_with(dataset: str, change) -> dict[str, Any]:
    document = base_document(dataset)
    change(document["payload"])
    return document


def test_equity_cost_sensitivity_is_optional_and_each_row_needs_its_cost_level():
    contract.validate_document(base_document("equity_curves"))  # 1.0.0 documents without it stay valid (MINOR)
    contract.validate_document(document_with("equity_curves", lambda p: p.update(cost_sensitivity=[dict(COST_ROW)])))
    without_cost = {k: v for k, v in COST_ROW.items() if k != "cost_bps"}
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(document_with("equity_curves", lambda p: p.update(cost_sensitivity=[without_cost])))
    assert [e.json_path for e in caught.value.errors] == ["$.payload.cost_sensitivity[0]"]
    with pytest.raises(contract.SchemaViolation):  # the summary row's own rules apply: psr0 is a probability
        contract.validate_document(document_with("equity_curves", lambda p: p.update(cost_sensitivity=[{**COST_ROW, "psr0": 1.5}])))


@pytest.mark.parametrize(
    ("dataset", "change", "json_path"),
    [
        ("evaluation", lambda p: p["dm"][0].update(loss="mae"), "$.payload.dm[0].loss"),
        ("evaluation", lambda p: p["dm"][0].update(p_holm=1.2), "$.payload.dm[0].p_holm"),
        ("evaluation", lambda p: p["losses"][0].update(rmse=-1.0), "$.payload.losses[0].rmse"),
        ("evaluation", lambda p: p.update(dm_fixed_b=0), "$.payload.dm_fixed_b"),
        ("evaluation", lambda p: p["calibration"].update(helps=-1), "$.payload.calibration.helps"),
        ("eda", lambda p: p["acf"][0].update(acf=[1.2, 0.1]), "$.payload.acf[0].acf[0]"),
        ("eda", lambda p: p["half_life"][0].update(sample="holdout"), "$.payload.half_life[0].sample"),
        ("eda", lambda p: p.update(period="dev"), "$.payload.period"),
    ],
)
def test_new_dataset_schemas_reject_out_of_range_values(dataset, change, json_path):
    with pytest.raises(contract.SchemaViolation) as caught:
        contract.validate_document(document_with(dataset, change))
    assert [e.json_path for e in caught.value.errors] == [json_path]
