"""``configs/models.yaml``: the model registry (``model_def``), the milestone 2 baseline settings and
the milestone 3 GARCH settings (``load_garch_settings``).

Read strictly, like the protocol (``ivf.protocol``): an unknown key, a missing key or a value of
the wrong type is an error, because a typo would otherwise change a forecast silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ivf import config
from ivf.protocol import ProtocolError, _exact_keys, _int, _read_yaml

MODELS_FILE = "models.yaml"
FAMILIES = {"naive", "arima", "har", "garch", "calibrated", "ml", "implied"}
FIT_SPACES = {"level", "log_level", "returns", "variance"}
MULTISTEP = {"direct", "iterated", "none"}
_MODEL_KEYS = {"model_id", "label", "family", "fit_space", "multistep", "targets", "milestone", "retired"}
_BASELINE_KEYS = {"symbols", "min_train_sessions", "hist_window", "arima"}
_ARIMA_KEYS = {"max_p", "max_q", "d_rule", "kpss_alpha"}
D_RULES = ("d0", "kpss")


@dataclass(frozen=True)
class ModelDef:
    model_id: str
    label: str
    family: str
    fit_space: str
    multistep: str
    targets: tuple[str, ...]
    milestone: int
    retired: bool


@dataclass(frozen=True)
class BaselineSettings:
    symbols: tuple[str, ...]
    min_train_sessions: int
    hist_window: int
    max_p: int
    max_q: int
    d_rule: str  # "d0" (default) or "kpss" (an option; amended M2, review A)
    kpss_alpha: float


def target_id_for(symbol: str) -> str:
    """``VIX`` -> ``vix_level`` (the protocol target); ``VIX3M`` -> ``vix3m_level`` (a secondary target)."""
    return f"{symbol.lower()}_level"


def load_models(path: Path | None = None) -> tuple[tuple[ModelDef, ...], BaselineSettings]:
    path = path or config.configs_dir() / MODELS_FILE
    document = _exact_keys(str(path), _read_yaml(path), {"models", "baselines", "garch"})
    models = []
    for i, entry in enumerate(document["models"]):
        where = f"{path}: models[{i}]"
        entry = _exact_keys(where, entry, _MODEL_KEYS)
        for key, allowed in (("family", FAMILIES), ("fit_space", FIT_SPACES), ("multistep", MULTISTEP)):
            if entry[key] not in allowed:
                raise ProtocolError(f"{where}.{key}: {entry[key]!r} is not one of {sorted(allowed)}")
        if not isinstance(entry["retired"], bool):
            raise ProtocolError(f"{where}.retired: expected true or false")
        models.append(
            ModelDef(
                model_id=entry["model_id"],
                label=entry["label"],
                family=entry["family"],
                fit_space=entry["fit_space"],
                multistep=entry["multistep"],
                targets=tuple(entry["targets"]),
                milestone=_int(f"{where}.milestone", entry["milestone"], 0),
                retired=entry["retired"],
            )
        )
    ids = [m.model_id for m in models]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"{path}: a model_id appears twice")
    base = _exact_keys(f"{path}: baselines", document["baselines"], _BASELINE_KEYS)
    arima = _exact_keys(f"{path}: baselines.arima", base["arima"], _ARIMA_KEYS)
    alpha = arima["kpss_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ProtocolError(f"{path}: baselines.arima.kpss_alpha must be in (0, 1)")
    settings = BaselineSettings(
        symbols=tuple(base["symbols"]),
        min_train_sessions=_int(f"{path}: baselines.min_train_sessions", base["min_train_sessions"], 1),
        hist_window=_int(f"{path}: baselines.hist_window", base["hist_window"], 1),
        max_p=_int(f"{path}: baselines.arima.max_p", arima["max_p"], 0),
        max_q=_int(f"{path}: baselines.arima.max_q", arima["max_q"], 0),
        d_rule=arima["d_rule"],
        kpss_alpha=float(alpha),
    )
    if settings.d_rule not in D_RULES:
        raise ProtocolError(f"{path}: baselines.arima.d_rule must be one of {list(D_RULES)}, not {settings.d_rule!r}")
    return tuple(models), settings


_GARCH_KEYS = {"dists", "max_p", "d"}
GARCH_DISTS = ("normal", "t")


@dataclass(frozen=True)
class GarchSettings:
    dists: tuple[str, ...]  # innovation distributions tried at every cutoff; the lower AIC wins
    max_p: int  # arima_garch_log's largest AR order
    d: int  # arima_garch_log: 0 (default; AR with a constant on ln IV) or 1 (AR on its changes, no constant)


def load_garch_settings(path: Path | None = None) -> GarchSettings:
    """The ``garch`` block of ``configs/models.yaml`` (milestone 3), read as strictly as the rest."""
    path = path or config.configs_dir() / MODELS_FILE
    document = _exact_keys(str(path), _read_yaml(path), {"models", "baselines", "garch"})
    block = _exact_keys(f"{path}: garch", document["garch"], _GARCH_KEYS)
    dists = tuple(block["dists"])
    if not dists or len(set(dists)) != len(dists) or any(d not in GARCH_DISTS for d in dists):
        raise ProtocolError(f"{path}: garch.dists must be distinct values from {list(GARCH_DISTS)}")
    d = block["d"]
    if isinstance(d, bool) or d not in (0, 1):
        raise ProtocolError(f"{path}: garch.d must be 0 or 1")
    return GarchSettings(dists=dists, max_p=_int(f"{path}: garch.max_p", block["max_p"], 0), d=d)


def models_for(defs, target_id: str, max_milestone: int) -> list[str]:
    """The expected model list for one target: registered, not retired, up to a milestone, in registry order."""
    return [d.model_id for d in defs if target_id in d.targets and not d.retired and d.milestone <= max_milestone]
