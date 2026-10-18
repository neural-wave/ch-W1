"""YAML experiment configuration: candidates, folds, locked block and promotion gate.

A config may `extends: other.yaml`: everything is inherited and deep-merged,
except `candidates`, which the child replaces when it defines them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .pipeline import ZONES, PipelineConfig

DEFAULT_GATE = {"min_rel_improvement": 0.005, "min_fold_share": 0.66, "max_zone_degradation": 0.02}


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: np.datetime64
    score_start: np.datetime64
    score_end: np.datetime64
    locked: bool = False

    def __post_init__(self):
        if not self.train_end < self.score_start <= self.score_end:
            raise ValueError(f"fold {self.name}: need train_end < score_start <= score_end")


def _day(value) -> np.datetime64:
    return np.datetime64(str(value), "D")


def _deep_merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    path = Path(path)
    content = path.read_bytes()
    raw = yaml.safe_load(content) or {}
    parent_sha = ""
    if parent := raw.pop("extends", None):
        base = load_config(path.parent / parent)
        parent_sha = base["_sha256"]
        candidates = raw.pop("candidates", None)
        raw = _deep_merge(base, raw)
        if candidates is not None:
            raw["candidates"] = candidates
    raw["_path"] = str(path)
    raw["_sha256"] = hashlib.sha256(content + parent_sha.encode()).hexdigest()
    return raw


def candidate_configs(raw: dict, names=None) -> list[PipelineConfig]:
    common = {"zones": raw.get("zones", list(ZONES)), "seed": raw.get("seed", 42)}
    defaults = _deep_merge(common, raw.get("defaults", {}))
    configs = [PipelineConfig.from_dict(_deep_merge(defaults, c)) for c in raw.get("candidates", [])]
    if names:
        missing = set(names) - {c.name for c in configs}
        if missing:
            raise ValueError(f"unknown candidates {sorted(missing)}")
        configs = [c for c in configs if c.name in set(names)]
    if not configs:
        raise ValueError("the config defines no candidates")
    if len({c.zones for c in configs}) != 1:
        raise ValueError("all candidates must share the same zones")
    return configs


def final_config(raw: dict, name: str | None = None) -> PipelineConfig:
    """The single candidate to train for submission (--candidate, `final: true`, or the only one)."""
    candidates = raw.get("candidates", [])
    if name:
        chosen = [c["name"] for c in candidates if c["name"] == name]
    else:
        chosen = [c["name"] for c in candidates if c.get("final")]
        if not chosen and len(candidates) == 1:
            chosen = [candidates[0]["name"]]
    if len(chosen) != 1:
        raise ValueError("select exactly one candidate (--candidate NAME or `final: true`)")
    return candidate_configs(raw, chosen)[0]


def monthly_folds(first_train_end, last_train_end, score_from_day: int = 19) -> list[Fold]:
    """One fold per month: train up to a month end, score day `score_from_day`..end of
    the next month, mimicking the 18-day gap before 19 October."""
    folds = []
    end, last = pd.Timestamp(str(first_train_end)), pd.Timestamp(str(last_train_end))
    while end <= last:
        month_start = end + pd.Timedelta(days=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        score_start = month_start + pd.Timedelta(days=score_from_day - 1)
        folds.append(Fold(f"m{month_start:%Y_%m}", _day(end.date()), _day(score_start.date()), _day(month_end.date())))
        end = month_end
    return folds


def folds_from_config(raw: dict, scheme: str | None = None) -> tuple[list[Fold], Fold | None]:
    spec = raw.get("folds", {})
    scheme = scheme or spec.get("scheme", "explicit")
    if scheme == "monthly":
        folds = monthly_folds(
            spec.get("first_train_end", "2025-12-31"),
            spec.get("last_train_end", "2026-07-31"),
            int(spec.get("score_from_day", 19)),
        )
    elif scheme == "explicit":
        folds = [
            Fold(f["name"], _day(f["train_end"]), _day(f["score_start"]), _day(f["score_end"]))
            for f in spec["list"]
        ]
    else:
        raise ValueError(f"unknown fold scheme {scheme!r}")
    locked = raw.get("locked")
    locked_fold = None
    if locked:
        locked_fold = Fold(locked["name"], _day(locked["train_end"]), _day(locked["score_start"]),
                           _day(locked["score_end"]), locked=True)
    return folds, locked_fold


def gate_from_config(raw: dict) -> dict:
    return {**DEFAULT_GATE, **(raw.get("gate") or {})}
