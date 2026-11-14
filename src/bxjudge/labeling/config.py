"""Validated labeling policy; separate from match adjudication thresholds."""

import math
from pathlib import Path

import yaml

from ..config import config_loader


@config_loader
def load_config(path: str | Path = "config/labeling.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = {
        "input_folder",
        "output_dir",
        "per_video",
        "energy_long_edge",
        "uniform_ratio",
        "min_gap_s",
        "launcher_max_aspect",
        "val_ratio",
        "max_new_tokens",
        "model_dir",
        "prompts",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("labeling config has missing or unknown keys")
    for key in [
        "per_video",
        "energy_long_edge",
        "max_new_tokens",
    ]:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["uniform_ratio", "min_gap_s", "launcher_max_aspect", "val_ratio"]:
        v = cfg[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError(f"invalid {key}")
    if (
        not 0 <= cfg["uniform_ratio"] <= 1
        or not 0 <= cfg["val_ratio"] < 1
        or cfg["launcher_max_aspect"] < 1
    ):
        raise ValueError("invalid labeling ratios")
    if not isinstance(cfg["model_dir"], str) or not cfg["model_dir"].strip():
        raise ValueError("model_dir must name the pinned local snapshot")
    prompts = cfg["prompts"]
    if not isinstance(prompts, dict):
        raise TypeError("prompts must be a mapping")
    if set(prompts) != {"bey", "bey_part", "launcher"} or any(
        not isinstance(p, str) or not p.strip() or "<" in p or ">" in p
        for p in prompts.values()
    ):
        raise ValueError("prompts must name all three classes without markup")
    cfg["prompts"] = {cls: prompts[cls].strip() for cls in ("bey", "bey_part", "launcher")}
    if len({p.casefold() for p in cfg["prompts"].values()}) != len(prompts):
        raise ValueError("class prompts must be distinct")
    return cfg
