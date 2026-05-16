#!/usr/bin/env python3
"""Entry point: load vesnet_config.json and run MLARM simulation via driver_vesnet."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

from driver_vesnet import (
    N128_RBF_UPSAMPLE_MAX,
    N128_TRAINED_ROOT_DEFAULT,
    N32_TRAINED_ROOT_DEFAULT,
    RBF_UPSAMPLE_DEFAULT,
    simulate,
)

DEFAULT_PARAMS: dict = {
    "input": "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/shear_N32.npy",
    "outfile": "debug",
    "output_dir": "./output",
    "logging": True,
    "log_file": "debug.log",
    "log_to_console": True,
    "resolution": 32,
    "num_steps": 100,
    "dt": 1e-5,
    "flow": {
        "name": "vortex",
        "speed": 400.0,
        "chanWidth": 2.5,
        "vortexSize": 2.5,
    },
    "rbf_params": {
        "nlayers": 5,
        "rbf_upsample": 4,
    },
    "repulsion_params": {
        "use_repulsion": False,
        "repulsion_strength": 1e4,
        "eta": 1 / 32,
    },
    "trained_root": N32_TRAINED_ROOT_DEFAULT,
    "inner_near_root": (
        "/work/09452/alberto47/ls6/vesicle_nearF2024/"
        "trained_disth_nocoords/inner_downsample32"
    ),
    "relaxed_shape": "relaxed_shape.npy",
}


def _deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            _deep_update(out[key], val)
        else:
            out[key] = val
    return out


def normalize_config(raw: dict) -> dict:
    """Accept flat vesnet_config keys or nested sections."""
    params = deepcopy(DEFAULT_PARAMS)
    raw = deepcopy(raw)

    if "flow" not in raw:
        flow_patch = {}
        if "bgFlow" in raw:
            flow_patch["name"] = raw.pop("bgFlow")
        for key in ("speed", "chanWidth", "vortexSize"):
            if key in raw:
                flow_patch[key] = raw.pop(key)
        if flow_patch:
            raw["flow"] = flow_patch

    if "rbf_params" not in raw:
        rbf_patch = {}
        for key in ("nlayers", "rbf_upsample"):
            if key in raw:
                rbf_patch[key] = raw.pop(key)
        if rbf_patch:
            raw["rbf_params"] = rbf_patch

    if "repulsion_params" not in raw:
        rep_patch = {}
        for key in ("use_repulsion", "repulsion_strength", "eta"):
            if key in raw:
                rep_patch[key] = raw.pop(key)
        if rep_patch:
            raw["repulsion_params"] = rep_patch

    if "filename" in raw and "outfile" not in raw:
        raw["outfile"] = raw.pop("filename")

    merged = _deep_update(params, raw)
    if int(merged.get("resolution", 32)) == 128:
        if "output_dir" not in raw:
            merged["output_dir"] = "./output_N128"
        # Adv / near / ten-adv norms are loaded from .npy under ../trained (mytorch N128 entry).
        if merged.get("trained_root") == N32_TRAINED_ROOT_DEFAULT:
            merged["trained_root"] = N128_TRAINED_ROOT_DEFAULT
        rbf = merged.setdefault("rbf_params", {})
        rbf["nlayers"] = 3
        rbf["rbf_upsample"] = min(
            int(rbf.get("rbf_upsample", RBF_UPSAMPLE_DEFAULT)), N128_RBF_UPSAMPLE_MAX
        )
    return merged


def load_params(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_config(raw)


def setup_logger(params: dict) -> logging.Logger:
    logger = logging.getLogger("vesnet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if params.get("logging", True):
        log_name = params.get("log_file")
        if not log_name:
            log_name = f"debug_{params.get('outfile', 'run')}.log"
        handler = logging.FileHandler(log_name, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if params.get("log_to_console", True):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


def apply_cli_overrides(params: dict, args: argparse.Namespace) -> dict:
    if args.input is not None:
        params["input"] = args.input
    if args.outfile is not None:
        params["outfile"] = args.outfile
    if args.resolution is not None:
        params["resolution"] = args.resolution
    if args.num_steps is not None:
        params["num_steps"] = args.num_steps
    if args.bg_flow is not None:
        params.setdefault("flow", {})["name"] = args.bg_flow
    if args.speed is not None:
        params.setdefault("flow", {})["speed"] = args.speed
    if args.nlayers is not None:
        params.setdefault("rbf_params", {})["nlayers"] = args.nlayers
    if args.rbf_upsample is not None:
        params.setdefault("rbf_params", {})["rbf_upsample"] = args.rbf_upsample
    return params


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run vesnet MLARM simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "vesnet_config.json",
        help="Path to JSON config (default: vesnet_config.json next to this script)",
    )
    parser.add_argument("--input", type=str, help="Override initial-condition path")
    parser.add_argument("--outfile", type=str, help="Output base name (without .bin)")
    parser.add_argument("--resolution", type=int, choices=(32, 128))
    parser.add_argument("--num-steps", type=int, dest="num_steps")
    parser.add_argument("--bg-flow", type=str, dest="bg_flow")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--nlayers", type=int)
    parser.add_argument("--rbf-upsample", type=int, dest="rbf_upsample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.config)
    params = apply_cli_overrides(params, args)
    logger = setup_logger(params)
    logger.info("Config: %s", args.config)
    logger.info("Resolution N=%d", params["resolution"])
    simulate(params, logger)


if __name__ == "__main__":
    main()
