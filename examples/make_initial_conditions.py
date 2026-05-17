#!/usr/bin/env python3
"""Build initial-condition .npy files (shape 2N x nv) for ex1 (ex2/ex3 use bundled .npy ICs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NEWTORCH = REPO_ROOT / "newtorch"
sys.path.insert(0, str(NEWTORCH))

from curve_batch_compile import Curve  # noqa: E402

N = 32
REDUCED_AREA = 0.65

EXAMPLES = ("ex1_one_ves_parabolic",)


def unit_ellipse(oc: Curve, ra: float = REDUCED_AREA) -> np.ndarray:
    x0 = oc.ellipse(N, torch.tensor([ra], dtype=torch.float32))
    _, _, length = oc.geomProp(x0)
    return (x0 / length).cpu().numpy()


def transform(
    x0: np.ndarray,
    center: tuple[float, float],
    angle: float = 0.0,
) -> np.ndarray:
    """Rotate and translate a single vesicle (2N x 1)."""
    c, s = np.cos(angle), np.sin(angle)
    x = x0[:N, 0].copy()
    y = x0[N:, 0].copy()
    xr = c * x - s * y + center[0]
    yr = s * x + c * y + center[1]
    return np.stack([xr, yr], axis=0)


def write_example(name: str, array: np.ndarray) -> Path:
    out = REPO_ROOT / "examples" / name / "initial.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, array.astype(np.float64))
    return out


def build_ex1(oc: Curve) -> np.ndarray:
    """Unit-length ellipse, tilted and shifted off the parabolic centerline (y = 0)."""
    x0 = unit_ellipse(oc)
    return transform(x0, center=(0.0, 0.065), angle=np.pi / 2)[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        choices=EXAMPLES,
        help="Example id(s) to build (default: ex1)",
    )
    args = parser.parse_args()
    names = args.examples or list(EXAMPLES)

    oc = Curve()
    for name in names:
        array = build_ex1(oc)
        path = write_example(name, array)
        print(f"{name}: wrote {path}  (2N={array.shape[0]}, nv={array.shape[1]})")


if __name__ == "__main__":
    main()
