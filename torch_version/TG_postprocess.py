#!/usr/bin/env python3
"""Render vesicle trajectories from a .bin file to PNG frames (for create_video.py)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tools.load_ves2d_file import load_ves2d_file


def _axis_limits(vesx, vesy, xlim, ylim, pad=0.05):
    if xlim is None:
        xmin, xmax = float(vesx.min()), float(vesx.max())
        dx = (xmax - xmin) or 1.0
        xlim = (xmin - pad * dx, xmax + pad * dx)
    if ylim is None:
        ymin, ymax = float(vesy.min()), float(vesy.max())
        dy = (ymax - ymin) or 1.0
        ylim = (ymin - pad * dy, ymax + pad * dy)
    return xlim, ylim


def plot_vesicle_data(
    file_name: str | Path,
    name: str,
    *,
    output_dir: str | Path = "output",
    xlim=None,
    ylim=None,
    dpi: int = 100,
) -> Path:
    """Write frames to ``output_dir/name/{1..ntime}.png``."""
    file_name = Path(file_name)
    output_dir = Path(output_dir)
    frames_dir = output_dir / name
    frames_dir.mkdir(parents=True, exist_ok=True)

    vesx, vesy, time, _n, _nv, xinit, yinit = load_ves2d_file(str(file_name))
    xlim, ylim = _axis_limits(vesx, vesy, xlim, ylim)
    fig, ax = plt.subplots(figsize=(8, 8))
    x1 = np.vstack((xinit, xinit[0, :]))
    y1 = np.vstack((yinit, yinit[0, :]))
    ax.plot(x1, y1, "red", linewidth=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"t = 0 {name}")
    fig.savefig(frames_dir / f"0.png", dpi=dpi)
        

    for it in tqdm(range(len(time)), desc=f"plot {name}"):
        fig, ax = plt.subplots(figsize=(8, 8))
        x1 = np.vstack((vesx[:, :, it], vesx[0, :, it]))
        y1 = np.vstack((vesy[:, :, it], vesy[0, :, it]))
        ax.plot(x1, y1, "red", linewidth=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"t = {it + 1} {name}")
        fig.savefig(frames_dir / f"{it + 1}.png", dpi=dpi)
        plt.close(fig)

    return frames_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", required=True, help="Path to simulation .bin file")
    p.add_argument("--name", required=True, help="Subfolder under output_dir for PNG frames")
    p.add_argument(
        "--output-dir",
        default="output",
        help="Parent directory for frame folders (default: output)",
    )
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_vesicle_data(
        args.bin,
        args.name,
        output_dir=args.output_dir,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
