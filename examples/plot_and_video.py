#!/usr/bin/env python3
"""Plot frames from an example .bin and encode a video (wraps torch_version postprocess tools)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_PKG = REPO_ROOT / "torch_version"
sys.path.insert(0, str(TORCH_PKG))

from create_video import create_video_from_images  # noqa: E402
from TG_postprocess import plot_vesicle_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Resolved example config.json")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    outfile = cfg.get("outfile", "run")
    output_dir = Path(cfg.get("output_dir", "output"))
    bin_path = output_dir / f"{outfile}.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(f"Simulation output not found: {bin_path}")

    frames_dir = plot_vesicle_data(
        bin_path,
        outfile,
        output_dir=output_dir,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        dpi=args.dpi,
    )
    video_path = output_dir / f"{outfile}.mp4"
    create_video_from_images(frames_dir, video_path, args.fps)
    print(f"Frames: {frames_dir}")
    print(f"Video:  {video_path}")


if __name__ == "__main__":
    main()
