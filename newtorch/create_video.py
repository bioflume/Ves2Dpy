#!/usr/bin/env python3
"""Assemble numbered PNG frames (1.png, 2.png, ...) into a video via OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def create_video_from_images(image_folder: str | Path, output_video_file: str | Path, fps: int) -> None:
    image_folder = Path(image_folder)
    images = []
    for i in range(1, 100_000_000):
        name = f"{i}.png"
        if (image_folder / name).is_file():
            images.append(name)
        else:
            break

    if not images:
        raise FileNotFoundError(f"No numbered PNG frames found in {image_folder}")

    print(f"A total of {len(images)} images")
    first_image_path = image_folder / images[0]
    frame = cv2.imread(str(first_image_path))
    if frame is None:
        raise OSError(f"Error reading the first image: {first_image_path}")

    height, width, _layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_video_file), fourcc, fps, (width, height))

    for idx, image_name in enumerate(tqdm(images, desc="encode video")):
        image_path = image_folder / image_name
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue
        video.write(frame)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} images...")

    video.release()
    print(f"Video saved as {output_video_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-folder", required=True, help="Folder with 1.png, 2.png, ...")
    p.add_argument("--output", required=True, help="Output video path (.mp4)")
    p.add_argument("--fps", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_video_from_images(args.image_folder, args.output, args.fps)


if __name__ == "__main__":
    main()
