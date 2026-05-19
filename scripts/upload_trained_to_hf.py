#!/usr/bin/env python3
"""
Stage trained Ves2Dpy assets from cluster paths and upload to Hugging Face Hub.

Run on a machine that has the trained .pth / .pt / .npy files (e.g. TACC login node).

Example (N=32):
  export HF_TOKEN=hf_...
  python scripts/upload_trained_to_hf.py \\
    --repo-id YOUR_ORG/ves2d-trained \\
    --resolution 32 \\
    --n32-trained-root /work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained \\
    --n32-inner-near-root /work/09452/alberto47/ls6/vesicle_nearF2024/trained_disth_nocoords/inner_downsample32 \\
    --n32-inner-near-weight /work/09452/alberto47/vista/Ves2Dpy/trained/2025ves_merged_disth_innerNearFourier.pth \\
    --n32-ten-self /work/09452/alberto47/ls6/vesicle_selften/save_models/Ves_2025Feb_downsample_selften_zerolevel_12blks_loss_0.01105_2242401_cuda2.pth

Example (N=128):
  python scripts/upload_trained_to_hf.py \\
    --repo-id YOUR_ORG/ves2d-trained \\
    --resolution 128 \\
    --n128-trained-root /work/09452/alberto47/vista/Ves2Dpy/trained

Then create the repo on huggingface.co (Model) if it does not exist, and re-run with --upload.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "trained_assets" / "manifest.json"


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def stage_resolution(
    resolution: int,
    staging: Path,
    *,
    trained_root: Path | None,
    inner_near_root: Path | None,
    extra_files: dict[str, Path],
) -> list[Path]:
    manifest = load_manifest()
    bundle = manifest["bundles"][str(resolution)]
    staged: list[Path] = []

    if trained_root is None:
        raise ValueError(f"--n{resolution}-trained-root is required for resolution {resolution}")

    hub_trained = staging / bundle["trained_root_subdir"]
    hub_trained.mkdir(parents=True, exist_ok=True)

    def copy_into_hub(rel: str, src: Path) -> None:
        dest = staging / bundle["trained_root_subdir"] / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            raise FileNotFoundError(f"Missing source file: {src}")
        shutil.copy2(src, dest)
        staged.append(dest)

    for rel in bundle.get("norms", {}).values():
        copy_into_hub(rel, trained_root / rel)

    for key, rel in bundle.get("weights", {}).items():
        if key in extra_files:
            copy_into_hub(rel, extra_files[key])
        else:
            copy_into_hub(rel, trained_root / rel)

    inner_sub = bundle.get("inner_near_root_subdir")
    if inner_sub and inner_near_root:
        hub_inner = staging / inner_sub
        hub_inner.mkdir(parents=True, exist_ok=True)
        for rel in bundle.get("inner_near_norms", {}).values():
            src = inner_near_root / rel
            dest = hub_inner / rel
            if not src.is_file():
                raise FileNotFoundError(f"Missing inner near norm: {src}")
            shutil.copy2(src, dest)
            staged.append(dest)

    return staged


def upload_folder(repo_id: str, folder: Path, revision: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message="Upload Ves2Dpy trained surrogates",
    )


def parse_args() -> argparse.Namespace:
    manifest = load_manifest()
    hints = manifest.get("cluster_upload_hints", {})

    p = argparse.ArgumentParser(description="Stage and upload Ves2Dpy trained assets to Hugging Face.")
    p.add_argument(
        "--repo-id",
        default=manifest.get("repo_id", "ves2d/ves2d-trained"),
        help="Hugging Face model repo id (org/name)",
    )
    p.add_argument("--revision", default=manifest.get("revision", "main"))
    p.add_argument(
        "--resolution",
        type=int,
        choices=(32, 128),
        required=True,
        help="Which bundle to stage/upload",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=REPO_ROOT / "trained_assets" / "_hf_staging",
        help="Local directory to assemble hub layout before upload",
    )
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload staging dir to Hugging Face (requires HF_TOKEN)",
    )
    p.add_argument("--private", action="store_true", help="Create/upload to a private model repo")
    p.add_argument("--dry-run", action="store_true", help="Only print planned copies")

    h32 = hints.get("32", {})
    h128 = hints.get("128", {})
    p.add_argument(
        "--n32-trained-root",
        type=Path,
        default=Path(h32["trained_root"]) if h32.get("trained_root") else None,
    )
    p.add_argument(
        "--n32-inner-near-root",
        type=Path,
        default=Path(h32["inner_near_root"]) if h32.get("inner_near_root") else None,
    )
    p.add_argument(
        "--n128-trained-root",
        type=Path,
        default=Path(h128["trained_root"]) if h128.get("trained_root") else None,
    )
    extra32 = h32.get("extra_files", {})
    p.add_argument(
        "--n32-inner-near-weight",
        type=Path,
        default=Path(extra32["2025ves_merged_disth_innerNearFourier.pth"])
        if extra32.get("2025ves_merged_disth_innerNearFourier.pth")
        else None,
    )
    p.add_argument(
        "--n32-ten-self",
        type=Path,
        default=Path(
            extra32[
                "ten_self/Ves_2025Feb_downsample_selften_zerolevel_12blks_loss_0.01105_2242401_cuda2.pth"
            ]
        )
        if extra32.get(
            "ten_self/Ves_2025Feb_downsample_selften_zerolevel_12blks_loss_0.01105_2242401_cuda2.pth"
        )
        else None,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    staging = args.staging_dir.resolve()
    if args.resolution == 32:
        extra: dict[str, Path] = {}
        if args.n32_inner_near_weight:
            extra["inner_near"] = args.n32_inner_near_weight
        if args.n32_ten_self:
            extra["ten_self"] = args.n32_ten_self
        if args.dry_run:
            print("Would stage N=32 from:", args.n32_trained_root, args.n32_inner_near_root, extra)
            return 0
        staged = stage_resolution(
            32,
            staging,
            trained_root=args.n32_trained_root,
            inner_near_root=args.n32_inner_near_root,
            extra_files=extra,
        )
    else:
        if args.dry_run:
            print("Would stage N=128 from:", args.n128_trained_root)
            return 0
        staged = stage_resolution(
            128,
            staging,
            trained_root=args.n128_trained_root,
            inner_near_root=None,
            extra_files={},
        )

    shutil.copy2(MANIFEST_PATH, staging / "manifest.json")
    print(f"Staged {len(staged)} files under {staging}")

    if not args.upload:
        print("Dry staging complete. Re-run with --upload to push to Hugging Face.")
        return 0

    upload_folder(args.repo_id, staging, args.revision, args.private)
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as exc:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from exc
