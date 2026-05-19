"""Download and resolve Ves2Dpy trained weights from Hugging Face Hub."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "trained_assets" / "manifest.json"

DEFAULT_REPO_ID = "ves2d/ves2d-trained"
DEFAULT_REVISION = "main"


@dataclass(frozen=True)
class AssetLayout:
    """Local paths after download (or when using an existing trained tree)."""

    resolution: int
    trained_root: Path
    inner_near_root: Path | None
    weights: dict[str, Path]

    def weight_path(self, key: str) -> str:
        return str(self.weights[key])


def _require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic model download. "
            "Install with: pip install huggingface_hub"
        ) from exc
    return hf_hub_download


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def default_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "VES2D_CACHE_DIR",
            Path.home() / ".cache" / "ves2d",
        )
    )


def default_repo_id() -> str:
    return os.environ.get("VES2D_HF_REPO", load_manifest().get("repo_id", DEFAULT_REPO_ID))


def default_revision() -> str:
    return os.environ.get("VES2D_HF_REVISION", load_manifest().get("revision", DEFAULT_REVISION))


def _bundle(resolution: int) -> dict[str, Any]:
    key = str(resolution)
    manifest = load_manifest()
    bundles = manifest["bundles"]
    if key not in bundles:
        raise ValueError(f"No asset bundle for resolution {resolution}; expected 32 or 128.")
    return bundles[key]


def _hub_files_for_resolution(resolution: int) -> list[str]:
    """Relative paths inside the HF repo for one resolution."""
    bundle = _bundle(resolution)
    files: list[str] = []
    prefix = bundle["trained_root_subdir"]
    for rel in bundle.get("norms", {}).values():
        files.append(f"{prefix}/{rel}")
    for rel in bundle.get("weights", {}).values():
        files.append(f"{prefix}/{rel}")
    inner = bundle.get("inner_near_root_subdir")
    if inner:
        for rel in bundle.get("inner_near_norms", {}).values():
            files.append(f"{inner}/{rel}")
    return files


def cache_roots(resolution: int, cache_dir: Path | None = None) -> tuple[Path, Path | None]:
    cache = cache_dir or default_cache_dir()
    bundle = _bundle(resolution)
    trained_root = cache / bundle["trained_root_subdir"]
    inner_sub = bundle.get("inner_near_root_subdir")
    inner_root = (cache / inner_sub) if inner_sub else None
    return trained_root, inner_root


def layout_from_cache(resolution: int, cache_dir: Path | None = None) -> AssetLayout:
    trained_root, inner_near_root = cache_roots(resolution, cache_dir)
    bundle = _bundle(resolution)
    weights = {
        key: trained_root / rel for key, rel in bundle["weights"].items()
    }
    return AssetLayout(
        resolution=resolution,
        trained_root=trained_root,
        inner_near_root=inner_near_root,
        weights=weights,
    )


def layout_from_trained_root(
    resolution: int,
    trained_root: str | Path,
    inner_near_root: str | Path | None = None,
) -> AssetLayout:
    """Build weight paths from an on-disk trained/ tree (cluster or local checkout)."""
    trained_root = Path(trained_root)
    bundle = _bundle(resolution)
    weights = {key: trained_root / rel for key, rel in bundle["weights"].items()}
    if resolution == 32 and inner_near_root is None:
        raise ValueError("inner_near_root is required for resolution 32")
    return AssetLayout(
        resolution=resolution,
        trained_root=trained_root,
        inner_near_root=Path(inner_near_root) if inner_near_root else None,
        weights=weights,
    )


def assets_present(layout: AssetLayout) -> bool:
    bundle = _bundle(layout.resolution)
    for rel in bundle.get("norms", {}).values():
        if not (layout.trained_root / rel).is_file():
            return False
    for path in layout.weights.values():
        if not path.is_file():
            return False
    if layout.inner_near_root is not None:
        for rel in bundle.get("inner_near_norms", {}).values():
            if not (layout.inner_near_root / rel).is_file():
                return False
    return True


def download_resolution_assets(
    resolution: int,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    force: bool = False,
) -> AssetLayout:
    """Download all manifest files for ``resolution`` into the local cache."""
    hf_hub_download = _require_hf_hub()
    repo_id = repo_id or default_repo_id()
    revision = revision or default_revision()
    cache = cache_dir or default_cache_dir()
    layout = layout_from_cache(resolution, cache)

    for hub_path in _hub_files_for_resolution(resolution):
        local_name = hub_path
        dest = cache / hub_path
        if dest.is_file() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=local_name,
            repo_type="model",
            revision=revision,
            local_dir=str(cache),
            local_dir_use_symlinks=False,
            token=token,
        )
        # hf_hub_download returns path under local_dir when local_dir is set
        if not dest.is_file() and Path(downloaded).is_file():
            Path(downloaded).replace(dest)

    if not assets_present(layout):
        missing = [str(p) for p in layout.weights.values() if not p.is_file()]
        raise FileNotFoundError(
            f"Download finished but assets are still missing for N={resolution}: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return layout


def ensure_resolution_assets(
    resolution: int,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> AssetLayout:
    """Return local layout, downloading from Hugging Face if needed."""
    layout = layout_from_cache(resolution, cache_dir)
    if force_download or not assets_present(layout):
        return download_resolution_assets(
            resolution,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            force=force_download,
        )
    return layout


def apply_layout_to_params(params: dict[str, Any], layout: AssetLayout) -> dict[str, Any]:
    """Inject trained_root, inner_near_root, and model_paths into a vesnet config dict."""
    out = dict(params)
    out["trained_root"] = str(layout.trained_root)
    if layout.inner_near_root is not None:
        out["inner_near_root"] = str(layout.inner_near_root)
    out["model_paths"] = {k: str(v) for k, v in layout.weights.items()}
    return out


def resolve_params_assets(
    params: dict[str, Any],
    *,
    prefer_hf: bool | None = None,
) -> dict[str, Any]:
    """
    Ensure trained assets are available.

    If ``prefer_hf`` is True (default when trained_root is unset or use_hf_hub is True),
    download or refresh from Hugging Face. Otherwise use explicit ``trained_root`` /
    ``inner_near_root`` and optional ``model_paths``.
    """
    resolution = int(params["resolution"])
    use_hf = prefer_hf
    if use_hf is None:
        use_hf = bool(params.get("use_hf_hub", True))
    explicit_root = params.get("trained_root")
    if explicit_root and not use_hf:
        layout = layout_from_trained_root(
            resolution,
            explicit_root,
            params.get("inner_near_root"),
        )
        out = dict(params)
        if "model_paths" not in out:
            out["model_paths"] = {k: str(v) for k, v in layout.weights.items()}
        return out

    if explicit_root and Path(explicit_root).is_dir() and assets_present(
        layout_from_trained_root(
            resolution, explicit_root, params.get("inner_near_root")
        )
    ):
        layout = layout_from_trained_root(
            resolution, explicit_root, params.get("inner_near_root")
        )
        return apply_layout_to_params(params, layout)

    layout = ensure_resolution_assets(
        resolution,
        repo_id=params.get("hf_repo"),
        revision=params.get("hf_revision"),
        cache_dir=Path(params["hf_cache_dir"]) if params.get("hf_cache_dir") else None,
        token=params.get("hf_token") or os.environ.get("HF_TOKEN"),
        force_download=bool(params.get("force_hf_download")),
    )
    return apply_layout_to_params(params, layout)
