# Shared setup for N=32 MLARM examples (source from run.sh, do not execute directly).
set -euo pipefail

_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_EXAMPLES_DIR}/.." && pwd)"
TORCH_PKG="${REPO_ROOT}/torch_version"
export PYTHONPATH="${TORCH_PKG}:${REPO_ROOT}/model_zoo_N32:${PYTHONPATH:-}"

example_ensure_hf_assets() {
  if [[ -n "${VES2D_TRAINED_ROOT:-}" && -n "${VES2D_INNER_NEAR_ROOT:-}" ]]; then
    return 0
  fi
  if [[ "${VES2D_USE_HF_HUB:-1}" == "0" ]]; then
    echo "Set VES2D_TRAINED_ROOT and VES2D_INNER_NEAR_ROOT, or set VES2D_USE_HF_HUB=1." >&2
    exit 1
  fi
  echo "Downloading N=32 trained assets from Hugging Face (set VES2D_HF_REPO to override repo)..."
  local roots
  roots="$(cd "${TORCH_PKG}" && python - <<'PY'
from tools.model_hub import ensure_resolution_assets
layout = ensure_resolution_assets(32)
print(layout.trained_root)
print(layout.inner_near_root)
PY
)"
  if [[ -z "${roots}" ]]; then
    echo "Failed to resolve trained asset paths from Hugging Face." >&2
    exit 1
  fi
  VES2D_TRAINED_ROOT="$(echo "${roots}" | sed -n '1p')"
  VES2D_INNER_NEAR_ROOT="$(echo "${roots}" | sed -n '2p')"
  export VES2D_TRAINED_ROOT VES2D_INNER_NEAR_ROOT
}

example_ensure_hf_assets

example_prepare() {
  local name="$1"
  EXAMPLE_DIR="${_EXAMPLES_DIR}/${name}"
  if [[ ! -d "${EXAMPLE_DIR}" ]]; then
    echo "Unknown example: ${name}" >&2
    exit 1
  fi
  python "${_EXAMPLES_DIR}/make_initial_conditions.py" "${name}"
}

example_resolve_config() {
  local config_in="$1"
  local config_out="$2"
  python - "${config_in}" "${config_out}" "${EXAMPLE_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

config_in, config_out, example_dir = sys.argv[1:4]
example_dir = Path(example_dir)
cfg = json.loads(Path(config_in).read_text(encoding="utf-8"))
cfg["trained_root"] = os.environ["VES2D_TRAINED_ROOT"]
cfg["inner_near_root"] = os.environ["VES2D_INNER_NEAR_ROOT"]
if os.environ.get("VES2D_SHEAR_IC") and cfg.get("input", "").endswith("shear_N32.npy"):
    cfg["input"] = os.environ["VES2D_SHEAR_IC"]
if os.environ.get("VES2D_TG_IC") and cfg.get("input", "").endswith("48vesTG_N32.npy"):
    cfg["input"] = os.environ["VES2D_TG_IC"]
for key in ("input", "output_dir", "relaxed_shape", "log_file"):
    if key not in cfg or cfg[key] in (None, ""):
        continue
    p = Path(cfg[key])
    if not p.is_absolute():
        cfg[key] = str((example_dir / p).resolve())
Path(config_out).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PY
}

example_run_vesnet() {
  local resolved_config="$1"
  shift
  cd "${TORCH_PKG}"
  python entry_vesnet.py --config "${resolved_config}" --resolution 32 "$@"
}

# Optional: VES2D_POSTPROCESS=1 or pass --postprocess on run.sh
example_wants_postprocess() {
  [[ "${VES2D_POSTPROCESS:-0}" == 1 ]] || [[ "${EXAMPLE_POSTPROCESS:-0}" == 1 ]]
}

example_parse_args() {
  EXAMPLE_POSTPROCESS=0
  EXAMPLE_RUN_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --postprocess)
        EXAMPLE_POSTPROCESS=1
        shift
        ;;
      *)
        EXAMPLE_RUN_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

example_maybe_postprocess() {
  local resolved_config="$1"
  shift
  example_wants_postprocess || return 0
  echo "Post-processing: plotting frames and building video..."
  python "${_EXAMPLES_DIR}/plot_and_video.py" --config "${resolved_config}" "$@"
}
