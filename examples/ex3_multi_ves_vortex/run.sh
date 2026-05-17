#!/usr/bin/env bash
# Eight vesicles from 48vesTG_N32.npy in vortex flow (N=32).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

example_parse_args "$@"
TG_IC="${VES2D_TG_IC:-${SCRIPT_DIR}/48vesTG_N32.npy}"
if [[ ! -f "${TG_IC}" ]]; then
  echo "Missing ${TG_IC}" >&2
  echo "Copy 48vesTG_N32.npy into ${SCRIPT_DIR} or set VES2D_TG_IC." >&2
  exit 1
fi
export VES2D_TG_IC="${TG_IC}"

EXAMPLE_DIR="${SCRIPT_DIR}"
RESOLVED="$(mktemp "${TMPDIR:-/tmp}/ves2d_ex3_config.XXXXXX.json")"
trap 'rm -f "${RESOLVED}"' EXIT
example_resolve_config "${SCRIPT_DIR}/config.json" "${RESOLVED}"
example_run_vesnet "${RESOLVED}" "${EXAMPLE_RUN_ARGS[@]}"
example_maybe_postprocess "${RESOLVED}"
