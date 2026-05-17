#!/usr/bin/env bash
# Two vesicles in shear flow; initial state from shear_N32.npy (N=32).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

SHEAR_IC="${VES2D_SHEAR_IC:-${SCRIPT_DIR}/shear_N32.npy}"
if [[ ! -f "${SHEAR_IC}" ]]; then
  echo "Missing ${SHEAR_IC}" >&2
  echo "Copy shear_N32.npy into ${SCRIPT_DIR} or set VES2D_SHEAR_IC." >&2
  exit 1
fi
export VES2D_SHEAR_IC="${SHEAR_IC}"

EXAMPLE_DIR="${SCRIPT_DIR}"
RESOLVED="$(mktemp "${TMPDIR:-/tmp}/ves2d_ex2_config.XXXXXX.json")"
trap 'rm -f "${RESOLVED}"' EXIT
example_resolve_config "${SCRIPT_DIR}/config.json" "${RESOLVED}"
example_run_vesnet "${RESOLVED}" "$@"
