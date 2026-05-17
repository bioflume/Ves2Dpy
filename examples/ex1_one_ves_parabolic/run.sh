#!/usr/bin/env bash
# One vesicle, off-center in parabolic channel flow (N=32).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_common.sh
source "${SCRIPT_DIR}/../_common.sh"

example_prepare "ex1_one_ves_parabolic"
RESOLVED="$(mktemp "${TMPDIR:-/tmp}/ves2d_ex1_config.XXXXXX.json")"
trap 'rm -f "${RESOLVED}"' EXIT
example_resolve_config "${SCRIPT_DIR}/config.json" "${RESOLVED}"
example_run_vesnet "${RESOLVED}" "$@"
