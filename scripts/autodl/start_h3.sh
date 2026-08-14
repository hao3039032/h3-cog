#!/usr/bin/env bash
set -Eeuo pipefail

H3_CODE_DIR="${H3_CODE_DIR:-/root/h3-cog}"
export COMFY_ROOT="${COMFY_ROOT:-/root/ComfyUI}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/root/autodl-tmp/weights}"
export MINIMAX_H3_LICENSE_ACCEPTED="${MINIMAX_H3_LICENSE_ACCEPTED:-1}"
export H3_PARALLEL_MODE="${H3_PARALLEL_MODE:-single}"
export H3_BAKED_WEIGHTS_VERIFIED="${H3_BAKED_WEIGHTS_VERIFIED:-0}"
export PORT="${PORT:-6006}"

cd "$H3_CODE_DIR"
log_dir="$(dirname "$WEIGHTS_DIR")/logs"
mkdir -p "$log_dir"
: >"$log_dir/h3-supervisor.log"

setsid /root/miniconda3/bin/python -u app.py \
  >>"$log_dir/h3-supervisor.log" 2>&1 &
app_pid=$!

terminate() {
  kill -TERM -- "-$app_pid" 2>/dev/null || true
}
trap terminate EXIT TERM INT

set +e
wait "$app_pid"
status=$?
set -e
terminate
wait "$app_pid" 2>/dev/null || true
exit "$status"
