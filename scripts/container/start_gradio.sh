#!/usr/bin/env bash
set -Eeuo pipefail

export PORT="${PORT:-7860}"
export COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
export H3_PARALLEL_MODE="${H3_PARALLEL_MODE:-single}"
export H3_BAKED_WEIGHTS_VERIFIED="${H3_BAKED_WEIGHTS_VERIFIED:-0}"
python_bin="$(command -v "${PYTHON_BIN:-python}")"

mkdir -p "$WEIGHTS_DIR" "$COMFY_ROOT/input" "$COMFY_ROOT/output" /tmp/gradio

app_pid=

terminate() {
    if [[ -n "$app_pid" ]]; then
        kill -TERM -- "-$app_pid" 2>/dev/null || true
    fi
}

trap terminate EXIT TERM INT
setsid "$python_bin" -u /src/app.py &
app_pid=$!

set +e
wait "$app_pid"
status=$?
set -e
terminate
wait "$app_pid" 2>/dev/null || true
trap - EXIT TERM INT
exit "$status"
