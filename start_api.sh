#!/usr/bin/env bash
# =====================================================================
#  indextts25-fastapi - start script (Linux / macOS)
#  Expected layout:   <index-tts>/indextts25-fastapi/start_api.sh
#  GPU=<n> PORT=<n> ./start_api.sh [extra run_api.py args...]
# =====================================================================
set -euo pipefail

WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_ROOT="$(dirname "$WRAPPER")"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"

cd "$INDEX_ROOT"
echo "[start_api] index-tts root: $INDEX_ROOT"
echo "[start_api] GPU=$GPU PORT=$PORT (BF16)"
exec uv run "$WRAPPER/run_api.py" --gpu "$GPU" --bf16 --host 0.0.0.0 --port "$PORT" "$@"
