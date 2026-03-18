#!/usr/bin/env bash
# start-ui.sh — launch the orchestrator API + Vite dev server locally
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────────
LOG_LEVEL="info"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-level|-l) LOG_LEVEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$LOG_LEVEL" in
  debug|info|warning|error|critical) ;;
  *) echo "ERROR: --log-level must be one of: debug info warning error critical" >&2; exit 1 ;;
esac

# Activate the project venv
source "$ROOT/.rag/bin/activate"

# Install any new orchestrator deps (idempotent)
pip install -e "$ROOT/orchestrator/" -q

# Start the FastAPI server in the background
rag-api --log-level "$LOG_LEVEL" &
API_PID=$!

# Ensure the background process is killed when this script exits
trap 'echo "Stopping API server (PID $API_PID)…"; kill "$API_PID" 2>/dev/null' EXIT

echo "Orchestrator API running at http://localhost:8080"
echo "Starting Vite dev server…"

cd "$ROOT/frontend"
npm install -q
npm run dev
