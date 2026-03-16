#!/usr/bin/env bash
# start-ui.sh — launch the orchestrator API + Vite dev server locally
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Activate the project venv
source "$ROOT/.rag/bin/activate"

# Install any new orchestrator deps (idempotent)
pip install -e "$ROOT/orchestrator/" -q

# Start the FastAPI server in the background
uvicorn orchestrator.api:app --port 8080 &
API_PID=$!

# Ensure the background process is killed when this script exits
trap 'echo "Stopping API server (PID $API_PID)…"; kill "$API_PID" 2>/dev/null' EXIT

echo "Orchestrator API running at http://localhost:8080"
echo "Starting Vite dev server…"

cd "$ROOT/frontend"
npm install -q
npm run dev
