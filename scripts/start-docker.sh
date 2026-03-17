#!/usr/bin/env bash
# start-docker.sh — start the orchestrator API + frontend via Docker Compose
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install Docker Desktop and re-run." >&2
  exit 1
fi

if ! docker compose version &>/dev/null; then
  echo "ERROR: 'docker compose' (v2) not found. Update Docker Desktop." >&2
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: .env file not found." >&2
  echo "       Run ./scripts/setup.sh first, then edit .env with your credentials." >&2
  exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "==> Starting RAGrig (orchestrator API + frontend)"
echo "    Orchestrator API: http://localhost:8080"
echo "    Frontend:         http://localhost:3003"
echo ""
echo "    Press Ctrl-C to stop."
echo ""

cd "$ROOT"
exec docker compose up --build "$@"
