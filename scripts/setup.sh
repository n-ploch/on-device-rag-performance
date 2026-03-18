#!/usr/bin/env bash
# setup.sh — bootstrap the RAGrig development environment
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.rag"

echo "==> Setting up RAGrig environment"

# ── Python version check ─────────────────────────────────────────────────────
PYTHON=$(command -v python3.13 2>/dev/null || command -v python3 2>/dev/null || true)
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3.13 not found. Install it and re-run this script." >&2
  exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VERSION" < "3.13" ]]; then
  echo "ERROR: Python 3.13+ required (found $PY_VERSION)." >&2
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtual environment at .rag/"
  "$PYTHON" -m venv "$VENV"
fi

source "$VENV/bin/activate"
echo "==> Using $(python --version) from $VENV"

# ── Upgrade pip silently ──────────────────────────────────────────────────────
pip install --upgrade pip -q

# ── Install packages in editable mode ────────────────────────────────────────
echo "==> Installing shared, worker, orchestrator, analysis (editable)"
pip install -e "$ROOT/shared" -q
pip install -e "$ROOT/worker" -q
pip install -e "$ROOT/orchestrator" -q
pip install -e "$ROOT/analysis[notebooks]" -q

# ── Install test dependencies ─────────────────────────────────────────────────
echo "==> Installing test dependencies"
pip install pytest pytest-asyncio -q

# ── Copy .env if not present ──────────────────────────────────────────────────
if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo ""
  echo "  Created .env from .env.example."
  echo "  Edit .env and set at minimum:"
  echo "    LOCAL_MODELS_DIR, LOCAL_COLLECTIONS_DIR, LOCAL_DATASETS_DIR"
  echo "    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL"
  echo "  WORKER_URL defaults to http://localhost:8000 (change if worker runs elsewhere)."
  echo ""
else
  echo "==> .env already exists — skipping copy"
fi

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source .rag/bin/activate"
echo ""
echo "Run tests with:  make test"
echo "Start worker:    ./scripts/start-worker.sh "
echo "Start UI:        ./scripts/start-ui.sh"
echo "Start Docker:    ./scripts/start-docker.sh"
