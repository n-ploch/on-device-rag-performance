#!/usr/bin/env bash
# start-worker.sh — start the RAGrig worker on the edge device (bare-metal)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.rag"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: Virtual environment not found at .rag/" >&2
  echo "       Run ./scripts/setup.sh first." >&2
  exit 1
fi
source "$VENV/bin/activate"

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# ── Parse arguments (overrides .env) ──────────────────────────────────────────
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
export LOG_LEVEL

# ── Required environment variable checks ─────────────────────────────────────
MISSING=()
[[ -z "${LOCAL_MODELS_DIR:-}" ]]      && MISSING+=("LOCAL_MODELS_DIR")
[[ -z "${LOCAL_COLLECTIONS_DIR:-}" ]] && MISSING+=("LOCAL_COLLECTIONS_DIR")
[[ -z "${LOCAL_DATASETS_DIR:-}" ]]    && MISSING+=("LOCAL_DATASETS_DIR")

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "ERROR: The following required environment variables are not set:" >&2
  for var in "${MISSING[@]}"; do
    echo "  - $var" >&2
  done
  echo "" >&2
  echo "Edit .env and set these paths to your local models, collections, and datasets." >&2
  exit 1
fi

# ── Validate paths exist ──────────────────────────────────────────────────────
for VAR in LOCAL_MODELS_DIR LOCAL_COLLECTIONS_DIR LOCAL_DATASETS_DIR; do
  DIR="${!VAR}"
  if [[ ! -d "$DIR" ]]; then
    echo "WARNING: $VAR=$DIR does not exist. Creating it now..." >&2
    mkdir -p "$DIR"
  fi
done

HOST="${WORKER_HOST:-0.0.0.0}"
PORT="${WORKER_PORT:-8000}"

echo "==> Starting RAGrig Worker"
echo "    Models:      $LOCAL_MODELS_DIR"
echo "    Collections: $LOCAL_COLLECTIONS_DIR"
echo "    Datasets:    $LOCAL_DATASETS_DIR"
echo "    Listening:   http://$HOST:$PORT"
echo "    Log level:   $LOG_LEVEL"
echo ""

cd "$ROOT"
exec rag-worker --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
