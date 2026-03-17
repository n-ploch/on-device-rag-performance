.PHONY: setup test worker orchestrator ui eval clean

# Default config path for 'make eval'
CONFIG ?= config/sample_config.yaml

VENV := .rag
ACTIVATE := source $(VENV)/bin/activate

# ── Bootstrap ─────────────────────────────────────────────────────────────────

setup:
	@bash scripts/setup.sh

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	@$(ACTIVATE) && pytest tests/unit -v

# ── Run services ──────────────────────────────────────────────────────────────

worker:
	@bash scripts/start-worker.sh

orchestrator:
	@$(ACTIVATE) && rag-api

ui:
	@bash scripts/start-ui.sh

# ── Run evaluation (CLI) ──────────────────────────────────────────────────────

eval:
	@$(ACTIVATE) && rag-orchestrator --config $(CONFIG)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
