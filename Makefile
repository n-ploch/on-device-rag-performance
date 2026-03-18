.PHONY: setup test clean

VENV := .rag
ACTIVATE := source $(VENV)/bin/activate

setup:
	@bash scripts/setup.sh

test:
	@$(ACTIVATE) && pytest tests/unit -v

clean:
	find . -type d -name "__pycache__" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./.rag/*" -exec rm -rf {} + 2>/dev/null || true
