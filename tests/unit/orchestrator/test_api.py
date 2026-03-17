"""Unit tests for the orchestrator FastAPI application (orchestrator/api.py)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from orchestrator.api import _AppState, app

# ---------------------------------------------------------------------------
# Minimal valid YAML fixtures
# ---------------------------------------------------------------------------

VALID_YAML = """\
dataset:
  id: scifact
  name: scifact
run_configs:
  - run_id: test_run
    retrieval:
      dataset_id: scifact
      model: intfloat/multilingual-e5-small
      quantization: fp16
      dimensions: 384
      k: 3
    generation:
      model: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
      quantization: q4_k_m
"""

LANGFUSE_YAML = """\
dataset:
  id: scifact
  name: scifact
observability:
  langfuse: true
run_configs:
  - run_id: test_run
    retrieval:
      dataset_id: scifact
      model: intfloat/multilingual-e5-small
      quantization: fp16
      dimensions: 384
      k: 3
    generation:
      model: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
      quantization: q4_k_m
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_state():
    """Isolate app state between tests."""
    app.state.data = _AppState()
    yield
    app.state.data = _AppState()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def client_with_config(client):
    """TestClient with a valid config already loaded."""
    client.post("/api/config/load", json={"content": VALID_YAML})
    return client


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_no_config(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["config_loaded"] is False
        assert data["run_id"] is None
        assert data["is_running"] is False

    def test_after_config_load(self, client_with_config):
        r = client_with_config.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["config_loaded"] is True
        assert data["run_id"] == "test_run"


# ---------------------------------------------------------------------------
# POST /api/config/load
# ---------------------------------------------------------------------------


class TestConfigLoad:
    def test_load_from_content_ok(self, client):
        r = client.post("/api/config/load", json={"content": VALID_YAML})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["yaml_text"] == VALID_YAML
        assert data["error"] is None

    def test_load_from_path_ok(self, client, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(VALID_YAML)
        r = client.post("/api/config/load", json={"path": str(cfg_file)})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_missing_path(self, client):
        r = client.post("/api/config/load", json={"path": "/nonexistent/path/config.yaml"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert "not found" in data["error"].lower()

    def test_invalid_yaml(self, client):
        r = client.post("/api/config/load", json={"content": "key: [unclosed"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert "yaml" in data["error"].lower()

    def test_schema_validation_error(self, client):
        r = client.post("/api/config/load", json={"content": "not_a_config: true\n"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert data["error"] is not None

    def test_no_path_or_content(self, client):
        r = client.post("/api/config/load", json={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/worker/url  +  POST /api/worker/url
# ---------------------------------------------------------------------------


class TestWorkerUrl:
    def test_get_returns_env_value(self, client, monkeypatch):
        monkeypatch.setenv("WORKER_URL", "http://myworker:9000")
        r = client.get("/api/worker/url")
        assert r.status_code == 200
        assert r.json()["url"] == "http://myworker:9000"

    def test_get_default(self, client, monkeypatch):
        monkeypatch.delenv("WORKER_URL", raising=False)
        r = client.get("/api/worker/url")
        assert r.status_code == 200
        assert r.json()["url"] == "http://localhost:8000"

    def test_set_updates_env(self, client, monkeypatch):
        monkeypatch.delenv("WORKER_URL", raising=False)
        r = client.post("/api/worker/url", json={"url": "http://newhost:8001/"})
        assert r.status_code == 200
        # Trailing slash is stripped by the endpoint
        assert r.json()["url"] == "http://newhost:8001"

    def test_set_reflected_in_get(self, client, monkeypatch):
        monkeypatch.delenv("WORKER_URL", raising=False)
        client.post("/api/worker/url", json={"url": "http://newhost:8001"})
        r = client.get("/api/worker/url")
        assert r.json()["url"] == "http://newhost:8001"


# ---------------------------------------------------------------------------
# GET /api/worker/check
# ---------------------------------------------------------------------------


class TestWorkerCheck:
    def _mock_healthy_client(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "backend": "cpu",
            "models_loaded": True,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    def test_worker_reachable(self, client, monkeypatch):
        monkeypatch.setenv("WORKER_URL", "http://worker:8000")
        with patch("orchestrator.api.httpx.AsyncClient", return_value=self._mock_healthy_client()):
            r = client.get("/api/worker/check")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["status"] == "healthy"
        assert data["backend"] == "cpu"
        assert data["models_loaded"] is True

    def test_worker_unreachable(self, client, monkeypatch):
        monkeypatch.setenv("WORKER_URL", "http://nowhere:9999")

        ctx = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("orchestrator.api.httpx.AsyncClient", return_value=ctx):
            r = client.get("/api/worker/check")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert "nowhere:9999" in data["error"]

    def test_worker_timeout(self, client, monkeypatch):
        monkeypatch.setenv("WORKER_URL", "http://slow:8000")

        ctx = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("orchestrator.api.httpx.AsyncClient", return_value=ctx):
            r = client.get("/api/worker/check")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "timed out" in r.json()["error"].lower()

    def test_worker_http_error(self, client, monkeypatch):
        monkeypatch.setenv("WORKER_URL", "http://worker:8000")

        mock_response = MagicMock()
        mock_response.status_code = 503
        ctx = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("bad", request=MagicMock(), response=mock_response)
        )
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("orchestrator.api.httpx.AsyncClient", return_value=ctx):
            r = client.get("/api/worker/check")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert "503" in data["error"]


# ---------------------------------------------------------------------------
# POST /api/run  +  POST /api/dry-run  (precondition checks only)
# ---------------------------------------------------------------------------


class TestRunPreconditions:
    def test_run_without_config(self, client):
        r = client.post("/api/run", json={})
        assert r.status_code == 422
        assert "configuration" in r.json()["detail"].lower()

    def test_dry_run_without_config(self, client):
        r = client.post("/api/dry-run", json={})
        assert r.status_code == 422

    def test_run_already_running(self, client_with_config):
        """409 when a run task is already active."""
        loop = asyncio.new_event_loop()
        never_done = loop.create_future()  # never resolves
        app.state.data.run_task = never_done  # type: ignore[assignment]
        r = client_with_config.post("/api/run", json={})
        assert r.status_code == 409
        assert "already running" in r.json()["detail"].lower()
        loop.close()

    def test_run_with_langfuse_missing_keys(self, client, monkeypatch):
        """422 when langfuse is enabled but credentials are absent."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        client.post("/api/config/load", json={"content": LANGFUSE_YAML})
        r = client.post("/api/run", json={})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "LANGFUSE_PUBLIC_KEY" in detail or "LANGFUSE_SECRET_KEY" in detail


# ---------------------------------------------------------------------------
# POST /api/stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_not_running(self, client):
        r = client.post("/api/stop", json={})
        assert r.status_code == 409
        assert "no evaluation" in r.json()["detail"].lower()

    def test_stop_while_running(self, client_with_config):
        loop = asyncio.new_event_loop()
        never_done = loop.create_future()
        app.state.data.run_task = never_done  # type: ignore[assignment]
        r = client_with_config.post("/api/stop", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert app.state.data.stop_event.is_set()
        loop.close()
