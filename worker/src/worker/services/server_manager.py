"""Manages llama-server processes for embedding and generation."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default ports for embedding and generation servers
DEFAULT_EMBEDDING_PORT = 8001
DEFAULT_GENERATION_PORT = 8002

# Health check configuration
HEALTH_CHECK_INTERVAL_SECONDS = 0.5
HEALTH_CHECK_TIMEOUT_SECONDS = 120


@dataclass
class ServerConfig:
    """Configuration for a llama-server instance."""

    model_path: Path
    port: int
    n_ctx: int = 2048
    n_gpu_layers: int = -1  # -1 = auto/all layers
    embedding: bool = False
    pooling: str = "mean"  # mean, cls, last
    parallel_slots: int = 1
    metrics: bool = True
    host: str = "127.0.0.1"
    n_threads: int | None = None  # CPU threads (-t); None = llama-server default
    n_batch: int | None = None  # Logical batch size (-b); None = llama-server default
    flash_attn: bool = False  # Enable flash attention (-fa)
    tensor_split: str | None = None  # GPU tensor split (-ts); comma-separated fractions for multi-GPU
    no_kv_offload: bool = False  # Disable KV cache GPU offload (-nkvo); useful for Metal with low RAM


@dataclass
class ServerMetrics:
    """Metrics scraped from llama-server /metrics endpoint."""

    prompt_tokens_total: int = 0
    tokens_predicted_total: int = 0
    prompt_seconds_total: float = 0.0
    tokens_predicted_seconds_total: float = 0.0
    n_decode_total: int = 0
    requests_processing: int = 0


@dataclass
class ServerProcess:
    """Wrapper around a running llama-server process."""

    process: subprocess.Popen
    config: ServerConfig
    url: str = field(init=False)

    def __post_init__(self):
        self.url = f"http://{self.config.host}:{self.config.port}"


class LlamaServerManager:
    """Manages llama-server processes for embedding and generation.

    Handles starting, stopping, health checking, and metrics scraping
    for llama-server instances running as subprocesses.
    """

    def __init__(
        self,
        llama_server_path: str | None = None,
        embedding_port: int = DEFAULT_EMBEDDING_PORT,
        generation_port: int = DEFAULT_GENERATION_PORT,
    ):
        """Initialize the server manager.

        Args:
            llama_server_path: Path to llama-server executable.
                              If None, uses 'llama-server' from PATH.
            embedding_port: Port for embedding server.
            generation_port: Port for generation server.
        """
        self._llama_server_path = llama_server_path or "llama-server"
        self._embedding_port = embedding_port
        self._generation_port = generation_port

        self._embedding_server: ServerProcess | None = None
        self._generation_server: ServerProcess | None = None
        self._http_client = httpx.AsyncClient(timeout=30.0)

    @property
    def embedding_url(self) -> str | None:
        """URL of the running embedding server, or None if not running."""
        return self._embedding_server.url if self._embedding_server else None

    @property
    def generation_url(self) -> str | None:
        """URL of the running generation server, or None if not running."""
        return self._generation_server.url if self._generation_server else None

    @property
    def is_embedding_running(self) -> bool:
        """Check if embedding server is running."""
        return (
            self._embedding_server is not None
            and self._embedding_server.process.poll() is None
        )

    @property
    def is_generation_running(self) -> bool:
        """Check if generation server is running."""
        return (
            self._generation_server is not None
            and self._generation_server.process.poll() is None
        )

    def _build_command(self, config: ServerConfig) -> list[str]:
        """Build the llama-server command line arguments."""
        cmd = [
            self._llama_server_path,
            "-m", str(config.model_path),
            "--host", config.host,
            "--port", str(config.port),
            "-c", str(config.n_ctx),
            "-ngl", str(config.n_gpu_layers),
        ]

        if config.n_threads is not None:
            cmd.extend(["-t", str(config.n_threads)])

        if config.n_batch is not None:
            cmd.extend(["-b", str(config.n_batch)])

        if config.embedding:
            cmd.append("--embedding")
            cmd.extend(["--pooling", config.pooling])

        if config.parallel_slots > 1:
            cmd.extend(["-np", str(config.parallel_slots)])
            cmd.append("-cb")  # continuous batching is only meaningful with multiple slots

        if config.metrics:
            cmd.append("--metrics")

        if config.flash_attn:
            cmd.append("-fa")

        if config.tensor_split and "," in config.tensor_split:
            cmd.extend(["-ts", config.tensor_split])

        if config.no_kv_offload:
            cmd.append("-nkvo")

        return cmd

    async def _wait_for_health(
        self,
        url: str,
        timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    ) -> bool:
        """Wait for server to become healthy.

        Args:
            url: Server base URL.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if server became healthy, False if timeout.
        """
        health_url = f"{url}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                response = await self._http_client.get(health_url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        logger.info("Server at %s is healthy", url)
                        return True
            except httpx.RequestError:
                pass  # Server not ready yet

            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

        logger.error("Server at %s failed to become healthy within %ds", url, timeout)
        return False

    async def start_embedding_server(
        self,
        model_path: Path,
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
        pooling: str = "mean",
        n_threads: int | None = None,
        n_batch: int | None = None,
        tensor_split: str | None = None,
    ) -> bool:
        """Start the embedding server.

        Args:
            model_path: Path to the GGUF embedding model.
            n_ctx: Context size (default 512 for embeddings).
            n_gpu_layers: GPU layers to offload (-1 = all).
            pooling: Pooling strategy (mean, cls, last).
            n_threads: CPU threads (-t); None = llama-server default.
            n_batch: Logical batch size (-b); None = llama-server default.
            tensor_split: GPU tensor split fractions for multi-GPU (-ts).

        Returns:
            True if server started successfully, False otherwise.
        """
        # Stop existing embedding server if running
        if self.is_embedding_running:
            logger.info("Stopping existing embedding server")
            await self.stop_embedding_server()

        config = ServerConfig(
            model_path=model_path,
            port=self._embedding_port,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,
            pooling=pooling,
            parallel_slots=1,
            metrics=True,
            n_threads=n_threads,
            n_batch=n_batch,
            # flash_attn and no_kv_offload are not forwarded: encoder-only embedding
            # models (e.g. E5/BERT) don't use causal attention or a KV cache, so
            # -fa may cause a startup failure and -nkvo is a no-op.
            tensor_split=tensor_split,
        )

        cmd = self._build_command(config)
        logger.info("Starting embedding server: %s", " ".join(cmd))

        try:
            # Start process with output redirected to prevent blocking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            self._embedding_server = ServerProcess(process=process, config=config)

            # Wait for server to become healthy
            if await self._wait_for_health(self._embedding_server.url):
                logger.info(
                    "Embedding server started on port %d (PID %d)",
                    self._embedding_port,
                    process.pid,
                )
                return True
            else:
                # Server failed to start, clean up
                await self.stop_embedding_server()
                return False

        except Exception as e:
            logger.error("Failed to start embedding server: %s", e)
            return False

    async def start_generation_server(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        parallel_slots: int = 4,
        n_threads: int | None = None,
        n_batch: int | None = None,
        flash_attn: bool = False,
        tensor_split: str | None = None,
        no_kv_offload: bool = False,
    ) -> bool:
        """Start the generation server.

        Args:
            model_path: Path to the GGUF generation model.
            n_ctx: Context size (default 2048 for RAG).
            n_gpu_layers: GPU layers to offload (-1 = all).
            parallel_slots: Number of parallel request slots.
            n_threads: CPU threads (-t); None = llama-server default.
            n_batch: Logical batch size (-b); None = llama-server default.
            flash_attn: Enable flash attention (-fa).
            tensor_split: GPU tensor split fractions for multi-GPU (-ts).
            no_kv_offload: Disable KV cache GPU offload (-nkvo).

        Returns:
            True if server started successfully, False otherwise.
        """
        # Stop existing generation server if running
        if self.is_generation_running:
            logger.info("Stopping existing generation server")
            await self.stop_generation_server()

        config = ServerConfig(
            model_path=model_path,
            port=self._generation_port,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=False,
            parallel_slots=parallel_slots,
            metrics=True,
            n_threads=n_threads,
            n_batch=n_batch,
            flash_attn=flash_attn,
            tensor_split=tensor_split,
            no_kv_offload=no_kv_offload,
        )

        cmd = self._build_command(config)
        logger.info("Starting generation server: %s", " ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            self._generation_server = ServerProcess(process=process, config=config)

            # Wait for server to become healthy
            if await self._wait_for_health(self._generation_server.url):
                logger.info(
                    "Generation server started on port %d (PID %d)",
                    self._generation_port,
                    process.pid,
                )
                return True
            else:
                await self.stop_generation_server()
                return False

        except Exception as e:
            logger.error("Failed to start generation server: %s", e)
            return False

    async def stop_embedding_server(self) -> None:
        """Stop the embedding server if running."""
        if self._embedding_server is not None:
            await self._stop_process(self._embedding_server)
            self._embedding_server = None

    async def stop_generation_server(self) -> None:
        """Stop the generation server if running."""
        if self._generation_server is not None:
            await self._stop_process(self._generation_server)
            self._generation_server = None

    async def stop_all(self) -> None:
        """Stop all running servers."""
        await self.stop_embedding_server()
        await self.stop_generation_server()

    async def _stop_process(self, server: ServerProcess) -> None:
        """Stop a server process gracefully."""
        process = server.process

        if process.poll() is not None:
            logger.debug("Process already terminated")
            return

        logger.info("Stopping server on port %d (PID %d)", server.config.port, process.pid)

        try:
            # Send SIGTERM for graceful shutdown
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.debug("Server terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("Server did not terminate gracefully, force killing")
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)

        except Exception as e:
            logger.error("Error stopping server: %s", e)

    async def health_check(self, server_type: str = "both") -> dict[str, bool]:
        """Check health of servers.

        Args:
            server_type: "embedding", "generation", or "both".

        Returns:
            Dict with health status for requested servers.
        """
        result = {}

        if server_type in ("embedding", "both") and self._embedding_server:
            try:
                response = await self._http_client.get(
                    f"{self._embedding_server.url}/health"
                )
                result["embedding"] = (
                    response.status_code == 200
                    and response.json().get("status") == "ok"
                )
            except httpx.RequestError:
                result["embedding"] = False

        if server_type in ("generation", "both") and self._generation_server:
            try:
                response = await self._http_client.get(
                    f"{self._generation_server.url}/health"
                )
                result["generation"] = (
                    response.status_code == 200
                    and response.json().get("status") == "ok"
                )
            except httpx.RequestError:
                result["generation"] = False

        return result

    async def scrape_metrics(self, server_type: str = "both") -> dict[str, ServerMetrics]:
        """Scrape Prometheus metrics from servers.

        Args:
            server_type: "embedding", "generation", or "both".

        Returns:
            Dict with ServerMetrics for requested servers.
        """
        result = {}

        if server_type in ("embedding", "both") and self._embedding_server:
            result["embedding"] = await self._scrape_server_metrics(
                self._embedding_server.url
            )

        if server_type in ("generation", "both") and self._generation_server:
            result["generation"] = await self._scrape_server_metrics(
                self._generation_server.url
            )

        return result

    async def _scrape_server_metrics(self, url: str) -> ServerMetrics:
        """Scrape metrics from a single server."""
        metrics = ServerMetrics()

        try:
            response = await self._http_client.get(f"{url}/metrics")
            if response.status_code != 200:
                return metrics

            text = response.text
            metrics = self._parse_prometheus_metrics(text)

        except httpx.RequestError as e:
            logger.warning("Failed to scrape metrics from %s: %s", url, e)

        return metrics

    def _parse_prometheus_metrics(self, text: str) -> ServerMetrics:
        """Parse Prometheus format metrics text."""
        metrics = ServerMetrics()

        patterns = {
            "prompt_tokens_total": r"llamacpp:prompt_tokens_total\s+(\d+)",
            "tokens_predicted_total": r"llamacpp:tokens_predicted_total\s+(\d+)",
            "prompt_seconds_total": r"llamacpp:prompt_seconds_total\s+([\d.]+)",
            "tokens_predicted_seconds_total": r"llamacpp:tokens_predicted_seconds_total\s+([\d.]+)",
            "n_decode_total": r"llamacpp:n_decode_total\s+(\d+)",
            "requests_processing": r"llamacpp:requests_processing\s+(\d+)",
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                if "." in value:
                    setattr(metrics, field_name, float(value))
                else:
                    setattr(metrics, field_name, int(value))

        return metrics

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_all()
        await self._http_client.aclose()
