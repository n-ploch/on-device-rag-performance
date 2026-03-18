"""Generation model client using llama-server HTTP API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from text generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    # Server-side timing (from llama-server timings)
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    predicted_per_token_ms: float = 0.0
    predicted_per_second: float = 0.0


class LlamaServerGenerator:
    """HTTP client for llama-server generation endpoint.

    Provides the same interface as the original Generator class but communicates
    with a running llama-server instance instead of loading the model in-process.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 300.0,
    ):
        """Initialize the generator client.

        Args:
            server_url: Base URL of the llama-server (e.g., "http://localhost:8002").
            timeout: HTTP request timeout in seconds (longer for generation).
        """
        self._server_url = server_url.rstrip("/")
        self._client = httpx.Client(base_url=self._server_url, timeout=timeout)
        logger.info("Initialized generator client for %s", self._server_url)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text completion for the given prompt.

        Args:
            prompt: The input prompt for generation.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling probability.
            stop: List of stop sequences.

        Returns:
            GenerationResult with text and token counts.

        Raises:
            RuntimeError: If the server request fails.
        """
        logger.debug(
            "Generating with max_tokens=%d, temp=%.2f",
            max_tokens,
            temperature,
        )

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        response = self._client.post("/v1/completions", json=payload)

        # llama-server returns 503 while loading the model; wait and retry
        retries = 0
        while response.status_code == 503 and retries < 10:
            wait = min(2**retries, 30)
            logger.info(
                "llama-server not ready (503 Loading model), retrying in %ds (attempt %d/10)...",
                wait,
                retries + 1,
            )
            time.sleep(wait)
            response = self._client.post("/v1/completions", json=payload)
            retries += 1

        if response.status_code != 200:
            raise RuntimeError(
                f"Generation request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        text = data["choices"][0]["text"]
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_ms=timings.get("prompt_ms", 0.0),
            predicted_ms=timings.get("predicted_ms", 0.0),
            predicted_per_token_ms=timings.get("predicted_per_token_ms", 0.0),
            predicted_per_second=timings.get("predicted_per_second", 0.0),
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate chat completion from messages.

        This uses the chat completions endpoint which applies the model's
        chat template automatically.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: List of stop sequences.

        Returns:
            GenerationResult with text and token counts.

        Raises:
            RuntimeError: If the server request fails.
        """
        logger.debug(
            "Chat generation with %d messages, max_tokens=%d",
            len(messages),
            max_tokens,
        )

        payload = {
            "model": "generation",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        response = self._client.post("/v1/chat/completions", json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Chat generation request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_ms=timings.get("prompt_ms", 0.0),
            predicted_ms=timings.get("predicted_ms", 0.0),
            predicted_per_token_ms=timings.get("predicted_per_token_ms", 0.0),
            predicted_per_second=timings.get("predicted_per_second", 0.0),
        )

    @property
    def server_url(self) -> str:
        """Return the server URL."""
        return self._server_url

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self._client.get("/health")
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
        except httpx.RequestError:
            pass
        return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._client.close()
        except Exception:
            pass
