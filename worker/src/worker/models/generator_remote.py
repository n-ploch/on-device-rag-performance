"""Generation client for remote OpenAI-compatible APIs.

Supports any API that implements the chat/completions endpoint (OpenRouter,
Groq, Together AI, etc.). Uses SSE streaming to measure real TTFT.

Features:
- Token bucket rate limiting (requests/second)
- Exponential backoff on transient errors (429, 500, 502, 503, 504)
- Fail-fast API key validation at construction time
- Streaming TTFT measurement via SSE parsing
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field

import httpx

from worker.models.generator_http import GenerationResult

logger = logging.getLogger(__name__)

_BACKOFF_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5


@dataclass
class TokenBucketRateLimiter:
    """Simple token bucket rate limiter (thread-unsafe; single-threaded worker use).

    Args:
        rps: Allowed requests per second. Bucket capacity is 1 token (no burst).
    """

    rps: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = 1.0  # start with a full token
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(1.0, self._tokens + elapsed * self.rps)
        self._last_refill = now

        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.rps
            logger.debug("Rate limiter: sleeping %.3fs to stay within %.2f rps", wait, self.rps)
            time.sleep(wait)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


class RemoteGenerator:
    """Generation client for remote OpenAI-compatible chat/completions API.

    Uses SSE streaming to measure time-to-first-token (TTFT). Results are
    returned as a GenerationResult compatible with LlamaServerGenerator.

    Mapping to GenerationResult fields:
      prompt_ms           = wall-clock time from request send to first content delta (TTFT)
      predicted_ms        = wall-clock time from first delta to [DONE]
      predicted_per_token_ms = 0 (not available without server internals)
      predicted_per_second   = completion_tokens / (predicted_ms / 1000), or 0
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        timeout: float = 300.0,
    ) -> None:
        """Initialize the remote generator.

        Args:
            base_url: Base URL of the OpenAI-compatible API.
            model: Model identifier to pass in each request.
            api_key: Bearer token for Authorization header.
            extra_headers: Additional headers merged into every request.
            rate_limiter: Optional token bucket rate limiter.
            timeout: HTTP request timeout in seconds.

        Raises:
            RuntimeError: If api_key is empty.
        """
        if not api_key:
            raise RuntimeError("api_key must not be empty")

        self._model = model
        self._rate_limiter = rate_limiter

        merged_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            merged_headers.update(extra_headers)

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=merged_headers,
            timeout=timeout,
        )
        logger.info("Initialized RemoteGenerator: base_url=%s, model=%s", base_url, model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a completion for the given prompt via chat/completions.

        The prompt is sent as a single user message. SSE streaming is used to
        capture TTFT. Exponential backoff retries on transient HTTP errors.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Optional stop sequences.

        Returns:
            GenerationResult with text, token counts, and timing measurements.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._stream_request(payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _BACKOFF_STATUSES or attempt == _MAX_RETRIES:
                    raise
                delay = min(2**attempt + random.uniform(0, 1), 60.0)
                logger.debug(
                    "Backoff attempt %d/%d: sleeping %.2fs after HTTP %d",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc.response.status_code,
                )
                time.sleep(delay)

        # Unreachable — loop above always raises on last attempt
        raise RuntimeError("Max retries exceeded")  # pragma: no cover

    def _stream_request(self, payload: dict) -> GenerationResult:
        """Execute one streaming request and parse SSE chunks.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
        """
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        request_start = time.perf_counter()
        first_token_time: float | None = None
        last_chunk_time: float = request_start

        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    last_chunk_time = time.perf_counter()
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON SSE line: %s", data_str[:120])
                    continue

                # Extract content delta
                choices = chunk.get("choices") or []
                if choices:
                    delta_content = choices[0].get("delta", {}).get("content") or ""
                    if delta_content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        text_parts.append(delta_content)

                # Usage may appear in the final chunk (stream_options.include_usage)
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        # Timing calculations
        ttft_ms: float = 0.0
        predicted_ms: float = 0.0

        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start) * 1000
            predicted_ms = max(0.0, (last_chunk_time - first_token_time) * 1000)
        else:
            # No content delta received — use total elapsed as TTFT proxy
            ttft_ms = (last_chunk_time - request_start) * 1000

        predicted_per_second = (
            completion_tokens / (predicted_ms / 1000)
            if predicted_ms > 0 and completion_tokens > 0
            else 0.0
        )

        logger.debug(
            "Remote generation complete: ttft=%.1fms, predicted=%.1fms, "
            "tokens=%d/%d, tps=%.1f",
            ttft_ms,
            predicted_ms,
            prompt_tokens,
            completion_tokens,
            predicted_per_second,
        )

        return GenerationResult(
            text="".join(text_parts).strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_ms=ttft_ms,           # TTFT: request_sent → first token
            predicted_ms=predicted_ms,   # generation tail: first token → [DONE]
            predicted_per_token_ms=0.0,  # not decomposed for remote
            predicted_per_second=predicted_per_second,
        )

    @property
    def model(self) -> str:
        """Return the model identifier used for requests."""
        return self._model

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
