"""Unit tests for RemoteGenerator and TokenBucketRateLimiter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import httpx
import pytest

from worker.models.generator_remote import RemoteGenerator, TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_lines(*chunks: dict, done: bool = True) -> list[str]:
    """Build a list of SSE data lines for use in mock iter_lines responses."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    if done:
        lines.append("data: [DONE]")
    return lines


def _delta_chunk(content: str, model: str = "m") -> dict:
    """Build an SSE chunk containing a content delta."""
    return {"choices": [{"delta": {"content": content}}], "usage": None}


def _usage_chunk(prompt: int, completion: int) -> dict:
    """Build the final SSE chunk carrying usage statistics."""
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def _error_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "http://x/chat/completions"))


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    def test_first_acquire_does_not_sleep(self):
        """First acquire should consume the initial token without sleeping."""
        limiter = TokenBucketRateLimiter(rps=1.0)
        with patch("worker.models.generator_remote.time.sleep") as mock_sleep:
            limiter.acquire()
        mock_sleep.assert_not_called()

    def test_second_acquire_sleeps(self):
        """Second immediate acquire should sleep to stay within RPS limit."""
        limiter = TokenBucketRateLimiter(rps=1.0)
        limiter.acquire()  # consume the initial token

        with patch("worker.models.generator_remote.time.sleep") as mock_sleep:
            # Fake that no time has passed so no refill occurs
            with patch("worker.models.generator_remote.time.monotonic", return_value=limiter._last_refill):
                limiter.acquire()

        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration > 0

    def test_refill_allows_next_acquire_without_sleep(self):
        """After sufficient time passes, a new token is available without sleeping."""
        limiter = TokenBucketRateLimiter(rps=2.0)
        limiter.acquire()  # consume initial token

        with patch("worker.models.generator_remote.time.sleep") as mock_sleep:
            # Fake that 1 second has passed → 2 tokens refilled, capped at 1
            start = limiter._last_refill
            with patch("worker.models.generator_remote.time.monotonic", return_value=start + 1.0):
                limiter.acquire()

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# RemoteGenerator construction
# ---------------------------------------------------------------------------


class TestRemoteGeneratorInit:
    def test_empty_api_key_raises(self):
        """Passing an empty API key must raise RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="api_key must not be empty"):
            RemoteGenerator(base_url="http://api.example.com", model="m", api_key="")

    def test_authorization_header_set(self):
        """Authorization: Bearer <key> must appear in the client headers."""
        gen = RemoteGenerator(base_url="http://api.example.com/v1", model="m", api_key="sk-test")
        assert gen._client.headers.get("authorization") == "Bearer sk-test"
        gen.close()

    def test_extra_headers_merged(self):
        """Extra headers from config are forwarded to the HTTP client."""
        gen = RemoteGenerator(
            base_url="http://api.example.com/v1",
            model="m",
            api_key="sk-test",
            extra_headers={"X-Custom": "val"},
        )
        assert gen._client.headers.get("x-custom") == "val"
        gen.close()

    def test_model_property(self):
        gen = RemoteGenerator(base_url="http://api.example.com/v1", model="my-model", api_key="sk")
        assert gen.model == "my-model"
        gen.close()


# ---------------------------------------------------------------------------
# RemoteGenerator.generate() — happy path
# ---------------------------------------------------------------------------


class TestRemoteGeneratorGenerate:
    def _make_generator(self, **kwargs) -> RemoteGenerator:
        return RemoteGenerator(base_url="http://api.example.com/v1", model="m", api_key="sk", **kwargs)

    def _mock_stream(self, gen: RemoteGenerator, sse_lines: list[str], status: int = 200):
        """Patch the httpx client's stream() context manager with fake SSE lines."""
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.raise_for_status = MagicMock()  # 2xx: no-op
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_cm.__exit__ = MagicMock(return_value=False)
        gen._client.stream = MagicMock(return_value=mock_cm)
        return mock_response

    def test_text_assembled_from_deltas(self):
        """Text is the concatenation of all content deltas, stripped."""
        gen = self._make_generator()
        chunks = [
            _delta_chunk("Hello"),
            _delta_chunk(" world"),
            _usage_chunk(5, 2),
        ]
        self._mock_stream(gen, _make_sse_lines(*chunks))

        result = gen.generate(prompt="hi")

        assert result.text == "Hello world"
        gen.close()

    def test_token_counts_from_usage_chunk(self):
        """prompt_tokens and completion_tokens come from the usage SSE chunk."""
        gen = self._make_generator()
        chunks = [_delta_chunk("answer"), _usage_chunk(prompt=12, completion=3)]
        self._mock_stream(gen, _make_sse_lines(*chunks))

        result = gen.generate(prompt="question")

        assert result.prompt_tokens == 12
        assert result.completion_tokens == 3
        gen.close()

    def test_tokens_per_second_calculated(self):
        """predicted_per_second = completion_tokens / (predicted_ms / 1000)."""
        gen = self._make_generator()
        chunks = [_delta_chunk("a"), _usage_chunk(5, 10)]
        self._mock_stream(gen, _make_sse_lines(*chunks))

        # Mock timing: first token arrives 0.1s after request, generation takes 1.0s
        call_times = iter([0.0, 0.1, 1.1])
        with patch("worker.models.generator_remote.time.perf_counter", side_effect=call_times):
            result = gen.generate(prompt="q")

        # prompt_ms ≈ 100 ms (TTFT), predicted_ms ≈ 1000 ms
        assert result.prompt_ms == pytest.approx(100.0, abs=1.0)
        assert result.predicted_ms == pytest.approx(1000.0, abs=1.0)
        # tokens_per_second = 10 / 1.0 = 10
        assert result.predicted_per_second == pytest.approx(10.0, abs=0.1)
        gen.close()

    def test_zero_tokens_gives_zero_tps(self):
        """predicted_per_second is 0 when completion_tokens is 0."""
        gen = self._make_generator()
        # No usage chunk — default to 0 tokens
        chunks = [_delta_chunk("hi")]
        self._mock_stream(gen, _make_sse_lines(*chunks))

        result = gen.generate(prompt="q")

        assert result.predicted_per_second == 0.0
        gen.close()

    def test_chat_completions_endpoint_used(self):
        """The request must go to /chat/completions."""
        gen = self._make_generator()
        self._mock_stream(gen, _make_sse_lines(_delta_chunk("ok")))

        gen.generate(prompt="hello")

        gen._client.stream.assert_called_once()
        call_kwargs = gen._client.stream.call_args
        assert call_kwargs[0][1] == "/chat/completions"
        gen.close()

    def test_stop_sequences_forwarded(self):
        """stop sequences are included in the request payload."""
        gen = self._make_generator()
        self._mock_stream(gen, _make_sse_lines(_delta_chunk("ok")))

        gen.generate(prompt="hello", stop=["</s>"])

        payload = gen._client.stream.call_args[1]["json"]
        assert payload["stop"] == ["</s>"]
        gen.close()

    def test_rate_limiter_acquire_called(self):
        """Rate limiter's acquire() is called before each request."""
        mock_limiter = MagicMock()
        gen = self._make_generator(rate_limiter=mock_limiter)
        self._mock_stream(gen, _make_sse_lines(_delta_chunk("ok")))

        gen.generate(prompt="q")

        mock_limiter.acquire.assert_called_once()
        gen.close()

    def test_no_rate_limiter_no_error(self):
        """No rate limiter configured → generate completes without error."""
        gen = self._make_generator(rate_limiter=None)
        self._mock_stream(gen, _make_sse_lines(_delta_chunk("ok")))

        result = gen.generate(prompt="q")

        assert result.text == "ok"
        gen.close()

    def test_non_json_sse_lines_skipped(self):
        """Non-JSON SSE lines (comments, blanks) are silently ignored."""
        gen = self._make_generator()
        raw_lines = [
            ": keep-alive",
            "",
            f"data: {json.dumps(_delta_chunk('hi'))}",
            "data: [DONE]",
        ]
        self._mock_stream(gen, raw_lines)

        result = gen.generate(prompt="q")

        assert result.text == "hi"
        gen.close()

    def test_stream_option_include_usage_in_payload(self):
        """stream_options.include_usage is set to True in the request."""
        gen = self._make_generator()
        self._mock_stream(gen, _make_sse_lines(_delta_chunk("ok")))

        gen.generate(prompt="hello")

        payload = gen._client.stream.call_args[1]["json"]
        assert payload.get("stream_options", {}).get("include_usage") is True
        gen.close()


# ---------------------------------------------------------------------------
# RemoteGenerator — retry / backoff
# ---------------------------------------------------------------------------


class TestRemoteGeneratorBackoff:
    def _make_generator(self) -> RemoteGenerator:
        return RemoteGenerator(base_url="http://api.example.com/v1", model="m", api_key="sk")

    def _http_error(self, status: int) -> httpx.HTTPStatusError:
        resp = _error_response(status)
        return httpx.HTTPStatusError(f"HTTP {status}", request=resp.request, response=resp)

    def test_no_retry_on_404(self):
        """404 is not in backoff statuses and must be raised immediately."""
        gen = self._make_generator()
        gen._client.stream = MagicMock(side_effect=self._http_error(404))

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            gen.generate(prompt="q")

        assert exc_info.value.response.status_code == 404
        assert gen._client.stream.call_count == 1
        gen.close()

    def test_retries_on_429(self):
        """429 triggers retries with exponential backoff up to _MAX_RETRIES."""
        from worker.models.generator_remote import _MAX_RETRIES

        gen = self._make_generator()

        # First N attempts → 429; last attempt succeeds
        success_lines = _make_sse_lines(_delta_chunk("ok"), _usage_chunk(1, 1))
        success_response = MagicMock()
        success_response.raise_for_status = MagicMock()
        success_response.iter_lines.return_value = iter(success_lines)
        success_cm = MagicMock()
        success_cm.__enter__ = MagicMock(return_value=success_response)
        success_cm.__exit__ = MagicMock(return_value=False)

        error_429 = self._http_error(429)

        attempt_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise error_429
            return success_cm

        gen._client.stream = MagicMock(side_effect=_side_effect)

        with patch("worker.models.generator_remote.time.sleep"):
            result = gen.generate(prompt="q")

        assert result.text == "ok"
        assert attempt_count == 3
        gen.close()

    def test_raises_after_max_retries_on_503(self):
        """All attempts return 503 → RuntimeError raised after last attempt."""
        from worker.models.generator_remote import _MAX_RETRIES

        gen = self._make_generator()
        gen._client.stream = MagicMock(side_effect=self._http_error(503))

        with patch("worker.models.generator_remote.time.sleep"):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                gen.generate(prompt="q")

        assert exc_info.value.response.status_code == 503
        assert gen._client.stream.call_count == _MAX_RETRIES + 1
        gen.close()

    def test_backoff_debug_logged_on_retry(self):
        """A debug log message must be emitted for each backoff sleep."""
        gen = self._make_generator()

        success_lines = _make_sse_lines(_delta_chunk("ok"))
        success_response = MagicMock()
        success_response.raise_for_status = MagicMock()
        success_response.iter_lines.return_value = iter(success_lines)
        success_cm = MagicMock()
        success_cm.__enter__ = MagicMock(return_value=success_response)
        success_cm.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise self._http_error(429)
            return success_cm

        gen._client.stream = MagicMock(side_effect=_side_effect)

        with patch("worker.models.generator_remote.time.sleep"):
            with patch("worker.models.generator_remote.logger") as mock_logger:
                gen.generate(prompt="q")

        # At least one debug call for the backoff sleep
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        backoff_calls = [c for c in debug_calls if "Backoff" in c or "backoff" in c.lower() or "sleeping" in c.lower()]
        assert len(backoff_calls) >= 1
        gen.close()
