"""Generation model wrapper using llama-cpp-python."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama


@dataclass
class GenerationResult:
    """Result from text generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Generator:
    """Wraps llama-cpp-python for text generation.

    Instantiated with a larger context window (n_ctx=2048) suitable for
    RAG generation tasks where retrieved context is prepended to the prompt.
    """

    def __init__(self, model_path: Path, n_ctx: int = 2048, n_gpu_layers: int = -1):
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text completion for the given prompt."""
        result = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )

        text = result["choices"][0]["text"]
        usage = result.get("usage", {})

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
