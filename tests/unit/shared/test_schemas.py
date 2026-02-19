"""Tests that define the shared Pydantic model contracts."""

import pytest
from pydantic import ValidationError

from shared_types.schemas import (
    GenerateRequest,
    GenerateResponse,
    RetrievalData,
    InferenceMeasurement,
    HardwareMeasurement,
    RetrievalConfig,
    GenerationConfig,
    RunConfig,
    ChunkingConfig,
)


class TestChunkingConfig:
    """Tests for ChunkingConfig schema."""

    def test_fixed_strategy(self):
        """Fixed chunking strategy with chunk_size and chunk_overlap."""
        config = ChunkingConfig(
            strategy="fixed",
            chunk_size=500,
            chunk_overlap=64,
        )
        assert config.strategy == "fixed"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 64

    def test_char_split_strategy(self):
        """Character split strategy with split_sequence."""
        config = ChunkingConfig(
            strategy="char_split",
            split_sequence=". ",
        )
        assert config.strategy == "char_split"
        assert config.split_sequence == ". "

    def test_strategy_required(self):
        """strategy is required."""
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=500)

    def test_fixed_requires_chunk_size(self):
        """Fixed strategy requires chunk_size."""
        with pytest.raises(ValidationError):
            ChunkingConfig(strategy="fixed", chunk_overlap=64)

    def test_chunk_size_must_be_positive(self):
        """chunk_size must be > 0."""
        with pytest.raises(ValidationError):
            ChunkingConfig(strategy="fixed", chunk_size=0)

    def test_chunk_overlap_defaults_to_zero(self):
        """chunk_overlap defaults to 0."""
        config = ChunkingConfig(strategy="fixed", chunk_size=500)
        assert config.chunk_overlap == 0

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValidationError."""
        with pytest.raises(ValidationError):
            ChunkingConfig(strategy="invalid", chunk_size=500)


class TestRetrievalConfig:
    """Tests for RetrievalConfig schema."""

    def test_valid_config_with_fixed_chunking(self):
        """Full config with fixed chunking strategy."""
        config = RetrievalConfig(
            model="intfloat/multilingual-e5-small",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(
                strategy="fixed",
                chunk_size=500,
                chunk_overlap=64,
            ),
            k=5,
        )
        assert config.model == "intfloat/multilingual-e5-small"
        assert config.quantization == "fp16"
        assert config.dimensions == 384
        assert config.chunking.strategy == "fixed"
        assert config.chunking.chunk_size == 500
        assert config.k == 5

    def test_valid_config_with_char_split_chunking(self):
        """Full config with char_split chunking strategy."""
        config = RetrievalConfig(
            model="intfloat/multilingual-e5-small",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(
                strategy="char_split",
                split_sequence=". ",
            ),
            k=3,
        )
        assert config.chunking.strategy == "char_split"
        assert config.chunking.split_sequence == ". "

    def test_k_must_be_positive(self):
        """k must be > 0."""
        with pytest.raises(ValidationError):
            RetrievalConfig(model="m", quantization="fp16", dimensions=384, k=0)
        with pytest.raises(ValidationError):
            RetrievalConfig(model="m", quantization="fp16", dimensions=384, k=-1)

    def test_dimensions_must_be_positive(self):
        """dimensions must be > 0."""
        with pytest.raises(ValidationError):
            RetrievalConfig(model="m", quantization="fp16", dimensions=0, k=3)

    def test_k_defaults_to_3(self):
        """k defaults to 3 if not specified."""
        config = RetrievalConfig(model="m", quantization="fp16", dimensions=384)
        assert config.k == 3

    def test_quantization_required(self):
        """quantization is required."""
        with pytest.raises(ValidationError):
            RetrievalConfig(model="m", dimensions=384)

    def test_chunking_optional(self):
        """chunking is optional."""
        config = RetrievalConfig(
            model="m",
            quantization="fp16",
            dimensions=384,
        )
        assert config.chunking is None


class TestGenerationConfig:
    """Tests for GenerationConfig schema."""

    def test_valid_config(self):
        """Valid generation config."""
        config = GenerationConfig(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            quantization="q4_k_m",
        )
        assert config.model == "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        assert config.quantization == "q4_k_m"

    def test_quantization_defaults(self):
        """quantization defaults to q4_k_m."""
        config = GenerationConfig(model="some/model")
        assert config.quantization == "q4_k_m"


class TestRunConfig:
    """Tests for RunConfig schema."""

    def test_inherits_retrieval_and_generation(self):
        """RunConfig contains both RetrievalConfig and GenerationConfig."""
        run = RunConfig(
            run_id="test_001",
            retrieval=RetrievalConfig(
                model="embed",
                quantization="fp16",
                dimensions=384,
                k=5,
            ),
            generation=GenerationConfig(model="llm", quantization="q4_k_m"),
        )
        assert run.run_id == "test_001"
        assert run.retrieval.k == 5
        assert run.retrieval.quantization == "fp16"
        assert run.generation.model == "llm"

    def test_run_id_required(self):
        """run_id is required."""
        with pytest.raises(ValidationError):
            RunConfig(
                retrieval=RetrievalConfig(model="m", quantization="fp16", dimensions=384),
                generation=GenerationConfig(model="m"),
            )

    def test_full_config_with_chunking(self):
        """RunConfig with full chunking configuration."""
        run = RunConfig(
            run_id="mistral_q4_baseline_001",
            retrieval=RetrievalConfig(
                model="intfloat/multilingual-e5-small",
                quantization="fp16",
                dimensions=384,
                chunking=ChunkingConfig(
                    strategy="fixed",
                    chunk_size=500,
                    chunk_overlap=64,
                ),
                k=3,
            ),
            generation=GenerationConfig(
                model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                quantization="q4_k_m",
            ),
        )
        assert run.retrieval.chunking.chunk_size == 500
        assert run.generation.quantization == "q4_k_m"


class TestGenerateRequest:
    """Tests for GenerateRequest schema."""

    def test_valid_request(self):
        """Minimum valid request structure."""
        req = GenerateRequest(
            claim_id="claim_001",
            input_prompt="Is aspirin effective?",
            run_config=RunConfig(
                run_id="run_001",
                retrieval=RetrievalConfig(model="embed", quantization="fp16", dimensions=384, k=3),
                generation=GenerationConfig(model="llm"),
            ),
        )
        assert req.claim_id == "claim_001"
        assert req.run_config.retrieval.k == 3

    def test_missing_claim_id_raises(self):
        """claim_id is required."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                input_prompt="test",
                run_config=RunConfig(
                    run_id="r1",
                    retrieval=RetrievalConfig(model="m", quantization="fp16", dimensions=384),
                    generation=GenerationConfig(model="m"),
                ),
            )

    def test_missing_input_prompt_raises(self):
        """input_prompt is required."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                claim_id="c1",
                run_config=RunConfig(
                    run_id="r1",
                    retrieval=RetrievalConfig(model="m", quantization="fp16", dimensions=384),
                    generation=GenerationConfig(model="m"),
                ),
            )

    def test_request_with_fixed_chunking(self):
        """Request includes fixed chunking config from run_config."""
        req = GenerateRequest(
            claim_id="claim_001",
            input_prompt="Is aspirin effective?",
            run_config=RunConfig(
                run_id="mistral_q4_baseline_001",
                retrieval=RetrievalConfig(
                    model="intfloat/multilingual-e5-small",
                    quantization="fp16",
                    dimensions=384,
                    chunking=ChunkingConfig(
                        strategy="fixed",
                        chunk_size=500,
                        chunk_overlap=64,
                    ),
                    k=3,
                ),
                generation=GenerationConfig(
                    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    quantization="q4_k_m",
                ),
            ),
        )
        assert req.run_config.retrieval.chunking.strategy == "fixed"
        assert req.run_config.retrieval.chunking.chunk_size == 500
        assert req.run_config.retrieval.chunking.chunk_overlap == 64

    def test_request_with_char_split_chunking(self):
        """Request includes char_split chunking config from run_config."""
        req = GenerateRequest(
            claim_id="claim_002",
            input_prompt="Does exercise help?",
            run_config=RunConfig(
                run_id="mistral_q4_k5_001",
                retrieval=RetrievalConfig(
                    model="intfloat/multilingual-e5-small",
                    quantization="fp16",
                    dimensions=384,
                    chunking=ChunkingConfig(
                        strategy="char_split",
                        split_sequence=". ",
                    ),
                    k=5,
                ),
                generation=GenerationConfig(
                    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    quantization="q4_k_m",
                ),
            ),
        )
        assert req.run_config.retrieval.chunking.strategy == "char_split"
        assert req.run_config.retrieval.chunking.split_sequence == ". "
        assert req.run_config.retrieval.k == 5


class TestGenerateResponse:
    """Tests for GenerateResponse schema."""

    def test_response_structure(self):
        """Response has all required fields."""
        resp = GenerateResponse(
            output="Answer text",
            retrieval_data=RetrievalData(
                cited_doc_ids=["doc1"],
                retrieved_chunks=["chunk1"],
            ),
            inference_measurement=InferenceMeasurement(
                e2e_latency_ms=1000.0,
                retrieval_latency_ms=100.0,
                ttft_ms=200.0,
                llm_generation_latency_ms=700.0,
                prompt_tokens=100,
                completion_tokens=50,
                tokens_per_second=71.4,
            ),
            hardware_measurement=HardwareMeasurement(
                max_ram_usage_mb=512.0,
                avg_cpu_utilization_pct=50.0,
                peak_cpu_temp_c=65.0,
                swap_in_bytes=0,
                swap_out_bytes=0,
            ),
        )
        assert resp.output == "Answer text"

    def test_peak_cpu_temp_optional(self):
        """Temperature can be None if sensor unavailable."""
        hw = HardwareMeasurement(
            max_ram_usage_mb=512.0,
            avg_cpu_utilization_pct=50.0,
            peak_cpu_temp_c=None,
            swap_in_bytes=0,
            swap_out_bytes=0,
        )
        assert hw.peak_cpu_temp_c is None

    def test_retrieval_data_length_match(self):
        """cited_doc_ids and retrieved_chunks must match length."""
        with pytest.raises(ValidationError):
            RetrievalData(
                cited_doc_ids=["doc1", "doc2"],
                retrieved_chunks=["only_one"],
            )
