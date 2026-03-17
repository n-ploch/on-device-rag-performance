Running a RAG pipeline on-device means every model choice influences answer quality as well as system performance and load. This analysis focuses on small Mistral-family models — Ministral 3 (3B and 8B parameters) and Mistral 7B v0.3 — evaluated across quantization levels from Q3 to Q8, with the goal of mapping accurate quality–cost tradeoffs for context-grounded answer generation. Llama 3.2 3B serves as a widely-used small-model alternative for direct comparison, and Mistral Large via API provides the quality ceiling: what a capable, unconstrained model would score on the same questions.

The experiment holds retrieval constant, with all configurations sharinh the same `e5-large` retriever and ChromaDB collection, so that differences in answer correctness, faithfulness, and hallucination can be attributed squarely to the generator. Alongside these RAG-specific quality metrics, every inference is instrumented with latency, throughput, and hardware utilisation measurements, enabling direct reasoning about which configurations deliver good answers at acceptable system cost.

---

## The Benchmarking Tool

All results in this post were collected using **[RAGrig](https://github.com/n-ploch/RAGrig)**, an open-source orchestration tool that instruments a full RAG pipeline with [OpenTelemetry](https://opentelemetry.io/) (OTEL) spans, making retrieval and generation fully traceable and exportable to any OTEL-compatible backend. The tool follows an **Orchestrator + Worker** architecture:

- The **Orchestrator** drives the evaluation loop and optionally ships OTEL spans to an external tracing backend (a self-hosted [Langfuse](https://langfuse.com/) instance was used for tracing experiments and LLM-based evaluation)
- The **Worker** runs quantized GGUF models via a `llama.cpp` server, making model deployment flexible.

---

## Experimental Setup

### RAG Pipeline

The benchmark uses a standard two-stage RAG pipeline with a fixed retrieval model and variable generation model:

- **Dataset:** eManuals subset from [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) — 61 questions drawn from consumer electronics manuals, each with ground-truth answers and relevant context chunks
- **Retrieval model:** `intfloat/multilingual-e5-large` (1024-dim embeddings, Q4 quantized), returning top-6 chunks via cosine similarity from a local ChromaDB collection
- **Generator prompt:** The system prompt instructs the model to answer from the provided context chunks; completion capped at 256 tokens
- **Evaluator (external):** Qwen3-235B scores each answer on three dimensions as judge. The LLM judges were implemented in Langfuse

### Generator Models Tested

| Family | Parameters | Quantizations |
|---|---|---|
| Llama 3.2 | 3B | IQ1, Q2, Q3, Q4, Q5, Q8 |
| Ministral 3 | 3B | Q4, Q5, Q8 |
| Ministral 3 | 8B | Q4, Q5 |
| Mistral 7B v0.3 | 7B | IQ3, Q3\_L, Q4, Q8 |
| Mistral Large 3 | 675B | API (quality ceiling) |

Generation parameters were held constant across all models:

```json
{
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false
}
```

### Hardware Under Test

All results were collected on an **Apple MacBook Pro with M1 chip** (10-Core CPU, 16-Core Neural Engine, 16 GB memory) leveraging Apple's Metal Performance Shaders (MPS).

### Running the Benchmark

One run configuration includes a dataset (with ground truth entries), a retrieval configuration, and a generation configuration. Each configuration ran 61 claims. Multiple repeated sessions were performed per configuration for stability analysis; this post focuses on cross-model comparisons averaging over all runs per configuration.

### Evaluation Metrics

#### Quality (LLM-as-judge)

| Metric | Range | Interpretation |
|---|---|---|
| **Correctness** | 0–1 | Does the answer correctly address the question? Higher is better. |
| **Hallucination** | 0–1 | Does the answer contain completely fabricated statements? Lower is better. |
| **Faithfulness** | 0–1 | Are all answer claims traceable to the retrieved context? Higher is better. |

**Faithfulness** (adapted from [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)): the judge breaks the answer into individual statements and checks each one for inferability from the context chunks. Score = verifiable statements / total statements.

**Hallucination**: identifies statements that cannot be traced to either the retrieved context or the ground truth. This metric is particularly discriminating on the eManuals dataset: because questions are tightly coupled to specific product documentation, any answer that reaches beyond the provided chunks tends to invent domain-specific details.

#### Latency & Hardware

| Metric | Description |
|---|---|
| **E2E latency p90 (ms)** | Wall-clock time from query to final token, 90th percentile |
| **TTFT p90 (ms)** | Time-to-first-token — reflects prompt encoding overhead, 90th percentile |
| **Tokens/sec** | Generation throughput, median |
| **Completion tokens** | Output length, median |
| **RAM (MB)** | Peak physical memory usage, median |

---

## Key Findings at a Glance

The table below summarises all 16 configurations. Three patterns stand out immediately and are explored in detail in the sections that follow.

![Overview table with configuration summary](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/overview_table.png)

<!-- | Model | Correctness | Hallucination | Faithfulness | Tokens/sec | Completion tokens | RAM (MB) | E2E p90 (ms) | TTFT p90 (ms) |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B IQ1 | 0.16 | 0.84 | 0.23 | 44.9 | 256 | 6,539 | 7,089 | 1,221 |
| Llama 3.2 3B Q2 | 0.67 | 0.25 | 0.83 | 47.9 | 256 | 8,657 | 6,737 | 1,272 |
| Llama 3.2 3B Q3 | 0.68 | 0.23 | 0.83 | 46.5 | 138 | 8,123 | 6,689 | 1,270 |
| Llama 3.2 3B Q4 | 0.65 | 0.21 | 0.82 | 51.4 | 150 | 7,595 | 6,401 | 1,261 |
| Llama 3.2 3B Q5 | 0.67 | 0.22 | 0.81 | 42.5 | 180 | 8,101 | 7,181 | 1,331 |
| Llama 3.2 3B Q8 | 0.63 | 0.26 | 0.82 | 41.2 | 184 | 8,614 | 7,346 | 1,210 |
| **Ministral3 3B Q4** | **0.74** | **0.21** | **0.83** | 48.1 | 256 | 7,323 | 6,714 | 1,375 |
| Ministral3 3B Q5 | 0.72 | 0.22 | 0.85 | 38.9 | 253 | 7,699 | 8,243 | 1,473 |
| Ministral3 3B Q8 | 0.73 | 0.21 | 0.83 | 37.9 | 256 | 8,453 | 8,091 | 1,213 |
| Ministral3 8B Q4 | 0.77 | 0.15 | 0.86 | 22.3 | 238 | 9,610 | 14,584 | 3,139 |
| Ministral3 8B Q5 | 0.75 | 0.14 | 0.86 | 18.4 | 232 | 10,218 | 17,374 | 3,334 |
| Mistral 7Bv0.3 IQ3 | 0.63 | 0.28 | 0.81 | 26.3 | 156 | 8,208 | 12,664 | 3,216 |
| Mistral 7Bv0.3 Q3L | 0.74 | 0.13 | 0.89 | 21.4 | 110 | 8,532 | 14,118 | 3,344 |
| **Mistral 7Bv0.3 Q4** | **0.70** | **0.15** | **0.90** | 25.9 | 127 | 8,853 | 12,337 | 3,072 |
| Mistral 7Bv0.3 Q8 | 0.73 | 0.12 | 0.90 | 21.1 | 117 | 11,170 | 13,946 | 2,787 |
| **Mistral Large (ceiling)** | **0.84** | **0.07** | **0.92** | 63.4 | 241 | 5,510 | 5,069 | 647 | -->

1. **Ministral 3B outperforms Llama 3.2 3B at comparable or lower system cost.** At Q4, Ministral 3B achieves correctness of 0.74 versus 0.65 for Llama — a gap of nearly 10 percentage points — while using *less* RAM (7,323 MB vs 7,595 MB).

2. **Scaling from 3B to 7/8B is a latency multiplier, but generates real quality improvements.** Moving from Ministral 3B to 8B approximately doubles E2E p90 latency (~6,700 ms → ~14,600 ms) and TTFT (~1,400 ms → ~3,100 ms). The quality return is real: hallucination drops ~6–10 percentage points and faithfulness improves by a similar margin.

3. **Mistral 7B consistently outputs fewer tokens.** Llama 3.2 3B and Ministral 3B produce a median of 180–256 completion tokens; Mistral 7B consistently outputs 110–156 tokens for the same questions. This verbosity gap keeps Mistral 7B E2E latency from scaling as badly as Ministral 8B.

---

## Retrieval Quality Baseline

Before comparing models, it's worth anchoring on the retrieval distribution. Because all generator models share the same fixed retrieval model (`e5-large`, Q4), they see identical retrieved context for each claim. The recall@6 distribution below sets the stage: when recall falls below ~0.4, the retrieved chunks are unlikely to contain the ground-truth answer, placing any generator in a difficult position. This threshold is used later in the low-recall robustness analysis.

The retrieval latency plot confirms that the retrieval step is a minor contributor to total E2E latency with low variance as compared to generation latency, thus it can be neglected in later comparison.

![Retrieval recall distribution and latency](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/retrieval_recall_distribution.png)

---

## Mistral Family Outperforms Llama 3.2

Across most quantization levels, Ministral and Mistral models deliver meaningfully better correctness than Llama 3.2 3B, with additional improvements in hallucination and faithfulness.

The trade-off: Mistral-family generation runs roughly 2 seconds slower on average, though TTFT is comparable at ~1 second. At Q4, Llama achieves ~51 tokens/sec vs ~48 for Ministral; at Q5, ~42 vs ~38. **Llama, however, consumes 300–400 MB more RAM at equivalent quantization**, suggesting less efficient memory access.

![Quality metrics and latency across all configurations](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/quality_metrics_all_models.png)

![RAM usage, tokens per second, and hardware metrics](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/ram_throughput_hardware.png)

---

## Quantization: Diminishing Returns, Collapse at the Extremes

Within each model family, returns diminish with increasing precision: moving from Q4 to Q5 to Q8 yields modest quality improvements at equally modest latency cost. **The exception is extreme compression** — IQ1 and Q2 for Llama 3.2, and IQ3 for Mistral 7B produce near-gibberish outputs, collapsing correctness toward zero and hallucination toward 1.0. The quality cliff is sharp, not gradual.

**Apple Silicon works best with Q4/Q8 quantizations**: Mistral 7B at Q4 *outperforms* lower quantization levels on generation latency, probably related to Apple Metal being optimized for Q4 and Q8. Q8 performs nearly as fast as Q5 for both Llama and Ministral, meaning there is very little latency penalty for choosing the highest quality.

The IQ quantizations are not only slower, they also drop substantially in quality — using low quantizations on hardware unoptimized for them yields no advantage at all.

### Llama 3.2 3B: Quality Across Quantization Levels

![Llama 3.2 3B quality across quantization levels](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/llama_quality_quantization.png)

### Ministral 3 (3B & 8B): Quality Across Quantization Levels

![Ministral quality across quantization levels](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/ministral_quality_quantization.png)

### Mistral 7B v0.3: Quality Across Quantization Levels

![Mistral 7B quality across quantization levels](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/mistral7b_quality_quantization.png)

### Latency Across Quantization Levels by Family

![Latency across quantization levels by family](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/latency_quantization_by_family.png)

---

## More Parameters Reduce Hallucination, But Latency Grows

Scaling parameter count brings clear improvements in faithfulness, hallucination, and correctness. Mistral 7B and Ministral 8B perform similarly on both hallucination and faithfulness, which is an interesting finding given Ministral's stronger benchmark scores. One hypothesis: the task structure here (short context, direct Q&A from documentation) may not allow Ministral to leverage its stronger reasoning capabilities. This would require further testing with longer contexts and more complex reasoning tasks.

The latency cost is substantial: **E2E latency more than doubles from Ministral 3B to 8B**, largely due to generation latency. The widening gap between generation latency and E2E latency reflects a growing TTFT (from ~1.2 s at 3B to ~3 s at 7B and beyond) as prompt encoding consumes more of the request budget.

### Quality Across Parameter Scales (Q4)

![Quality metrics across parameter scales](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/quality_by_parameter_scale.png)

### Latency Breakdown Across Parameter Scales

![Latency breakdown across parameter scales](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/latency_by_parameter_scale.png)

### Quality by Prompt Length

A quick look at short vs. long context (top 25% of prompt token lengths vs. bottom 75%) does not give a conclusive answer on whether Ministral benefits from longer context. Ministral tends to perform better on shorter contexts, aligning with other models in this benchmark.

![Quality by prompt length for Mistral family](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/quality_prompt_length.png)

---

## The Trade-Off: Mistral 7B Q3_L vs. Ministral 3B Q8

Two configurations with nearly identical RAM footprints (~8.5 GB) arrive at very different operating points:

| Model | Correctness | Hallucination | Faithfulness | Tokens/sec | Completion tokens | RAM (MB) | E2E p90 (ms) | TTFT p90 (ms) |
|---|---|---|---|---|---|---|---|---|
| Ministral3 3B Q8 | 0.73 | 0.21 | 0.83 | 37.9 | 256 | 8,453 | 8,091 | 1,213 |
| Mistral 7Bv0.3 Q3L | 0.74 | 0.13 | 0.89 | 21.4 | 110 | 8,532 | 14,118 | 3,344 |

- **Ministral 3B Q8** achieves similar correctness and much higher token throughput, with a lower TTFT (~1.2 s vs ~3.3 s) while generating more tokens per answer.
- **Mistral 7B Q3_L** scores better on faithfulness (0.89 vs 0.83) and hallucination (0.13 vs 0.21) on average, but at the cost of significantly higher p90 latency (14,118 ms vs 8,091 ms E2E).

**The choice depends on the use case:** if you need fast throughput, Ministral 3B Q8 is the stronger pick. If source fidelity and reduced hallucination are the priority, e.g. in citation-sensitive applications, Mistral 7B Q3_L delivers.

---

## When Retrieval Fails: Model Robustness Across Recall Regimes

Comparing low-recall (recall@6 < 0.4) and high-recall (recall@6 ≥ 0.8) claims side by side reveals how much of each model's quality is *retrieval-dependent* versus *inherent*. When context is good, most models score reasonably well. When context is poor, differences compound: faithfulness and hallucination degrade for all models, but larger models hold up more gracefully.

![Quality metrics by recall regime](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/quality_recall_regime.png)

### Quality by Recall Regime — Full Table

![Full Table quality metrics by recall regime](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/recall_table.png)

<!-- | Model | Corr Low | Corr High | Hall Low | Hall High | Faith Low | Faith High | Tokens Low | Tokens High |
|---|---|---|---|---|---|---|---|---|
| Llama 3.2 3B IQ1 | 0.13 | 0.17 | 0.83 | 0.84 | 0.26 | 0.25 | 256 | 256 |
| Llama 3.2 3B Q2 | 0.52 | 0.71 | 0.10 | 0.24 | 0.84 | 0.86 | 256 | 256 |
| Llama 3.2 3B Q3 | 0.54 | 0.70 | 0.33 | 0.21 | 0.83 | 0.86 | 172 | 132 |
| Llama 3.2 3B Q4 | 0.52 | 0.67 | 0.18 | 0.24 | 0.85 | 0.81 | 142 | 164 |
| Llama 3.2 3B Q5 | 0.48 | 0.70 | 0.30 | 0.22 | 0.78 | 0.81 | 168 | 177 |
| Llama 3.2 3B Q8 | 0.49 | 0.64 | 0.36 | 0.25 | 0.76 | 0.83 | 194 | 174 |
| Ministral3 3B Q4 | 0.61 | 0.75 | 0.17 | 0.22 | 0.82 | 0.86 | 247 | 256 |
| Ministral3 3B Q5 | 0.57 | 0.74 | 0.23 | 0.21 | 0.86 | 0.86 | 234 | 236 |
| Ministral3 3B Q8 | 0.58 | 0.76 | 0.24 | 0.20 | 0.85 | 0.85 | 256 | 246 |
| Ministral3 8B Q4 | 0.67 | 0.77 | 0.14 | 0.15 | 0.86 | 0.85 | 211 | 227 |
| Ministral3 8B Q5 | 0.66 | 0.75 | 0.10 | 0.14 | 0.88 | 0.87 | 209 | 190 |
| Mistral 7Bv0.3 IQ3 | 0.60 | 0.61 | 0.19 | 0.31 | 0.82 | 0.82 | 115 | 154 |
| Mistral 7Bv0.3 Q3L | 0.63 | 0.78 | 0.10 | 0.13 | 0.94 | 0.90 | 92 | 118 |
| Mistral 7Bv0.3 Q4 | 0.59 | 0.72 | 0.15 | 0.15 | 0.88 | 0.91 | 98 | 134 |
| Mistral 7Bv0.3 Q8 | 0.61 | 0.73 | 0.08 | 0.13 | 0.88 | 0.92 | 82 | 124 |
| Mistral Large (ceiling) | 0.72 | 0.86 | 0.09 | 0.07 | 0.95 | 0.95 | 177 | 241 | -->

**Key observations:**

- **Mistral 7B and Ministral 8B show their strength under low recall.** Their stronger instruction-following allows for better faithfulness and lower hallucination when context is poor, where smaller models tend to confabulate.
- **Quantization level has surprisingly little impact on hallucination under low recall** — except at extreme compression (IQ1/IQ3). For robustness to poor retrieval, model architecture matters far more than bit-width in the Q4–Q8 range.
- **The correctness gap between low and high recall** shows how much headroom each model has when given good context — a useful reminder that fixing retrieval is usually a better investment than scaling the model.

---

## Conclusion

**The Mistral family earns its memory footprint; Llama does not.** Llama 3.2 3B consumes 300–400 MB *more* RAM than comparable Ministral configurations while delivering lower correctness, faithfulness, and hallucination scores. The extra memory overhead is not being converted into better answers. Ministral 3B Q4 or Q5 uses less system memory, generates faster, and produces better RAG outputs. The case for Llama 3.2 on this task is not supported by the data.

**Quantization interacts with hardware in a non-obvious way.** Apple Silicon's MPS backend is optimised for 4-bit and 8-bit weight formats: Q8 is nearly as fast as Q5 for both Llama and Ministral, making it the obvious choice when RAM is available — you pay effectively no extra latency for the highest quality within the safe compression range. Extreme quantization inverts this logic entirely: IQ1 and IQ3 still consume ~6–7 GB RAM and require several seconds of inference time while producing near-worthless outputs.

**Parameter count is the most expensive knob to turn.** Ministral 8B's TTFT is 2× compared to Ministral 3B and its E2E latency climbs sharply — but the gains in faithfulness and hallucination are modest. Mistral 7B achieves better safety metrics than Ministral 8B without the same per-token latency increase, making it the more system-efficient path to RAG reliability.

**Mistral 7B is the most stable generator across retrieval conditions.** Its output stability, combined with a memory footprint comparable to Ministral 3B Q8, makes Mistral 7B the better choice when retrieval quality cannot be guaranteed. There is also an efficiency argument: Mistral 7B produces significantly fewer completion tokens (82–124) than Ministral (200–256), avoiding wasted compute on low-signal generations.

### Verdict

For on-device RAG where both answer quality and system load matter, **Mistral 7B Q4 on MPS provides the best quality per RAM-byte and the best generation latency with the highest adherence to context.** Ministral 3B Q8, being close in quality to its larger sibling, is the right pick when throughput is the priority and the 7B latencies are unacceptable. Both are above Llama 3.2 in RAG quality at comparable or lower system cost, especially when retrieval fails.

### RAGrig

This benchmark marks the first field test of RAGrig as versatile evaluation harness. The Orchestrator + Worker architecture made it straightforward to run repeated configurations on edge hardware, while OTEL instrumentation ensured standardized trace capture. The pipeline is deliberately modular: swapping in a different retriever, adding a re-ranker, or extending to agentic retrieval requires only changes to the pipeline itself; the tracing and repeatability guarantees around it remain intact, backed by llama.cpp's portability and OTEL's ecosystem reach.

### Limitations

Though trends are clearly identifiable, the variance driven partly by LLM-judges and by run-to-run model variance results in wide error bars. More experimental repetitions would tighten confidence intervals, though some of that variance may be intrinsic to stochastic generation at temperature 0.7 and unlikely to disappear entirely. Hardware constraints also cap the scope of the study: 16 GB unified memory limits testable models, making 10B+ parameter models inaccessible.

### Open Questions

- A clearer Q4 → Q5 → Q8 quality ordering within each family requires more experimental runs
- Cross-platform comparisons (NVIDIA CUDA, CPU-only) are directly supported by the OTEL-native benchmarking framework but not yet available
- Do Ministral's increased reasoning capabilities pay off in a different RAG setting — longer context, higher content complexity, more complex pipeline?

---

## Appendix: Detailed Distributions

### Latency Metrics (median, p25–p75 error bars)

![Appendix: quality metrics across all configurations](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_quality_metrics.png)

### Latency Metrics (median, p25–p75 error bars)

![Appendix: latency metrics across all configurations](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_latency_metrics.png)

### Generation Metrics (median, p25–p75 error bars)

![Appendix: generation metrics across all configurations](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_generation_metrics.png)

### Hardware Metrics (median, p25–p75 error bars)

![Appendix: hardware metrics across all configurations](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_hardware_metrics.png)

### Per-Claim Quality Profiles

#### Correctness — Per-Claim Profile Across All Runs

![Correctness per-claim profile](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_correctness_per_claim.png)

#### Faithfulness — Per-Claim Profile Across All Runs

![Faithfulness per-claim profile](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_faithfulness_per_claim.png)

#### Hallucination — Per-Claim Profile Across All Runs

![Hallucination per-claim profile](https://raw.githubusercontent.com/n-ploch/RAGrig/main/blogs/assets/appendix_hallucination_per_claim.png)
