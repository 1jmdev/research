# Overviews & runtime roadmap

Hand-written syntheses of the ranked paper folders. Each claim links to the paper file, which holds the abstract,
extracted results tables and compute estimates. Paper numbers are the authors' claims.

| Area | Overview | Paper folders |
| --- | --- | --- |
| Quantization (sub-1-bit → FP8, FP4 formats, low-precision training) | [quantization.md](quantization.md) | [papers/quantization](../papers/quantization/README.md) |
| KV cache (quantization, eviction, low-rank, offload, reuse) | [kv-cache.md](kv-cache.md) | [papers/kv-cache](../papers/kv-cache/README.md) |
| Attention & sequence mixers (sparse, linear, SSM, hybrids, kernels) | [attention.md](attention.md) | [papers/attention](../papers/attention/README.md) |
| Decoding (speculative, MTP, Jacobi, diffusion LLMs, early exit) | [decoding.md](decoding.md) | [papers/decoding](../papers/decoding/README.md) |
| Model conversion (AR→diffusion, linearization, MHA→MLA, upcycling) + **compute costs** | [model-conversion.md](model-conversion.md) | [papers/model-conversion](../papers/model-conversion/README.md) |
| Compression (pruning, activation sparsity, low-rank, distillation) | [compression.md](compression.md) | [papers/compression](../papers/compression/README.md) |
| Mixture of Experts | [mixture-of-experts.md](mixture-of-experts.md) | [papers/mixture-of-experts](../papers/mixture-of-experts/README.md) |
| Serving systems, kernels & hardware | [serving-systems.md](serving-systems.md) | [papers/serving-systems](../papers/serving-systems/README.md) |
| Training (optimizers, scaling laws, RL, data, PEFT) | [training.md](training.md) | [papers/training](../papers/training/README.md) |
| Reasoning efficiency & test-time compute | [reasoning.md](reasoning.md) | [papers/reasoning](../papers/reasoning/README.md) |
| Context & visual-token compression | [context-compression.md](context-compression.md) | [papers/context-compression](../papers/context-compression/README.md) |
| What frontier models use (tech reports) | [models-and-architectures.md](models-and-architectures.md) | [papers/models-and-architectures](../papers/models-and-architectures/README.md) |

## Roadmap: building a state-of-the-art LLM runtime (2026)

Ordered by payoff ÷ effort for a new engine. Each item links to the evidence.

### Tier 1: table stakes
1. **Paged KV + radix prefix cache + host/SSD tier.** LMCache / Strata style ([kv-cache](kv-cache.md#tldr-for-a-runtime-builder)).
2. **Continuous batching + chunked prefill + PD disaggregation hooks** ([serving](serving-systems.md)).
3. **Attention engine**: FlashInfer-class paged decode for GQA, MLA (absorbed) and sliding window with sinks, plus
   FP8 attention ([attention](attention.md)).
4. **Weight formats**: W4A16 (GPTQ/AWQ layouts), FP8 W8A8, and **NVFP4/MXFP4** on Blackwell
   ([quantization](quantization.md#tldr-for-a-runtime-builder)).
5. **MoE**: grouped GEMM, shared experts, EP with overlapped all-to-all ([MoE](mixture-of-experts.md)).

### Tier 2: the big multipliers
6. **Generic draft/verify engine**: MTP heads, EAGLE-3 heads, block-diffusion drafters (DFlash) and self-drafting
   (Jacobi Forcing), with tree attention and KV rollback ([decoding](decoding.md#tldr-for-a-runtime-builder)).
7. **KV quantization (K4V2 → 2-bit with rotations)** and query-agnostic eviction (KVzip / DMS)
   ([kv-cache](kv-cache.md)).
8. **Hybrid-model executor**: recurrent-state cache (Gated DeltaNet / KDA / Mamba-2) alongside paged KV, including
   state checkpoints for prefix caching ([attention](attention.md#what-to-implement-first-runtime)).
9. **Top-k sparse attention kernels** (NSA / DSA / CSA / MSA) with cross-layer index reuse ([attention](attention.md)).
10. **Block-diffusion decode mode** for dLLMs (LLaDA2, Dream, SDAR, Fast-dLLM v2): block KV cache plus
    confidence-threshold parallel unmasking ([decoding](decoding.md#diffusion-language-models-dllms)).

### Tier 3: differentiators
11. Agent-aware scheduling: program graphs, tool-call idle offload, KV reuse across agents
    ([serving](serving-systems.md), [kv-cache](kv-cache.md)).
12. Phase-aware precision: NVFP4 prefill with BF16/FP8 decode ([quantization](quantization.md)).
13. Reasoning-budget control and early exit of CoT ([reasoning](reasoning.md)).
14. Visual-token pruning before the LLM ([context compression](context-compression.md)).
15. Activation-sparsity kernels (batch-friendly: Polar Sparsity) ([compression](compression.md)).
16. Determinism mode (verify/rollback) and FP8 RL rollouts that match the trainer ([serving](serving-systems.md),
    [training](training.md)).

## Roadmap: building / converting models cheaply

| Goal | Cheapest known path | ≈ cost | Details |
| --- | --- | --- | --- |
| Make a model 2–6× faster at decode, same weights | EAGLE-3 / MTP drafter or DFlash drafter | ~10–100 H100-h per model | [decoding](decoding.md) |
| Parallel decoding without a drafter | Jacobi Forcing distillation | ~10²–10³ H100-h (8B) | [decoding](decoding.md), [conversion](model-conversion.md) |
| AR → block-diffusion LM | Fast-dLLM v2 / Efficient-DLM / NBDiff recipe | ~1B tokens ≈ 30–100 H100-h (8B) up to 10⁴ | [conversion](model-conversion.md) |
| Long-context efficiency on an existing model | linearize to hybrid (RADLADS / HALO / Zebra-Llama) | 0.35–11B tokens ≈ 12–400 H100-h (8B) | [conversion](model-conversion.md) |
| Shrink the KV cache 5–10× on an existing model | MHA/GQA → MLA (TransMLA / X-EcoMLA) | ~100–300 H100-h (1–8B) | [conversion](model-conversion.md) |
| 4-bit weights, near-lossless | GPTQ-class PTQ (GPTAQ, SINQ calibration-free) | < 1–5 H100-h | [quantization](quantization.md) |
| Ternary model | PTQ ternarization (CAT-Q / PTQTP) → optional QAT | ~1 GPU-h (PTQ) / 10–14 GPU-days (QAT) | [quantization](quantization.md) |
| Smaller dense model | structured prune + distill (Minitron / DarwinLM) | 10–100B tokens | [compression](compression.md) |
