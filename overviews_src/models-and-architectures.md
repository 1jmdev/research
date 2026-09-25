# Models & architectures: what the 2025–2026 frontier models actually use

> Synthesis of [`papers/models-and-architectures`](../papers/models-and-architectures/README.md). The
> [technical reports folder](../papers/models-and-architectures/technical-reports/README.md) is the best single source
> of "what works at scale". Each report lists architecture, data, training compute and post-training recipes.

## Architecture patterns in open frontier models (from the tech reports)

| Model family | Report(s) | Attention | FFN | Notable |
| --- | --- | --- | --- | --- |
| DeepSeek V3.2 → V4 → V4.1 | [[2512.02556\|V3.2]], [[2606.19348\|V4]], [[2609.19969\|V4.1-Flash]] | MLA → **DSA** (V3.2: lightning indexer + top-k) → **CSA + HCA** hybrid (V4: compressed sparse + heavily compressed attention; 10% of V3.2's KV at 1M context) → CSA2 with cross-layer KV reuse + **FP4 KV** (V4.1: 890 B/token of global KV) | fine-grained MoE + shared experts (V4-Pro 1.6T/49B active) | MTP; FP8 training; **mHC** hyper-connections ([[2512.24880]]); **Muon**; causal encoder-decoder (V4.1: 8B active in prefill, 16B in decode); Engram conditional memory ([[2601.07372]]) |
| Qwen3 / Qwen3-Next / 3.5 | [[2505.09388\|Qwen3]], [[2608.30320\|Qwen3.8-Next design]] | GQA → **Gated DeltaNet + gated attention hybrid** (3:1) | MoE (128–512 experts) | thinking-budget control; MTP; gated attention ([[2505.06708]]) |
| Kimi K2 / Linear | Kimi K2, [[2510.26692\|Kimi Linear]] | MLA (K2); **KDA + MLA hybrid** (Kimi Linear: −75% KV, 6× decode at 1M, beats full MLA) | 1T MoE (32B active) | MuonClip optimizer; open KDA kernels + vLLM |
| GLM-4.5 / 5 | [[2508.06471\|GLM-4.5]], [[2602.15763\|GLM-5]] | GQA with many heads; partial RoPE; QK-norm | MoE 355B/32B | MTP; Muon; async agent RL |
| MiniMax-01 / M1 / M2 | [[2501.08313\|MiniMax-01]], [[2506.13585\|MiniMax-M1]] | **Lightning (linear) attention hybrid** (7:1) → full attention again in M2; [[2606.13392\|MSA]] | MoE | 1M context; CISPO RL |
| Ling / Ring 2.x | [[2510.22115\|Ling-1T]], [[2510.19338\|Ring-linear]] | hybrid linear | MoE, ~1/32 activation | FP8 kernels; 1T reasoning |
| NVIDIA Nemotron-H / Nano 2 / 3 | Nemotron reports | **Mamba-2 hybrid** | dense → MoE (Nemotron 3) | NVFP4 pre-training ([[2509.25149]]) |
| gpt-oss | gpt-oss model card | alternating dense + banded sliding window, attention sinks | MoE | MXFP4 weights at release |
| Gemma 3 | [[2503.19786\|Gemma 3]] | 5:1 local:global sliding window | dense | QAT checkpoints |
| Diffusion LMs | LLaDA2.0 ([[2512.15745]]), Dream ([[2508.15487]]), Mercury ([[2506.17298]]), Seed Diffusion ([[2508.02193]]) | bidirectional / block | MoE (LLaDA2.0) | 1–2k tok/s coding models |

**Takeaways for a runtime that must run everything:** support MLA (absorbed and naive paths), GQA with sinks and
sliding windows, DSA/NSA/MSA top-k sparse attention, Gated DeltaNet/KDA/Mamba-2 recurrent states, MoE with shared
experts plus EP, MTP heads as drafters, FP8/MXFP4/NVFP4 weights, and block-diffusion decoding.

## Architecture research worth knowing

* [[2512.17351|Physics of LMs 4.1 — Canon layers]]: cheap local token mixing (short convs) lifts weak architectures
  (NoPE ≈ RoPE; linear attention ≈ Mamba2/GDN). Many 2026 hybrids add short convolutions for this reason.
* Residual-stream width: [[2512.24880|mHC]] (manifold-constrained hyper-connections, used in DeepSeek-V4) and
  [[2607.14530|xHC]] (expanded hyper-connections with a memory-traffic-aware Flash variant).
* Massive activations: [[2605.08504]] traces them to a single layer; fixing them helps both quality and quantization.

## Small language models ([folder](../papers/models-and-architectures/small-language-models/README.md))

* Routing between small and large models: [[2505.21600|R2R]] (token-level routing between R1-1.5B and R1-32B,
  2.8× faster at equal quality).
* Edge-optimal SLM design: [[2501.16273|Return of the Encoder]] (encoder-decoder SLMs: 47% lower first-token latency,
  4.7× throughput on edge) and Nemotron-Flash (latency-optimal hybrid SLM).
* Reasoning in SLMs: [[2502.11569|ThinkSLM]] benchmark (quantized, pruned and distilled SLMs compared).
