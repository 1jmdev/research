# Models & architectures: what the 2025–2026 frontier models actually use

> Synthesis of [`papers/models-and-architectures`](../papers/models-and-architectures/README.md). The
> [technical reports folder](../papers/models-and-architectures/technical-reports/README.md) is the best single source
> of "what works at scale". Each report lists architecture, data, training compute and post-training recipes.

## Architecture patterns in open frontier models (from the tech reports)

| Model family | Report(s) | Attention | FFN | Notable |
| --- | --- | --- | --- | --- |
| DeepSeek V3.2 → V4 → V4.1 | [V3.2](../papers/models-and-architectures/technical-reports/2512.02556-deepseek-v3-2-pushing-the-frontier-of-open-large-language-models.md), [V4](../papers/models-and-architectures/technical-reports/2606.19348-deepseek-v4-towards-highly-efficient-million-token-context-intelligenc.md), [V4.1-Flash](../papers/models-and-architectures/technical-reports/2609.19969-deepseek-v4-1-flash-pushing-the-limits-of-kv-cache-compression.md) | MLA → **DSA** (V3.2: lightning indexer + top-k) → **CSA + HCA** hybrid (V4: compressed sparse + heavily compressed attention; 10% of V3.2's KV at 1M context) → CSA2 with cross-layer KV reuse + **FP4 KV** (V4.1: 890 B/token of global KV) | fine-grained MoE + shared experts (V4-Pro 1.6T/49B active) | MTP; FP8 training; **mHC** hyper-connections ([mHC](../papers/models-and-architectures/novel-architectures/2512.24880-mhc-manifold-constrained-hyper-connections.md)); **Muon**; causal encoder-decoder (V4.1: 8B active in prefill, 16B in decode); Engram conditional memory ([Conditional Memory via Scalable Lookup](../papers/models-and-architectures/novel-architectures/2601.07372-conditional-memory-via-scalable-lookup-a-new-axis-of-sparsity-for-larg.md)) |
| Qwen3 / Qwen3-Next / 3.5 | [Qwen3](../papers/models-and-architectures/technical-reports/2505.09388-qwen3-technical-report.md), [Qwen3.8-Next design](../papers/models-and-architectures/technical-reports/2608.30320-on-the-design-of-qwen3-8-next-architecture-evaluation-efficiency-and-t.md) | GQA → **Gated DeltaNet + gated attention hybrid** (3:1) | MoE (128–512 experts) | thinking-budget control; MTP; gated attention ([Gated Attention for Large Language Models](../papers/attention/attention-variants/2505.06708-gated-attention-for-large-language-models-non-linearity-sparsity-and-a.md)) |
| Kimi K2 / Linear | Kimi K2, [Kimi Linear](../papers/attention/linear-attention/2510.26692-kimi-linear-an-expressive-efficient-attention-architecture.md) | MLA (K2); **KDA + MLA hybrid** (Kimi Linear: −75% KV, 6× decode at 1M, beats full MLA) | 1T MoE (32B active) | MuonClip optimizer; open KDA kernels + vLLM |
| GLM-4.5 / 5 | [GLM-4.5](../papers/models-and-architectures/technical-reports/2508.06471-glm-4-5-agentic-reasoning-and-coding-arc-foundation-models.md), [GLM-5](../papers/models-and-architectures/technical-reports/2602.15763-glm-5-from-vibe-coding-to-agentic-engineering.md) | GQA with many heads; partial RoPE; QK-norm | MoE 355B/32B | MTP; Muon; async agent RL |
| MiniMax-01 / M1 / M2 | [MiniMax-01](../papers/models-and-architectures/technical-reports/2501.08313-minimax-01-scaling-foundation-models-with-lightning-attention.md), [MiniMax-M1](../papers/models-and-architectures/technical-reports/2506.13585-minimax-m1-scaling-test-time-compute-efficiently-with-lightning-attent.md) | **Lightning (linear) attention hybrid** (7:1) → full attention again in M2; [MSA](../papers/attention/sparse-attention/2606.13392-minimax-sparse-attention.md) | MoE | 1M context; CISPO RL |
| Ling / Ring 2.x | [Ling-1T](../papers/models-and-architectures/technical-reports/2510.22115-every-activation-boosted-scaling-general-reasoner-to-1-trillion-open-l.md), [Ring-linear](../papers/models-and-architectures/technical-reports/2510.19338-every-attention-matters-an-efficient-hybrid-architecture-for-long-cont.md) | hybrid linear | MoE, ~1/32 activation | FP8 kernels; 1T reasoning |
| NVIDIA Nemotron-H / Nano 2 / 3 | Nemotron reports | **Mamba-2 hybrid** | dense → MoE (Nemotron 3) | NVFP4 pre-training ([Pretraining Large Language Models with NVFP4](../papers/quantization/low-precision-training/2509.25149-pretraining-large-language-models-with-nvfp4.md)) |
| gpt-oss | gpt-oss model card | alternating dense + banded sliding window, attention sinks | MoE | MXFP4 weights at release |
| Gemma 3 | [Gemma 3](../papers/models-and-architectures/technical-reports/2503.19786-gemma-3-technical-report.md) | 5:1 local:global sliding window | dense | QAT checkpoints |
| Diffusion LMs | LLaDA2.0 ([LLaDA2.0](../papers/decoding/diffusion-language-models/2512.15745-llada2-0-scaling-up-diffusion-language-models-to-100b.md)), Dream ([Dream 7B](../papers/decoding/diffusion-language-models/2508.15487-dream-7b-diffusion-large-language-models.md)), Mercury ([Mercury](../papers/decoding/diffusion-language-models/2506.17298-mercury-ultra-fast-language-models-based-on-diffusion.md)), Seed Diffusion ([Seed Diffusion](../papers/decoding/diffusion-language-models/2508.02193-seed-diffusion-a-large-scale-diffusion-language-model-with-high-speed.md)) | bidirectional / block | MoE (LLaDA2.0) | 1–2k tok/s coding models |

**Takeaways for a runtime that must run everything:** support MLA (absorbed and naive paths), GQA with sinks and
sliding windows, DSA/NSA/MSA top-k sparse attention, Gated DeltaNet/KDA/Mamba-2 recurrent states, MoE with shared
experts plus EP, MTP heads as drafters, FP8/MXFP4/NVFP4 weights, and block-diffusion decoding.

## Architecture research worth knowing

* [Physics of LMs 4.1 — Canon layers](../papers/models-and-architectures/novel-architectures/2512.17351-physics-of-language-models-part-4-1-architecture-design-and-the-magic.md): cheap local token mixing (short convs) lifts weak architectures
  (NoPE ≈ RoPE; linear attention ≈ Mamba2/GDN). Many 2026 hybrids add short convolutions for this reason.
* Residual-stream width: [mHC](../papers/models-and-architectures/novel-architectures/2512.24880-mhc-manifold-constrained-hyper-connections.md) (manifold-constrained hyper-connections, used in DeepSeek-V4) and
  [xHC](../papers/models-and-architectures/novel-architectures/2607.14530-xhc-expanded-hyper-connections.md) (expanded hyper-connections with a memory-traffic-aware Flash variant).
* Massive activations: [A Single Layer to Explain Them All](../papers/models-and-architectures/novel-architectures/2605.08504-a-single-layer-to-explain-them-all-understanding-massive-activations-i.md) traces them to a single layer; fixing them helps both quality and quantization.

## Small language models ([folder](../papers/models-and-architectures/small-language-models/README.md))

* Routing between small and large models: [R2R](../papers/decoding/speculative-decoding/2505.21600-r2r-efficiently-navigating-divergent-reasoning-paths-with-small-large.md) (token-level routing between R1-1.5B and R1-32B,
  2.8× faster at equal quality).
* Edge-optimal SLM design: [Return of the Encoder](../papers/models-and-architectures/small-language-models/2501.16273-return-of-the-encoder-maximizing-parameter-efficiency-for-slms.md) (encoder-decoder SLMs: 47% lower first-token latency,
  4.7× throughput on edge) and Nemotron-Flash (latency-optimal hybrid SLM).
* Reasoning in SLMs: [ThinkSLM](../papers/models-and-architectures/small-language-models/2502.11569-towards-reasoning-ability-of-small-language-models.md) benchmark (quantized, pruned and distilled SLMs compared).
