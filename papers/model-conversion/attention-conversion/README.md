# Attention conversion (MHA/GQA → MLA, GQA uptraining, sparse retrofit)

Retrofitting the attention of a pretrained model: MHA2MLA, TransMLA, X-EcoMLA, GQA conversion, retrofitting sparse/sliding attention.

**15 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** This is attention-to-attention retrofitting: the model stays a Transformer, but its KV cache and attention cost
shrink. There are two families.

1. **MHA/GQA → MLA** (DeepSeek-style latent KV). [TransMLA](2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md) proves MLA is strictly more expressive than GQA
   at equal KV size. It converts via **RoRoPE** (concentrate the RoPE signal into one head) + **FreqFold** + joint KV
   low-rank factorization. MHA2MLA does partial-RoPE removal + SVD. The payoff is **reusing DeepSeek's MLA kernels**
   (FlashMLA, absorbed MQA decode in vLLM/SGLang) and ~93–97% KV reduction.

   The cheap part is initialization: [CARE](2603.17946-care-covariance-aware-and-rank-enhanced-decomposition-for-enabling-mul.md) (ICLR'26) makes it **activation-aware** (covariance-weighted
   SVD, per-layer rank allocation), which cuts the healing needed afterwards. Healing takes **1–6B tokens** at 7–8B.
2. **Full → sparse / sliding-window attention.** Full-attention models are intrinsically sparse. Only some heads are
   "retrieval heads"; the rest can become streaming/SWA heads.
   * [RTPurbo](2605.16928-full-attention-strikes-back-transferring-full-attention-into-sparse-wi.md) converts in **a few hundred training steps**: head specialization + low-dimensional retrieval
     index + dynamic top-p.
   * [SWAA](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md) shows naive SWA collapses. The fix is a recipe: keep some full-attention layers, sinks,
     interleave, and fine-tune briefly.
   * LongCat's [LoZA](2512.23966-efficient-context-scaling-with-longcat-zigzag-attention.md) and DeepSeek-V3.2's DSA show the same thing at frontier scale, applied during
     mid-training.

A new consideration is **hardware fit.** MLA's absorbed MQA path suits H100 compute/bandwidth ratios but loses head-axis
tensor parallelism and gains nothing from MTP on H20-class GPUs. [GQLA](2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) fixes this with group-indexed
up-projections that allow two equivalent decode paths.

### Conversion cost table (hand-checked)

| Method | Conversion | Tokens / steps | H100-h | Basis |
| --- | --- | ---: | ---: | --- |
| [RTPurbo](2605.16928-full-attention-strikes-back-transferring-full-attention-into-sparse-wi.md) | Full → head-wise sparse (retrieval/streaming heads) | ~1M-token alignment + ~600 steps at 48K ctx | **~10–50** | estimate from steps (H20) |
| [SWAA](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md) | Full → SWA hybrid | short fine-tune | **~15** (Qwen3-4B), **~36** (30B-A3B) | reported: 8×H20 × 12 h / 30 h |
| [NLL-guided layer selection](2606.27791-nll-guided-full-attention-layer-selection-for-training-free-sliding-wi.md) | Full → SWA hybrid, training-free | calibration only | **<1** | training-free |
| [CARE](2603.17946-care-covariance-aware-and-rank-enhanced-decomposition-for-enabling-mul.md) (ICLR'26) | GQA → MLA (activation-aware init) | 0B one-shot; 1–3B healing | **~0 / ~35–100** | estimate (8B) |
| [TransMLA](2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md) | GQA → MLA (Llama-2-7B, Qwen) | 6B | **~180** | estimate |
| [MHA2MLA](2502.14837-towards-economical-inference-enabling-deepseek-s-multi-head-latent-att.md) (ACL'25) | MHA/GQA → MLA | 0.6–1% of pretraining tokens (12K steps) | **~100–350** (7B) | estimate |
| [X-EcoMLA](2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md) | Attention → MLA via KD from a larger teacher | 3.6–7B | **~70–140 MI300-h** | reported |
| [LoZA](2512.23966-efficient-context-scaling-with-longcat-zigzag-attention.md) (LongCat-Flash) | Full → ZigZag sparse during mid-training | 500B + 40B long-context | part of the mid-training budget | frontier scale |

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [TransMLA](2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md) | Any GQA → MLA; RoRoPE + FreqFold decouple RoPE; balanced KV factorization; direct compatibility with DeepSeek code | 93% KV compressed on Llama-2-7B, **10.6× speedup at 8K**; 6B tokens to restore quality |
| 2 | [RTPurbo: Full Attention Strikes Back](2605.16928-full-attention-strikes-back-transferring-full-attention-into-sparse-wi.md) | Retrieval vs streaming heads + low-dim retrieval index + dynamic top-p; a few hundred steps | Near-lossless long-context and reasoning at high sparsity |
| 3 | [MHA2MLA](2502.14837-towards-economical-inference-enabling-deepseek-s-multi-head-latent-att.md) (ACL'25) | Contribution-aware partial-RoPE removal + SVD joint KV projection | Up to **96.87% KV reduction** (Llama-2-7B) with 0.6–1% of the data; composes with KV quantization |
| 4 | [CARE](2603.17946-care-covariance-aware-and-rank-enhanced-decomposition-for-enabling-mul.md) (ICLR'26) | Covariance-aware, rank-enhanced factorization + per-layer rank allocation | Much better one-shot MLA init; fewer healing tokens |
| 5 | [SWAA](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md) | Diagnoses SWA collapse (train/inference mismatch + no distant access); a recipe of FA layers + sinks + interleaving + light fine-tuning | Recovers long-context quality with SWA efficiency; ~12 h on 8×H20 for 4B |
| 6 | [GQLA](2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) | MLA variant whose weights admit **both** absorbed-MQA and GQA-style decode paths | Hardware-adaptive MLA: tensor parallelism + MTP gains on H20 |
| 7 | [LoZA](2512.23966-efficient-context-scaling-with-longcat-zigzag-attention.md) (Meituan) | Convert full attention to ZigZag sparse during mid-training | 1M-token context; >50% prefill speedup and >30% decode savings at 256K |
| 8 | [X-EcoMLA](2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md) | SVD init + distillation from a larger teacher ("dark knowledge") | 6.4× KV compression on Llama-3.2-1B, no loss, 70 MI300 GPU-h |
| 9 | [Attention Editing](2604.05688-attention-editing-a-versatile-framework-for-cross-architecture-attenti.md) | General framework: any trained attention → MLA or hybrid SWA without strict structural requirements | Practical cross-architecture conversion |
| 10 | [Zebra-Llama](2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) (NeurIPS'25) | MLA + Mamba2 hybrid composed from a pretrained Transformer | Extreme KV reduction (see [transformer-to-linear-or-hybrid](../transformer-to-linear-or-hybrid/README.md)) |

**Also useful.**
* Other modalities: [MHA2MLA-VLM](2601.11464-mha2mla-vlm-enabling-deepseek-s-economical-multi-head-latent-attention.md) (VLMs), [Whisper-MLA](2603.00563-whisper-mla-reducing-gpu-memory-consumption-of-asr-models-based-on-mha.md) (ASR).
* Domain case study: [YouZhi](2606.05868-youzhi-towards-high-concurrency-financial-llms-via-adaptive-gqa-to-mla.md) (layer-adaptive FreqFold on Ascend).
* Small-model study: [Latent Multi-Head Attention for Small Language Models](2506.09342-latent-multi-head-attention-for-small-language-models.md) (MLA + RoPE at half rank is a Pareto improvement at 30M).

**Recommendation.**
* For serving an existing GQA model at long context, **GQA → MLA (TransMLA + CARE init, 1–6B tokens, ≈50–200 H100-h)**
  pays for itself immediately. It gets 4–10× smaller KV plus DeepSeek's MLA kernels.
* For a lower-effort win, do head-level full → sparse/SWA conversion (RTPurbo/SWAA, ≈10–40 H100-h). This keeps GQA
  kernels.
* For a runtime, MLA support needs both the **absorbed decode path** (latent KV, MQA-style) and the **expanded prefill
  path**. Watch GQLA if you target non-H100 hardware.

## 🏆 Best of the best by impact score (top 10)

1. **[TransMLA: Multi-Head Latent Attention Is All You Need](2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md)** (2025-06) — TransMLA, a framework that seamlessly converts any GQA-based pre-trained model into an MLA-based model, enables direct compatibility with DeepSeek's codebase, allowing these models to fully leverage DeepSeek-specific …  
   _score 10.76 · 28 cites · 69▲ HF · [code](https://github.com/fxmeng/TransMLA)_
2. **[Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps](2605.16928-full-attention-strikes-back-transferring-full-attention-into-sparse-wi.md)** (2026-06) — This work proposes RTPurbo, which retains the full KV cache only for retrieval heads and introduces a lightweight token indexer for sparse attention, and suggests that strong sparse inference can be obtained from …  
   _score 6.65 · 3 cites · 90▲ HF_
3. **[Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](2502.14837-towards-economical-inference-enabling-deepseek-s-multi-head-latent-att.md)** (2025-10) — This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, it removes RoPE from dimensions of queries and keys that …  
   _score 6.07 · Accepted to ACL 2025 · 36 cites · [code](https://github.com/JT-Ushio/MHA2MLA)_
4. **[SWAA: Sliding Window Attention Adaptation for Efficient and Quality Preserving Long Context Processing](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md)** (2026-03) — This work proposes Sliding Window Attention Adaptation (SWAA), a plug and play toolkit of recipes that adapts FA models to SWA without costly pretraining, and systematically combines four core strategies to tackle …  
   _score 4.91 · 2 cites · 21▲ HF · [code](https://github.com/yuyijiong/sliding-window-attention-adaptation) · ~14–36 H100-h_
5. **[Zebra-Llama: Towards Extremely Efficient Hybrid Models](2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md)** (2026-01) — This work introduces a family of 1B, 3B, and 8B hybrid models by combining State Space Models (SSMs) and Multi-head Latent Attention (MLA) layers, using a refined initialization and post-training pipeline to efficiently …  
   _score 4.52 · Neural Information Processing Systems (Neural Inf Process Sy · 17 cites · [code](https://github.com/AMD-AGI/AMD-Hybrid-Models)_
6. **[GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding](2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md)** (2026-07) — Group-Query Latent Attention (GQLA), a minimal modification of MLA whose trained weights expose two algebraically equivalent decoding paths over the same parameters: an MQA-absorb path identical to MLA's, and a GQA path …  
   _score 4.3 · 1 cites · 12▲ HF · [code](https://github.com/MuLabPKU/TransArch)_
7. **[CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention](2603.17946-care-covariance-aware-and-rank-enhanced-decomposition-for-enabling-mul.md)** (2026-03) — CARE is proposed, a Covariance-Aware, Rank-Enhanced MLA conversion pipeline under a fixed KV width that outperforms a uniform-rank SVD baseline on Qwen3-4B/30B-A3B-Instruct-2507 and Llama-3.1-8B/70B-Instruct, reducing …  
   _score 2.44 · Accepted at ICLR 2026 · 3 cites · [code](https://github.com/FutureMLS-Lab/CARE)_
8. **[Efficient Context Scaling with LongCat ZigZag Attention](2512.23966-efficient-context-scaling-with-longcat-zigzag-attention.md)** (2026-01) — LongCat ZigZag Attention (LoZA) is introduced, which is a sparse attention scheme designed to transform any existing full-attention models into sparse versions with rather limited compute budget, enabling efficient …  
   _score 2.42 · 6 cites_
9. **[X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression](2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md)** (2025-09) — X-EcoMLA is proposed to deploy post training distillation to enable the upcycling of Transformer-based attention into an efficient hybrid MLA variant through lightweight post-training adaptation, bypassing the need for …  
   _score 1.53 · 4 cites · [code](https://github.com/AMD-AGI/AMD-Hybrid-Models) · ~91–140 H100-h_
10. **[Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion](2604.05688-attention-editing-a-versatile-framework-for-cross-architecture-attenti.md)** (2026-04) — Attention Editing is presented, a practical framework for converting already-trained large language models (LLMs) with new attention architectures without re-pretraining from scratch, demonstrating that large-scale …  
   _score 0.96 · 2 cites_

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [SWAA: Sliding Window Attention Adaptation for Efficient and Quality Preserving L](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md) | 14 | 36 | H20 | Training of each SWAA configuration takes approximately 12 hours on an 8*H20 GPU server for Qwen3-4B and 30 hours for Qwen3-30B-A3B.… |
| [X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV ](2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md) | 91 | 140 | MI300X | The experimental results show that our proposed method can effectively compress the KV cache while preserving the performance on the benchmarks; specifically, f… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [TransMLA: Multi-Head Latent Attention Is All You Need](2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md) | 2025-06-12 | 10.76 | 28 | 69 |  | [✓](https://github.com/fxmeng/TransMLA) | TransMLA, a framework that seamlessly converts any GQA-based pre-trained model into an MLA-based model, enables direct compatibility with DeepSeek's codebase, … |
| 2 | [Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps](2605.16928-full-attention-strikes-back-transferring-full-attention-into-sparse-wi.md) | 2026-06-08 | 6.65 | 3 | 90 |  |  | This work proposes RTPurbo, which retains the full KV cache only for retrieval heads and introduces a lightweight token indexer for sparse attention, and … |
| 3 | [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](2502.14837-towards-economical-inference-enabling-deepseek-s-multi-head-latent-att.md) | 2025-10-03 | 6.07 | 36 | 0 | Accepted to ACL 2025 | [✓](https://github.com/JT-Ushio/MHA2MLA) | This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for … |
| 4 | [SWAA: Sliding Window Attention Adaptation for Efficient and Quality Preserving Long Context Processing](2512.10411-swaa-sliding-window-attention-adaptation-for-efficient-and-quality-pre.md) | 2026-03-26 | 4.91 | 2 | 21 |  | [✓](https://github.com/yuyijiong/sliding-window-attention-adaptation) | This work proposes Sliding Window Attention Adaptation (SWAA), a plug and play toolkit of recipes that adapts FA models to SWA without costly pretraining, and … |
| 5 | [Zebra-Llama: Towards Extremely Efficient Hybrid Models](2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) | 2026-01-20 | 4.52 | 17 | 0 | Neural Information Processing Systems (N | [✓](https://github.com/AMD-AGI/AMD-Hybrid-Models) | This work introduces a family of 1B, 3B, and 8B hybrid models by combining State Space Models (SSMs) and Multi-head Latent Attention (MLA) layers, using a … |
| 6 | [GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding](2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) | 2026-07-21 | 4.3 | 1 | 12 |  | [✓](https://github.com/MuLabPKU/TransArch) | Group-Query Latent Attention (GQLA), a minimal modification of MLA whose trained weights expose two algebraically equivalent decoding paths over the same … |
| 7 | [CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention](2603.17946-care-covariance-aware-and-rank-enhanced-decomposition-for-enabling-mul.md) | 2026-03-18 | 2.44 | 3 | 0 | Accepted at ICLR 2026 | [✓](https://github.com/FutureMLS-Lab/CARE) | CARE is proposed, a Covariance-Aware, Rank-Enhanced MLA conversion pipeline under a fixed KV width that outperforms a uniform-rank SVD baseline on … |
| 8 | [Efficient Context Scaling with LongCat ZigZag Attention](2512.23966-efficient-context-scaling-with-longcat-zigzag-attention.md) | 2026-01-06 | 2.42 | 6 | 0 |  |  | LongCat ZigZag Attention (LoZA) is introduced, which is a sparse attention scheme designed to transform any existing full-attention models into sparse versions … |
| 9 | [X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression](2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md) | 2025-09-08 | 1.53 | 4 | 0 |  | [✓](https://github.com/AMD-AGI/AMD-Hybrid-Models) | X-EcoMLA is proposed to deploy post training distillation to enable the upcycling of Transformer-based attention into an efficient hybrid MLA variant through … |
| 10 | [Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion](2604.05688-attention-editing-a-versatile-framework-for-cross-architecture-attenti.md) | 2026-04-07 | 0.96 | 2 | 0 |  |  | Attention Editing is presented, a practical framework for converting already-trained large language models (LLMs) with new attention architectures without … |
| 11 | [Whisper-MLA: Reducing GPU Memory Consumption of ASR Models based on MHA2MLA Conversion](2603.00563-whisper-mla-reducing-gpu-memory-consumption-of-asr-models-based-on-mha.md) | 2026-02-28 | 0.85 | 2 | 0 | IEEE International Conference on Acousti |  | Empirical results indicate that applying MLA exclusively to decoder self-attention yields the desired balance between performance and memory efficiency, and … |
| 12 | [NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation](2606.27791-nll-guided-full-attention-layer-selection-for-training-free-sliding-wi.md) | 2026-06-26 | 0.8 | 1 | 0 |  |  | NLL-guided layer selection is proposed, a training-free method that directly measures each layer's importance by computing the negative log-likelihood … |
| 13 | [Latent Multi-Head Attention for Small Language Models](2506.09342-latent-multi-head-attention-for-small-language-models.md) | 2025-06-16 | 0.72 | 3 | 0 |  |  | A key finding is that MLA+RoPE with half-rank latent dimensions achieves a 45% KV-cache memory reduction while incurring only a 0.3% increase in validation … |
| 14 | [MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention across Vision-Language Models](2601.11464-mha2mla-vlm-enabling-deepseek-s-economical-multi-head-latent-attention.md) | 2026-01-16 | 0.7 | 0 | 0 | AAAI Conference on Artificial Intelligen |  | This work presents MHA2MLA-VLM, a parameter-efficient and multimodal-aware framework for converting off-the-shelf VLMs to MLA, and introduces … |
| 15 | [YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition](2606.05868-youzhi-towards-high-concurrency-financial-llms-via-adaptive-gqa-to-mla.md) | 2026-06-04 | 0.0 | 0 | 0 |  |  | YouZhi-LLM, a highly efficient financial LLM empowered by a comprehensive structural transition and training pipeline natively built on the Huawei Ascend … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection](../../kv-cache/low-rank-and-latent/2503.01586-elitekv-scalable-kv-cache-compression-via-rope-frequency-selection-and.md) | KV cache low-rank / latent / head compression | 1.96 |
