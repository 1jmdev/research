# Model conversion: AR → diffusion, Transformer → linear/hybrid, MHA → MLA, dense → MoE (2025–2026)

> Synthesis of [`papers/model-conversion`](../papers/model-conversion/README.md). The key question is how much
> compute a conversion costs, so every leaf README has a **💻 compute table** (auto-extracted, in H100-hours), and
> this page adds token-budget estimates.

## How to read compute costs

Most conversion papers report **training tokens** rather than GPU-hours. A consistent rule of thumb:

```
H100-hours ≈ 6 · N_params · D_tokens / (989e12 FLOP/s · MFU · 3600)
           ≈ N · D / 2.4e17          (MFU ≈ 40%, BF16 dense)
```

Add about 30–50% when a teacher forward pass is needed (distillation). For MoE, use **active** parameters for N.

| Tokens → | 1B | 10B | 100B | 1T |
| --- | ---: | ---: | ---: | ---: |
| 1B model | 4 | 42 | 420 | 4.2k |
| 8B model | 33 | 330 | 3.3k | 33k |
| 32B model | 130 | 1.3k | 13k | 130k |
| 72B model | 300 | 3k | 30k | 300k |

## TL;DR

| Conversion | Best current recipe | Token budget | ≈ H100-h (8B) | Notes |
| --- | --- | --- | --- | --- |
| **AR → block-diffusion LM** | [Fast-dLLM v2](../papers/decoding/diffusion-llm-inference/2509.26328-fast-dllm-v2-efficient-block-diffusion-llm.md), [Efficient-DLM](../papers/model-conversion/ar-to-diffusion/2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md), [NBDiff](../papers/model-conversion/ar-to-diffusion/2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md), [SDAR](../papers/model-conversion/ar-to-diffusion/2510.06303-sdar-a-synergistic-diffusion-autoregression-paradigm-for-scalable-sequ.md), [LLaDA2.0](../papers/decoding/diffusion-language-models/2512.15745-llada2-0-scaling-up-diffusion-language-models-to-100b.md) (100B MoE, 3-phase block-size WSD) | **~1B** (Fast-dLLM v2) to hundreds of B (Dream: 580B) | ~30 → ~17k | Keep causal *across* blocks. Block-wise attention preserves AR weights better than full bidirectional ([Efficient-DLM](../papers/model-conversion/ar-to-diffusion/2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md)). |
| **AR → causal parallel decoder** (no bidirectional attention) | [Jacobi Forcing](../papers/decoding/jacobi-and-parallel-decoding/2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md), [WeDLM](../papers/decoding/diffusion-llm-inference/2512.22737-wedlm-reconciling-diffusion-language-models-with-standard-causal-atten.md) (topological reordering, causal dLLM) | on-policy trajectories; small | tens to hundreds | Keeps exact prefix-KV caching, so the runtime needs little change. |
| **AR → dLLM, data-efficient** | [OPDLM](../papers/model-conversion/ar-to-diffusion/2606.06712-data-efficient-autoregressive-to-diffusion-language-models-via-on-poli.md) (on-policy distillation, 15–7000× fewer tokens), [REPR-ALIGN](../papers/model-conversion/ar-to-diffusion/2605.06885-don-t-retrain-align-adapting-autoregressive-lms-to-diffusion-lms-via-r.md) (4× faster adaptation) | ≪1B possible | <30 | Newest (2026); verify on your model. |
| **Transformer → linear / hybrid** | [RADLADS](../papers/model-conversion/transformer-to-linear-or-hybrid/2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md) (Qwen2.5 7B/32B/72B → RWKV-variants), [HALO → HypeNet](../papers/model-conversion/transformer-to-linear-or-hybrid/2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md) (Qwen3 → hybrid with HyPE), [Zebra-Llama](../papers/model-conversion/attention-conversion/2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) (SSM + MLA hybrid), [Jet-Nemotron / PostNAS](../papers/models-and-architectures/technical-reports/2508.15884-jet-nemotron-efficient-language-model-with-post-neural-architecture-se.md) | 0.35–0.7B (RADLADS), 2.3B (HALO), 7–11B (Zebra-Llama) | 12–370 | Layer choice matters: [KL-guided layer selection](../papers/model-conversion/transformer-to-linear-or-hybrid/2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md), [FlashMorph](../papers/model-conversion/transformer-to-linear-or-hybrid/2606.30562-morphing-into-hybrid-attention-models.md). |
| **MHA/GQA → MLA** | [TransMLA](../papers/model-conversion/attention-conversion/2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md), [X-EcoMLA](../papers/model-conversion/attention-conversion/2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md), [TransGQLA](../papers/model-conversion/attention-conversion/2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) | 3.6B tokens for 6.4× KV compression on Llama-3.2-1B (X-EcoMLA: **70 MI300 GPU-h ≈ 90 H100-h**) | ~100–300 | Unlocks DeepSeek MLA kernels (FlashMLA) in vLLM/SGLang. TransMLA: 93% KV compressed, 10.6× at 8K on Llama-2-7B. |
| **Dense → MoE (upcycling)** | [Drop-Upcycling](../papers/model-conversion/dense-to-moe-upcycling/2502.19261-drop-upcycling-training-sparse-mixture-of-experts-with-partial-re-init.md) (partial re-init), [Expert Upcycling](../papers/model-conversion/dense-to-moe-upcycling/2604.19835-expert-upcycling-shifting-the-compute-efficient-frontier-of-mixture-of.md) (grow experts during CPT, −32% GPU-h at equal loss), [upcycling scaling laws](../papers/model-conversion/dense-to-moe-upcycling/2502.03009-scaling-laws-for-upcycling-mixture-of-experts-language-models.md) | 100B–1T (continued pre-training) | 3k–30k | Upcycling saves compute only below a dense-token threshold ([Scaling Laws for Upcycling Mixture-of-Experts Language Models](../papers/model-conversion/dense-to-moe-upcycling/2502.03009-scaling-laws-for-upcycling-mixture-of-experts-language-models.md)). |
| **Tokenizer swap** | [TokAlign](../papers/model-conversion/tokenizer-and-vocab-transfer/2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md) (~5k steps), [ALM cross-tokenizer distillation](../papers/model-conversion/tokenizer-and-vocab-transfer/2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md), [MATT](../papers/model-conversion/tokenizer-and-vocab-transfer/2510.21954-model-aware-tokenizer-transfer.md) ("a few GPU-hours") | ≪1B | 1–30 | Byte-level interfaces ([Cross-Tokenizer LLM Distillation through a Byte-Level Interface](../papers/model-conversion/tokenizer-and-vocab-transfer/2604.07466-cross-tokenizer-llm-distillation-through-a-byte-level-interface.md)) are the simplest cross-tokenizer KD baseline. |

## AR → diffusion in detail ([folder](../papers/model-conversion/ar-to-diffusion/README.md))

* **Why convert rather than train from scratch?** LLaDA-8B from scratch cost about **0.13M H800 GPU-hours on 2.3T
  tokens** (extracted from [LLaDA](../papers/decoding/diffusion-language-models/2502.09992-large-language-diffusion-models.md)'s full text). Dream-7B initialised from Qwen2.5 used 580B tokens.
  Fast-dLLM v2 matches its AR parent with **~1B tokens**, about 500× less data.
* **Best-practice ingredients (2025–26):**
  1. **Block diffusion** ([BD3-LM](../papers/decoding/diffusion-language-models/2503.09573-block-diffusion-interpolating-between-autoregressive-and-diffusion-lan.md)): causal across blocks, bidirectional inside. This preserves the
     AR prior and enables KV caching.
  2. **Progressive block-size schedule.** Start at block size 1, which *is* AR, and grow it:
     [NBDiff](../papers/model-conversion/ar-to-diffusion/2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md) (next-token → next-block) and [LLaDA2.0](../papers/decoding/diffusion-language-models/2512.15745-llada2-0-scaling-up-diffusion-language-models-to-100b.md) (warm-up → full-sequence diffusion
     → decay back to small blocks).
  3. **Keep weight distributions close to the AR model**: [Efficient-DLM](../papers/model-conversion/ar-to-diffusion/2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md). Block-wise attention beats
     full bidirectional for retention. Efficient-DLM 8B reports +5.4% accuracy and 4.5× throughput versus Dream-7B.
  4. **Complementary masking and token-level noise rescheduling** (Dream, [Dream 7B](../papers/decoding/diffusion-language-models/2508.15487-dream-7b-diffusion-large-language-models.md)).
* **Multimodal conversions:** [DiffusionVL](../papers/model-conversion/ar-to-diffusion/2512.15713-diffusionvl-translating-any-autoregressive-models-into-diffusion-visio.md) (<5% of prior data, 2× faster),
  [Fast-dVLM](../papers/model-conversion/ar-to-diffusion/2604.06832-fast-dvlm-efficient-block-diffusion-vlm-via-direct-conversion-from-aut.md) (6× end-to-end with SGLang + FP8) and SDAR-VL.
* **Serving implication:** converted block-dLLMs run on a normal paged-KV engine with a *block* decode step
  (parallel unmask k tokens, commit, append to KV). See [decoding](decoding.md) for Fast-dLLM caching and D2F.

## Transformer → linear / hybrid in detail ([folder](../papers/model-conversion/transformer-to-linear-or-hybrid/README.md))

Recipe that emerged in 2025 (RADLADS / HALO / Zebra-Llama / LoLCATs lineage):

1. **Attention weight transfer.** Initialise linear-attention Q/K/V (and gates) from softmax weights. [Liger](../papers/model-conversion/transformer-to-linear-or-hybrid/2503.01496-liger-linearizing-large-language-models-to-gated-recurrent-structures.md)
   repurposes key weights as gates, so no new parameters are needed.
2. **Hidden-state alignment** per layer: match the attention outputs of the teacher.
3. **KL distillation** of logits end-to-end.
4. **Short fine-tune.**
5. **Keep a few full-attention layers**, chosen by a data-driven score ([Distilling to Hybrid Attention Models via KL-Guided Layer Selection](../papers/model-conversion/transformer-to-linear-or-hybrid/2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md), [FlashMorph](../papers/model-conversion/transformer-to-linear-or-hybrid/2606.30562-morphing-into-hybrid-attention-models.md))
   rather than a fixed ratio. Hybrids at 1:3–1:6 full:linear preserve recall; see [attention](attention.md).

Results to calibrate expectations:
* [RADLADS](../papers/model-conversion/transformer-to-linear-or-hybrid/2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md): 350–700M tokens (<0.005% of the teacher's pre-training) for Qwen2.5 7B, 32B and 72B.
* [HypeNet](../papers/model-conversion/transformer-to-linear-or-hybrid/2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md): 2.3B tokens, better length generalisation than the parent Qwen3.
* [Zebra-Llama](../papers/model-conversion/attention-conversion/2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md): KV cache down to 2–4% of the original with 97–100% of zero-shot accuracy, from 7–11B
  tokens and an 8B teacher.
* [Lizard](../papers/model-conversion/transformer-to-linear-or-hybrid/2507.09025-lizard-an-efficient-linearization-framework-for-large-language-models.md): near-lossless MMLU, +9–24 points over earlier linearisation.
* [Jet-Nemotron](../papers/models-and-architectures/technical-reports/2508.15884-jet-nemotron-efficient-language-model-with-post-neural-architecture-se.md) (PostNAS): freeze the MLPs, search the attention blocks. 2B model with **53.6×
  generation throughput** versus comparable full-attention models.

## MHA/GQA → MLA ([folder](../papers/model-conversion/attention-conversion/README.md))

TransMLA ([TransMLA](../papers/model-conversion/attention-conversion/2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md)) shows that GQA is a special case of MLA, so any GQA checkpoint can be re-parameterised
(RoPE split + low-rank KV joint compression) and then briefly fine-tuned. X-EcoMLA ([X-EcoMLA](../papers/model-conversion/attention-conversion/2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md)) adds teacher
distillation for extreme compression: 6.4× KV at equal score with 3.6B tokens and 70 MI300 GPU-hours on
Llama-3.2-1B. Variants exist for VLMs (MHA2MLA-VLM, [MHA2MLA-VLM](../papers/model-conversion/attention-conversion/2601.11464-mha2mla-vlm-enabling-deepseek-s-economical-multi-head-latent-attention.md)) and ASR (Whisper-MLA, [Whisper-MLA](../papers/model-conversion/attention-conversion/2603.00563-whisper-mla-reducing-gpu-memory-consumption-of-asr-models-based-on-mha.md)).
[GQLA](../papers/model-conversion/attention-conversion/2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) keeps both an MLA-absorb path and a GQA path in one set of weights, so it adapts to hardware.

## Dense → MoE ([folder](../papers/model-conversion/dense-to-moe-upcycling/README.md))

* [Drop-Upcycling](../papers/model-conversion/dense-to-moe-upcycling/2502.19261-drop-upcycling-training-sparse-mixture-of-experts-with-partial-re-init.md) (ICLR'25): re-initialise part of each copied expert to break symmetry. It beats plain
  sparse upcycling at scale.
* [Scaling laws for upcycling](../papers/model-conversion/dense-to-moe-upcycling/2502.03009-scaling-laws-for-upcycling-mixture-of-experts-language-models.md) (ICML'25): there is a dense-token budget beyond which upcycling stops
  saving compute versus MoE from scratch.
* Training-free MoE-fication: [ToMoE](../papers/model-conversion/dense-to-moe-upcycling/2501.15316-tomoe-converting-dense-large-language-models-to-mixture-of-experts-thr.md) (dynamic structural pruning into experts) and
  [DOT-MoE](../papers/model-conversion/dense-to-moe-upcycling/2606.01666-dot-moe-differentiable-optimal-transport-for-moefication.md) (optimal-transport split, keeps 90% quality at 50% active parameters).

## Model merging ([folder](../papers/model-conversion/model-merging/README.md))

* [Model merging in pre-training](../papers/model-conversion/model-merging/2505.12082-model-merging-in-pre-training-of-large-language-models.md) (ByteDance Seed): merge constant-LR checkpoints to *predict and
  replace* annealing. This is a compute saver for pre-training runs.
* [Activation-informed merging](../papers/model-conversion/model-merging/2502.02421-activation-informed-merging-of-large-language-models.md) (up to +40% on benchmarks), [MergeME](../papers/model-conversion/model-merging/2502.00997-mergeme-model-merging-techniques-for-homogeneous-and-heterogeneous-moe.md) (merging MoEs) and
  [test-time model merging](../papers/model-conversion/model-merging/2505.14136-local-mixtures-of-experts-essentially-free-test-time-training-via-mode.md).
* Security: [Merge Hijacking](https://arxiv.org/abs/2505.23561) (backdoors survive merging).
