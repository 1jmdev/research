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
| **AR → block-diffusion LM** | [[2509.26328\|Fast-dLLM v2]], [[2512.14067\|Efficient-DLM]], [[2512.06776\|NBDiff]], [[2510.06303\|SDAR]], [[2512.15745\|LLaDA2.0]] (100B MoE, 3-phase block-size WSD) | **~1B** (Fast-dLLM v2) to hundreds of B (Dream: 580B) | ~30 → ~17k | Keep causal *across* blocks. Block-wise attention preserves AR weights better than full bidirectional ([[2512.14067]]). |
| **AR → causal parallel decoder** (no bidirectional attention) | [[2512.14681\|Jacobi Forcing]], [[2512.22737\|WeDLM]] (topological reordering, causal dLLM) | on-policy trajectories; small | tens to hundreds | Keeps exact prefix-KV caching, so the runtime needs little change. |
| **AR → dLLM, data-efficient** | [[2606.06712\|OPDLM]] (on-policy distillation, 15–7000× fewer tokens), [[2605.06885\|REPR-ALIGN]] (4× faster adaptation) | ≪1B possible | <30 | Newest (2026); verify on your model. |
| **Transformer → linear / hybrid** | [[2505.03005\|RADLADS]] (Qwen2.5 7B/32B/72B → RWKV-variants), [[2601.22156\|HALO → HypeNet]] (Qwen3 → hybrid with HyPE), [[2505.17272\|Zebra-Llama]] (SSM + MLA hybrid), [[2508.15884\|Jet-Nemotron / PostNAS]] | 0.35–0.7B (RADLADS), 2.3B (HALO), 7–11B (Zebra-Llama) | 12–370 | Layer choice matters: [[2512.20569\|KL-guided layer selection]], [[2606.30562\|FlashMorph]]. |
| **MHA/GQA → MLA** | [[2502.07864\|TransMLA]], [[2503.11132\|X-EcoMLA]], [[2605.15250\|TransGQLA]] | 3.6B tokens for 6.4× KV compression on Llama-3.2-1B (X-EcoMLA: **70 MI300 GPU-h ≈ 90 H100-h**) | ~100–300 | Unlocks DeepSeek MLA kernels (FlashMLA) in vLLM/SGLang. TransMLA: 93% KV compressed, 10.6× at 8K on Llama-2-7B. |
| **Dense → MoE (upcycling)** | [[2502.19261\|Drop-Upcycling]] (partial re-init), [[2604.19835\|Expert Upcycling]] (grow experts during CPT, −32% GPU-h at equal loss), [[2502.03009\|upcycling scaling laws]] | 100B–1T (continued pre-training) | 3k–30k | Upcycling saves compute only below a dense-token threshold ([[2502.03009]]). |
| **Tokenizer swap** | [[2506.03523\|TokAlign]] (~5k steps), [[2503.20083\|ALM cross-tokenizer distillation]], [[2510.21954\|MATT]] ("a few GPU-hours") | ≪1B | 1–30 | Byte-level interfaces ([[2604.07466]]) are the simplest cross-tokenizer KD baseline. |

## AR → diffusion in detail ([folder](../papers/model-conversion/ar-to-diffusion/README.md))

* **Why convert rather than train from scratch?** LLaDA-8B from scratch cost about **0.13M H800 GPU-hours on 2.3T
  tokens** (extracted from [[2502.09992|LLaDA]]'s full text). Dream-7B initialised from Qwen2.5 used 580B tokens.
  Fast-dLLM v2 matches its AR parent with **~1B tokens**, about 500× less data.
* **Best-practice ingredients (2025–26):**
  1. **Block diffusion** ([[2503.09573|BD3-LM]]): causal across blocks, bidirectional inside. This preserves the
     AR prior and enables KV caching.
  2. **Progressive block-size schedule.** Start at block size 1, which *is* AR, and grow it:
     [[2512.06776|NBDiff]] (next-token → next-block) and [[2512.15745|LLaDA2.0]] (warm-up → full-sequence diffusion
     → decay back to small blocks).
  3. **Keep weight distributions close to the AR model**: [[2512.14067|Efficient-DLM]]. Block-wise attention beats
     full bidirectional for retention. Efficient-DLM 8B reports +5.4% accuracy and 4.5× throughput versus Dream-7B.
  4. **Complementary masking and token-level noise rescheduling** (Dream, [[2508.15487]]).
* **Multimodal conversions:** [[2512.15713|DiffusionVL]] (<5% of prior data, 2× faster),
  [[2604.06832|Fast-dVLM]] (6× end-to-end with SGLang + FP8) and SDAR-VL.
* **Serving implication:** converted block-dLLMs run on a normal paged-KV engine with a *block* decode step
  (parallel unmask k tokens, commit, append to KV). See [decoding](decoding.md) for Fast-dLLM caching and D2F.

## Transformer → linear / hybrid in detail ([folder](../papers/model-conversion/transformer-to-linear-or-hybrid/README.md))

Recipe that emerged in 2025 (RADLADS / HALO / Zebra-Llama / LoLCATs lineage):

1. **Attention weight transfer.** Initialise linear-attention Q/K/V (and gates) from softmax weights. [[2503.01496|Liger]]
   repurposes key weights as gates, so no new parameters are needed.
2. **Hidden-state alignment** per layer: match the attention outputs of the teacher.
3. **KL distillation** of logits end-to-end.
4. **Short fine-tune.**
5. **Keep a few full-attention layers**, chosen by a data-driven score ([[2512.20569]], [[2606.30562|FlashMorph]])
   rather than a fixed ratio. Hybrids at 1:3–1:6 full:linear preserve recall; see [attention](attention.md).

Results to calibrate expectations:
* [[2505.03005|RADLADS]]: 350–700M tokens (<0.005% of the teacher's pre-training) for Qwen2.5 7B, 32B and 72B.
* [[2601.22156|HypeNet]]: 2.3B tokens, better length generalisation than the parent Qwen3.
* [[2505.17272|Zebra-Llama]]: KV cache down to 2–4% of the original with 97–100% of zero-shot accuracy, from 7–11B
  tokens and an 8B teacher.
* [[2507.09025|Lizard]]: near-lossless MMLU, +9–24 points over earlier linearisation.
* [[2508.15884|Jet-Nemotron]] (PostNAS): freeze the MLPs, search the attention blocks. 2B model with **53.6×
  generation throughput** versus comparable full-attention models.

## MHA/GQA → MLA ([folder](../papers/model-conversion/attention-conversion/README.md))

TransMLA ([[2502.07864]]) shows that GQA is a special case of MLA, so any GQA checkpoint can be re-parameterised
(RoPE split + low-rank KV joint compression) and then briefly fine-tuned. X-EcoMLA ([[2503.11132]]) adds teacher
distillation for extreme compression: 6.4× KV at equal score with 3.6B tokens and 70 MI300 GPU-hours on
Llama-3.2-1B. Variants exist for VLMs (MHA2MLA-VLM, [[2601.11464]]) and ASR (Whisper-MLA, [[2603.00563]]).
[[2605.15250|GQLA]] keeps both an MLA-absorb path and a GQA path in one set of weights, so it adapts to hardware.

## Dense → MoE ([folder](../papers/model-conversion/dense-to-moe-upcycling/README.md))

* [[2502.19261|Drop-Upcycling]] (ICLR'25): re-initialise part of each copied expert to break symmetry. It beats plain
  sparse upcycling at scale.
* [[2502.03009|Scaling laws for upcycling]] (ICML'25): there is a dense-token budget beyond which upcycling stops
  saving compute versus MoE from scratch.
* Training-free MoE-fication: [[2501.15316|ToMoE]] (dynamic structural pruning into experts) and
  [[2606.01666|DOT-MoE]] (optimal-transport split, keeps 90% quality at 50% active parameters).

## Model merging ([folder](../papers/model-conversion/model-merging/README.md))

* [[2505.12082|Model merging in pre-training]] (ByteDance Seed): merge constant-LR checkpoints to *predict and
  replace* annealing. This is a compute saver for pre-training runs.
* [[2502.02421|Activation-informed merging]] (up to +40% on benchmarks), [[2502.00997|MergeME]] (merging MoEs) and
  [[2505.14136|test-time model merging]].
* Security: [[2505.23561|Merge Hijacking]] (backdoors survive merging).
