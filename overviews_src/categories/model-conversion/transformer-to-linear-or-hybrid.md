**Verdict.** Converting a pretrained Transformer into a linear-attention or SSM **hybrid** is cheap and works. Converting
it into a **pure** linear model is cheap but still lossy on recall-heavy and long-context tasks. 2025–26 fixed three
things:

1. **Staged distillation recipe** (MOHAWK → RADLADS/HALO/Llamba):
   * **Stage 1: attention-output (hidden-state) alignment**, per layer, with MLPs frozen and copied;
   * **Stage 2: end-to-end logit KD**;
   * **Stage 3: short fine-tune / long-context stage.**

   Initialize Q/K/V/O from the teacher. The whole recipe takes **0.35–3B tokens** at 7–72B.
2. **Which layers keep full attention.** This decides quality far more than the linear block itself. Heuristic even
   spacing is beaten by KL-guided layer importance ([[2512.20569]]), joint optimization ([[2606.30562|FlashMorph]]: 20M
   tokens, 2.1 GPU-hours) and redundancy detection via self-speculation ([[2505.22135|RAD]]). Retrieval-critical heads
   matter most ([[2602.11374|retrieval-aware distillation]]). Keep **~1/8–1/4 of the layers as full attention**.
3. **Better student blocks.**
   * Gated DeltaNet / RWKV-7 / mLSTM beat plain linear attention.
   * Add SWA + sinks in the same layer (Liger, Lizard, xLSTM hybrid).
   * Use hybrid-aware position encoding (HALO's HyPE: RoPE in attention, NoPE-style in RNN).
   * Don't let the SWA branch dominate ([[2510.05901|component imbalance]]).

**Evaluate generation, not log-likelihood.** Distilled hybrids look fine on multiple-choice perplexity ranking and then
fail at generation ([[2603.26556|When Perplexity Lies]]).

### Conversion cost table (hand-checked)

"Reported" means from the paper. "Estimate" means 6·N·D at 40% H100 MFU (≈1.42·10¹⁸ FLOP per H100-h), which
**overestimates** stages with frozen MLPs.

| Method | Teacher → student | Tokens | H100-h | Basis |
| --- | --- | ---: | ---: | --- |
| [[2606.30562\|FlashMorph]] (layer selection only) | → hybrid | 20M | **~2** | reported: 2.1 GPU-h |
| [[2503.01496\|Liger]] (ICML'25) | Llama-3-8B / Mistral → GLA-style + SWA | 0.02B (LoRA) | **~1–5** | estimate |
| [[2507.09025\|Lizard]] (ACL'26) | Llama-3-8B → gated linear + SWA + meta-sinks | ~0.7B | **~25** | estimate |
| [[2505.03005\|RADLADS]] | Qwen2.5 7B / 32B / 72B → RWKV-6/7 variants | 0.35–0.7B | **~20 (7B) … ~600–800 (72B)** | 72B reported "< $2,000"; 7B estimate |
| [[2501.15570\|ARWKV]] | Qwen2.5-32B → RWKV-6 (QRWKV6) | — | **~130** | reported: 16×MI300X × 8 h |
| [[2601.22156\|HALO / HypeNet]] | Qwen3 → attention-RNN hybrid | 2.3B | **~80** (8B) | estimate |
| [[2603.15590\|xLSTM hybrid distill]] | 8B-class → mLSTM + SWA/sink | ~2B (+5B per expert) | **~110** (+ experts) | estimate |
| [[2505.17272\|Zebra-Llama]] | Llama-3.x 1–8B → MLA + Mamba2 hybrid | 6.8B SFT + ILD | **~100–400** | 1 node 8×MI300 |
| [[2502.14458\|Llamba]] | Llama-3.1-8B → Mamba-2 (MOHAWK) | 12B | **~400** | estimate |
| [[2605.08301\|Priming]] | Qwen3-8B → hybrid SSM | <150B | **~5K** | estimate |
| [[2508.15884\|Jet-Nemotron]] | Qwen2.5-1.5B → PostNAS hybrid (JetBlock) | 400B | **~8.2K training + ~10K search** | reported (Table: H100 GPU-hours) |

Pretraining an 8B hybrid from scratch costs ≳100K H100-h. Conversion is **100–10,000× cheaper**.

### Hand ranking

| # | Paper | Target | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.03005\|RADLADS]] | Pure RWKV-6/7 variants | 3-step protocol (attention hidden-state alignment → KL distillation → fine-tune) with teacher-initialized projections; two simplified RWKV blocks | **7B/32B/72B converted** with 350–700M tokens; 72B < $2K; near-teacher quality. Most practical pure-linear recipe |
| 2 | [[2601.22156\|HALO + HypeNet]] | Attention–RNN hybrid | <3B-token cross-architecture distillation; **HyPE** position scheme + attention scaling for length generalization | Converted Qwen3 hybrids keep long-context performance, where most distilled hybrids fail |
| 3 | [[2508.15884\|Jet-Nemotron]] (NVIDIA) | Hybrid via PostNAS | Start from a pretrained model with **frozen MLPs**; search full-attention placement, then the linear block (JetBlock, dynamic conv), then hardware-aware hyperparameters | 2B beats Qwen3-1.7B on MMLU-Pro at **47× generation throughput** (64K ctx, H100) |
| 4 | [[2503.01496\|Liger]] (ICML'25) | Gated recurrent + SWA | Reuse key-projection weights to build gates (**no new params**); intra-layer hybrid attention; LoRA | 93% of Llama-3-8B recovered with 0.02B tokens |
| 5 | [[2512.20569\|KL-guided layer selection]] / [[2606.30562\|FlashMorph]] | Hybrid | Choose which layers stay softmax by the KL they cause (or by joint optimization of a relaxed mixture) | Big quality gains at the same attention ratio; FlashMorph costs ~2 GPU-h |
| 6 | [[2502.14458\|Llamba]] (Cartesia) | Pure Mamba-2 | MOHAWK at scale (1B/3B/8B), 8–12B tokens | Higher throughput and batch sizes at comparable benchmarks; on-device friendly |
| 7 | [[2603.15590\|Effective distillation to hybrid xLSTM]] | mLSTM + SWA/sink | "Lossless" defined as win-and-tie vs the teacher; per-head output gates; expert distillation + merging | Closest to lossless among distilled students |
| 8 | [[2507.09025\|Lizard]] (ACL'26) | Gated linear + SWA + meta-memory sinks | RoPE-free linear branch trained to match RoPE softmax outputs | +9.4–24.5 MMLU over prior linearizations; 50%-attention hybrid ≈ teacher |
| 9 | [[2605.08301\|Priming]] | Hybrid SSM at 8B+ | Initialize SSM layers from attention weights + two short phases (<0.5% of the pretraining budget); P2P sequence parallelism for SSM training | Opens the hybrid design space without pretraining |
| 10 | [[2505.17272\|Zebra-Llama]] | MLA + Mamba2 | Mix MLA (compressed KV) and Mamba2 layers with SMART layer selection | KV cache down to a few % of the original at small quality loss |
| 11 | [[2502.13145\|mmMamba]] | Multimodal SSM | Three-stage progressive distillation from a decoder-only VLM | Linear-complexity VLM from academic compute |
| 12 | [[2603.26556\|When Perplexity Lies]] | Evaluation | Generation-based evaluation exposes distillation failures hidden by likelihood ranking | **Adopt this eval protocol** |

**Also useful.**
* Initialization: [[2606.16429|Taylor-Calibrate]] (principled GDN init), [[2507.03340]] (degrees of freedom for
  linear-attention feature maps), [[2607.07706]] (analysis-driven linearization).
* Mamba students: [[2510.19266|Attention Bridge]], [[2604.14191|Attention → Mamba recipe]].
* Other: [[2509.18467|LAWCAT]] (conv across tokens), [[2506.09316|DSLA]] (dual-state), [[2604.24715]] (long-context-aware
  upcycling), [[2608.02689|Stuck on "A"]] (interface injury in KDA linearization at 0.6B).
* Linear attention in dLLMs: [[2608.06628]].
* Reasoning in distilled SSMs: [[2504.10449|M1]] (Mamba reasoning via distillation + RL).

**Recommendation.**
* Pick a **GDN / RWKV-7 / Mamba-2 hybrid with ~1:3–1:7 full-attention layers**.
* Place the full-attention layers by KL-guided or joint selection, not uniformly.
* Run the RADLADS/HALO staged recipe on **1–3B tokens (≈50–500 H100-h at 7–8B)**.
* Finish with a long-context stage and a generation-based eval.
* For a runtime this means supporting **mixed layer types in one model**: a paged KV cache for attention layers and a
  fixed-size state cache for recurrent layers, with prefix-cache snapshots of the recurrent state.
