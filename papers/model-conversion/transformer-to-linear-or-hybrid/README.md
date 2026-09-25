# Transformer → linear / SSM / hybrid conversion (linearization)

Distilling or converting pretrained attention models into linear-attention, Mamba or hybrid models (LoLCATs, Mamba-in-Llama, Liger, RAD).

**32 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Converting a pretrained Transformer into a linear-attention or SSM **hybrid** is cheap and works. Converting
it into a **pure** linear model is cheap but still lossy on recall-heavy and long-context tasks. 2025–26 fixed three
things:

1. **Staged distillation recipe** (MOHAWK → RADLADS/HALO/Llamba):
   * **Stage 1: attention-output (hidden-state) alignment**, per layer, with MLPs frozen and copied;
   * **Stage 2: end-to-end logit KD**;
   * **Stage 3: short fine-tune / long-context stage.**

   Initialize Q/K/V/O from the teacher. The whole recipe takes **0.35–3B tokens** at 7–72B.
2. **Which layers keep full attention.** This decides quality far more than the linear block itself. Heuristic even
   spacing is beaten by KL-guided layer importance ([Distilling to Hybrid Attention Models via KL-Guided Layer Selection](2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md)), joint optimization ([FlashMorph](2606.30562-morphing-into-hybrid-attention-models.md): 20M
   tokens, 2.1 GPU-hours) and redundancy detection via self-speculation ([RAD](2505.22135-rad-redundancy-aware-distillation-for-hybrid-models-via-self-speculati.md)). Retrieval-critical heads
   matter most ([retrieval-aware distillation](2602.11374-retrieval-aware-distillation-for-transformer-ssm-hybrids.md)). Keep **~1/8–1/4 of the layers as full attention**.
3. **Better student blocks.**
   * Gated DeltaNet / RWKV-7 / mLSTM beat plain linear attention.
   * Add SWA + sinks in the same layer (Liger, Lizard, xLSTM hybrid).
   * Use hybrid-aware position encoding (HALO's HyPE: RoPE in attention, NoPE-style in RNN).
   * Don't let the SWA branch dominate ([component imbalance](2510.05901-untangling-component-imbalance-in-hybrid-linear-attention-conversion-m.md)).

**Evaluate generation, not log-likelihood.** Distilled hybrids look fine on multiple-choice perplexity ranking and then
fail at generation ([When Perplexity Lies](2603.26556-when-perplexity-lies-generation-focused-distillation-of-hybrid-sequenc.md)).

### Conversion cost table (hand-checked)

"Reported" means from the paper. "Estimate" means 6·N·D at 40% H100 MFU (≈1.42·10¹⁸ FLOP per H100-h), which
**overestimates** stages with frozen MLPs.

| Method | Teacher → student | Tokens | H100-h | Basis |
| --- | --- | ---: | ---: | --- |
| [FlashMorph](2606.30562-morphing-into-hybrid-attention-models.md) (layer selection only) | → hybrid | 20M | **~2** | reported: 2.1 GPU-h |
| [Liger](2503.01496-liger-linearizing-large-language-models-to-gated-recurrent-structures.md) (ICML'25) | Llama-3-8B / Mistral → GLA-style + SWA | 0.02B (LoRA) | **~1–5** | estimate |
| [Lizard](2507.09025-lizard-an-efficient-linearization-framework-for-large-language-models.md) (ACL'26) | Llama-3-8B → gated linear + SWA + meta-sinks | ~0.7B | **~25** | estimate |
| [RADLADS](2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md) | Qwen2.5 7B / 32B / 72B → RWKV-6/7 variants | 0.35–0.7B | **~20 (7B) … ~600–800 (72B)** | 72B reported "< $2,000"; 7B estimate |
| [ARWKV](2501.15570-arwkv-pretrain-is-not-what-we-need-an-rnn-attention-based-language-mod.md) | Qwen2.5-32B → RWKV-6 (QRWKV6) | — | **~130** | reported: 16×MI300X × 8 h |
| [HALO / HypeNet](2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md) | Qwen3 → attention-RNN hybrid | 2.3B | **~80** (8B) | estimate |
| [xLSTM hybrid distill](2603.15590-effective-distillation-to-hybrid-xlstm-architectures.md) | 8B-class → mLSTM + SWA/sink | ~2B (+5B per expert) | **~110** (+ experts) | estimate |
| [Zebra-Llama](../attention-conversion/2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) | Llama-3.x 1–8B → MLA + Mamba2 hybrid | 6.8B SFT + ILD | **~100–400** | 1 node 8×MI300 |
| [Llamba](2502.14458-llamba-scaling-distilled-recurrent-models-for-efficient-language-proce.md) | Llama-3.1-8B → Mamba-2 (MOHAWK) | 12B | **~400** | estimate |
| [Priming](2605.08301-priming-hybrid-state-space-models-from-pre-trained-transformers.md) | Qwen3-8B → hybrid SSM | <150B | **~5K** | estimate |
| [Jet-Nemotron](../../models-and-architectures/technical-reports/2508.15884-jet-nemotron-efficient-language-model-with-post-neural-architecture-se.md) | Qwen2.5-1.5B → PostNAS hybrid (JetBlock) | 400B | **~8.2K training + ~10K search** | reported (Table: H100 GPU-hours) |

Pretraining an 8B hybrid from scratch costs ≳100K H100-h. Conversion is **100–10,000× cheaper**.

### Hand ranking

| # | Paper | Target | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [RADLADS](2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md) | Pure RWKV-6/7 variants | 3-step protocol (attention hidden-state alignment → KL distillation → fine-tune) with teacher-initialized projections; two simplified RWKV blocks | **7B/32B/72B converted** with 350–700M tokens; 72B < $2K; near-teacher quality. Most practical pure-linear recipe |
| 2 | [HALO + HypeNet](2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md) | Attention–RNN hybrid | <3B-token cross-architecture distillation; **HyPE** position scheme + attention scaling for length generalization | Converted Qwen3 hybrids keep long-context performance, where most distilled hybrids fail |
| 3 | [Jet-Nemotron](../../models-and-architectures/technical-reports/2508.15884-jet-nemotron-efficient-language-model-with-post-neural-architecture-se.md) (NVIDIA) | Hybrid via PostNAS | Start from a pretrained model with **frozen MLPs**; search full-attention placement, then the linear block (JetBlock, dynamic conv), then hardware-aware hyperparameters | 2B beats Qwen3-1.7B on MMLU-Pro at **47× generation throughput** (64K ctx, H100) |
| 4 | [Liger](2503.01496-liger-linearizing-large-language-models-to-gated-recurrent-structures.md) (ICML'25) | Gated recurrent + SWA | Reuse key-projection weights to build gates (**no new params**); intra-layer hybrid attention; LoRA | 93% of Llama-3-8B recovered with 0.02B tokens |
| 5 | [KL-guided layer selection](2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md) / [FlashMorph](2606.30562-morphing-into-hybrid-attention-models.md) | Hybrid | Choose which layers stay softmax by the KL they cause (or by joint optimization of a relaxed mixture) | Big quality gains at the same attention ratio; FlashMorph costs ~2 GPU-h |
| 6 | [Llamba](2502.14458-llamba-scaling-distilled-recurrent-models-for-efficient-language-proce.md) (Cartesia) | Pure Mamba-2 | MOHAWK at scale (1B/3B/8B), 8–12B tokens | Higher throughput and batch sizes at comparable benchmarks; on-device friendly |
| 7 | [Effective distillation to hybrid xLSTM](2603.15590-effective-distillation-to-hybrid-xlstm-architectures.md) | mLSTM + SWA/sink | "Lossless" defined as win-and-tie vs the teacher; per-head output gates; expert distillation + merging | Closest to lossless among distilled students |
| 8 | [Lizard](2507.09025-lizard-an-efficient-linearization-framework-for-large-language-models.md) (ACL'26) | Gated linear + SWA + meta-memory sinks | RoPE-free linear branch trained to match RoPE softmax outputs | +9.4–24.5 MMLU over prior linearizations; 50%-attention hybrid ≈ teacher |
| 9 | [Priming](2605.08301-priming-hybrid-state-space-models-from-pre-trained-transformers.md) | Hybrid SSM at 8B+ | Initialize SSM layers from attention weights + two short phases (<0.5% of the pretraining budget); P2P sequence parallelism for SSM training | Opens the hybrid design space without pretraining |
| 10 | [Zebra-Llama](../attention-conversion/2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) | MLA + Mamba2 | Mix MLA (compressed KV) and Mamba2 layers with SMART layer selection | KV cache down to a few % of the original at small quality loss |
| 11 | [mmMamba](2502.13145-multimodal-mamba-decoder-only-multimodal-state-space-model-via-quadrat.md) | Multimodal SSM | Three-stage progressive distillation from a decoder-only VLM | Linear-complexity VLM from academic compute |
| 12 | [When Perplexity Lies](2603.26556-when-perplexity-lies-generation-focused-distillation-of-hybrid-sequenc.md) | Evaluation | Generation-based evaluation exposes distillation failures hidden by likelihood ranking | **Adopt this eval protocol** |

**Also useful.**
* Initialization: [Taylor-Calibrate](2606.16429-taylor-calibrate-principled-initialization-for-hybrid-linear-attention.md) (principled GDN init), [Degrees of Freedom for Linear Attention](2507.03340-degrees-of-freedom-for-linear-attention-distilling-softmax-attention-w.md) (degrees of freedom for
  linear-attention feature maps), [The Key to Going Linear](2607.07706-the-key-to-going-linear-analysis-driven-transformer-linearization.md) (analysis-driven linearization).
* Mamba students: [Attention Bridge](2510.19266-data-efficient-any-transformer-to-mamba-distillation-via-attention-bri.md), [Attention → Mamba recipe](2604.14191-attention-to-mamba-a-recipe-for-cross-architecture-distillation.md).
* Other: [LAWCAT](2509.18467-lawcat-efficient-distillation-from-quadratic-to-linear-attention-with.md) (conv across tokens), [DSLA](2506.09316-on-the-fly-adaptive-distillation-of-transformer-to-dual-state-linear-a.md) (dual-state), [Long-Context Aware Upcycling](2604.24715-long-context-aware-upcycling-a-new-frontier-for-hybrid-llm-scaling.md) (long-context-aware
  upcycling), [Stuck on "A"](2608.02689-stuck-on-a-diagnosing-and-repairing-interface-injury-in-attention-to-k.md) (interface injury in KDA linearization at 0.6B).
* Linear attention in dLLMs: [Retrofitting Linear Attention into Diffusion Language Models](2608.06628-retrofitting-linear-attention-into-diffusion-language-models.md).
* Reasoning in distilled SSMs: [M1](../../attention/state-space-and-recurrent/2504.10449-m1-towards-scalable-test-time-compute-with-mamba-reasoning-models.md) (Mamba reasoning via distillation + RL).

**Recommendation.**
* Pick a **GDN / RWKV-7 / Mamba-2 hybrid with ~1:3–1:7 full-attention layers**.
* Place the full-attention layers by KL-guided or joint selection, not uniformly.
* Run the RADLADS/HALO staged recipe on **1–3B tokens (≈50–500 H100-h at 7–8B)**.
* Finish with a long-context stage and a generation-based eval.
* For a runtime this means supporting **mixed layer types in one model**: a paged KV cache for attention layers and a
  fixed-size state cache for recurrent layers, with prefix-cache snapshots of the recurrent state.

## 🏆 Best of the best by impact score (top 10)

1. **[RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale](2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md)** (2026-01) — RADLADS is presented, a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant architectures, and models converted from popular Qwen2.5 open …  
   _score 10.26 · 20 cites · 36▲ HF · [code](https://huggingface.co/collections/recursal)_
2. **[Liger: Linearizing Large Language Models to Gated Recurrent Structures](2503.01496-liger-linearizing-large-language-models-to-gated-recurrent-structures.md)** (2025-05) — Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters, and repurposes the pretrained key matrix weights to construct diverse gating mechanisms, …  
   _score 10.2 · Accepted by ICML 2025 · 28 cites · 18▲ HF · [code](https://github.com/OpenSparseLLMs/Linearization)_
3. **[Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation](2502.13145-multimodal-mamba-decoder-only-multimodal-state-space-model-via-quadrat.md)** (2025-03) — This work proposes mmMamba, a framework for developing linear-complexity native multimodal state space models through progressive distillation from existing MLLMs using moderate academic computational resources, and an …  
   _score 8.2 · 12 cites · 38▲ HF · [code](https://github.com/hustvl/mmMamba)_
4. **[Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts](2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md)** (2026-01) — HypeNet is presented, a hybrid architecture with superior length generalization enabled by a novel position encoding scheme (named HyPE) and various architectural modifications that achieves performance comparable to …  
   _score 7.9 · 14 cites · 15▲ HF · [code](https://github.com/thunlp/hybrid-linear-attention)_
5. **[Lizard: An Efficient Linearization Framework for Large Language Models](2507.09025-lizard-an-efficient-linearization-framework-for-large-language-models.md)** (2026-04) — Lizard is a linearization framework that transforms pretrained Transformer-based Large Language Models into subquadratic architectures that achieves near-lossless recovery of its teacher model's performance, …  
   _score 7.12 · ACL 2026 · 8 cites · 19▲ HF · [code](https://github.com/EleutherAI/lm-evaluation-harness)_
6. **[ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer](2501.15570-arwkv-pretrain-is-not-what-we-need-an-rnn-attention-based-language-mod.md)** (2025-01) — This series of models distilled from Qwen 2.5, based on pure native RWKV-7 attention, which aims to make RNN more expressive and demonstrates state tracking ability beyond transformers, is introduced.  
   _score 7.11 · 9 cites · 24▲ HF · [code](https://github.com/yynil/RWKVInside) · ~166 H100-h_
7. **[Effective Distillation to Hybrid xLSTM Architectures](2603.15590-effective-distillation-to-hybrid-xlstm-architectures.md)** (2026-07) — This work introduces an effective distillation pipeline for xLSTM-based students, and proposes an additional merging stage, where individually linearized experts are combined into a single model.  
   _score 5.84 · 3 cites · 33▲ HF_
8. **[Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing](2502.14458-llamba-scaling-distilled-recurrent-models-for-efficient-language-proce.md)** (2025-02) — This work introduces Llamba, a family of efficient recurrent language models distilled from Llama-3.x into the Mamba architecture, which achieves higher inference throughput and handle significantly larger batch sizes …  
   _score 5.81 · 33 cites · [code](https://github.com/cartesia-ai/edge)_
9. **[Distilling to Hybrid Attention Models via KL-Guided Layer Selection](2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md)** (2025-12) — A simple and efficient recipe for layer selection that uses layer importance scores derived from a small amount of training on generic text data is described, which is more effective than existing approaches for layer …  
   _score 5.09 · 15 cites · [code](https://github.com/fla-org/hybrid-distillation)_
10. **[Morphing into Hybrid Attention Models](2606.30562-morphing-into-hybrid-attention-models.md)** (2026-06) — This work forms hybrid layer selection as a budget-constrained subset optimization problem, and proposes FlashMorph (Fast LAyer Selection for Hybrid MORPHing), an effective, efficient and scalable layer selection method …  
   _score 4.87 · 0 cites · 49▲ HF · [code](https://github.com/LanDisen/FlashMorph) · ~2.1 H100-h_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge](2510.19266-data-efficient-any-transformer-to-mamba-distillation-via-attention-bri.md)** (2026-09-16; 0▲, 3 cites) — Cross-architecture distillation via Attention Bridge (CAB) is proposed, a distillation framework that transfers attention-related representations from Transformer teachers to …
- **[When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models](2603.26556-when-perplexity-lies-generation-focused-distillation-of-hybrid-sequenc.md)** (2026-09-17; 0▲, 1 cites) — GenDistill, a multi-stage pipeline for distilling a pretrained Transformer into an efficient Hybrid Kimi Delta Attention (Hybrid-KDA) student, finds that log-likelihood-based …
- **[Stuck on "A": Diagnosing and Repairing Interface Injury in Attention-to-KDA Linearization of a 0.6B Language Model](2608.02689-stuck-on-a-diagnosing-and-repairing-interface-injury-in-attention-to-k.md)** (2026-08-03; 0▲, 0 cites) — The engineering lessons -- including an FP32-master failure mode in which bf16 optimizer updates are silently swallowed -- that made convergence possible at this budget are …
- **[Retrofitting Linear Attention into Diffusion Language Models](2608.06628-retrofitting-linear-attention-into-diffusion-language-models.md)** (2026-08-06; 0▲, 0 cites) — This work introduces block-hybrid attention, which retains exact softmax attention within the active denoising block while applying linear attention over previous blocks, and …
- **[The Key to Going Linear: Analysis-Driven Transformer Linearization](2607.07706-the-key-to-going-linear-analysis-driven-transformer-linearization.md)** (2026-07-08; 0▲, 0 cites) — It is shown that softmax relies on key-dependent, rank-1 orthogonal projections, elucidating why delta-style networks outperform purely gated accumulation and introducing …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [Morphing into Hybrid Attention Models](2606.30562-morphing-into-hybrid-attention-models.md) | 2.1 | 2.1 | unspecified | FlashMorph uses only 20M tokens for hybrid layer selection, requiring 2.5\times 10^{17} FLOPs and 2.1 GPU hours.… |
| [ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born ](2501.15570-arwkv-pretrain-is-not-what-we-need-an-rnn-attention-based-language-mod.md) | 166 | 166 | MI300X | We work with QRWK 32B † https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1 based on RWKV-6 architecture, another approach that reduces the entire … |
| [RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decodi](2505.22135-rad-redundancy-aware-distillation-for-hybrid-models-via-self-speculati.md) | 184 | 184 | A100 | The training settings for the hybrid models are as follows (training takes 2-3 days on 8×A100 GPUs): Using the base model Llama3.2-3B-Instruct as the teacher \m… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale](2505.03005-radlads-rapid-attention-distillation-to-linear-attention-decoders-at-s.md) | 2026-01-22 | 10.26 | 20 | 36 |  | [✓](https://huggingface.co/collections/recursal) | RADLADS is presented, a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant … |
| 2 | [Liger: Linearizing Large Language Models to Gated Recurrent Structures](2503.01496-liger-linearizing-large-language-models-to-gated-recurrent-structures.md) | 2025-05-07 | 10.2 | 28 | 18 | Accepted by ICML 2025 | [✓](https://github.com/OpenSparseLLMs/Linearization) | Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters, and repurposes the pretrained key … |
| 3 | [Multimodal Mamba: Decoder-only Multimodal State Space Model via Quadratic to Linear Distillation](2502.13145-multimodal-mamba-decoder-only-multimodal-state-space-model-via-quadrat.md) | 2025-03-18 | 8.2 | 12 | 38 |  | [✓](https://github.com/hustvl/mmMamba) | This work proposes mmMamba, a framework for developing linear-complexity native multimodal state space models through progressive distillation from existing … |
| 4 | [Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Cont](2601.22156-hybrid-linear-attention-done-right-efficient-distillation-and-effectiv.md) | 2026-01-29 | 7.9 | 14 | 15 |  | [✓](https://github.com/thunlp/hybrid-linear-attention) | HypeNet is presented, a hybrid architecture with superior length generalization enabled by a novel position encoding scheme (named HyPE) and various … |
| 5 | [Lizard: An Efficient Linearization Framework for Large Language Models](2507.09025-lizard-an-efficient-linearization-framework-for-large-language-models.md) | 2026-04-18 | 7.12 | 8 | 19 | ACL 2026 | [✓](https://github.com/EleutherAI/lm-evaluation-harness) | Lizard is a linearization framework that transforms pretrained Transformer-based Large Language Models into subquadratic architectures that achieves … |
| 6 | [ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer](2501.15570-arwkv-pretrain-is-not-what-we-need-an-rnn-attention-based-language-mod.md) | 2025-01-26 | 7.11 | 9 | 24 |  | [✓](https://github.com/yynil/RWKVInside) | This series of models distilled from Qwen 2.5, based on pure native RWKV-7 attention, which aims to make RNN more expressive and demonstrates state tracking … |
| 7 | [Effective Distillation to Hybrid xLSTM Architectures](2603.15590-effective-distillation-to-hybrid-xlstm-architectures.md) | 2026-07-06 | 5.84 | 3 | 33 |  |  | This work introduces an effective distillation pipeline for xLSTM-based students, and proposes an additional merging stage, where individually linearized … |
| 8 | [Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing](2502.14458-llamba-scaling-distilled-recurrent-models-for-efficient-language-proce.md) | 2025-02-23 | 5.81 | 33 | 0 |  | [✓](https://github.com/cartesia-ai/edge) | This work introduces Llamba, a family of efficient recurrent language models distilled from Llama-3.x into the Mamba architecture, which achieves higher … |
| 9 | [Distilling to Hybrid Attention Models via KL-Guided Layer Selection](2512.20569-distilling-to-hybrid-attention-models-via-kl-guided-layer-selection.md) | 2025-12-23 | 5.09 | 15 | 0 |  | [✓](https://github.com/fla-org/hybrid-distillation) | A simple and efficient recipe for layer selection that uses layer importance scores derived from a small amount of training on generic text data is described, … |
| 10 | [Morphing into Hybrid Attention Models](2606.30562-morphing-into-hybrid-attention-models.md) | 2026-06-29 | 4.87 | 0 | 49 |  | [✓](https://github.com/LanDisen/FlashMorph) | This work forms hybrid layer selection as a budget-constrained subset optimization problem, and proposes FlashMorph (Fast LAyer Selection for Hybrid MORPHing), … |
| 11 | [TPTT: Transforming Pretrained Transformers into Titans](2506.17671-tptt-transforming-pretrained-transformers-into-titans.md) | 2025-08-31 | 3.07 | 1 | 5 |  | [✓](https://github.com/fabienfrfr/tptt) | Experiments on models with approximately 1 billion parameters suggest potential improvements in both efficiency and accuracy compared to baseline models, and … |
| 12 | [Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge](2510.19266-data-efficient-any-transformer-to-mamba-distillation-via-attention-bri.md) | 2026-09-16 | 3.05 | 3 | 0 |  |  | Cross-architecture distillation via Attention Bridge (CAB) is proposed, a distillation framework that transfers attention-related representations from … |
| 13 | [TransMamba: Fast Universal Architecture Adaption from Transformers to Mamba](2502.15130-transmamba-fast-universal-architecture-adaption-from-transformers-to-m.md) | 2025-10-09 | 2.6 | 10 | 0 |  | [✓](https://github.com/chen-xw/TransMamba-main) | A cross-architecture knowledge transfer paradigm, termed TransMamba, that facilitates the reuse of Transformer pre-trained knowledge and proposes a two-stage … |
| 14 | [LAWCAT: Efficient Distillation from Quadratic to Linear Attention with Convolution across Tokens for Long Cont](2509.18467-lawcat-efficient-distillation-from-quadratic-to-linear-attention-with.md) | 2025-11-04 | 2.55 | 5 | 0 | Conference on Empirical Methods in Natur | [✓](https://github.com/zeyuliu1037/LAWCAT) | LAWCAT (Linear Attention with Convolution Across Time), a novel linearization framework designed to efficiently transfer the capabilities of pre-trained … |
| 15 | [RAD: Redundancy-Aware Distillation for Hybrid Models via Self-Speculative Decoding](2505.22135-rad-redundancy-aware-distillation-for-hybrid-models-via-self-speculati.md) | 2025-05-28 | 2.35 | 9 | 0 |  |  | RAD (Redundancy-Aware Distillation), a novel framework that uses self-speculative decoding as a diagnostic tool to identify redundant attention layers within … |
| 16 | [Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation](2606.16429-taylor-calibrate-principled-initialization-for-hybrid-linear-attention.md) | 2026-06-15 | 2.29 | 0 | 4 |  | [✓](https://github.com/FutureMLS-Lab/Taylor-Calibrate) | Taylor-Calibrate is proposed, a lightweight initialization method for hybrid GDN students that gives substantially stronger zero-shot students, with up to an … |
| 17 | [On-the-Fly Adaptive Distillation of Transformer to Dual-State Linear Attention](2506.09316-on-the-fly-adaptive-distillation-of-transformer-to-dual-state-linear-a.md) | 2025-06-17 | 2.12 | 4 | 0 | International Conference on Machine Lear | [✓](https://github.com/utnslab/DSLA-Serve) | DSLA is proposed, a novel design that maintains two specialized hidden states-one for preserving historical context and one for tracking recency-thereby … |
| 18 | [Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](2510.05901-untangling-component-imbalance-in-hybrid-linear-attention-conversion-m.md) | 2025-10-10 | 1.96 | 3 | 0 |  |  | This work identifies a critical flaw: existing hybrid methods inadvertently bypass the linear component, relying almost entirely on SWA, and proposes three … |
| 19 | [Priming: Hybrid State Space Models From Pre-trained Transformers](2605.08301-priming-hybrid-state-space-models-from-pre-trained-transformers.md) | 2026-05-08 | 1.81 | 4 | 0 |  |  | A model zoo of primed Hybrid models for long-context reasoning and instruction following is released, together with the Priming training and inference code … |
| 20 | [Attention to Mamba: A Recipe for Cross-Architecture Distillation](2604.14191-attention-to-mamba-a-recipe-for-cross-architecture-distillation.md) | 2026-04-01 | 1.63 | 2 | 0 |  |  | This work proposes a principled two-stage approach to distill knowledge from a traditional Transformer into a linearized version of Attention, using an … |
| 21 | [When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models](2603.26556-when-perplexity-lies-generation-focused-distillation-of-hybrid-sequenc.md) | 2026-09-17 | 1.52 | 1 | 0 |  |  | GenDistill, a multi-stage pipeline for distilling a pretrained Transformer into an efficient Hybrid Kimi Delta Attention (Hybrid-KDA) student, finds that … |
| 22 | [STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs](2602.02180-still-selecting-tokens-for-intra-layer-hybrid-attention-to-linearize-l.md) | 2026-02-02 | 1.48 | 2 | 0 |  |  | STILL introduces a Self-Saliency Score with strong local-global consistency, enabling accurate token selection using sliding-window computation, and retains … |
| 23 | [Retrieval-Aware Distillation for Transformer-SSM Hybrids](2602.11374-retrieval-aware-distillation-for-transformer-ssm-hybrids.md) | 2026-02-11 | 1.4 | 4 | 0 |  |  | Retrieval-aware distillation is proposed, which converts a pretrained Transformer into a hybrid student by preserving only these retrieval-critical heads and … |
| 24 | [Degrees of Freedom for Linear Attention: Distilling Softmax Attention with Optimal Feature Efficiency](2507.03340-degrees-of-freedom-for-linear-attention-distilling-softmax-attention-w.md) | 2025-07-04 | 1.22 | 2 | 0 | Neural Information Processing Systems (N |  | A principled method to automatically determine the feature dimension in linear attention using the concept of statistical degrees of freedom, which represent … |
| 25 | [Distill-then-Replace: Efficient Task-Specific Hybrid Attention Model Construction](2601.11667-distill-then-replace-efficient-task-specific-hybrid-attention-model-co.md) | 2026-06-02 | 1.22 | 2 | 0 |  |  | DtR (Distill-then-Replace) is proposed, which first transfers weights from the pretrained full-attention modules to its linear attention counterparts through … |
| 26 | [What Matters in Linearizing Language Models? A Comparative Study of Architecture, Scale, and Task Adaptation](2504.14366-what-matters-in-linearizing-language-models-a-comparative-study-of-arc.md) | 2026-01-28 | 0.92 | 1 | 0 |  | [✓](https://github.com/huggingface/alignment-handbook) | It is shown that performance gaps are established early and persist through asymptotic maturity at 10B tokens, suggesting that state resolution is a more … |
| 27 | [Stuck on "A": Diagnosing and Repairing Interface Injury in Attention-to-KDA Linearization of a 0.6B Language M](2608.02689-stuck-on-a-diagnosing-and-repairing-interface-injury-in-attention-to-k.md) | 2026-08-03 | 0.5 | 0 | 0 |  | [✓](https://github.com/Sisyphbaous-DT-Project/open-qingyi) | The engineering lessons -- including an FP32-master failure mode in which bf16 optimizer updates are silently swallowed -- that made convergence possible at … |
| 28 | [Retrofitting Linear Attention into Diffusion Language Models](2608.06628-retrofitting-linear-attention-into-diffusion-language-models.md) | 2026-08-06 | 0.5 | 0 | 0 |  | [✓](https://github.com/Diuven/LLaDA-Hybrid) | This work introduces block-hybrid attention, which retains exact softmax attention within the active denoising block while applying linear attention over … |
| 29 | [Distil-xLSTM: Learning Attention Mechanisms through Recurrent Structures](2503.18565-distil-xlstm-learning-attention-mechanisms-through-recurrent-structure.md) | 2025-03-24 | 0.24 | 1 | 0 |  |  | This work proposes Distil-xLSTM, an xLSTM-based Small Language Model (SLM) trained by distilling knowledge from a Large Language Model (LLM) that shows … |
| 30 | [LayerBoost: Layer-Aware Attention Reduction for Efficient LLMs](2604.22050-layerboost-layer-aware-attention-reduction-for-efficient-llms.md) | 2026-05-14 | 0.0 | 0 | 0 |  |  | LayerBoost is a layer-aware attention reduction method that selectively modifies the attention mechanism based on the sensitivity of individual transformer … |
| 31 | [Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling](2604.24715-long-context-aware-upcycling-a-new-frontier-for-hybrid-llm-scaling.md) | 2026-04-27 | 0.0 | 0 | 0 |  |  | Across 1B- and 3B-scale settings (Llama- and Qwen-based variants), HyLo delivers consistently strong short- and long-context performance and significantly … |
| 32 | [The Key to Going Linear: Analysis-Driven Transformer Linearization](2607.07706-the-key-to-going-linear-analysis-driven-transformer-linearization.md) | 2026-07-08 | 0.0 | 0 | 0 |  |  | It is shown that softmax relies on key-dependent, rank-1 orthogonal projections, elucidating why delta-style networks outperform purely gated accumulation and … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Zebra-Llama: Towards Extremely Efficient Hybrid Models](../attention-conversion/2505.17272-zebra-llama-towards-extremely-efficient-hybrid-models.md) | Attention conversion (MHA/GQA → MLA, GQA uptraining, sparse retrofit) | 4.52 |
| [MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling](../../attention/hybrid-architectures/2602.11761-minicpm-sala-hybridizing-sparse-and-linear-attention-for-efficient-lon.md) | Hybrid architectures (attention + SSM/linear layers) | 3.88 |
| [X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression](../attention-conversion/2503.11132-x-ecomla-upcycling-pre-trained-attention-into-mla-for-efficient-and-ex.md) | Attention conversion (MHA/GQA → MLA, GQA uptraining, sparse retrofit) | 1.53 |
