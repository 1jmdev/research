# Training: optimizers, scaling laws, pre-training, RL, data, PEFT, tokenization (2025–2026)

> Synthesis of [`papers/training`](../papers/training/README.md). Low-precision (FP8/FP4) training is in
> [quantization](quantization.md#low-precision-training-fp8--fp4). Full model recipes are in the
> [technical reports](../papers/models-and-architectures/technical-reports/README.md).

## Model-builder's checklist (2026 consensus)

| Decision | 2026 default | Evidence |
| --- | --- | --- |
| Optimizer | **Muon** (with weight decay and per-parameter update-scale matching) for matrices, AdamW for embeddings/norms | [Muon is scalable](../papers/training/optimizers/2502.16982-muon-is-scalable-for-llm-training.md) (Moonlight, ~2× compute efficiency vs AdamW), used in Kimi K2 (MuonClip), GLM-4.5, [Qwen3.8-Next](../papers/models-and-architectures/technical-reports/2608.30320-on-the-design-of-qwen3-8-next-architecture-evaluation-efficiency-and-t.md). Distributed variant: [Dion](../papers/training/optimizers/2504.05295-dion-distributed-orthonormalized-updates.md). |
| Stability | QK-norm / gated attention; spike-aware clipping; avoid outlier formation | [SPAM](../papers/training/optimizers/2501.06842-spam-spike-aware-adam-with-momentum-reset-for-stable-llm-training.md), [Outlier-Safe Pre-Training](../papers/quantization/_general/2506.19697-outlier-safe-pre-training-for-robust-4-bit-quantization-of-large-langu.md) (no Adam-induced outliers, so 4-bit works: 35.7 vs 26.5 avg), [Spectral-sphere optimizer](../papers/training/optimizers/2601.08393-controlled-llm-training-on-spectral-sphere.md) |
| LR schedule / hyper-params | WSD schedule + checkpoint merging instead of long decay; power-law HP fits | [Step Law](../papers/training/scaling-laws/2503.04715-predictable-scale-part-i-step-law-optimal-hyperparameter-scaling-law-i.md) (3,700 models, ~1M H800-h, optimal LR/batch formula), [merging in pre-training](../papers/model-conversion/model-merging/2505.12082-model-merging-in-pre-training-of-large-language-models.md) |
| Precision | FP8 today, NVFP4 next | see [quantization](quantization.md) |
| MoE sparsity | activation ratio ~3–10%, granular experts | [Efficiency leverage scaling laws](../papers/training/scaling-laws/2507.17702-towards-greater-leverage-scaling-laws-for-efficient-mixture-of-experts.md) (Ling: 7× compute efficiency), [optimal sparsity](../papers/training/scaling-laws/2501.12370-parameters-vs-flops-scaling-laws-for-optimal-sparsity-for-mixture-of-e.md) |
| Data | domain-organised web, model-based refinement, token-level filtering | [WebOrganizer](../papers/training/data-curation-and-synthetic-data/2502.10341-organize-the-web-constructing-domains-enhances-pre-training-data-curat.md), [RefineX](../papers/training/data-curation-and-synthetic-data/2507.03253-refinex-learning-to-refine-pre-training-data-at-scale-from-expert-guid.md), [token-level filtering](../papers/training/data-curation-and-synthetic-data/2601.21571-shaping-capabilities-with-token-level-data-filtering.md) |
| Data-limited regime | more epochs + heavy regularization, ensembling; or diffusion LMs | [Pre-training under infinite compute](../papers/training/scaling-laws/2509.14786-pre-training-under-infinite-compute.md), [DLMs are super data learners](../papers/decoding/diffusion-language-models/2511.03276-diffusion-language-models-are-super-data-learners.md) |

## Scaling laws ([folder](../papers/training/scaling-laws/README.md))

* MoE: [Towards Greater Leverage](../papers/training/scaling-laws/2507.17702-towards-greater-leverage-scaling-laws-for-efficient-mixture-of-experts.md) (efficiency leverage depends on activation ratio and compute; granularity has an optimum) and
  [Parameters vs FLOPs](../papers/training/scaling-laws/2501.12370-parameters-vs-flops-scaling-laws-for-optimal-sparsity-for-mixture-of-e.md) (optimal sparsity grows with compute).
* Hyper-parameters: [Step Law](../papers/training/scaling-laws/2503.04715-predictable-scale-part-i-step-law-optimal-hyperparameter-scaling-law-i.md).
* Precision: [FP quantization-training scaling laws](../papers/quantization/low-precision-training/2501.02423-scaling-laws-for-floating-point-quantization-training.md) (exponent bits matter slightly more than
  mantissa; gives the optimal E/M split per bit-width) and [QAT scaling law](../papers/quantization/4-bit-integer/2505.14302-scaling-law-for-quantization-aware-training.md).
* Multimodal: [native multimodal scaling](../papers/training/scaling-laws/2504.07951-scaling-laws-for-native-multimodal-models.md) (early fusion is as good as late fusion; MoE helps).
* Data: [loss-to-loss scaling](../papers/training/scaling-laws/2502.12120-llms-on-the-line-data-determines-loss-to-loss-scaling-laws.md) (the pre-training data determines the trend) and
  [L²M](../papers/training/scaling-laws/2503.04725-l-2-m-mutual-information-scaling-law-for-long-context-language-modelin.md) (mutual-information scaling says how big the long-context state must be).

## RL for reasoning ([folder](../papers/training/rl-for-reasoning/README.md))

* **Algorithms:**
  * [GSPO](../papers/training/rl-for-reasoning/2507.18071-group-sequence-policy-optimization.md) (Qwen): sequence-level importance ratios; stabilises MoE RL. The most-cited 2025 RL
    algorithm paper.
  * [SAPO](../papers/training/rl-for-reasoning/2511.20347-soft-adaptive-policy-optimization.md): soft gating instead of hard clipping.
  * DAPO / Dr.GRPO lineage (see folder).
* **What RL actually does:** [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](../papers/training/rl-for-reasoning/2504.13837-does-reinforcement-learning-really-incentivize-reasoning-capacity-in-l.md) (NeurIPS'25). RLVR improves pass@1 but base models win at large pass@k,
  so current RL mostly *sharpens* existing capability. [1-shot RLVR](../papers/training/rl-for-reasoning/2504.20571-reinforcement-learning-for-reasoning-in-large-language-models-with-one.md): a single example lifts
  Qwen2.5-Math-1.5B on MATH500 from 36% to 73.6%.
* **Systems that matter to a runtime author:**
  * Rollout generation is 70–90% of RL wall-clock, so the inference engine *is* the RL bottleneck.
  * Use asynchronous and disaggregated rollouts ([RollArt](../papers/training/rl-for-reasoning/2512.22560-rollart-disaggregated-multi-task-agentic-rl-training-at-scale.md)) and routing replay for MoE
    ([R3](../papers/training/rl-for-reasoning/2510.11370-stabilizing-moe-reinforcement-learning-by-aligning-training-and-infere.md)).
  * Use FP8 end-to-end ([Jet-RL](../papers/quantization/low-precision-training/2601.14243-jet-rl-enabling-on-policy-fp8-reinforcement-learning-with-unified-trai.md)) and train MTP drafters during RL ([Breaking Entropy Bounds](../papers/training/rl-for-reasoning/2606.12370-breaking-entropy-bounds-accelerating-rl-training-via-mtp-with-rejectio.md)).
  * Use tree-structured trajectories with shared prefixes ([Tree Training](https://arxiv.org/abs/2511.00413), 6.2×).
* **Reward models:** [RM-R1](../papers/training/rl-for-reasoning/2505.02387-rm-r1-reward-modeling-as-reasoning.md) (reasoning reward models) and [ReasonFlux-PRM](../papers/reasoning/test-time-scaling/2506.18896-reasonflux-prm-trajectory-aware-prms-for-long-chain-of-thought-reasoni.md).
* On-policy distillation as an RL alternative: see [compression](compression.md#distillation-folder).

## PEFT ([folder](../papers/training/parameter-efficient-finetuning/README.md))

* [LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!](../papers/reasoning/efficient-reasoning/2502.07374-llms-can-easily-learn-to-reason-from-demonstrations-structure-not-cont.md): LoRA on only 17k long-CoT samples teaches Qwen2.5-32B o1-preview-level reasoning. *Structure*,
  not content, matters.
* [How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?](../papers/training/parameter-efficient-finetuning/2502.14502-how-much-knowledge-can-you-pack-into-a-lora-adapter-without-harming-ll.md) studies how much new knowledge a LoRA adapter can hold before it harms the model.
* [Evaluating Parameter Efficient Methods for RLVR](../papers/training/rl-for-reasoning/2512.23165-evaluating-parameter-efficient-methods-for-rlvr.md) benchmarks PEFT for RLVR: DoRA, AdaLoRA and MiSS beat plain LoRA.
* Optimizers for adapters: [Riemannian LoRA with Muon](../papers/training/parameter-efficient-finetuning/2507.12142-lora-meets-riemannion-muon-optimizer-for-parametrization-independent-l.md). Hypernetwork-generated adapters:
  [Drag-and-Drop LLMs](../papers/training/parameter-efficient-finetuning/2506.16406-drag-and-drop-llms-zero-shot-prompt-to-weights.md) (prompt → LoRA weights, zero-shot).

## Tokenization ([folder](../papers/training/tokenization/README.md))

Text-tokenizer research for LLMs is in this folder and in
[model conversion → tokenizer transfer](../papers/model-conversion/tokenizer-and-vocab-transfer/README.md). Key
themes: vocabulary scaling, byte-level / latent-patch models (BLT lineage), and tokenizer transfer for multilingual
models. The folder also contains speech, action and recommendation tokenizers, which are useful when building
omni-modal models.
