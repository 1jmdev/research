# Training: optimizers, scaling laws, pre-training, RL, data, PEFT, tokenization (2025–2026)

> Synthesis of [`papers/training`](../papers/training/README.md). Low-precision (FP8/FP4) training is in
> [quantization](quantization.md#low-precision-training-fp8--fp4). Full model recipes are in the
> [technical reports](../papers/models-and-architectures/technical-reports/README.md).

## Model-builder's checklist (2026 consensus)

| Decision | 2026 default | Evidence |
| --- | --- | --- |
| Optimizer | **Muon** (with weight decay and per-parameter update-scale matching) for matrices, AdamW for embeddings/norms | [[2502.16982\|Muon is scalable]] (Moonlight, ~2× compute efficiency vs AdamW), used in Kimi K2 (MuonClip), GLM-4.5, [[2608.30320\|Qwen3.8-Next]]. Distributed variant: [[2504.05295\|Dion]]. |
| Stability | QK-norm / gated attention; spike-aware clipping; avoid outlier formation | [[2501.06842\|SPAM]], [[2506.19697\|Outlier-Safe Pre-Training]] (no Adam-induced outliers, so 4-bit works: 35.7 vs 26.5 avg), [[2601.08393\|Spectral-sphere optimizer]] |
| LR schedule / hyper-params | WSD schedule + checkpoint merging instead of long decay; power-law HP fits | [[2503.04715\|Step Law]] (3,700 models, ~1M H800-h, optimal LR/batch formula), [[2505.12082\|merging in pre-training]] |
| Precision | FP8 today, NVFP4 next | see [quantization](quantization.md) |
| MoE sparsity | activation ratio ~3–10%, granular experts | [[2507.17702\|Efficiency leverage scaling laws]] (Ling: 7× compute efficiency), [[2501.12370\|optimal sparsity]] |
| Data | domain-organised web, model-based refinement, token-level filtering | [[2502.10341\|WebOrganizer]], [[2507.03253\|RefineX]], [[2601.21571\|token-level filtering]] |
| Data-limited regime | more epochs + heavy regularization, ensembling; or diffusion LMs | [[2509.14786\|Pre-training under infinite compute]], [[2511.03276\|DLMs are super data learners]] |

## Scaling laws ([folder](../papers/training/scaling-laws/README.md))

* MoE: [[2507.17702]] (efficiency leverage depends on activation ratio and compute; granularity has an optimum) and
  [[2501.12370]] (optimal sparsity grows with compute).
* Hyper-parameters: [[2503.04715|Step Law]].
* Precision: [[2501.02423|FP quantization-training scaling laws]] (exponent bits matter slightly more than
  mantissa; gives the optimal E/M split per bit-width) and [[2505.14302|QAT scaling law]].
* Multimodal: [[2504.07951|native multimodal scaling]] (early fusion is as good as late fusion; MoE helps).
* Data: [[2502.12120|loss-to-loss scaling]] (the pre-training data determines the trend) and
  [[2503.04725|L²M]] (mutual-information scaling says how big the long-context state must be).

## RL for reasoning ([folder](../papers/training/rl-for-reasoning/README.md))

* **Algorithms:**
  * [[2507.18071|GSPO]] (Qwen): sequence-level importance ratios; stabilises MoE RL. The most-cited 2025 RL
    algorithm paper.
  * [[2511.20347|SAPO]]: soft gating instead of hard clipping.
  * DAPO / Dr.GRPO lineage (see folder).
* **What RL actually does:** [[2504.13837]] (NeurIPS'25). RLVR improves pass@1 but base models win at large pass@k,
  so current RL mostly *sharpens* existing capability. [[2504.20571|1-shot RLVR]]: a single example lifts
  Qwen2.5-Math-1.5B on MATH500 from 36% to 73.6%.
* **Systems that matter to a runtime author:**
  * Rollout generation is 70–90% of RL wall-clock, so the inference engine *is* the RL bottleneck.
  * Use asynchronous and disaggregated rollouts ([[2512.22560|RollArt]]) and routing replay for MoE
    ([[2510.11370|R3]]).
  * Use FP8 end-to-end ([[2601.14243|Jet-RL]]) and train MTP drafters during RL ([[2606.12370]]).
  * Use tree-structured trajectories with shared prefixes ([[2511.00413|Tree Training]], 6.2×).
* **Reward models:** [[2505.02387|RM-R1]] (reasoning reward models) and [[2506.18896|ReasonFlux-PRM]].
* On-policy distillation as an RL alternative: see [compression](compression.md#distillation-folder).

## PEFT ([folder](../papers/training/parameter-efficient-finetuning/README.md))

* [[2502.07374]]: LoRA on only 17k long-CoT samples teaches Qwen2.5-32B o1-preview-level reasoning. *Structure*,
  not content, matters.
* [[2502.14502]] studies how much new knowledge a LoRA adapter can hold before it harms the model.
* [[2512.23165]] benchmarks PEFT for RLVR: DoRA, AdaLoRA and MiSS beat plain LoRA.
* Optimizers for adapters: [[2507.12142|Riemannian LoRA with Muon]]. Hypernetwork-generated adapters:
  [[2506.16406|Drag-and-Drop LLMs]] (prompt → LoRA weights, zero-shot).

## Tokenization ([folder](../papers/training/tokenization/README.md))

Text-tokenizer research for LLMs is in this folder and in
[model conversion → tokenizer transfer](../papers/model-conversion/tokenizer-and-vocab-transfer/README.md). Key
themes: vocabulary scaling, byte-level / latent-patch models (BLT lineage), and tokenizer transfer for multilingual
models. The folder also contains speech, action and recommendation tokenizers, which are useful when building
omni-modal models.
