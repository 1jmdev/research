# Decoding: speculative, MTP, Jacobi / parallel, diffusion LLMs, 2025–2026

> Synthesis of [`papers/decoding`](../papers/decoding/README.md). Converting AR checkpoints into diffusion or
> parallel decoders is covered in [model conversion](model-conversion.md), with compute costs.

## TL;DR for a runtime builder

Tokens per forward pass is the lever. The families below are converging: MTP heads, block-diffusion drafters and
Jacobi-trained models all yield *k* candidate tokens per step, verified causally.

| Approach | Needs | Typical lossless speedup (bs=1) | Batch-friendliness |
| --- | --- | --- | --- |
| EAGLE-3-style draft head ([EAGLE-3](../papers/decoding/speculative-decoding/2503.01840-eagle-3-scaling-up-inference-acceleration-of-large-language-models-via.md)) | train ~1 layer head per model | 3–6.5× | degrades at high batch; 1.38× at bs=64 in SGLang |
| Native MTP heads (DeepSeek-V3 style) + spec verify ([FastMTP](../papers/decoding/multi-token-prediction/2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md), [DSpark](../papers/decoding/speculative-decoding/2607.05147-dspark-confidence-scheduled-speculative-decoding-with-semi-autoregress.md)) | MTP trained in pre- or post-training | ~2× (MTP-1), more with semi-AR drafting | good; production at DeepSeek |
| Block-diffusion drafter ([DFlash](../papers/decoding/speculative-decoding/2602.06036-dflash-block-diffusion-for-flash-speculative-decoding.md)) | train a small block-diffusion drafter | **>6×**, 2.5× over EAGLE-3 | good (one forward drafts a block) |
| Parallel/tree heads ([JetSpec](../papers/decoding/speculative-decoding/2606.18394-jetspec-breaking-the-scaling-ceiling-of-speculative-decoding-with-para.md), [Domino](../papers/decoding/speculative-decoding/2605.29707-domino-decoupling-causal-modeling-from-autoregressive-drafting-in-spec.md)) | train heads | up to 9.6× (math), 5.8× SGLang throughput | medium |
| Jacobi Forcing model ([Fast and Accurate Causal Parallel Decoding using Jacobi Forcing](../papers/decoding/jacobi-and-parallel-decoding/2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md)) | distill the AR model on its own Jacobi trajectories | ~3.8–4× wall-clock | exact KV reuse (stays causal) |
| Diffusion LLM + cache + parallel unmasking ([Fast-dLLM](../papers/decoding/diffusion-llm-inference/2505.22618-fast-dllm-training-free-acceleration-of-diffusion-llm-by-enabling-kv-c.md)) | a dLLM | up to 27.6× throughput vs. naive dLLM | depends on block design |
| AR↔diffusion hybrid in one model ([TiDAR](../papers/decoding/diffusion-llm-inference/2511.08923-tidar-think-in-diffusion-talk-in-autoregression.md)) | pre-train/convert | 4.7–5.9× tokens/s at AR quality | designed for serving |

**Recommendation:** implement one generic *draft → tree/causal verify* engine that accepts drafts from (a) MTP heads,
(b) EAGLE heads, (c) a block-diffusion drafter and (d) the model itself (Jacobi / self-speculation). All of them
reduce to the same verify kernel with tree attention masks and KV rollback.

## Speculative decoding ([folder](../papers/decoding/speculative-decoding/README.md))

* **Heads:** [EAGLE-3](../papers/decoding/speculative-decoding/2503.01840-eagle-3-scaling-up-inference-acceleration-of-large-language-models-via.md) (NeurIPS'25) drops feature prediction, fuses multi-layer features and trains
  with "training-time test". It is the de-facto baseline: up to 6.5×, and 1.4× over EAGLE-2.
* **Beyond autoregressive drafting:**
  * [DFlash](../papers/decoding/speculative-decoding/2602.06036-dflash-block-diffusion-for-flash-speculative-decoding.md) (ICML'26) drafts a whole block with a lightweight *block-diffusion* model.
  * [Domino](../papers/decoding/speculative-decoding/2605.29707-domino-decoupling-causal-modeling-from-autoregressive-drafting-in-spec.md) decouples causal modelling from AR drafting.
  * [JetSpec](../papers/decoding/speculative-decoding/2606.18394-jetspec-breaking-the-scaling-ceiling-of-speculative-decoding-with-para.md) does parallel tree drafting with branch-wise causal conditioning (9.64× on MATH-500 on
    H100).
* **Production:** [DSpark](../papers/decoding/speculative-decoding/2607.05147-dspark-confidence-scheduled-speculative-decoding-with-semi-autoregress.md) (DeepSeek-V4 serving) uses confidence-scheduled semi-AR drafting with
  load-aware verification: +60–85% per-user speed at matched throughput versus MTP-1.
* **Training objective:** [LK losses](../papers/decoding/speculative-decoding/2602.23881-lk-losses-direct-acceptance-rate-optimization-for-speculative-decoding.md) optimize acceptance rate directly (+8–10% acceptance length).
  [Breaking Entropy Bounds](../papers/training/rl-for-reasoning/2606.12370-breaking-entropy-bounds-accelerating-rl-training-via-mtp-with-rejectio.md) trains MTP drafters with a TV loss for up to 95% acceptance during RL.
* **Large vocabularies:** [FR-Spec](../papers/decoding/speculative-decoding/2502.14856-fr-spec-accelerating-large-vocabulary-language-models-via-frequency-ra.md) restricts the drafter's LM head to frequent tokens, which matters
  for 128K–256K vocabularies.
* **Relaxed verification:** [Judge Decoding](../papers/decoding/speculative-decoding/2501.19309-judge-decoding-faster-speculative-sampling-requires-going-beyond-model.md) (a learned judge accepts "good enough" tokens, 9× on
  Llama-405B) and [Reward-guided SD](../papers/decoding/speculative-decoding/2501.19324-reward-guided-speculative-decoding-for-efficient-llm-reasoning.md) (PRM-gated acceptance for reasoning, up to 4.4× fewer FLOPs).
* **Reasoning-level speculation:** [Lookahead Reasoning](../papers/decoding/speculative-decoding/2506.19830-scaling-speculative-decoding-with-lookahead-reasoning.md) speculates whole reasoning *steps*, raising
  SD speedup from 1.4× to 2.1×.
* **Benchmarking:** [SPEED-Bench](../papers/decoding/speculative-decoding/2604.09557-speed-bench-a-unified-and-diverse-benchmark-for-speculative-decoding.md) (ICML'26) shows synthetic prompts overestimate real throughput, and
  that the optimal draft length depends on batch size.

## Multi-token prediction ([folder](../papers/decoding/multi-token-prediction/README.md))

* As a **pre-training objective** (DeepSeek-V3, Qwen3-Next, GLM-4.5, MiMo and others use MTP heads, then reuse them
  as drafters): [MuToR](../papers/decoding/multi-token-prediction/2505.10518-multi-token-prediction-needs-registers.md) (register tokens, works in fine-tuning), [Token Order Prediction](../papers/decoding/multi-token-prediction/2508.19228-predicting-the-order-of-upcoming-tokens-improves-language-modeling.md)
  (one extra unembedding instead of extra layers), [L-MTP](../papers/decoding/multi-token-prediction/2505.17505-l-mtp-leap-multi-token-prediction-beyond-adjacent-context-for-large-la.md) (leap/skip-token prediction).
* **Retrofitting MTP into an existing model:** [Your LLM knows the future](../papers/decoding/multi-token-prediction/2507.11851-your-llm-knows-the-future-uncovering-its-multi-token-prediction-potent.md) (gated LoRA + masked future
  tokens) and [FastMTP](../papers/decoding/multi-token-prediction/2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md) (align MTP training with inference, 2.03× lossless).
* **Training-free:** [ESP](../papers/decoding/multi-token-prediction/2603.17942-efficient-training-free-multi-token-prediction-via-embedding-space-pro.md) (ICML'26) probes mask embeddings, 7–11% longer acceptance than lookahead.
* **MTP during RL:** [Breaking Entropy Bounds](../papers/training/rl-for-reasoning/2606.12370-breaking-entropy-bounds-accelerating-rl-training-via-mtp-with-rejectio.md) and [Joint Training of Multi-Token Prediction in Reinforcement Learning via Optimal Coefficient Calibration](../papers/training/rl-for-reasoning/2605.28184-joint-training-of-multi-token-prediction-in-reinforcement-learning-via.md) show MTP acceptance is bounded by policy entropy, and how to
  co-train MTP heads during RL.

## Jacobi / consistency parallel decoding ([folder](../papers/decoding/jacobi-and-parallel-decoding/README.md))

* **[Jacobi Forcing](../papers/decoding/jacobi-and-parallel-decoding/2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md)** is the key 2025 result. It is a progressive distillation in which the model is
  trained on its *own* Jacobi decoding trajectories, turning an AR model into a parallel decoder while keeping
  **causal attention and exact KV reuse**. This sidesteps dLLMs' bidirectional-attention mismatch. It reports 3.8×
  wall-clock speedup on code and math, and ~4× with multi-block decoding plus rejection recycling.
* Latent reasoning variant: [PCCoT](../papers/decoding/jacobi-and-parallel-decoding/2506.18582-parallel-continuous-chain-of-thought-with-jacobi-iteration.md) runs Jacobi iterations over continuous thought tokens, about 50%
  less train and inference time.
* Semantic-level parallelism: [PASTA](../papers/decoding/jacobi-and-parallel-decoding/2502.11517-learning-to-keep-a-promise-scaling-language-model-decoding-parallelism.md) learns to mark independent spans and decode them asynchronously
  (1.2–1.9×).

## Diffusion language models (dLLMs)

### Models & training ([folder](../papers/decoding/diffusion-language-models/README.md))
* [Block Diffusion (BD3-LM)](../papers/decoding/diffusion-language-models/2503.09573-block-diffusion-interpolating-between-autoregressive-and-diffusion-lan.md) (ICLR'25) interpolates AR and diffusion: AR across blocks, diffusion
  within a block, which gives KV caching and flexible length. Almost every fast dLLM since builds on it.
* **Scale:** LLaDA (8B, from scratch) → **LLaDA2.0** (100B MoE, converted, see
  [model conversion](model-conversion.md)), Dream-7B, DiffuCoder ([DiffuCoder](../papers/decoding/diffusion-language-models/2506.20639-diffucoder-understanding-and-improving-masked-diffusion-models-for-cod.md)), Mercury and Gemini Diffusion
  (commercial). Tech reports are in [`technical-reports`](../papers/models-and-architectures/technical-reports/README.md).
* **Data efficiency:** [DLMs are super data learners](../papers/decoding/diffusion-language-models/2511.03276-diffusion-language-models-are-super-data-learners.md). With limited unique data, dLLMs beat AR models
  by training for many more epochs.
* **RL for dLLMs:** [d1 / diffu-GRPO](../papers/decoding/diffusion-language-models/2504.12216-d1-scaling-reasoning-in-diffusion-large-language-models-via-reinforcem.md), [SPG](../papers/decoding/diffusion-language-models/2510.09541-spg-sandwiched-policy-gradient-for-masked-diffusion-language-models.md) (sandwiched ELBO bounds),
  [TraceRL/TraDo](../papers/decoding/diffusion-language-models/2509.06949-revolutionizing-reinforcement-learning-framework-for-diffusion-large-l.md) (TraDo-8B beats Qwen2.5-7B-Instruct on math).
* **Sampling order matters:** [Train for the worst, plan for the best](../papers/decoding/diffusion-language-models/2502.06768-train-for-the-worst-plan-for-the-best-understanding-token-ordering-in.md) (adaptive unmasking order takes
  Sudoku from <7% to ~90%) and [ReMDM](../papers/decoding/diffusion-language-models/2503.00307-remasking-discrete-diffusion-models-with-inference-time-scaling.md) (remasking sampler for inference-time scaling).
* Tooling: [dLLM](../papers/decoding/diffusion-language-models/2602.22661-dllm-simple-diffusion-language-modeling.md), an open-source framework for training, inference and evaluation.

### Inference acceleration ([folder](../papers/decoding/diffusion-llm-inference/README.md))
* **KV/feature caching for bidirectional models:** [Fast-dLLM](../papers/decoding/diffusion-llm-inference/2505.22618-fast-dllm-training-free-acceleration-of-diffusion-llm-by-enabling-kv-c.md) (approximate block-wise KV cache plus
  confidence-threshold parallel decoding, up to 27.6× throughput) and [dKV-Cache](../papers/decoding/diffusion-llm-inference/2505.15781-dkv-cache-the-cache-for-diffusion-language-models.md) (2–10×).
* **Block-diffusion forcing:** [D2F](../papers/decoding/diffusion-llm-inference/2508.09192-diffusion-llms-can-do-faster-than-ar-inference-via-discrete-diffusion.md) (faster-than-AR dLLM inference via inter-block pipelining) and
  [Fast-dLLM v2](../papers/decoding/diffusion-llm-inference/2509.26328-fast-dllm-v2-efficient-block-diffusion-llm.md) (adapts AR models into block dLLMs with ~1B tokens, 2.5× over AR decoding).
* **Learned or adaptive parallelism:** [dParallel](../papers/decoding/diffusion-llm-inference/2509.26488-dparallel-learnable-parallel-decoding-for-dllms.md) (LLaDA-8B from 256 to 30 steps, 8.5× on GSM8K) and
  [APD](../papers/decoding/diffusion-llm-inference/2506.00413-accelerating-diffusion-llms-via-adaptive-parallel-decoding.md) (a small AR model gates the parallel sample count).
* **Hybrid:** [TiDAR](../papers/decoding/diffusion-llm-inference/2511.08923-tidar-think-in-diffusion-talk-in-autoregression.md) drafts in diffusion and samples in AR within one forward, closing the quality gap
  at 4.7–5.9× tokens/s.
* Safety caveat: [DIJA](https://arxiv.org/abs/2507.11097) shows masked-infilling jailbreaks unique to dLLMs.

## Early exit & layer skipping ([folder](../papers/decoding/early-exit-and-layer-skipping/README.md))

* Most 2025 "early exit" work is really **early exit of the reasoning chain**: [DEER](../papers/reasoning/efficient-reasoning/2504.15895-dynamic-early-exit-in-reasoning-models.md) (−19–80% CoT
  length, +accuracy) and [S-GRPO](../papers/reasoning/efficient-reasoning/2505.07686-s-grpo-early-exit-via-reinforcement-learning-in-reasoning-models.md). See also [reasoning efficiency](reasoning.md).
* Layer-level: [FlexiDepth](../papers/decoding/early-exit-and-layer-skipping/2503.23798-adaptive-layer-skipping-in-pre-trained-llms.md) (plug-in router and adapter on a frozen model), [SpecEE](../papers/decoding/early-exit-and-layer-skipping/2504.08850-specee-accelerating-large-language-model-inference-with-speculative-ea.md)
  (ISCA'25, speculative early exit with a reduced-vocab predictor), [AdaDecode](../papers/decoding/early-exit-and-layer-skipping/2506.03700-adadecode-accelerating-llm-decoding-with-adaptive-layer-parallelism.md) (adaptive layer
  parallelism, 1.73× with output parity) and [layer dropout in pre-training](../papers/decoding/early-exit-and-layer-skipping/2609.05275-don-t-drop-dropout-optimizing-layer-sparsity-for-efficient-llm-trainin.md) (saves 25% of training
  FLOPs and enables early exit or self-speculation later).
* Depth as a first-class axis: [Mixture-of-Depths Attention](../papers/models-and-architectures/novel-architectures/2603.15619-mixture-of-depths-attention.md) reaches 97.3% of FA-2 efficiency at 64K.

## Constrained / structured decoding ([folder](../papers/decoding/constrained-and-structured/README.md))

* Engines: [Flexible GCD](../papers/decoding/constrained-and-structured/2502.05111-flexible-and-efficient-grammar-constrained-decoding.md) (17.7× faster grammar preprocessing), [STATIC](../papers/decoding/constrained-and-structured/2602.22647-vectorizing-the-trie-efficient-constrained-decoding-for-llm-based-gene.md) (vectorized trie
  for TPU/GPU, 0.033 ms/step), and XGrammar-style pushdown automata (pre-2025).
* Accuracy caveat: [CRANE](../papers/decoding/constrained-and-structured/2502.09061-crane-reasoning-with-constrained-llm-generation.md). Constraining the *whole* output to a strict grammar hurts reasoning; let the
  model reason freely, then constrain the answer span (+10 points).
* Benchmarks: [JSONSchemaBench](../papers/decoding/constrained-and-structured/2501.10868-jsonschemabench-a-rigorous-benchmark-of-structured-outputs-for-languag.md) (10K real schemas) and [StructEval](https://arxiv.org/abs/2505.20139).

## Sampling ([folder](../papers/decoding/sampling-and-strategies/README.md))

[Top-H](../papers/decoding/sampling-and-strategies/2509.02510-top-h-decoding-adapting-the-creativity-and-coherence-with-bounded-entr.md) (NeurIPS'25, entropy-bounded; beats min-p on creative writing by up to 25.6%),
[Min-k](../papers/decoding/sampling-and-strategies/2604.11012-min-k-sampling-decoupling-truncation-from-temperature-scaling-via-rela.md) (temperature-invariant truncation) and [AutoDeco](../papers/decoding/sampling-and-strategies/2510.26697-the-end-of-manual-decoding-towards-truly-end-to-end-language-models.md) (the model predicts its own
temperature and top-p per token).
