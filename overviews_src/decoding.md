# Decoding: speculative, MTP, Jacobi / parallel, diffusion LLMs, 2025–2026

> Synthesis of [`papers/decoding`](../papers/decoding/README.md). Converting AR checkpoints into diffusion or
> parallel decoders is covered in [model conversion](model-conversion.md), with compute costs.

## TL;DR for a runtime builder

Tokens per forward pass is the lever. The families below are converging: MTP heads, block-diffusion drafters and
Jacobi-trained models all yield *k* candidate tokens per step, verified causally.

| Approach | Needs | Typical lossless speedup (bs=1) | Batch-friendliness |
| --- | --- | --- | --- |
| EAGLE-3-style draft head ([[2503.01840]]) | train ~1 layer head per model | 3–6.5× | degrades at high batch; 1.38× at bs=64 in SGLang |
| Native MTP heads (DeepSeek-V3 style) + spec verify ([[2509.18362\|FastMTP]], [[2607.05147\|DSpark]]) | MTP trained in pre- or post-training | ~2× (MTP-1), more with semi-AR drafting | good; production at DeepSeek |
| Block-diffusion drafter ([[2602.06036\|DFlash]]) | train a small block-diffusion drafter | **>6×**, 2.5× over EAGLE-3 | good (one forward drafts a block) |
| Parallel/tree heads ([[2606.18394\|JetSpec]], [[2605.29707\|Domino]]) | train heads | up to 9.6× (math), 5.8× SGLang throughput | medium |
| Jacobi Forcing model ([[2512.14681]]) | distill the AR model on its own Jacobi trajectories | ~3.8–4× wall-clock | exact KV reuse (stays causal) |
| Diffusion LLM + cache + parallel unmasking ([[2505.22618\|Fast-dLLM]]) | a dLLM | up to 27.6× throughput vs. naive dLLM | depends on block design |
| AR↔diffusion hybrid in one model ([[2511.08923\|TiDAR]]) | pre-train/convert | 4.7–5.9× tokens/s at AR quality | designed for serving |

**Recommendation:** implement one generic *draft → tree/causal verify* engine that accepts drafts from (a) MTP heads,
(b) EAGLE heads, (c) a block-diffusion drafter and (d) the model itself (Jacobi / self-speculation). All of them
reduce to the same verify kernel with tree attention masks and KV rollback.

## Speculative decoding ([folder](../papers/decoding/speculative-decoding/README.md))

* **Heads:** [[2503.01840|EAGLE-3]] (NeurIPS'25) drops feature prediction, fuses multi-layer features and trains
  with "training-time test". It is the de-facto baseline: up to 6.5×, and 1.4× over EAGLE-2.
* **Beyond autoregressive drafting:**
  * [[2602.06036|DFlash]] (ICML'26) drafts a whole block with a lightweight *block-diffusion* model.
  * [[2605.29707|Domino]] decouples causal modelling from AR drafting.
  * [[2606.18394|JetSpec]] does parallel tree drafting with branch-wise causal conditioning (9.64× on MATH-500 on
    H100).
* **Production:** [[2607.05147|DSpark]] (DeepSeek-V4 serving) uses confidence-scheduled semi-AR drafting with
  load-aware verification: +60–85% per-user speed at matched throughput versus MTP-1.
* **Training objective:** [[2602.23881|LK losses]] optimize acceptance rate directly (+8–10% acceptance length).
  [[2606.12370]] trains MTP drafters with a TV loss for up to 95% acceptance during RL.
* **Large vocabularies:** [[2502.14856|FR-Spec]] restricts the drafter's LM head to frequent tokens, which matters
  for 128K–256K vocabularies.
* **Relaxed verification:** [[2501.19309|Judge Decoding]] (a learned judge accepts "good enough" tokens, 9× on
  Llama-405B) and [[2501.19324|Reward-guided SD]] (PRM-gated acceptance for reasoning, up to 4.4× fewer FLOPs).
* **Reasoning-level speculation:** [[2506.19830|Lookahead Reasoning]] speculates whole reasoning *steps*, raising
  SD speedup from 1.4× to 2.1×.
* **Benchmarking:** [[2604.09557|SPEED-Bench]] (ICML'26) shows synthetic prompts overestimate real throughput, and
  that the optimal draft length depends on batch size.

## Multi-token prediction ([folder](../papers/decoding/multi-token-prediction/README.md))

* As a **pre-training objective** (DeepSeek-V3, Qwen3-Next, GLM-4.5, MiMo and others use MTP heads, then reuse them
  as drafters): [[2505.10518|MuToR]] (register tokens, works in fine-tuning), [[2508.19228|Token Order Prediction]]
  (one extra unembedding instead of extra layers), [[2505.17505|L-MTP]] (leap/skip-token prediction).
* **Retrofitting MTP into an existing model:** [[2507.11851|Your LLM knows the future]] (gated LoRA + masked future
  tokens) and [[2509.18362|FastMTP]] (align MTP training with inference, 2.03× lossless).
* **Training-free:** [[2603.17942|ESP]] (ICML'26) probes mask embeddings, 7–11% longer acceptance than lookahead.
* **MTP during RL:** [[2606.12370]] and [[2605.28184]] show MTP acceptance is bounded by policy entropy, and how to
  co-train MTP heads during RL.

## Jacobi / consistency parallel decoding ([folder](../papers/decoding/jacobi-and-parallel-decoding/README.md))

* **[[2512.14681|Jacobi Forcing]]** is the key 2025 result. It is a progressive distillation in which the model is
  trained on its *own* Jacobi decoding trajectories, turning an AR model into a parallel decoder while keeping
  **causal attention and exact KV reuse**. This sidesteps dLLMs' bidirectional-attention mismatch. It reports 3.8×
  wall-clock speedup on code and math, and ~4× with multi-block decoding plus rejection recycling.
* Latent reasoning variant: [[2506.18582|PCCoT]] runs Jacobi iterations over continuous thought tokens, about 50%
  less train and inference time.
* Semantic-level parallelism: [[2502.11517|PASTA]] learns to mark independent spans and decode them asynchronously
  (1.2–1.9×).

## Diffusion language models (dLLMs)

### Models & training ([folder](../papers/decoding/diffusion-language-models/README.md))
* [[2503.09573|Block Diffusion (BD3-LM)]] (ICLR'25) interpolates AR and diffusion: AR across blocks, diffusion
  within a block, which gives KV caching and flexible length. Almost every fast dLLM since builds on it.
* **Scale:** LLaDA (8B, from scratch) → **LLaDA2.0** (100B MoE, converted, see
  [model conversion](model-conversion.md)), Dream-7B, DiffuCoder ([[2506.20639]]), Mercury and Gemini Diffusion
  (commercial). Tech reports are in [`technical-reports`](../papers/models-and-architectures/technical-reports/README.md).
* **Data efficiency:** [[2511.03276|DLMs are super data learners]]. With limited unique data, dLLMs beat AR models
  by training for many more epochs.
* **RL for dLLMs:** [[2504.12216|d1 / diffu-GRPO]], [[2510.09541|SPG]] (sandwiched ELBO bounds),
  [[2509.06949|TraceRL/TraDo]] (TraDo-8B beats Qwen2.5-7B-Instruct on math).
* **Sampling order matters:** [[2502.06768|Train for the worst, plan for the best]] (adaptive unmasking order takes
  Sudoku from <7% to ~90%) and [[2503.00307|ReMDM]] (remasking sampler for inference-time scaling).
* Tooling: [[2602.22661|dLLM]], an open-source framework for training, inference and evaluation.

### Inference acceleration ([folder](../papers/decoding/diffusion-llm-inference/README.md))
* **KV/feature caching for bidirectional models:** [[2505.22618|Fast-dLLM]] (approximate block-wise KV cache plus
  confidence-threshold parallel decoding, up to 27.6× throughput) and [[2505.15781|dKV-Cache]] (2–10×).
* **Block-diffusion forcing:** [[2508.09192|D2F]] (faster-than-AR dLLM inference via inter-block pipelining) and
  [[2509.26328|Fast-dLLM v2]] (adapts AR models into block dLLMs with ~1B tokens, 2.5× over AR decoding).
* **Learned or adaptive parallelism:** [[2509.26488|dParallel]] (LLaDA-8B from 256 to 30 steps, 8.5× on GSM8K) and
  [[2506.00413|APD]] (a small AR model gates the parallel sample count).
* **Hybrid:** [[2511.08923|TiDAR]] drafts in diffusion and samples in AR within one forward, closing the quality gap
  at 4.7–5.9× tokens/s.
* Safety caveat: [[2507.11097|DIJA]] shows masked-infilling jailbreaks unique to dLLMs.

## Early exit & layer skipping ([folder](../papers/decoding/early-exit-and-layer-skipping/README.md))

* Most 2025 "early exit" work is really **early exit of the reasoning chain**: [[2504.15895|DEER]] (−19–80% CoT
  length, +accuracy) and [[2505.07686|S-GRPO]]. See also [reasoning efficiency](reasoning.md).
* Layer-level: [[2503.23798|FlexiDepth]] (plug-in router and adapter on a frozen model), [[2504.08850|SpecEE]]
  (ISCA'25, speculative early exit with a reduced-vocab predictor), [[2506.03700|AdaDecode]] (adaptive layer
  parallelism, 1.73× with output parity) and [[2609.05275|layer dropout in pre-training]] (saves 25% of training
  FLOPs and enables early exit or self-speculation later).
* Depth as a first-class axis: [[2603.15619|Mixture-of-Depths Attention]] reaches 97.3% of FA-2 efficiency at 64K.

## Constrained / structured decoding ([folder](../papers/decoding/constrained-and-structured/README.md))

* Engines: [[2502.05111|Flexible GCD]] (17.7× faster grammar preprocessing), [[2602.22647|STATIC]] (vectorized trie
  for TPU/GPU, 0.033 ms/step), and XGrammar-style pushdown automata (pre-2025).
* Accuracy caveat: [[2502.09061|CRANE]]. Constraining the *whole* output to a strict grammar hurts reasoning; let the
  model reason freely, then constrain the answer span (+10 points).
* Benchmarks: [[2501.10868|JSONSchemaBench]] (10K real schemas) and [[2505.20139|StructEval]].

## Sampling ([folder](../papers/decoding/sampling-and-strategies/README.md))

[[2509.02510|Top-H]] (NeurIPS'25, entropy-bounded; beats min-p on creative writing by up to 25.6%),
[[2604.11012|Min-k]] (temperature-invariant truncation) and [[2510.26697|AutoDeco]] (the model predicts its own
temperature and top-p per token).
