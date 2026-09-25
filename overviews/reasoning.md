# Reasoning efficiency & test-time compute (2025–2026)

> Synthesis of [`papers/reasoning`](../papers/reasoning/README.md). Reasoning models now spend most serving
> compute on **decode tokens**. Every token saved here is a speedup no kernel can match.

## TL;DR for a runtime builder

* **Expose reasoning-budget controls in the API.** Frontier models support "thinking budgets" and adaptive
  think/no-think, and the literature shows 20–80% token cuts at equal accuracy:
  * [DEER](../papers/reasoning/efficient-reasoning/2504.15895-dynamic-early-exit-in-reasoning-models.md): early exit at reasoning-transition points, −19–80% CoT length with +0.3–5% accuracy.
  * [Train long, think short](../papers/reasoning/efficient-reasoning/2508.08940-train-long-think-short-curriculum-learning-for-efficient-reasoning.md): a GRPO curriculum tightens the token budget over training.
  * [CGRS](../papers/reasoning/efficient-reasoning/2508.05337-efficient-reasoning-for-large-reasoning-language-models-via-certainty.md): certainty-guided suppression of "wait" reflections, −18–42% tokens.
  * [Long-to-short via model merging](../papers/reasoning/efficient-reasoning/2503.20641-unlocking-efficient-long-to-short-llm-reasoning-with-model-merging.md): merge a long-CoT and a short model, −55% length at no training
    cost.
  * [SEAL](../papers/reasoning/efficient-reasoning/2504.07986-seal-steerable-reasoning-calibration-of-large-language-models-for-free.md): steer away from reflection/transition thoughts, −12–50% tokens with +11% accuracy,
    training-free.
* **Architectures for long generation:** [SambaY / Phi4-mini-Flash-Reasoning](../papers/attention/hybrid-architectures/2507.06607-decoder-hybrid-decoder-architecture-for-efficient-reasoning-with-long.md) (decoder-hybrid-decoder
  with gated memory units, **10× decode throughput** at 32K generation) and M1 (hybrid Mamba reasoning,
  see [attention](attention.md)).
* **Quantize reasoning models carefully:** [ParoQuant](../papers/quantization/4-bit-integer/2511.10645-paroquant-pairwise-rotation-quantization-for-efficient-reasoning-llm-i.md) (pairwise Givens rotations, +2.4% over AWQ on
  reasoning). Long CoT accumulates quantization error; see [quantization](quantization.md).
* **KV compression for long reasoning traces:** [RPC](../papers/kv-cache/eviction-and-token-selection/2505.13866-reasoning-path-compression-compressing-generation-trajectories-for-eff.md) and [TriAttention](../papers/kv-cache/eviction-and-token-selection/2604.04921-triattention-efficient-long-reasoning-with-trigonometric-kv-compressio.md) (AIME25 at full
  accuracy with 10.7× less KV). See [KV cache](kv-cache.md).

## Failure modes to measure ([efficient reasoning folder](../papers/reasoning/efficient-reasoning/README.md))

* **Underthinking:** o1-like models switch thoughts too early ([Thoughts Are All Over the Place](../papers/reasoning/efficient-reasoning/2501.18585-thoughts-are-all-over-the-place-on-the-underthinking-of-o1-like-llms.md)); a thought-switching penalty (TIP)
  fixes it at decode time.
* **Overthinking on ill-posed questions:** [MiP-Overthinking](../papers/reasoning/efficient-reasoning/2504.06514-missing-premise-exacerbates-overthinking-are-reasoning-models-losing-c.md), which spreads through distillation.
* Survey: [Efficient Reasoning Models](../papers/reasoning/efficient-reasoning/2504.10903-efficient-reasoning-models-a-survey.md) (shorter / smaller / faster).

## Test-time scaling ([folder](../papers/reasoning/test-time-scaling/README.md))

* Search + self-evolution: [rStar-Math](../papers/reasoning/test-time-scaling/2501.04519-rstar-math-small-llms-can-master-math-reasoning-with-self-evolved-deep.md) (MCTS with a PRM lifts Qwen2.5-Math-7B on MATH from 58.8% to
  90.0%).
* Code: [S*](../papers/reasoning/test-time-scaling/2502.14382-s-test-time-scaling-for-code-generation.md) (parallel + sequential scaling; a 3B model beats GPT-4o-mini).
* **Parallel reasoning** is the runtime-relevant trend: many branches sharing a prefix. It needs efficient prefix
  sharing, branch pruning and fork/join in the scheduler.
  * [Adaptive Parallel Reasoning](../papers/reasoning/test-time-scaling/2504.15466-learning-adaptive-parallel-reasoning-with-language-models.md): the model spawns and joins threads itself.
  * [Hogwild! Inference](../papers/reasoning/test-time-scaling/2504.06261-hogwild-inference-parallel-llm-generation-via-concurrent-attention.md): concurrent workers share a KV cache.
* Survey: [A Survey on Test-Time Scaling in Large Language Models](../papers/reasoning/test-time-scaling/2503.24235-a-survey-on-test-time-scaling-in-large-language-models-what-how-where.md).

## Latent & looped reasoning ([folder](../papers/reasoning/latent-and-looped/README.md))

* [Recurrent-depth (Huginn)](../papers/reasoning/latent-and-looped/2502.05171-scaling-up-test-time-compute-with-latent-reasoning-a-recurrent-depth-a.md) scales test-time compute by iterating a recurrent block in latent space.
  Runtime implication: variable depth per token, and KV caches shared across iterations.
* [Parallel Loop Transformer](../papers/reasoning/latent-and-looped/2510.24824-parallel-loop-transformer-for-efficient-test-time-computation-scaling.md) gets looped-model accuracy at near non-looped latency (cross-loop
  parallelism). [SMELT](../papers/training/scaling-laws/2609.01343-smelt-scaling-laws-for-compute-matched-moe-looped-transformers.md) gives compute-matched looped-MoE scaling laws.
  [LoopUS](../papers/reasoning/latent-and-looped/2605.11011-loopus-recasting-pretrained-llms-into-looped-latent-refinement-models.md) recasts a pretrained LLM as a looped latent-refinement model without training from scratch.
* Continuous thoughts: [System-1.5](../papers/reasoning/latent-and-looped/2505.18962-system-1-5-reasoning-traversal-in-language-and-latent-spaces-with-dyna.md) (latent shortcuts, >20× faster on GSM8K), [SemCoT](../papers/reasoning/latent-and-looped/2510.24940-semcot-accelerating-chain-of-thought-reasoning-through-semantically-al.md),
  [HRPO](../papers/reasoning/latent-and-looped/2505.18454-hybrid-latent-reasoning-via-reinforcement-learning.md) (RL for hybrid latent reasoning) and [PCCoT](../papers/decoding/jacobi-and-parallel-decoding/2506.18582-parallel-continuous-chain-of-thought-with-jacobi-iteration.md) (Jacobi iteration over latent
  thoughts).
* Survey: [A Survey on Latent Reasoning](../papers/reasoning/latent-and-looped/2507.06203-a-survey-on-latent-reasoning.md).
