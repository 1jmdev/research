# Reasoning efficiency & test-time compute (2025–2026)

> Synthesis of [`papers/reasoning`](../papers/reasoning/README.md). Reasoning models now spend most serving
> compute on **decode tokens**. Every token saved here is a speedup no kernel can match.

## TL;DR for a runtime builder

* **Expose reasoning-budget controls in the API.** Frontier models support "thinking budgets" and adaptive
  think/no-think, and the literature shows 20–80% token cuts at equal accuracy:
  * [[2504.15895|DEER]]: early exit at reasoning-transition points, −19–80% CoT length with +0.3–5% accuracy.
  * [[2508.08940|Train long, think short]]: a GRPO curriculum tightens the token budget over training.
  * [[2508.05337|CGRS]]: certainty-guided suppression of "wait" reflections, −18–42% tokens.
  * [[2503.20641|Long-to-short via model merging]]: merge a long-CoT and a short model, −55% length at no training
    cost.
  * [[2504.07986|SEAL]]: steer away from reflection/transition thoughts, −12–50% tokens with +11% accuracy,
    training-free.
* **Architectures for long generation:** [[2507.06607|SambaY / Phi4-mini-Flash-Reasoning]] (decoder-hybrid-decoder
  with gated memory units, **10× decode throughput** at 32K generation) and M1 (hybrid Mamba reasoning,
  see [attention](attention.md)).
* **Quantize reasoning models carefully:** [[2511.10645|ParoQuant]] (pairwise Givens rotations, +2.4% over AWQ on
  reasoning). Long CoT accumulates quantization error; see [quantization](quantization.md).
* **KV compression for long reasoning traces:** [[2505.13866|RPC]] and [[2604.04921|TriAttention]] (AIME25 at full
  accuracy with 10.7× less KV). See [KV cache](kv-cache.md).

## Failure modes to measure ([efficient reasoning folder](../papers/reasoning/efficient-reasoning/README.md))

* **Underthinking:** o1-like models switch thoughts too early ([[2501.18585]]); a thought-switching penalty (TIP)
  fixes it at decode time.
* **Overthinking on ill-posed questions:** [[2504.06514|MiP-Overthinking]], which spreads through distillation.
* Survey: [[2504.10903|Efficient Reasoning Models]] (shorter / smaller / faster).

## Test-time scaling ([folder](../papers/reasoning/test-time-scaling/README.md))

* Search + self-evolution: [[2501.04519|rStar-Math]] (MCTS with a PRM lifts Qwen2.5-Math-7B on MATH from 58.8% to
  90.0%).
* Code: [[2502.14382|S*]] (parallel + sequential scaling; a 3B model beats GPT-4o-mini).
* **Parallel reasoning** is the runtime-relevant trend: many branches sharing a prefix. It needs efficient prefix
  sharing, branch pruning and fork/join in the scheduler.
  * [[2504.15466|Adaptive Parallel Reasoning]]: the model spawns and joins threads itself.
  * [[2504.06261|Hogwild! Inference]]: concurrent workers share a KV cache.
* Survey: [[2503.24235]].

## Latent & looped reasoning ([folder](../papers/reasoning/latent-and-looped/README.md))

* [[2502.05171|Recurrent-depth (Huginn)]] scales test-time compute by iterating a recurrent block in latent space.
  Runtime implication: variable depth per token, and KV caches shared across iterations.
* [[2510.24824|Parallel Loop Transformer]] gets looped-model accuracy at near non-looped latency (cross-loop
  parallelism). [[2609.01343|SMELT]] gives compute-matched looped-MoE scaling laws.
  [[2605.11011|LoopUS]] recasts a pretrained LLM as a looped latent-refinement model without training from scratch.
* Continuous thoughts: [[2505.18962|System-1.5]] (latent shortcuts, >20× faster on GSM8K), [[2510.24940|SemCoT]],
  [[2505.18454|HRPO]] (RL for hybrid latent reasoning) and [[2506.18582|PCCoT]] (Jacobi iteration over latent
  thoughts).
* Survey: [[2507.06203]].
