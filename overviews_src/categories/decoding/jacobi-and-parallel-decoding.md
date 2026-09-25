**Verdict.** Parallel decoding *inside* an AR model now has three practical forms.

1. **Jacobi / consistency-trained decoding.** Jacobi Forcing is the 2025 state of the art. It uses progressive
   distillation on the model's own Jacobi trajectories with a noise schedule. The model stays **causal**, so exact KV
   reuse is kept (unlike dLLMs). It adds rejection-recycling and multi-block decoding: **up to 3.8× speedup** on code
   and math at near-AR quality.
2. **Masked-token prediction in an AR model.** Set Block Decoding (Meta) samples several *non-consecutive* future
   tokens in parallel with discrete-diffusion solvers. There are no architecture changes and KV caching stays exact
   (3–5× fewer forward passes).
3. **Semantic / structural parallelism.** The model decides to fork independent branches:
   * Multiverse: MapReduce-style reasoning with Multiverse Attention;
   * PASTA: learned asynchronous decoding;
   * ASPD, parallel reasoning within one sequence;
   * intra-prompt parallel QA (IPPD, HPD).

Before chasing speedups, read [[2605.30851|How much parallelism is "free"?]]. The hardware only offers a bounded number
of near-free positions per forward, set by memory-bound slack and kernel granularity.

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2512.14681\|Jacobi Forcing]] | Causal parallel decoder | Progressive distillation on the model's own Jacobi trajectories (noise-aware, packed sequences) + rejection recycling + multi-block decoding | Up to **3.8×** with AR-level quality; beats AR→dLLM conversions on speed/quality |
| 2 | [[2509.04185\|Set Block Decoding]] (Meta) | NTP + masked prediction | Fine-tune to fill **sets of future tokens** in parallel; use diffusion solvers (EB-sampler); exact KV cache | 3–5× fewer forward passes at equal accuracy (Llama-3.1-8B, Qwen-3-8B) |
| 3 | [[2506.09991\|Multiverse]] (NeurIPS'25) | Native parallel reasoning | Map → parallel Process → Reduce inside the model; Multiverse Attention + an engine; converts AR models with ~1K examples | Parallel reasoning with real wall-clock speedup at AR-level accuracy |
| 4 | [[2502.11517\|PASTA]] (ICML'25) | Learned async decoding | The model annotates independent chunks (PASTA-LANG); an interpreter decodes them in parallel | Pareto-better speed/quality than heuristic parallel decoding |
| 5 | [[2505.21189\|One-step text generation]] (EMNLP'25) | Analysis | Frozen LLMs can emit **hundreds of tokens in one forward pass** from two learned embeddings | Shows latent multi-token capacity |
| 6 | [[2506.18582\|PCCoT]] (EMNLP'25) | Latent CoT + Jacobi | Jacobi iteration over continuous thought tokens | ~50% less training and inference time for latent CoT |
| 7 | [[2504.20456\|Any-subset AR + ASSD]] | Correct parallel sampling | Any-subset AR models can **self-verify** parallel drafts, giving the exact joint distribution | Provable correctness, unlike dLLM parallel sampling |
| 8 | [[2508.08895\|ASPD]] | Intrinsic parallelism | Extract parallelizable branches from AR outputs; branch-invisible masks with shared position ids | Latency reduction without quality loss |
| 9 | [[2605.30851\|Near-Free Parallelism]] | Systems | Predicts how many positions per forward are ~free for dense, MoE and attention layers | Sets realistic speedup ceilings |

**Runtime requirements.**
* Support **multi-position decode steps**: a verify/accept loop with tree or block masks and KV commit/rollback, plus
  **fork/join of branches sharing a prefix** (Multiverse/PASTA). This is the same machinery speculative decoding needs,
  so build it once.
