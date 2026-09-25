**Verdict.** Weight sparsity is the weakest compression lever for LLM inference in 2025–26. Quantization gives more
memory and speed per accuracy point, and the two only partly compose. What changed:

* **2:4 is too aggressive for reasoning models.** Qwen3 drops from 54% to 15% on a reasoning benchmark at one-shot 2:4
  ([[2603.05232|SlideSparse]]). Milder patterns recover accuracy while keeping hardware speedups:
  * **(2N−2):2N**, e.g. 6:8 via SlideSparse, reaching **1.33× measured** of a 4/3 ceiling;
  * **8:16** ([[2507.03052]]);
  * **mixed sparse + dense GEMM** splits ([[2606.10445|SpenseGPT]]: first one-shot 2:4 with a real 1.2× end-to-end
    decode speedup on B200 FP8).
* **Unstructured 50% is finally worth something on GPUs.** It needs kernels built for *low* sparsity:
  [[2511.13061|MACKO]] SpMV (1.5× memory, 1.2–1.5× faster than dense at 50%) and [[2607.08786]] (sparse tensor
  cores + CUDA cores, beats dense on HBM GPUs). CPU decode benefits too ([[2502.12444|SparAMX]] on AMX).
* **Better one-shot masks.**
  * [[2503.04992|Wanda++]]: regional gradients, 7B in <10 min;
  * [[2505.23049|DenoiseRotator]] (NeurIPS'25): rotations *concentrate importance* before pruning, closing 58% of the
    2:4 gap on Llama-3-70B;
  * [[2510.05528|ARMOR]] (ICLR'26): 2:4 core wrapped by small block-diagonal factors.
* **Learned masks / sparsity-aware training** are the way to *near-lossless* 2:4:
  * [[2502.00258|ProxSparse]] (ICML'25): proximal mask learning;
  * [[2509.25996|CAST]]: 2:4 Llama-2-7B at +0.09 PPL with 2% of pretraining tokens, ≈1.2K H100-h.
* **Evaluate beyond perplexity.** On [[2505.19433|ACBench]] (ICML'25), even 4-bit models keep tool use but lose
  10–15% on real-world agentic applications. [[2606.09080|Beyond FLOPs]] maps which pruning type actually speeds up
  prefill vs decode.

### Hand ranking

| # | Paper | Kind | Key idea | Result / cost |
| ---: | --- | --- | --- | --- |
| 1 | [[2603.05232\|SlideSparse]] | Pattern + kernels | (2N−2):2N sparsity executed on 2:4 sparse tensor cores via sliding windows; integrated in vLLM across FP4/INT8/FP8/BF16 | 6:8 reaches 1.33× (≈ the 4/3 bound) and keeps reasoning accuracy that 2:4 destroys |
| 2 | [[2505.23049\|DenoiseRotator]] (NeurIPS'25) | One-shot, plug-in | Learn orthogonal rotations that concentrate importance, then prune with SparseGPT/Wanda | 2:4 Llama-3-70B perplexity gap −58% (8.1 → 3.4) |
| 3 | [[2509.25996\|CAST]] | Sparsity-aware training | Continuous, differentiable 2:4 training + sparse-model scaling law | Near-lossless 2:4 Llama-2-7B with 2% of pretraining tokens (~1.2K H100-h est.) |
| 4 | [[2511.13061\|MACKO]] | Kernel | Storage format + SpMV for 30–90% unstructured sparsity | First real memory and speed win at 50% (1.5× on Llama-2-7B) |
| 5 | [[2510.05528\|ARMOR]] (ICLR'26) | One-shot 2:4 | Factorize W ≈ A·(2:4 core)·B with small block-diagonal wrappers | Beats state-of-the-art 2:4 while keeping 2:4 speed and memory |
| 6 | [[2606.10445\|SpenseGPT]] | One-shot hybrid | Choose which GEMMs run sparse vs dense | 1.2× end-to-end decode on B200 FP8 at preserved accuracy (Qwen3-32B) |
| 7 | [[2502.00258\|ProxSparse]] (ICML'25) | Learned 2:4 mask | Regularized mask learning with a proximal solver; no weight updates | Best learned semi-structured masks across 7 models |
| 8 | [[2503.04992\|Wanda++]] (ACL'25) | One-shot | Regional (block-level) gradients + regional optimization | Up to 32% better perplexity than Wanda; <10 min for 7B on one H100 |
| 9 | [[2505.19433\|ACBench]] (ICML'25) | Evaluation | Agentic capabilities under quantization and pruning | 4-bit quantization keeps workflow/tool use (1–3% drop) but real-world agentic tasks drop 10–15%; pruning compared on the same suite |
| 10 | [[2603.05168\|Sparse-BitNet]] | Sparsity × ternary | 1.58-bit models tolerate N:M sparsity better than FP models | Joint ternary + N:M training; up to 1.30× with a custom sparse tensor core |
| 11 | [[2607.08786\|Moderately unstructured SpMM]] (DAC'26) | Kernel | Hybrid sparse-tensor-core + CUDA-core execution at ~50% | 1.64× over SpInfer kernel, 1.41× end-to-end over FlashLLM |
| 12 | [[2606.09080\|Beyond FLOPs]] | Benchmark | GEMM-centric taxonomy of pruning and its *real* acceleration | Where each pruning family is Pareto-optimal in prefill vs decode |

**Also useful.**
* Mask learning: [[2608.23048]], [[2605.06402|SparseForge]], [[2501.18015]] (proximal 2:4), [[2509.23410|PATCH]]
  (tile-level hybrid sparsity).
* One-shot variants: [[2603.05878|ROSE]], [[2501.16376|SwiftPrune]], [[2503.22451|STADE]], [[2501.18980]],
  [[2608.00481|F-Wanda]], [[2607.07557|PALS]] (layerwise ratios).
* Joint pruning + quantization: [[2506.10205|AWP]] (ICML'25), [[2502.01705]] (binarization + N:M).
* Flat minima: [[2506.06866|SAFE]] (ICML'25).
* Calibration data: [[2606.03328]] (multi-source calibration matters at high sparsity).
* Mamba: [[2505.08299]].

**Recommendation.**
* *Runtime:* support 2:4 **and** 6:8/8:16 sparse GEMMs with FP8/FP4 operands and per-GEMM sparse/dense selection. For
  batch-1 decode, a low-sparsity SpMV (MACKO-style) is the only way unstructured 50% pays off. Don't expect more than
  ~1.2–1.5× end to end.
* *Model builders:* if you want sparsity, **train for it** (CAST/ProxSparse, 1–5% of pretraining tokens) and prefer
  milder N:M. Stack it on quantization only after checking agentic evaluations.
