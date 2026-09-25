**Verdict.** dLLM inference has its own toolbox. The canonical training-free stack is **Fast-dLLM**:
* block-wise **approximate KV cache**, with DualCache for prefix and suffix;
* **confidence-threshold parallel unmasking**.

On top of that come:
* adaptive caching: dKV-Cache, dLLM-Cache, Elastic-Cache, d²Cache;
* early answer commitment (Prophet);
* sparse attention (SparseD, Sparse-dLLM).

To actually **beat AR speed** you need training:
* **D2F**: discrete diffusion forcing gives inter-block pipelined parallelism, 2.5× over LLaMA3/Qwen2.5;
* **Fast-dLLM v2**: block-diffusion adaptation of AR models in ~1B tokens, 2.5× over AR;
* **dParallel / d3LLM / DMax**: certainty-forcing, pseudo-trajectory or self-refinement distillation for aggressive
  parallelism;
* **LLaDA2.1**: token-to-token editing lets it lower unmasking thresholds safely.

Always check speedup claims on **ParallelBench**. Parallel decoding degrades when token dependencies are strong, which
standard math and code benchmarks under-detect.

### Hand ranking

| # | Paper | Training? | Key idea | Result / cost |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.22618\|Fast-dLLM]] (NVIDIA) | Free | Block-wise approximate KV cache (DualCache) + **confidence-aware parallel decoding** (theory: safe when confidence exceeds a threshold) | Up to 27.6× over vanilla LLaDA/Dream with small accuracy loss. **Baseline for every dLLM engine** |
| 2 | [[2509.26328\|Fast-dLLM v2]] | ~1B tokens | Block diffusion + complementary mask adapts **AR → block dLLM**; hierarchical block/sub-block cache | Matches the AR model with up to 2.5× faster decoding. 7B: 64 A100 × 12 h ≈ **250 H100-h** |
| 3 | [[2508.09192\|D2F: Discrete Diffusion Forcing]] | Distill | Block-AR with KV cache + predict later blocks before earlier ones finish; asymmetric distillation + pipelined parallel decoding | **First open dLLM faster than AR** (2.5× over LLaMA3/Qwen2.5); ~31 H100-h distill (8 A100 × 12 h) |
| 4 | [[2602.08676\|LLaDA 2.1]] | Model | **Token-to-token editing** added to mask-to-token decoding; Speedy vs Quality modes | Aggressive thresholds, with errors corrected by editing |
| 5 | [[2505.15781\|dKV-Cache]] (NeurIPS'25) | Free | **Delayed**, conditioned KV caching (cache a token's KV a step after it is decoded) | 2–10× speed, near-lossless |
| 6 | [[2510.14973\|Elastic-Cache]] | Free | Recompute KV only where drift is large: skip shallow layers, cache distant MASKs block-wise; use the most-attended token as a drift bound | Large speedups at the same accuracy |
| 7 | [[2508.19982\|Prophet]] | Free | **Early answer convergence**: answers are fixed after ~half the steps; commit early | Up to 3.4× fewer steps |
| 8 | [[2509.26488\|dParallel]] | Distill | **Certainty-forcing distillation** on the model's own trajectories | LLaDA GSM8K steps 256→30 (8.5×) |
| 9 | [[2601.07568\|d3LLM]] (ICML'26) | Distill | Pseudo-trajectory distillation + entropy-based multi-block decoding with KV refresh | Ultra-fast dLLM with a good accuracy-parallelism balance |
| 10 | [[2511.08923\|TiDAR]] (NVIDIA) | Architecture | **Diffusion drafting + AR sampling in one forward pass** (structured masks) | AR-level quality at 4.7–5.9× throughput. See also [[2609.04010]] |
| 11 | [[2510.04767\|ParallelBench]] (ICLR'26) | Benchmark | Information-theoretic analysis + tasks that expose dependency-breaking | **Use it to validate parallel decoding** |
| 12 | [[2506.00413\|APD]] (NeurIPS'25) | Free + small AR | Multiplicative mixture of dLLM marginals and a small AR model's joint (inverse speculative decoding) | Adaptive parallelism at high quality |
| 13 | [[2604.08302\|DMax]] | Training | On-policy uniform training + **soft parallel decoding** in embedding space (self-revision) | Aggressive parallelism without error accumulation |
| 14 | [[2509.24014\|SparseD]] / [[2508.02558\|Sparse-dLLM]] | Free | Head-specific sparse patterns reused across steps; delayed bidirectional sparse caching | Long-context dLLM speedups |
| 15 | [[2510.08666\|dInfer]] | System | Modular dLLM inference framework (model, diffusion manager, decoder, KV manager) | >1,100 tok/s on HumanEval (LLaDA-MoE, 8×H800). A practical engine reference |

**Serving engine design for dLLMs.**
1. **Block scheduler.** Current block + optional look-ahead blocks (D2F/multi-block).
2. **KV policy per block.** Exact for committed blocks (block-causal models); approximate or delayed for the active
   block.
3. **Unmasking policy.** Confidence threshold, entropy bound (EB-sampler), or learned.
4. **Editing/remasking step.** For T2T-capable models (LLaDA 2.1) and ReMDM samplers.
5. **Fused bidirectional-within-block attention kernels.** See [[2609.26796|Flash-dLLM]] for IO-aware caching.
6. **Batching.** Batch across requests at the *block-step* level.
