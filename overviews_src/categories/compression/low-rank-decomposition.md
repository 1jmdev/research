**Verdict.** Post-training SVD compression of LLM weights (W ≈ U·Vᵀ) is a crowded field of 80+ papers with steady but
**small** gains. At equal memory, 20–40% SVD compression usually loses to 4-bit weight quantization, and naive low-rank
factors don't even cut *peak* memory or latency without a dedicated runtime. Where the progress is:

1. **Better truncation and rank allocation.** Whiten by calibration activations (SVD-LLM lineage), then optimize
   per-matrix ranks:
   * [[2503.12340|SVD-LLM V2]]: truncation-loss-based heterogeneous ratios;
   * [[2502.02723|Dobi-SVD]]: differentiable truncation positions + activation-side reconstruction;
   * [[2602.02848|ZS-SVD]]: global zero-sum component selection;
   * [[2604.01609|Swift-SVD]]: closed-form optimum via one eigendecomposition, 3–70× faster compression;
   * [[2505.12942|A³]]: attention-aware analytic factorization, Llama-3.1-70B PPL 4.69 vs 7.87 for the prior best.
2. **Beyond plain low rank.**
   * Sparse dictionary learning: [[2509.22075|CoSpaDi]], [[2508.04581|MASA]] (−66.7% attention parameters by sharing
     dictionary atoms across layers).
   * Sparse + low-rank: [[2603.01376|3BASiL]].
   * Quantized + low-rank, W ≈ Q + LR: [[2512.03383|UniQL]] for Transformers, SSMs and hybrids, 4–5.7× memory,
     2.7–3.4× throughput on edge; [[2506.02077]].
3. **Runtime co-design is mandatory.** [[2605.08314|FlashSVD v1.5]] shows factorized models only speed up with
   phase-specific kernels, packed MLPs, dense-KV decode and CUDA graphs: up to 2.55× decode and 1.48× on average.
   [[2508.01506|FlashSVD]] fixes activation memory with streaming rank-aware kernels.
4. **Native low-rank *pretraining* is becoming viable.** [[2602.12429|Spectron]] (ICML'26) bounds the spectral norm of
   updates to stop loss spikes and gives compute-optimal scaling laws for factorized transformers. See also
   [[2505.21732|LaX]] and [[2508.02668|LOST]].
5. **Low-rank for reasoning speed.** [[2505.07861|Caprese]] recovers reasoning lost by efficient FFN approximations
   with low-rank distillation (~2B fewer active parameters on 8–9B models; 16% lower TTNT).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2605.08314\|FlashSVD v1.5]] | Runtime | Thin low-rank serving path: phase-specific kernels, dense-KV decode, packed MLP, per-layer CUDA-graph replay | Up to 2.55× decode / 2.39× end to end; **the piece that makes SVD compression pay off** |
| 2 | [[2503.12340\|SVD-LLM V2]] (NAACL'25) | Post-training SVD | Heterogeneous per-matrix ratios from theoretical truncation loss + loss-optimized truncation | State of the art among SVD methods across 5 LLMs |
| 3 | [[2502.02723\|Dobi-SVD]] | Differentiable SVD | Learn truncation positions; reconstruct weights from truncated activations; handles SVD "injection" loss | Principled activation-side SVD; strong at high compression ratios |
| 4 | [[2505.12942\|A³]] | Attention-aware low rank | Decompose QK, OV and MLP functionally; no extra kernel launches (reduced head dim) | Llama-3.1-70B: PPL 4.69 vs 7.87 previous state of the art at equal budget; also compresses KV |
| 5 | [[2512.03383\|UniQL]] | Quant + low-rank for edge | Structured weight sorting, quantization-aware SVD, state-aware sorting for SSMs; one model, many sizes | 4–5.7× memory, 2.7–3.4× throughput within 5% accuracy (Llama3, Qwen2.5, Mamba2, Nemotron-H) |
| 6 | [[2602.12429\|Spectron: native low-rank pretraining]] (ICML'26) | Pretraining | Spectral renormalization + orthogonalization of factor updates; no full-rank guidance | Stable factorized pretraining with compute-optimal scaling laws |
| 7 | [[2509.22075\|CoSpaDi]] | Dictionary learning | Calibration-guided sparse dictionary instead of a low-rank basis | Beats SVD and structured pruning at 20–40% compression; composes with quantization |
| 8 | [[2604.01609\|Swift-SVD]] | Fast post-training | Aggregate output-activation covariance, single eigendecomposition, effective-rank-based allocation | Optimal layer-wise approximation; 3–70× faster compression |
| 9 | [[2505.07861\|Caprese]] | Low-rank distillation | Low-rank corrections distilled to recover reasoning lost by FFN sparsification | ~2B fewer active parameters, >16% lower TTNT, shorter responses |
| 10 | [[2603.01376\|3BASiL]] | Sparse + low-rank | Algorithmic S+LR decomposition with transformer-matching | 2:4 + rank 64: >30% smaller PPL gap on Llama-3-8B; 2.5× faster compression |
| 11 | [[2508.04581\|MASA]] | Cross-layer sharing | Q/K/V/O as combinations of shared dictionary atoms | −66.7% attention parameters at parity (small scale) |
| 12 | [[2508.01506\|FlashSVD]] | Runtime (memory) | Stream factor tiles through SRAM; never materialize full activations | −70% peak activation memory |

**Also useful.**
* Rank allocation: [[2502.01403|AdaSVD]], [[2602.03051|SAES-SVD]], [[2506.20353|DipSVD]], [[2509.25622]],
  [[2510.19389|ARA]], [[2512.13733]], [[2512.03062]].
* Fisher-weighted / Hessian-aware: [[2505.17974]], [[2604.00821|Optimal Brain Decomposition]].
* Activation-space methods: [[2505.23966|FLAT-LLM]], [[2507.03828|IMPACT]].
* Tensor decompositions: [[2606.03465]] (rethinking their role), [[2505.14871|Saten]].
* Training-side: [[2509.18993|CR-Net]], [[2512.01980|Low-Rank Prehab]] (prepare networks before SVD).
* Delta compression of fine-tunes: [[2604.16940|D-QRELO]].

**Recommendation.**
* *Runtime builders:* only invest in low-rank serving if you target edge memory tiers or combine it with quantization
  (Q + LR). Then you need FlashSVD-style fused factor kernels, or the savings disappear.
* *Model builders:* post-hoc SVD is a last resort after quantization and pruning + distillation. If you want low-rank
  efficiency, train it natively (Spectron) or use MLA/low-rank attention designs from the start.
