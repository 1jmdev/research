**Verdict.** Mixed precision is how you hit an arbitrary memory budget, such as "fit a 70B model in 24 GB".

* **Granularity.**
  * Layer-wise allocation is solved: AMQ search, closed-form (BAQ), DP (SignRoundV2), and information-theoretic
    fractional-bit allocation (Q-Palette).
  * The frontier is **block-level** allocation inside a matrix that stays hardware-aligned (ScaleBITS, SFMP, MicroMix).
  * Multi-precision "any-bit" models are one checkpoint serving many bit-widths (MatGPTQ, AnyBCQ, QuEPT).
* **Caution.** [[2609.01587]] shows sensitivity proxies often pick the wrong layers. Always validate against a
  global-uniform baseline at the same budget.

| # | Paper | Kind | Key idea | Headline / cost |
| ---: | --- | --- | --- | --- |
| 1 | [[2509.20214\|Q-Palette]] (NeurIPS'25) | PTQ + kernels | Optimal bit allocation for Gaussianized (rotated) weights. Builds a **palette of fractional-bit quantizers** (trellis, vector, scalar) with optimized CUDA kernels and picks per layer under memory and latency budgets | Pareto-dominant weight-only PTQ; data-aware variant ~79 GPU-h, data-free near zero |
| 2 | [[2509.12019\|AMQ]] (EMNLP'25) | AutoML | Search space pruning, quantization proxy, quality predictor, iterative search over 10¹⁰⁰ configurations | Best-per-memory model; ~5 GPU-h search |
| 3 | [[2508.02343\|MicroMix]] (ICLR'26) | PTQ + kernel | Per-channel **MXFP4/MXFP6/MXFP8** mix with a single Blackwell GEMM producing BF16 | Uses FP4 tensor cores where safe |
| 4 | [[2501.01144\|BlockDialect]] (ICML'25) | Format | Per-block choice from a "formatbook" of FP4 variants (DialectFP4), integer-friendly | Energy-efficient W4A4 |
| 5 | [[2602.17698\|ScaleBITS]] | PTQ | Bi-directional channel reordering, then **hardware-aligned block partitions** and a scalable global allocator | Continuous Pareto front below 4 bits |
| 6 | [[2602.03537\|MatGPTQ]] | PTQ | One-shot **Matryoshka** INT8 → 4 → 2 via bit slicing, with open kernels | One checkpoint, many precisions |
| 7 | [[2504.14152\|FGMP]] (NVIDIA) | PTQ + HW | Fisher-weighted block-level choice of FP8 vs FP4 for weights and activations | Fisher computed in <3 min on an A100 for 7B |
| 8 | [[2603.17891\|RAMP]] | RL policy | SAC policy over layer features; transfers zero-shot across models; GGUF export | Useful for llama.cpp pipelines |
| 9 | [[2511.06516\|TAQ]] | PTQ | Task-aware allocation from hidden states of task prompts | 1.5–6 A40-h per model |
| 10 | [[2608.30564\|Q-Strata]] (EMNLP'26) | MoE | Bi-level allocation: per-block Pareto frontier cache plus a model-level outer search over experts | MoE-specific; see also [[2606.00079\|BitsMoE]] |
| 11 | [[2506.12024\|FlexQuant]] (EMNLP'25) | Dynamic | **Token-wise** precision switching guided by entropy, KL-based layer choice | Runtime precision scheduling |
| 12 | [[2504.03717\|RaanA]] | PTQ | RaBitQ-H randomized VQ + optimal AllocateBits; little calibration | Fast, flexible bit budgets |

Also notable:
* [[2608.27513|DAMP]]: decay-aware mixed precision for linear-attention **recurrent state**.
* [[2604.13440]]: forward-only KL sensitivity for hybrid SSM-Transformers.
* [[2607.02893|Variable Bit-width Quantization]]: learned per-group precision.

**Runtime recommendations.**
1. Support **per-layer** heterogeneous weight formats in the model loader (GGUF already does). Also support **per-block
   format flags** in at least one kernel: one bit in the scale selects the grid (IF4, BlockDialect, MicroMix style).
2. Prefer bit-plane or Matryoshka layouts if you want run-time precision switching. They let a scheduler degrade
   precision under load without reloading weights.
