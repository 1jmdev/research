**Verdict.** MoE models are memory-bound by their *total* parameters, so compression targets the expert pool. Four levers,
in order of reliability:

1. **Expert-level mixed-precision quantization.** Cold experts get fewer bits, hot and super experts more. Calibrate so
   every expert sees data.
   * [[2505.03804|MoEQuant]]: expert-balanced self-sampling + affinity-guided quantization; +10 HumanEval points on
     DeepSeekMoE-16B at 4-bit.
   * [[2605.23078|GEMQ]] (ICML'26): global linear program over experts + router fine-tuning.
   * [[2508.01625|EAC-MoE]]: calibrate routers after quantization, because low-bit weights shift expert selection.
   * [[2511.15015|DynaExq]]: *runtime* precision switching of hot/cold experts under an HBM budget; Qwen3-80B
     73.1 → 77.6% vs static PTQ at equal memory.
   * See also [`quantization/mixed-precision`](../../quantization/mixed-precision/README.md).
2. **One-shot expert pruning beats expert merging for generative tasks.**
   * [[2510.13999|REAP]]: router-weighted expert activation pruning, 20B–1T models; near-lossless code generation on
     Qwen3-Coder-480B and Kimi-K2 at **50% of experts removed**.
   * [[2604.04356|REAM]]: merge-then-prune; the MC vs generation trade-off depends on the calibration mix.
   * Domain-specific pruning is very effective: [[2504.06792|EASY-EP]], [[2605.28042]] (translation specialists keep
     25–50% of experts).
   * **Always re-calibrate the router** after pruning or merging ([[2603.02217]]).
3. **Shared-basis factorization of experts** (experts are highly redundant):
   * [[2508.05257|MoBE]]: shared basis matrices; −24–30% parameters on Qwen3-235B, DeepSeek-V3 and Kimi-K2 at 1–2%
     accuracy drop;
   * [[2502.17298|D²-MoE]] (ICML'25): shared base + SVD deltas;
   * [[2506.23266|Sub-MoE]]: subspace merging.
4. **Prune + distill** for a *smaller sibling MoE*:
   * [[2506.18349|SlimMoE]] (Microsoft): Phi-3.5-MoE 42B → Phi-mini-MoE 7.6B;
   * [[2605.08738|SlimQwen]]: Qwen3-Next-80B-A3B → 23B-A2B; pruning a pretrained MoE beats training from scratch, and
     progressive schedules beat one-shot;
   * [[2605.28207]]: MoE → *dense* students via diversity-aware expert scoring, +6.3 pp over dense-to-dense pruning
     after ~4B tokens.

**Compute reduction (not memory):** [[2605.18643|ZEDA]] adds zero-output experts + self-distillation so post-trained
MoEs skip >50% of expert FLOPs (1.2× end to end). [[2511.15690|MoDES]] does training-free expert skipping for
multimodal MoEs.

### Hand ranking

| # | Paper | Lever | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.13999\|REAP the Experts]] (Cerebras) | One-shot expert pruning | Score = router gate × expert activation norm (bounds reconstruction error) | Near-lossless code at 50% experts pruned on Qwen3-Coder-480B / Kimi-K2; beats merging on generative tasks |
| 2 | [[2508.05257\|MoBE]] | Shared basis | W = A_i·(Σ α_ij B_j): expert-specific A, shared basis B | −24–30% parameters on 235B–1T MoEs at 1–2% accuracy drop |
| 3 | [[2511.15015\|DynaExq]] | Runtime mixed precision | Online precision allocation for hot/cold experts under a hard HBM budget | +4.5 pts vs static PTQ at equal memory; 2.73× over offloading at batch 32 |
| 4 | [[2505.03804\|MoEQuant]] | PTQ | Expert-balanced self-sampling calibration + affinity-guided quantization | +10 pts HumanEval on DeepSeekMoE-16B at 4-bit |
| 5 | [[2605.08738\|SlimQwen]] | Prune + distill at pretraining scale | Depth, width and expert pruning of Qwen3-Next + large-scale continual KD; progressive schedules | Qwen3-Next-80B-A3B → 23B-A2B, competitive |
| 6 | [[2506.18349\|SlimMoE]] | Prune + staged distill | Slim experts in stages instead of dropping them | Phi-mini-MoE (7.6B/2.4B active) and Phi-tiny-MoE released |
| 7 | [[2605.18643\|ZEDA]] | Dynamic compute | Zero-output experts + two-stage self-distillation + group balance loss | >50% of expert FLOPs skipped at marginal loss; 1.2× end to end |
| 8 | [[2502.17298\|D²-MoE]] (ICML'25) | Delta decomposition | Fisher-weighted shared base + SVD-compressed deltas + semi-dynamic pruning | +13% over other compressors at 40–60% compression |
| 9 | [[2605.23078\|GEMQ]] (ICML'26) | Mixed-precision PTQ | Global LP over expert bit-widths + router fine-tuning | Better than per-layer allocation at the same average bits |
| 10 | [[2508.01625\|EAC-MoE]] | Quant + pruning | Router calibration against quantization-induced selection bias; task-frequency expert pruning | Lower memory and faster inference with minimal loss |
| 11 | [[2605.28207\|MoE → dense]] | Distillation | Diversity-aware expert scoring to build a dense student | +6.3 pp over dense→dense at matched parameters after ~4B tokens |

**Also useful.**
* Pruning scoring: [[2606.15716]] (unified expert-scoring formulation), [[2603.06003|EvoESAP]] (speculative-acceptance
  proxy for non-uniform pruning), [[2509.16105|DiEP]], [[2509.22299|HEAPr]], [[2509.10377|DERN]], [[2507.00390|MoNE]],
  [[2511.19822|Mosaic Pruning]].
* Merging: [[2510.14436|MergeMoE]], [[2511.04805|PuzzleMoE]] (bit-packed sparse expert merging), [[2510.16138]]
  (Nash bargaining).
* Quantization: [[2602.11184|KBVQ-MoE]] (vector quantization), [[2509.02512|MoPEQ]], [[2606.05688]]
  (routing-consistent quantization).

**Recommendation.**
* *Runtime:* support **per-expert bit-width** (e.g. FP8 / INT4 / 2-bit) with grouped-GEMM kernels that handle mixed
  formats, plus optional runtime re-quantization of cold experts (DynaExq).
* *Deployment:* for domain deployments, prune 25–50% of experts with REAP using in-domain calibration, then
  re-calibrate the router. Keep super experts and shared experts at full precision.
