**Verdict.** INT4 is solved for **weight-only** (W4A16): GPTQ/AWQ-class methods are near-lossless on ≥7B models. The
active frontier is **W4A4 / W4A4KV4**, where the best recipe is:

1. a function-preserving transform: Hadamard, then learned rotation, scaling, or the provably near-optimal WUSH transform;
2. a GPTQ-style solver;
3. some fine-grained grouping.

Three empirical findings matter more than any single method:

* **Reasoning models:** W8A8 and W4A16 are lossless. W4A4 and KV4 hurt long chain-of-thought (CoT) much more than
  perplexity shows ([[2504.04823]]).
* **Long context:** 4-bit methods can drop up to **59%** on >64K-token inputs while 8-bit holds (~0.8%) ([[2505.20276]]).
* **Agents:** 4-bit models can match on flat scores yet amplify failures in multi-turn tool use ([[2607.27275]]).

### Hand ranking

| # | Paper | Kind | Key idea | Headline | Cost |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2504.04823\|Quantization Hurts Reasoning?]] (COLM'25) | Study | Weights, KV cache and activations on R1-distill, QwQ and Qwen3, 1.5–70B | **W8A8 and W4A16 are lossless**; lower bits hurt hard tasks and small models; quantized models also think longer | — |
| 2 | [[2512.00956\|WUSH]] (ICML'26) | PTQ transform | **Closed-form near-optimal block transform**: Hadamard backbone plus a data-dependent second-moment part; non-orthogonal; covers INT and FP | Beats the best Hadamard baselines at W4A4; fused kernel | Closed form (cheap) |
| 3 | [[2501.13987\|OSTQuant]] (ICLR'25) | PTQ | Learnable orthogonal + scaling transforms guided by a quantization-space-utilization metric (QSUR) and KL-top loss | 99.5% of FP at W4; closes 32% of the W4A4KV4 gap on Llama-3-8B | **Llama-3-8B in 20 min on one A800** (≈0.1 H100-h) |
| 4 | [[2502.09720\|NestQuant]] (ICML'25) | PTQ | **Gosset (E8) nested-lattice** quantizer for W, A and KV; information-theoretically motivated | Llama-3-8B W4A4KV4 PPL 6.6 vs 7.3 for SpinQuant/OSTQuant | — |
| 5 | [[2603.04956\|WaterSIC]] | PTQ theory | Shows GPTQ can be arbitrarily far from the information-theoretic limit; **waterfilling per-column rates** stay within 0.255 bit of it | New state of the art from 1 to 4 bits | — |
| 6 | [[2511.10645\|ParoQuant]] (ICLR'26) | PTQ + kernel | Hardware-efficient **independent Givens (pairwise) rotations** plus channel scaling, fused kernel | Targets reasoning models; small overhead | — |
| 7 | [[2507.04610\|any4]] (ICML'25, Meta) | Format | Learned **per-row 4-bit codebook** (LUT); tinygemm library | Beats int4/fp4/nf4 with no preprocessing; calibrate on **one sample** | — |
| 8 | [[2505.14302\|QAT Scaling Law]] | Study | 268 W4A4 QAT runs: error falls with N, rises with tokens D and group size | **Keep FC2 (down_proj) input at 8-bit** in W4A4 | 276K A100-h (≈**88K H100-h**) total for the study |
| 9 | [[2503.22879\|Quamba2]] (ICML'25) | PTQ (SSM) | W8A8/W4A8/W4A16 for Mamba-1/2 via sort-and-cluster and per-state-group quantization | Reference for quantizing SSMs | — |
| 10 | [[2502.00425\|MQuant]] (MM'25) | PTQ (MLLM) | Modality-specific **static** scales; flags that online Hadamard creates new outliers | <1% loss W4A8; +23% prefill, +100% decode | — |
| 11 | [[2509.11177\|OBR]] | Joint | Hessian-based compensation between pruning and quantization | W4A4KV4 + 50% sparsity, up to 4.72× | Training-free |
| 12 | [[2601.22347\|PeRQ]] | PTQ | Non-asymptotic theory of **block Hadamard**; permute to spread ℓ₁ mass before block rotation | Better small-block rotations | Greedy calibration |
| 13 | [[2503.01483\|KurTail]] (EMNLP'25) | PTQ | Kurtosis-minimizing learned rotation, layer-wise | +13.3% MMLU vs QuaRot, +2.6% vs SpinQuant | 70B in ~1 H100-h |
| 14 | [[2608.20953\|Quantization-Aware Healing]] | Recipe | Distill the pruned + 4-bit student from the **original** (not the pruned BF16) model | GPT-OSS 120B→60B→MXFP4 beats its BF16 source on 7 of 9 benchmarks | — |
| 15 | [[2608.02703\|ARCHead]] | PTQ | Compresses the usually-BF16 **LM head** (quantized low-rank core + INT4 residual) | 3.7–3.9× smaller head, rel. PPL 1.007 | — |

Also useful:
* [[2505.11574]]: 332 examples and 5 minutes of fine-tuning restore W4 math reasoning.
* [[2506.12044]]: residual-stream magnitude predicts which inputs break.
* [[2606.08761|APEX4]]: pure W4A4 kernels via intra-SM rebalancing.
* [[2604.02556]]: fast NF4 dequantization.
* [[2608.25188]]: "The Great Inversion", a survey of transform/format co-design.

**Runtime checklist.**
1. **Kernels.** Ship W4A16 (Marlin/Machete-style) and W4A8 with INT8 activations. W4A4 is worth it only for
   prefill-heavy or batch serving.
2. **Rotations.** Support an online **block-Hadamard** (size 32–128) fused into the preceding RMSNorm or GEMM epilogue.
   It is needed for W4A4, KV4 and FP4.
3. **LM head.** Keep it at ≥8-bit or use ARCHead-style compression; naive INT4 of the head costs 14–16% perplexity.
4. **Down projection.** Keep the down_proj input at 8 bits in W4A4 (the QAT scaling-law result).
5. **Evaluation.** Gate every quantized release on long-context (RULER at 64K+), reasoning (AIME/GPQA) and agentic
   evaluations, not on perplexity.
