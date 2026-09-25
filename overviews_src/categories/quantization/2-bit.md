**Verdict.** 2-bit is where post-training quantization (PTQ) either works or collapses.

* **Weight-only 2-bit on ≥30B models is usable.** BPDQ runs Qwen2.5-72B at 2 bits on a single RTX 3090 with 83.9%
  GSM8K, against 90.8% at FP16. KronQ keeps Llama-3-70B at 7.93 PPL where GPTQ diverges.
* **Small models (≤8B) at 2 bits still need QAT or VQ.**
* **Vector and trellis quantizers remain the accuracy frontier** (QTIP/AQLM lineage, Leech lattice), but 2026 work
  shows most of the gap to scalar quantization is an *optimization* gap (UniSVQ, GSQ, BPDQ).
* **Scalar formats that run on integer kernels are catching up.**

2-bit *weights* with 4-bit activations and KV cache (W2A4KV4) needs QAT (RCP). The "Two Failure Modes" paper explains
why 2-bit PTQ fails: early-layer **computation collapse**, which training-free repair cannot fix.

### Hand ranking

| # | Paper | Kind | Key idea | Headline | Cost (hand-checked) |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2607.07964\|KronQ]] (COLM'26) | PTQ | GPTQ objective with **K-FAC Hessian H_X ⊗ H_G**: output-side incoherence rotation plus inter-layer bit allocation; H_G cancels in the update, so the cost stays GPTAQ-like | Llama-3-70B 2-bit PPL 7.93 (GPTQ > 2000) | One backward pass over the calibration set, then GPTQ cost: a few GPU-h for 70B |
| 2 | [[2602.04163\|BPDQ]] | PTQ | **Variable grid** from bit-planes × scalar coefficients, refined with a Hessian | Qwen2.5-72B 2-bit on one RTX 3090: GSM8K 83.9% (GPTQ/AWQ < 41%) | Single-GPU PTQ |
| 3 | [[2606.10520\|UniSVQ]] (ICML'26) | PTQ | Codewords = affine transform of **integer lattices**, so VQ-like accuracy runs on INT kernels | Beats SQ and VQ baselines at 2 bits | Block-wise fine-tune |
| 4 | [[2512.04746\|SignRoundV2]] (Intel) | PTQ | Gradient × error sensitivity → DP bit allocation; pre-tuning scale search | Near-lossless at 4–5 bits, robust at 2 bits and MXFP4. Ships in AutoRound | 1×A100 |
| 5 | [[2605.29843\|HARP]] | PTQ add-on | **Learned butterfly rotation** initialized at randomized Hadamard, drop-in for QuIP#-style incoherence | Consistent gains over RHT at 2–4 bits | 70B fit: **31–80 H100-h**; 8B stats ~3 H100-h |
| 6 | [[2602.05367\|RaBiT]] (ICML'26) | QAT | Residual binarization with enforced residual hierarchy (fixes inter-path co-adaptation). Matmul-free | Matches VQ at 2 bits, **4.49×** vs FP16 on an RTX 4090 | QAT, about 200M tokens |
| 7 | [[2606.10531\|LC-QAT]] (ICML'26) | QAT | VQ-QAT via learned affine maps over discrete vectors; no codebook lookup in the training forward pass | Data-efficient 2-bit QAT | — |
| 8 | [[2502.15779\|RCP]] (EMNLP'25) | QAT | Rotation + learnable non-uniform partitions + W2A4 GEMV kernel | First **W2A4KV4**: Llama-2-7B +2.84 PPL, 5.29× memory reduction | QAT |
| 9 | [[2505.00850\|ICQuant]] | PTQ add-on | Index-code outliers at **~0.3 bit** overhead instead of ~1 bit | Lifts plain scalar quantizers to state-of-the-art at 2–3 bits | 1×RTX 4090 |
| 10 | [[2509.09679\|ButterflyQuant]] | PTQ | Learnable butterfly orthogonal transforms (Hadamard is a special case); O(n log n) | Best rotation-based 2-bit | 4×H100 |
| 11 | [[2604.19884\|Two Failure Modes]] (ACL F'26) | Analysis | Signal degradation (repairable) vs **computation collapse** (not repairable without training) | Tells you when to stop trying PTQ | — |
| 12 | [[2510.03274\|Quant-dLLM]] | PTQ | 2-bit for **diffusion LLMs**: masked calibration simulation plus any-order quantizer | First usable 2-bit dLLMs | 1×A800 |
| 13 | [[2604.08118\|OA-EM]] | PTQ (AQ) | Output-aware EM codebook **initialization** fixes AQLM's 2-bit failures | Better after PV-tuning | — |
| 14 | [[2504.07389\|TaCQ]] (COLM'25) | Mixed PTQ | Keep "task circuit" weights in 16-bit | Big gains at 2–3 bits | Qwen2.5-7B: ~3.3 h on A6000 (≈1 H100-h) |
| 15 | [[2605.24144\|EVA]] (ISCA'26) | Hardware | VQ-decode GEMV architecture without codebook bank conflicts | For accelerator designers | — |

Also relevant:
* [[2602.21233|AngelSlim]]: Tencent's toolkit; HY-1.8B-int2 is the "first industrially viable 2-bit model".
* [[2506.09104|UPQ]]: FP16→INT4→INT2 progressive plus distillation QAT for instruction-tuned models, 30B tokens.
* [[2609.02652]]: fused Leech-lattice decode kernels.
* [[2606.02823|Qift]]: shift-friendly zero-free W2 grid for rotated W2A4.

**Recommendations.**
1. **Runtime.** Support a 2-bit group-wise scalar format with **non-uniform levels** (LUT of 4 values per group:
   BPDQ, UniSVQ, GANQ-style). Pure uniform INT2 is the weakest option. Add one VQ decode path (QTIP/AQLM-style, E8/Leech
   lattice) if you target maximum accuracy per byte.
2. **Pipeline.** Rotation (Hadamard or learned butterfly) → Hessian-aware solver (GPTQ/GPTAQ/KronQ) → outlier index
   coding or mixed precision on sensitive layers → optional short QAT/distillation.
3. **Evaluation.** Do not trust perplexity at 2 bits. Check reasoning (GSM8K/AIME) and long-context tasks; collapse shows
   up there first.
