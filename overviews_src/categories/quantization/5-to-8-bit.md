**Verdict.** 8-bit is lossless for inference in practice.

* **Formats.** W8A8 in either INT8 or FP8 is near-lossless for reasoning, long context and agents. MXINT8 slightly
  beats MXFP8 ([[2510.25602]]).
* **Where the research is.** It has moved to **systems questions**:
  * one checkpoint serving two precisions (NestedFP, QStore);
  * INT6 kernels (FlexQ);
  * 8-bit for state-space duality (SSDi8);
  * where to spend "the next bit" (global beats targeted).
* **Distribution-lossless** (sampling the same text as BF16) needs about 5–6.6 bits ([[2605.02404]]). 6-bit is the
  sweet spot if you need an exact-behaviour replica, for example as a speculative-decoding draft or an RL rollout model.

| # | Paper | Why it matters |
| ---: | --- | --- |
| 1 | [[2506.02024\|NestedFP]] (NeurIPS'25) | **FP8 overlaid inside FP16 weights**: one memory footprint serves both precisions. Switch to FP8 under load spikes for SLOs with no second copy. A great serving trick |
| 2 | [[2609.01587\|Structure of Quantization Damage]] (NeurIPS'26) | Causal study over 9 models: damage is **diffuse**, and recovering 75% of the gap takes half the layers. At equal budget, **spend the extra bit globally** (finer groups everywhere) rather than on "important" layers |
| 3 | [[2508.04405\|FlexQ]] | Post-training INT6 weights with selective 8-bit activations and a **W6A6/W6A8 GPU kernel**. Accuracy close to W8A8 at 25% less memory |
| 4 | [[2505.04081\|QStore]] (VLDB) | Lossless **joint storage of a BF16 + INT8 pair** (store INT8 plus the conditional residual): −55% storage, 1.6× faster load |
| 5 | [[2505.20839\|FireQ]] | INT4 weights × FP8 activations GEMM, RoPE-aware KV/query quantization, 3-stage prefill pipeline on FlashAttention-3 |
| 6 | [[2608.21952\|SSDi8]] (ICLR'26 W) | Persistent **INT8 path for Mamba-2 SSD** by decoupling element-wise ops from matmuls |
| 7 | [[2605.26339\|QAM-W]] | 2-D (paired-coordinate) Lloyd-Max codebook after block-Hadamard: **±0.4% of BF16 PPL at ~5.5 bpw** |
| 8 | [[2608.06763\|CubicQuant]] | Parametric cubic non-uniform levels on a dense integer code stream (1–8 bits); uniform INT is a special case, so kernels stay simple |
| 9 | [[2502.12346\|QuZO]] (EMNLP'25) | Zeroth-order fine-tuning with forward passes only, on INT8/INT4 models (no straight-through estimator) |
| 10 | [[2603.22324\|DAQ]] | Data-free FP8 that preserves the **post-training delta** (ΔW direction), recovering style and behaviour lost under standard FP8 |

**Runtime notes.**
* FP8 (E4M3) with per-channel weight scales and per-token dynamic activation scales is the default serving format on
  Hopper and newer.
* **NestedFP-style dual precision** is a cheap way to add a "degraded mode" to the scheduler.
* For MoE or large-vocabulary models, check that the LM head and router stay at ≥8 bits.
