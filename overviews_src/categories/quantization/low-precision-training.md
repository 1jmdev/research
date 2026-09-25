**Verdict.** **FP8 training is production-standard. NVFP4 pretraining is now demonstrated at frontier scale.** NVIDIA
trained a 12B hybrid Mamba-Transformer on **10T tokens** in NVFP4 with loss matching FP8.

The recipe has converged. Linear layers use NVFP4:
* **2-D (16×16) weight scaling**, so the forward and backward quantizations agree;
* a **random Hadamard transform** on the wgrad inputs;
* **stochastic rounding** on gradients;
* a few BF16 layers, usually the last ones.

Current research improves the *unbiased gradient estimator*: MS-EDEN in Quartet II has more than 2× lower error than
stochastic rounding. It also removes the last high-precision pieces: master weights (ECO), optimizer states, attention.

A separate line matters for post-training: **FP8/FP4 in RL rollouts**. BF16 training with FP8 rollouts is off-policy
and collapses on long horizons. Unify the precision (Jet-RL) or correct the mismatch (FP8-RL, QaRL, AIS).

### Hand ranking

| # | Paper | Scope | Key idea | Headline | Compute |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2509.25149\|Pretraining LLMs with NVFP4]] (NVIDIA) | Pretrain | 2-D weight scaling, RHT on wgrad, SR on gradients, a few BF16 layers | **12B hybrid, 10T tokens**, FP8-level loss and downstream accuracy | Frontier-scale run on GB200 |
| 2 | [[2601.22813\|Quartet II / MS-EDEN]] | Pretrain | Unbiased microscaled rounding that moves randomness to the scale (EDEN), **>2× lower error than SR**; full NVFP4 linear layers; CUDA kernels | Beats prior NVFP4 recipes | 38B-token runs on B200 |
| 3 | [[2505.14669\|Quartet]] (NeurIPS'25) | Pretrain | Low-precision scaling law. **Forward = QuEST (min-MSE, Hadamard + clip), backward = RTN/SR**. All matmuls MXFP4 | "FP4 training can be optimal" in the large-data regime | ~6K H100-h of experiments |
| 4 | [[2510.27527\|TetraJet-v2]] (ICML'26) | Pretrain | Unbiased double-block NVFP4, **OsciReset** (weight oscillation), OutControl (outliers) | Best NVFP4 fully-quantized training (FQT) at the time; 212B tokens | — |
| 5 | [[2505.19115\|FP4 All the Way]] (Intel) | Pretrain | Sweeps block, scale format and rounding; NVFP4 (16, E4M3) optimal; **SR backward, RTN forward**; FQT fails when grad-norm < √3 × quantization noise (switch to higher precision late in training) | 7B on 1T tokens | 256 Gaudi2 × 30 days ≈ **79K H100-h** |
| 6 | [[2510.04212\|Why Low-Precision FlashAttention Fails]] (ICLR'26) | Analysis | Loss explosions come from low-rank attention representations **plus biased BF16 rounding** in FlashAttention; a minimal kernel fix | Explains a notorious instability | — |
| 7 | [[2601.14243\|Jet-RL]] | RL | Identical FP8 precision flow for training and rollout (on-policy) | BF16-train + FP8-rollout collapses; Jet-RL is stable and faster | — |
| 8 | [[2601.18150\|FP8-RL]] (veRL) | RL system | FP8 weight sync every step, FP8 KV cache, importance-sampling mismatch correction; vLLM/SGLang | Practical stack | — |
| 9 | [[2601.22101\|ECO]] | Optimizer | **No master weights**: inject the requantization error into momentum | Near-baseline quality; large savings for MoE | — |
| 10 | [[2502.20586\|Training LLMs with MXFP4]] (AISTATS) | Pretrain | Stochastic rounding + RHT to bound SR variance | Near-lossless to 6.7B; >50% of FLOPs in MXFP4 | 210B tokens |
| 11 | [[2506.08027\|MXFP8 recipes]] (NVIDIA) | Pretrain | MXFP8-E4M3 everywhere with a specific conversion (rounding of the scale) | Matches BF16 at 8B / **15T tokens**; in Transformer Engine | — |
| 12 | [[2505.20524\|FOG: fully FP8 GEMMs]] (NeurIPS'25) | Architecture | Architecture changes (no pre-norm gains, etc.) that prevent outliers, so **all GEMMs incl. attention** run in FP8 | +43% throughput vs BF16 | 450B-token runs |
| 13 | [[2509.00404\|Metis]] | Pretrain | Spectral split of anisotropic tensors; random-projection subspace | W4A4G4 on Llama-3-8B/100B tokens with a <0.4% gap | — |
| 14 | [[2501.17116\|FP4 training (MSRA)]] (ICML'25) | Pretrain | Differentiable gradient estimator + outlier clamp and compensation | First FP4 LLM training framework (simulated) | 100B tokens |
| 15 | [[2603.00040\|Attn-QAT]] | Attention | Stable **FP4 attention** QAT: low-precision recomputation in backward plus a high-precision auxiliary output | 1.1–1.5× over SageAttention3 | ~1–53 H100-h |

Also important:
* [[2501.02423]]: FP scaling law. Exponent bits matter a bit more than mantissa, and there is an optimal precision per
  compute.
* [[2501.02625|HALO]]: Hadamard-assisted INT8/FP8 fine-tuning with FSDP.
* [[2605.09825]], [[2607.04422|Full-Stack FP4]], [[2606.20381|UFP4]], [[2603.10444]] (mean bias), [[2609.02846|UE5M3]]:
  FP4 stability fixes.
* [[2609.22870]]: full-pipeline FP8 RL.
* [[2603.02731]]: FP4 MoE training on **Hopper**.
* [[2507.16099|TorchAO]]: the practical toolkit.

**For a training stack.**
1. **FP8.** Use FP8 (E4M3 forward, E5M2 or E4M3 gradients with per-block scaling) everywhere except the LM head and
   embeddings.
2. **NVFP4.** On Blackwell, follow the NVIDIA recipe: 2-D weight scales, RHT on wgrad, SR on gradients, the last ~15%
   of layers in BF16. Switch to higher precision for the final LR-decay phase.
3. **RL.** Keep rollout and trainer precision identical, or apply an importance-sampling correction. Otherwise rewards
   collapse on long chains of thought.
