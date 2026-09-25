**Verdict.** 3-bit is the "free lunch" edge. With a good solver, weight-only 3-bit on ≥7B models loses about 1–2 points
on common benchmarks.

* **Accuracy.** Statistically-lossless quantization (SLQ) is **task-lossless at 3.3–4.7 bits** but only
  **distribution-lossless at 5.0–6.6 bits**. So 3-bit models answer benchmark questions correctly but do not
  sample the same text. That matters for speculative decoding with a quantized draft, and for RL rollouts.
* **Hidden cost.** Quantized *reasoning* models think longer. [[2606.25519|Token Inflation]] shows INT3/INT4 can
  preserve accuracy but raise chain-of-thought length enough to cancel the per-token speedup.
* **Main research lines.** Non-uniform/LUT quantizers (GANQ, PoT) and better GPTQ compensation (FOEM).

| # | Paper | Kind | Key idea | Headline | Cost |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2605.02404\|SLQ: Statistically-Lossless Quantization]] | PTQ + theory | Defines task-lossless vs distribution-lossless; **Expected Acceptance Rate (EAR)** metric; γ² variance law ⇒ asymmetric quantization is required for distribution losslessness; ILP bit allocation | TL at 3.3–4.7 b, DL at 5.0–6.6 b, **1.7–3.7× speedup** vs BF16; works on MoE | ~2.5 h per bit-width on A6000 for 8B |
| 2 | [[2501.12956\|GANQ]] (ICML'25) | PTQ | Mixed-integer QP for **LUT-based non-uniform** weights, GPU-adaptive alternating solver | Beats uniform at 3/4 bits; up to 2.57× on an RTX 4090 (LUT mpGEMM) | 7B in ~1 h on one RTX 4090 (≈0.2 H100-h) |
| 3 | [[2507.11017\|FOEM]] (AAAI'26) | PTQ | GPTQ with the **first-order term** (latent weights drift during compensation) | Llama-3-8B 3-bit: −17.3% PPL vs GPTQ; composes with GPTAQ/QuaRot | GPTQ-level |
| 4 | [[2505.02390\|DeepSeek quantization study]] | Empirical | Multi-bit evaluation of the full **DeepSeek-V3/R1 671B**; dynamic DQ3_K_M recipe | Q4 ≈ FP8; DQ3_K_M fits one 8×GPU node | — |
| 5 | [[2608.11045\|ReRound]] | Calibration-free PTQ | Diffusion prior over the model's own weights decides midpoint-ambiguous rounding | Improves 3/4-bit RTN with **no data** | 2×RTX 4090 |
| 6 | [[2604.11080\|ReSpinQuant]] (ICML'26) | PTQ | Layer-wise rotation accuracy but **fusable offline** via residual subspace rotation | No online rotation overhead | — |
| 7 | [[2507.11959\|PoTPTQ]] | PTQ + kernel | Power-of-two levels with a bitwise dequant kernel | 3.67× on V100, 1.63× on RTX 4090 vs dequant baseline | — |
| 8 | [[2605.04084\|FASQ]] | Calibration-free PQ | Product quantization with a continuous size knob (27–49% of FP16) and LUT-free CUDA kernels | Beats 4-bit GPTQ/AWQ at 37–42% size on Llama-3-8B | — |
| 9 | [[2608.21134\|Llama-Mobile]] (Meta) | QAD | **2.7-bit S3D8 format** for Arm CPUs; self-generated distillation data | Llama-3.2-11B-Vision in 3.7 GB, W2.7A8 | — |
| 10 | [[2608.08910\|Tied Trit-Planes]] | Format | PTQTP planes folded into one 4.06-bit code; SSD-streamed MoE experts | DeepSeek-V4-Flash 284B-A13B served from disk | — |
| 11 | [[2603.27914\|ITQ3_S]] | Format | FWHT-rotated interleaved ternary 3-bit with inverse FWHT fused into the shared-memory load | llama.cpp-style 3-bit format | — |

**Recommendations.**
* **Default 3-bit path:** Hadamard rotation + GPTQ/FOEM + **asymmetric** group-wise quantization (g = 64–128).
* **If you use a quantized model as a speculative draft or RL rollout policy,** measure EAR or acceptance rate, not
  benchmark accuracy.
* **Budget for token inflation.** Measure end-to-end time-to-answer on reasoning models, not tokens per second.
