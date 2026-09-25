# ~3-bit quantization

Weight quantization around 3 bits per weight.

**23 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Quantization](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/quantization.md](../../../overviews/quantization.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** 3-bit is the "free lunch" edge. With a good solver, weight-only 3-bit on ≥7B models loses about 1–2 points
on common benchmarks.

* **Accuracy.** Statistically-lossless quantization (SLQ) is **task-lossless at 3.3–4.7 bits** but only
  **distribution-lossless at 5.0–6.6 bits**. So 3-bit models answer benchmark questions correctly but do not
  sample the same text. That matters for speculative decoding with a quantized draft, and for RL rollouts.
* **Hidden cost.** Quantized *reasoning* models think longer. [Token Inflation](2606.25519-quantization-inflates-reasoning-token-inflation-as-a-hidden-cost-of-lo.md) shows INT3/INT4 can
  preserve accuracy but raise chain-of-thought length enough to cancel the per-token speedup.
* **Main research lines.** Non-uniform/LUT quantizers (GANQ, PoT) and better GPTQ compensation (FOEM).

| # | Paper | Kind | Key idea | Headline | Cost |
| ---: | --- | --- | --- | --- | --- |
| 1 | [SLQ: Statistically-Lossless Quantization](2605.02404-statistically-lossless-quantization-of-large-language-models.md) | PTQ + theory | Defines task-lossless vs distribution-lossless; **Expected Acceptance Rate (EAR)** metric; γ² variance law ⇒ asymmetric quantization is required for distribution losslessness; ILP bit allocation | TL at 3.3–4.7 b, DL at 5.0–6.6 b, **1.7–3.7× speedup** vs BF16; works on MoE | ~2.5 h per bit-width on A6000 for 8B |
| 2 | [GANQ](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md) (ICML'25) | PTQ | Mixed-integer QP for **LUT-based non-uniform** weights, GPU-adaptive alternating solver | Beats uniform at 3/4 bits; up to 2.57× on an RTX 4090 (LUT mpGEMM) | 7B in ~1 h on one RTX 4090 (≈0.2 H100-h) |
| 3 | [FOEM](2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md) (AAAI'26) | PTQ | GPTQ with the **first-order term** (latent weights drift during compensation) | Llama-3-8B 3-bit: −17.3% PPL vs GPTQ; composes with GPTAQ/QuaRot | GPTQ-level |
| 4 | [DeepSeek quantization study](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md) | Empirical | Multi-bit evaluation of the full **DeepSeek-V3/R1 671B**; dynamic DQ3_K_M recipe | Q4 ≈ FP8; DQ3_K_M fits one 8×GPU node | — |
| 5 | [ReRound](2608.11045-reround-reconstructive-rounding-to-resolve-midpoint-ambiguity-in-calib.md) | Calibration-free PTQ | Diffusion prior over the model's own weights decides midpoint-ambiguous rounding | Improves 3/4-bit RTN with **no data** | 2×RTX 4090 |
| 6 | [ReSpinQuant](2604.11080-respinquant-efficient-layer-wise-llm-quantization-via-subspace-residua.md) (ICML'26) | PTQ | Layer-wise rotation accuracy but **fusable offline** via residual subspace rotation | No online rotation overhead | — |
| 7 | [PoTPTQ](2507.11959-potptq-a-two-step-power-of-two-post-training-for-llms.md) | PTQ + kernel | Power-of-two levels with a bitwise dequant kernel | 3.67× on V100, 1.63× on RTX 4090 vs dequant baseline | — |
| 8 | [FASQ](2605.04084-fasq-flexible-accelerated-subspace-quantization-for-calibration-free-l.md) | Calibration-free PQ | Product quantization with a continuous size knob (27–49% of FP16) and LUT-free CUDA kernels | Beats 4-bit GPTQ/AWQ at 37–42% size on Llama-3-8B | — |
| 9 | [Llama-Mobile](2608.21134-llama-mobile-efficient-2-7-bit-quantization-of-vlms.md) (Meta) | QAD | **2.7-bit S3D8 format** for Arm CPUs; self-generated distillation data | Llama-3.2-11B-Vision in 3.7 GB, W2.7A8 | — |
| 10 | [Tied Trit-Planes](2608.08910-tied-trit-planes-constraining-ptqtp-to-a-uniform-nine-level-quantizer.md) | Format | PTQTP planes folded into one 4.06-bit code; SSD-streamed MoE experts | DeepSeek-V4-Flash 284B-A13B served from disk | — |
| 11 | [ITQ3_S](2603.27914-itq3-s-high-fidelity-3-bit-llm-inference-via-interleaved-ternary-quant.md) | Format | FWHT-rotated interleaved ternary 3-bit with inverse FWHT fused into the shared-memory load | llama.cpp-style 3-bit format | — |

**Recommendations.**
* **Default 3-bit path:** Hadamard rotation + GPTQ/FOEM + **asymmetric** group-wise quantization (g = 64–128).
* **If you use a quantized model as a speculative draft or RL rollout policy,** measure EAR or acceptance rate, not
  benchmark accuracy.
* **Budget for token inflation.** Measure end-to-end time-to-answer on reasoning models, not tokens per second.

## 🏆 Best of the best by impact score (top 10)

1. **[GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md)** (2025-06) — GANQ (GPU-Adaptive Non-Uniform Quantization), a layer-wise post-training non-uniform quantization framework optimized for hardware-efficient lookup table-based mpGEMM, which achieves superior quantization performance by …  
   _score 3.44 · International Conference on Machine Learning (ICML) · 8 cites · [code](https://github.com/Evans-Z/GANQ) · ~0.17–0.51 H100-h_
2. **[First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md)** (2025-11) — FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation, is proposed, and it is revealed that FOEM consistently outperforms the classical GPTQ method.  
   _score 3.11 · Accepted by AAAI 2026 · 8 cites · [code](https://github.com/Xingyu-Zheng/FOEM)_
3. **[Statistically-Lossless Quantization of Large Language Models](2605.02404-statistically-lossless-quantization-of-large-language-models.md)** (2026-08) — This paper explores the middle ground of statistically-lossless compression, examining three complementary aspects of what losslessness means for quantized LLMs, and proves a gamma-squared variance law showing that …  
   _score 3.03 · 3 cites · [code](https://github.com/IST-DASLab/SLQ)_
4. **[Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md)** (2025-06) — This technical report presents the first quantitative evaluation of multi-bitwidth quantization across the complete DeepSeek model spectrum and proposes DQ3_K_M, a dynamic 3-bit quantization method that significantly …  
   _score 2.69 · 5 cites · [code](https://github.com/UnicomAI/DeepSeek-Eval)_
5. **[DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models](2504.09223-dl-qat-weight-decomposed-low-rank-quantization-aware-training-for-larg.md)** (2025-04) — Weight-Decomposed Low-Rank Quantization-Aware Training (DL-QAT), which merges the advantages of QAT while training only less than 1% of the total parameters, and shows significant improvements over the baseline method …  
   _score 2.63 · Conference on Empirical Methods in Natural Language Processi · 12 cites_
6. **[Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](2603.17809-fine-grained-post-training-quantization-for-large-vision-language-mode.md)** (2026-03) — Inspired by axiomatic attribution in mechanistic interpretability, a fine-grained quantization strategy on Quantization-aware Integrated Gradients (QIG), which leverages integrated gradients to quantitatively evaluate …  
   _score 2.44 · Accepted by CVPR 2026 Main C · 3 cites · [code](https://github.com/ucas-xiang/QIG)_
7. **[Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs](2608.21134-llama-mobile-efficient-2-7-bit-quantization-of-vlms.md)** (2026-08) — This work presents a framework for quantizing VLMs for efficient inference on resource-constrained hardware, which combines a quantization pipeline that uses the model itself to generate training data and does not …  
   _score 2.42 · 0 cites · 8▲ HF_
8. **[ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization](2608.11045-reround-reconstructive-rounding-to-resolve-midpoint-ambiguity-in-calib.md)** (2026-08) — The ReRound strategy represents a new approach for low-bit quantization and consistently outperforms standard RTN for 3-bit and 4-bit weight quantization across a range of such models, and applies to AI models beyond …  
   _score 2.03 · 0 cites · 3▲ HF · [code](https://github.com/louisYen/ReRound)_
9. **[FBQuant: FeedBack Quantization for Large Language Models](2501.16385-fbquant-feedback-quantization-for-large-language-models.md)** (2025-05) — FeedBack Quantization (FBQuant), a novel approach inspired by negative feedback mechanisms in automatic control, which inherently ensures that the reconstructed weights remain bounded by the quantization process, …  
   _score 1.93 · Accepted to IJCAI 2025 · 6 cites_
10. **[VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models](2602.01037-veq-modality-adaptive-quantization-for-moe-vision-language-models.md)** (2026-02) — Visual Expert Quantization (VEQ), a dual-aware quantization framework designed to simultaneously accommodate cross-modal differences and heterogeneity between experts, is proposed, demonstrating superior robustness …  
   _score 1.6 · 3 cites · [code](https://github.com/guangshuoqin/VEQ)_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Beyond Activation Alignment:The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization](2607.00908-beyond-activation-alignment-the-alignment-diversity-tradeoff-in-task-a.md)** (2026-07-01; 0▲, 1 cites) — TASA (Task-Aware Sensitivity Analysis), a two-level framework that jointly optimizes calibration-data composition and mixed-precision bit allocation, is proposed and results show …
- **[Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models](2606.25519-quantization-inflates-reasoning-token-inflation-as-a-hidden-cost-of-lo.md)** (2026-06-29; 0▲, 0 cites) — Across mathematical reasoning, code generation, scientific question answering, and agentic tool-use benchmarks, it is found that INT4/INT3 quantization can preserve accuracy but …
- **[RDQ: Residual Distribution Quantization for Large Language Models](2607.10137-rdq-residual-distribution-quantization-for-large-language-models.md)** (2026-09-10; 0▲, 0 cites) — RDQ (Residual Distribution Quantization), a PTQ framework whose central contribution is Cascaded Error Compensation, a sequential calibration procedure that captures the actual …
- **[Tied Trit-Planes: Constraining PTQTP to a Uniform Nine-Level Quantizer, with a Persistent Folded Format for Disk-Streamed Mixture-of-Experts Serving](2608.08910-tied-trit-planes-constraining-ptqtp-to-a-uniform-nine-level-quantizer.md)** (2026-08-09; 0▲, 0 cites) — This work is the first to impose that identity as a constraint inside PTQTP's solver, a known balanced-ternary identity, and applies it to the routed experts of …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md) | 0.17 | 0.51 | RTX 4090 | For instance, GANQ processes the LLaMA-2-7B model on a single NVIDIA RTX 4090 GPU in approximately one hour, using only 128 samples, each containing 2,048 token… |

## Bit-width map

| Bits | # papers | Top papers |
| --- | ---: | --- |
| 1.58 (ternary) | 2 | [ITQ3_S: High-Fidelity 3-bit LLM Inference via Inte](2603.27914-itq3-s-high-fidelity-3-bit-llm-inference-via-interleaved-ternary-quant.md); [Tied Trit-Planes: Constraining PTQTP to a Uniform ](2608.08910-tied-trit-planes-constraining-ptqtp-to-a-uniform-nine-level-quantizer.md) |
| 2 | 1 | [Llama-Mobile: Efficient 2.7-Bit Quantization of VL](2608.21134-llama-mobile-efficient-2-7-bit-quantization-of-vlms.md) |
| 3 | 21 | [GANQ: GPU-Adaptive Non-Uniform Quantization for La](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md); [First-Order Error Matters: Accurate Compensation f](2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md); [Statistically-Lossless Quantization of Large Langu](2605.02404-statistically-lossless-quantization-of-large-language-models.md); [Quantitative Analysis of Performance Drop in DeepS](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md) |
| 4 | 13 | [GANQ: GPU-Adaptive Non-Uniform Quantization for La](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md); [First-Order Error Matters: Accurate Compensation f](2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md); [Statistically-Lossless Quantization of Large Langu](2605.02404-statistically-lossless-quantization-of-large-language-models.md); [Quantitative Analysis of Performance Drop in DeepS](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md) |
| 5–6 | 1 | [Beyond Activation Alignment:The Alignment-Diversit](2607.00908-beyond-activation-alignment-the-alignment-diversity-tradeoff-in-task-a.md) |
| 8 | 1 | [Quantitative Analysis of Performance Drop in DeepS](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md) |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models](2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md) | 2025-06-09 | 3.44 | 8 | 0 | International Conference on Machine Lear | [✓](https://github.com/Evans-Z/GANQ) | GANQ (GPU-Adaptive Non-Uniform Quantization), a layer-wise post-training non-uniform quantization framework optimized for hardware-efficient lookup table-based … |
| 2 | [First-Order Error Matters: Accurate Compensation for Quantized Large Language Models](2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md) | 2025-11-14 | 3.11 | 8 | 0 | Accepted by AAAI 2026 | [✓](https://github.com/Xingyu-Zheng/FOEM) | FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation, is proposed, and it is revealed … |
| 3 | [Statistically-Lossless Quantization of Large Language Models](2605.02404-statistically-lossless-quantization-of-large-language-models.md) | 2026-08-09 | 3.03 | 3 | 0 |  | [✓](https://github.com/IST-DASLab/SLQ) | This paper explores the middle ground of statistically-lossless compression, examining three complementary aspects of what losslessness means for quantized … |
| 4 | [Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md) | 2025-06-13 | 2.69 | 5 | 0 |  | [✓](https://github.com/UnicomAI/DeepSeek-Eval) | This technical report presents the first quantitative evaluation of multi-bitwidth quantization across the complete DeepSeek model spectrum and proposes … |
| 5 | [DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models](2504.09223-dl-qat-weight-decomposed-low-rank-quantization-aware-training-for-larg.md) | 2025-04-12 | 2.63 | 12 | 0 | Conference on Empirical Methods in Natur |  | Weight-Decomposed Low-Rank Quantization-Aware Training (DL-QAT), which merges the advantages of QAT while training only less than 1% of the total parameters, … |
| 6 | [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gr](2603.17809-fine-grained-post-training-quantization-for-large-vision-language-mode.md) | 2026-03-18 | 2.44 | 3 | 0 | Accepted by CVPR 2026 Main C | [✓](https://github.com/ucas-xiang/QIG) | Inspired by axiomatic attribution in mechanistic interpretability, a fine-grained quantization strategy on Quantization-aware Integrated Gradients (QIG), which … |
| 7 | [Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs](2608.21134-llama-mobile-efficient-2-7-bit-quantization-of-vlms.md) | 2026-08-21 | 2.42 | 0 | 8 |  |  | This work presents a framework for quantizing VLMs for efficient inference on resource-constrained hardware, which combines a quantization pipeline that uses … |
| 8 | [ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in Calibration-Free LLM Quantization](2608.11045-reround-reconstructive-rounding-to-resolve-midpoint-ambiguity-in-calib.md) | 2026-08-11 | 2.03 | 0 | 3 |  | [✓](https://github.com/louisYen/ReRound) | The ReRound strategy represents a new approach for low-bit quantization and consistently outperforms standard RTN for 3-bit and 4-bit weight quantization … |
| 9 | [FBQuant: FeedBack Quantization for Large Language Models](2501.16385-fbquant-feedback-quantization-for-large-language-models.md) | 2025-05-23 | 1.93 | 6 | 0 | Accepted to IJCAI 2025 |  | FeedBack Quantization (FBQuant), a novel approach inspired by negative feedback mechanisms in automatic control, which inherently ensures that the … |
| 10 | [VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models](2602.01037-veq-modality-adaptive-quantization-for-moe-vision-language-models.md) | 2026-02-01 | 1.6 | 3 | 0 |  | [✓](https://github.com/guangshuoqin/VEQ) | Visual Expert Quantization (VEQ), a dual-aware quantization framework designed to simultaneously accommodate cross-modal differences and heterogeneity between … |
| 11 | [PoTPTQ: A Two-step Power-of-Two Post-training for LLMs](2507.11959-potptq-a-two-step-power-of-two-post-training-for-llms.md) | 2025-07-16 | 1.47 | 7 | 0 |  |  | This work proposes a novel POT quantization framework for LLM weights that outperforms state-of-the-art accuracy in extremely low-precision number formats, and … |
| 12 | [ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](2604.11080-respinquant-efficient-layer-wise-llm-quantization-via-subspace-residua.md) | 2026-05-28 | 1.37 | 1 | 0 | ICML 2026 |  | ReSpinQuant is proposed, a quantization framework that reconciles the high expressivity of layer-wise adaptation with only negligible inference overhead, and … |
| 13 | [ROSAQ: Rotation-based Saliency-Aware Weight Quantization for Efficiently Compressing Large Language Models](2506.13472-rosaq-rotation-based-saliency-aware-weight-quantization-for-efficientl.md) | 2025-06-17 | 1.01 | 2 | 0 |  | [✓](https://github.com/AutoGPTQ/AutoGPTQ) | The rotation-based saliency-aware weight quantization (ROSAQ), which identifies salient channels in the projection feature space, not in the original feature … |
| 14 | [Beyond Activation Alignment:The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization](2607.00908-beyond-activation-alignment-the-alignment-diversity-tradeoff-in-task-a.md) | 2026-07-01 | 0.83 | 1 | 0 |  |  | TASA (Task-Aware Sensitivity Analysis), a two-level framework that jointly optimizes calibration-data composition and mixed-precision bit allocation, is … |
| 15 | [HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning](2601.21626-hero-q-a-general-framework-for-stable-low-bit-quantization-via-hessian.md) | 2026-06-17 | 0.75 | 1 | 0 |  |  | The Hessian Robust Quantization (HeRo Q) algorithm is proposed, which applies a lightweight, learnable rotation-compression matrix to the weight space prior to … |
| 16 | [FASQ: Flexible Accelerated Subspace Quantization for Calibration-Free LLM Compression](2605.04084-fasq-flexible-accelerated-subspace-quantization-for-calibration-free-l.md) | 2026-04-22 | 0.57 | 1 | 0 |  |  | FASQ (Flexible Accelerated Subspace Quantization), a calibration-free framework that applies product quantization to LLM weight matrices, is presented, … |
| 17 | [ITQ3_S: High-Fidelity 3-bit LLM Inference via Interleaved Ternary Quantization with Rotation-Domain Smoothing](2603.27914-itq3-s-high-fidelity-3-bit-llm-inference-via-interleaved-ternary-quant.md) | 2026-03-31 | 0.52 | 1 | 0 |  |  | We present ITQ3_S (Interleaved Ternary Quantization -- Specialized), a novel 3-bit weight quantization format for LLMs integrating TurboQuant (TQ), a … |
| 18 | [Adaptive Layer-Wise Transformations for Post-Training Quantization of Large Language Models](2511.17809-adaptive-layer-wise-transformations-for-post-training-quantization-of.md) | 2025-11-21 | 0.36 | 1 | 0 |  |  | This paper proposes an outlier-guided layer selection method using robust $z$-score normalization that achieves comparable performance to differentiable search … |
| 19 | [CafeQ: Calibration-free Quantization via Learned Transformations and Adaptive Rounding](2511.19705-cafeq-calibration-free-quantization-via-learned-transformations-and-ad.md) | 2025-11-24 | 0.36 | 1 | 0 |  |  | This paper proposes algorithms to optimize transformations and adaptive rounding without access to any calibration data, and achieves performance comparable to … |
| 20 | [QuantX: A Framework for Hardware-Aware Quantization of Generative AI Workloads](2505.07531-quantx-a-framework-for-hardware-aware-quantization-of-generative-ai-wo.md) | 2025-09-12 | 0.0 | 0 | 0 |  |  | This manuscript provides insights into the LLM quantization process that motivated the range of recipes and options that are incorporated in QuantX, a tailored … |
| 21 | [Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models](2606.25519-quantization-inflates-reasoning-token-inflation-as-a-hidden-cost-of-lo.md) | 2026-06-29 | 0.0 | 0 | 0 |  |  | Across mathematical reasoning, code generation, scientific question answering, and agentic tool-use benchmarks, it is found that INT4/INT3 quantization can … |
| 22 | [RDQ: Residual Distribution Quantization for Large Language Models](2607.10137-rdq-residual-distribution-quantization-for-large-language-models.md) | 2026-09-10 | 0.0 | 0 | 0 |  |  | RDQ (Residual Distribution Quantization), a PTQ framework whose central contribution is Cascaded Error Compensation, a sequential calibration procedure that … |
| 23 | [Tied Trit-Planes: Constraining PTQTP to a Uniform Nine-Level Quantizer, with a Persistent Folded Format for Di](2608.08910-tied-trit-planes-constraining-ptqtp-to-a-uniform-nine-level-quantizer.md) | 2026-08-09 | 0.0 | 0 | 0 |  |  | This work is the first to impose that identity as a constraint inside PTQTP's solver, a known balanced-ternary identity, and applies it to the routed experts … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](../../mixture-of-experts/compression/2602.11184-kbvq-moe-klt-guided-svd-with-bias-corrected-vector-quantization-for-mo.md) | MoE compression (expert pruning, merging, quantization) | 2.62 |
| [RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization](../_general/2505.03803-rwkvquant-quantizing-the-rwkv-family-with-proxy-guided-hybrid-of-scala.md) | Quantization — general / analysis / surveys | 1.57 |
| [ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](../_general/2606.00494-projq-project-and-quantize-for-adapter-aware-llm-compression.md) | Quantization — general / analysis / surveys | 1.2 |
| [LCD: Advancing Extreme Low-Bit Clustering for Large Language Models via Knowledge Distillation](../2-bit/2506.12038-lcd-advancing-extreme-low-bit-clustering-for-large-language-models-via.md) | ~2-bit quantization (2 – 2.x bits) | 0.26 |
