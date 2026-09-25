# Quantization: state of the art, 2025–2026

> Hand-written synthesis of the top-ranked papers in [`papers/quantization`](../papers/quantization/README.md) and
> [`papers/kv-cache/quantization`](../papers/kv-cache/quantization/README.md). Paper links go to the per-paper files,
> which hold abstracts, tables and extracted compute. Numbers are the authors' claims.

## TL;DR for a runtime builder

1. **The safe default is weight-only 4-bit (W4A16) with a GPTQ-class solver.** The largest study of quantized
   *reasoning* models ([Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models](../papers/quantization/4-bit-integer/2504.04823-quantization-hurts-reasoning-an-empirical-study-on-quantized-reasoning.md)) finds W8A8 and W4A16 lossless. Lower bit-widths, and especially aggressive
   activation or KV quantization, hurt long chain-of-thought tasks more than perplexity suggests.
2. **On Blackwell, FP4 is the new INT4.** Support both **NVFP4** (16-element blocks with an FP8-E4M3 scale) and
   **MXFP4** (32-element blocks with a power-of-two E8M0 scale). NVFP4 is more accurate out of the box, and MXFP4
   needs extra tricks:
   * [MR-GPTQ](../papers/quantization/4-bit-floating-point/2509.23202-bridging-the-gap-between-promise-and-performance-for-microscaling-fp4.md) shows that the classic rotation/outlier tricks break with NVFP4's small groups, and that
     MXFP4's power-of-two scales cause large errors. Block-wise Hadamard transforms plus format-aware GPTQ fix this,
     with a reported 3.6× layer-wise and 2.2× end-to-end speedup over FP16 on B200.
   * [OAS + MBS](../papers/quantization/4-bit-floating-point/2603.08713-unveiling-the-potential-of-quantization-with-mxfp4-strategies-for-quan.md) (software-only scaling fixes) shrink the MXFP4-vs-NVFP4 accuracy gap from about 10% to
     under 1%. [Block rotation](../papers/quantization/4-bit-floating-point/2511.04214-block-rotation-is-all-you-need-for-mxfp4-quantization.md), [DuQuant++](../papers/quantization/4-bit-floating-point/2604.17789-duquant-fine-grained-rotation-enhances-microscaling-fp4-quantization.md), [BATQuant](../papers/quantization/4-bit-floating-point/2603.16590-batquant-outlier-resilient-mxfp4-quantization-via-learnable-block-wise.md) and the
     [M2XFP](../papers/quantization/4-bit-floating-point/2601.19213-m2xfp-a-metadata-augmented-microscaling-data-format-for-efficient-low.md) format attack the same gap.
   * [Four Over Six](../papers/quantization/4-bit-floating-point/2512.02010-four-over-six-more-accurate-nvfp4-quantization-with-adaptive-block-sca.md) adds adaptive block scaling to NVFP4, which helps both inference and NVFP4
     pre-training.
   * [INT vs FP](../papers/quantization/4-bit-floating-point/2510.25602-int-v-s-fp-a-comprehensive-study-of-fine-grained-low-bit-quantization.md): at 8 bits, **MXINT8 beats MXFP8** in both accuracy and hardware cost. At 4 bits FP
     usually wins, but **NVINT4 can beat NVFP4** once outlier mitigation such as a Hadamard transform is applied.
     Worth reading before choosing kernel formats.
3. **For W4A4 / W4A8, use rotations.** Rotation methods (QuaRot/SpinQuant lineage) are the standard.
   * [OSTQuant](../papers/quantization/4-bit-integer/2501.13987-ostquant-refining-large-language-model-quantization-with-orthogonal-an.md): learned orthogonal + scaling transforms; keeps 99.5% of FP accuracy at W4-only and
     closes 32% of the remaining gap at W4A4KV4 on Llama-3-8B.
   * [WUSH](../papers/quantization/4-bit-integer/2512.00956-wush-near-optimal-adaptive-transforms-for-llm-quantization.md): provably near-optimal non-orthogonal transforms, with a fused kernel reaching up to 5.8×
     per-layer throughput via FP4 matmul.
   * [DartQuant](../papers/quantization/_general/2511.04063-dartquant-efficient-rotational-distribution-calibration-for-llm-quanti.md): makes rotation calibration 47× faster and 10× less memory-hungry on 70B models.
4. **Calibration-free is now competitive.** [SINQ](../papers/quantization/_general/2509.22944-sinq-sinkhorn-normalized-quantization-for-calibration-free-low-precisi.md) (Sinkhorn-normalised dual scales) halves the
   perplexity gap of plain RTN with no calibration data and near-zero overhead. Useful for "quantize on load" in a
   runtime.
5. **Serving trick: quantize only prefill.** [Mix-Quant](../papers/quantization/4-bit-floating-point/2605.20315-mix-quant-quantized-prefilling-precise-decoding-for-agentic-llms.md) runs NVFP4 for prefill and BF16 for decode.
   Agentic and long-context workloads are prefill-heavy, so this gives up to 3× prefill speedup while keeping decode
   exact.

## Improvements to the GPTQ solver

GPTQ is still the backbone. 2025–26 work mostly improves its objective:

| Method | Idea | Headline |
| --- | --- | --- |
| [The Geometry of LLM Quantization](../papers/quantization/_general/2507.18553-the-geometry-of-llm-quantization-gptq-as-babai-s-nearest-plane-algorit.md) | GPTQ (back-to-front) **is** Babai's nearest-plane algorithm on the Hessian lattice | Gives error bounds and principled ordering |
| [GPTAQ](../papers/quantization/_general/2504.02692-gptaq-efficient-finetuning-free-quantization-for-asymmetric-calibratio.md) | Asymmetric calibration: match the *full-precision* layer input | About 20 extra lines over GPTQ; quantizes a 405B model on one GPU |
| [QEP](../papers/quantization/_general/2504.09629-quantization-error-propagation-revisiting-layer-wise-post-training-qua.md) | Propagate and compensate error accumulated across layers | General add-on to layer-wise PTQ |
| [GuidedQuant](../papers/quantization/_general/2505.07004-guidedquant-large-language-model-quantization-via-exploiting-end-loss.md) | End-loss gradient guidance plus a non-uniform scalar quantizer | ICML 2025 |
| [FOEM](../papers/quantization/3-bit/2507.11017-first-order-error-matters-accurate-compensation-for-quantized-large-la.md) | Keep first-order terms in the error compensation | 3-bit Llama-3-8B: −17.3% PPL |
| [KronQ](../papers/quantization/2-bit/2607.07964-kronq-llm-quantization-via-kronecker-factored-hessian.md) | Kronecker-factored Hessian with gradient covariance | 2-bit Llama-3-70B at 7.93 PPL, where GPTQ diverges (>2000) |

## Bit-width by bit-width

### ~4-bit (INT4 / FP4): production grade
See [`4-bit-integer`](../papers/quantization/4-bit-integer/README.md) and
[`4-bit-floating-point`](../papers/quantization/4-bit-floating-point/README.md).
* [any4](../papers/quantization/4-bit-integer/2507.04610-any4-learned-4-bit-numeric-representation-for-llms.md) learns a 4-bit numeric codebook per matrix and beats int4, fp4 and nf4 with no preprocessing.
  It maps well to LUT-based GEMV.
* [Quamba2](../papers/quantization/4-bit-integer/2503.22879-quamba2-a-robust-and-scalable-post-training-quantization-framework-for.md) covers W8A8, W4A8 and W4A16 for **Mamba 1/2** (SSM-specific PTQ).
* [MQuant](../papers/quantization/4-bit-integer/2502.00425-mquant-unleashing-the-inference-potential-of-multimodal-large-language.md) does full static W4A8 for multimodal LLMs: under 1% loss and up to 30% lower latency.
* [OBR](../papers/quantization/4-bit-integer/2509.11177-optimal-brain-restoration-for-joint-quantization-and-sparsification-of.md) combines quantization and 50% sparsity (W4A4KV4 + 2:4-style sparsity) for up to 4.72× speedup.

### ~3-bit
Mostly GPTQ-solver improvements plus non-uniform / LUT quantizers:
* [GANQ](../papers/quantization/3-bit/2501.12956-ganq-gpu-adaptive-non-uniform-quantization-for-large-language-models.md): GPU-adaptive non-uniform quantization for LUT mpGEMM, up to 2.57× on an RTX 4090.
* [Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](../papers/quantization/3-bit/2505.02390-quantitative-analysis-of-performance-drop-in-deepseek-model-quantizati.md): a dynamic 3-bit `DQ3_K_M` recipe for **DeepSeek-V3/R1 671B**. 4-bit is near-lossless versus FP8
  and enables single-node deployment.

### ~2-bit: the frontier of practical PTQ
See [`2-bit`](../papers/quantization/2-bit/README.md).
* [RaBiT](../papers/quantization/2-bit/2602.05367-rabit-residual-aware-binarization-training-for-accurate-and-efficient.md) (residual binarization training) matches vector-quantization methods at 2 bits and runs 4.49×
  faster than FP16 on an RTX 4090.
* [BPDQ](../papers/quantization/2-bit/2602.04163-bpdq-bit-plane-decomposition-quantization-on-a-variable-grid-for-large.md) uses bit-plane decomposition on a variable grid. At 2 bits, Qwen2.5-72B runs on a **single
  RTX 3090** with 83.9% GSM8K (versus 90.8% at 16-bit).
* [SignRoundV2](../papers/quantization/2-bit/2512.04746-signroundv2-toward-closing-the-performance-gap-in-extremely-low-bit-po.md) (Intel AutoRound) adds gradient-guided layer-wise bit allocation plus scale search.
* Vector and trellis quantizers (AQLM/QuIP#/QTIP lineage) remain the accuracy leaders but need dedicated
  decode kernels. See [EVA](../papers/quantization/2-bit/2605.24144-eva-accelerating-llm-decoding-via-an-efficient-vector-quantization-arc.md) (ISCA'26) for VQ-decode hardware, and [Q-Palette](../papers/quantization/mixed-precision/2509.20214-q-palette-fractional-bit-quantizers-toward-optimal-bit-allocation-for.md) for trellis,
  vector and scalar quantizers with optimized kernels.

### 1–1.58-bit (binary / ternary)
See [`1-bit-and-ternary`](../papers/quantization/1-bit-and-ternary/README.md).
* **Native ternary training:** [BitNet v2](../papers/quantization/1-bit-and-ternary/2504.18415-bitnet-v2-native-4-bit-activations-with-hadamard-transformation-for-1.md) adds 4-bit activations to 1.58-bit weights via online
  Hadamard (H-BitLinear). [QuEST](../papers/quantization/1-bit-and-ternary/2502.05003-quest-stable-training-of-llms-with-1-bit-weights-and-activations.md) shows stable QAT scaling laws all the way down to 1-bit weights and
  activations, with GPU kernels.
* **Post-training ternarization now rivals BitNet:**
  * [CAT-Q](../papers/quantization/1-bit-and-ternary/2606.26650-cat-q-cost-efficient-and-accurate-ternary-quantization-for-llms.md) ternarizes pre-trained 1.7–235B models with 512 calibration samples. It reportedly beats
    BitNet b1.58 v1/v2 models trained on 100B tokens, a roughly 100,000× reduction in training tokens.
  * [TWLA](../papers/quantization/1-bit-and-ternary/2606.13054-twla-achieving-ternary-weights-and-low-bit-activations-for-llms-via-po.md) reaches W1.58A4 by PTQ.
  * [PTQTP](../papers/quantization/1-bit-and-ternary/2509.16989-ptqtp-post-training-quantization-to-trit-planes-for-large-language-mod.md) (dual trit-planes, multiplication-free) needs about **one hour** of quantization versus
    **10–14 GPU-days** for training-based ternary methods.
* **Packing matters for kernels:** [Sherry](../papers/quantization/1-bit-and-ternary/2601.07892-sherry-hardware-efficient-1-25-bit-ternary-quantization-via-fine-grain.md) packs 3:4-sparse ternary weights at 1.25 bits so blocks of 4
  weights fit in 5 bits, keeping power-of-two alignment. [HBLLM](../papers/quantization/1-bit-and-ternary/2512.00862-hbllm-wavelet-enhanced-high-fidelity-1-bit-quantization-for-llms.md) reaches 1.08 bits via wavelets.

### Below 1 bit
See [`sub-1-bit`](../papers/quantization/sub-1-bit/README.md). [LittleBit](../papers/quantization/sub-1-bit/2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md) uses latent factorization down
to **0.1 bpw** (Llama-2-13B under 0.9 GB). [NanoQuant](../papers/quantization/sub-1-bit/2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md) is the first PTQ to binary and sub-1-bit
(ICML 2026). [BTC-LLM](../papers/quantization/sub-1-bit/2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md) uses binary codebooks. Quality is still well below 2-bit methods; treat as research.

### Mixed precision and bit allocation
See [`mixed-precision`](../papers/quantization/mixed-precision/README.md).
* [Q-Palette](../papers/quantization/mixed-precision/2509.20214-q-palette-fractional-bit-quantizers-toward-optimal-bit-allocation-for.md): fractional-bit quantizers plus optimal allocation (NeurIPS 2025).
* [MicroMix](../papers/quantization/mixed-precision/2508.02343-micromix-efficient-mixed-precision-quantization-with-microscaling-form.md): per-layer MXFP4/6/8 mix with a co-designed GEMM kernel, near-FP16 at about 5 bits on
  average.
* [AMQ](../papers/quantization/mixed-precision/2509.12019-amq-enabling-automl-for-mixed-precision-weight-only-quantization-of-la.md) (AutoML search) and [ScaleBITS](../papers/quantization/mixed-precision/2602.17698-scalebits-scalable-bitwidth-search-for-hardware-aligned-mixed-precisio.md) (hardware-aligned search).
* [The Structure of Quantization Damage in LLMs](../papers/quantization/5-to-8-bit/2609.01587-the-structure-of-quantization-damage-in-llms-why-the-next-bit-should-b.md) shows quantization damage is *diffuse*. Spend extra bits globally rather than chasing task
  circuits.

## Low-precision training (FP8 → FP4)

See [`low-precision-training`](../papers/quantization/low-precision-training/README.md).
* [NVFP4 pre-training](../papers/quantization/low-precision-training/2509.25149-pretraining-large-language-models-with-nvfp4.md) (NVIDIA) trains a **12B model on 10T tokens in NVFP4**, the longest public 4-bit
  run. The recipe: random Hadamard transforms, 2-D block scaling, stochastic rounding on gradients, and a few
  BF16 layers.
* [Quartet](../papers/quantization/low-precision-training/2505.14669-quartet-native-fp4-training-can-be-optimal-for-large-language-models.md) (native FP4 training can be compute-optimal, with a low-precision scaling law) and
  [Quartet II](../papers/quantization/low-precision-training/2601.22813-quartet-ii-accurate-llm-pre-training-in-nvfp4-by-improved-unbiased-gra.md) (unbiased MS-EDEN gradient estimator, Blackwell kernels up to 4.2× over BF16).
* [FP4 All the Way](../papers/quantization/low-precision-training/2505.19115-fp4-all-the-way-fully-quantized-training-of-llms.md) (fully quantized training up to 200B tokens), [TetraJet-v2](../papers/quantization/low-precision-training/2510.27527-tetrajet-v2-accurate-nvfp4-training-for-large-language-models-with-osc.md) (NVFP4 with
  oscillation suppression), [MXFP4 training](../papers/quantization/low-precision-training/2502.20586-training-llms-with-mxfp4.md), and [Metis](../papers/quantization/low-precision-training/2509.00404-metis-training-llms-with-fp4-quantization.md) (spectral splitting; W4A4G4 on
  Llama-3-8B/100B tokens with 0.4% loss gap).
* Pitfall: [Why Low-Precision Transformer Training Fails](../papers/quantization/low-precision-training/2510.04212-why-low-precision-transformer-training-fails-an-analysis-on-flash-atte.md) explains *why* low-precision FlashAttention training blows up (a biased rounding error
  accumulates). Read it before writing low-precision attention backward kernels.
* RL: [Jet-RL](../papers/quantization/low-precision-training/2601.14243-jet-rl-enabling-on-policy-fp8-reinforcement-learning-with-unified-trai.md) shows BF16-train + FP8-rollout is unstable over long rollouts; unified FP8 precision flow
  is needed.

## What does it cost? (H100-hours)

Rough orders of magnitude from the extracted compute sentences (see the compute tables in each sub-category README):

| Approach | Typical cost | Examples |
| --- | --- | --- |
| RTN / calibration-free (SINQ, HQQ-style) | seconds–minutes per model | [SINQ](../papers/quantization/_general/2509.22944-sinq-sinkhorn-normalized-quantization-for-calibration-free-low-precisi.md) |
| GPTQ-class PTQ, 7–70B | minutes to a few GPU-hours; one GPU is enough even for 405B ([GPTAQ](../papers/quantization/_general/2504.02692-gptaq-efficient-finetuning-free-quantization-for-asymmetric-calibratio.md)) | [GPTAQ](../papers/quantization/_general/2504.02692-gptaq-efficient-finetuning-free-quantization-for-asymmetric-calibratio.md), [The Geometry of LLM Quantization](../papers/quantization/_general/2507.18553-the-geometry-of-llm-quantization-gptq-as-babai-s-nearest-plane-algorit.md) |
| Rotation learning (SpinQuant/OSTQuant style) | a few GPU-hours at 70B; DartQuant cuts this 47× | [DartQuant](../papers/quantization/_general/2511.04063-dartquant-efficient-rotational-distribution-calibration-for-llm-quanti.md) |
| Ternary PTQ | ≈1 GPU-hour ([PTQTP](../papers/quantization/1-bit-and-ternary/2509.16989-ptqtp-post-training-quantization-to-trit-planes-for-large-language-mod.md)), 512 samples ([CAT-Q](../papers/quantization/1-bit-and-ternary/2606.26650-cat-q-cost-efficient-and-accurate-ternary-quantization-for-llms.md)) | |
| QAT / ternary training from a checkpoint | 10–14 GPU-days per model (quoted in [PTQTP](../papers/quantization/1-bit-and-ternary/2509.16989-ptqtp-post-training-quantization-to-trit-planes-for-large-language-mod.md)) | [QuEST](../papers/quantization/1-bit-and-ternary/2502.05003-quest-stable-training-of-llms-with-1-bit-weights-and-activations.md) |
| FP4 pre-training from scratch | same as BF16 pre-training minus the speedup; 12B/10T is 10⁵–10⁶ H100-h | [Pretraining Large Language Models with NVFP4](../papers/quantization/low-precision-training/2509.25149-pretraining-large-language-models-with-nvfp4.md) |

## Open problems worth working on

* 2-bit PTQ that keeps **long-CoT reasoning** accuracy. Studies ([Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models](../papers/quantization/4-bit-integer/2504.04823-quantization-hurts-reasoning-an-empirical-study-on-quantized-reasoning.md), [Attend to Your Own Thoughts](../papers/quantization/1-bit-and-ternary/2608.01078-attend-to-your-own-thoughts-breaking-the-barrier-for-post-training-qua.md)) show reasoning
  degrades first.
* Unified kernels for LUT / codebook / trellis decode at batch > 1 (most VQ kernels are GEMV-only).
* MXFP4 accuracy parity without per-model rotation calibration.
* Quantization of **MoE experts** and **SSM/linear-attention states**. See [Quamba2](../papers/quantization/4-bit-integer/2503.22879-quamba2-a-robust-and-scalable-post-training-quantization-framework-for.md) and the MoE
  compression folder.
