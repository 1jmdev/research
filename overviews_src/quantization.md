# Quantization: state of the art, 2025–2026

> Hand-written synthesis of the top-ranked papers in [`papers/quantization`](../papers/quantization/README.md) and
> [`papers/kv-cache/quantization`](../papers/kv-cache/quantization/README.md). Paper links go to the per-paper files,
> which hold abstracts, tables and extracted compute. Numbers are the authors' claims.

## TL;DR for a runtime builder

1. **The safe default is weight-only 4-bit (W4A16) with a GPTQ-class solver.** The largest study of quantized
   *reasoning* models ([[2504.04823]]) finds W8A8 and W4A16 lossless. Lower bit-widths, and especially aggressive
   activation or KV quantization, hurt long chain-of-thought tasks more than perplexity suggests.
2. **On Blackwell, FP4 is the new INT4.** Support both **NVFP4** (16-element blocks with an FP8-E4M3 scale) and
   **MXFP4** (32-element blocks with a power-of-two E8M0 scale). NVFP4 is more accurate out of the box, and MXFP4
   needs extra tricks:
   * [[2509.23202|MR-GPTQ]] shows that the classic rotation/outlier tricks break with NVFP4's small groups, and that
     MXFP4's power-of-two scales cause large errors. Block-wise Hadamard transforms plus format-aware GPTQ fix this,
     with a reported 3.6× layer-wise and 2.2× end-to-end speedup over FP16 on B200.
   * [[2603.08713|OAS + MBS]] (software-only scaling fixes) shrink the MXFP4-vs-NVFP4 accuracy gap from about 10% to
     under 1%. [[2511.04214|Block rotation]], [[2604.17789|DuQuant++]], [[2603.16590|BATQuant]] and the
     [[2601.19213|M2XFP]] format attack the same gap.
   * [[2512.02010|Four Over Six]] adds adaptive block scaling to NVFP4, which helps both inference and NVFP4
     pre-training.
   * [[2510.25602|INT vs FP]]: at 8 bits, **MXINT8 beats MXFP8** in both accuracy and hardware cost. At 4 bits FP
     usually wins, but **NVINT4 can beat NVFP4** once outlier mitigation such as a Hadamard transform is applied.
     Worth reading before choosing kernel formats.
3. **For W4A4 / W4A8, use rotations.** Rotation methods (QuaRot/SpinQuant lineage) are the standard.
   * [[2501.13987|OSTQuant]]: learned orthogonal + scaling transforms; keeps 99.5% of FP accuracy at W4-only and
     closes 32% of the remaining gap at W4A4KV4 on Llama-3-8B.
   * [[2512.00956|WUSH]]: provably near-optimal non-orthogonal transforms, with a fused kernel reaching up to 5.8×
     per-layer throughput via FP4 matmul.
   * [[2511.04063|DartQuant]]: makes rotation calibration 47× faster and 10× less memory-hungry on 70B models.
4. **Calibration-free is now competitive.** [[2509.22944|SINQ]] (Sinkhorn-normalised dual scales) halves the
   perplexity gap of plain RTN with no calibration data and near-zero overhead. Useful for "quantize on load" in a
   runtime.
5. **Serving trick: quantize only prefill.** [[2605.20315|Mix-Quant]] runs NVFP4 for prefill and BF16 for decode.
   Agentic and long-context workloads are prefill-heavy, so this gives up to 3× prefill speedup while keeping decode
   exact.

## Improvements to the GPTQ solver

GPTQ is still the backbone. 2025–26 work mostly improves its objective:

| Method | Idea | Headline |
| --- | --- | --- |
| [[2507.18553]] | GPTQ (back-to-front) **is** Babai's nearest-plane algorithm on the Hessian lattice | Gives error bounds and principled ordering |
| [[2504.02692\|GPTAQ]] | Asymmetric calibration: match the *full-precision* layer input | About 20 extra lines over GPTQ; quantizes a 405B model on one GPU |
| [[2504.09629\|QEP]] | Propagate and compensate error accumulated across layers | General add-on to layer-wise PTQ |
| [[2505.07004\|GuidedQuant]] | End-loss gradient guidance plus a non-uniform scalar quantizer | ICML 2025 |
| [[2507.11017\|FOEM]] | Keep first-order terms in the error compensation | 3-bit Llama-3-8B: −17.3% PPL |
| [[2607.07964\|KronQ]] | Kronecker-factored Hessian with gradient covariance | 2-bit Llama-3-70B at 7.93 PPL, where GPTQ diverges (>2000) |

## Bit-width by bit-width

### ~4-bit (INT4 / FP4): production grade
See [`4-bit-integer`](../papers/quantization/4-bit-integer/README.md) and
[`4-bit-floating-point`](../papers/quantization/4-bit-floating-point/README.md).
* [[2507.04610|any4]] learns a 4-bit numeric codebook per matrix and beats int4, fp4 and nf4 with no preprocessing.
  It maps well to LUT-based GEMV.
* [[2503.22879|Quamba2]] covers W8A8, W4A8 and W4A16 for **Mamba 1/2** (SSM-specific PTQ).
* [[2502.00425|MQuant]] does full static W4A8 for multimodal LLMs: under 1% loss and up to 30% lower latency.
* [[2509.11177|OBR]] combines quantization and 50% sparsity (W4A4KV4 + 2:4-style sparsity) for up to 4.72× speedup.

### ~3-bit
Mostly GPTQ-solver improvements plus non-uniform / LUT quantizers:
* [[2501.12956|GANQ]]: GPU-adaptive non-uniform quantization for LUT mpGEMM, up to 2.57× on an RTX 4090.
* [[2505.02390]]: a dynamic 3-bit `DQ3_K_M` recipe for **DeepSeek-V3/R1 671B**. 4-bit is near-lossless versus FP8
  and enables single-node deployment.

### ~2-bit: the frontier of practical PTQ
See [`2-bit`](../papers/quantization/2-bit/README.md).
* [[2602.05367|RaBiT]] (residual binarization training) matches vector-quantization methods at 2 bits and runs 4.49×
  faster than FP16 on an RTX 4090.
* [[2602.04163|BPDQ]] uses bit-plane decomposition on a variable grid. At 2 bits, Qwen2.5-72B runs on a **single
  RTX 3090** with 83.9% GSM8K (versus 90.8% at 16-bit).
* [[2512.04746|SignRoundV2]] (Intel AutoRound) adds gradient-guided layer-wise bit allocation plus scale search.
* Vector and trellis quantizers (AQLM/QuIP#/QTIP lineage) remain the accuracy leaders but need dedicated
  decode kernels. See [[2605.24144|EVA]] (ISCA'26) for VQ-decode hardware, and [[2509.20214|Q-Palette]] for trellis,
  vector and scalar quantizers with optimized kernels.

### 1–1.58-bit (binary / ternary)
See [`1-bit-and-ternary`](../papers/quantization/1-bit-and-ternary/README.md).
* **Native ternary training:** [[2504.18415|BitNet v2]] adds 4-bit activations to 1.58-bit weights via online
  Hadamard (H-BitLinear). [[2502.05003|QuEST]] shows stable QAT scaling laws all the way down to 1-bit weights and
  activations, with GPU kernels.
* **Post-training ternarization now rivals BitNet:**
  * [[2606.26650|CAT-Q]] ternarizes pre-trained 1.7–235B models with 512 calibration samples. It reportedly beats
    BitNet b1.58 v1/v2 models trained on 100B tokens, a roughly 100,000× reduction in training tokens.
  * [[2606.13054|TWLA]] reaches W1.58A4 by PTQ.
  * [[2509.16989|PTQTP]] (dual trit-planes, multiplication-free) needs about **one hour** of quantization versus
    **10–14 GPU-days** for training-based ternary methods.
* **Packing matters for kernels:** [[2601.07892|Sherry]] packs 3:4-sparse ternary weights at 1.25 bits so blocks of 4
  weights fit in 5 bits, keeping power-of-two alignment. [[2512.00862|HBLLM]] reaches 1.08 bits via wavelets.

### Below 1 bit
See [`sub-1-bit`](../papers/quantization/sub-1-bit/README.md). [[2506.13771|LittleBit]] uses latent factorization down
to **0.1 bpw** (Llama-2-13B under 0.9 GB). [[2602.06694|NanoQuant]] is the first PTQ to binary and sub-1-bit
(ICML 2026). [[2506.12040|BTC-LLM]] uses binary codebooks. Quality is still well below 2-bit methods; treat as research.

### Mixed precision and bit allocation
See [`mixed-precision`](../papers/quantization/mixed-precision/README.md).
* [[2509.20214|Q-Palette]]: fractional-bit quantizers plus optimal allocation (NeurIPS 2025).
* [[2508.02343|MicroMix]]: per-layer MXFP4/6/8 mix with a co-designed GEMM kernel, near-FP16 at about 5 bits on
  average.
* [[2509.12019|AMQ]] (AutoML search) and [[2602.17698|ScaleBITS]] (hardware-aligned search).
* [[2609.01587]] shows quantization damage is *diffuse*. Spend extra bits globally rather than chasing task
  circuits.

## Low-precision training (FP8 → FP4)

See [`low-precision-training`](../papers/quantization/low-precision-training/README.md).
* [[2509.25149|NVFP4 pre-training]] (NVIDIA) trains a **12B model on 10T tokens in NVFP4**, the longest public 4-bit
  run. The recipe: random Hadamard transforms, 2-D block scaling, stochastic rounding on gradients, and a few
  BF16 layers.
* [[2505.14669|Quartet]] (native FP4 training can be compute-optimal, with a low-precision scaling law) and
  [[2601.22813|Quartet II]] (unbiased MS-EDEN gradient estimator, Blackwell kernels up to 4.2× over BF16).
* [[2505.19115|FP4 All the Way]] (fully quantized training up to 200B tokens), [[2510.27527|TetraJet-v2]] (NVFP4 with
  oscillation suppression), [[2502.20586|MXFP4 training]], and [[2509.00404|Metis]] (spectral splitting; W4A4G4 on
  Llama-3-8B/100B tokens with 0.4% loss gap).
* Pitfall: [[2510.04212]] explains *why* low-precision FlashAttention training blows up (a biased rounding error
  accumulates). Read it before writing low-precision attention backward kernels.
* RL: [[2601.14243|Jet-RL]] shows BF16-train + FP8-rollout is unstable over long rollouts; unified FP8 precision flow
  is needed.

## What does it cost? (H100-hours)

Rough orders of magnitude from the extracted compute sentences (see the compute tables in each sub-category README):

| Approach | Typical cost | Examples |
| --- | --- | --- |
| RTN / calibration-free (SINQ, HQQ-style) | seconds–minutes per model | [[2509.22944]] |
| GPTQ-class PTQ, 7–70B | minutes to a few GPU-hours; one GPU is enough even for 405B ([[2504.02692\|GPTAQ]]) | [[2504.02692]], [[2507.18553]] |
| Rotation learning (SpinQuant/OSTQuant style) | a few GPU-hours at 70B; DartQuant cuts this 47× | [[2511.04063]] |
| Ternary PTQ | ≈1 GPU-hour ([[2509.16989\|PTQTP]]), 512 samples ([[2606.26650\|CAT-Q]]) | |
| QAT / ternary training from a checkpoint | 10–14 GPU-days per model (quoted in [[2509.16989\|PTQTP]]) | [[2502.05003]] |
| FP4 pre-training from scratch | same as BF16 pre-training minus the speedup; 12B/10T is 10⁵–10⁶ H100-h | [[2509.25149]] |

## Open problems worth working on

* 2-bit PTQ that keeps **long-CoT reasoning** accuracy. Studies ([[2504.04823]], [[2608.01078]]) show reasoning
  degrades first.
* Unified kernels for LUT / codebook / trellis decode at batch > 1 (most VQ kernels are GEMV-only).
* MXFP4 accuracy parity without per-model rotation calibration.
* Quantization of **MoE experts** and **SSM/linear-attention states**. See [[2503.22879|Quamba2]] and the MoE
  compression folder.
