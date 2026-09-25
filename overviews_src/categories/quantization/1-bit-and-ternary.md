**Verdict.** Two ways to get a ternary {-1, 0, +1} model now both work:

* **Native training.** BitNet b1.58 2B4T is the proof point: 2B parameters, 4T tokens, on par with FP models of the same
  size.
* **Post-training ternarization of existing checkpoints.** CAT-Q and ScaleQ-1.58 cover 1.7B–235B, including MoE and
  reasoning models, in hours to days on one 8×A100 node.

Plain 1-bit (binary) PTQ is still lossy. Anything below ~1.5 bits needs structure: salient channels, wavelets, codebooks.
The main runtime problem is no longer accuracy but **packing and kernels**. 1.58-bit does not align to bytes; see Sherry
(1.25-bit, 3:4 sparse) and the "Breaking the 1.58-bit barrier" line.

### Hand ranking

| # | Paper | Kind | Why it matters | Cost (hand-checked, H100-h at A100 = 0.32×) |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.12285\|BitNet b1.58 2B4T]] | Native pretrain | Reference open ternary LLM (weights and GPU/CPU kernels). Matches FP 2B models at a fraction of memory and energy | 4T-token pretrain (not disclosed in GPU-h) |
| 2 | [[2606.26650\|CAT-Q]] (ICML'26 oral) | PTQ | Learnable modulation plus softened (differentiable) ternarization with a sliding-layer pipeline. **First to ternarize 14B–235B models** | 1–60 h on 8×A100 → **~2.6–154 H100-h** |
| 3 | [[2608.01078\|ScaleQ-1.58 / AYOT]] | PTQ | CAT-Q plus calibrating on the model's **own reasoning traces**. Without this, ternary reasoning models collapse on math and code | 4–240 h on 8×A100 → **~10–614 H100-h**; 4M calibration tokens |
| 4 | [[2510.13998\|BitNet Distillation]] | FT → 1.58b | Qwen → ternary per downstream task (SubLN + MiniLM attention distillation + short continual pretraining). 10× memory, 2.65× CPU speed | Short CPT (~10B tokens) |
| 5 | [[2504.18415\|BitNet v2]] | Native pretrain | **W1.58A4** via online Hadamard before activation quantization (H-BitLinear). Enables INT4 batched matmuls | 100B-token runs |
| 6 | [[2502.05003\|QuEST]] (ICML'25) | QAT theory | Hadamard + Gaussian-fitted clipping + "trust" gradient estimator. **4-bit W&A is Pareto-optimal** for QAT, and training is stable down to W1A1 | Up to 160B-token runs |
| 7 | [[2502.11880\|Bitnet.cpp]] (ACL'25) | Kernels | Ternary LUT (TL) and I2_S mpGEMM for CPU. Up to 6.25× over FP16 on edge CPUs. **Copy these kernels** | — |
| 8 | [[2509.16989\|PTQTP]] | PTQ | Two ternary **trit-planes** + scales; multiplication-free. Keeps math and code ability where other sub-2-bit PTQ collapses. 4.63× end-to-end | About 1 GPU-hour per model |
| 9 | [[2601.07892\|Sherry]] (ACL'26) | QAT + format | **1.25-bit**: 3:4 sparse ternary, so 4 weights pack into 5 bits and align with SIMD. Fixes the 2-bit-packing waste | Edge models ≤3B |
| 10 | [[2606.13054\|TWLA]] (ICML'26) | PTQ | W1.58**A4** without retraining. Beats 2-bit PTQ methods | 2×A6000 |
| 11 | [[2509.23809\|Tequila]] | QAT | Diagnoses **deadzone trapping** (weights stuck at the 0/±1 boundary) and repurposes them as dynamic biases | — |
| 12 | [[2512.00862\|HBLLM]] (NeurIPS'25) | PTQ binary | Haar-wavelet 1-bit. Llama-2-13B PPL 6.71 at 1.08 bits | 1×A800 |
| 13 | [[2502.13179\|PTQ1.61]] (ACL'25) | PTQ | 1-D structured salient-channel mask at 0.0002 bit per weight; salient channels in 4-bit | ~2 h preprocessing per model |
| 14 | [[2510.10467\|AnyBCQ]] (ICLR'26) | PTQ + kernel | Multi-precision **bit-plane** BCQ: one model serves 2/3/4-bit by activating more planes | — |
| 15 | [[2605.00422\|BWLA]] (ACL'26) | PTQ | **W1A6** by Orthogonal-Kronecker transform and proximal SVD | 2×A6000 |
| 16 | [[2512.02901\|Fairy2i]] | QAT | Converts real layers to widely-linear **complex** form with weights in {±1, ±i}; reuses real checkpoints | — |

Worth a look in the tail:
* [[2604.20913|FairyFuse]]: fused ternary CPU kernels.
* [[2506.23025|Spectra 1.1]]: ternary scaling laws plus TriRun kernels.
* [[2609.16338]]: entropy-coded ternary below 1.58 bits.
* [[2606.18114|Ternary Mamba]]: ternary SSMs.
* [[2604.06798|MoBiE]]: binary MoE experts.
* [[2505.08823]]: an extra RMSNorm before each quantized linear is enough to fine-tune to 1.58 bits.

**Runtime checklist.**
1. **Weight formats.** Ship I2_S (2-bit-packed ternary with a scale) as the baseline and a TL (LUT) path for CPUs.
   Consider a 1.25-bit 3:4 format for bandwidth-bound decode.
2. **Activations.** Ternary weights with INT8 activations are the norm. W1.58A4 needs an online Hadamard (BitNet v2,
   TWLA), so fuse the FWHT into the previous op.
3. **Validation.** Evaluate ternary PTQ'd reasoning models on long chain-of-thought (AIME, LiveCodeBench), not on
   perplexity. ScaleQ shows perplexity and MMLU hide the collapse.
