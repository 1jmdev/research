**Verdict.** Activation (contextual) sparsity means skipping FFN neurons, channels or attention heads whose activations
are ~0 for this token. It is the most **decode-friendly** sparsity at **batch size 1**, where you save the *weight reads*
of the skipped channels. The catch has always been batching: different tokens activate different neurons, so savings
vanish as the batch grows. [[2505.14884|Polar Sparsity]] quantifies this: FFN sparsity decays with batch size, while
**attention-head sparsity stays batch-invariant**. The state of play:

* **Training-free, magnitude-based** (TEAL lineage, 2024) is the baseline. 2025 improvements:
  * [[2504.19449|R-Sparse]] (ICLR'25): rank-aware, uses input channels + singular components, 50% model-level sparsity,
    +43% end to end;
  * [[2505.19427|WINA]]: weight-norm-informed, +2.94% over TEAL;
  * [[2507.01299|LaRoSA]] (ICML'25): layerwise *rotations* before top-k; 40% sparsity, +0.17 PPL, 1.30× wall-clock.

  Expect **40–50% sparsity, 1.2–1.5× decode** on SwiGLU models.
* **Architectures that train sparsity in** get far more:
  * [[2506.06644|Spark Transformer]] (Google, NeurIPS'25): 8% of FFN neurons active + ≤256 attended tokens,
    Gemma-2 recipe, 2.5× fewer FLOPs, 1.79× CPU / 1.40× GPU decode;
  * [[2507.08771|BlockFFN]]: ReLU+RMSNorm-routed chunk-level sparsity that stays sparse across **consecutive tokens**,
    so it composes with speculative decoding; 3.67× on edge devices;
  * [[2603.23198|Sparser, Faster, Lighter]]: L1 regularization → >99% FFN sparsity plus kernels;
  * [[2503.16672|2:4 activation sparsity]] (Squared-ReLU): 1.3× FFN in training and inference.
* **Sparsity grows with model size** ([[2509.00454|universal properties]]), and dLLMs show it too.

### Hand ranking

| # | Paper | Training? | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2506.06644\|Spark Transformer]] (NeurIPS'25) | Pretrain | Statistical top-k (no sort) in FFN **and** attention, with a dedicated predictor sub-dimension | 8% active FFN neurons; 2.5× fewer FLOPs; 1.79× CPU / 1.40× GPU decode at Gemma-2 quality |
| 2 | [[2504.19449\|R-Sparse]] (ICLR'25) | Free | Sparsify **input channels** and low-rank components (no output-neuron predictor) | 50% model-level sparsity at comparable accuracy; 43% end-to-end gain with custom kernels |
| 3 | [[2505.14884\|Polar Sparsity]] | Free + kernels | FFN sparsity dies with batch size, head sparsity doesn't → **selective head attention** kernels | Up to 2.2× end to end in *batched* serving |
| 4 | [[2507.08771\|BlockFFN]] | Pretrain (MoE-style) | ReLU+RMSNorm differentiable router; chunk-level sparsity (CLS) aware training | >80% token-level and 70% 8-token chunk sparsity; 3.67× on end-side devices; works with speculative decoding |
| 5 | [[2507.01299\|LaRoSA]] (ICML'25) | Free | Layerwise orthogonal rotations make activations top-k-friendly; consistent model-level sparsity | Llama-2-7B at 40%: +0.17 PPL, 1.30× wall-clock; beats TEAL and CATS |
| 6 | [[2505.19427\|WINA]] (Microsoft) | Free | Select neurons by \|x\|·‖W_col‖₂ (weight-informed), with optimality bounds | New training-free frontier (+2.94% vs TEAL) |
| 7 | [[2503.16672\|2:4 activation sparsity]] (Meta) | Squared-ReLU models | Map naturally sparse Squared-ReLU activations onto 2:4 sparse tensor cores | Up to 1.3× faster FFN forward **and backward** with no accuracy loss |
| 8 | [[2603.23198\|Sparser, Faster, Lighter]] | Pretrain | L1 on activations → >99% unstructured FFN sparsity + custom kernels | Throughput, energy and memory gains that grow with scale |
| 9 | [[2602.00397\|FastForward]] | Light training | Predictive FFN sparsity for **prefill** + error-compensation net + layer scheduler | 1.45× compute-bound prefill at 50% FFN sparsity (<6% loss on LongBench) |
| 10 | [[2604.03258\|SoLA]] | Free | Keep the few high-contribution FFN components; low-rank the rest | Llama-2-70B at 30%: PPL 6.95 → 4.44 vs prior state of the art |
| 11 | [[2509.00454\|Universal properties of activation sparsity]] | Analysis | Cross-family study of FFN activation sparsity, incl. dLLMs | Achievable sparsity grows with model size |
| 12 | [[2505.17701\|COUNTDOWN]] | Free | Sparsify the down-projection using the linear-combination coefficients | Up to 90% of computation skipped at ~5.5% loss (ideal predictor) |

**Also useful.**
* Dual weight + activation sparsity: [[2506.20194|DuoGPT]], [[2608.01536|Celty]] (SpMSpV kernel + SIMT co-design).
* N:M activation sparsity: [[2508.02128|Amber Pruner]] (prefill), [[2509.22166]] (16:32 patterns for next-gen
  hardware).
* Robustness and allocation: [[2512.12744|SPON]], [[2603.12272|ActTail]], [[2602.14452|WiSparse]],
  [[2607.27591|Prox]].
* Predictors: [[2507.14179]], [[2603.14110]] (SVD).
* On-device: [[2511.04477]] (dynamic sparsity with quantized weights), [[2607.18081|SelectInfer]].
* Channel-MoE: [[2511.09323|Mixture-of-Channels]].

**Recommendation.**
* *Runtime (edge / batch size 1):* implement a fused "sparse GEMV" that reads only active weight columns, combined with
  quantized weights (group-aligned). Use R-Sparse/WINA/LaRoSA thresholds calibrated offline. For batched serving,
  exploit **head sparsity** (Polar) rather than FFN sparsity.
* *Model builders:* if you control pretraining, add ReLU²/top-k FFN sparsity (Spark, BlockFFN) from the start. It is
  close to free in quality and gives 2–4× on-device decode. Make it chunk-consistent so speculative decoding still
  helps.
