# Activation / contextual sparsity

Exploiting sparse activations at inference (ReLU-fication, TEAL, contextual sparsity, neuron-level skipping).

**33 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Compression: pruning, sparsity, low-rank, distillation](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/compression.md](../../../overviews/compression.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Activation (contextual) sparsity means skipping FFN neurons, channels or attention heads whose activations
are ~0 for this token. It is the most **decode-friendly** sparsity at **batch size 1**, where you save the *weight reads*
of the skipped channels. The catch has always been batching: different tokens activate different neurons, so savings
vanish as the batch grows. [Polar Sparsity](2505.14884-polar-sparsity-high-throughput-batched-llm-inferencing-with-scalable-c.md) quantifies this: FFN sparsity decays with batch size, while
**attention-head sparsity stays batch-invariant**. The state of play:

* **Training-free, magnitude-based** (TEAL lineage, 2024) is the baseline. 2025 improvements:
  * [R-Sparse](2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md) (ICLR'25): rank-aware, uses input channels + singular components, 50% model-level sparsity,
    +43% end to end;
  * [WINA](2505.19427-wina-weight-informed-neuron-activation-for-accelerating-large-language.md): weight-norm-informed, +2.94% over TEAL;
  * [LaRoSA](2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md) (ICML'25): layerwise *rotations* before top-k; 40% sparsity, +0.17 PPL, 1.30× wall-clock.

  Expect **40–50% sparsity, 1.2–1.5× decode** on SwiGLU models.
* **Architectures that train sparsity in** get far more:
  * [Spark Transformer](2506.06644-spark-transformer-reactivating-sparsity-in-ffn-and-attention.md) (Google, NeurIPS'25): 8% of FFN neurons active + ≤256 attended tokens,
    Gemma-2 recipe, 2.5× fewer FLOPs, 1.79× CPU / 1.40× GPU decode;
  * [BlockFFN](2507.08771-blockffn-towards-end-side-acceleration-friendly-mixture-of-experts-wit.md): ReLU+RMSNorm-routed chunk-level sparsity that stays sparse across **consecutive tokens**,
    so it composes with speculative decoding; 3.67× on edge devices;
  * [Sparser, Faster, Lighter](2603.23198-sparser-faster-lighter-transformer-language-models.md): L1 regularization → >99% FFN sparsity plus kernels;
  * [2:4 activation sparsity](2503.16672-accelerating-transformer-inference-and-training-with-2-4-activation-sp.md) (Squared-ReLU): 1.3× FFN in training and inference.
* **Sparsity grows with model size** ([universal properties](2509.00454-universal-properties-of-activation-sparsity-in-modern-large-language-m.md)), and dLLMs show it too.

### Hand ranking

| # | Paper | Training? | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [Spark Transformer](2506.06644-spark-transformer-reactivating-sparsity-in-ffn-and-attention.md) (NeurIPS'25) | Pretrain | Statistical top-k (no sort) in FFN **and** attention, with a dedicated predictor sub-dimension | 8% active FFN neurons; 2.5× fewer FLOPs; 1.79× CPU / 1.40× GPU decode at Gemma-2 quality |
| 2 | [R-Sparse](2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md) (ICLR'25) | Free | Sparsify **input channels** and low-rank components (no output-neuron predictor) | 50% model-level sparsity at comparable accuracy; 43% end-to-end gain with custom kernels |
| 3 | [Polar Sparsity](2505.14884-polar-sparsity-high-throughput-batched-llm-inferencing-with-scalable-c.md) | Free + kernels | FFN sparsity dies with batch size, head sparsity doesn't → **selective head attention** kernels | Up to 2.2× end to end in *batched* serving |
| 4 | [BlockFFN](2507.08771-blockffn-towards-end-side-acceleration-friendly-mixture-of-experts-wit.md) | Pretrain (MoE-style) | ReLU+RMSNorm differentiable router; chunk-level sparsity (CLS) aware training | >80% token-level and 70% 8-token chunk sparsity; 3.67× on end-side devices; works with speculative decoding |
| 5 | [LaRoSA](2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md) (ICML'25) | Free | Layerwise orthogonal rotations make activations top-k-friendly; consistent model-level sparsity | Llama-2-7B at 40%: +0.17 PPL, 1.30× wall-clock; beats TEAL and CATS |
| 6 | [WINA](2505.19427-wina-weight-informed-neuron-activation-for-accelerating-large-language.md) (Microsoft) | Free | Select neurons by \|x\|·‖W_col‖₂ (weight-informed), with optimality bounds | New training-free frontier (+2.94% vs TEAL) |
| 7 | [2:4 activation sparsity](2503.16672-accelerating-transformer-inference-and-training-with-2-4-activation-sp.md) (Meta) | Squared-ReLU models | Map naturally sparse Squared-ReLU activations onto 2:4 sparse tensor cores | Up to 1.3× faster FFN forward **and backward** with no accuracy loss |
| 8 | [Sparser, Faster, Lighter](2603.23198-sparser-faster-lighter-transformer-language-models.md) | Pretrain | L1 on activations → >99% unstructured FFN sparsity + custom kernels | Throughput, energy and memory gains that grow with scale |
| 9 | [FastForward](2602.00397-fast-forward-accelerating-llm-prefill-with-predictive-ffn-sparsity.md) | Light training | Predictive FFN sparsity for **prefill** + error-compensation net + layer scheduler | 1.45× compute-bound prefill at 50% FFN sparsity (<6% loss on LongBench) |
| 10 | [SoLA](2604.03258-sola-leveraging-soft-activation-sparsity-and-low-rank-decomposition-fo.md) | Free | Keep the few high-contribution FFN components; low-rank the rest | Llama-2-70B at 30%: PPL 6.95 → 4.44 vs prior state of the art |
| 11 | [Universal properties of activation sparsity](2509.00454-universal-properties-of-activation-sparsity-in-modern-large-language-m.md) | Analysis | Cross-family study of FFN activation sparsity, incl. dLLMs | Achievable sparsity grows with model size |
| 12 | [COUNTDOWN](2505.17701-countdown-contextually-sparse-activation-filtering-out-unnecessary-wei.md) | Free | Sparsify the down-projection using the linear-combination coefficients | Up to 90% of computation skipped at ~5.5% loss (ideal predictor) |

**Also useful.**
* Dual weight + activation sparsity: [DuoGPT](2506.20194-duogpt-training-free-dual-sparsity-through-activation-aware-pruning-in.md), [Celty](2608.01536-celty-spmspv-gpu-kernel-and-simt-co-design-for-efficient-dual-sparse-l.md) (SpMSpV kernel + SIMT co-design).
* N:M activation sparsity: [Amber Pruner](2508.02128-amber-pruner-leveraging-n-m-activation-sparsity-for-efficient-prefill.md) (prefill), [Motivating Next-Gen Accelerators with Flexible (N](2509.22166-motivating-next-gen-accelerators-with-flexible-n-m-activation-sparsity.md) (16:32 patterns for next-gen
  hardware).
* Robustness and allocation: [SPON](2512.12744-resting-neurons-active-insights-robustifying-activation-sparsity-in-ll.md), [ActTail](2603.12272-acttail-global-activation-sparsity-in-large-language-models.md), [WiSparse](2602.14452-wisparse-boosting-llm-inference-efficiency-with-weight-aware-mixed-act.md),
  [Prox](2607.27591-prox-training-free-ffn-activation-sparsity-via-approximate-intermediat.md).
* Predictors: [A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering](2507.14179-a-sparsity-predicting-approach-for-large-language-models-via-activatio.md), [SVD Contextual Sparsity Predictors for Fast LLM Inference](2603.14110-svd-contextual-sparsity-predictors-for-fast-llm-inference.md) (SVD).
* On-device: [Enabling Dynamic Sparsity in Quantized LLM Inference](2511.04477-enabling-dynamic-sparsity-in-quantized-llm-inference.md) (dynamic sparsity with quantized weights), [SelectInfer](2607.18081-selectinfer-selective-neuron-loading-and-computation-for-on-device-llm.md).
* Channel-MoE: [Mixture-of-Channels](2511.09323-mixture-of-channels-exploiting-sparse-ffns-for-efficient-llms-pre-trai.md).

**Recommendation.**
* *Runtime (edge / batch size 1):* implement a fused "sparse GEMV" that reads only active weight columns, combined with
  quantized weights (group-aligned). Use R-Sparse/WINA/LaRoSA thresholds calibrated offline. For batched serving,
  exploit **head sparsity** (Polar) rather than FFN sparsity.
* *Model builders:* if you control pretraining, add ReLU²/top-k FFN sparsity (Spark, BlockFFN) from the start. It is
  close to free in quality and gives 2–4× on-device decode. Make it chunk-consistent so speculative decoding still
  helps.

## 🏆 Best of the best by impact score (top 10)

1. **[SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression](2604.03258-sola-leveraging-soft-activation-sparsity-and-low-rank-decomposition-fo.md)** (2026-03) — A novel training-free compression method for LLMs, named SoLA, which leverages Soft activation sparsity and Low-rAnk decomposition is proposed, which exhibits remarkable improvement in both language modeling and …  
   _score 6.21 · AAAI Conference on Artificial Intelligence (National Confere · 20 cites · [code](https://github.com/xinhaoH/SoLA)_
2. **[R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md)** (2025-04) — R-Sparse is introduced, a training-free activation sparsity approach capable of achieving high sparsity levels in advanced LLMs that replaces the linear layers in LLMs with a rank-aware sparse inference method that …  
   _score 5.28 · ICLR 2025 · 24 cites · [code](https://github.com/VITA-Group/R-Sparse) · ~0.16 H100-h_
3. **[BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity](2507.08771-blockffn-towards-end-side-acceleration-friendly-mixture-of-experts-wit.md)** (2025-07) — A novel MoE architecture, BlockFFN, is introduced, as well as its efficient training and deployment techniques, and efficient acceleration kernels are implemented, combining activation sparsity and speculative decoding …  
   _score 4.59 · 6 cites · 10▲ HF · [code](https://github.com/thunlp/BlockFFN)_
4. **[WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference](2505.19427-wina-weight-informed-neuron-activation-for-accelerating-large-language.md)** (2026-02) — WINA (Weight Informed Neuron Activation) is proposed, a novel, simple, and training-free sparse activation framework that jointly considers hidden state magnitudes and the column-wise $\ell_2$-norms of weight matrices …  
   _score 4.56 · 2 cites · 11▲ HF · [code](https://github.com/microsoft/wina)_
5. **[La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md)** (2026-01) — LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning, is introduced.  
   _score 3.81 · ICML 2025 · 5 cites · [code](https://github.com/alibaba/EfficientAI) · ~6.4 H100-h_
6. **[Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity](2505.14884-polar-sparsity-high-throughput-batched-llm-inferencing-with-scalable-c.md)** (2025-11) — This work introduces Polar Sparsity, the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar …  
   _score 3.63 · NeurIPS 2025 · 7 cites · [code](https://github.com/susavlsh10/Polar-Sparsity)_
7. **[Spark Transformer: Reactivating Sparsity in FFN and Attention](2506.06644-spark-transformer-reactivating-sparsity-in-ffn-and-attention.md)** (2025-10) — The Spark Transformer is introduced, a novel architecture that achieves a high level of activation sparsity in both FFN and the attention mechanism while maintaining model quality, parameter count, and standard training …  
   _score 3.08 · NeurIPS 2025 · 7 cites_
8. **[Accelerating Transformer Inference and Training with 2:4 Activation Sparsity](2503.16672-accelerating-transformer-inference-and-training-with-2-4-activation-sp.md)** (2025-03) — This work exploits the intrinsic sparsity found in Squared-ReLU activations to provide this acceleration with no accuracy loss, and achieves up to 1.3x faster Feed Forward Network (FFNs) in both the forwards and …  
   _score 2.58 · 12 cites_
9. **[DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](2506.20194-duogpt-training-free-dual-sparsity-through-activation-aware-pruning-in.md)** (2025-11) — DuoGPT is proposed, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity by extending the Optimal Brain Compression framework with …  
   _score 2.26 · NeurIPS 2025 · 6 cites_
10. **[Universal Properties of Activation Sparsity in Modern Large Language Models](2509.00454-universal-properties-of-activation-sparsity-in-modern-large-language-m.md)** (2026-02) — This work introduces a general framework for evaluating sparsity robustness in contemporary LLMs and conducts a systematic investigation of this phenomenon in their feedforward~(FFN) layers, uncovering universal …  
   _score 2.22 · ICLR 2026 · 2 cites_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Celty: SpMSpV GPU Kernel and SIMT Co-Design for Efficient Dual-Sparse LLM Inference](2608.01536-celty-spmspv-gpu-kernel-and-simt-co-design-for-efficient-dual-sparse-l.md)** (2026-09-05; 0▲, 0 cites) — Celty, a co-designed sparse format, GPU kernel, and SIMT microarchitecture for SpMSpV in LLM inference is proposed, which enables vectorized loading of compressed weight columns …
- **[Sparse Inter-Layer Dependencies of Transformer FFN Neurons](2607.11990-sparse-inter-layer-dependencies-of-transformer-ffn-neurons.md)** (2026-07-13; 0▲, 1 cites) — A training-free attribution method that estimates the relative influence of upstream neurons and attention outputs on a target neuron's activation and identifies candidate sparse …
- **[Sensitivity-Aware Thresholding and Token Routing for Activation Sparsification in Large Language Models](2607.08991-sensitivity-aware-thresholding-and-token-routing-for-activation-sparsi.md)** (2026-07-09; 0▲, 0 cites) — SATS improves over the threshold-based sparsification baseline at matched actual sparsity and token routing yields a more favorable quality-throughput trade-off than static …
- **[SelectInfer: Selective Neuron Loading and Computation for On-Device LLMs](2607.18081-selectinfer-selective-neuron-loading-and-computation-for-on-device-llm.md)** (2026-07-20; 0▲, 0 cites) — Evaluation across multiple datasets shows that SelectInfer achieves significant reductions in memory footprint and computation while preserving task performance, making it a …
- **[Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs](2607.27591-prox-training-free-ffn-activation-sparsity-via-approximate-intermediat.md)** (2026-07-30; 0▲, 0 cites) — Prox is a two-stage training-free framework for sparse SwiGLU FFNs that outperforms training-free baselines at all sparsity levels, achieves up to a $1.99\times end-to-end …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md) | 0.16 | 0.16 | A6000 | The overhead of the search process is minimal, taking approximately one hour on a single A6000 GPU for the Llama-2-7B model.… |
| [La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md) | 6.4 | 6.4 | A100 | The computation of {\mathbf{Q}} is performed on 8x80G A100 GPUs, taking approximately 12 minutes to complete for the LLaMA3 70B model.… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression](2604.03258-sola-leveraging-soft-activation-sparsity-and-low-rank-decomposition-fo.md) | 2026-03-12 | 6.21 | 20 | 0 | AAAI Conference on Artificial Intelligen | [✓](https://github.com/xinhaoH/SoLA) | A novel training-free compression method for LLMs, named SoLA, which leverages Soft activation sparsity and Low-rAnk decomposition is proposed, which exhibits … |
| 2 | [R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md) | 2025-04-28 | 5.28 | 24 | 0 | ICLR 2025 | [✓](https://github.com/VITA-Group/R-Sparse) | R-Sparse is introduced, a training-free activation sparsity approach capable of achieving high sparsity levels in advanced LLMs that replaces the linear layers … |
| 3 | [BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity](2507.08771-blockffn-towards-end-side-acceleration-friendly-mixture-of-experts-wit.md) | 2025-07-30 | 4.59 | 6 | 10 |  | [✓](https://github.com/thunlp/BlockFFN) | A novel MoE architecture, BlockFFN, is introduced, as well as its efficient training and deployment techniques, and efficient acceleration kernels are … |
| 4 | [WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference](2505.19427-wina-weight-informed-neuron-activation-for-accelerating-large-language.md) | 2026-02-18 | 4.56 | 2 | 11 |  | [✓](https://github.com/microsoft/wina) | WINA (Weight Informed Neuron Activation) is proposed, a novel, simple, and training-free sparse activation framework that jointly considers hidden state … |
| 5 | [La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation](2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md) | 2026-01-04 | 3.81 | 5 | 0 | ICML 2025 | [✓](https://github.com/alibaba/EfficientAI) | LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional … |
| 6 | [Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity](2505.14884-polar-sparsity-high-throughput-batched-llm-inferencing-with-scalable-c.md) | 2025-11-11 | 3.63 | 7 | 0 | NeurIPS 2025 | [✓](https://github.com/susavlsh10/Polar-Sparsity) | This work introduces Polar Sparsity, the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial … |
| 7 | [Spark Transformer: Reactivating Sparsity in FFN and Attention](2506.06644-spark-transformer-reactivating-sparsity-in-ffn-and-attention.md) | 2025-10-23 | 3.08 | 7 | 0 | NeurIPS 2025 |  | The Spark Transformer is introduced, a novel architecture that achieves a high level of activation sparsity in both FFN and the attention mechanism while … |
| 8 | [Accelerating Transformer Inference and Training with 2:4 Activation Sparsity](2503.16672-accelerating-transformer-inference-and-training-with-2-4-activation-sp.md) | 2025-03-20 | 2.58 | 12 | 0 |  |  | This work exploits the intrinsic sparsity found in Squared-ReLU activations to provide this acceleration with no accuracy loss, and achieves up to 1.3x faster … |
| 9 | [DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](2506.20194-duogpt-training-free-dual-sparsity-through-activation-aware-pruning-in.md) | 2025-11-13 | 2.26 | 6 | 0 | NeurIPS 2025 |  | DuoGPT is proposed, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity by … |
| 10 | [Universal Properties of Activation Sparsity in Modern Large Language Models](2509.00454-universal-properties-of-activation-sparsity-in-modern-large-language-m.md) | 2026-02-18 | 2.22 | 2 | 0 | ICLR 2026 |  | This work introduces a general framework for evaluating sparsity robustness in contemporary LLMs and conducts a systematic investigation of this phenomenon in … |
| 11 | [COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection](2505.17701-countdown-contextually-sparse-activation-filtering-out-unnecessary-wei.md) | 2025-10-27 | 1.73 | 1 | 0 | EMNLP 2025 |  | This work hypothesizes that the sparsity of the FFNN layer lies globally in the form of a linear combination over its internal down projection matrix, and … |
| 12 | [Sparse Inter-Layer Dependencies of Transformer FFN Neurons](2607.11990-sparse-inter-layer-dependencies-of-transformer-ffn-neurons.md) | 2026-07-13 | 1.6 | 1 | 0 |  |  | A training-free attribution method that estimates the relative influence of upstream neurons and attention outputs on a target neuron's activation and … |
| 13 | [Bug or Feature$^2$: Weight Drift, Activation Sparsity and Spikes](2605.17659-bug-or-feature-2-weight-drift-activation-sparsity-and-spikes.md) | 2026-05-20 | 1.27 | 0 | 1 |  | [✓](https://github.com/On-Point-RND/BugOrFeature) | It is proved that under MSE or cross-entropy loss, the gradient with respect to positive pre-activations is non-negative in expectation at initialization, … |
| 14 | [Celty: SpMSpV GPU Kernel and SIMT Co-Design for Efficient Dual-Sparse LLM Inference](2608.01536-celty-spmspv-gpu-kernel-and-simt-co-design-for-efficient-dual-sparse-l.md) | 2026-09-05 | 1.2 | 0 | 0 | ICCAD 2026 | [✓](https://github.com/RuokaiYin/Celty) | Celty, a co-designed sparse format, GPU kernel, and SIMT microarchitecture for SpMSpV in LLM inference is proposed, which enables vectorized loading of … |
| 15 | [Adaptive Rank Allocation: Speeding Up Modern Transformers with RaNA Adapters](2503.18216-adaptive-rank-allocation-speeding-up-modern-transformers-with-rana-ada.md) | 2025-05-06 | 1.18 | 2 | 0 | ICLR 2025 |  | RaNA is introduced as a robust solution for improving inference efficiency in modern Transformer architectures and improves perplexity by up to 7 points and … |
| 16 | [Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models](2508.02128-amber-pruner-leveraging-n-m-activation-sparsity-for-efficient-prefill.md) | 2025-08-04 | 0.77 | 3 | 0 |  |  | Amber Pruner is introduced, a training-free N:M activation sparsity method designed specifically for the prefill stage, targeting the acceleration of linear … |
| 17 | [$μ$-MoE: Test-Time Pruning as Micro-Grained Mixture-of-Experts](2505.18451-moe-test-time-pruning-as-micro-grained-mixture-of-experts.md) | 2025-05-24 | 0.7 | 3 | 0 |  |  | Several experiments demonstrate that $\mu$-MoE can dynamically adapt to task/prompt-dependent structured sparsity on the fly, yet achieving reduced complexity … |
| 18 | [Motivating Next-Gen Accelerators with Flexible (N:M) Activation Sparsity via Benchmarking Lightweight Post-Tra](2509.22166-motivating-next-gen-accelerators-with-flexible-n-m-activation-sparsity.md) | 2026-04-24 | 0.7 | 0 | 0 | Annual Meeting of the Association for Co |  | This work presents a comprehensive analysis of methods for post-training N:M activation pruning in LLMs, demonstrating that pruning activations enables … |
| 19 | [Resting Neurons, Active Insights: Robustifying Activation Sparsity in LLMs via Spontaneity](2512.12744-resting-neurons-active-insights-robustifying-activation-sparsity-in-ll.md) | 2026-05-21 | 0.7 | 0 | 0 | ICML 2026 |  | This work reframes activation sparsity as a representational alignment problem and introduces **Spontaneous Neurons (SPON)**, a lightweight mechanism inspired … |
| 20 | [Sparser, Faster, Lighter Transformer Language Models](2603.23198-sparser-faster-lighter-transformer-language-models.md) | 2026-05-08 | 0.5 | 0 | 0 |  | [✓](https://github.com/SakanaAI/sparser-faster-llms) | A new sparse packing format and set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs are introduced, … |
| 21 | [ActTail: Global Activation Sparsity in Large Language Models](2603.12272-acttail-global-activation-sparsity-in-large-language-models.md) | 2026-02-18 | 0.45 | 1 | 0 |  |  | ActTail is proposed, a TopK magnitude-based activation sparsity method with global activation sparsity allocation grounded in Heavy-Tailed Self-Regularization … |
| 22 | [Fast Forward: Accelerating LLM Prefill with Predictive FFN Sparsity](2602.00397-fast-forward-accelerating-llm-prefill-with-predictive-ffn-sparsity.md) | 2026-01-30 | 0.43 | 1 | 0 |  |  | FastForward is introduced, a predictive sparsity framework that accelerates LLM prefill through block-wise, context-aware FFN sparsity, substantially reducing … |
| 23 | [Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference](2511.09323-mixture-of-channels-exploiting-sparse-ffns-for-efficient-llms-pre-trai.md) | 2025-11-12 | 0.35 | 1 | 0 |  |  | Mixture-of-Channels (MoC), a novel FFN architecture that selectively activates only the Top-K most relevant channels per token determined by SwiGLU's native … |
| 24 | [A Sparsity Predicting Approach for Large Language Models via Activation Pattern Clustering](2507.14179-a-sparsity-predicting-approach-for-large-language-models-via-activatio.md) | 2025-07-11 | 0.0 | 0 | 0 |  |  | Our method achieves up to 79.34% clustering precision, outperforming standard binary clustering approaches while maintaining minimal degradation in perplexity … |
| 25 | [Enabling Dynamic Sparsity in Quantized LLM Inference](2511.04477-enabling-dynamic-sparsity-in-quantized-llm-inference.md) | 2025-11-06 | 0.0 | 0 | 0 |  |  | This study proposes a set of techniques that realize dynamic sparse inference under low-bit quantization and achieves up to 1.55x faster decoding throughput … |
| 26 | [WiSparse: Boosting LLM Inference Efficiency with Weight-Aware Mixed Activation Sparsity](2602.14452-wisparse-boosting-llm-inference-efficiency-with-weight-aware-mixed-act.md) | 2026-02-16 | 0.0 | 0 | 0 |  |  | Weight-aware Mixed-Granularity Training-free Activation Sparsity (WiSparse), which leverages both activation and weight information for adaptive sparsity … |
| 27 | [SVD Contextual Sparsity Predictors for Fast LLM Inference](2603.14110-svd-contextual-sparsity-predictors-for-fast-llm-inference.md) | 2026-03-14 | 0.0 | 0 | 0 |  |  | The proposed framework provides a fast, training-free method for building sparse pattern predictors using truncation-aware singular value decomposition of the … |
| 28 | [Dynamic sparsity in tree-structured feed-forward layers at scale](2604.08565-dynamic-sparsity-in-tree-structured-feed-forward-layers-at-scale.md) | 2026-03-18 | 0.0 | 0 | 0 |  |  | This work demonstrates that tree-structured feed-forward layers provide a scalable and controllable mechanism for sparsifying large transformer models and … |
| 29 | [Towards the Connection between Activation Sparsity and Flat Minima](2605.25612-towards-the-connection-between-activation-sparsity-and-flat-minima.md) | 2026-05-25 | 0.0 | 0 | 0 | IEEE Transactions on Pattern Analysis an |  | This work finds that the flatness of loss landscapes is also closely related to the MLP activation sparsity and can serve as a weaker assumption because it … |
| 30 | [End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inference](2606.27743-end-to-end-dynamic-sparsity-for-resource-adaptive-llm-inference.md) | 2026-06-26 | 0.0 | 0 | 0 |  |  | This work proposes Learning to Allocate (L2A), an end-to-end framework for resource-adaptive inference that introduces lightweight, budget-conditioned and … |
| 31 | [Sensitivity-Aware Thresholding and Token Routing for Activation Sparsification in Large Language Models](2607.08991-sensitivity-aware-thresholding-and-token-routing-for-activation-sparsi.md) | 2026-07-09 | 0.0 | 0 | 0 |  |  | SATS improves over the threshold-based sparsification baseline at matched actual sparsity and token routing yields a more favorable quality-throughput … |
| 32 | [SelectInfer: Selective Neuron Loading and Computation for On-Device LLMs](2607.18081-selectinfer-selective-neuron-loading-and-computation-for-on-device-llm.md) | 2026-07-20 | 0.0 | 0 | 0 |  |  | Evaluation across multiple datasets shows that SelectInfer achieves significant reductions in memory footprint and computation while preserving task … |
| 33 | [Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs](2607.27591-prox-training-free-ffn-activation-sparsity-via-approximate-intermediat.md) | 2026-07-30 | 0.0 | 0 | 0 |  |  | Prox is a two-stage training-free framework for sparse SwiGLU FFNs that outperforms training-free baselines at all sparsity levels, achieves up to a … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [SmallThinker: A Family of Efficient Large Language Models Natively Trained for Local Deployment](../../models-and-architectures/small-language-models/2507.20984-smallthinker-a-family-of-efficient-large-language-models-natively-trai.md) | Small language models (≤ ~4B) | 7.19 |
| [Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](../../model-conversion/dense-to-moe-upcycling/2502.04416-analytical-ffn-to-moe-restructuring-via-activation-pattern-analysis.md) | Dense → MoE upcycling & MoE-fication | 7.12 |
| [Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM](../../serving-systems/memory-and-offloading/2502.16963-make-llm-inference-affordable-to-everyone-augmenting-gpu-memory-with-n.md) | Memory management & offloading (weights, activations) | 4.93 |
| [WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning](../structured-pruning/2607.28418-wide-boosting-adaptive-llm-inference-via-token-level-dynamic-width-pru.md) | Structured pruning (layers, heads, width, experts) | 1.57 |
| [SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention](../../attention/sparse-attention/2604.13847-sparsebalance-load-balanced-long-context-training-with-dynamic-sparse.md) | Sparse attention (trainable & training-free) | 0.0 |
