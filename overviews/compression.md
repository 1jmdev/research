# Compression: pruning, sparsity, low-rank, distillation (2025–2026)

> Synthesis of [`papers/compression`](../papers/compression/README.md). Quantization has its own page
> ([quantization](quantization.md)); MoE-specific compression is in [MoE](mixture-of-experts.md).

## TL;DR

* **Weight pruning is mostly a 2:4-kernel story now.** Unstructured 50% sparsity rarely speeds up GPU inference.
  [MACKO](../papers/compression/unstructured-and-semi-structured-pruning/2511.13061-macko-sparse-matrix-vector-multiplication-for-low-sparsity.md) is the first SpMV format to get 1.2–1.5× at 50% unstructured sparsity on GPUs. 2:4
  semi-structured sparsity gets real speedups on Ampere+ tensor cores. Improve 2:4 accuracy with
  [ARMOR](../papers/compression/unstructured-and-semi-structured-pruning/2510.05528-armor-high-performance-semi-structured-pruning-via-adaptive-matrix-fac.md) (adaptive matrix factorization, ICLR'26), [ProxSparse](../papers/compression/unstructured-and-semi-structured-pruning/2502.00258-proxsparse-regularized-learning-of-semi-structured-sparsity-masks-for.md) (learned masks) and
  [DenoiseRotator](../papers/compression/unstructured-and-semi-structured-pruning/2505.23049-denoiserotator-enhance-pruning-robustness-for-llms-via-importance-conc.md) (rotate to concentrate importance first: −58% PPL gap on Llama-3-70B 2:4). BitNet
  models are naturally friendly to N:M sparsity ([Sparse-BitNet](../papers/compression/unstructured-and-semi-structured-pruning/2603.05168-sparse-bitnet-1-58-bit-llms-are-naturally-friendly-to-semi-structured.md)).
* **Activation sparsity is the better inference lever**: it needs no retraining and composes with quantization.
  * [R-Sparse](../papers/compression/activation-sparsity/2504.19449-r-sparse-rank-aware-activation-sparsity-for-efficient-llm-inference.md) (ICLR'25): rank-aware, 50% model-level sparsity, 43% end-to-end speedup.
  * [LaRoSA](../papers/compression/activation-sparsity/2507.01299-la-rosa-enhancing-llm-efficiency-via-layerwise-rotated-sparse-activati.md) (rotate, then sparsify activations; 1.30× at 40% on Llama-2-7B) and
    [WINA](../papers/compression/activation-sparsity/2505.19427-wina-weight-informed-neuron-activation-for-accelerating-large-language.md) (weight-informed, beats TEAL).
  * [Polar Sparsity](../papers/compression/activation-sparsity/2505.14884-polar-sparsity-high-throughput-batched-llm-inferencing-with-scalable-c.md) (NeurIPS'25) is the first to scale *contextual* sparsity to **large batches**, via
    head-level attention sparsity: 2.2× end-to-end.
  * Architectures designed for it: [Spark Transformer](../papers/compression/activation-sparsity/2506.06644-spark-transformer-reactivating-sparsity-in-ffn-and-attention.md) (8% FFN neurons active, top-256 attention;
    1.79× on CPU) and [BlockFFN](../papers/compression/activation-sparsity/2507.08771-blockffn-towards-end-side-acceleration-friendly-mixture-of-experts-wit.md) (chunk-level sparsity plus speculative decoding on devices).
* **Structured (depth/width) pruning + distillation** is how small models are made from big ones (Minitron
  lineage). [DarwinLM](../papers/compression/structured-pruning/2502.07780-darwinlm-evolutionary-structured-pruning-of-large-language-models.md) (evolutionary search, 5× less healing data than ShearedLlama),
  [ReplaceMe](../papers/compression/structured-pruning/2505.02819-replaceme-network-simplification-via-depth-pruning-and-transformer-blo.md) (replace pruned blocks by a linear map, training-free, 25% pruning at 90% quality),
  [Probe Pruning](../papers/compression/structured-pruning/2502.15618-probe-pruning-accelerating-llms-through-dynamic-pruning-via-model-prob.md) (dynamic, batch-wise).
* **Low-rank (SVD) compression** matured. [SVD-LLM V2](../papers/compression/low-rank-decomposition/2503.12340-svd-llm-v2-optimizing-singular-value-truncation-for-large-language-mod.md), [Dobi-SVD](../papers/compression/low-rank-decomposition/2502.02723-dobi-svd-differentiable-svd-for-llm-compression-and-some-new-perspecti.md) (truncate
  *activations*), [Swift-SVD](../papers/compression/low-rank-decomposition/2604.01609-swift-svd-theoretical-optimality-meets-practical-efficiency-in-low-ran.md) (closed form, 3–70× faster compression), [CoSpaDi](../papers/compression/low-rank-decomposition/2509.22075-cospadi-compressing-llms-via-calibration-guided-sparse-dictionary-lear.md) (sparse
  dictionary learning instead of one subspace) and [UniQL](../papers/compression/low-rank-decomposition/2512.03383-uniql-unified-quantization-and-low-rank-compression-for-adaptive-edge.md) (joint quantization + low-rank with on-device
  configurable rates). At 20–40% compression they now rival structured pruning.
* **Lossless weight compression** (entropy coding of BF16 exponents) is free memory: [Huff-LLM](../papers/compression/_general/2502.00922-huff-llm-end-to-end-lossless-compression-for-efficient-llm-inference.md).
  DFloat11-style work shows ~30% savings with exact outputs.

## Distillation ([folder](../papers/compression/knowledge-distillation/README.md))

2026's big shift is **on-policy (self-)distillation**, which sits between SFT and RL:

* [Rethinking on-policy distillation](../papers/compression/knowledge-distillation/2604.13016-rethinking-on-policy-distillation-of-large-language-models-phenomenolo.md): phenomenology plus recipe. OPD works when teacher and student
  share thinking patterns; the signal concentrates on a small set of high-probability tokens (97–99% of the mass).
* [Revisiting OPD](../papers/compression/knowledge-distillation/2603.25562-revisiting-on-policy-distillation-empirical-failure-modes-and-simple-f.md): three failure modes (token imbalance, unreliable teacher on student prefixes,
  tokenizer mismatch) and fixes (+19.8%).
* Self-distillation as RL: [OPSD](../papers/compression/knowledge-distillation/2601.18734-self-distilled-reasoner-on-policy-self-distillation-for-large-language.md), [SDPO](../papers/training/rl-for-reasoning/2601.20802-reinforcement-learning-via-self-distillation.md), [RLSD](../papers/compression/knowledge-distillation/2604.03128-self-distilled-rlvr.md) and
  [SDAR](../papers/training/rl-for-reasoning/2605.15155-self-distilled-agentic-reinforcement-learning.md) (agentic). Caveat: [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?](../papers/compression/knowledge-distillation/2603.24472-why-does-self-distillation-sometimes-degrade-the-reasoning-capability.md) shows self-distillation can suppress uncertainty expression
  and hurt out-of-distribution reasoning by up to 40%.
* Continual learning: [SDFT](../papers/compression/knowledge-distillation/2601.19897-self-distillation-enables-continual-learning.md) (learn from demonstrations on-policy, with less forgetting).
* Cross-tokenizer distillation is covered in [model conversion](model-conversion.md).

## Practical recipe for a small, fast model (2026)

1. Start from a strong teacher.
2. Prune depth and width with a search method (DarwinLM / Minitron-style) → **distil** on ~10–100B tokens.
3. Optionally convert attention to hybrid ([model conversion](model-conversion.md)).
4. Quantize (W4A16 or NVFP4) → add activation sparsity at inference.
5. Evaluate **agentic** ability, not just perplexity: [ACBench](../papers/compression/unstructured-and-semi-structured-pruning/2505.19433-can-compressed-llms-truly-act-an-empirical-evaluation-of-agentic-capab.md) finds 4-bit keeps tool use (−1–3%)
   but loses 10–15% on real-world applications.
