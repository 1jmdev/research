# Compression: pruning, sparsity, low-rank, distillation (2025–2026)

> Synthesis of [`papers/compression`](../papers/compression/README.md). Quantization has its own page
> ([quantization](quantization.md)); MoE-specific compression is in [MoE](mixture-of-experts.md).

## TL;DR

* **Weight pruning is mostly a 2:4-kernel story now.** Unstructured 50% sparsity rarely speeds up GPU inference.
  [[2511.13061|MACKO]] is the first SpMV format to get 1.2–1.5× at 50% unstructured sparsity on GPUs. 2:4
  semi-structured sparsity gets real speedups on Ampere+ tensor cores. Improve 2:4 accuracy with
  [[2510.05528|ARMOR]] (adaptive matrix factorization, ICLR'26), [[2502.00258|ProxSparse]] (learned masks) and
  [[2505.23049|DenoiseRotator]] (rotate to concentrate importance first: −58% PPL gap on Llama-3-70B 2:4). BitNet
  models are naturally friendly to N:M sparsity ([[2603.05168|Sparse-BitNet]]).
* **Activation sparsity is the better inference lever**: it needs no retraining and composes with quantization.
  * [[2504.19449|R-Sparse]] (ICLR'25): rank-aware, 50% model-level sparsity, 43% end-to-end speedup.
  * [[2507.01299|LaRoSA]] (rotate, then sparsify activations; 1.30× at 40% on Llama-2-7B) and
    [[2505.19427|WINA]] (weight-informed, beats TEAL).
  * [[2505.14884|Polar Sparsity]] (NeurIPS'25) is the first to scale *contextual* sparsity to **large batches**, via
    head-level attention sparsity: 2.2× end-to-end.
  * Architectures designed for it: [[2506.06644|Spark Transformer]] (8% FFN neurons active, top-256 attention;
    1.79× on CPU) and [[2507.08771|BlockFFN]] (chunk-level sparsity plus speculative decoding on devices).
* **Structured (depth/width) pruning + distillation** is how small models are made from big ones (Minitron
  lineage). [[2502.07780|DarwinLM]] (evolutionary search, 5× less healing data than ShearedLlama),
  [[2505.02819|ReplaceMe]] (replace pruned blocks by a linear map, training-free, 25% pruning at 90% quality),
  [[2502.15618|Probe Pruning]] (dynamic, batch-wise).
* **Low-rank (SVD) compression** matured. [[2503.12340|SVD-LLM V2]], [[2502.02723|Dobi-SVD]] (truncate
  *activations*), [[2604.01609|Swift-SVD]] (closed form, 3–70× faster compression), [[2509.22075|CoSpaDi]] (sparse
  dictionary learning instead of one subspace) and [[2512.03383|UniQL]] (joint quantization + low-rank with on-device
  configurable rates). At 20–40% compression they now rival structured pruning.
* **Lossless weight compression** (entropy coding of BF16 exponents) is free memory: [[2502.00922|Huff-LLM]].
  DFloat11-style work shows ~30% savings with exact outputs.

## Distillation ([folder](../papers/compression/knowledge-distillation/README.md))

2026's big shift is **on-policy (self-)distillation**, which sits between SFT and RL:

* [[2604.13016|Rethinking on-policy distillation]]: phenomenology plus recipe. OPD works when teacher and student
  share thinking patterns; the signal concentrates on a small set of high-probability tokens (97–99% of the mass).
* [[2603.25562|Revisiting OPD]]: three failure modes (token imbalance, unreliable teacher on student prefixes,
  tokenizer mismatch) and fixes (+19.8%).
* Self-distillation as RL: [[2601.18734|OPSD]], [[2601.20802|SDPO]], [[2604.03128|RLSD]] and
  [[2605.15155|SDAR]] (agentic). Caveat: [[2603.24472]] shows self-distillation can suppress uncertainty expression
  and hurt out-of-distribution reasoning by up to 40%.
* Continual learning: [[2601.19897|SDFT]] (learn from demonstrations on-policy, with less forgetting).
* Cross-tokenizer distillation is covered in [model conversion](model-conversion.md).

## Practical recipe for a small, fast model (2026)

1. Start from a strong teacher.
2. Prune depth and width with a search method (DarwinLM / Minitron-style) → **distil** on ~10–100B tokens.
3. Optionally convert attention to hybrid ([model conversion](model-conversion.md)).
4. Quantize (W4A16 or NVFP4) → add activation sparsity at inference.
5. Evaluate **agentic** ability, not just perplexity: [[2505.19433|ACBench]] finds 4-bit keeps tool use (−1–3%)
   but loses 10–15% on real-world applications.
