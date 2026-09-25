# KV cache low-rank / latent / head compression

Compressing KV along the hidden/head dimension: low-rank projection, latent KV (MLA-style), head merging.

**16 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [KV cache](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/kv-cache.md](../../../overviews/kv-cache.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** The KV cache has a lot of **channel/rank redundancy**, especially in keys: selection needs only
O(log N) dimensions, while values need the full width ([Thin Keys, Full Values](2603.04427-thin-keys-full-values-reducing-kv-cache-via-low-dimensional-attention.md)). There are two routes.

* **Post-training** low-rank projection of K (and less of V). This is training-free or lightly calibrated: LeanK,
  ReCalKV, KQ-SVD, StiefAttention. It typically gives 50–75% K-cache reduction.
* **Architectural latent KV (MLA).** The strongest result, but it needs training or conversion:
  * DeepSeek MLA caches a ~512-d latent per token;
  * conversion methods (TransMLA, MHA2MLA, X-EcoMLA, CARE; see
    [`model-conversion/attention-conversion`](../../model-conversion/attention-conversion/README.md)) retrofit it
    into GQA models;
  * serving it well under tensor parallelism needs TPLA-style sharding.

The sound objective is to preserve the **QKᵀ inner product** (KQ-SVD, SAKI), not the keys themselves. Methods that
reconstruct K alone are provably suboptimal.

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [LeanK](2508.02215-leank-learnable-k-cache-channel-pruning-for-efficient-decoding.md) (MSR, EMNLP'25) | Learned static mask | Two-stage training of a hardware-aligned **K-channel pruning mask** plus a custom decode kernel | −70% K cache, −16–18% V cache, 1.3–1.45× attention speed |
| 2 | [TPLA](2508.15881-tpla-tensor-parallel-latent-attention-for-efficient-disaggregated-pref.md) | MLA serving | Shard the latent **and** per-head input dimension across TP ranks, then all-reduce; every head still sees the full latent | Makes MLA's small cache survive tensor parallelism; drop-in for MLA checkpoints |
| 3 | [KQ-SVD](2512.05916-kq-svd-compressing-the-kv-cache-with-provable-guarantees-on-attention.md) | Closed form | Optimal low-rank decomposition of the **attention matrix (K·Qᵀ)**, not of K | Provable fidelity; beats K-only SVD |
| 4 | [ReCalKV](2505.24357-recalkv-low-rank-kv-cache-compression-via-head-reordering-and-offline.md) | Post-training | Head-similarity reordering + grouped SVD for K; offline calibration for V | High compression with small loss |
| 5 | [Thin Keys, Full Values](2603.04427-thin-keys-full-values-reducing-kv-cache-via-low-dimensional-attention.md) | Theory + retrofit | Selection needs O(log N) dims; factor W_K by truncated SVD | Retrofits any model without pretraining from scratch |
| 6 | [LRQK](2510.23649-efficient-low-rank-attention-for-long-context-inference-in-large-langu.md) (NeurIPS'25) | Low-rank proxy + offload | Rank-r Q/K factors give proxy scores; top-k tokens fetched from a GPU/CPU cache | Long context on small GPUs |
| 7 | [STAR-KV](2606.08382-star-kv-low-rank-kv-cache-compression-via-soft-thresholding-for-adapti.md) | Adaptive rank | Differentiable soft-threshold rank per head/block plus low-rank-aware quantization | 75% KV compression, up to 20× combined |
| 8 | [OjaKV](2509.21623-ojakv-context-aware-online-low-rank-kv-cache-compression.md) (ACL'26) | Online subspace | **Online Oja updates** of the projection subspace; full rank for first and recent tokens | Robust to distribution shift |
| 9 | [StiefAttention](2601.21686-don-t-be-so-stief-learning-kv-cache-low-rank-approximation-over-the-st.md) | Post-training | Learn orthonormal bases on the Stiefel manifold minimizing **decoder-layer output** error; rank allocation | Better than SVD proxies |
| 10 | [SAKI](2608.03228-saki-score-aware-low-rank-key-indexing-with-random-matrix-noise-correc.md) | Index | Score-aware low-rank key index with random-matrix noise correction | Better top-k recall for sparse retrieval |

**Runtime notes.**
* If you control the architecture, **use MLA** (or GQA + a small head dim for K): it is the only 5–10×
  KV reduction with no quality loss.
* Implement the MLA "absorb" path (FlashMLA; see [TyphoonMLA](../../attention/kernels-and-io-aware/2509.21081-typhoonmla-a-mixed-naive-absorb-mla-kernel-for-shared-prefix.md) for shared prefixes) and TPLA sharding.
* For existing GQA models, low-rank K projection is cheap to add: it is one extra small GEMM fused into the K
  projection, and it composes with quantization and eviction.

## 🏆 Best of the best by impact score (top 10)

1. **[LeanK: Learnable K Cache Channel Pruning for Efficient Decoding](2508.02215-leank-learnable-k-cache-channel-pruning-for-efficient-decoding.md)** (2025-08) — This work proposes LeanK, a learning-based method that prunes unimportant key (K) cache channels by leveraging static channel sparsity by learning channel-wise static mask that could satisfy specific sparsity ratio and …  
   _score 5.56 · Conference on Empirical Methods in Natural Language Processi · 6 cites · 12▲ HF_
2. **[ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration](2505.24357-recalkv-low-rank-kv-cache-compression-via-head-reordering-and-offline.md)** (2025-09) — This work proposes ReCalKV, a post-training low-rank KV cache compression approach with tailored strategies for Keys and Values, which consistently outperforms existing low-rank compression methods, achieving high …  
   _score 4.54 · 15 cites · [code](https://github.com/XIANGLONGYAN/ReCalKV)_
3. **[SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning](2508.15212-spark-query-aware-unstructured-sparsity-with-recoverable-kv-cache-chan.md)** (2025-11) — SPARK is a training-free plug-and-play method that applies unstructured sparsity by pruning KV at the channel level, while dynamically restoring the pruned entries during attention score computation, and reduces KV …  
   _score 3.46 · accepted to AAAI 2026 · 6 cites · [code](https://github.com/Xnhyacinth/SparK)_
4. **[KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity](2512.05916-kq-svd-compressing-the-kv-cache-with-provable-guarantees-on-attention.md)** (2025-12) — This work introduces KQ-SVD, a simple and computationally efficient method that directly performs an optimal low-rank decomposition of the attention matrix via a closed-form solution and preserves attention outputs with …  
   _score 3.22 · 6 cites · [code](https://github.com/DamienLesens/KQ-SVD)_
5. **[TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill and Decode Inference](2508.15881-tpla-tensor-parallel-latent-attention-for-efficient-disaggregated-pref.md)** (2025-08) — Tensor-Parallel Latent Attention (TPLA) is proposed, a scheme that partitions both the latent representation and each head's input dimension across devices, performs attention independently per shard, and then combines …  
   _score 3.2 · 2 cites · 10▲ HF_
6. **[Efficient Low Rank Attention for Long-Context Inference in Large Language Models](2510.23649-efficient-low-rank-attention-for-long-context-inference-in-large-langu.md)** (2025-12) — Low Rank Query and Key attention is introduced, a two-stage framework that jointly decomposes full-precision query and key matrices into compact rank-\(r\) factors during the prefill stage, and then employs these …  
   _score 3.14 · NeurIPS 2025 · 4 cites · [code](https://github.com/tenghuilee/LRQK)_
7. **[STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control](2606.08382-star-kv-low-rank-kv-cache-compression-via-soft-thresholding-for-adapti.md)** (2026-06) — STAR-KV is proposed, an adaptive low-rank KV cache compression framework with fine-grained rank control that achieves up to 75% KV cache compression and up to 20x overall KV cache reduction when combined with …  
   _score 3.12 · 6 cites · [code](https://github.com/PriyanshBhatnagar/STAR-KV)_
8. **[OjaKV: Context-Aware Online Low-Rank KV Cache Compression](2509.21623-ojakv-context-aware-online-low-rank-kv-cache-compression.md)** (2026-04) — OjaKV is introduced, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation that achieves its strongest gains on very long-context benchmarks that require complex reasoning, …  
   _score 2.89 · Annual Meeting of the Association for Computational Linguist · 6 cites_
9. **[KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](2507.11273-kv-latent-dimensional-level-kv-cache-reduction-with-frequency-aware-ro.md)** (2025-07) — Down-sampling the Key-Value vector dimensions into a latent space can significantly reduce the KV Cache footprint and improve inference speed, only with a small amount of extra training, less than 1\% of pre-training …  
   _score 2.82 · ACL 2025 · 8 cites · [code](https://github.com/ShiLuohe/KV-Latent)_
10. **[KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs](2602.05929-kv-core-benchmarking-data-dependent-low-rank-compressibility-of-kv-cac.md)** (2026-02) — This study establishes a principled evaluation framework and the first large-scale benchmark of kv-cache compressibility in LLMs, offering insights for dynamic, data-aware compression and data-centric model development.  
   _score 2.32 · 3 cites · 2▲ HF_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[SAKI: Score-Aware Low-Rank Key Indexing with Random-Matrix Noise Correction for KV Retrieval](2608.03228-saki-score-aware-low-rank-key-indexing-with-random-matrix-noise-correc.md)** (2026-08-05; 0▲, 0 cites) — This work derives the expected attention score distortion caused by rank r key compression and shows that it yields a covariance weighted low rank objective, which motivates SAKI, …
- **[PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression](2608.23843-puzzlekv-page-wise-low-rank-decomposition-for-kv-cache-compression.md)** (2026-08-24; 0▲, 0 cites) — P PuzzleKV is proposed, a training- and calibration-free method that treats each completed page as an independent compression unit that achieves more than 96% of Full KV …

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [LeanK: Learnable K Cache Channel Pruning for Efficient Decoding](2508.02215-leank-learnable-k-cache-channel-pruning-for-efficient-decoding.md) | 2025-08-04 | 5.56 | 6 | 12 | Conference on Empirical Methods in Natur |  | This work proposes LeanK, a learning-based method that prunes unimportant key (K) cache channels by leveraging static channel sparsity by learning channel-wise … |
| 2 | [ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration](2505.24357-recalkv-low-rank-kv-cache-compression-via-head-reordering-and-offline.md) | 2025-09-27 | 4.54 | 15 | 0 |  | [✓](https://github.com/XIANGLONGYAN/ReCalKV) | This work proposes ReCalKV, a post-training low-rank KV cache compression approach with tailored strategies for Keys and Values, which consistently outperforms … |
| 3 | [SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning](2508.15212-spark-query-aware-unstructured-sparsity-with-recoverable-kv-cache-chan.md) | 2025-11-12 | 3.46 | 6 | 0 | accepted to AAAI 2026 | [✓](https://github.com/Xnhyacinth/SparK) | SPARK is a training-free plug-and-play method that applies unstructured sparsity by pruning KV at the channel level, while dynamically restoring the pruned … |
| 4 | [KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity](2512.05916-kq-svd-compressing-the-kv-cache-with-provable-guarantees-on-attention.md) | 2025-12-05 | 3.22 | 6 | 0 |  | [✓](https://github.com/DamienLesens/KQ-SVD) | This work introduces KQ-SVD, a simple and computationally efficient method that directly performs an optimal low-rank decomposition of the attention matrix via … |
| 5 | [TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill and Decode Inference](2508.15881-tpla-tensor-parallel-latent-attention-for-efficient-disaggregated-pref.md) | 2025-08-25 | 3.2 | 2 | 10 |  |  | Tensor-Parallel Latent Attention (TPLA) is proposed, a scheme that partitions both the latent representation and each head's input dimension across devices, … |
| 6 | [Efficient Low Rank Attention for Long-Context Inference in Large Language Models](2510.23649-efficient-low-rank-attention-for-long-context-inference-in-large-langu.md) | 2025-12-23 | 3.14 | 4 | 0 | NeurIPS 2025 | [✓](https://github.com/tenghuilee/LRQK) | Low Rank Query and Key attention is introduced, a two-stage framework that jointly decomposes full-precision query and key matrices into compact rank-\(r\) … |
| 7 | [STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control](2606.08382-star-kv-low-rank-kv-cache-compression-via-soft-thresholding-for-adapti.md) | 2026-06-07 | 3.12 | 6 | 0 |  | [✓](https://github.com/PriyanshBhatnagar/STAR-KV) | STAR-KV is proposed, an adaptive low-rank KV cache compression framework with fine-grained rank control that achieves up to 75% KV cache compression and up to … |
| 8 | [OjaKV: Context-Aware Online Low-Rank KV Cache Compression](2509.21623-ojakv-context-aware-online-low-rank-kv-cache-compression.md) | 2026-04-16 | 2.89 | 6 | 0 | Annual Meeting of the Association for Co |  | OjaKV is introduced, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation that achieves its strongest gains on … |
| 9 | [KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](2507.11273-kv-latent-dimensional-level-kv-cache-reduction-with-frequency-aware-ro.md) | 2025-07-15 | 2.82 | 8 | 0 | ACL 2025 | [✓](https://github.com/ShiLuohe/KV-Latent) | Down-sampling the Key-Value vector dimensions into a latent space can significantly reduce the KV Cache footprint and improve inference speed, only with a … |
| 10 | [KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs](2602.05929-kv-core-benchmarking-data-dependent-low-rank-compressibility-of-kv-cac.md) | 2026-02-07 | 2.32 | 3 | 2 |  |  | This study establishes a principled evaluation framework and the first large-scale benchmark of kv-cache compressibility in LLMs, offering insights for … |
| 11 | [EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection](2503.01586-elitekv-scalable-kv-cache-compression-via-rope-frequency-selection-and.md) | 2025-03-03 | 1.96 | 7 | 0 |  |  | EliteKV, a flexible modification framework for RoPE-based models supporting variable KV cache compression ratios, first identifies the intrinsic frequency … |
| 12 | [Thin Keys, Full Values: Reducing KV Cache via Low-Dimensional Attention Selection](2603.04427-thin-keys-full-values-reducing-kv-cache-via-low-dimensional-attention.md) | 2026-03-28 | 0.93 | 2 | 0 |  |  | Factored keys are introduced, which exploit this asymmetry to physically shrink the KV cache of any pretrained model without retraining from scratch -- unlike … |
| 13 | [SWAN: Sparse Winnowed Attention for Reduced Inference Memory via Decompression-Free KV-Cache Compression](2511.18936-swan-sparse-winnowed-attention-for-reduced-inference-memory-via-decomp.md) | 2025-11-24 | 0.0 | 0 | 0 |  |  | This work introduces SWAN, a novel, fine-tuning-free framework that eliminates significant computational overhead from explicit decompression steps, and uses … |
| 14 | [Don't be so Stief! Learning KV Cache low-rank approximation over the Stiefel manifold](2601.21686-don-t-be-so-stief-learning-kv-cache-low-rank-approximation-over-the-st.md) | 2026-05-29 | 0.0 | 0 | 0 |  |  | StiefAttention is introduced, a post-training KV-cache compression method that learns orthonormal projection bases by directly minimizing decoder-layer output … |
| 15 | [SAKI: Score-Aware Low-Rank Key Indexing with Random-Matrix Noise Correction for KV Retrieval](2608.03228-saki-score-aware-low-rank-key-indexing-with-random-matrix-noise-correc.md) | 2026-08-05 | 0.0 | 0 | 0 |  |  | This work derives the expected attention score distortion caused by rank r key compression and shows that it yields a covariance weighted low rank objective, … |
| 16 | [PuzzleKV: Page-Wise Low-Rank Decomposition for KV Cache Compression](2608.23843-puzzlekv-page-wise-low-rank-decomposition-for-kv-cache-compression.md) | 2026-08-24 | 0.0 | 0 | 0 |  |  | P PuzzleKV is proposed, a training- and calibration-free method that treats each completed page as an independent compression unit that achieves more than 96% … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](../cross-layer-sharing/2503.18893-xkv-cross-layer-kv-cache-compression-via-aligned-singular-vector-extra.md) | KV cache cross-layer sharing & merging | 6.59 |
| [IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference](../eviction-and-token-selection/2605.25475-indexmem-learned-kv-cache-eviction-with-latent-memory-for-long-context.md) | KV cache eviction / token selection / sparse retrieval | 2.62 |
| [SALS: Sparse Attention in Latent Space for KV cache Compression](../../attention/sparse-attention/2510.24273-sals-sparse-attention-in-latent-space-for-kv-cache-compression.md) | Sparse attention (trainable & training-free) | 2.22 |
| [Quality Recovery for Quantized KV Caches via Low-Rank Attention Adaptation](../quantization/2609.04263-quality-recovery-for-quantized-kv-caches-via-low-rank-attention-adapta.md) | KV cache quantization | 0.0 |
