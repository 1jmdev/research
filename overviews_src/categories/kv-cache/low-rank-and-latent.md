**Verdict.** The KV cache has a lot of **channel/rank redundancy**, especially in keys: selection needs only
O(log N) dimensions, while values need the full width ([[2603.04427|Thin Keys, Full Values]]). There are two routes.

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
| 1 | [[2508.02215\|LeanK]] (MSR, EMNLP'25) | Learned static mask | Two-stage training of a hardware-aligned **K-channel pruning mask** plus a custom decode kernel | −70% K cache, −16–18% V cache, 1.3–1.45× attention speed |
| 2 | [[2508.15881\|TPLA]] | MLA serving | Shard the latent **and** per-head input dimension across TP ranks, then all-reduce; every head still sees the full latent | Makes MLA's small cache survive tensor parallelism; drop-in for MLA checkpoints |
| 3 | [[2512.05916\|KQ-SVD]] | Closed form | Optimal low-rank decomposition of the **attention matrix (K·Qᵀ)**, not of K | Provable fidelity; beats K-only SVD |
| 4 | [[2505.24357\|ReCalKV]] | Post-training | Head-similarity reordering + grouped SVD for K; offline calibration for V | High compression with small loss |
| 5 | [[2603.04427\|Thin Keys, Full Values]] | Theory + retrofit | Selection needs O(log N) dims; factor W_K by truncated SVD | Retrofits any model without pretraining from scratch |
| 6 | [[2510.23649\|LRQK]] (NeurIPS'25) | Low-rank proxy + offload | Rank-r Q/K factors give proxy scores; top-k tokens fetched from a GPU/CPU cache | Long context on small GPUs |
| 7 | [[2606.08382\|STAR-KV]] | Adaptive rank | Differentiable soft-threshold rank per head/block plus low-rank-aware quantization | 75% KV compression, up to 20× combined |
| 8 | [[2509.21623\|OjaKV]] (ACL'26) | Online subspace | **Online Oja updates** of the projection subspace; full rank for first and recent tokens | Robust to distribution shift |
| 9 | [[2601.21686\|StiefAttention]] | Post-training | Learn orthonormal bases on the Stiefel manifold minimizing **decoder-layer output** error; rank allocation | Better than SVD proxies |
| 10 | [[2608.03228\|SAKI]] | Index | Score-aware low-rank key index with random-matrix noise correction | Better top-k recall for sparse retrieval |

**Runtime notes.**
* If you control the architecture, **use MLA** (or GQA + a small head dim for K): it is the only 5–10×
  KV reduction with no quality loss.
* Implement the MLA "absorb" path (FlashMLA; see [[2509.21081|TyphoonMLA]] for shared prefixes) and TPLA sharding.
* For existing GQA models, low-rank K projection is cheap to add: it is one extra small GEMM fused into the K
  projection, and it composes with quantization and eviction.
