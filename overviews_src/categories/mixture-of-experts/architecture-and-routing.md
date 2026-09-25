**Verdict.** The frontier MoE design has converged:
* **fine-grained experts** (64–512, top-4 to top-10) plus 1–2 **shared experts**;
* **sigmoid or softmax top-k routing with auxiliary-loss-free bias balancing**, with balance measured over the **global
  batch** (DeepSeek-V3 / Qwen3);
* **~3–10% activation ratio** (DeepSeek-V3 37B/671B, Qwen3-235B-A22B, Ling, Nemotron 3).

2025–26 research refined *how sparse* and *which experts*, rather than replacing the recipe.

**Sparsity and scaling laws** (most in [`training/scaling-laws`](../../training/scaling-laws/README.md)):
* [[2507.17702|Efficiency Leverage]] (Ling): a 0.85B-active MoE matches a 6.1B dense at >7× less compute. Leverage is
  set by activation ratio and granularity.
* [[2501.12370|Parameters vs FLOPs]]: an optimal sparsity exists for a given budget.
* [[2502.05172|Joint MoE scaling laws]] (ICML'25): MoE can be **more memory-efficient** than dense.
* [[2508.18672|Optimal sparsity for reasoning]]: reasoning needs *active FLOPs* and data per parameter, not just total
  parameters; memorization likes sparsity.
* [[2608.10605|Compute-optimal ≠ cluster-optimal]]: with FLOPs alone, sparser is always better; the real optimum comes
  from **systems constraints**.
* [[2506.12119]]: MoE beats dense under *strictly equal* compute, memory and data (with data reuse).

**Architecture refinements that stuck:**
* **LatentMoE** ([[2601.18089]], NVIDIA): experts operate in a down-projected latent space, giving better accuracy per
  FLOP and per parameter. Used in Nemotron-3 Super/Ultra.
* **Router–expert coupling and specialization:**
  * [[2512.23447|ERC loss]]: each expert must respond most to its own router's proxy token;
  * [[2505.22323|orthogonality + variance losses]] (NeurIPS'25 oral): +23.79%;
  * [[2501.13074|Autonomy-of-Experts]] (ICML'25): experts self-select by activation norm, no router.
* **Parameter sharing across depth:**
  * [[2605.06665|UniPool]]: one global expert pool read by per-layer routers; 42–67% of the expert parameters at equal
    or better quality;
  * [[2506.18945|Chain-of-Experts]]: iterative intra-layer expert communication.
* **Scaling embeddings instead of experts:** [[2601.21204|LongCat-Flash-Lite]] puts 30B of 68.5B parameters into
  n-gram embeddings.

**Serving-aware designs:**
* [[2604.20156|Temporally extended MoE]]: switch experts rarely (options framework), so offloaded experts stay resident.
* [[2507.08771|BlockFFN]]: chunk-consistent sparsity for speculative decoding.
* [[2605.27358|MobileMoE]]: an on-device sweet spot of moderate sparsity + fine-grained + shared experts; 2.2–3.4×
  faster decode than dense on phones.

**Reality check.** Expert "specialization" mostly reflects hidden-state geometry, not domains ([[2604.09780|The Myth of
Expert Specialization]]). Prompt-level routing does not predict rollout routing, which matters for prefetching.
**Super experts** ([[2507.23279]]) are a few experts that create the massive activations and attention sinks. Never
prune or aggressively quantize them.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2601.18089\|LatentMoE]] (NVIDIA) | Architecture | Route and compute experts in a compressed latent dimension | Best accuracy per FLOP and per parameter; adopted by Nemotron-3 Super/Ultra |
| 2 | [[2507.17702\|Towards Greater Leverage]] (Ling) | Scaling law | "Efficiency leverage" as a function of activation ratio, granularity and compute | 0.85B-active ≈ 6.1B dense at >7× less compute (1T tokens) |
| 3 | [[2505.22323\|Advancing Expert Specialization]] (NeurIPS'25 oral) | Training loss | Orthogonality loss (distinct token types per expert) + variance loss (sharper routing), compatible with balance loss | Up to +23.79% over auxiliary-loss MoE baselines; no architecture change |
| 4 | [[2512.23447\|ERC loss: coupling experts and routers]] | Training loss | Router embeddings as proxy tokens; each expert must respond most to its own proxy, and vice versa | Scales with the number of experts, not tokens; better MoEs at 3–15B over trillions of tokens |
| 5 | [[2508.18672\|Optimal sparsity for reasoning]] | Scaling study | Separate active FLOPs and tokens per parameter | Reasoning wants active compute and data; GRPO/TTC don't change the trend |
| 6 | [[2608.10605\|Compute-optimal is not cluster-optimal]] | Systems-aware scaling | Add all-to-all, memory and cluster constraints (MOSAIC) | Real optimal sparsity comes from the cluster, not FLOPs |
| 7 | [[2605.06665\|UniPool]] | Architecture | Global shared expert pool + per-layer routers + pool-level balance loss | Matches layer-wise MoE with 41.6–66.7% of the expert parameters |
| 8 | [[2501.13074\|Autonomy-of-Experts]] (ICML'25) | Router-free | Experts pre-compute (low-rank) activations; the top norms proceed | Beats router-based MoE at 700M–4B |
| 9 | [[2507.23279\|Super Experts]] | Analysis | A handful of experts cause massive activations and attention sinks; model-specific, data-agnostic | Pruning them collapses the model; protect them in compression |
| 10 | [[2507.07024\|FlexOlmo]] (NeurIPS'25) | Modular data | Independently trained domain experts merged with domain-informed routing, no joint training; opt-in/out at inference | Surpasses a standard MoE trained without data restrictions at equal FLOPs |
| 11 | [[2605.27358\|MobileMoE]] | On-device | Memory+compute-optimal MoE shape for phones + QAT | Beats OLMoE-1B-7B with 60% fewer parameters; 2.2–3.4× faster decode than dense |
| 12 | [[2604.20156\|Temporally extended MoE]] | Serving-aware | Options framework: keep the same experts for spans of tokens | Low switching rates at up to 90% of base accuracy after light conversion |
| 13 | [[2506.18945\|Chain-of-Experts]] | Architecture | Sequential expert iterations inside a layer with re-routing | 2 iterations ≈ 3× width; 17.6–42% less memory |
| 14 | [[2603.11535\|Expert threshold routing]] | Routing | Causal per-token thresholds: dynamic compute + balance without auxiliary loss | 1.6× token efficiency vs token-choice at 2.4B |

**Also useful.**
* Routing variants: [[2511.06494]] (route by sequence), [[2508.12801|max-score routing]], [[2606.17952|SoftMoE]],
  [[2506.21328|latent prototype routing]], [[2506.14038]] (similarity-preserving balance).
* Router post-training: [[2511.07419|RoMA]] (router-only fine-tuning for generalization), [[2504.07964|C3PO]]
  (test-time expert re-mixing).
* Expert granularity: [[2602.05711|OmniMoE]] (atomic experts + Cartesian-product router),
  [[2602.06154|MoSE]] (slimmable experts), [[2603.04971]] (virtual width).
* Attention MoE: [[2505.07260|UMoE]] (NeurIPS'25).
* Other: [[2508.07785|Grove MoE]] (big.LITTLE adjugate experts), [[2605.06663|EMO]] (emergent modularity),
  [[2601.21420|ConceptMoE]], [[2503.05447|Linear-MoE]].
* Survey: [[2503.07137]].

**Recommendation for model builders.**
* Default to DeepSeek-V3-style fine-grained + shared experts, aux-loss-free global-batch balancing, 1:16–1:32
  activation, and a specialization loss (orthogonality/ERC). Consider LatentMoE for better FLOP efficiency.
* Choose sparsity with systems-aware laws: your all-to-all and HBM budget, not FLOPs alone.
* For reasoning-heavy models, don't starve active parameters.

**For runtime builders.** Plan for top-8+ routing over 128–512 experts, shared experts, and latent-space experts. Keep
**super experts** in high precision and pinned in GPU memory. Routing is stable within spans but not predictable from
the prompt, so prefetch on recent-token routing.
