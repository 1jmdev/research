# KV cache cross-layer sharing & merging

Sharing or merging KV across layers or heads (CLA, YOCO-style, MiniCache).

**21 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [KV cache](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/kv-cache.md](../../../overviews/kv-cache.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** "Cross-layer sharing" covers two different things.

**(a) Depth-wise sharing inside one model** (YOCO, CLA and descendants).
* Architectural variants (YOCO++, SkipV1Former, FusedKV, HySparse) need **pretraining** but give 2× or more KV savings.
  HySparse is the most interesting: full-attention layers act as the **oracle token selector and KV provider** for the
  sparse layers that follow them.
* Post-training variants (xKV, CommonKV, Stochastic KV Routing) exploit aligned singular vectors across layers.
  They reach up to 8× with small loss.

**(b) Sharing KV across requests, agents or models.** This is a systems topic:
* multi-LoRA agents share the base part of the cache (LRAgent);
* fuzzy/semantic sharing (SemShareKV, SparseX);
* cross-model KV translation.

**Security warning.** Any cross-tenant KV sharing is a **timing side channel** ([SafeKV](2508.08438-selective-kv-cache-sharing-to-mitigate-timing-side-channels-in-llm-inf.md)).

| # | Paper | Type | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [HySparse](2602.03560-hysparse-a-hybrid-sparse-attention-architecture-with-oracle-token-sele.md) | Architecture | Interleave 1 full-attention layer with N sparse layers that **reuse its top-k selection and its KV** | Cuts both compute and KV without proxy selectors; a strong design for new models |
| 2 | [xKV](2503.18893-xkv-cross-layer-kv-cache-compression-via-aligned-singular-vector-extra.md) (ICML'26) | Post-training | CKA shows dominant singular vectors of KV align across layers; **joint low-rank subspace per layer group** + selective reconstruction | Up to 8× KV compression, multi-turn safe |
| 3 | [LRAgent](2602.01053-lragent-efficient-kv-cache-sharing-for-multi-lora-llm-agents.md) (ICML'26) | Multi-LoRA serving | Split KV into a **shared base component** plus a small adapter-dependent component | Large memory savings for multi-agent LoRA systems |
| 4 | [Stochastic KV Routing](2604.22782-stochastic-kv-routing-enabling-adaptive-depth-wise-cache-sharing.md) | Training | Train with random depth-wise KV routing so layers can **drop their cache** at inference | Adaptive depth-wise sharing with no TTFT penalty |
| 5 | [SkipV1Former](2510.16807-improving-model-representation-and-reducing-kv-cache-via-skip-connecti.md) (NeurIPS'25) | Architecture | Each layer reuses **half its V heads from layer 1** | Better perplexity *and* ~25% less KV |
| 6 | [YOCO++](2604.13556-yoco-enhancing-yoco-with-kv-residual-connections-for-efficient-llm-inf.md) | Architecture | YOCO plus weighted KV residuals from the bottom layer | Best cross-layer method at 50% KV; beats the standard Transformer |
| 7 | [FusedKV](2512.03870-reconstructing-kv-caches-with-cross-layer-fusion-for-enhanced-transfor.md) | Architecture | Top-layer KV = learned fusion of bottom- and middle-layer KV (post-RoPE) | Closes the CLA/YOCO vs GQA quality gap |
| 8 | [SafeKV](2508.08438-selective-kv-cache-sharing-to-mitigate-timing-side-channels-in-llm-inf.md) | Security | Detect sensitive prefixes and isolate them in the radix-tree cache | Mitigates cross-tenant timing side channels at low cost |
| 9 | [SparseX](2606.01751-sparsex-efficient-segment-level-kv-cache-sharing-for-interleaved-llm-s.md) | Serving | Segment-level non-prefix KV reuse with sparse recomputation of affected tokens | Reuse beyond exact prefixes |
| 10 | [Cross-model KV translation](2608.30963-a-universal-context-reuse-layer-for-cross-model-kv-sharing.md) | Serving | Translate KV from one model to another (even across families) | Skip prefill on the target model |

**For model builders.** When designing a new model, combine GQA or MLA with **cross-layer KV sharing** (YOCO-style
halves, or HySparse full/sparse interleaving). Use sliding-window layers for the rest. Gemma-3/4, Hunyuan and several
2026 hybrids already mix these.

**For runtimes.** Support "KV aliasing", where layer *l* reads layer *k*'s pages, in the paged allocator. It is a small
change that unlocks every architecture in this category.

## 🏆 Best of the best by impact score (top 10)

1. **[LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents](2602.01053-lragent-efficient-kv-cache-sharing-for-multi-lora-llm-agents.md)** (2026-05) — LRAgent, a KV cache sharing framework for multi-LoRA agents, and Flash-LoRA-Attention, a kernel that reorders attention computation to avoid materializing the low-rank cache to full dimension, to achieve throughput and …  
   _score 7.48 · ICML 2026 · 8 cites · 8▲ HF · [code](https://github.com/hjeon2k/LRAgent)_
2. **[HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing](2602.03560-hysparse-a-hybrid-sparse-attention-architecture-with-oracle-token-sele.md)** (2026-02) — This work introduces Hybrid Sparse Attention (HySparse), a new architecture that interleaves each full attention layer with several sparse attention layers that achieves substantial performance gains while reducing KV …  
   _score 6.66 · 9 cites · 48▲ HF_
3. **[xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](2503.18893-xkv-cross-layer-kv-cache-compression-via-aligned-singular-vector-extra.md)** (2026-05) — xKV is a post-training method that uses Singular Value Decomposition to reduce the size of KV-Cache in LLMs with long context windows, improving compression rates and accuracy without significant performance degradation.  
   _score 6.59 · ICML 2026 · 9 cites · 5▲ HF · [code](https://github.com/abdelfattah-lab/xKV)_
4. **[Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference](2508.08438-selective-kv-cache-sharing-to-mitigate-timing-side-channels-in-llm-inf.md)** (2026-02) — SafeKV restores the efficiency of KV reuse while enforcing strong, practical privacy for multi-tenant LLM inference, and integrates lightweight detection and isolation directly into the serving runtime to eliminate …  
   _score 5.58 · 26 cites_
5. **[Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing](2604.22782-stochastic-kv-routing-enabling-adaptive-depth-wise-cache-sharing.md)** (2026-04) — It is demonstrated that dropping a layer's cache offers efficient optimization without information loss, and for larger models in data-constrained settings, this approach is suggestive of a regularization-like effect, …  
   _score 3.36 · 2 cites · 8▲ HF_
6. **[SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching](2509.24832-semsharekv-efficient-kvcache-sharing-for-semantically-similar-prompts.md)** (2025-12) — SemShareKV is proposed, a KV cache sharing and compression framework that accelerates LLM inference by reusing KVCache in semantically similar prompts by applying fuzzy token matching using locality-sensitive hashing on …  
   _score 3.21 · IJCNLP-AACL · 8 cites · [code](https://github.com/JasperZhao666/SemShareKV-public.git)_
7. **[Improving Model Representation and Reducing KV Cache via Skip Connections with First Value Heads](2510.16807-improving-model-representation-and-reducing-kv-cache-via-skip-connecti.md)** (2025-10) — This work proposes SkipV1Former, a Transformer variant that uses skip connections from the first layer's Value heads to strengthen model representation and reduce KV cache, and proposes a recipe for uptraining existing …  
   _score 3.01 · Neural Information Processing Systems (Neural Inf Process Sy · 4 cites · [code](https://github.com/Zhoutong-Wu/SkipV1Former)_
8. **[CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing](2508.16134-commonkv-compressing-kv-cache-with-cross-layer-parameter-sharing.md)** (2025-08) — Inspired by the high similarity observed in cross-layer hidden states, Singular Value Decomposition (SVD) is utilized to achieve weight sharing across adjacent parameters, resulting in a more easily mergeable latent KV …  
   _score 1.54 · 7 cites_
9. **[A Universal Context-Reuse Layer for Cross-Model KV Sharing](2608.30963-a-universal-context-reuse-layer-for-cross-model-kv-sharing.md)** (2026-08) — Results provide initial evidence that KV states can serve as transferable computational representations rather than strictly model-local caches, and motivate context mobility as a systems abstraction for reducing …  
   _score 1.52 · 1 cites_
10. **[Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers](2512.03870-reconstructing-kv-caches-with-cross-layer-fusion-for-enhanced-transfor.md)** (2026-02) — This work proposes FusedKV, whose top-layer KV caches are a learnable fusion of the most informative ones from the bottom and middle layers, and FusedKV-Lite, an cross-layer sharing approach, where top-layer KV caches …  
   _score 1.43 · 4 cites_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[RippleKV: Cross-Layer KV Cache Allocation via Perturbation Propagation](2608.08684-ripplekv-cross-layer-kv-cache-allocation-via-perturbation-propagation.md)** (2026-08-09; 0▲, 0 cites) — RippleKV is proposed, which allocates cache across layers by estimating how perturbations to each layer's value cache affect the final predictive distribution, and achieves the …
- **[KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs](2608.15943-kv-pipe-on-the-relation-between-kv-sharing-and-pipeline-parallel-effic.md)** (2026-08-16; 0▲, 0 cites) — KV layout is identified as a system--architecture degree of freedom for jointly improving pipeline-parallel training efficiency and long-context inference.
- **[Shared Global KV with Layer-Specific Local History](2609.28006-shared-global-kv-with-layer-specific-local-history.md)** (2026-09-23; 0▲, 0 cites) — This work derives a sufficient suffix schedule that reduces upper-layer construction work while preserving the complete cache in exact arithmetic, and derives a sufficient suffix …

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents](2602.01053-lragent-efficient-kv-cache-sharing-for-multi-lora-llm-agents.md) | 2026-05-31 | 7.48 | 8 | 8 | ICML 2026 | [✓](https://github.com/hjeon2k/LRAgent) | LRAgent, a KV cache sharing framework for multi-LoRA agents, and Flash-LoRA-Attention, a kernel that reorders attention computation to avoid materializing the … |
| 2 | [HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing](2602.03560-hysparse-a-hybrid-sparse-attention-architecture-with-oracle-token-sele.md) | 2026-02-03 | 6.66 | 9 | 48 |  |  | This work introduces Hybrid Sparse Attention (HySparse), a new architecture that interleaves each full attention layer with several sparse attention layers … |
| 3 | [xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](2503.18893-xkv-cross-layer-kv-cache-compression-via-aligned-singular-vector-extra.md) | 2026-05-27 | 6.59 | 9 | 5 | ICML 2026 | [✓](https://github.com/abdelfattah-lab/xKV) | xKV is a post-training method that uses Singular Value Decomposition to reduce the size of KV-Cache in LLMs with long context windows, improving compression … |
| 4 | [Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference](2508.08438-selective-kv-cache-sharing-to-mitigate-timing-side-channels-in-llm-inf.md) | 2026-02-09 | 5.58 | 26 | 0 |  |  | SafeKV restores the efficiency of KV reuse while enforcing strong, practical privacy for multi-tenant LLM inference, and integrates lightweight detection and … |
| 5 | [Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing](2604.22782-stochastic-kv-routing-enabling-adaptive-depth-wise-cache-sharing.md) | 2026-04-03 | 3.36 | 2 | 8 |  |  | It is demonstrated that dropping a layer's cache offers efficient optimization without information loss, and for larger models in data-constrained settings, … |
| 6 | [SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching](2509.24832-semsharekv-efficient-kvcache-sharing-for-semantically-similar-prompts.md) | 2025-12-16 | 3.21 | 8 | 0 | IJCNLP-AACL | [✓](https://github.com/JasperZhao666/SemShareKV-public.git) | SemShareKV is proposed, a KV cache sharing and compression framework that accelerates LLM inference by reusing KVCache in semantically similar prompts by … |
| 7 | [Improving Model Representation and Reducing KV Cache via Skip Connections with First Value Heads](2510.16807-improving-model-representation-and-reducing-kv-cache-via-skip-connecti.md) | 2025-10-23 | 3.01 | 4 | 0 | Neural Information Processing Systems (N | [✓](https://github.com/Zhoutong-Wu/SkipV1Former) | This work proposes SkipV1Former, a Transformer variant that uses skip connections from the first layer's Value heads to strengthen model representation and … |
| 8 | [CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing](2508.16134-commonkv-compressing-kv-cache-with-cross-layer-parameter-sharing.md) | 2025-08-22 | 1.54 | 7 | 0 |  |  | Inspired by the high similarity observed in cross-layer hidden states, Singular Value Decomposition (SVD) is utilized to achieve weight sharing across adjacent … |
| 9 | [A Universal Context-Reuse Layer for Cross-Model KV Sharing](2608.30963-a-universal-context-reuse-layer-for-cross-model-kv-sharing.md) | 2026-08-31 | 1.52 | 1 | 0 |  |  | Results provide initial evidence that KV states can serve as transferable computational representations rather than strictly model-local caches, and motivate … |
| 10 | [Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers](2512.03870-reconstructing-kv-caches-with-cross-layer-fusion-for-enhanced-transfor.md) | 2026-02-19 | 1.43 | 4 | 0 |  |  | This work proposes FusedKV, whose top-layer KV caches are a learnable fusion of the most informative ones from the bottom and middle layers, and FusedKV-Lite, … |
| 11 | [UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs](2502.00439-uniattn-reducing-inference-costs-via-softmax-unification-for-post-trai.md) | 2026-01-22 | 1.33 | 4 | 0 |  |  | Experiments show that UniAttn matches the performance of standard post-training while significantly reducing inference costs, outperforming existing efficient … |
| 12 | [SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving](2606.01751-sparsex-efficient-segment-level-kv-cache-sharing-for-interleaved-llm-s.md) | 2026-06-07 | 1.25 | 2 | 0 |  |  | SarseX is model-agnostic, training-free, and compatible with Prefix Cache, and it provides unified support for common online serving scenarios including … |
| 13 | [RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inference](2606.09937-rksc-reasoning-aware-kv-cache-sharing-and-confident-early-exit-for-mul.md) | 2026-06-07 | 1.2 | 0 | 0 | Accepted to the ICML 2026 Worksh | [✓](https://github.com/AnirudhSekar/RKSC) | RKSC (Reasoning-Aware KV Cache Sharing), a training-free inference framework that eliminates two structural redundancies in multi-branch LLM reasoning … |
| 14 | [ReasonCache: Accelerating Large Reasoning Model Serving through KV Cache Sharing](2507.21433-reasoncache-accelerating-large-reasoning-model-serving-through-kv-cach.md) | 2026-05-14 | 0.63 | 1 | 0 | International Workshop on Quality of Ser |  | Experimental evaluation demonstrates that ReasonCache achieves a peak throughput improvement of 89.2% and an average gain of 40-60%, leading to more responsive … |
| 15 | [KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging](2603.00907-kvslimmer-theoretical-insights-and-practical-optimizations-for-asymmet.md) | 2026-03-08 | 0.5 | 0 | 0 |  | [✓](https://github.com/lianjunl13-sudo/KVSlimmer) | This work introduces KVSlimmer, an efficient algorithm that captures exact Hessian information through a mathematically exact formulation, and derives a … |
| 16 | [Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing](2507.08045-krul-efficient-state-restoration-for-multi-turn-conversations-with-dyn.md) | 2025-08-26 | 0.3 | 1 | 0 |  |  | Krul is presented, a multi-turn LLM inference system that enables accurate and efficient KV cache restoration and introduces three key innovations: a … |
| 17 | [CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill](2602.16054-claa-cross-layer-attention-aggregation-for-accelerating-llm-prefill.md) | 2026-02-17 | 0.0 | 0 | 0 |  |  | This work introduces an Answer-Informed Oracle, which defines ground-truth token importance by measuring attention from generated answers back to the prompt, … |
| 18 | [YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference](2604.13556-yoco-enhancing-yoco-with-kv-residual-connections-for-efficient-llm-inf.md) | 2026-04-15 | 0.0 | 0 | 0 |  |  | YOCO++ is proposed, an enhanced YOCO that incorporates a weighted residual connection between the KVs of each bottom-half layer and the bottom layer that … |
| 19 | [RippleKV: Cross-Layer KV Cache Allocation via Perturbation Propagation](2608.08684-ripplekv-cross-layer-kv-cache-allocation-via-perturbation-propagation.md) | 2026-08-09 | 0.0 | 0 | 0 |  |  | RippleKV is proposed, which allocates cache across layers by estimating how perturbations to each layer's value cache affect the final predictive distribution, … |
| 20 | [KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs](2608.15943-kv-pipe-on-the-relation-between-kv-sharing-and-pipeline-parallel-effic.md) | 2026-08-16 | 0.0 | 0 | 0 |  |  | KV layout is identified as a system--architecture degree of freedom for jointly improving pipeline-parallel training efficiency and long-context inference. |
| 21 | [Shared Global KV with Layer-Specific Local History](2609.28006-shared-global-kv-with-layer-specific-local-history.md) | 2026-09-23 | 0.0 | 0 | 0 |  |  | This work derives a sufficient suffix schedule that reduces upper-layer construction work while preserving the complete cache in exact arithmetic, and derives … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse](../prefix-caching-and-reuse/2503.16525-kvshare-an-llm-service-system-with-efficient-and-effective-multi-tenan.md) | Prefix caching & KV reuse (RAG, multi-turn, agents) | 4.12 |
| [TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing](../prefix-caching-and-reuse/2604.03143-tokendance-scaling-multi-agent-llm-serving-via-collective-kv-cache-sha.md) | Prefix caching & KV reuse (RAG, multi-turn, agents) | 2.11 |
| [FlowMM: Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference](../eviction-and-token-selection/2511.05534-flowmm-cross-modal-information-flow-guided-kv-cache-merging-for-effici.md) | KV cache eviction / token selection / sparse retrieval | 1.74 |
| [SwiftCache: Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing](../offloading-and-hierarchical-storage/2606.16135-swiftcache-efficient-llm-serving-for-multi-turn-conversations-with-het.md) | KV cache offloading & hierarchical storage | 1.3 |
| [WeightedKV: Attention Scores Weighted Key-Value Cache Merging for Large Language Models](../eviction-and-token-selection/2503.01330-weightedkv-attention-scores-weighted-key-value-cache-merging-for-large.md) | KV cache eviction / token selection / sparse retrieval | 1.26 |
| [Enabling KV Caching of Shared Prefix for Diffusion Language Models](../../decoding/diffusion-llm-inference/2606.07571-enabling-kv-caching-of-shared-prefix-for-diffusion-language-models.md) | Diffusion LLM inference acceleration | 1.2 |
| [Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction](../eviction-and-token-selection/2605.18053-protection-is-nearly-all-you-need-structural-protection-dominates-scor.md) | KV cache eviction / token selection / sparse retrieval | 1.14 |
| [KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language Models](../eviction-and-token-selection/2512.06727-kv-car-kv-cache-compression-using-autoencoders-and-kv-reuse-in-large-l.md) | KV cache eviction / token selection / sparse retrieval | 0.0 |
| [SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation](../eviction-and-token-selection/2607.16213-selkv-selective-kv-cache-merging-with-per-token-merge-or-drop-and-atte.md) | KV cache eviction / token selection / sparse retrieval | 0.0 |
| [Affix Cache for Diffusion Large Language Models](../../decoding/diffusion-llm-inference/2608.26140-affix-cache-for-diffusion-large-language-models.md) | Diffusion LLM inference acceleration | 0.0 |
| [KVShareArena: KV-Cache Reuse Across Contexts and Model Checkpoints](../prefix-caching-and-reuse/2609.10266-kvsharearena-kv-cache-reuse-across-contexts-and-model-checkpoints.md) | Prefix caching & KV reuse (RAG, multi-turn, agents) | 0.0 |
| [HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](../../attention/sparse-attention/2609.26368-hysparse2-hybrid-sparse-attention-with-two-level-kv-sharing.md) | Sparse attention (trainable & training-free) | 0.0 |
