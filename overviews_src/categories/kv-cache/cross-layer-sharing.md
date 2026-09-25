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

**Security warning.** Any cross-tenant KV sharing is a **timing side channel** ([[2508.08438|SafeKV]]).

| # | Paper | Type | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2602.03560\|HySparse]] | Architecture | Interleave 1 full-attention layer with N sparse layers that **reuse its top-k selection and its KV** | Cuts both compute and KV without proxy selectors; a strong design for new models |
| 2 | [[2503.18893\|xKV]] (ICML'26) | Post-training | CKA shows dominant singular vectors of KV align across layers; **joint low-rank subspace per layer group** + selective reconstruction | Up to 8× KV compression, multi-turn safe |
| 3 | [[2602.01053\|LRAgent]] (ICML'26) | Multi-LoRA serving | Split KV into a **shared base component** plus a small adapter-dependent component | Large memory savings for multi-agent LoRA systems |
| 4 | [[2604.22782\|Stochastic KV Routing]] | Training | Train with random depth-wise KV routing so layers can **drop their cache** at inference | Adaptive depth-wise sharing with no TTFT penalty |
| 5 | [[2510.16807\|SkipV1Former]] (NeurIPS'25) | Architecture | Each layer reuses **half its V heads from layer 1** | Better perplexity *and* ~25% less KV |
| 6 | [[2604.13556\|YOCO++]] | Architecture | YOCO plus weighted KV residuals from the bottom layer | Best cross-layer method at 50% KV; beats the standard Transformer |
| 7 | [[2512.03870\|FusedKV]] | Architecture | Top-layer KV = learned fusion of bottom- and middle-layer KV (post-RoPE) | Closes the CLA/YOCO vs GQA quality gap |
| 8 | [[2508.08438\|SafeKV]] | Security | Detect sensitive prefixes and isolate them in the radix-tree cache | Mitigates cross-tenant timing side channels at low cost |
| 9 | [[2606.01751\|SparseX]] | Serving | Segment-level non-prefix KV reuse with sparse recomputation of affected tokens | Reuse beyond exact prefixes |
| 10 | [[2608.30963\|Cross-model KV translation]] | Serving | Translate KV from one model to another (even across families) | Skip prefill on the target model |

**For model builders.** When designing a new model, combine GQA or MLA with **cross-layer KV sharing** (YOCO-style
halves, or HySparse full/sparse interleaving). Use sliding-window layers for the rest. Gemma-3/4, Hunyuan and several
2026 hybrids already mix these.

**For runtimes.** Support "KV aliasing", where layer *l* reads layer *k*'s pages, in the paged allocator. It is a small
change that unlocks every architecture in this category.
