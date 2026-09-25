**Verdict.** This is the most crowded sub-field in the corpus: 240 papers, most of them "H2O/SnapKV + a new
importance score". Three things matter in practice.

1. **Query-agnostic vs query-aware.**
   * SnapKV-style *query-aware* eviction breaks when you **reuse the compressed cache** for another question (prefix
     caching, multi-turn, agents). The Pitfalls paper and ShotKV show real failures: ignored instructions, system-prompt
     leakage.
   * Query-agnostic methods (KVzip, Expected Attention, Compactor, KVzap) are the ones compatible with a serving system
     that shares prefixes.
2. **Eviction during decoding for reasoning models** is a different problem from prefill compression. Long chains of
   thought are redundant (R-KV, RPC, TriAttention, LazyEviction), but reasoning-critical heads must keep a full cache
   (RLKV).
3. **Learned eviction beats heuristics.** A small amount of training wins: DMS needs only 1K steps for 8×, and
   retention gates and Fast KVzip gates are cheap. Architecture-level solutions compete with all of this (MLA, sparse
   attention; see the attention and low-rank categories). Prefer sparse *retrieval* (keep everything, read only top-k)
   when memory allows.

### Hand ranking

| # | Paper | Setting | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.23416\|KVzip]] (NeurIPS'25) | Query-agnostic, prefill | Score KV pairs by how well the LLM can **reconstruct the context** from them | 3–4× smaller cache, ~2× faster FlashAttention decode, negligible loss; **reusable across queries** |
| 2 | [[2506.05345\|DMS: Inference-time hyper-scaling]] (NVIDIA, NeurIPS'25) | Learned, decode | Dynamic memory sparsification with **delayed eviction**, retrofitted in **1K training steps** | 8× compression beats training-free sparse attention; spend the saved memory on longer/more reasoning for higher accuracy at equal compute |
| 3 | [[2604.04921\|TriAttention]] | Reasoning, decode | Pre-RoPE Q/K **concentrate around fixed centers**, so key importance is a trigonometric series of distance | Stable importance for long reasoning where post-RoPE scoring fails |
| 4 | [[2505.24133\|R-KV]] (NeurIPS'25) | Reasoning, decode | Importance + **redundancy** scoring for chain-of-thought tokens | ~100% of full-KV accuracy at 10% cache |
| 5 | [[2510.00636\|Expected Attention]] (NVIDIA kvpress) | Query-agnostic | Closed-form expected attention from the future-query distribution; FlashAttention-compatible | Principled, plug-in; code in kvpress |
| 6 | [[2510.08525\|RLKV]] | Reasoning | Uses RL to find **which heads matter for reasoning**; full cache for those, compressed for the rest | Head-level allocation for generative reasoning |
| 7 | [[2505.13866\|RPC]] (NeurIPS'25) | Reasoning, decode | Periodic compression of the reasoning path using a recent-query selector window | QwQ-32B 1.6× throughput, −1.2% AIME |
| 8 | [[2502.01068\|FastKV]] (ACL'26) | Prefill + decode | Full context until a Token-Selective Propagation layer, then only important tokens (decouples prefill savings from KV budget) | Speeds up **both** prefill and decode |
| 9 | [[2502.14051\|RocketKV]] (ICML'25) | Decode | Coarse permanent eviction + fine hybrid top-k sparse attention | Up to 400× compression, 3.7× end-to-end |
| 10 | [[2601.17668\|Fast KVzip]] | Learned gates | Sink-attention gating modules trained with forward passes only | Near-lossless at high ratios, negligible overhead |
| 11 | [[2503.12491\|CAKE]] (ICLR'25) | Layer budgets | Cascading per-layer "cake-slicing" allocation from spatial and temporal attention dispersion | Layer-aware budgets |
| 12 | [[2504.15364\|KeyDiff]] (NeurIPS'25) | Streaming, edge | Keep **geometrically distinctive keys**; no attention scores needed (FlashAttention-friendly) | <0.04% gap at 8K budget |
| 13 | [[2605.09649\|Make Each Token Count]] | Learned global | Learned retention gates plus one global cross-layer/head policy; eviction can **beat** the full cache | Beats full-KV on long context |
| 14 | [[2502.00299\|ChunkKV]] (NeurIPS'25) | Prefill | Semantic chunks as the eviction unit plus cross-layer index reuse | Better than token-level eviction |
| 15 | [[2506.15745\|InfiniPot-V]] (NeurIPS'25) | Streaming video | Hard memory cap: temporal redundancy + value-norm ranking | Length-independent memory for video |

**Must-read cautionary papers.**
* [[2510.00231|The Pitfalls of KV Cache Compression]]: instructions get dropped and system prompts leak.
* [[2502.01941|KVFundaBench / ShotKV]]: retrieval stays robust while reasoning collapses.
* [[2512.12008|Hold Onto That Thought]] and [[2607.01520|The risk of KV cache compression]].
* [[2609.03430|Random Attention]] (151▲): random eviction is a strong baseline for reasoning.
* [[2605.17613|VeriCache]]: turns a lossy cache into lossless inference by verification, speculative-decoding style.

**Runtime recommendations.**
* Implement **block-granular (page-level) eviction** ([[2509.04377|PagedEviction]], [[2603.08743|Zipage]]) so eviction
  composes with paged attention. Score pages with a query-agnostic score (KVzip / Expected Attention) at prefill end.
  Add an optional learned-gate path (DMS / Fast KVzip) for models you can fine-tune.
* For reasoning decode, keep a **sink + recent window** and evict mid-trace tokens by redundancy (R-KV). Always keep
  full KV for a small set of heads (RLKV-style allocation).
* Never apply query-aware eviction to a shared prefix cache.
