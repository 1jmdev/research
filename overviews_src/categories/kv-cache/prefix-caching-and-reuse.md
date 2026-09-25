**Verdict.** Exact prefix caching (radix tree / APC) is table stakes. The research has moved to three problems.

1. **Non-prefix reuse** (position-independent caching, PIC). RAG chunks and agent tool schemas recur at *different
   positions*. Solutions:
   * selective recomputation of the tokens whose cross-chunk attention matters (CacheBlend lineage: Cache-Craft,
     ProphetKV, CacheClip, MEPIC);
   * light fine-tuning so the model tolerates independently encoded chunks (KVLink, KV Packet).
2. **Agent and multi-agent workloads.** Workflow-aware eviction (KVFlow), cross-agent KV communication (KVCOMM), and
   multi-LoRA cache sharing. Real-provider data ([[2506.02634|KVCache in the Wild]], [[2601.06007|Don't Break the Cache]])
   shows reuse is skewed and predictable per request category.
3. **Hybrid models.** Linear-attention and SSM layers have *per-request recurrent state*, not per-token KV, which
   breaks prefix caching. Fixes: [[2607.01299|HYPIC]], [[2608.30310|Tail-Replay]], [[2605.05219|Sparse Prefix
   Caching]], [[2608.11231|LinearKV]].

### Hand ranking

| # | Paper | Problem | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2506.02634\|KVCache Cache in the Wild]] (Alibaba, ATC'25) | Workload | First production characterization of KV$ reuse: skewed, single-turn reuse as important as multi-turn, predictable per category | Workload-aware eviction policy; **read before designing a cache** |
| 2 | [[2502.16002\|KVLink]] (NeurIPS'25) | PIC via fine-tune | Precompute per-document KV; fix positions at inference; add trainable link tokens so documents can attend across each other | Accurate reuse of independently encoded documents; ~27× fewer GPU-hours per million requests in their RAG setting |
| 3 | [[2510.12872\|KVCOMM]] (NeurIPS'25) | Multi-agent | Estimate each agent's **KV offset** for shared context from anchor examples; reuse instead of re-prefill | Training-free; large prefill savings for agent chains |
| 4 | [[2507.07400\|KVFlow]] (NeurIPS'25) | Agent workflows | **Agent Step Graph** → steps-to-execution-based eviction + proactive prefetch (replaces LRU) | Fewer misses in multi-agent workflows |
| 5 | [[2604.13226\|KV Packet]] | PIC without recompute | Cached documents are immutable "packets" wrapped with **trainable soft-token adapters** (self-distilled) | Near-zero recompute FLOPs, near full-recompute accuracy |
| 6 | [[2502.15734\|Cache-Craft]] (SIGMOD'25) | RAG chunk cache | Chunk-cache manager: which chunks to store, which tokens to recompute | Practical RAG KV reuse |
| 7 | [[2508.18572\|Strata]] | Hierarchical caching | GPU-assisted I/O to beat paged-layout fragmentation + **cache-loading-aware scheduling** | Long-context prefix caching that stays compute-bound |
| 8 | [[2512.16822\|MEPIC]] | PIC memory | Page-aligned, position-canonical chunk KV so different requests *share pages* | Memory-efficient PIC |
| 9 | [[2606.17107\|Models Take Notes at Prefill]] | Editable KV | Downstream KV holds memoized conclusions; **editing one field's KV is not enough** (<1% of the decision). Shows editable and composable KV | Correctness warning for "KV patching" systems |
| 10 | [[2602.22603\|SideQuest]] | Agentic reasoning | The LRM itself decides which context tokens to drop, as a parallel auxiliary task | Model-driven compaction for deep research |
| 11 | [[2607.01299\|HYPIC]] | Hybrid PIC | Cache the segment-cumulative transition operator for linear-attention layers | First PIC for hybrid models |
| 12 | [[2603.04428\|Persistent Q4 KV]] | Edge agents | Persist per-agent **4-bit** KV to disk and reload into attention | 76–136× TTFT reduction on Apple M4 |

**Security.**
* [[2603.10726|PrefixWall]] and [[2605.23640|CachePrune]]: prefix-cache timing side channels.
* [[2609.04748]]: quantization amplifies *cache-induced output divergence*. The same request can give different answers
  with and without a cache hit. Test determinism.

**Runtime recommendations.**
* **Radix-tree prefix cache.** Key blocks by (token ids, model, LoRA, multimodal hashes), with cache-aware routing
  ([[2602.06502|DualMap]], [[2608.19677|CacheRoute]]).
* **Chunk-level PIC mode.** An opt-in mode for RAG and agents: store per-chunk KV with position-free keys (pre-RoPE or
  re-rotated), then recompute ~10–15% of tokens (CacheBlend-style) or use models fine-tuned for concatenation.
* **Hybrid models.** Checkpoint the recurrent state every N tokens so that prefix hits can resume from the nearest
  checkpoint (Tail-Replay / Sparse Prefix Caching).
