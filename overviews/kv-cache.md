# KV cache: state of the art, 2025–2026

> Synthesis of the top papers in [`papers/kv-cache`](../papers/kv-cache/README.md). Related:
> [attention](attention.md) (architectures that shrink KV by design: MLA, GQA, hybrids, sparse attention) and
> [serving systems](serving-systems.md) (disaggregation, KV transfer).

## TL;DR for a runtime builder

The KV cache is now the dominant memory cost in long-context and agentic serving. Build the runtime around five
composable mechanisms, roughly in order of payoff:

1. **Paged, prefix-shared KV with a radix/tree index** (vLLM/SGLang style), plus a **host/SSD tier**.
   * [LMCache](../papers/kv-cache/offloading-and-hierarchical-storage/2510.09665-lmcache-an-efficient-kv-cache-layer-for-enterprise-scale-llm-inference.md) is the reference open-source KV layer: it moves KV out of GPU memory and shares it
     across engines and queries, for up to 15× throughput with vLLM on multi-round QA and document workloads.
   * [Strata](../papers/kv-cache/prefix-caching-and-reuse/2508.18572-strata-hierarchical-context-caching-for-long-context-language-model-se.md): hierarchical context caching in SGLang, up to 5× lower TTFT than vLLM + LMCache.
   * [DualPath](../papers/kv-cache/offloading-and-hierarchical-storage/2602.21548-dualpath-breaking-the-storage-bandwidth-bottleneck-in-agentic-llm-infe.md) (agentic inference): storage bandwidth becomes the bottleneck. Load KV into *decode*
     engines and forward it to prefill, for about 1.9× serving throughput.
   * Production trace study: [KV$ in the wild](../papers/kv-cache/prefix-caching-and-reuse/2506.02634-kvcache-cache-in-the-wild-characterizing-and-optimizing-kvcache-cache.md) (ATC'25) gives the workload-aware eviction policy.
2. **KV quantization to 2–4 bits.** It stacks with everything else.
   * Keys need more precision than values ([More for keys, less for values](../papers/kv-cache/quantization/2502.15075-quantize-what-counts-more-for-keys-less-for-values.md)): e.g. 4-bit K + 2-bit V.
   * Rotation + normalization is the 2026 recipe. [KVarN](../papers/kv-cache/quantization/2606.03458-kvarn-variance-normalized-kv-cache-quantization-mitigates-error-accumu.md) (Hadamard + dual-axis variance
     normalization) is calibration-free and holds state of the art at **2-bit on MATH500/AIME/HumanEval**.
     [OSCAR](../papers/kv-cache/quantization/2605.17757-oscar-offline-spectral-covariance-aware-rotation-for-2-bit-kv-cache-qu.md) (offline covariance-aware rotation) keeps Qwen3-8B within 1.4 points at INT2, versus
     collapse for naive rotation, with ~8× memory savings and up to 7× throughput at large batch.
   * [CommVQ](../papers/kv-cache/quantization/2506.18879-commvq-commutative-vector-quantization-for-kv-cache-compression.md): RoPE-commutative additive VQ gets **1-bit KV** with minimal loss (Llama-3.1-8B, 128K
     context, one RTX 4090).
   * [KVTuner](../papers/kv-cache/quantization/2502.04420-kvtuner-sensitivity-aware-layer-wise-mixed-precision-kv-cache-quantiza.md): layer-wise mixed KV precision (≈3.25 bits average), searched offline.
   * [XQuant](../papers/kv-cache/quantization/2508.10395-xquant-breaking-the-memory-wall-for-llm-inference-with-kv-cache-remate.md) caches quantized *layer inputs X* and rematerializes K/V: 2× from the start, up to 7.7× with
     under 0.1 PPL loss. It trades FLOPs for bandwidth, which is the right trade on new GPUs.
3. **Token eviction / selection**, ideally *query-agnostic* so the compressed cache can be reused.
   * [KVzip](../papers/kv-cache/eviction-and-token-selection/2505.23416-kvzip-query-agnostic-kv-cache-compression-with-context-reconstruction.md) (NeurIPS'25): query-agnostic eviction scored by context reconstruction. 3–4× smaller
     cache and ~2× faster FlashAttention decode, and it is reusable across queries.
   * [DMS: Inference-time hyper-scaling](../papers/kv-cache/eviction-and-token-selection/2506.05345-inference-time-hyper-scaling-with-kv-cache-compression.md) (NVIDIA): *learned* eviction with only ~1K training steps gives
     8× compression. The saved memory buys more reasoning tokens (+12 AIME24 points on Qwen-R1-32B at equal compute).
   * [CAKE](../papers/kv-cache/eviction-and-token-selection/2503.12491-cake-cascading-and-adaptive-kv-cache-eviction-with-layer-preferences.md) (layer-adaptive budgets, 3.2% of the cache on LongBench), [ChunkKV](../papers/kv-cache/eviction-and-token-selection/2502.00299-chunkkv-semantic-preserving-kv-cache-compression-for-efficient-long-co.md) (evict
     semantic chunks, reuse indices across layers), [Expected Attention](../papers/kv-cache/eviction-and-token-selection/2510.00636-expected-attention-kv-cache-compression-by-estimating-attention-from-f.md) (score keys against the
     predicted *future* query distribution) and [FastKV](../papers/kv-cache/eviction-and-token-selection/2502.01068-fastkv-decoupling-of-context-reduction-and-kv-cache-compression-for-pr.md) (decouple prefill token pruning from the KV
     budget: 1.8× prefill, 2.9× decode).
   * Reasoning traces: [RPC](../papers/kv-cache/eviction-and-token-selection/2505.13866-reasoning-path-compression-compressing-generation-trajectories-for-eff.md) compresses the model's *own* generated reasoning path (QwQ-32B 1.6×
     throughput for −1.2 AIME points).
4. **Low-rank / latent KV** for models you can fine-tune briefly.
   * Post-training projections: [ReCalKV](../papers/kv-cache/low-rank-and-latent/2505.24357-recalkv-low-rank-kv-cache-compression-via-head-reordering-and-offline.md), [KQ-SVD](../papers/kv-cache/low-rank-and-latent/2512.05916-kq-svd-compressing-the-kv-cache-with-provable-guarantees-on-attention.md) (closed-form optimal low-rank of the
     attention matrix), [Thin keys](../papers/kv-cache/low-rank-and-latent/2603.04427-thin-keys-full-values-reducing-kv-cache-via-low-dimensional-attention.md) (factored keys: 75% key-cache savings after a small QK fine-tune).
   * Cross-layer SVD: [xKV](../papers/kv-cache/cross-layer-sharing/2503.18893-xkv-cross-layer-kv-cache-compression-via-aligned-singular-vector-extra.md) (up to 8× compression, 4.2× end-to-end with selective recompute).
   * If you can convert the architecture, see MHA→MLA conversion in [model conversion](model-conversion.md).
5. **Reuse KV across prompts, documents and agents** (non-prefix reuse).
   * RAG chunks: [KVLink](../papers/kv-cache/prefix-caching-and-reuse/2502.16002-kvlink-accelerating-large-language-models-via-efficient-kv-cache-reuse.md) (re-position cached docs plus trainable link tokens, −96% TTFT),
     [Cache-Craft](../papers/kv-cache/prefix-caching-and-reuse/2502.15734-cache-craft-managing-chunk-caches-for-efficient-retrieval-augmented-ge.md) (SIGMOD'25, production RAG), [CacheClip](../papers/kv-cache/prefix-caching-and-reuse/2510.10129-cacheclip-accelerating-rag-with-effective-kv-cache-reuse.md) (small aux model picks tokens to
     recompute).
   * Multi-agent: [KVCOMM](../papers/kv-cache/prefix-caching-and-reuse/2510.12872-kvcomm-online-cross-context-kv-cache-communication-for-efficient-llm-b.md) (70%+ reuse, 7.8× speedup), [KVFlow](../papers/kv-cache/prefix-caching-and-reuse/2507.07400-kvflow-efficient-prefix-caching-for-accelerating-llm-based-multi-agent.md) (agent-step-graph-aware
     prefix eviction, 1.8–2.2× over SGLang), [LRAgent](../papers/kv-cache/cross-layer-sharing/2602.01053-lragent-efficient-kv-cache-sharing-for-multi-lora-llm-agents.md) (shared KV for multi-LoRA agents).
   * Across models: [Cache-to-Cache](../papers/kv-cache/_general/2510.03215-cache-to-cache-direct-semantic-communication-between-large-language-mo.md) (ICLR'26) fuses one LLM's KV into another, 2.5× faster than text
     communication. [Cross-Model KV Cache Transfer in LLM Families](../papers/kv-cache/offloading-and-hierarchical-storage/2608.03893-cross-model-kv-cache-transfer-in-llm-families-a-closed-form-linear-map.md) maps KV between family members with a closed-form linear map.

## Storage-format tricks worth stealing

* **Transform coding** of KV (PCA + adaptive quant + entropy coding): [KVTC](../papers/kv-cache/offloading-and-hierarchical-storage/2511.01815-kv-cache-transform-coding-for-compact-storage-in-llm-inference.md) (ICLR'26) reports **20×**
  compression, and 40× for some workloads, for off-GPU storage and transfer.
* **Bit-plane keys** for sparse decode over offloaded caches: [Fathom](../papers/kv-cache/offloading-and-hierarchical-storage/2609.17652-fathom-per-query-read-depth-for-sparse-decoding-over-offloaded-kv-cach.md) reads only as many bit-planes per
  channel as each query needs.
* **Near-storage attention**: [HILOS](../papers/kv-cache/offloading-and-hierarchical-storage/2502.09921-a-cost-effective-near-storage-processing-solution-for-offline-inferenc.md) (ASPLOS'26), 7.9× offline throughput.
* Vector-DB-style KV retrieval for 1M-token contexts: [RetroInfer](../papers/kv-cache/offloading-and-hierarchical-storage/2505.02922-retroinfer-a-vector-storage-engine-for-scalable-long-context-llm-infer.md) (VLDB'26), 4.4× over full attention
  at 120K.

## Streaming / multimodal

Video LLMs are the heaviest KV users. Query-agnostic memory caps are standard there:
[InfiniPot-V](../papers/kv-cache/eviction-and-token-selection/2506.15745-infinipot-v-memory-constrained-kv-cache-compression-for-streaming-vide.md) (−94% peak memory), [StreamMem](../papers/kv-cache/_general/2508.15717-streammem-query-agnostic-kv-cache-memory-for-streaming-video-understan.md), [HERMES](../papers/kv-cache/_general/2601.14724-hermes-kv-cache-as-hierarchical-memory-for-efficient-streaming-video-u.md) (10× faster TTFT) and
[ReKV](../papers/kv-cache/_general/2503.00540-streaming-video-question-answering-with-in-context-video-kv-cache-retr.md). For VideoLLMs, quantize values **per-channel** ([VidKV](../papers/kv-cache/quantization/2503.16257-plug-and-play-1-x-bit-kv-cache-quantization-for-video-large-language-m.md), 1.x-bit KV).

## Security note

Prefix / KV sharing leaks timing information across tenants. See [SafeKV](../papers/kv-cache/cross-layer-sharing/2508.08438-selective-kv-cache-sharing-to-mitigate-timing-side-channels-in-llm-inf.md) for selective isolation
inside a radix-tree cache manager.

## Open problems

* Eviction that survives **multi-turn reuse** (most methods are query-aware and one-shot; KVzip is an exception).
* Joint optimisation of eviction × quantization × low-rank under one budget. See
  [KV-COBRA](../papers/kv-cache/eviction-and-token-selection/2609.24298-kv-cobra-kv-cache-compression-via-co-optimized-bit-rank-allocation.md) for per-head bit/rank co-allocation.
* KV for **hybrid models** (linear/SSM state + a few attention layers). Cross-model state transfer is just starting
  ([LatentPort](../papers/kv-cache/offloading-and-hierarchical-storage/2609.25053-latentport-beyond-kv-cache-cross-model-transfer-of-recurrent-memory-in.md)).
