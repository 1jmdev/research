# KV cache: state of the art, 2025–2026

> Synthesis of the top papers in [`papers/kv-cache`](../papers/kv-cache/README.md). Related:
> [attention](attention.md) (architectures that shrink KV by design: MLA, GQA, hybrids, sparse attention) and
> [serving systems](serving-systems.md) (disaggregation, KV transfer).

## TL;DR for a runtime builder

The KV cache is now the dominant memory cost in long-context and agentic serving. Build the runtime around five
composable mechanisms, roughly in order of payoff:

1. **Paged, prefix-shared KV with a radix/tree index** (vLLM/SGLang style), plus a **host/SSD tier**.
   * [[2510.09665|LMCache]] is the reference open-source KV layer: it moves KV out of GPU memory and shares it
     across engines and queries, for up to 15× throughput with vLLM on multi-round QA and document workloads.
   * [[2508.18572|Strata]]: hierarchical context caching in SGLang, up to 5× lower TTFT than vLLM + LMCache.
   * [[2602.21548|DualPath]] (agentic inference): storage bandwidth becomes the bottleneck. Load KV into *decode*
     engines and forward it to prefill, for about 1.9× serving throughput.
   * Production trace study: [[2506.02634|KV$ in the wild]] (ATC'25) gives the workload-aware eviction policy.
2. **KV quantization to 2–4 bits.** It stacks with everything else.
   * Keys need more precision than values ([[2502.15075|More for keys, less for values]]): e.g. 4-bit K + 2-bit V.
   * Rotation + normalization is the 2026 recipe. [[2606.03458|KVarN]] (Hadamard + dual-axis variance
     normalization) is calibration-free and holds state of the art at **2-bit on MATH500/AIME/HumanEval**.
     [[2605.17757|OSCAR]] (offline covariance-aware rotation) keeps Qwen3-8B within 1.4 points at INT2, versus
     collapse for naive rotation, with ~8× memory savings and up to 7× throughput at large batch.
   * [[2506.18879|CommVQ]]: RoPE-commutative additive VQ gets **1-bit KV** with minimal loss (Llama-3.1-8B, 128K
     context, one RTX 4090).
   * [[2502.04420|KVTuner]]: layer-wise mixed KV precision (≈3.25 bits average), searched offline.
   * [[2508.10395|XQuant]] caches quantized *layer inputs X* and rematerializes K/V: 2× from the start, up to 7.7× with
     under 0.1 PPL loss. It trades FLOPs for bandwidth, which is the right trade on new GPUs.
3. **Token eviction / selection**, ideally *query-agnostic* so the compressed cache can be reused.
   * [[2505.23416|KVzip]] (NeurIPS'25): query-agnostic eviction scored by context reconstruction. 3–4× smaller
     cache and ~2× faster FlashAttention decode, and it is reusable across queries.
   * [[2506.05345|DMS: Inference-time hyper-scaling]] (NVIDIA): *learned* eviction with only ~1K training steps gives
     8× compression. The saved memory buys more reasoning tokens (+12 AIME24 points on Qwen-R1-32B at equal compute).
   * [[2503.12491|CAKE]] (layer-adaptive budgets, 3.2% of the cache on LongBench), [[2502.00299|ChunkKV]] (evict
     semantic chunks, reuse indices across layers), [[2510.00636|Expected Attention]] (score keys against the
     predicted *future* query distribution) and [[2502.01068|FastKV]] (decouple prefill token pruning from the KV
     budget: 1.8× prefill, 2.9× decode).
   * Reasoning traces: [[2505.13866|RPC]] compresses the model's *own* generated reasoning path (QwQ-32B 1.6×
     throughput for −1.2 AIME points).
4. **Low-rank / latent KV** for models you can fine-tune briefly.
   * Post-training projections: [[2505.24357|ReCalKV]], [[2512.05916|KQ-SVD]] (closed-form optimal low-rank of the
     attention matrix), [[2603.04427|Thin keys]] (factored keys: 75% key-cache savings after a small QK fine-tune).
   * Cross-layer SVD: [[2503.18893|xKV]] (up to 8× compression, 4.2× end-to-end with selective recompute).
   * If you can convert the architecture, see MHA→MLA conversion in [model conversion](model-conversion.md).
5. **Reuse KV across prompts, documents and agents** (non-prefix reuse).
   * RAG chunks: [[2502.16002|KVLink]] (re-position cached docs plus trainable link tokens, −96% TTFT),
     [[2502.15734|Cache-Craft]] (SIGMOD'25, production RAG), [[2510.10129|CacheClip]] (small aux model picks tokens to
     recompute).
   * Multi-agent: [[2510.12872|KVCOMM]] (70%+ reuse, 7.8× speedup), [[2507.07400|KVFlow]] (agent-step-graph-aware
     prefix eviction, 1.8–2.2× over SGLang), [[2602.01053|LRAgent]] (shared KV for multi-LoRA agents).
   * Across models: [[2510.03215|Cache-to-Cache]] (ICLR'26) fuses one LLM's KV into another, 2.5× faster than text
     communication. [[2608.03893]] maps KV between family members with a closed-form linear map.

## Storage-format tricks worth stealing

* **Transform coding** of KV (PCA + adaptive quant + entropy coding): [[2511.01815|KVTC]] (ICLR'26) reports **20×**
  compression, and 40× for some workloads, for off-GPU storage and transfer.
* **Bit-plane keys** for sparse decode over offloaded caches: [[2609.17652|Fathom]] reads only as many bit-planes per
  channel as each query needs.
* **Near-storage attention**: [[2502.09921|HILOS]] (ASPLOS'26), 7.9× offline throughput.
* Vector-DB-style KV retrieval for 1M-token contexts: [[2505.02922|RetroInfer]] (VLDB'26), 4.4× over full attention
  at 120K.

## Streaming / multimodal

Video LLMs are the heaviest KV users. Query-agnostic memory caps are standard there:
[[2506.15745|InfiniPot-V]] (−94% peak memory), [[2508.15717|StreamMem]], [[2601.14724|HERMES]] (10× faster TTFT) and
[[2503.00540|ReKV]]. For VideoLLMs, quantize values **per-channel** ([[2503.16257|VidKV]], 1.x-bit KV).

## Security note

Prefix / KV sharing leaks timing information across tenants. See [[2508.08438|SafeKV]] for selective isolation
inside a radix-tree cache manager.

## Open problems

* Eviction that survives **multi-turn reuse** (most methods are query-aware and one-shot; KVzip is an exception).
* Joint optimisation of eviction × quantization × low-rank under one budget. See
  [[2609.24298|KV-COBRA]] for per-head bit/rank co-allocation.
* KV for **hybrid models** (linear/SSM state + a few attention layers). Cross-model state transfer is just starting
  ([[2609.25053|LatentPort]]).
