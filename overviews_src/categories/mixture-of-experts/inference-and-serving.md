**Verdict.** MoE serving splits into two regimes with different bottlenecks.

**1. Datacenter (expert parallelism, EP).** The problems are all-to-all communication, load imbalance across EP ranks
and decode being memory-bound. The 2025–26 toolkit:
* **Device-initiated EP communication**: DeepEP, [[2512.19849|UCCL-EP]] (portable to AMD/EFA, up to 2.1×),
  [[2603.13606|NCCL EP]]. [[2506.04667|FlashMoE]] fuses dispatch, compute and combine into **one persistent kernel**.
* **Load balancing at serving time**:
  * EPLB-style replication: [[2509.25041|GRACE-MoE]], [[2603.28768|CRAFT]], [[2606.04101|UltraEP]] (94% of the
    ideal balanced throughput at rack scale);
  * [[2601.17111|Least-Loaded EP]] for extreme imbalance: 1.9× on gpt-oss-120b;
  * [[2512.09277|METRO]]: in memory-bound decode, balance **activated experts, not tokens**; up to 4.1× decode
    throughput over EPLB at fixed SLO.
* **Stragglers**: [[2503.05066|capacity-aware token drop/expansion]] gives 1.85× on Mixtral at no quality loss.
* **Batch-aware routing** (cut the *unique* experts per decode batch): [[2511.02237|Opportunistic Expert Activation]]
  (−39% MoE-layer decode latency on Qwen3-30B), [[2602.07616|SERE]], [[2602.07265|XShare]].
* **Attention–FFN disaggregation (AFD)**: [[2504.02263|MegaScale-Infer]] (ping-pong micro-batches, 1.9× per GPU) and
  [[2512.13525|Janus]] (4.7× per GPU). But [[2602.09721|AFD is not universal]]: it pays off on superpod-class
  interconnects with coarser experts.

**2. Memory-constrained (single GPU / edge / CPU+GPU).** The problem is fitting experts, so they are offloaded to CPU
DRAM or SSD and fetched on demand. What matters:
* **Routing locality**: [[2505.16056|not all models suit offloading]]; a cache of ~2× the active experts suffices for
  models with good local consistency.
* **Prediction and prefetch**: [[2609.18063|Edge0]] uses *trained* next-layer routing prediction as the routing itself,
  running a 35B MoE at 20 tok/s in 3 GiB of active memory from SSD. See also [[2602.03921|SpecMD]].
* **Hybrid CPU–GPU execution**: [[2504.05897|HybriMoE]] on kTransformers, 1.33× prefill / 1.70× decode.
* **Pipelined batching**: [[2502.06888|Klotski]], [[2503.09716|MoE-Gen]] (8–31× offline throughput on one GPU).
* **Routers biased toward cached experts**: [[2605.27081|ReMoE]] (+26% reuse), [[2508.18983|SMoE substitution]].
* **Energy**: [[2508.06978|SSD offload is energy-harmful]]; Flash read energy must drop ~10× before SSD tiers are
  efficient.

### Hand ranking

| # | Paper | Regime | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.02263\|MegaScale-Infer]] (ByteDance) | Datacenter | Disaggregate attention and FFN per layer; ping-pong micro-batch pipeline; M2N communication library | Up to 1.90× per-GPU throughput at production scale |
| 2 | [[2506.04667\|FlashMoE]] (NeurIPS'25) | Datacenter kernel | Single persistent kernel fusing dispatch/compute/combine with device-initiated RDMA | Up to 9× GPU utilization, 6× lower latency, 5.7× throughput |
| 3 | [[2512.09277\|METRO: balance activated experts, not tokens]] | Datacenter decode | In the memory-bound regime, minimize the maximum *number of activated experts per GPU* | −11–22% decode latency; up to 4.11× decode throughput vs EPLB at fixed SLO |
| 4 | [[2511.02237\|Opportunistic Expert Activation]] | Decode, training-free | Tokens piggyback on experts already loaded for other tokens in the batch | −39% / −15% MoE decode latency on Qwen3-30B / 235B at batch 16, no accuracy loss |
| 5 | [[2512.13525\|Janus]] | Datacenter | Attention/MoE disaggregation + adaptive two-phase comm + SLO-aware joint scaling of both pools | Up to 4.7× per-GPU throughput under token-level SLOs |
| 6 | [[2504.05897\|HybriMoE]] | CPU+GPU | Dynamic intra-layer CPU–GPU scheduling, impact-driven prefetch, score-based caching | 1.33× prefill, 1.70× decode over kTransformers |
| 7 | [[2609.18063\|Edge0: serving 35B MoEs from SSD]] | Edge/SSD | Trained one-token-ahead routing prediction used *as* routing + int4 + recovery LoRA | 35B MoE at 20 tok/s in 3 GiB active memory on a 24 GB machine |
| 8 | [[2503.05066\|Capacity-aware inference]] | Datacenter | Drop excess tokens from overloaded experts; expand to under-loaded ones | 30% faster at 0.9% loss (OLMoE); 1.85× on Mixtral with a slight quality gain |
| 9 | [[2505.16056\|Local routing consistency]] | Model selection | SRP/SCH metrics for how well a fixed expert set covers a token segment | Which MoEs suit offloading; cache ≈ 2× active experts |
| 10 | [[2512.19849\|UCCL-EP]] | Communication | Portable GPU-initiated EP with a CPU proxy | DeepEP-level on NVIDIA; up to 2.1× on AMD/EFA; +40% SGLang throughput |
| 11 | [[2606.04101\|UltraEP]] | Rack-scale EP | Near-optimal replication and relay fan-out on rack-scale nodes | 94.3% of ideal balanced throughput at 256 GPUs; imbalance 4.0 → 1.04 |
| 12 | [[2602.09721\|Challenges of AFD]] | Analysis | When attention–FFN disaggregation beats EP | Superpod interconnect + coarse experts favour AFD; otherwise EP wins |
| 13 | [[2503.09716\|MoE-Gen]] | Single-GPU offline | Module-based batching (accumulate tokens per module) | 8–31× throughput over FlexGen/MoE-Lightning/DeepSpeed |
| 14 | [[2505.11415\|MoE-CAP]] | Benchmark | Cost–accuracy–performance trade-off; sparse MBU/MFU metrics | Use S-MBU/S-MFU to evaluate MoE systems |

**Also useful.**
* Prefetch/offload: [[2502.05370|FineMoE]], [[2505.05950|FloE]], [[2601.21198|ZipMoE]] (lossless compression on
  device), [[2606.21868|WiSP]] (working-set paging), [[2512.12990|SliceMoE]] (bit-sliced caching),
  [[2606.15453]], [[2608.12103]] (kernel-managed tiering for trillion-parameter MoE), [[2608.14333]] (high-bandwidth flash).
* Placement: [[2502.06643|MoETuner]], [[2503.04398|Semantic Parallelism]], [[2607.00466|ELDR]] (expert-locality-aware
  decode routing in PD-disaggregated vLLM).
* Overlap: [[2607.19539]] (tile-level signalling), [[2511.11505|FarSkip-Collective]].
* Speculative decoding for MoE: see [`decoding/speculative-decoding`](../../decoding/speculative-decoding/README.md)
  (MoESD, MoE-Spec, cost-aware SD).
* MoE dLLM offload: [[2605.20179|TIDE]].
* Edge-native: [[2608.16157|FreeToken]].

**Runtime checklist.**
1. DeepEP/UCCL-style dispatch/combine with FP8 payloads, overlapped with shared-expert and attention compute (two
   micro-batches).
2. EPLB-style redundant experts, rebalanced online. In decode, balance **activated experts per rank**.
3. Grouped-GEMM kernels for fine-grained experts, or a persistent fused MoE kernel.
4. Batch-aware routing option (opportunistic activation) for memory-bound decode.
5. For single-node or edge: expert cache (~2× active) + learned or speculative prefetch + CPU-side expert execution for
   misses (kTransformers-style).
6. Keep super experts and shared experts resident and in high precision.
