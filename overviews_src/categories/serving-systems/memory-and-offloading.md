**Verdict.** GPU memory management has moved past PagedAttention into three directions.
(KV-specific offloading lives in [`kv-cache/offloading-and-hierarchical-storage`](../../kv-cache/offloading-and-hierarchical-storage/README.md).)

1. **Heterogeneous caches in one pool.** Modern models mix full attention, sliding window, SSM state, cross-attention
   and vision embeddings, each with different page sizes and lifetimes. [[2503.18292|Jenga]] (SOSP'25) uses a two-level
   allocator (LCM page size) + per-layer-type caching; up to 79.6% better memory utilization and 4.92× (1.8× average)
   throughput in vLLM. It is the basis of vLLM's hybrid KV manager. For hybrid Mamba models, see asymmetric paging
   ([[2605.22416]]).
2. **Elastic memory across models and phases.**
   * [[2505.04021|Prism]] (OSDI'26): *GPU memory ballooning* (kvcached) reclaims KV memory across co-served models;
     production on 10K+ GPUs.
   * [[2506.15155|eLLM]]: unifies activation and KV memory.
   * [[2608.23658]]: elastic KV reserve, and why chunked prefill already closes most of the gap.
   * [[2608.13263|vToken]]: token-level reclaimable KV.
3. **New memory tiers.**
   * CPU/host offload with SLO guarantees: [[2502.08182|Select-N]], [[2506.03296|APEX]].
   * Head-wise offload for extreme context: [[2502.12574|HeadInfer]], 4M tokens with an 8B model on a 24 GB GPU.
   * Peer-GPU caching: [[2602.00328|Harvest]].
   * CXL: [[2609.10790]] (Kubernetes-native composable CXL).
   * **High-Bandwidth Flash (HBF)**: [[2607.10186|FlashAccel]] (+2.49× throughput per GPU with six HBF stacks). But
     [[2608.11668|HBF Sucks?]] shows a drop-in HBF KV tier can make the *faster* device give a slower system (write
     wear, thermal limits). Use it for weights and reused KV, not transient KV.
   * Agentic idle windows: [[2606.00866|MORI]] offloads KV of agents waiting on tools, measured on Claude Code traces:
     +20–71% throughput.

Training-side counterpart: [[2604.05091|MegaTrain]] trains 120B-parameter models at full precision on **one H200 + 1.5 TB
host memory** via streamed stateless layer templates; 1.84× ZeRO-3 offload.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2503.18292\|Jenga]] (SOSP'25) | Heterogeneous KV/embedding memory | LCM-page two-level allocator + layer-type-specific caching and eviction | Up to +79.6% memory utilization, 4.92× (1.8× average) throughput in vLLM |
| 2 | [[2505.04021\|Prism]] (OSDI'26) | Multi-model | GPU memory ballooning (kvcached) for dynamic cross-model sharing | Production across 10K+ GPUs; open-source driver |
| 3 | [[2502.12574\|HeadInfer]] | Offload | Head-wise KV offload to CPU with roofline-guided overlap | Llama-3-8B at 1M tokens: 207 → 17 GB GPU memory; 4M tokens on an RTX 4090 |
| 4 | [[2606.00866\|MORI]] | Agentic offload | Offload by relative idleness during tool calls, with a movable GPU/CPU partition | +20–71% throughput, −18–43% TTFT on Claude Code traces |
| 5 | [[2608.11668\|HBF Sucks?]] | Characterization | Full-stack HBF study for KV-centric serving (H100/B200) | Drop-in HBF KV tier: 2–5.5× worse latency; selective placement makes it worthwhile |
| 6 | [[2607.10186\|FlashAccel]] | HBF architecture | HBF stacks inside HBM GPUs with layouts for weights and KV + an HBF-aware storage layer | 2.49× throughput per GPU, 1.93× energy efficiency at 100 ms latency |
| 7 | [[2604.05091\|MegaTrain]] | Training offload | Stream weights/gradients per layer from host with stateless layer templates | 120B training on a single H200; 1.84× ZeRO-3 offload at 14B |
| 8 | [[2504.06319\|Asynchronous KV prefetching]] | Kernel | Prefetch KV into L2 during compute, hiding HBM latency | 2.15× attention-kernel efficiency; 1.97× end to end vs FA3 (H20) |
| 9 | [[2502.16963\|NDP-DIMM augmentation]] | Consumer GPU | Hot neurons on GPU, cold neurons on near-data-processing DIMMs | Affordable large-model inference on one consumer GPU |
| 10 | [[2605.22416\|Asymmetric paging for hybrid Mamba-Transformers]] | Hybrid models | Different virtual-memory policies for KV (grows) vs SSM state (fixed) | Correct and efficient paging for Jamba-style models |

**Also useful.**
* Offloaded inference on consumer devices: [[2504.03664|PIPO]], [[2605.02189|PipeMax]], [[2504.12526|MOM]].
* Host–GPU bandwidth: [[2512.16056|MultiPath memory access]], [[2609.13592|BOOST]], [[2604.26074|DAK]].
* Supernodes: [[2602.00748|HyperOffload]].
* LPDDR accelerators: [[2512.09427|ODMA]].
* Lossless transfer compression: [[2605.30728|Invariant Bit Packing]], [[2505.06901|Ecco]].
* RAG on one GPU: [[2504.15302|RAGDoll]].

**Runtime checklist.**
* A **hybrid KV manager** (Jenga-style) for mixed layer types.
* Ballooning/elastic KV for multi-model co-serving.
* Tiered placement (HBM → host DRAM/CXL → HBF/SSD) with reuse-aware and write-budgeted policies.
* Idle-window offload for agent sessions.
* L2 prefetch in attention kernels.
