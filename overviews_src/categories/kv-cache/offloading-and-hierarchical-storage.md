**Verdict.** In 2025–26 the KV cache became a **first-class storage object**. It is persisted across requests, engines
and nodes in a GPU HBM → CPU DRAM → CXL → NVMe/SSD → remote-storage hierarchy.

Agentic, multi-turn workloads make **KV load I/O**, not compute, the bottleneck (DualPath). The production reference is
**LMCache** (vLLM/SGLang connector). The research frontier:
* **Compress what you store.** KVTC reaches 20× with transform coding (PCA + adaptive quantization + entropy coding).
* **Sparse decoding directly over offloaded KV.** RetroInfer, OasisKV, Fathom and HGCA read only the top-k tokens from
  host memory.
* **CXL shared memory** as a rack-scale KV pool (Beluga, TraCT).

### Hand ranking

| # | Paper | Layer | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.09665\|LMCache]] (151 cites) | Production KV layer | Extract, store and share KV out of vLLM/SGLang; offload **and** PD transfer; chunked, pipelined I/O; control API | The de-facto open KV cache layer; large TTFT and throughput wins with prefix reuse |
| 2 | [[2602.21548\|DualPath]] (DeepSeek) | Disaggregated storage I/O | Load KV **storage→decode then RDMA→prefill** as well as storage→prefill, balancing storage-NIC load | 1.87× offline throughput, 1.96× agent runs/s |
| 3 | [[2511.01815\|KVTC]] (NVIDIA, ICLR'26) | Compression for storage | **PCA decorrelation + adaptive quantization + entropy coding** of KV (media-codec style) | Up to 20× (40× in some cases) with reasoning accuracy preserved |
| 4 | [[2505.02922\|RetroInfer]] (VLDB'26) | Sparse over offload | KV cache as a **vector-storage engine**: wave index (attention-aware) + wave buffer across GPU/CPU | Long-context throughput at full-attention accuracy |
| 5 | [[2608.08097\|OasisKV]] | Sparse prefetch | Use **speculative-decoding drafts to predict future important tokens** and prefetch them to HBM | Decode KV beyond HBM with little stall |
| 6 | [[2511.20172\|Beluga]] (SIGMOD'26) | CXL | GPUs and CPUs share a large CXL memory pool for KV | Replaces RDMA pools; lower latency |
| 7 | [[2512.18194\|TraCT]] | CXL rack-scale | CXL shared memory as both the **PD transfer substrate** and a rack-wide prefix cache | Removes the NIC hop from disaggregated serving |
| 8 | [[2609.17652\|Fathom]] | Sparse scan | 4-bit K stored as **bit planes**; each query reads only as many planes per channel as it needs (reverse water-filling) | 1.67× faster decode at 1M tokens vs Double Sparsity/Loki/SparQ |
| 9 | [[2507.03153\|HGCA]] | CPU–GPU attention | Dense attention on recent GPU KV + sparse attention on CPU KV, merged by log-sum-exp | Near-full attention quality with offload |
| 10 | [[2509.00105\|AdaptCache]] (SOSP'25) | Tiered + lossy | Chooses compression algorithm, rate and device per KV entry to maximize DRAM hits | Lower delay, higher quality than SSD-heavy caches |
| 11 | [[2605.18071\|KVDrive]] | GPU/DRAM/SSD | Joint placement, pipeline scheduling and cross-tier coordination | Sustained long-context throughput on small GPUs |
| 12 | [[2608.03893\|Cross-model KV transfer]] | Reuse | Closed-form per-head ridge mapper between model sizes in a family | Skip prefill when switching models mid-conversation |

Also worth reading:
* [[2512.10576|ESS]]: offload-centric latent-cache management for DeepSeek-V3.2 sparse attention.
* [[2606.19746|SAC]]: CXL KV for sparse-attention models.
* [[2605.03375|Tutti]]: practical SSD KV.
* [[2601.19910]]: PCIe bottleneck analysis for KV offloading.

**Runtime architecture checklist.**
1. **Allocator.** Paged KV with a stable, hashable block identity (token-content hash plus model/LoRA/adapter id). This
   enables prefix caching, offload, cross-instance sharing and PD transfer with one mechanism.
2. **Async tiering.** An asynchronous tier manager (HBM ↔ pinned DRAM ↔ NVMe via GDS/io_uring), with **layer-wise
   pipelined loading** so layer *i* computes while layer *i+1* loads.
3. **Storage codec.** Optionally compress KV blocks when writing to lower tiers: KVTC-style or FP8/INT4. Decompress on
   the GPU.
4. **Sparse decode.** For million-token agents, keep the full KV on CPU/CXL and a small top-k working set on the GPU
   (OasisKV, RetroInfer). This is the natural pairing with sparse-attention models such as DeepSeek-V3.2 DSA and NSA.
