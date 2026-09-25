# Serving systems, kernels & hardware (2025–2026)

> Synthesis of [`papers/serving-systems`](../papers/serving-systems/README.md). KV-cache storage and reuse are in
> [KV cache](kv-cache.md); attention kernels are in [attention](attention.md).

## TL;DR: the 2026 reference architecture for an LLM runtime

1. **Prefill/decode disaggregation (PD)** with a shared, tiered KV store (HBM → DRAM → CXL/SSD → remote).
   * CXL makes KV transfer cheap: [[2512.18194|TraCT]] (rack-scale CXL shared memory on Dynamo: −9.8× TTFT,
     1.6× throughput) and [[2511.20172|Beluga]] (SIGMOD'26: −89.6% TTFT, 7.35× vLLM throughput vs. RDMA).
   * PD can span datacenters: [[2604.15039|Prefill-as-a-Service]] moves long-context prefill to compute-dense
     clusters over commodity Ethernet (+54% throughput for a 1T hybrid model).
   * [[2510.18121|Core Attention Disaggregation]] applies the same idea to *training*: long-context attention runs on
     separate servers (512 H200, 512K context, 1.35×).
2. **Wide expert parallelism for MoE** with fine-grained comm/compute overlap: [[2502.19811|Comet]] (1.71×
   end-to-end, production at ByteDance). See [MoE](mixture-of-experts.md).
3. **Agent-aware scheduling.** Agents have tool-call idle windows, long shared prefixes and program structure.
   * [[2602.13692|ThunderAgent]]: program-aware scheduler, 1.5–3.6× serving and 1.8–3.9× RL rollout.
   * [[2606.00866|MORI]]: offload idle agents' KV during tool calls; +20–71% throughput on Claude Code traces.
   * [[2601.22705|CONCUR]]: congestion control for agent admission, up to 4.1×.
   * [[2507.07400|KVFlow]] (in the KV-cache folder).
4. **Specialised engines per workload shape.** [[2505.07203|PrefillOnly]] (SOSP'25) for scoring and embedding;
   determinism when needed via [[2601.17768|LLM-42]] (non-deterministic fast path + verify/rollback).
5. **Auto-configuration**: [[2601.06288|AIConfigurator]] picks TP/EP/PD splits and batch sizes without GPU
   profiling (+40% dense, +50% MoE, 30 s search).

## GPU kernels & compilers ([folder](../papers/serving-systems/gpu-kernels-and-compilers/README.md))

* **Know the hardware:** [[2512.02189|Blackwell microbenchmarks]] (memory subsystem, 5th-gen tensor cores,
  FP4/FP6/FP8 throughput vs. H200). FlashAttention-4 ([[2603.05451]]) shows why Blackwell needs new pipelining.
* **Compilers:** [[2510.14719|Tawa]] (automatic warp specialisation from tile programs; matches hand-written
  FA-3, 1.1× over cuBLAS GEMM).
* **LLM-written kernels are real now:** [[2602.19128|K-Search]] (2.1× average over evolutionary baselines, 14.3× on
  MoE kernels) and [[2603.24517|AVO]] (7-day autonomous search beats FA-4 by up to 10.5%). Benchmark:
  [[2605.04956|KernelBenchX]] (fusion tasks still fail 72% of the time).
* **Lossless compression that speeds things up:** [[2603.17435|ZipServ]] (ASPLOS'26), −30% model size, 2.21×
  kernel-level and 1.22× end-to-end over vLLM.

## Hardware & co-design ([folder](../papers/serving-systems/hardware-accelerators/README.md))

* Model–system co-design from the labs: [[2505.09343|Insights into DeepSeek-V3]] (ISCA'25; MLA, MoE, FP8, MTP and
  network-topology lessons, plus hardware wish-list) and [[2507.19427|Step-3]] (attention–FFN disaggregation;
  4,039 tok/s/GPU at a 50 ms TPOT SLA on Hopper).
* PIM / near-memory:
  * [[2502.07578|CENT]] (ASPLOS'25, GPU-free CXL-PIM, 2.3× throughput, 2.9× less energy).
  * [[2502.15470|PAPI]] (dynamic PIM/compute kernel mapping).
  * [[2507.10178|Pimba]] (MICRO'25, PIM for post-transformer state updates; MX formats are Pareto-optimal).
  * [[2502.16963|Hermes NDP-DIMM]] (Llama-2-70B on a consumer GPU at 13.75 tok/s).
* LUT-based low-bit compute: [[2503.06862|FIGLUT]] (HPCA'25) and [[2511.06174|LUT-LLM]] (FPGA, VQ).
* New memory tiers: [[2607.10186|High-Bandwidth Flash]] (2.5× throughput per GPU) and [[2505.06901|Ecco]]
  (ISCA'25, entropy-coded cache compression).

## Edge & local ([folder](../papers/serving-systems/edge-and-on-device/README.md))

* [[2502.11880|Bitnet.cpp]]: ternary LUT mpGEMM kernels. This is the reference for sub-2-bit CPU inference.
* Home clusters: [[2504.08791|prima.cpp]] (70B across mixed consumer devices, 5–17× lower TPOT than llama.cpp,
  exo and dllama).
* Hybrid local/cloud: [[2502.15964|Minions]] (a local model reads the documents; 5.7× cheaper at 97.9% quality) and
  [[2505.17052|SpecEdge]] (edge GPU drafts, server verifies).
* 1M+ contexts on one consumer GPU: [[2502.12574|HeadInfer]] (head-wise KV offload, 4M tokens with an 8B model).
  [[2608.08097|OasisKV]] adds look-ahead sparse prefetch from host memory.

## Energy ([folder](../papers/serving-systems/energy-and-cost/README.md))

[[2504.17674|Energy considerations of inference optimisations]] (ACL'25) finds that proper optimisation cuts energy
by up to 73%. For measurement tooling, use [[2512.03024|TokenPowerBench]].
