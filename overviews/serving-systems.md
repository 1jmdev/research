# Serving systems, kernels & hardware (2025–2026)

> Synthesis of [`papers/serving-systems`](../papers/serving-systems/README.md). KV-cache storage and reuse are in
> [KV cache](kv-cache.md); attention kernels are in [attention](attention.md).

## TL;DR: the 2026 reference architecture for an LLM runtime

1. **Prefill/decode disaggregation (PD)** with a shared, tiered KV store (HBM → DRAM → CXL/SSD → remote).
   * CXL makes KV transfer cheap: [TraCT](../papers/kv-cache/offloading-and-hierarchical-storage/2512.18194-tract-disaggregated-llm-serving-with-cxl-shared-memory-kv-cache-at-rac.md) (rack-scale CXL shared memory on Dynamo: −9.8× TTFT,
     1.6× throughput) and [Beluga](../papers/kv-cache/offloading-and-hierarchical-storage/2511.20172-beluga-a-cxl-based-memory-architecture-for-scalable-and-efficient-llm.md) (SIGMOD'26: −89.6% TTFT, 7.35× vLLM throughput vs. RDMA).
   * PD can span datacenters: [Prefill-as-a-Service](../papers/serving-systems/disaggregated-prefill-decode/2604.15039-prefill-as-a-service-kvcache-of-next-generation-models-could-go-cross.md) moves long-context prefill to compute-dense
     clusters over commodity Ethernet (+54% throughput for a 1T hybrid model).
   * [Core Attention Disaggregation](../papers/serving-systems/distributed-inference-and-parallelism/2510.18121-efficient-long-context-language-model-training-by-core-attention-disag.md) applies the same idea to *training*: long-context attention runs on
     separate servers (512 H200, 512K context, 1.35×).
2. **Wide expert parallelism for MoE** with fine-grained comm/compute overlap: [Comet](../papers/serving-systems/distributed-inference-and-parallelism/2502.19811-comet-fine-grained-computation-communication-overlapping-for-mixture-o.md) (1.71×
   end-to-end, production at ByteDance). See [MoE](mixture-of-experts.md).
3. **Agent-aware scheduling.** Agents have tool-call idle windows, long shared prefixes and program structure.
   * [ThunderAgent](../papers/serving-systems/scheduling-and-batching/2602.13692-thunderagent-a-simple-fast-and-program-aware-agentic-inference-system.md): program-aware scheduler, 1.5–3.6× serving and 1.8–3.9× RL rollout.
   * [MORI](../papers/serving-systems/memory-and-offloading/2606.00866-idleness-is-relative-exploiting-tool-call-idle-windows-for-offloading.md): offload idle agents' KV during tool calls; +20–71% throughput on Claude Code traces.
   * [CONCUR](../papers/serving-systems/scheduling-and-batching/2601.22705-concur-high-throughput-agentic-batch-inference-of-llm-via-congestion-b.md): congestion control for agent admission, up to 4.1×.
   * [KVFlow](../papers/kv-cache/prefix-caching-and-reuse/2507.07400-kvflow-efficient-prefix-caching-for-accelerating-llm-based-multi-agent.md) (in the KV-cache folder).
4. **Specialised engines per workload shape.** [PrefillOnly](../papers/serving-systems/scheduling-and-batching/2505.07203-prefillonly-an-inference-engine-for-prefill-only-workloads-in-large-la.md) (SOSP'25) for scoring and embedding;
   determinism when needed via [LLM-42](../papers/serving-systems/_general/2601.17768-llm-42-enabling-determinism-in-llm-inference-with-verified-speculation.md) (non-deterministic fast path + verify/rollback).
5. **Auto-configuration**: [AIConfigurator](../papers/serving-systems/_general/2601.06288-aiconfigurator-lightning-fast-configuration-optimization-for-multi-fra.md) picks TP/EP/PD splits and batch sizes without GPU
   profiling (+40% dense, +50% MoE, 30 s search).

## GPU kernels & compilers ([folder](../papers/serving-systems/gpu-kernels-and-compilers/README.md))

* **Know the hardware:** [Blackwell microbenchmarks](../papers/serving-systems/hardware-accelerators/2512.02189-microbenchmarking-nvidia-s-blackwell-architecture-an-in-depth-architec.md) (memory subsystem, 5th-gen tensor cores,
  FP4/FP6/FP8 throughput vs. H200). FlashAttention-4 ([FlashAttention-4](../papers/attention/kernels-and-io-aware/2603.05451-flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmet.md)) shows why Blackwell needs new pipelining.
* **Compilers:** [Tawa](../papers/serving-systems/gpu-kernels-and-compilers/2510.14719-tawa-automatic-warp-specialization-for-modern-gpus-with-asynchronous-r.md) (automatic warp specialisation from tile programs; matches hand-written
  FA-3, 1.1× over cuBLAS GEMM).
* **LLM-written kernels are real now:** [K-Search](../papers/serving-systems/gpu-kernels-and-compilers/2602.19128-k-search-llm-kernel-generation-via-co-evolving-intrinsic-world-model.md) (2.1× average over evolutionary baselines, 14.3× on
  MoE kernels) and [AVO](../papers/serving-systems/gpu-kernels-and-compilers/2603.24517-avo-agentic-variation-operators-for-autonomous-evolutionary-search.md) (7-day autonomous search beats FA-4 by up to 10.5%). Benchmark:
  [KernelBenchX](../papers/serving-systems/gpu-kernels-and-compilers/2605.04956-kernelbenchx-a-comprehensive-benchmark-for-evaluating-llm-generated-gp.md) (fusion tasks still fail 72% of the time).
* **Lossless compression that speeds things up:** [ZipServ](../papers/compression/_general/2603.17435-zipserv-fast-and-memory-efficient-llm-inference-with-hardware-aware-lo.md) (ASPLOS'26), −30% model size, 2.21×
  kernel-level and 1.22× end-to-end over vLLM.

## Hardware & co-design ([folder](../papers/serving-systems/hardware-accelerators/README.md))

* Model–system co-design from the labs: [Insights into DeepSeek-V3](../papers/serving-systems/hardware-accelerators/2505.09343-insights-into-deepseek-v3-scaling-challenges-and-reflections-on-hardwa.md) (ISCA'25; MLA, MoE, FP8, MTP and
  network-topology lessons, plus hardware wish-list) and [Step-3](../papers/serving-systems/disaggregated-prefill-decode/2507.19427-step-3-is-large-yet-affordable-model-system-co-design-for-cost-effecti.md) (attention–FFN disaggregation;
  4,039 tok/s/GPU at a 50 ms TPOT SLA on Hopper).
* PIM / near-memory:
  * [CENT](../papers/serving-systems/hardware-accelerators/2502.07578-pim-is-all-you-need-a-cxl-enabled-gpu-free-system-for-large-language-m.md) (ASPLOS'25, GPU-free CXL-PIM, 2.3× throughput, 2.9× less energy).
  * [PAPI](../papers/serving-systems/hardware-accelerators/2502.15470-papi-exploiting-dynamic-parallelism-in-large-language-model-decoding-w.md) (dynamic PIM/compute kernel mapping).
  * [Pimba](../papers/serving-systems/hardware-accelerators/2507.10178-pimba-a-processing-in-memory-acceleration-for-post-transformer-large-l.md) (MICRO'25, PIM for post-transformer state updates; MX formats are Pareto-optimal).
  * [Hermes NDP-DIMM](../papers/serving-systems/memory-and-offloading/2502.16963-make-llm-inference-affordable-to-everyone-augmenting-gpu-memory-with-n.md) (Llama-2-70B on a consumer GPU at 13.75 tok/s).
* LUT-based low-bit compute: [FIGLUT](../papers/serving-systems/hardware-accelerators/2503.06862-figlut-an-energy-efficient-accelerator-design-for-fp-int-gemm-using-lo.md) (HPCA'25) and [LUT-LLM](../papers/serving-systems/hardware-accelerators/2511.06174-lut-llm-efficient-large-language-model-inference-with-memory-based-com.md) (FPGA, VQ).
* New memory tiers: [High-Bandwidth Flash](../papers/serving-systems/memory-and-offloading/2607.10186-flashaccel-leveraging-high-bandwidth-flash-hbf-for-high-throughput-llm.md) (2.5× throughput per GPU) and [Ecco](../papers/serving-systems/memory-and-offloading/2505.06901-ecco-improving-memory-bandwidth-and-capacity-for-llms-via-entropy-awar.md)
  (ISCA'25, entropy-coded cache compression).

## Edge & local ([folder](../papers/serving-systems/edge-and-on-device/README.md))

* [Bitnet.cpp](../papers/quantization/1-bit-and-ternary/2502.11880-bitnet-cpp-efficient-edge-inference-for-ternary-llms.md): ternary LUT mpGEMM kernels. This is the reference for sub-2-bit CPU inference.
* Home clusters: [prima.cpp](../papers/serving-systems/edge-and-on-device/2504.08791-prima-cpp-fast-30-70b-llm-inference-on-heterogeneous-and-low-resource.md) (70B across mixed consumer devices, 5–17× lower TPOT than llama.cpp,
  exo and dllama).
* Hybrid local/cloud: [Minions](../papers/serving-systems/edge-and-on-device/2502.15964-minions-cost-efficient-collaboration-between-on-device-and-cloud-langu.md) (a local model reads the documents; 5.7× cheaper at 97.9% quality) and
  [SpecEdge](../papers/serving-systems/scheduling-and-batching/2505.17052-specedge-scalable-edge-assisted-serving-framework-for-interactive-llms.md) (edge GPU drafts, server verifies).
* 1M+ contexts on one consumer GPU: [HeadInfer](../papers/serving-systems/memory-and-offloading/2502.12574-headinfer-memory-efficient-llm-inference-by-head-wise-offloading.md) (head-wise KV offload, 4M tokens with an 8B model).
  [OasisKV](../papers/kv-cache/offloading-and-hierarchical-storage/2608.08097-oasiskv-scaling-in-decode-kv-cache-beyond-hbm-with-lookahead-sparse-pr.md) adds look-ahead sparse prefetch from host memory.

## Energy ([folder](../papers/serving-systems/energy-and-cost/README.md))

[Energy considerations of inference optimisations](../papers/serving-systems/energy-and-cost/2504.17674-energy-considerations-of-large-language-model-inference-and-efficiency.md) (ACL'25) finds that proper optimisation cuts energy
by up to 73%. For measurement tooling, use [TokenPowerBench](../papers/serving-systems/energy-and-cost/2512.03024-tokenpowerbench-benchmarking-the-power-consumption-of-llm-inference.md).
