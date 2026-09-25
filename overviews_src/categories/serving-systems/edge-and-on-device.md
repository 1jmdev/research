**Verdict.** On-device inference in 2025–26 is about three things.

1. **Using the whole SoC.** Heterogeneous GPU+NPU(+CPU) execution beats single-accelerator engines by large margins:
   * [[2501.14794|HeteroInfer]] (SOSP'25): 1.34–6× over GPU-only and NPU-only engines;
   * [[2506.24045|Agent.xpu]]: concurrent reactive and proactive agent flows on NPU+iGPU; ≥91% lower reactive latency;
   * NPUs need **hardware-aware quantization**: [[2509.23324|mobile-NPU test-time scaling]] uses tile-aligned group
     quantization + LUT softmax (up to 19× GEMM speedup), so small models with test-time scaling beat larger ones;
   * NPU-first model design: [[2512.02924|AutoNeural]].
2. **Kernels that actually realize the memory savings.** [[2605.30571|Memory-bound but not bandwidth-limited]]: at
   batch 1, many INT4 paths barely beat BF16 (NF4 59 ms vs 62 ms BF16), and only tuned INT4 kernels reach the floor
   (ExLlamaV2 17 ms). **Kernel quality > quantization format** on edge.
   * Engines: [[2506.10443|MNN-LLM]] (up to 8.6× over mainstream mobile frameworks; DRAM-Flash hybrid storage),
     llama.cpp / MLX ([[2511.05502|Apple Silicon runtime comparison]]; [[2601.19139]]: native continuous batching +
     content-hash vision prefix caching on M4 Max).
3. **Device–cloud collaboration.**
   * [[2502.15964|MinionS]] (Hazy Research): the local model executes decomposed subtasks over document chunks, and
     the cloud model plans. **5.7× lower cost at 97.9% of cloud quality**.
   * Edge-drafted speculative decoding: [[2506.09397|SLED]], [[2505.21594]] (early-exit target), [[2511.21669|DSD]],
     [[2505.17052|SpecEdge]].
   * Home clusters: [[2504.08791|prima.cpp]] runs 70B across consumer devices; 32B at 26 tok/s with speculation.

Model design for edge is its own lever: [[2602.10377|hardware co-design scaling laws]] (roofline + loss model, 19.4%
lower PPL at Qwen2.5-0.5B latency on Jetson Orin); small models in
[`models-and-architectures/small-language-models`](../../models-and-architectures/small-language-models/README.md);
MoE on phones in [`mixture-of-experts/architecture-and-routing`](../../mixture-of-experts/architecture-and-routing/README.md)
(MobileMoE).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2501.14794\|HeteroInfer: characterizing mobile SoCs]] (SOSP'25) | Engine | Profile GPU/NPU/memory behaviour → layer-level and tensor-level heterogeneous parallelism | 1.34–6.02× over GPU-only and NPU-only engines; negligible interference |
| 2 | [[2502.15964\|Minions / MinionS]] | Local–cloud protocol | Remote model decomposes; local model runs many parallel subtasks over chunks | 5.7× lower cost at 97.9% of remote-only quality |
| 3 | [[2605.30571\|Memory-bound but not bandwidth-limited]] | Measurement | Batch-1 decode on L4-class GPUs vs the memory floor per quantization path | Only tuned int4 kernels realize the savings. **Profile the kernel, not the format** |
| 4 | [[2506.10443\|MNN-LLM]] | Mobile engine | Instruction-set-aware weight layout, DRAM-Flash hybrid storage, multicore balance, mixed precision | Up to 8.6× faster than mainstream mobile LLM frameworks |
| 5 | [[2509.23324\|Test-time scaling on mobile NPUs]] | NPU | Tile-aligned group quantization + LUT nonlinearities on Snapdragon NPUs | 19× mixed-precision GEMM; small model + TTS beats bigger models |
| 6 | [[2506.24045\|Agent.xpu]] | Agentic on SoC | Operator–accelerator affinity; preemptive scheduling of reactive vs proactive flows | 1.2–4.9× proactive throughput; ≥91% lower reactive latency |
| 7 | [[2504.08791\|prima.cpp]] | Home cluster | Piped-ring parallelism + heterogeneity-aware scheduler (Halda) over consumer devices | 70B at 674 ms/token; 32B + speculation at 26 tok/s |
| 8 | [[2602.10377\|Hardware co-design scaling laws]] | Model design | Joint loss-vs-architecture law + roofline latency; search 1,942 architectures on Jetson | −19.4% perplexity at equal latency vs Qwen2.5-0.5B |
| 9 | [[2601.19139\|Native LLM/MLLM inference on Apple Silicon]] | Engine | MLX-native continuous batching + content-hashed image prefix cache | 525 tok/s on M4 Max; 28× faster repeated image queries |
| 10 | [[2608.15018\|S2-MoE]] | MoE on edge | Self-speculative decoding for MoE in llama.cpp with reuse-aware expert gating | Up to 5.3× (≈2× average) over AR decoding |
| 11 | [[2512.02924\|AutoNeural]] | NPU co-design | NPU-friendly VLM (MobileNet-style vision + SSM/Transformer LM) for stable INT4/8/16 | Real-time on Qualcomm SA8295P; 4× longer context |

**Also useful.**
* Profiling and benchmarks: [[2510.06126|lm-Meter]], [[2602.11506|RooflineBench]], [[2609.19169|SiliconBench]],
  [[2504.00002]] (mobile measurement study), [[2508.00904]] (hardware-agnostic performance forecasting).
* Multi-tenant LoRA on edge: [[2507.01438|EdgeLoRA]].
* Energy: [[2508.09173|Camel]] (GPU frequency + batch tuning).
* Distributed edge: [[2504.08242|Jupiter]], [[2508.20375|CoFormer]].
* Edge VLMs: [[2510.11496|AndesVL]], [[2512.14052|HyperVL]].
* Small-batch runtime state: [[2606.20537]] (execution-state capsules).
* Survey: [[2507.16731]] (edge–cloud collaboration).

**Runtime checklist for edge.**
* Per-SoC backends: CPU (NEON/AMX/SME), GPU (Metal/Vulkan/OpenCL), NPU (QNN/ANE).
* Layer-level heterogeneous scheduling.
* NPU-aligned group quantization.
* **Batch-1 GEMV kernels benchmarked against the memory floor.**
* Flash-backed weights with DRAM caching.
* Local speculative drafting with optional cloud verification.
* Prefix caching for images and system prompts.
