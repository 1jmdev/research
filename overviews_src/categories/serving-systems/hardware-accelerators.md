**Verdict.** This is mostly architecture research (200+ papers: PIM, FPGA, ASIC, wafer-scale, photonics). For a
*software* runtime builder it is useful in three ways:

1. **What the hardware roadmap rewards.** Decode is bandwidth- and capacity-bound, so future gains come from memory,
   not FLOPs.
   * [[2601.05047|Challenges and research directions for LLM inference hardware]] lists High-Bandwidth Flash (10×
     capacity at HBM-like bandwidth), processing-near-memory / 3D memory-logic stacking, and low-latency interconnect.
   * [[2505.09343|Insights into DeepSeek-V3]] (ISCA'25): hardware lessons from MLA, MoE, FP8 training and a multi-plane
     network. Asks for finer-grained scaling in tensor cores, scale-up/scale-out convergence and low-latency fabrics.
2. **Production non-NVIDIA serving.** [[2506.12708|CloudMatrix384]] (384 Ascend 910 NPUs on a unified bus) serves
   DeepSeek-R1 at 6,688 prefill / 1,943 decode tok/s per NPU (<50 ms TPOT) with INT8. Useful as a design reference for
   peer-to-peer all-to-all plus disaggregated pools.
3. **Low-bit and LUT hardware validates ternary/2-bit model design.** LUT/ternary accelerators ([[2509.13765|TENET]],
   [[2502.16473|TerEffic]], [[2511.06174|LUT-LLM]], [[2503.06862|FIGLUT]]) show 4–21× energy efficiency over A100 for
   ternary or LUT-quantized models, and TENET-ASIC is 2.7× faster. This matters if you build BitNet-style models (see
   [`quantization/1-bit-and-ternary`](../../quantization/1-bit-and-ternary/README.md)).

Also notable:
* PIM for decode: [[2502.07578|CENT]] (CXL PIM, GPU-free, 5.2× tokens per dollar), [[2502.15470|PAPI]] (dynamic
  GPU↔PIM kernel mapping), [[2507.10178|Pimba]] (PIM for SSM + attention with MX arithmetic).
* Wafer-scale inference: [[2502.04563|WaferLLM]] (OSDI'25, 10–20× over A100 clusters), [[2510.25258|MoEntwine]].
* Blackwell microbenchmarks: [[2512.02189]] (FP4/FP6 tensor-core characterization vs H200).

### Hand ranking (runtime-relevant first)

| # | Paper | Kind | Key point | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.09343\|Insights into DeepSeek-V3]] (ISCA'25) | Co-design retrospective | MLA, MoE, FP8 and multi-plane network choices made *because* of H800 limits | Blueprint of hardware asks from a frontier lab |
| 2 | [[2506.12708\|Serving LLMs on CloudMatrix384]] (Huawei) | Production system | Peer-to-peer serving (disaggregated prefill/decode/cache) over a UB supernode; large-scale EP; INT8 | 1,943 decode tok/s/NPU at <50 ms TPOT; 538 tok/s/NPU at 15 ms |
| 3 | [[2601.05047\|Challenges for LLM inference hardware]] | Position | Memory capacity/bandwidth and latency are the constraints; HBF, PNM, 3D stacking, fast interconnect | Where hardware is heading; plan runtimes for tiered memory |
| 4 | [[2512.02189\|Microbenchmarking Blackwell]] | Characterization | Tensor-core, memory and precision behaviour of B200 (FP4/FP6/FP8) | 1.55× GPT-1.3B training vs H200; 32% better energy efficiency |
| 5 | [[2502.04563\|WaferLLM]] (OSDI'25) | Wafer-scale | Mesh-aware GEMM/GEMV and parallelism for hundreds of thousands of cores | 10–20× over A100 clusters running SGLang/vLLM |
| 6 | [[2502.07578\|CENT: PIM is all you need]] | CXL-PIM | GPU-free decode on CXL memory with PIM; pipeline and tensor parallelism across devices | 2.3× throughput, 2.9× less energy, 5.2× tokens per dollar vs GPUs |
| 7 | [[2509.13765\|TENET]] | Ternary ASIC/FPGA | Sparse ternary LUT core + dynamic activation N:M sparsity | 21.1× energy efficiency (ASIC) and 2.7× lower latency vs A100 |
| 8 | [[2507.10178\|Pimba]] | PIM for SSM/attention | Shared state-update engine with MX arithmetic for post-transformer models | Up to 4.1× over GPU, 2.1× over GPU+PIM |
| 9 | [[2502.15470\|PAPI]] | Heterogeneous PIM | Runtime-dynamic mapping of decode kernels to PIM or compute units | 1.8× over a heterogeneous accelerator; 11.1× over PIM-only |
| 10 | [[2511.06174\|LUT-LLM]] | FPGA | Memory-based (table lookup) inference with a conversion recipe | Qwen3-1.7B: 1.1–3.3× faster, 3–6.6× more energy-efficient than GPU |
| 11 | [[2602.18568\|RPU: reasoning processing unit]] | Architecture | Bandwidth-first chip for long reasoning decode | Simulated 45× lower latency vs H100 at iso-TDP (Llama-3-405B) |

**Also useful.**
* FPGA: [[2502.15260|LightMamba]], [[2504.16266|TeLLMe]], [[2507.03308|Hummingbird]], [[2509.19873|SpecMamba]].
* Near-memory MoE: [[2509.09420|HD-MoE]].
* Attention in hardware: [[2507.11331|SystolicAttention]], [[2505.05772|STARC]] (sparse attention on PIM).
* Reliability: [[2503.24053|ReaLM]] (statistical ABFT).
* Datacenter design: [[2506.15006]], [[2503.20377|UB-Mesh]].
* Edge KV on eDRAM: [[2510.16040|Kelle]].

**Runtime takeaway.** Design the runtime around **memory tiers** (HBM → CXL/DRAM → HBF/SSD), bandwidth-bound decode
kernels and pluggable backends (Ascend/AMD/wafer-scale). Keep a ternary/LUT path, because emerging hardware rewards it
most.
