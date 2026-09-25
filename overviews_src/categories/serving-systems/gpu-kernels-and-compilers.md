**Verdict.** Two stories run in parallel here. Attention kernels themselves (FlashAttention-3/4, FlashInfer, FlashMLA)
live in [`attention/kernels-and-io-aware`](../../attention/kernels-and-io-aware/README.md).

**1. Mega-kernels and deeper fusion are the biggest runtime win for low-batch decode.** Launch overhead and inter-kernel
HBM round trips dominate at batch 1–16. Options:
* [[2512.22219|MPK / Mirage Persistent Kernel]]: compile the whole model into SM-level task graphs run in one
  persistent kernel; up to 1.7× lower end-to-end latency.
* [[2505.22758|FlashFormer]]: a whole-transformer kernel.
* [[2508.18850|ClusterFusion]]: fuse QKV-proj + attention + out-proj via thread-block-cluster collectives; 1.61×
  end-to-end on H100. [[2604.23553|ClusterFusion++]] covers the full block.
* Dynamic shapes: [[2604.13327|Event Tensor]], [[2605.11581|Ada-MK]], [[2604.15379|Fleet]] (multi-die).
* MoE: [[2609.04244|MonoMoE]] (quantized MoE decode mega-kernel).

**2. LLMs now write competitive kernels, but benchmarks are leaky.**
* [[2502.10517|KernelBench]] (ICML'25) defined the task.
* RL-trained kernel models reach expert level on many operators:
  * [[2507.14111|CUDA-L1]]: contrastive RL, 1.42× median over KernelBench baselines;
  * [[2602.24286|CUDA Agent]]: large-scale agentic RL; faster than torch.compile on 100/100/92% of L1/L2/L3 tasks;
  * [[2602.05885|Dr. Kernel]] (Triton, unbiased multi-turn RL);
  * [[2507.05687|AutoTriton]], [[2510.17891|TritonRL]].
* Evolutionary agents beat hand-tuned SOTA on the hardest targets: [[2603.24517|AVO]] (7 days of autonomous evolution
  beats cuDNN by 3.5% and **FlashAttention-4 by 10.5%** on B200), [[2512.02551|CUDA-L2]] (HGEMM +19% over cuBLAS),
  [[2506.13131|AlphaEvolve]] (Google's stack).
* **Correctness is the weak spot.** Allclose-on-one-shape oracles pass wrong kernels: [[2606.20128|The Correctness
  Illusion]], and [[2609.21058]] found a "283×" kernel that wrote 0.3% of its output. Use multi-shape, scale-invariant
  checks and equivalence checking ([[2511.12638]]).

**3. New tile-level DSLs and compilers make hand-writing easier:**
* [[2504.17577|TileLang]]: decouples scheduling from dataflow; used for DeepSeek kernels;
* [[2504.12984|Tilus]]: arbitrary low-bit types;
* [[2510.14719|Tawa]]: automatic warp specialization; matches CUTLASS FA3;
* [[2504.16214|Hexcute]]: layout synthesis; [[2604.14825|Nautilus]]: auto-scheduling;
* [[2604.23466]]: evaluation of NVIDIA CUDA Tile.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2512.22219\|MPK: Mega-Kernelizing Tensor Programs]] | Compiler + runtime | Tensor program → SM-level task graph; decentralized in-kernel scheduler in one persistent kernel | Up to 1.7× lower end-to-end latency; near hardware limits |
| 2 | [[2603.24517\|AVO: agentic variation operators]] (NVIDIA) | Agentic evolution | Agent as the variation operator in evolutionary search, with micro-architecture reasoning | MHA kernels beating cuDNN (+3.5%) and FlashAttention-4 (+10.5%) on B200; transfers to GQA |
| 3 | [[2602.24286\|CUDA Agent]] | Agentic RL | Scalable data synthesis + skill-augmented dev environment (verify + profile) + stable long-horizon RL | Faster than torch.compile on 100/100/92% of KernelBench L1/L2/L3; ~40% above frontier proprietary models on L3 |
| 4 | [[2508.18850\|ClusterFusion]] | Fusion | Cluster-level collectives (DSMEM) to fuse QKV-proj → attention → out-proj | 1.61× average end-to-end latency on H100 |
| 5 | [[2504.17577\|TileLang]] | DSL | Composable tiled programming, scheduling primitives separate from dataflow | State-of-the-art kernels with far less code; basis of many 2025 kernels |
| 6 | [[2507.14111\|CUDA-L1]] | RL | Contrastive RL from speedup rewards only | 1.42× median, up to 120× over KernelBench references; ~2.8× vs torch.compile |
| 7 | [[2512.02551\|CUDA-L2]] | RL for GEMM | RL-driven search over HGEMM configurations | +19.2% over cuBLAS and +11.4% over cuBLASLt autotuning (offline) |
| 8 | [[2606.20128\|The Correctness Illusion]] / [[2609.21058\|How much of a real workload?]] | Verification | Fixed-shape allclose checks accept wrong kernels; scale-invariant oracles | **Mandatory reading before trusting AI-generated kernels** |
| 9 | [[2510.14719\|Tawa]] | Compiler | Automatic warp specialization with asynchronous references (Hopper/Blackwell) | 1.1× over cuBLAS GEMM; matches CUTLASS FA3 |
| 10 | [[2502.10517\|KernelBench]] (ICML'25) | Benchmark | 250 PyTorch → CUDA tasks with the fast_p metric | The standard benchmark (use with the fixes above) |
| 11 | [[2601.00227\|FlashInfer-Bench]] | Benchmark → deployment | Real LLM-serving kernel tasks + a path to deploy AI-written kernels in FlashInfer | Closes the loop from agent to production |
| 12 | [[2505.22758\|FlashFormer]] | Whole-model kernel | Entire forward pass in one kernel for low-batch inference | Speedups across sizes and quantizations |

**Also useful.**
* Cross-vendor: [[2505.16968|CASS]] (CUDA → HIP transpilation), [[2511.13274|KForge]], [[2609.04523|MaxKernel]] (TPU),
  [[2601.07160|AscendKernelGen]].
* Benchmarks: [[2502.14752|TritonBench]], [[2507.17773|MultiKernelBench]], [[2605.04956|KernelBenchX]],
  [[2605.23215|FastKernels]] (production), [[2608.17379|PTXBench]].
* Agent frameworks: [[2509.07506|Astra]] (from SGLang kernels), [[2511.01884|CudaForge]], [[2510.16996|STARK]],
  [[2507.23194|GEAK]] (AMD).
* Kernels for sampling: [[2607.20475|SonicSampler]] (fused sampling + speculative verification).
* Low-bit engines: [[2508.15601|TurboMind]], [[2603.27462|RSR-core]].
* SASS scheduling: [[2501.08071|CuAsmRL]].

**Runtime recommendation.**
* For latency-critical decode, move to **persistent mega-kernels** (MPK-style) or at least cluster-level fusion of the
  attention block and CUDA-graph everything.
* Write new kernels in TileLang/Triton/CuTe DSL.
* Use agentic RL/evolution (CUDA Agent, AVO-style) to squeeze hot kernels per GPU generation, but gate every generated
  kernel with **multi-shape, scale-invariant correctness tests** and equivalence checks.
