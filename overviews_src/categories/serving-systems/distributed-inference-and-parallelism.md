**Verdict.** This bucket covers parallelism for both **inference** and **training** (180 papers). The shared theme is
*hiding or removing communication*.

**Inference:**
* **Compute–communication overlap for TP/EP**:
  * [[2502.19811|Comet]] (ByteDance): fine-grained MoE overlap, 1.96× per MoE layer, deployed on 10K-GPU clusters,
    millions of GPU-h saved;
  * [[2505.11329|TokenWeave]] (Microsoft): token-split overlap for small low-latency batches, sometimes faster than the
    same model with *no* communication;
  * [[2511.06605|DMA-offloaded]] collectives;
  * **device-initiated point-to-point**: [[2510.27656|fabric-lib]] (Perplexity), KV transfer and MoE dispatch on
    EFA/CX-7.
* **Architecture that removes sync points**: [[2501.06589|Ladder-residual]] overlaps the TP all-reduce with the next
  block's compute; Llama-3.1-8B converted with 3B tokens. [[2502.20727|Sync-Point Drop]] does something similar.
* **Parallelism that adapts to load**:
  * [[2509.16495|Shift Parallelism]] (Snowflake): switch between TP (low latency) and a KV-compatible sequence
    parallelism (high throughput) on the fly;
  * [[2503.06433|Seesaw]]: re-shard between prefill and decode, 1.36–1.78× over vLLM;
  * [[2604.12171|PipeLive]]: live PP reconfiguration.
* **Multi-million-token decode**: [[2507.07120|Helix Parallelism]] (NVIDIA) shards KV over sequence for attention and
  reuses the same GPUs as TP for FFN. 1.5× lower token latency and 32× larger batches for DeepSeek-R1 on Blackwell.
* **Determinism**: [[2511.17826|deterministic inference across TP sizes]] gives bitwise-identical logits between vLLM
  (TP>1) and FSDP training, which removes the RL training/inference mismatch.

**Training:**
* **Long context**:
  * [[2510.18121|Core Attention Disaggregation]] (DistCA): stateless attention on dedicated servers, 1.35× at 512K;
  * [[2502.21231|ByteScale]]: 2M-token context on >12K GPUs, up to 7.89×;
  * [[2510.10620|DCP]], [[2602.21196|Untied Ulysses]], [[2506.13996|Arctic Long Sequence Training]];
  * linear attention: [[2502.07563|LASP-2]], [[2507.01004|ZeCO]] (All-Scan, 1M tokens on 64 GPUs ≈ 16K on one).
* **MoE**: [[2504.14960|MoE Parallel Folding]] (Megatron-Core) gives different parallel layouts to attention and MoE
  layers; 49.3% MFU on Mixtral-8×22B. See also [[2603.07685]].
* **Scale and reliability**:
  * [[2510.20171|Collective communication for 100K+ GPUs]] (Meta, Llama 4);
  * [[2602.00277|FT-HSDP]]: effective training time at 100K GPUs from 44% to 80%;
  * [[2509.03018|Mycroft]]: collective-communication tracing.
* **Low-communication / decentralized**: [[2508.15706|SparseLoCo]] (top-k + 2-bit pseudo-gradients beats DiLoCo),
  [[2506.10911|NoLoCo]] (no all-reduce), [[2604.21428|Decoupled DiLoCo]], [[2507.00217|CrossPipe]] (cross-DC
  pipelines).
* **Post-training**: [[2601.19362|ODC]] (ICLR'26) replaces FSDP collectives with parameter-server-style point-to-point
  for imbalanced RL batches, +36%.

### Hand ranking

| # | Paper | Side | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.19811\|Comet]] | MoE (train + infer) | Data-dependency analysis → fine-grained overlap of expert GEMMs with all-to-all | 1.96× per MoE layer, 1.71× end to end; production at 10K-GPU scale |
| 2 | [[2507.07120\|Helix Parallelism]] (NVIDIA) | Inference, ultra-long | KV-parallel attention + TP FFN on the same GPUs with a temporal pipeline | Up to 1.5× lower TTL; 32× larger batches for DeepSeek-R1 on Blackwell |
| 3 | [[2511.17826\|Deterministic inference across TP sizes]] | Inference/RL | TP-invariant, batch-invariant kernels | Bitwise identical vLLM ↔ FSDP; zero RL train/infer mismatch |
| 4 | [[2505.11329\|TokenWeave]] | Inference TP | Split tokens into overlapping waves; fused all-reduce + RMSNorm using few SMs | Up to 1.28× lower latency, 1.19× throughput; sometimes beats the no-communication baseline |
| 5 | [[2510.18121\|Core Attention Disaggregation]] | Training long context | Move parameter-free core attention to balanced attention servers | 1.35× at 512K on 512 H200; removes DP/PP stragglers |
| 6 | [[2509.16495\|Shift Parallelism]] | Inference | Dynamic switch TP ↔ KV-invariant SP depending on load | Better latency–throughput trade-off than TP or DP on dynamic traffic |
| 7 | [[2504.14960\|MoE Parallel Folding]] | MoE training | Decouple attention and MoE parallel mappings; token dispatcher for both dropping modes | 49.3% MFU (Mixtral-8×22B), 39% (Qwen2-57B-A14B) on H100 |
| 8 | [[2501.06589\|Ladder-residual]] | Architecture | Route residuals so TP communication overlaps with the next block | ~30% faster TP inference at 70B; retrofit Llama-3.1-8B with 3B tokens |
| 9 | [[2602.00277\|FT-HSDP on 100K GPUs]] | Training reliability | Fault-tolerant all-reduce; replicas drop and rejoin asynchronously | Effective training time 44% → 80% |
| 10 | [[2508.15706\|SparseLoCo]] | Low-comm training | Top-k sparsified + 2-bit pseudo-gradients with error feedback in DiLoCo | Beats DiLoCo in quality and compression (178M–2B, MoE) |
| 11 | [[2502.21231\|ByteScale]] | Long-context training | Dynamic hybrid DP×CP for mixed long/short data + balance scheduler | Up to 7.89× at 256K–2048K context, 12K GPUs |
| 12 | [[2510.27656\|fabric-lib]] | Communication | Portable RDMA point-to-point (CX-7 + EFA) for KV transfer, weight updates and MoE | 400 Gbps; MoE decode latency better than DeepEP on CX-7 |

**Also useful.**
* Pipeline: [[2503.01328|PipeOffload]], [[2504.14519|SlimPipe]], [[2606.30634]] (asynchronous PP),
  [[2505.01099]] (Nesterov for async PP).
* Balance: [[2503.17924|WLB-LLM]], [[2503.19050|Mist]].
* Multimodal training: [[2503.11367|Cornstarch]] (ICML'26), [[2503.23830|OrchMLLM]].
* Topology: [[2509.15940]] (topology-aware alignment on 9,600 GPUs).
* Inference prefetch: [[2501.08192|PRESERVE]].
* Long-context inference: [[2502.12085|APB]].

**Runtime checklist.**
* Overlap TP all-reduce with compute (TokenWeave-style or Ladder-residual models).
* Runtime switching between TP and SP/DP.
* Helix-style KV-sequence sharding for million-token decode.
* **Deterministic, TP-invariant kernels** as an option for RL.
* Portable point-to-point RDMA for KV/MoE traffic.
