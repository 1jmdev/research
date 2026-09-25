**Verdict.** Small models (0.1–4B) are where runtime choices matter most: they run on CPU, phone and edge, and at
batch 1. The 2025–26 lessons are:

1. **Data-centric training beats cleverness.** [[2502.02737|SmolLM2]] (Hugging Face) uses a multi-stage, manually
   rebalanced 11T-token mix with new FineMath, Stack-Edu and SmolTalk datasets, all released. It beats Qwen2.5-1.5B and
   Llama-3.2-1B. [[2509.24945|MobileLLM-R1]] shows ~2T well-chosen tokens suffice for sub-billion reasoners.
   [[2602.02522|IMU-1]] (430M on 72B tokens) approaches models trained on 56× more data.
2. **Design for latency, not parameter count.** [[2511.18890|Nemotron-Flash]] (NVIDIA, NeurIPS'25) searches depth/width
   ratios and operator mixes (attention vs linear) for *measured* latency. It gets +5.5% accuracy with 1.3–1.9× lower
   latency and 18.7–45.6× higher throughput vs Qwen3-1.7B/0.6B. Also [[2603.15954|MobileLLM-Flash]],
   [[2511.06719|MobileLLM-Pro]] (128K context, robust to 4-bit), [[2608.20210|Daedalus-150M]] (conv-attention hybrid
   for CPU).
3. **Sparse small models for local deployment.** [[2507.20984|SmallThinker]] (PowerInfer) combines a native MoE with
   sparse FFNs and pre-attention routing that hides SSD/flash latency. 4B-A0.6B and 21B-A3B run at >20 tok/s on
   consumer CPUs in 1 GB / 8 GB. See also MobileMoE in the MoE area.
4. **Small reasoners can reach frontier-level math and code.**
   * [[2511.06221|VibeThinker-1.5B]] (Weibo): diversity-driven SFT → MaxEnt-guided RL for **$7,800**; beats
     Magistral Medium and matches GPT-OSS-20B-medium on math.
   * [[2606.16140|VibeThinker-3B]]: 94.3 AIME'26, 80.2 LiveCodeBench v6.
   * But [[2506.07712|long-CoT SFT hurts small models first]] ("through the valley") and needs large-scale SFT to
     recover.
   * [[2502.11569]]: 72 SLMs × 17 benchmarks.
5. **Derive small models from big ones** (pruning + distillation): [[2601.08584|Ministral 3]] (cascade distillation),
   Minitron (see compression/structured-pruning).
6. **SLMs as the default for agents**: [[2506.02153|Small Language Models are the Future of Agentic AI]] (NVIDIA
   position), [[2510.03847]] (SLM-default, LLM-fallback). Agentic small models: [[2602.13367|Nanbeige4.1-3B]] (up to
   600 tool turns), [[2604.21590|AgenticQwen]], [[2506.22760|Jan-nano]].

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.02737\|SmolLM2]] | Recipe | Multi-stage data-centric training with manual rebalancing + new specialized datasets | 1.7B beats Qwen2.5-1.5B and Llama-3.2-1B; fully open data |
| 2 | [[2511.18890\|Nemotron-Flash]] (NeurIPS'25) | Latency-optimal design | Search depth/width and operator mix (hybrid attention/linear) against real latency; weight normalization, meta tokens | +5.5% accuracy; 1.3–1.9× lower latency; 18.7–45.6× throughput vs Qwen3-1.7B/0.6B |
| 3 | [[2511.06221\|VibeThinker-1.5B]] | Small reasoner | Spectrum-to-Signal: diversity-maximizing SFT then MaxEnt-guided RL | $7.8K training cost; beats much larger models on AIME/LiveCodeBench |
| 4 | [[2507.20984\|SmallThinker]] | Local-first MoE | Two-level sparsity (MoE + sparse FFN) + pre-attention router for I/O prefetch | 4B-A0.6B / 21B-A3B at >20 tok/s on consumer CPUs in 1 GB / 8 GB |
| 5 | [[2509.24945\|MobileLLM-R1]] (Meta) | Sub-billion reasoning | Curated/resampled ~2T tokens; open recipe | Strong sub-1B reasoning with far less data than Qwen3-0.6B's 36T |
| 6 | [[2504.05299\|SmolVLM]] | Small VLM | Aggressive visual tokenization (pixel shuffle), balanced encoder/LM split | 256M VLM in <1 GB GPU memory; good video understanding |
| 7 | [[2601.08584\|Ministral 3]] (Mistral) | Derived family | Cascade distillation: iterative pruning + continued training with distillation | 3B/8B/14B base/instruct/reasoning with vision, Apache-2.0 |
| 8 | [[2506.07712\|Through the Valley]] | Training science | Long-CoT SFT first degrades SLMs; error accumulation | Needs large SFT before RL; guidance for small reasoners |
| 9 | [[2606.16140\|VibeThinker-3B]] | Small reasoner | Verifiable-reasoning focus with claim-level TTS | 94.3 AIME'26 (97.1 with TTS), 80.2 LiveCodeBench v6 |
| 10 | [[2511.06719\|MobileLLM-Pro]] | On-device | Implicit positional distillation for 128K context; QAT-robust | Minor regression at 4-bit; long context on device |
| 11 | [[2506.02153\|SLMs are the future of agentic AI]] | Position | Economics and capability of SLMs for agent subtasks | Heterogeneous SLM-first agent systems |

**Also useful.**
* More families: [[2512.06266|Nanbeige4-3B]] (23T tokens), [[2511.19496|Xmodel-2.5]], [[2512.07612|PCMind-2.1]].
* Small VLMs: [[2603.06569|Penguin-VL]] (LLM-initialized vision encoder), [[2505.09498|Flash-VL 2B]].
* Encoder-decoder SLMs: [[2501.16273]].
* Micro LMs for instant responses: [[2604.19642]].
* Research framework: [[2509.16413|Pico]].
* Surveys: [[2501.05465]], [[2505.07460]].

**For runtimes.** SLMs are **latency- and overhead-bound**, not FLOP-bound:
* CUDA graphs or mega-kernels;
* low-bit GEMV (INT4/ternary) tuned per CPU/NPU;
* LM-head optimizations for 150K+ vocabularies;
* MoE-on-CPU with flash-backed experts (SmallThinker-style prefetch);
* hybrid attention/linear kernels (Nemotron-Flash uses DeltaNet-style layers).
