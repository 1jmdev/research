**Verdict.** This is the catch-all serving bucket: routing between models, platforms, simulators, determinism,
agent-workload studies and surveys. Four themes are actionable.

1. **Numerical nondeterminism is a correctness problem, not a curiosity.**
   * [[2506.09501|Understanding numerical nondeterminism]]: under BF16 + greedy decoding, reasoning models vary by **up
     to 9% accuracy and 9K tokens** of response length across GPU count, type and batch size. The fix is LayerCast:
     16-bit weights, FP32 compute.
   * [[2601.17768|LLM-42]]: determinism *on demand* via verified speculation (fixed-shape replay of candidate tokens),
     paying overhead only for traffic that needs it.
   * [[2605.19537|The Silent Hyperparameter]]: the inference backend itself shifts benchmark scores. Report the serving
     stack.
   * See also [[2511.17826]] (TP-invariant determinism for RL).
2. **Model routing and cascades.**
   * [[2502.08773|UniRoute]] (Google): route to *unseen* LLMs via cluster-based representations with excess-risk
     bounds.
   * [[2510.08731|vLLM Semantic Router]]: decide *when to reason*; −47% latency and −48.5% tokens at +10.2 pts on
     MMLU-Pro.
   * [[2502.20576|OmniRouter]]: budget-constrained global routing.
   * Survey: [[2603.04445]].
3. **Agent workloads reshape the front end.**
   * [[2607.29678|TokTier]]: stateful incremental tokenization, because re-tokenizing long agent contexts dominates TTFT
     tails; −16–34% median TTFT.
   * [[2606.16824|CacheWise]]: tool-metadata-guided KV eviction for coding agents; 2–2.6× fewer evictions, up to 3.5×
     faster sessions.
   * [[2510.18586|TokenCake]]: KV-centric multi-agent serving.
   * Characterization: [[2608.15127]].
4. **Production platforms and configuration.**
   * [[2504.03648|AIBrix]] (vLLM control plane): distributed KV, LoRA management, autoscaling; +50% throughput and −70%
     latency from cross-node KV reuse.
   * [[2501.14417|DeepServe]] (Huawei): serverless at scale on Ascend; NPU-fork scales to 64 instances in seconds.
   * [[2601.06288|AIConfigurator]] (NVIDIA): framework-agnostic configuration search in ~30 s; +40–50% over defaults.
   * Simulators: [[2602.23036|LLMServingSim 2.0]], [[2508.03148|Frontier]], [[2601.00397|Revati]].

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2506.09501\|Numerical sources of nondeterminism]] | Correctness | Trace accuracy and length drift to reduction-order changes under BF16; LayerCast | Up to 9% accuracy / 9K-token variance removed with FP32 compute on 16-bit weights |
| 2 | [[2601.17768\|LLM-42]] | Determinism | Speculate with fast kernels, verify with fixed-shape reductions, roll back violations | Determinism with overhead proportional to deterministic traffic |
| 3 | [[2510.08731\|vLLM Semantic Router: when to reason]] | Routing | Classify queries; enable reasoning mode only when it helps | +10.2 pts MMLU-Pro with −47.1% latency and −48.5% tokens |
| 4 | [[2502.08773\|UniRoute]] | Routing | Represent each LLM by per-cluster errors → route among unseen models | Theory-backed routing across 30+ unseen LLMs |
| 5 | [[2504.03648\|AIBrix]] | Platform | Cloud-native control plane for vLLM: distributed KV pool, LoRA, autoscaling, diagnostics | +50% throughput, −70% latency via cross-node KV reuse |
| 6 | [[2601.06288\|AIConfigurator]] | Tuning | Performance models for TRT-LLM/vLLM/SGLang → configuration search without GPUs | +40% dense, +50% MoE vs defaults in ~30 s |
| 7 | [[2607.29678\|TokTier]] | Front end | Exact stateful CPU+GPU incremental tokenization for agent sessions | −16–34% median TTFT; 1,821 req/s at 50 ms P99 with 4 cores + 1 GPU |
| 8 | [[2606.16824\|CacheWise]] | Agent KV | Prefix-aware scheduling + reuse-aware eviction from tool-call metadata | 2–2.6× fewer evictions; up to 3.5× faster sessions |
| 9 | [[2503.08311\|Mind the Memory Gap]] | Characterization | Large-batch decode is DRAM-bandwidth-bound even at large batches; batching advisor | Free memory for co-located replicas |
| 10 | [[2602.00269\|VoxServe]] | Speech LMs | Streaming-centric serving abstraction for speech LMs | 10–20× throughput at comparable latency |
| 11 | [[2501.14417\|DeepServe]] (ATC'25) | Serverless | Pre-warmed pods, DRAM pre-loading, NPU-fork, mixed PD-disaggregated/colocated | Scale to 64 instances in seconds; a year in production |
| 12 | [[2605.19537\|The Silent Hyperparameter]] | Reproducibility | Same model and prompts across backends | Backend defaults in logit processing shift scores |

**Also useful.**
* Surveys: [[2505.01658]] (25 inference engines), [[2504.19720|Taming the Titans]], [[2506.21901]].
* Engine bugs: [[2506.09713]].
* Cold start: [[2502.15524|HydraServe]].
* Steering at serving time: [[2509.25175|EasySteer]].
* RAG: [[2502.20969|TeleRAG]].
* Semantic caching: [[2508.07675]].
* Time budgets: [[2512.21859|TimeBill]].
* CPU bottlenecks: [[2603.22774]].
* Agents building serving systems: [[2605.06068|VibeServe]], [[2602.19594|ISO-Bench]].

**Runtime checklist.**
* An opt-in deterministic mode (fixed reduction order or LLM-42-style verification).
* FP32-accumulate everywhere for reasoning evals.
* Routing hooks: model cascade plus a reasoning on/off switch.
* Stateful tokenization and tool-aware KV eviction for agents.
* A config search tool and a simulator.
