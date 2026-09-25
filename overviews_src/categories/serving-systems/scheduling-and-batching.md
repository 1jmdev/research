**Verdict.** Continuous batching + chunked prefill + paged KV + prefix caching (vLLM / SGLang / LightLLM baseline) is
solved. Throughput-optimality is even proven for Sarathi-style and SGLang-style schedulers
([[2504.07347|throughput-optimal scheduling]]). The 2025–26 frontier moved **up the stack**:

1. **Agents are programs, not requests.** Schedule by *program* (the whole multi-call trajectory):
   * [[2502.13965|Autellix]]: program-level preemption, 4–15× throughput at equal latency vs vLLM;
   * [[2511.02230|Continuum]]: keep an agent's KV cache pinned with a **time-to-live across tool calls**, >8× lower
     job completion time on SWE-Bench/BFCL/OpenHands;
   * [[2602.13692|ThunderAgent]]: program-aware; 1.5–3.6× serving and 1.8–3.9× RL-rollout throughput;
   * [[2605.00528|SAGA]]: workflow-atomic scheduling;
   * [[2601.22705|CONCUR]]: congestion control on the number of active agents so KV doesn't thrash; 4.09× batch agentic
     throughput;
   * [[2603.18897|PASTE]]: run tool execution speculatively in parallel with generation, −43.5% task time.
2. **Goodput under SLOs, not raw throughput.**
   * [[2507.10150|Past-Future scheduler]] (ASPLOS; LightLLM): predicts future KV demand from past output-length
     distributions, 2–3× goodput;
   * [[2504.08784|SLOs-Serve]]: multi-SLO, 2.2× per-GPU capacity;
   * [[2505.23022|Scorpio]]: TTFT/TPOT guards with deadline reordering and admission control;
   * [[2504.20068|JITServe]].
3. **Output-length uncertainty is the core scheduling input.** Lengths are heavy-tailed, and log-t fits them.
   [[2604.00499|TIE]] schedules on a tail-inflated expectation: 2.31× lower per-token latency online, 1.42× offline
   throughput.
4. **Specialized engines for special workloads.**
   * [[2505.07203|PrefillOnly]]: classification/embedding-style requests keep only the last layer's KV and use exact
     JCT → SRJF; 4× QPS.
   * [[2510.24051|Pie]] (SOSP'25): *programmable* serving; app logic as WebAssembly "inferlets" controlling KV and
     generation; 1.3–3.4× on agentic workflows.
   * [[2605.13779|MinT]]: million-scale LoRA catalogs over 1T-class bases.
5. **Evaluate correctly.** [[2507.09019|On evaluating LLM serving systems]] catalogues benchmarking anti-patterns (wrong
   metrics, unrealistic arrivals, ignoring the prefill/decode split).

### Hand ranking

| # | Paper | Focus | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.13965\|Autellix]] | Agentic | Program-level scheduling (preempt/prioritize LLM calls by the program's completed work) | 4–15× program throughput at equal latency vs vLLM |
| 2 | [[2511.02230\|Continuum]] | Agentic multi-turn | KV cache TTL across tool calls + program-level FCFS | >8× lower average JCT on SWE-Bench/BFCL/OpenHands (8B–355B) |
| 3 | [[2507.10150\|Past-Future scheduler / LightLLM]] (ASPLOS) | SLA goodput | Predict future peak KV memory from historical output-length distribution | 2–3× goodput under heavy load |
| 4 | [[2602.13692\|ThunderAgent]] | Agentic + RL rollouts | LLM Programs abstraction managing KV, sandbox state and tool assets together | 1.5–3.6× serving, 1.8–3.9× RL rollout, 4.2× disk savings |
| 5 | [[2510.24051\|Pie]] (SOSP'25) | Programmability | WebAssembly inferlets own KV and decoding logic inside the server | 3–12% overhead on standard tasks; 1.3–3.4× on agentic workflows |
| 6 | [[2504.08784\|SLOs-Serve]] | Multi-SLO | DP-based token allocation across prefill/decode per SLO class + multi-replica routing | 2.2× per-GPU capacity |
| 7 | [[2604.00499\|TIE: uncertainty-aware length scheduling]] | Length prediction | Heavy-tailed (log-t) length distributions + tail-inflated expectation | 2.31× lower per-token latency; 1.42× offline throughput |
| 8 | [[2505.07203\|PrefillOnly]] | Special engine | Store only the last layer's KV for single-token outputs; exact JCT → SRJF | Up to 4× QPS without P99 inflation |
| 9 | [[2601.22705\|CONCUR]] | Agentic batch | Cache-aware congestion control of concurrent agents | 4.09× (Qwen3-32B) and 1.9× (DeepSeek-V3) batch throughput |
| 10 | [[2504.07347\|Throughput-optimal scheduling]] | Theory | Fluid-limit analysis of batched LLM queues incl. agent DAGs | Sarathi/SGLang-style are throughput-optimal; vanilla vLLM/FasterTransformer are not |
| 11 | [[2603.18897\|PASTE]] | Agentic | Speculative tool execution in parallel with LLM generation + joint scheduling | −43.5% task completion time |
| 12 | [[2502.07115\|Online scheduling with KV constraints]] | Theory + algorithm | Hindsight-optimal integer program + online algorithm with guarantees | Beats heuristics on Llama2-70B traces |

**Also useful.**
* Prefix-aware scheduling: [[2502.04677]] (NP-hardness under RadixAttention), [[2501.14312]] (locality-aware fair
  scheduling), [[2603.15202]].
* Load balancing across instances: [[2508.03611|Astrolabe]], [[2505.24095|SkyWalker]] (cross-region),
  [[2601.17855]].
* Multi-LoRA: [[2512.20210|P-LoRA]], [[2511.22880]].
* Co-location and autoscaling: [[2501.14808|HyGen]] (online + offline co-location), [[2501.08090|Chiron]].
* Streaming: [[2510.02758|TokenFlow]].
* Reasoning-model serving study: [[2510.18672]].
* Code interpreters: [[2604.00491]] (execute as you generate).
* Queueing theory: [[2605.04595]], [[2508.01002]].

**Runtime checklist.**
* Program/session IDs in the API.
* **KV pinning with TTL** across tool calls, and prefix-aware routing across replicas.
* Admission control and deadline-aware ordering per SLO class.
* A length-distribution predictor (not a point estimate).
* A concurrency limiter for agentic batch jobs.
* A prefill-only fast path.
* Hooks to run tool calls speculatively.
