**Verdict.** Energy work splits into **measurement** (how many joules per token or query, and why) and **control** (DVFS,
placement, routing). The findings that matter for runtimes:

* **Decode is insensitive to GPU clock.** Decode is 77–91% of inference time and memory-bound, so dropping the SM
  clock saves a lot: [[2501.08219]] reports −42% energy for 1–6% more latency. Phase-aware DVFS is almost free money:
  * [[2508.16449|GreenLLM]]: −34% energy on Alibaba/Azure traces at no throughput loss;
  * [[2509.04827|VoltanaLLM]]: U-shaped energy–frequency curve; per-iteration frequency;
  * [[2602.18755|DualScale]] (PD-disaggregated);
  * [[2608.21719|PowerSlider]] (demand response with flexible SLOs).
* **Low precision saves energy only when compute-bound**; batching helps most in memory-bound phases ([[2601.22362]]).
  GPU choice matters per model ([[2604.09048|Watt Counts]]).
* **Measure the right unit.**
  * [[2511.07885|Intelligence per Watt]] (Stanford): task accuracy per watt for local models; local inference can
    absorb a large share of queries.
  * [[2504.13359|Cost-of-Pass]]: expected dollar cost of a *correct* answer. Reasoning models win on hard math despite
    higher per-token cost.
  * [[2606.11690]]: per-token calculators misprice self-hosting by 1/utilization.
* **Footprint baselines**: [[2505.09598|How Hungry is AI?]] (per-query energy, water and carbon across 30 models; the
  most energy-intensive models use >29 Wh per long prompt) and [[2503.05804]] (full lifecycle of training a model
  family; 493 t CO₂).
* **Test-time compute has an energy bill**: [[2505.14733]], [[2603.20224]] (energy-per-token), [[2609.19499]]
  (candidate-generation schedule matters).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2508.16449\|GreenLLM]] | Control | Length-based queues + separate prefill/decode DVFS with hysteresis to hold tail TBT | −34% energy vs default DVFS; no throughput loss |
| 2 | [[2501.08219\|Energy–performance trade-offs across GPU scaling]] | Measurement | Decode dominates and is frequency-insensitive | −42% energy for +1–6% latency via down-clocking |
| 3 | [[2511.07885\|Intelligence per Watt]] | Metric | Accuracy per watt across local model–accelerator pairs on real queries | Local LMs answer most real single-turn queries; large headroom in local accelerators |
| 4 | [[2504.13359\|Cost-of-Pass]] | Economics | Expected cost to obtain a correct answer; compare to human-expert cost | Picks the right model class per task; most inference-time tricks rarely pay for themselves |
| 5 | [[2509.04827\|VoltanaLLM]] | Control | Phase-specific iteration-level frequency + state-space request routing in PD serving | Exploits U-shaped energy–frequency sweet spots |
| 6 | [[2505.09598\|How Hungry is AI?]] | Footprint | Energy, water and carbon per query from public API performance + infrastructure factors | >65× spread between models; scaled footprints for reference |
| 7 | [[2512.03024\|TokenPowerBench]] | Benchmark | Phase-attributed power per request without special meters | Reproducible energy benchmarking |
| 8 | [[2502.05043\|EcoServe]] | Carbon | Carbon-aware provisioning using offline batch share and hardware heterogeneity | Lower operational + embodied carbon |
| 9 | [[2601.22362\|Quantization, batching and serving in LLM energy use]] | Measurement | When do lower precision and batching save energy on H100? | Precision helps only when compute-bound; batching helps memory-bound phases |
| 10 | [[2606.11690\|Concurrency-aware cost estimation]] | Economics | Little's law-based cost model with utilization | Per-token calculators understate self-hosting cost by 1/U |

**Also useful.**
* Energy models: [[2602.05695|SweetSpot]], [[2603.17280]] (the 1/W law), [[2507.11417]] (simulation + grid co-simulation).
* Diagnosis: [[2601.22076|Where do the joules go?]], [[2511.05597]].
* MoE power: [[2605.21427|PALS]].
* Mobile DVFS: [[2507.02135]].
* Edge VLMs: [[2607.09520]] (power is constant; time is what varies).
* Heterogeneous GPUs: [[2511.00807|FREESH]].
* Position: [[2605.11733]] (evaluate inference as energy-to-token production).

**Runtime checklist.**
* Per-phase DVFS: low clocks for decode, high for prefill, tuned against the TBT SLO.
* Energy counters per request (TokenPowerBench-style).
* Route easy queries to small/local models by cost-of-pass.
* Report **J/token and $/correct answer** next to tok/s.
