**Verdict.** Prefill–decode (PD) disaggregation (DistServe / Splitwise / Mooncake, 2024) is now the default for large
deployments. The 2025–26 work refines **where the boundary goes** and **when not to disaggregate**:

1. **Disaggregate further: attention vs FFN (AFD).** [[2507.19427|Step-3]] co-designs the model (MFA attention, 38B
   active) with AFD. It reaches **4,039 tok/s/GPU at 50 ms TPOT** on Hopper vs DeepSeek-V3's 2,324 in the same setup.
   Provisioning theory in [[2601.21351]]; design-space limits in [[2605.28302]] and the MoE AFD notes.
2. **Disaggregate less: hybrid/elastic.** A fixed split wastes capacity when TTFT/TPOT SLOs or loads shift:
   * [[2508.01989|TaiChi]]: unify aggregation and disaggregation with differentiated instances, up to +77% goodput;
   * [[2504.19867|semi-PD]]: disaggregated compute, unified storage, so no KV migration;
   * intra-GPU multiplexing: [[2504.14489|MuxWise]] (2.2× goodput), [[2507.06608|Nexus]],
     [[2511.04791|DuetServe]] (SM partitioning only when needed);
   * [[2504.09285|DynaServe]]: split a request at any token boundary.
3. **Multi-turn and agentic changes the prefill picture.** *Append-prefill* (reusing cached KV) is 10× less disruptive
   than full prefill, so run it on the decode node ([[2603.13358|PPD]], −68% turn-2+ TTFT). Schedule by conversation, not
   turn ([[2606.01839|ConServe]]).
4. **KV transfer is the tax.**
   * Selective or mixed-precision KV transfer: [[2607.28150|SmartGen]], [[2606.08635|SpectrumKV]].
   * Compute directly on compressed KV: [[2502.03589|HACK]], up to −70.9% JCT.
   * **Cross-datacenter prefill** is viable for hybrid models with small KV ([[2604.15039|Prefill-as-a-Service]]: 1T
     hybrid, +54% throughput, −64% P90 TTFT).
5. **Autoscaling the pools together.**
   * [[2508.19559|HeteroScale]] (ByteDance, tens of thousands of GPUs): one robust metric jointly scales prefill and
     decode pools; +26.6 pts GPU utilization, hundreds of thousands of GPU-h saved daily.
   * [[2512.03416|TokenScale]]: "token velocity" as a leading indicator + convertible decoders.
6. **Multimodal EPD** (encode–prefill–decode): [[2501.05460|EPD disaggregation]] (ICML'25) and
   [[2602.02204|vLLM-Omni]] for any-to-any models (−91% JCT).

### Hand ranking

| # | Paper | Axis | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2507.19427\|Step-3: model-system co-design]] (StepFun) | AFD + architecture | Multi-matrix factorization attention + attention–FFN disaggregation designed together for decode cost | 4,039 tok/s/GPU at 50 ms TPOT (4K, FP8, no MTP) vs 2,324 for DeepSeek-V3 |
| 2 | [[2508.19559\|HeteroScale]] (ByteDance) | Autoscaling | Production study of autoscaling signals → a single metric that scales P and D pools in balance on heterogeneous GPUs | +26.6 pts utilization across tens of thousands of GPUs |
| 3 | [[2604.15039\|Prefill-as-a-Service]] | Cross-datacenter PD | Externalize prefill for small-KV hybrid models with congestion-aware scheduling | 1T hybrid: +54% throughput, −64% P90 TTFT; ~15% gain at equal cost |
| 4 | [[2508.01989\|TaiChi]] | Hybrid PD | Differentiated prefill-heavy/decode-heavy instances + latency-shifting schedulers; aggregation ↔ disaggregation by SLO | Up to +77% goodput |
| 5 | [[2504.14489\|MuxWise]] | Intra-GPU multiplexing | Bubble-less prefill/decode multiplexing + contention-tolerant estimator + SLO-aware dispatch | 2.20× average (3.06× max) peak goodput |
| 6 | [[2603.13358\|PPD: not all prefills are equal]] | Multi-turn | Route append-prefills to decode nodes; full prefills to prefill nodes | −68% turn-2+ TTFT; relieves KV transfer congestion |
| 7 | [[2504.19867\|semi-PD]] | Architecture | Phase-wise disaggregated *compute* over unified *storage* (no KV copy) + SLO-aware partitioning | 1.27–2.58× lower latency (DeepSeek); 1.55–1.72× more SLO-compliant requests |
| 8 | [[2512.03416\|TokenScale]] | Autoscaling | Token velocity metric + convertible decoders that absorb prefill bursts | SLO attainment 50–88% → 80–96% at 4–14% lower cost |
| 9 | [[2503.20552\|Adrenaline]] | Attention offload | Offload part of decode attention to under-used prefill GPUs | 1.68× throughput |
| 10 | [[2501.05460\|EPD disaggregation]] (ICML'25) | Multimodal | Separate encode from prefill; cache multimodal tokens; intra-request encoder parallelism | Up to 22× larger batches; −71% TTFT |
| 11 | [[2602.02204\|vLLM-Omni]] | Any-to-any | Stage graph where each stage (LLM or diffusion) is served independently with connectors | Up to −91.4% JCT |
| 12 | [[2502.03589\|HACK]] | KV transfer | Homomorphic compute on quantized KV (skip dequantization) | Up to −70.9% JCT vs a disaggregated baseline |

**Also useful.**
* Scheduling theory: [[2508.06133]] (variable prefill/decode lengths; NP-hard; Sorted-F),
  [[2602.02987]] (asymptotically optimal), [[2606.17081]] (price of anarchy between pools).
* Heterogeneous hardware: [[2502.07903|HexGen-2]], [[2606.29986]] (memory-heterogeneous accelerators), [[2603.12707]].
* Hybrid Mamba models: [[2603.15530|DUET]].
* KV state transfer: [[2606.07684]] (semantic cache distillation).
* Pipelines: [[2506.10470|TD-Pipe]].
* Multi-round: [[2602.14516|AMPD]] (ICML'26).
* Multimodal: [[2505.12658|HydraInfer]], [[2509.24381|RServe]].

**Runtime checklist.**
* Support aggregated, disaggregated and intra-GPU-multiplexed modes, switchable by SLO.
* Route append-prefill locally.
* Stream KV layer by layer, mixed-precision and selective.
* Keep the KV pool shared with prefix caching (Mooncake-style store).
* Autoscale P and D pools jointly from one leading indicator.
* For MoE at scale, add AFD with ping-pong micro-batches.
