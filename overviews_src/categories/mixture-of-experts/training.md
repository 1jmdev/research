**Verdict.** Training MoEs well is now mostly **systems + three algorithmic details**.

**The algorithmic details:**
1. **Balance loss over the global batch, not the micro-batch.** Micro-batch balancing forces uniform routing *within a
   sequence* and kills domain specialization ([[2501.11873|Demons in the Detail]], Qwen). Combine with DeepSeek-style
   aux-loss-free bias updates (theory in [[2512.03915]]).
2. **Give the router dense gradients.** [[2504.12463|Default MoE]] (NeurIPS'25) substitutes an EMA of each expert's
   output for non-selected experts, so the router learns from all experts at ~no extra cost.
3. **Hyperparameter transfer across width, depth, expert count and expert size:** µP for MoE ([[2508.09752]]),
   DMFT-justified parameterization ([[2601.20205]], ICML'26), [[2605.14200]] (see training/scaling-laws).

**The systems side:**
* **Kernels**:
  * [[2512.14080|SonicMoE]]: IO- and tile-aware forward/backward, minimal activation caching, token rounding to tile
    multiples;
  * [[2604.19241|UniEP]] megakernel;
  * [[2601.05296|MoEBlaze]] (memory wall).
* **Parallelism and communication**:
  * [[2505.11432|MegaScale-MoE]]: 352B MoE on 1,440 Hopper GPUs at 1.41M tok/s, 1.88× Megatron-LM;
  * Megatron-Core MoE and MoE Parallel Folding (see [`serving-systems/distributed-inference-and-parallelism`](../../serving-systems/parallelism-and-distributed/README.md));
  * load-adaptive expert re-layout: [[2602.11686|LAER-MoE]], [[2504.19925|SYMI]], [[2502.02581|Themis]];
  * all-to-all load balancing: [[2510.19262|RailS]].
* **Memory**:
  * [[2607.19058|tiered optimizer state]]: factored second moment for experts, exact for backbone and router; 2.6% of
    AdamW's state;
  * [[2609.14306]]: flatten every memory peak in long-context MoE training.
* **RL on MoE** is its own problem: routing differs between the inference engine and the trainer, which destabilizes
  RL.
  * [[2510.11370|R3: rollout routing replay]] replays the inference routes in training.
  * Load-balance RL stages with routing foresight: [[2606.11867|ForeMoE]], [[2605.08639|ReLibra]].
* **Growth / reuse**: [[2510.08008|orthogonal MoE growth]] recycles checkpoints (layer copying + noisy expert
  duplication); +10.6% accuracy vs from scratch at equal extra compute, up to 70B.
* **Train for inference locality**: [[2607.08780|Sticky Routing]], [[2608.18261]] (cacheable routers) and
  [[2509.26520|Matryoshka MoE]] (elastic top-k at inference).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2501.11873\|Demons in the Detail: global-batch LBL]] (Qwen) | Algorithm | Compute load-balancing loss on global-batch expert frequencies (sync across DP) | Better perplexity, downstream and **domain specialization** up to 42.8B / 400B tokens. Now standard |
| 2 | [[2512.14080\|SonicMoE]] | Kernels | Memory-efficient forward/backward with minimal cached activations + IO-overlapped kernels + tile-aware token rounding | Faster than DeepGEMM baseline on 7B MoE; +1.16× from token rounding at high sparsity |
| 3 | [[2505.11432\|MegaScale-MoE]] (ByteDance) | System | Per-module parallelism for attention vs FFN, inter/intra-operator overlap, low-precision communication | 1.88× over Megatron-LM (352B MoE, 1,440 GPUs) |
| 4 | [[2510.11370\|R3: routing replay for MoE RL]] | RL stability | Record inference-engine routes, replay in training | Removes training–inference policy KL spikes; beats GSPO/TIS stabilization |
| 5 | [[2504.12463\|Default MoE: dense backprop]] (NeurIPS'25) | Algorithm | EMA "default" outputs for unselected experts → dense router gradient | Beats TopK routing with negligible overhead |
| 6 | [[2601.20205\|HP transfer with MoE layers]] (ICML'26) | Parameterization | DMFT-derived scaling over width, depth, number of experts and expert size | Reliable HP transfer 51M → 2B+ and to longer horizons |
| 7 | [[2510.08008\|Beyond Sunk Costs: orthogonal MoE growth]] | Reuse | Grow depth (interpositional copying) and width (noisy expert duplication) from checkpoints | +10.6% vs from scratch at equal extra compute (to 70B, 1T tokens) |
| 8 | [[2607.19058\|Tiered optimizer state]] | Memory | Different optimizer-state fidelity for backbone, experts and router | State at 2.6% of AdamW's; 81.4 → 31.3 GB peak |
| 9 | [[2505.20225\|FLAME-MoE]] | Open platform | Fully open MoE suite (64 experts, top-8, 2 shared), logs, checkpoints, routing traces | +3.4 pts over dense at equal FLOPs; research testbed |
| 10 | [[2602.11686\|LAER-MoE]] | System | Fully sharded expert parameters restored per device on the fly → re-layout for balance each step | Load-adaptive expert placement without migration stalls |

**Also useful.**
* Systems: [[2501.10714|FSMoE]], [[2510.00207|FlowMoE]], [[2508.13337|X-MoE]] (Frontier supercomputer),
  [[2504.03871|HeterMoE]] (mixed GPU generations), [[2508.09591|HierMoE]], [[2608.03676|TAOT]], [[2607.01844]].
* Routing training: [[2512.13996|DTop-p]] (PI-controlled dynamic top-p), [[2505.13380|CompeteSMoE]],
  [[2605.30992]] (eigenvector routers), [[2604.04230]] (three phases of routing balance).
* Continual pretraining: [[2503.05029]] (router robustness).
* Modular post-training: [[2604.18473]] (train separately, merge as MoE).
* Hardware: [[2501.03905|MixNet]] (optical fabric), [[2603.07006|Mozart]] (wafer-scale chiplets).

**Recommendation.**
* Global-batch balancing + aux-free bias + a small sequence-level balance term; dense-gradient router trick.
* µP-style HP transfer from small proxies.
* FP8 grouped GEMM (see [`quantization/low-precision-training`](../../quantization/low-precision-training/README.md)).
* DeepEP-style overlapped all-to-all.
* **Routing replay** for any RL stage.
* If the model will be served on memory-limited hardware, add a routing-locality objective during training.
