**Verdict.** Scaling-law work moved from "N vs D" to **everything else**: hyperparameters, data mixtures, sparsity,
depth, inference cost, and new axes of compute. What a model builder should take from 2025–26:

1. **Hyperparameters are predictable. Stop sweeping at scale.**
   * [[2503.04715|Step Law]] (StepFun): optimal LR and batch size as power laws in N and D, with a broad convex optimum.
   * [[2505.13738|Power Lines]] (Cerebras): optimal weight decay (via the AdamW timescale), B_opt and B_crit are power
     laws in **D, independent of N**.
   * [[2505.01618|CompleteP]] (NeurIPS'25): depth-wise parameterization that transfers hyperparameters over depth *and*
     keeps all layers non-lazy; 12–34% compute savings.
   * Full-stack transfer across modules, width, depth, batch and duration: [[2512.22382]].
   * Norm-based view: [[2510.03871|Optimal scaling needs optimal norm]].
2. **Data dominates downstream behaviour.** [[2502.12120|LLMs on the Line]] (ICML'25): loss-to-loss scaling is set by
   the pretraining data; architecture and tokenizer matter little. [[2507.09404|Scaling laws for optimal data
   mixtures]] (Apple) predicts optimal domain weights from small runs, for LLMs, native multimodal models and LVMs.
3. **New compute axes.**
   * [[2505.10475|Parallel Scaling Law (ParScale)]]: P parallel streams with learnable transforms ≈ scaling parameters
     by O(log P), at much lower memory/latency cost than parameter scaling.
   * Looped MoE: [[2609.01343|SMELT]] saves 6.8–18% FLOPs on the compute-optimal frontier.
   * DiLoCo scales predictably ([[2503.09799]]).
4. **Inference-aware scaling.** Include inference cost in the objective:
   * [[2501.18107|Scaling inference-efficient LMs]]: wider and shallower models; Morph-1B is 1.8× faster at equal
     accuracy.
   * [[2510.18245|Conditional scaling law over architecture]]: +2.1% accuracy and +42% throughput vs Llama-3.2 at equal
     budget.
5. **Data-constrained regime.** [[2509.14786|Pre-training under infinite compute]]: with fixed data, heavy
   regularization + epoching + **ensembles** lower the loss asymptote; 17.5× data efficiency. Diffusion LMs also win
   here (see decoding/diffusion-language-models).
6. **MoE laws**: [[2501.12370]], [[2502.05172]], [[2507.17702]], [[2608.10605]] (see
   [`mixture-of-experts/architecture-and-routing`](../../mixture-of-experts/architecture-and-routing/README.md)).
   **Native multimodal**: [[2504.07951]] (ICCV'25) finds early fusion ≥ late fusion, and MoE helps.

### Hand ranking

| # | Paper | Axis | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2503.04715\|Step Law: optimal hyperparameter scaling]] | Hyperparameters | ~3,700 runs → LR*(N,D), B*(N,D) power laws; convex, broad optimum | Plug-and-play LR/batch for any N, D, data recipe |
| 2 | [[2505.01618\|CompleteP]] (NeurIPS'25) | Parameterization | Depth-µP with α=1 residual scaling → hyperparameter transfer over depth + non-lazy learning | 12–34% compute savings; flexible width/depth |
| 3 | [[2505.13738\|Power Lines]] | Weight decay + batch | Optimal AdamW timescale is constant in tokens; B_opt and B_crit ∝ D^α independent of N | Predict λ_opt and batch size before a big run |
| 4 | [[2507.09404\|Scaling laws for optimal data mixtures]] (NeurIPS'25) | Data | Loss as a function of (N, D, domain weights) fitted from small runs | Optimal mixtures for LLM, NMM and LVM pretraining; extrapolates |
| 5 | [[2505.10475\|Parallel Scaling Law]] (NeurIPS'25) | New axis | P learned input transforms + parallel forward passes + learned aggregation | P streams ≈ O(log P)× parameters, with far less memory and latency growth |
| 6 | [[2502.12120\|LLMs on the Line]] (ICML'25) | Data vs architecture | Loss-to-loss scaling across 6K+ checkpoints | Data determines transfer; architecture can be chosen for efficiency |
| 7 | [[2509.14786\|Pre-training under infinite compute]] | Data-constrained | Regularization + epoching + ensembling, judged by loss asymptote | 17.5× data efficiency on math mid-training |
| 8 | [[2510.18245\|Scaling laws meet architecture]] | Inference-aware | Conditional scaling law over hidden size, MLP ratio, GQA | +42% throughput and +2.1% accuracy vs Llama-3.2 |
| 9 | [[2504.07951\|Scaling laws for native multimodal models]] (ICCV'25) | Multimodal | 457 models: early vs late fusion | Early fusion wins at small scale, simpler to deploy; MoE learns modality-specific weights |
| 10 | [[2503.12811\|Multi-power law for LR schedules]] (ICLR'25) | Schedules | Predict whole loss curves across LR schedules from a few fits | Optimized schedule beats WSD slightly |
| 11 | [[2503.09799\|Scaling laws for DiLoCo]] | Distributed | DiLoCo scaling with replicas and model size | Scales better than DP with model size when tuned |
| 12 | [[2502.06042\|Forgetting scaling laws]] | Fine-tuning | Forgetting vs pretraining-data injection | **Injecting 1% pretraining data prevents forgetting** |

**Also useful.**
* Evaluation methodology: [[2508.13144|Signal and Noise]] (pick low-noise benchmarks for scaling decisions),
  [[2502.18969|(Mis)Fitting survey]], [[2502.06857|Gemstones]] (sensitivity of scaling prescriptions).
* Skills: [[2503.10061]] (knowledge vs reasoning scale differently).
* Code: [[2512.13472]].
* Compression laws: [[2502.16440]], [[2504.04342]].
* Theory: [[2505.10465]] (superposition → robust scaling), [[2507.02119]] (supercollapse),
  [[2503.04725|L²M]] (long-context mutual information).
* Refined laws: [[2506.10972|Farseer]], [[2608.07222|Skaling]].
* Batch schedules: [[2602.14208]].
* MoE hyperparameters: [[2608.20061]], [[2605.14200]], [[2609.08690]].

**Recommendation.**
* Set LR/batch/weight decay from Step Law + Power Lines; use CompleteP/µP for width and depth.
* Choose the data mixture with a fitted mixture law.
* Add an inference-cost term (wider/shallower, GQA ratio) before fixing the architecture.
* For data-limited domains, prefer many epochs with strong regularization or ensembles, or a diffusion objective.
