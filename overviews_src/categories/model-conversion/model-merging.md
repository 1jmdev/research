**Verdict.** This is a big, noisy area (100+ papers). Most "new merging operator" papers report small gains on ViT/T5
task-vector benchmarks that don't transfer to modern LLMs ([[2511.21437|in-the-wild study]]). What actually matters for
someone building models:

1. **Merging checkpoints inside pretraining is the highest-ROI use.** [[2505.12082|Model Merging in Pre-training]]
   (PMA, ByteDance Seed; up to 100B+ MoE) shows that averaging constant-LR checkpoints **reproduces the gains of LR
   annealing** and predicts annealed performance, which cuts ablation cost. [[2507.17634|WSM]] goes further: a
   decay-free LR schedule where checkpoint merging *is* the decay.
2. **Weighted souping of post-trained variants gets state-of-the-art results almost for free.**
   [[2511.13254|Souper-Model / SoCE]] (Meta) picks a category expert per weakly-correlated benchmark cluster and uses
   optimized weights instead of a uniform average. It is state of the art on BFCL.
3. **Capability transfer by task vectors between siblings.** Examples:
   * [[2509.01363|Reasoning Vectors]]: θ_GRPO − θ_SFT added to another model;
   * [[2502.09056|Typhoon2-R1 one-day recipe]]: language-specific model + R1-distilled reasoning model merged for $120;
   * long-to-short reasoning merges ([[2503.20641]]) that shorten CoT;
   * [[2505.22697|re-basin task vectors]] (ICML'25): carry a fine-tune to a *new base release* without training.
4. **Principled interference handling, when merging many experts:**
   * spectral/subspace methods: [[2502.04959|Iso-C/Iso-CTS]] (ICML'25), [[2502.10339|STAR]], [[2603.06242|DC-Merge]];
   * data-free least-squares on task vectors: [[2503.08099|WUDI]] (ICML'25);
   * activation-informed methods: [[2502.02421|AIM]], [[2505.14009|ACM]];
   * LoRA-specific: [[2509.17786|Core Space]], [[2505.22934|OSRM]].

   [[2509.24244|Merging scaling laws]] (ICML'26) shows a size-dependent floor plus **diminishing returns in the number
   of experts**, so plan how many experts to merge.

**Cost.** Weight-space merges take **minutes of CPU/GPU**. Evolutionary/Bayesian searches cost 10–1,000 GPU-h of
*evaluation* ([[2505.11427|Mergenetic]], [[2502.10436|MERGE³]] on consumer GPUs, [[2605.14386|Darwin]]). PMA in
pretraining costs nothing extra; it replaces annealing runs.

### Hand ranking

| # | Paper | Setting | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.12082\|PMA: Model Merging in Pre-training]] (NeurIPS'25) | Pretraining, dense + MoE to >100B | Average checkpoints from the constant-LR phase (SMA/EMA/WMA studied) | ≈ annealing gains without annealing; predicts final performance; stabilizes spikes (PMA-init). **Adopt in every pretraining run** |
| 2 | [[2511.13254\|Souper-Model (SoCE)]] | Post-training soups | Category-aware expert selection + optimized non-uniform weights | State of the art on BFCL; robust gains across domains |
| 3 | [[2502.04959\|Iso-C / Iso-CTS]] (ICML'25) | Multi-task merging | **Flatten the singular spectrum** of the merged task matrix (isotropic) + common/task-specific subspaces | State of the art across task counts and scales |
| 4 | [[2503.08099\|WUDI merging]] (ICML'25) | Data-free | Task vectors span the input subspace of linear layers → solve interference by least squares using the task vectors themselves | +10.9% over data-free baselines; beats test-time adaptation |
| 5 | [[2509.24244\|Model merging scaling laws]] (ICML'26) | Planning | Power law in model size and number of experts; holds across methods | Choose the number of experts vs base size under a budget |
| 6 | [[2509.01363\|Reasoning Vectors]] | Capability transfer | θ_GRPO − θ_SFT from identically initialized Qwen2.5 → add to other models | +GSM8K/HumanEval gains; subtracting it drops GSM8K 11.8% |
| 7 | [[2502.09056\|Language-specific → reasoning in one day]] | Recipe | Data selection + merge a language-specific LLM with an R1-distilled model | R1-level reasoning in Thai for **$120 of compute** |
| 8 | [[2509.17786\|Core Space merging]] (NeurIPS'25) | LoRA merging | Merge LoRAs in a shared core basis (lossless projection) | State of the art at a fraction of full-matrix merge cost |
| 9 | [[2505.22697\|Re-basin of task vectors]] (ICML'25) | Base-model update | Permute heads (spectral) to move fine-tunes to a new base release, data-free | Keeps fine-tunes alive across base upgrades |
| 10 | [[2505.10833\|MergeBench]] (NeurIPS'25) | Benchmark | Domain-specialized LLM merging at scale (Llama/Gemma 2–9B) | Shows the gap to multi-task training; **use it for evaluation** |
| 11 | [[2502.02421\|AIM]] (NeurIPS'25) / [[2505.14009\|ACM]] | Activation-aware | Protect weights important in activation space; layer-wise coefficients from activation similarity | Plug-in gains for any merge method; ACM shortens reasoning (System 1/2) |
| 12 | [[2605.14386\|Darwin Family]] | Evolutionary | 14-dim merge genome + diagnostic layer-importance ("MRI-trust") weighting | Training-free improvements at 4–35B; merges Transformer + Mamba parts |

**Also useful.**
* Heterogeneous fusion (different architectures/vocabularies): [[2505.13893|InfiGFusion]], [[2604.01674]],
  [[2501.00061]].
* MoE-aware merging: [[2502.00997|MergeME]], [[2606.03391]] (merging breaks routing; training-free calibration).
* Merging as data-mixture search: [[2505.16066|Merge to Mix]], [[2602.04937]]. See also the training-data notes: OptiMer,
  MergeMix.
* Failure analyses: [[2506.14126]] (overtrained experts merge worse), [[2607.11997]], [[2603.09463]] (merging collapse),
  [[2605.25846]] (limits for multilinguality in pretraining).
* SFT + RLVR vector synthesis: [[2605.00610]]. Merging for DiLoCo aggregation: [[2607.03011]].
* Survey: [[2603.09938]].

**Recommendation for model builders.**
1. **Pretraining.** Keep constant-LR checkpoints and use PMA/WSM merging instead of (or before) annealing.
2. **Post-training.** Train domain/skill specialists from the same base, then do SoCE-style weighted souping or
   Iso-CTS/WUDI for many experts. Measure against a multi-task baseline on MergeBench.
3. **Capability grafting.** Use reasoning or long-to-short task vectors between siblings; re-basin when the base updates.
4. For **runtimes**, merging is offline. The relevant runtime feature is **multi-LoRA serving**, which lets you defer
   merging to request time.
