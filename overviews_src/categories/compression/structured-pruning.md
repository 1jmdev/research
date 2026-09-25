**Verdict.** Structured pruning (width: heads/channels/SSM groups; depth: layers/blocks) gives **real, kernel-free speedups**.
It only works well as **prune → distill** with a meaningful token budget. The winning industrial recipe is NVIDIA's
**Minitron / Puzzle** line:
* activation-based importance for width (heads, FFN channels, embedding dim, Mamba groups) plus depth;
* then **logit distillation** from the parent on ~1–10% of pretraining tokens;
* for MoE/hybrids, NAS-style per-block choices (Puzzle).

This produces production models (Nemotron-H 8B → 4B with up to 40× fewer tokens than training from scratch;
Puzzle-75B-A9B from a larger hybrid MoE). DarwinLM shows the pruning search should be **training-aware**.

**The big 2025–26 caveat: training-free pruning breaks generation and reasoning.**
* [[2603.24652|Demystifying When Pruning Works]] (ICML'26) explains why: logit perturbations are amplified by softmax
  and accumulate over decode steps. Multiple-choice and retrieval survive; generation does not.
* Removing even 1–2 layers can collapse test-time scaling on long reasoning ([[2510.22228]], [[2602.01997]]).
  Plain SFT does not repair it.
* Recovery must be **on-policy**: [[2607.13124|ShortOPD]] uses short-to-long on-policy distillation from the unpruned
  teacher and gets 1.6–4.4× over SFT/KD/SeqKD recovery.
* When the token budget is small, pruning a big parent beats training a small model from scratch
  ([[2606.14150]]).

**Phase-specific pruning is a new runtime angle.** [[2602.03295|POP]] prunes for prefill only, using independent KV
projections to keep the cache consistent: 1.37× faster prefill. [[2509.04467|PDTrim]] removes different blocks for the
prefill and decode instances of a PD-disaggregated deployment. [[2503.18908|FFN Fusion]] runs consecutive FFNs **in
parallel**, as used in Llama-Nemotron Ultra 253B: 1.71× latency.

### Cost table

| Method | Setting | Budget | H100-h | Basis |
| --- | --- | --- | ---: | --- |
| [[2505.02819\|ReplaceMe]] (NeurIPS'25) | Depth prune 25% + linear replacement, training-free | calibration only | **<1** | reported (no healing) |
| [[2505.24680\|LinearPatch]] (NeurIPS'25) | Layer pruning + Hadamard/scaling patch | 5K samples, 30 min on 1 GPU | **~0.5** | reported |
| [[2502.07780\|DarwinLM]] (COLM'26) | Evolutionary training-aware search + post-training | 5× less data than Sheared-LLaMA | **~520** | reported: 40×H100 × 13 h |
| [[2607.13124\|ShortOPD]] | Recovery by on-policy distillation | 8.5 h vs 35.9 h for fixed-horizon OPD | **tens–hundreds** | reported wall-clock (node) |
| [[2504.11409\|Minitron-SSM]] | Nemotron-H 8B → 4B (width + depth + SSM groups) + KD | up to 40× fewer tokens than from scratch | **~2–7K** | estimate (≈150–400B tokens at 4B) |
| [[2607.04371\|Puzzle-75B-A9B]] | Hybrid MoE NAS-pruning + distillation | undisclosed | large | production |

### Hand ranking

| # | Paper | Type | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.11409\|Minitron-SSM]] (NVIDIA) | Width + depth + SSM + KD | Group-aware Mamba pruning (keeps SSM head structure) + Minitron KD retraining | Nemotron-H 8B → 4B, beats similarly sized models, **2× faster**, up to 40× fewer tokens |
| 2 | [[2502.07780\|DarwinLM]] (COLM'26) | Training-aware search | Evolutionary search over non-uniform structured sparsity; offspring get short, increasing training before selection | State of the art on Llama-2-7B / Llama-3.1-8B / Qwen-2.5-14B; beats Sheared-LLaMA with 5× less data |
| 3 | [[2503.18908\|FFN Fusion]] (NVIDIA) | Depth parallelization | Identify runs of FFNs with low inter-dependency and execute them **in parallel** (fewer sequential steps) | Ultra-253B-Base from 405B: 1.71× latency, 35× lower per-token cost; complements quantization |
| 4 | [[2607.13124\|ShortOPD]] | Recovery | On-policy distillation from the dense teacher with short-to-long rollout horizons | ~9× the unrecovered score; 1.6–4.4× standard recovery; ¼ the time of fixed-horizon OPD |
| 5 | [[2603.24652\|Demystifying When Pruning Works]] (ICML'26) | Analysis | Embedding/logit spaces robust, probability space fragile → generation errors compound | Explains why perplexity/MC results mislead |
| 6 | [[2607.04371\|Nemotron-Labs-3-Puzzle-75B-A9B]] | MoE/hybrid NAS pruning | Puzzle-style block-level choices incl. expert and Mamba pruning | Deployment-optimized hybrid MoE that keeps parent accuracy |
| 7 | [[2505.02819\|ReplaceMe]] (NeurIPS'25) | Training-free depth pruning | Replace pruned blocks with an estimated linear transform merged into the previous layer | ~90% performance at 25% depth pruning with no healing |
| 8 | [[2602.03295\|POP: Prefill-Only Pruning]] | Stage-aware | Prune only for prefill; full model for decode; independent KV projections for cache integrity | Up to 1.37× prefill speedup at minimal loss (Llama-3.1, Qwen3-VL, Gemma-3) |
| 9 | [[2510.22228\|Layer pruning harms test-time scaling]] | Analysis | Long-CoT performance collapses after removing 1–2 layers; SFT recovery fails | **Do not depth-prune reasoning models without on-policy recovery** |
| 10 | [[2505.20155\|Pangu Light]] | Width + depth + re-init | Weight re-initialization after pruning (CLAP / SLNP) + Post-RMSNorm absorption | Pangu Light-32B beats Qwen3-32B in accuracy and throughput on Ascend |
| 11 | [[2510.14444\|A Free Lunch: retraining after pruning]] | Recovery study | Reconstruct at the block/sub-block level, not per matrix; simple criteria become competitive with scale | Cheap post-pruning adaptation is practical |
| 12 | [[2606.14150\|Small LLMs: pruning vs training from scratch]] | Study | Compare pruned parents vs scratch at matched tokens | With limited tokens, pruning wins; fine-grained pruning keeps its advantage |

**Also useful.**
* Training-free width pruning: [[2505.22689|SlimLLM]] (ICML'25), [[2509.14230|NIRVANA]], [[2501.17771|2SSP]],
  [[2503.09657|Týr-the-Pruner]] (NeurIPS'25), [[2504.21174|AMP]].
* Dynamic / input-aware pruning: [[2502.15618|Probe Pruning]] (ICLR'25), [[2501.02086|Instruction-Following Pruning]]
  (ICML'25), [[2502.04348|PuDDing]] (ICML'25), [[2503.07605|SEAP]], [[2506.04179|SkipGPT]], [[2607.28418|WIDE]].
* Depth pruning: [[2507.18212|Prune&Comp]], [[2502.19159|sliding-window merging]], [[2510.15304]], [[2506.20480|GPTailor]].
* Reasoning-aware pruning: [[2601.18091]], [[2604.25098]], [[2511.18864]], [[2512.02185]].
* Head pruning via sinks: [[2601.06787|BOS sink heads]].
* Recovery: [[2502.12594|PASER]], [[2609.06974|OverRep]].
* Language case study: [[2603.11881|Bielik-Minitron-7B]].
* SSM pruning: [[2502.18886]], [[2501.17088|Mamba-Shedder]], [[2506.09613|SparseSSM]].

**Recommendation.**
* *Model builders.* To make a smaller sibling, use Minitron-style width-first pruning of the big model + logit KD on
  **≥50–100B tokens**, then on-policy distillation for reasoning. Avoid depth pruning of reasoning models.
* *Runtime builders.* Structured pruning needs no special kernels, but **non-uniform layer shapes** (per-layer head and
  FFN counts, Puzzle-style heterogeneous blocks) must be supported. Phase-specific weights (POP/PDTrim) fit naturally
  into PD disaggregation. FFN Fusion needs a graph-level "parallel FFN" op.
