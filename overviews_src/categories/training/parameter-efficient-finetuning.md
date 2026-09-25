**Verdict.** The LoRA-variant zoo produced little that survives careful tuning.
[[2602.04998|Learning Rate Matters: Vanilla LoRA May Suffice]] finds that with per-method LR/batch/rank sweeps, most
variants land within noise of plain LoRA. They mainly prefer different learning rates. The live directions are
elsewhere:

1. **Hypernetworks that generate adapters (prompt → weights).**
   * [[2506.06105|Text-to-LoRA]] (Sakana, ICML'25): builds a LoRA from a task description in one forward pass.
   * [[2506.16406|Drag-and-Drop LLMs]]: prompts → LoRA in seconds; up to 12,000× cheaper than fine-tuning, +30% over
     training LoRAs on unseen tasks.
   * [[2602.06358|SHINE]]: context → LoRA, turning in-context knowledge into parameters.
   * [[2501.06252|Transformer²]] (ICLR'25): singular-value "expert vectors" mixed per prompt.
2. **Knowledge injection and continual learning.**
   * [[2502.14502|How much knowledge fits in a LoRA]]: mixing known and new facts helps, but external QA still degrades.
   * [[2510.15103|Sparse memory fine-tuning]] (Meta): update only memory-layer slots highly activated by the new
     knowledge, with far less forgetting than LoRA or full fine-tuning.
   * [[2605.30260|parametric memory law]] for LoRA recall.
3. **Efficiency of PEFT training itself:**
   * [[2506.16500|SparseLoRA]] (ICML'25): contextual sparsity in the frozen base; 2.2× less compute, 1.6× faster;
   * [[2502.13533|LoRAM]]: train on a pruned model, infer on the full one;
   * [[2510.00206|LoRAFusion]] and [[2603.22276|fused DoRA kernels]];
   * quantized/zeroth-order fine-tuning: [[2505.13430]], [[2502.08141|LowRA]] (<2-bit);
   * multi-LoRA concurrent training: [[2508.02932|PLoRA]].
4. **Serving-aware adapters.** [[2504.12397|Activated LoRA]] (IBM) applies the adapter only *after* its invocation
   point, so the base model's KV cache is reused: adapters as cheap "intrinsics". PEFT at million-adapter scale:
   [[2606.02437]] / MinT.
5. **Optimization geometry.** [[2507.12142|Riemannian LoRA with Muon]], [[2502.01235|LoRA-One]] (one full-gradient step
   init), [[2502.09376]] (LoRA converges to a low-rank minimum or fails loudly).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2602.04998\|Learning Rate Matters: Vanilla LoRA may suffice]] | Benchmark | Full hyperparameter sweeps for LoRA variants across tasks and scales | All variants within 1–2% of vanilla LoRA once LR is tuned. **Tune LR before trying variants** |
| 2 | [[2506.16406\|Drag-and-Drop LLMs]] (NeurIPS'25) | Prompt → weights | Condition a hyper-convolutional decoder on prompt embeddings to emit full LoRA matrices | Seconds per adapter; up to 30% over trained LoRAs zero-shot; 12,000× less overhead |
| 3 | [[2506.06105\|Text-to-LoRA]] (ICML'25) | Hypernetwork | Task description → LoRA in one forward pass | Matches task-specific adapters; generalizes to unseen tasks |
| 4 | [[2510.15103\|Sparse memory fine-tuning]] | Continual learning | Memory-layer model; update only slots specific to new data (TF-IDF vs pretraining usage) | Much less forgetting than full fine-tuning/LoRA at equal new-fact learning |
| 5 | [[2506.16500\|SparseLoRA]] (ICML'25) | Training speed | SVD-based contextual sparsity estimator skips base-weight channels in the forward pass | Up to 2.2× less compute, 1.6× faster at equal accuracy |
| 6 | [[2504.12397\|Activated LoRA]] | Serving | Adapter activates mid-sequence; reuse the base model's KV for the prefix | Instant switching between base and adapters without re-prefill |
| 7 | [[2501.06252\|Transformer²]] (ICLR'25) | Self-adaptive | RL-trained singular-value scaling vectors, mixed per prompt at inference | Beats LoRA with far fewer parameters |
| 8 | [[2502.14502\|Knowledge in a LoRA adapter]] | Knowledge injection | Measure how many new facts fit and what breaks | Known+new mixtures work best; general QA still regresses |
| 9 | [[2502.13533\|LoRAM: train small, infer large]] (ICLR'25) | Memory | Train LoRA on a pruned model, recover to the full model for inference | QLoRAM cuts parameter storage 15.8× for Llama-3.1-70B LoRA training and beats a LoRA-trained 8B |
| 10 | [[2507.12142\|LoRA meets Riemannion]] | Optimization | Parameterization-independent Riemannian Muon for low-rank factors | Consistent gains on LLM and diffusion fine-tuning |
| 11 | [[2502.16894\|GOAT]] | LoRA-MoE | SVD-structured MoE of LoRAs + scaling aligned with full fine-tuning | Closes much of the gap to full fine-tuning |
| 12 | [[2602.06358\|SHINE]] | Context → LoRA | In-context hypernetwork maps documents to adapters in one pass | Answers questions about a context without seeing it at inference |

**Also useful.**
* Rank and initialization: [[2502.12171|GoRA]] (NeurIPS'25), [[2503.02659]].
* Multi-task interference: [[2504.07448|LoRI]].
* MoE fine-tuning: [[2601.04823|DR-LoRA]], [[2603.24044|MoE-Sieve]].
* SSM PEFT: [[2503.03499|State-offset Tuning]].
* Benchmarks: [[2511.21285|PEFT-Bench]].
* Zeroth-order: [[2506.09034|FZOO]].
* Shadow networks: [[2604.19254|ShadowPEFT]].
* Surveys: [[2501.13787]], [[2501.00365]].

**Recommendation.**
* *Model builders:* plain LoRA (or full fine-tuning for RL-scale changes) with a properly tuned LR; high rank for
  knowledge, low rank suffices for RL.
* *Runtime builders:* multi-LoRA batching (Punica/S-LoRA-style kernels), **aLoRA-style KV reuse**, adapter paging,
  and hypernetwork-generated adapters as a new request type.
