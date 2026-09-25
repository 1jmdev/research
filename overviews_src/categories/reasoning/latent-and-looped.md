**Verdict.** There are two routes to "reasoning without spelling out every token".

1. **Looped / recurrent-depth models** reuse a block many times, giving more compute per token without more parameters.
   This is the more mature route.
   * [[2502.05171|Huginn: recurrent depth]] (3.5B, 800B tokens): scales test-time compute in latent space up to the
     equivalent of 50B parameters.
   * [[2510.25741|Ouro / LoopLM]] (ByteDance): pretrained looped models (1.4B/2.6B, 7.7T tokens) match 12B-class LLMs.
     The gain comes from better *knowledge manipulation*, not capacity.
   * Stability and scaling laws: [[2604.12946|Parcae]] (spectral-norm-constrained injection; 1.3B reaches 87.5% of a
     2× Transformer) and [[2604.21106|iso-depth laws]] (what one recurrence is worth; hyperconnections help).
   * Theory: [[2502.17416|looped transformers simulate T CoT steps with T loops]] (ICLR'25).
   * Latency fixes: [[2510.24824|Parallel Loop Transformer]] (loops across tokens in parallel, near-zero extra latency)
     and [[2606.18023|LoopCoder-v2]] (saturation at two loops explained).
   * Retrofitting existing models: [[2511.07384]], [[2605.11011|LoopUS]].
   * Tiny recursive models for puzzles: [[2510.04871|TRM]] (7M parameters, 45% ARC-AGI-1), [[2512.14693|URM]].
2. **Continuous / latent chain-of-thought** replaces text thoughts with hidden vectors. Promising but fragile at scale.
   * Training-free soft tokens: [[2505.15778|Soft Thinking]] (probability-weighted embedding mixtures; +2.5% accuracy
     with −22% tokens), [[2510.05069|SwiReasoning]] (switch latent ↔ explicit by confidence),
     [[2601.08808|Multiplex Thinking]] (token-wise branch-and-merge trained with RL).
   * Trained: [[2502.21074|CODI]] (self-distillation into continuous space; first implicit CoT to match explicit CoT on
     GSM8K at GPT-2 scale), [[2509.20317|SIM-CoT]] (step-level supervision stabilizes latents),
     [[2505.16552|CoLaR]] (dynamic compression + RL).
   * Reality check: [[2509.19170|Soft Tokens, Hard Truths]] finds continuous-CoT RL helps diversity and OOD retention,
     but the best deployment is often *train soft, infer hard*. [[2503.07604]] shows implicit reasoning often uses
     shortcuts.
   * Multi-agent latent communication: [[2511.20639|LatentMAS]] (shared latent working memory; −70–84% tokens, 4×
     faster).

### Hand ranking

| # | Paper | Route | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.25741\|Ouro: Scaling Latent Reasoning via Looped LMs]] | Looped pretraining | Weight-shared looped blocks with learned early exit, trained on 7.7T tokens | 1.4B/2.6B LoopLMs match up to 12B standard LLMs |
| 2 | [[2502.05171\|Recurrent-depth latent reasoning (Huginn)]] | Looped | Prelude / recurrent core / coda; random loop counts in training | Test-time depth scaling up to ~50B-parameter-equivalent compute |
| 3 | [[2604.12946\|Parcae]] | Looped stability | Spectral-norm constraint on loop injection; looped scaling laws | 1.3B: +2.99 CORE vs Transformer; 87.5% of a 2× larger model |
| 4 | [[2510.24824\|Parallel Loop Transformer]] | Looped serving | Cross-loop parallelism + shared KV with gated sliding-window attention | Looped-model accuracy at almost standard latency and memory |
| 5 | [[2505.15778\|Soft Thinking]] | Latent CoT (training-free) | Concept tokens = probability-weighted embedding mixtures; cold-stop on entropy | +2.48 pass@1 with −22.4% tokens |
| 6 | [[2502.21074\|CODI]] (EMNLP'25) | Latent CoT (trained) | Teacher (explicit) / student (implicit) self-distillation on one token's hidden state | Implicit CoT matching explicit CoT on GSM8K (GPT-2); +28.2% over prior implicit methods |
| 7 | [[2502.17416\|Power of looped transformers]] (ICLR'25) | Theory | k-layer model looped L times ≈ kL-layer model on reasoning; simulates CoT | Justifies looped reasoning; regularization via looping |
| 8 | [[2601.08808\|Multiplex Thinking]] | Latent + RL | Token-wise branch-and-merge of K samples into one multiplex token; on-policy RL | Beats discrete CoT and RL baselines from pass@1 to pass@1024 with shorter outputs |
| 9 | [[2510.04871\|TRM: Tiny Recursive Model]] | Recursive small nets | 2-layer network recursively refining latent answer and state | 7M parameters: 45% ARC-AGI-1, 8% ARC-AGI-2 |
| 10 | [[2511.20639\|LatentMAS]] | Latent multi-agent | Agents exchange last-layer embeddings through a shared latent working memory | +14.6% accuracy, −70.8–83.7% tokens, 4× faster |
| 11 | [[2509.19170\|Soft Tokens, Hard Truths]] | Analysis | RL with continuous CoT at scale | Soft training + hard inference works best; better OOD preservation |
| 12 | [[2604.21106\|Iso-depth scaling laws for looped LMs]] | Scaling | Equivalent unique-parameter value of one recurrence | Truncated-BPTT under-trains loops; hyperconnections help |

**Also useful.**
* Pondering in pretraining: [[2505.20674|PonderLM]].
* Adaptive loops: [[2511.08577|Think-at-Hard]], [[2602.11451|LoopFormer]] (elastic depth), [[2606.04438|LoopMoE]].
* Latent test-time scaling: [[2510.07745]] (parallel TTS for latent models), [[2505.13308|LatentSeek]].
* Concept-level models: [[2512.24617|Dynamic Large Concept Models]].
* Theory: [[2505.12514]] (reasoning by superposition).
* Surveys: [[2505.16782]], [[2507.06203]].

**For runtimes.**
* Looped models need **per-request loop count / early-exit control**, KV handling for weight-tied blocks (shared vs
  per-loop KV; PLT shares across loops), and CUDA graphs per loop count.
* Soft-token decoding needs an "embedding-mixture input" path that bypasses the discrete token lookup.
* Latent multi-agent systems need hidden-state transfer APIs between sessions.
