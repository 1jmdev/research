**Verdict.** Token-level sampling research splits into three camps, and only one of them changes what a runtime ships.

1. **Truncation samplers** (top-k / top-p / min-p / top-nσ / top-H / min-k / p-less). Gains over well-tuned min-p or
   top-p are small and often within noise. [[2506.13681|Min-p, Max Exaggeration]] shows how easily they are overstated.
   The robust finding is **temperature-invariance**: logit-space truncations (top-nσ, [[2604.11012|Min-k]]) stay
   coherent at high temperature, where probability-space ones collapse. Ship a few, default to one, and expose the rest.
2. **Learned or adaptive decoding.** [[2510.26697|AutoDeco]] adds tiny heads that predict per-token temperature and top-p,
   which matches or beats hand-tuned settings. Selective sampling switches greedy ↔ sampling at "sensitive" positions.
   This is the direction that removes knobs from users.
3. **Sequence-level sampling as reasoning.** Sampling from the *power distribution* p(y|x)^α recovers much of RL's
   reasoning gain with **no training**. [[2510.14901|Reasoning with Sampling]] does it with MCMC (slow);
   [[2602.10273|Power-SMC]] makes it low-latency with SMC. Reward-guided SMC ([[2604.16453]]) is the same machinery. This is
   test-time compute, and it needs runtime support for particles, resampling and prefix-shared KV.

Contrastive/layer-contrastive decoding (DoLa lineage: LayerCake, ActLCD) improves factuality a little, but costs an extra
pass or intermediate-layer logits. It is niche for serving.

**Systems notes.** Sampling is a real kernel cost at 150K–260K vocabularies. Use sort-free top-k/top-p
([[2602.01518|Qrita]]: pivot-based selection), certified sub-vocabulary LM heads ([[2511.21702|CSV-Decode]]), and
KV sharing across beams via a trie ([[2502.00085]]). Also, the **sign-branched multiplicative repetition penalty** shipped by
HF/vLLM/llama.cpp is gauge-dependent and **corrupts structured output** ([[2607.09791]]); use additive/frequency
penalties instead.

### Hand ranking

| # | Paper | Kind | Key idea | Result / use |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.26697\|AutoDeco: The End of Manual Decoding]] | Learned decoding | Lightweight heads predict **token-level temperature and top-p** alongside logits; trained cheaply on a frozen model | Matches or beats expert-tuned static settings on 8 benchmarks; also steerable by instructions ("be less random"). **Ship-able** |
| 2 | [[2602.10273\|Power-SMC]] | Sequence-level sampling | SMC for p^α (α>1) instead of Metropolis–Hastings; low-latency particles | RL-like reasoning gains, training-free, at a fraction of MCMC latency. See [[2510.14901\|Reasoning with Sampling]] |
| 3 | [[2504.15266\|Roll the dice & look before you leap]] (ICML'25) | Analysis | NTP + temperature is myopic for "creative leap" tasks; teacherless/diffusion training and **seed-conditioning** (noise at the input) beat output-temperature randomness | Motivates MTP/diffusion and input-noise sampling for diversity |
| 4 | [[2509.02510\|Top-H]] (NeurIPS'25) | Truncation | Entropy-constrained mass maximization (NP-hard) → greedy approximation: keep tokens while entropy stays bounded | Up to +25% over min-p at high temperature on creative writing, with coherence kept |
| 5 | [[2604.11012\|Min-k sampling]] (ACL'26) | Truncation | Truncate by **relative logit dynamics** among top candidates; temperature-invariant, robust to long-tail noise | Decouples truncation from temperature; beats top-nσ |
| 6 | [[2510.01218\|Selective sampling]] | Adaptive | Learned "sampling risk" metric switches greedy ↔ high-temperature per position | Diversity without math accuracy loss |
| 7 | [[2602.01518\|Qrita]] | Kernel | Pivot-based top-k/top-p with Gaussian σ-truncation; no sort, deterministic | Faster than sort-based and exact, unlike stochastic approximations. **Runtime-relevant** |
| 8 | [[2511.21702\|CSV-Decode]] | LM-head efficiency | Offline clusters of vocab embeddings; centroid+radius bounds give certified sub-vocabularies each step | Exact top-k certification with sparse LM-head compute |
| 9 | [[2505.19371\|Foundations of top-k decoding]] (NeurIPS'25) | Theory | Decoding = recovering a sparse distribution; top-k as ℓ0-regularized Bregman projection; generalizations | Principled way to design new truncations |
| 10 | [[2502.00085\|Trie-based beam search]] | Systems | Beams share one KV cache through a prefix trie | Large memory savings for beam search (MHA, GQA, SWA) |
| 11 | [[2507.04404\|LayerCake]] / [[2505.23657\|ActLCD]] (EMNLP'25) | Contrastive | Token-type-aware layer-contrastive decoding; RL policy decides *when* to contrast | Factuality gains, training-free, extra compute |
| 12 | [[2506.13681\|Min-p, Max Exaggeration]] | Critique | Re-analysis of min-p's human and LLM-judge evidence | Claimed min-p gains largely do not hold. **Be skeptical of sampler papers** |
| 13 | [[2607.09791\|Sign-branched repetition penalties]] | Bug report | Multiplicative penalty flips direction with the sign of the logit (gauge-dependent) | Corrupts JSON across engines; prefer additive alternatives |

**Also useful.**
* Test-time search via decoding: [[2601.15296|Entropy-Tree]] (branch only at uncertain positions),
  [[2602.18232|Thinking by Subtraction]], [[2604.16453|reward-guided SMC]], [[2503.00029|streaming look-ahead]].
* Adaptive/learned: [[2603.09065|Learning Adaptive LLM Decoding]], [[2603.18428]], [[2508.20757|GUARD]],
  [[2507.03038|Cautious NTP]].
* Temperature science: [[2608.14665]] (the best pass@k temperature rises with budget), [[2609.15476|temperature fragility]],
  [[2606.13233|ReSET]] (step-aware temperature to rescue NVFP4 reasoning models).
* Cheaper contrastive decoding: [[2608.12913]], [[2601.21744|Temporal Guidance]].
* Consistency: [[2503.00831|recycled Gumbel noise]].

**For a runtime.**
* Fused sampling kernel: temperature → logit-space truncation (top-nσ / min-k) → probability-space (top-p / min-p), using
  sort-free selection.
* **Per-request, per-token parameters** so that AutoDeco-style models can drive them.
* **Seeded, reproducible Gumbel sampling**, deterministic across batch sizes.
* Additive repetition and frequency penalties.
* Particle/SMC primitives (fork, weight, resample) over **prefix-shared KV**. These serve power sampling, reward-guided
  SMC, beam search and Entropy-Tree with the same code.
