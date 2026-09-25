**Verdict.** Outside attention (see the attention area) and MoE, the architecture changes that 2025–26 frontier labs
actually adopted are about **the residual stream, normalization, and memory**.

1. **Richer residual streams.**
   * [[2512.24880|mHC: manifold-constrained hyper-connections]] (DeepSeek) widens the residual into N streams and
     projects mixing matrices onto a manifold that restores identity-mapping stability. It trains stably at scale with
     efficient kernels.
   * [[2603.15031|Attention Residuals]] (Moonshot) lets each layer *attend over previous layers' outputs*; the block
     variant scales. Integrated into Kimi Linear 48B-A3B on 1.4T tokens, it fixes PreNorm dilution.
   * Also [[2502.12170|MUDDFormer]] (dense dynamic connections; 1.8–2.4× compute-equivalent),
     [[2601.00417|Deep Delta Learning]], [[2607.14530|xHC]], [[2502.09245|LIMe]].
   * Motivation: [[2505.13898|deep models don't use depth efficiently]] (NeurIPS'25). The second half of layers mostly
     refines.
2. **Normalization and the curse of depth.**
   * [[2502.05795|The Curse of Depth]] (NeurIPS'25): Pre-LN variance grows with depth so deep layers ≈ identity; fixed
     by LayerNorm scaling (1/√depth).
   * [[2503.10622|DyT: Transformers without normalization]] (Meta, CVPR'25): tanh(αx) replaces norms;
     [[2512.10938|Derf]] improves on it.
   * [[2503.04598|HybridNorm]], [[2502.02732|Peri-LN]], [[2601.19895|Post-LN is back]], [[2602.08064|SiameseNorm]].
3. **Memory as a new sparsity axis** (parameters that are looked up, not multiplied):
   * [[2601.07372|Engram: conditional memory via scalable lookup]] (DeepSeek): hashed n-gram embedding memory. At 27B
     iso-parameter/iso-FLOPs it beats MoE; bigger gains on *reasoning* than knowledge; host-memory prefetch because
     addressing is deterministic.
   * [[2502.01637|SCONE]] (NeurIPS'25): scale n-gram embeddings off-accelerator; 1B beats 1.9B at half the FLOPs.
   * [[2601.21204|LongCat-Flash-Lite]]: 30B of embeddings in a 68.5B model.
   * Memory layers: [[2508.18756|UltraMemV2]] (MoE parity at 120B total).
   * [[2510.02375|Hierarchical memories]], [[2607.27919|Memory Decoder at scale]].
4. **Adaptive depth and recursion**: [[2507.10524|Mixture-of-Recursions]] (NeurIPS'25; token-level recursion depth with
   recursion-wise KV caching), [[2603.15619|Mixture-of-Depths Attention]], [[2605.20613|HRM-Text]]. Looped models are in
   [`reasoning/latent-and-looped`](../../reasoning/latent-and-looped/README.md).
5. **Beyond next-token prediction.**
   * [[2510.27688|CALM: continuous autoregressive LMs]] predicts K-token chunks as one continuous vector (autoencoder
     reconstruction >99.9%), so K× fewer generation steps.
   * [[2507.02092|Energy-Based Transformers]], [[2509.14252|LLM-JEPA]], concept models ([[2508.05305|SONAR-LLM]],
     [[2609.10715|NCP]]).
6. **Science of architecture**: [[2512.17351|Physics of LMs 4.1: Canon layers]] (cheap local token mixing; roughly 2×
   reasoning depth across attention, linear and SSM backbones).

### Hand ranking

| # | Paper | Area | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2601.07372\|Engram: conditional memory]] (DeepSeek) | Memory sparsity | O(1) hashed n-gram lookup memory beside MoE; memory/MoE allocation law | Beats iso-parameter/iso-FLOP MoE at 27B; reasoning gains (e.g. BBH) > knowledge gains; host prefetch |
| 2 | [[2512.24880\|mHC]] (DeepSeek) | Residual | Hyper-connections constrained to a manifold (restore identity mapping) + infrastructure optimizations | Stable, scalable multi-stream residuals with real gains at scale |
| 3 | [[2603.15031\|Attention Residuals]] (Moonshot) | Residual | Softmax attention over previous layer outputs; Block AttnRes for memory | Mitigates PreNorm dilution in Kimi Linear 48B-A3B on 1.4T tokens |
| 4 | [[2502.05795\|The Curse of Depth]] (NeurIPS'25) | Normalization | Scale LayerNorm output by 1/√layer | Deep layers contribute again; better pretraining and SFT |
| 5 | [[2507.10524\|Mixture-of-Recursions]] (NeurIPS'25) | Adaptive depth | Shared recursive block + per-token router picking recursion depth; KV only for active tokens | Better perplexity and throughput than vanilla and recursive baselines at equal FLOPs |
| 6 | [[2503.10622\|DyT: Transformers without Normalization]] (CVPR'25) | Normalization | tanh(αx) drop-in for LayerNorm/RMSNorm | Matches normalized models, mostly without tuning |
| 7 | [[2510.27688\|CALM: Continuous Autoregressive LMs]] | Objective | Autoencode K tokens into a vector; predict vectors with an energy/likelihood-free head | K× fewer generative steps at matched quality |
| 8 | [[2502.01637\|SCONE: scaling embedding layers]] (NeurIPS'25) | Embedding scaling | Frequent n-gram embeddings from an offline model, stored off-accelerator | 1B accelerator-resident beats 1.9B at ~half the inference FLOPs |
| 9 | [[2512.17351\|Physics of LMs 4.1: Canon layers]] | Architecture science | Lightweight local-mixing Canon layers across architectures | ~2× reasoning depth; enables fair architecture comparisons |
| 10 | [[2502.12170\|MUDDFormer]] (ICML'25) | Residual | Multiway dynamic dense connections across layers | Matches Transformers trained with 1.8–2.4× compute |
| 11 | [[2508.18756\|UltraMemV2]] | Memory layers | Memory-layer design at 120B total / 2.5B active | Parity with MoE; activation density matters more than total sparse parameters |
| 12 | [[2505.13898\|Do LMs use their depth efficiently?]] (NeurIPS'25) | Analysis | Layer contribution and skip analysis | Second-half layers mostly refine the residual; motivates residual/depth redesign |

**Also useful.**
* Brain-inspired: [[2509.26507|Dragon Hatchling (BDH)]].
* Nested learning (Titans follow-up): [[2512.24695]].
* Memory models: [[2502.06049|LM2]], [[2602.03359|MeKi]].
* Chain-of-Model elastic sizes: [[2505.11820]].
* Automated architecture discovery: [[2603.29640|ASI-Evolve]].
* Scale vectors in norms: [[2605.26895]].
* Survey: [[2508.09834|Speed Always Wins]].

**For runtimes.**
* Support multi-stream residuals (mHC/HC).
* **Host-memory embedding/memory tables with deterministic prefetch** (Engram, SCONE).
* Per-token recursion depth (MoR) with depth-aware KV.
* Continuous-vector decoding heads (CALM).
* Normalization-free blocks (DyT), which simplify fused kernels.
