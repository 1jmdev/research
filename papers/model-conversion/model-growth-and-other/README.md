# Model growth & other architecture conversion

Depth/width growth, layer stacking, re-using checkpoints across architectures, converting modality or precision paradigm.

**8 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🏆 Best of the best by impact score (top 10)

1. **[BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs](2604.02045-bidirlm-from-text-to-omnimodal-bidirectional-encoders-by-adapting-and.md)** (2026-04) — This work identifies the key factors driving successful adaptation of causal generative language models, highlighting the critical role of an often-omitted prior masking phase, and introduces a dual strategy combining …  
   _score 5.68 · 3 cites · 33▲ HF · [code](https://github.com/embeddings-benchmark/mteb) · ~0.01–92 H100-h_
2. **[Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](2504.06225-encoder-decoder-gemma-improving-the-quality-efficiency-trade-off-via-a.md)** (2025-04) — It is argued that adaptation not only enables inheriting the capability of decoder-only LLMs but also reduces the demand for computation compared to pretraining from scratch, to achieve a more favorable …  
   _score 4.16 · 26 cites_
3. **[Efficient Construction of Model Family through Progressive Training Using Model Expansion](2504.00623-efficient-construction-of-model-family-through-progressive-training-us.md)** (2026-03) — An efficient method for constructing model families via progressive training, where smaller models are incrementally expanded to larger sizes to create a complete model family, which reduces total computational cost by …  
   _score 3.29 · accepted by COLM 2025 as a c · 9 cites_
4. **[Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate](2507.07129-growing-transformers-modular-composition-and-layer-wise-expansion-on-a.md)** (2026-05) — The evidence supports a narrow claim: useful continued learning can proceed above a frozen minimal interface under a bounded active trainable-parameter budget, with a clear tradeoff against dense monolithic training in …  
   _score 2.63 · 1 cites · 3▲ HF · [code](https://github.com/AVBochkov/PGT)_
5. **[Progressive Depth Up-scaling via Optimal Transport](2508.08011-progressive-depth-up-scaling-via-optimal-transport.md)** (2025-08) — Optimal Transport Depth Up-Scaling achieves better overall performance and offers improved training efficiency than existing methods for continual pre-training and supervised fine-tuning across different model sizes.  
   _score 1.88 · 3 cites_
6. **[Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets](2608.11233-retrofitting-recurrent-depth-into-a-pretrained-language-model-installa.md)** (2026-07) — An iterative transformer can perform deeper reasoning in latent space faster than comparable or larger models fine-tuned on the same task, in a system-level comparison.  
   _score 1.58 · 1 cites · [code](https://github.com/mshapiro123/recurrent-qwen-svgd)_
7. **[When is Warmstarting Effective for Scaling Language Models?](2605.13405-when-is-warmstarting-effective-for-scaling-language-models.md)** (2026-05) — It is shown that preserving the base model's initial post-growth performance is not necessary for strong final performance, and that simple, architecture-agnostic growth strategies can outperform more complex …  
   _score 0.0 · 0 cites_
8. **[KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](2609.27294-kite-kv-invariant-transformer-expansion-for-efficient-agentic-llm-scal.md)** (2026-09) — KV-Invariant Transformer Expansion (KITE) is introduced, a scaling paradigm that trains the model from a smaller size to a larger size (i.e., saving training costs via upcycling), while places newly added parameters in …  
   _score 0.0 · 0 cites_

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing](2604.02045-bidirlm-from-text-to-omnimodal-bidirectional-encoders-by-adapting-and.md) | 0.01 | 92 | MI250 | We merge them at a 50% ratio 8 We provide a detailed analysis for merge ratios \in\{0,0.25,0.5,0.75,1\} in §​ F.3 (cos sim: 0.97) and perform 500 fine-tuning st… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs](2604.02045-bidirlm-from-text-to-omnimodal-bidirectional-encoders-by-adapting-and.md) | 2026-04-02 | 5.68 | 3 | 33 |  | [✓](https://github.com/embeddings-benchmark/mteb) | This work identifies the key factors driving successful adaptation of causal generative language models, highlighting the critical role of an often-omitted … |
| 2 | [Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](2504.06225-encoder-decoder-gemma-improving-the-quality-efficiency-trade-off-via-a.md) | 2025-04-08 | 4.16 | 26 | 0 |  |  | It is argued that adaptation not only enables inheriting the capability of decoder-only LLMs but also reduces the demand for computation compared to … |
| 3 | [Efficient Construction of Model Family through Progressive Training Using Model Expansion](2504.00623-efficient-construction-of-model-family-through-progressive-training-us.md) | 2026-03-16 | 3.29 | 9 | 0 | accepted by COLM 2025 as a c |  | An efficient method for constructing model families via progressive training, where smaller models are incrementally expanded to larger sizes to create a … |
| 4 | [Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate](2507.07129-growing-transformers-modular-composition-and-layer-wise-expansion-on-a.md) | 2026-05-02 | 2.63 | 1 | 3 |  | [✓](https://github.com/AVBochkov/PGT) | The evidence supports a narrow claim: useful continued learning can proceed above a frozen minimal interface under a bounded active trainable-parameter budget, … |
| 5 | [Progressive Depth Up-scaling via Optimal Transport](2508.08011-progressive-depth-up-scaling-via-optimal-transport.md) | 2025-08-11 | 1.88 | 3 | 0 |  |  | Optimal Transport Depth Up-Scaling achieves better overall performance and offers improved training efficiency than existing methods for continual pre-training … |
| 6 | [Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Rete](2608.11233-retrofitting-recurrent-depth-into-a-pretrained-language-model-installa.md) | 2026-07-31 | 1.58 | 1 | 0 |  | [✓](https://github.com/mshapiro123/recurrent-qwen-svgd) | An iterative transformer can perform deeper reasoning in latent space faster than comparable or larger models fine-tuned on the same task, in a system-level … |
| 7 | [When is Warmstarting Effective for Scaling Language Models?](2605.13405-when-is-warmstarting-effective-for-scaling-language-models.md) | 2026-05-13 | 0.0 | 0 | 0 |  |  | It is shown that preserving the base model's initial post-growth performance is not necessary for strong final performance, and that simple, … |
| 8 | [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](2609.27294-kite-kv-invariant-transformer-expansion-for-efficient-agentic-llm-scal.md) | 2026-09-23 | 0.0 | 0 | 0 |  |  | KV-Invariant Transformer Expansion (KITE) is introduced, a scaling paradigm that trains the model from a smaller size to a larger size (i.e., saving training … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Curriculum-Guided Layer Scaling for Language Model Pretraining](../../training/pretraining-recipes-and-efficiency/2506.11389-curriculum-guided-layer-scaling-for-language-model-pretraining.md) | Pre-training recipes & efficiency | 4.0 |
| [SCORE: Replacing Layer Stacking with Contractive Recurrent Depth](../../models-and-architectures/novel-architectures/2603.10544-score-replacing-layer-stacking-with-contractive-recurrent-depth.md) | Novel architectures & architecture analysis | 0.88 |
| [Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD Compression for LLMs](../../compression/low-rank-decomposition/2606.19993-activation-and-influence-aware-ranks-air-function-preserving-svd-compr.md) | Low-rank decomposition & weight factorization | 0.7 |
