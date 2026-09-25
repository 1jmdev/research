# Model growth & other architecture conversion

Depth/width growth, layer stacking, re-using checkpoints across architectures, converting modality or precision paradigm.

**8 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🏆 Best of the best by impact score (top 10)

1. **[BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs](2604.02045-bidirlm-from-text-to-omnimodal-bidirectional-encoders-by-adapting-and.md)** (2026-04) — Adapting causal generative language models into bidirectional encoders through systematic ablation and novel merging strategies achieves superior performance across multiple modalities.  
   _score 4.38 · 0 cites · 33▲ HF · [code](https://github.com/embeddings-benchmark/mteb) · ~0.01–92 H100-h_
2. **[Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate](2507.07129-growing-transformers-modular-composition-and-layer-wise-expansion-on-a.md)** (2026-05) — Transformers with frozen embeddings enable efficient scaling through modular composition and layer-wise growth, improving performance on reasoning tasks without catastrophic forgetting.  
   _score 2.04 · 0 cites · 3▲ HF · [code](https://github.com/AVBochkov/PGT)_
3. **[Efficient Construction of Model Family through Progressive Training Using Model Expansion](2504.00623-efficient-construction-of-model-family-through-progressive-training-us.md)** (2026-03) — Through extensive experiments on a model family ranging from 1B to 8B parameters, we show that our approach reduces total computational cost by approximately 25% while maintaining comparable performance to independently …  
   _score 0.7 · accepted by COLM 2025 as a c · 0 cites_
4. **[Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](2504.06225-encoder-decoder-gemma-improving-the-quality-efficiency-trade-off-via-a.md)** (2025-04) — For example, Gemma 2B-2B outperforms Gemma 2B by $\sim$7\% after instruction tuning.  
   _score 0.0 · 0 cites_
5. **[Progressive Depth Up-scaling via Optimal Transport](2508.08011-progressive-depth-up-scaling-via-optimal-transport.md)** (2025-08) — Scaling Large Language Models (LLMs) yields performance gains but incurs substantial training costs.  
   _score 0.0 · 0 cites_
6. **[When is Warmstarting Effective for Scaling Language Models?](2605.13405-when-is-warmstarting-effective-for-scaling-language-models.md)** (2026-05) — Across our experiments on dense MLPs and dense language models, we find that a $2\times$ growth factor is the most reliable in yielding convergence speedups, with gains most pronounced under 20 tokens/parameter budgets …  
   _score 0.0 · 0 cites_
7. **[Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets](2608.11233-retrofitting-recurrent-depth-into-a-pretrained-language-model-installa.md)** (2026-07) — The adapter matched the full block overall (83.8% versus 84.0%), led through depth 11, and trailed beyond.  
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
| 1 | [BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs](2604.02045-bidirlm-from-text-to-omnimodal-bidirectional-encoders-by-adapting-and.md) | 2026-04-02 | 4.38 | 0 | 33 |  | [✓](https://github.com/embeddings-benchmark/mteb) | Adapting causal generative language models into bidirectional encoders through systematic ablation and novel merging strategies achieves superior performance … |
| 2 | [Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate](2507.07129-growing-transformers-modular-composition-and-layer-wise-expansion-on-a.md) | 2026-05-02 | 2.04 | 0 | 3 |  | [✓](https://github.com/AVBochkov/PGT) | Transformers with frozen embeddings enable efficient scaling through modular composition and layer-wise growth, improving performance on reasoning tasks … |
| 3 | [Efficient Construction of Model Family through Progressive Training Using Model Expansion](2504.00623-efficient-construction-of-model-family-through-progressive-training-us.md) | 2026-03-16 | 0.7 | 0 | 0 | accepted by COLM 2025 as a c |  | Through extensive experiments on a model family ranging from 1B to 8B parameters, we show that our approach reduces total computational cost by approximately … |
| 4 | [Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation](2504.06225-encoder-decoder-gemma-improving-the-quality-efficiency-trade-off-via-a.md) | 2025-04-08 | 0.0 | 0 | 0 |  |  | For example, Gemma 2B-2B outperforms Gemma 2B by $\sim$7\% after instruction tuning. |
| 5 | [Progressive Depth Up-scaling via Optimal Transport](2508.08011-progressive-depth-up-scaling-via-optimal-transport.md) | 2025-08-11 | 0.0 | 0 | 0 |  |  | Scaling Large Language Models (LLMs) yields performance gains but incurs substantial training costs. |
| 6 | [When is Warmstarting Effective for Scaling Language Models?](2605.13405-when-is-warmstarting-effective-for-scaling-language-models.md) | 2026-05-13 | 0.0 | 0 | 0 |  |  | Across our experiments on dense MLPs and dense language models, we find that a $2\times$ growth factor is the most reliable in yielding convergence speedups, … |
| 7 | [Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Rete](2608.11233-retrofitting-recurrent-depth-into-a-pretrained-language-model-installa.md) | 2026-07-31 | 0.0 | 0 | 0 |  |  | The adapter matched the full block overall (83.8% versus 84.0%), led through depth 11, and trailed beyond. |
| 8 | [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](2609.27294-kite-kv-invariant-transformer-expansion-for-efficient-agentic-llm-scal.md) | 2026-09-23 | 0.0 | 0 | 0 |  |  | KV-Invariant Transformer Expansion (KITE) is introduced, a scaling paradigm that trains the model from a smaller size to a larger size (i.e., saving training … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Curriculum-Guided Layer Scaling for Language Model Pretraining](../../training/pretraining-recipes-and-efficiency/2506.11389-curriculum-guided-layer-scaling-for-language-model-pretraining.md) | Pre-training recipes & efficiency | 1.2 |
| [Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD Compression for LLMs](../../compression/low-rank-decomposition/2606.19993-activation-and-influence-aware-ranks-air-function-preserving-svd-compr.md) | Low-rank decomposition & weight factorization | 0.7 |
| [SCORE: Replacing Layer Stacking with Contractive Recurrent Depth](../../models-and-architectures/novel-architectures/2603.10544-score-replacing-layer-stacking-with-contractive-recurrent-depth.md) | Novel architectures & architecture analysis | 0.0 |
