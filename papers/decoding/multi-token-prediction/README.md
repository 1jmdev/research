# Multi-token prediction (MTP)

Training/inference with multiple future-token heads or objectives (DeepSeek-V3 MTP, future-token prediction, register tokens for MTP).

**25 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Decoding: speculative, parallel, MTP, diffusion](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/decoding.md](../../../overviews/decoding.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Multi-token prediction (MTP) has two uses.

1. **A pretraining auxiliary loss.** DeepSeek-V3-style sequential MTP modules are now standard in DeepSeek, Qwen3-Next,
   GLM-4.5, MiMo and others. The quality gains from *exact* future-token MTP are inconsistent at small scale, so softer
   targets work better:
   * token order prediction (TOP);
   * future summaries (FSP);
   * register tokens (MuToR) for fine-tuning.
2. **A built-in speculative drafter.** The MTP head drafts and the main model verifies. The main levers:
   * **self-distillation**: FastMTP lifts 3rd-token acceptance from 2% to 36%, and MTP-D adds 7.5%;
   * **recursive/shared heads**;
   * **windowed draft attention** at million-token context ([Windowed-MTP](2607.21535-windowed-mtp-removing-the-full-context-draft-kv-tax-at-million-token-c.md); otherwise the draft's full-KV
     read dominates).

A third, emerging use is converting a pretrained AR model into a **standalone multi-token generator** with no separate
verifier (MTP via self-distillation, MARS, K-Forcing). It is a cousin of diffusion conversion (see
[`model-conversion/ar-to-diffusion`](../../model-conversion/ar-to-diffusion/README.md)).

### Hand ranking

| # | Paper | Use | Key idea | Result / cost |
| ---: | --- | --- | --- | --- |
| 1 | [FastMTP](2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md) | MTP → SD | Fine-tune **one shared-weight MTP head** on self-distilled data for recursive drafting + language-aware vocabulary compression | 2.03× over NTP (+82% vs vanilla MTP); <1 day on one H20 node (~4 H100-h) |
| 2 | [Your LLM Knows the Future](2507.11851-your-llm-knows-the-future-uncovering-its-multi-token-prediction-potent.md) (Apple) | Retrofit MTP | Mask-token inputs + **gated LoRA** (keeps NTP exact) + learnable sampler head | ~5× speedup on code/math with no quality loss |
| 3 | [Token Order Prediction (TOP)](2508.19228-predicting-the-order-of-upcoming-tokens-improves-language-modeling.md) | Pretraining aux | Rank upcoming tokens by proximity (learning-to-rank) instead of exact MTP; one extra unembedding | Beats NTP, MTP and DeepSeek-MTP at 340M/1.8B/7B; 7B run ≈ 3K H100-h |
| 4 | [MuToR: MTP needs registers](2505.10518-multi-token-prediction-needs-registers.md) (NeurIPS'25) | FT / pretrain aux | Interleave **register tokens** that predict future targets; no architecture change | MTP benefits carried to SFT |
| 5 | [Future Summary Prediction](2510.14751-beyond-multi-token-prediction-pretraining-llms-with-future-summaries.md) (Meta) | Pretraining aux | Predict a **summary** (bag-of-words or learned reverse-LM embedding) of the long-term future | Gains on reasoning/planning at 3B/8B with 1T tokens |
| 6 | [MTP-D](2603.23911-self-distillation-for-multi-token-prediction.md) | Pretraining + SD | Self-distillation for MTP heads + **looped extension** to 8–16 heads | +7.5% acceptance; +35% speed from head extension |
| 7 | [MTP via self-distillation](2602.06019-multi-token-prediction-via-self-distillation.md) | Standalone MTP | Online distillation turns an AR model into a multi-token model with the **same implementation**, no verifier | >3× faster, <5% GSM8K drop |
| 8 | [MARS](2604.07023-mars-enabling-autoregressive-models-multi-token-generation.md) | Standalone MTP | Mask-autoregression fine-tuning on instruction data; no new params; still callable as NTP | Multiple tokens per pass with no NTP degradation |
| 9 | [ESP: embedding-space probing](2603.17942-efficient-training-free-multi-token-prediction-via-embedding-space-pro.md) (ICML'26) | Training-free MTP | Probe with mask tokens from the embedding space + dynamic tree | Beats training-free baselines (LADE); vs EAGLE-3's 200–300 GPU-h training |
| 10 | [L-MTP](2505.17505-l-mtp-leap-multi-token-prediction-beyond-adjacent-context-for-large-la.md) (NeurIPS'25) | Leap MTP | Predict **non-adjacent** future tokens + a matching decoding strategy | Better long-range dependency and speed |
| 11 | [Windowed-MTP](2607.21535-windowed-mtp-removing-the-full-context-draft-kv-tax-at-million-token-c.md) | Serving | Sliding window + sink for the **draft head's** attention only | Keeps MTP speculation profitable at 1M context |
| 12 | [How transformers learn to plan via MTP](2604.11912-how-transformers-learn-to-plan-via-multi-token-prediction.md) (COLM'26) | Theory | MTP induces reverse reasoning (attend to goal, trace back) via gradient decoupling | Explains the planning benefit |

**For a runtime.**
* Treat a model's MTP head as an EAGLE-like drafter: chain or tree drafting with the MTP module reusing the main
  model's KV.
* Window the draft attention at long context.
* Choose speculation depth by entropy or load ([EntMTP](2606.27550-entmtp-accelerating-llm-inference-with-entropy-guided-multi-token-pred.md)).

**For model builders.** Train with 1–2 MTP modules (DeepSeek-V3 recipe) or TOP/FSP-style soft auxiliaries. Then
**self-distill the MTP head** before release (FastMTP) to maximize acceptance.

## 🏆 Best of the best by impact score (top 10)

1. **[Multi-Token Prediction Needs Registers](2505.10518-multi-token-prediction-needs-registers.md)** (2025-05) — MuToR is proposed, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets, making it especially …  
   _score 7.47 · Neural Information Processing Systems (Neural Inf Process Sy · 13 cites · 13▲ HF · [code](https://github.com/nasosger/MuToR)_
2. **[Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](2507.11851-your-llm-knows-the-future-uncovering-its-multi-token-prediction-potent.md)** (2025-07) — This work proposes a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques to realize this potential and enable simultaneous prediction …  
   _score 6.17 · 43 cites_
3. **[Predicting the Order of Upcoming Tokens Improves Language Modeling](2508.19228-predicting-the-order-of-upcoming-tokens-improves-language-modeling.md)** (2026-02) — Token order prediction (TOP) is proposed, which trains models to order upcoming tokens by their proximity using a learning-to-rank loss, which requires only a single additional unembedding layer compared to MTP's …  
   _score 5.37 · 3 cites · 23▲ HF · [code](https://github.com/zaydzuhri/token-order-prediction) · ~3.0k H100-h_
4. **[MARS: Enabling Autoregressive Models Multi-Token Generation](2604.07023-mars-enabling-autoregressive-models-multi-token-generation.md)** (2026-04) — This work introduces MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per forward pass, and develops a block-level KV caching strategy …  
   _score 4.64 · 0 cites · 36▲ HF · [code](https://github.com/Xalp/MARS)_
5. **[Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](2510.14751-beyond-multi-token-prediction-pretraining-llms-with-future-summaries.md)** (2026-03) — This work proposes future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations.  
   _score 3.74 · 13 cites · [code](https://github.com/chenwu98/algorithmic-creativity)_
6. **[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](2505.17505-l-mtp-leap-multi-token-prediction-beyond-adjacent-context-for-large-la.md)** (2025-09) — L-MTP is an innovative token prediction method that extends the capabilities of multi-token prediction by introducing a leap-based mechanism, which enhances the model's ability to capture long-range dependencies but …  
   _score 3.73 · Accepted by NeurIPS 2025 · 14 cites · [code](https://github.com/Xiaohao-Liu/L-MTP)_
7. **[Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](2603.17942-efficient-training-free-multi-token-prediction-via-embedding-space-pro.md)** (2026-05) — A training-free method for multi-token prediction in large language models using mask tokens from the embedding space enables parallel token generation with improved throughput and accuracy.  
   _score 3.34 · Accepted at ICML 2026 · 0 cites · 10▲ HF · ~300 H100-h_
8. **[Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits](2511.11346-fast-and-expressive-multi-byte-prediction-with-probabilistic-circuits.md)** (2026-06) — This work investigates the trade-off between expressiveness and latency in MTP within the framework of probabilistic circuits (PCs) and rigorously study the optimal trade-off between expressiveness and latency when …  
   _score 3.22 · 9 cites_
9. **[Improving Large Language Models with Concept-Aware Fine-Tuning](2506.07833-improving-large-language-models-with-concept-aware-fine-tuning.md)** (2025-06) — Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned, is introduced, which is the first to bring the multi-token setting to the post-training phase, thus …  
   _score 3.15 · 3 cites · 4▲ HF · [code](https://github.com/michaelchen-lab/caft-llm)_
10. **[FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction](2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md)** (2025-09) — FastMTP is introduced, a simple yet effective method that improves multi-step draft quality by aligning MTP training with its inference pattern, significantly enhancing speculative decoding performance and reducing …  
   _score 2.98 · 12 cites · ~3.6 H100-h_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[How Transformers Learn to Plan via Multi-Token Prediction](2604.11912-how-transformers-learn-to-plan-via-multi-token-prediction.md)** (2026-07-26; 0▲, 1 cites) — It is proved that MTP induces a two-stage reverse reasoning process: the model first attends to the end node and then reconstructs the path by tracing intermediate nodes backward, …
- **[Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context](2607.21535-windowed-mtp-removing-the-full-context-draft-kv-tax-at-million-token-c.md)** (2026-07-23; 0▲, 0 cites) — Across three architecture families at 1M context on a single GPU in SGLang, windowing cuts the per-decode-step cost over the shipping native MTP draft by +28% to +44%, an …
- **[AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction](2608.00434-adamtp-an-adaptive-training-paradigm-for-multi-token-prediction.md)** (2026-08-01; 0▲, 0 cites) — AdaMTP is proposed, an adaptive training paradigm that dynamically aligns the prediction horizon with the intrinsic predictability of the sequence, and consistently outperforms …
- **[LoopMTP: A looped transformer guided by latent multi-token prediction](2608.03624-loopmtp-a-looped-transformer-guided-by-latent-multi-token-prediction.md)** (2026-08-04; 0▲, 0 cites) — Multi-token prediction (MTP) supplies exactly the dense, forward-looking supervision the loop is missing, by softly aligning the hidden state of loop $t$ with the embedding of the …
- **[Hierarchical Latent Prediction for Language Models](2608.05806-hierarchical-latent-prediction-for-language-models.md)** (2026-08-06; 0▲, 0 cites) — Hierarchical Latent Prediction (HiLP) is introduced, which introduces an auxiliary higher-level abstract latent to help reduce the error accumulation effect in latent-space …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction](2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md) | 3.6 | 3.6 | H20 | The entire training process was completed in less than 1 day on a single H20 server, demonstrating low training cost.… |
| [Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](2603.17942-efficient-training-free-multi-token-prediction-via-embedding-space-pro.md) | 300 | 300 | unspecified | The trade-off is clear: EAGLE3 requires 200–300 GPU hours of training per model (training time can increase for larger model sizes), whereas training-free metho… |
| [Predicting the Order of Upcoming Tokens Improves Language Modeling](2508.19228-predicting-the-order-of-upcoming-tokens-improves-language-modeling.md) | 3.0k | 3.0k | H200 | The 7B models require 2 weeks of training time each on the 8xH200 node available to us.… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [Multi-Token Prediction Needs Registers](2505.10518-multi-token-prediction-needs-registers.md) | 2025-05-15 | 7.47 | 13 | 13 | Neural Information Processing Systems (N | [✓](https://github.com/nasosger/MuToR) | MuToR is proposed, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked … |
| 2 | [Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential](2507.11851-your-llm-knows-the-future-uncovering-its-multi-token-prediction-potent.md) | 2025-07-16 | 6.17 | 43 | 0 |  |  | This work proposes a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques … |
| 3 | [Predicting the Order of Upcoming Tokens Improves Language Modeling](2508.19228-predicting-the-order-of-upcoming-tokens-improves-language-modeling.md) | 2026-02-16 | 5.37 | 3 | 23 |  | [✓](https://github.com/zaydzuhri/token-order-prediction) | Token order prediction (TOP) is proposed, which trains models to order upcoming tokens by their proximity using a learning-to-rank loss, which requires only a … |
| 4 | [MARS: Enabling Autoregressive Models Multi-Token Generation](2604.07023-mars-enabling-autoregressive-models-multi-token-generation.md) | 2026-04-08 | 4.64 | 0 | 36 |  | [✓](https://github.com/Xalp/MARS) | This work introduces MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per … |
| 5 | [Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](2510.14751-beyond-multi-token-prediction-pretraining-llms-with-future-summaries.md) | 2026-03-25 | 3.74 | 13 | 0 |  | [✓](https://github.com/chenwu98/algorithmic-creativity) | This work proposes future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving … |
| 6 | [L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](2505.17505-l-mtp-leap-multi-token-prediction-beyond-adjacent-context-for-large-la.md) | 2025-09-22 | 3.73 | 14 | 0 | Accepted by NeurIPS 2025 | [✓](https://github.com/Xiaohao-Liu/L-MTP) | L-MTP is an innovative token prediction method that extends the capabilities of multi-token prediction by introducing a leap-based mechanism, which enhances … |
| 7 | [Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](2603.17942-efficient-training-free-multi-token-prediction-via-embedding-space-pro.md) | 2026-05-28 | 3.34 | 0 | 10 | Accepted at ICML 2026 |  | A training-free method for multi-token prediction in large language models using mask tokens from the embedding space enables parallel token generation with … |
| 8 | [Fast and Expressive Multi-Byte Prediction with Probabilistic Circuits](2511.11346-fast-and-expressive-multi-byte-prediction-with-probabilistic-circuits.md) | 2026-06-02 | 3.22 | 9 | 0 |  |  | This work investigates the trade-off between expressiveness and latency in MTP within the framework of probabilistic circuits (PCs) and rigorously study the … |
| 9 | [Improving Large Language Models with Concept-Aware Fine-Tuning](2506.07833-improving-large-language-models-with-concept-aware-fine-tuning.md) | 2025-06-13 | 3.15 | 3 | 4 |  | [✓](https://github.com/michaelchen-lab/caft-llm) | Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned, is introduced, which is the first to bring the … |
| 10 | [FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction](2509.18362-fastmtp-accelerating-llm-inference-with-enhanced-multi-token-predictio.md) | 2025-09-16 | 2.98 | 12 | 0 |  |  | FastMTP is introduced, a simple yet effective method that improves multi-step draft quality by aligning MTP training with its inference pattern, significantly … |
| 11 | [Multi-Token Prediction via Self-Distillation](2602.06019-multi-token-prediction-via-self-distillation.md) | 2026-04-23 | 2.49 | 5 | 0 |  | [✓](https://github.com/sgl-project/sglang) | This work considers a new approach for converting a pretrained autoregressive language model from a slow single next token prediction model into a fast … |
| 12 | [HAMburger: Accelerating LLM Inference via Token Smashing](2505.20438-hamburger-accelerating-llm-inference-via-token-smashing.md) | 2025-05-26 | 1.92 | 6 | 0 |  |  | HAMburger is introduced, a Hierarchically Auto-regressive Model that redefines resource allocation in LLMs by moving beyond uniform computation and storage per … |
| 13 | [How Transformers Learn to Plan via Multi-Token Prediction](2604.11912-how-transformers-learn-to-plan-via-multi-token-prediction.md) | 2026-07-26 | 1.72 | 1 | 0 | COLM 2026 |  | It is proved that MTP induces a two-stage reverse reasoning process: the model first attends to the end node and then reconstructs the path by tracing … |
| 14 | [On multi-token prediction for efficient LLM inference](2502.09419-on-multi-token-prediction-for-efficient-llm-inference.md) | 2025-02-13 | 1.65 | 5 | 0 |  |  | This work systematically investigates multi-token prediction capabilities within LLMs pre-trained for next-token prediction (NTP) and shows that while joint … |
| 15 | [Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction](2509.23186-understanding-and-enhancing-the-planning-capability-of-language-models.md) | 2025-09-27 | 1.27 | 5 | 0 |  |  | This work theoretically analyzes the Multi-Token Prediction paradigm using a Transformer architecture composed of a shared output head and a transfer layer and … |
| 16 | [Self-Distillation for Multi-Token Prediction](2603.23911-self-distillation-for-multi-token-prediction.md) | 2026-03-25 | 1.27 | 3 | 0 |  |  | This work proposes MTP-D, a simple yet effective self-distillation method with minimal additional training cost, which boosts MTP head acceptance rates while … |
| 17 | [Pre-Training Curriculum for Multi-Token Prediction in Language Models](2505.22757-pre-training-curriculum-for-multi-token-prediction-in-language-models.md) | 2025-05-28 | 1.19 | 2 | 0 | Accepted to ACL 2025 (Main |  | A curriculum learning strategy for MTP training is proposed, exploring two variants: a forward curriculum, which gradually increases the complexity of the … |
| 18 | [K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling](2606.10820-k-forcing-joint-next-k-token-decoding-via-push-forward-language-modeli.md) | 2026-06-10 | 0.5 | 0 | 0 |  | [✓](https://github.com/alibaba-damo-academy/K-Forcing) | K-Forcing is introduced, a push-forward language modeling paradigm for joint next-k-token decoding that distills an existing AR model into a conditional … |
| 19 | [Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs](2605.27255-pair-in-pair-out-latent-multi-token-prediction-for-efficient-llms.md) | 2026-05-29 | 0.0 | 0 | 0 |  |  | PIPO trains a lightweight confidence head that decides whether draft tokens should be accepted, and observes that On-Policy Distillation naturally matches the … |
| 20 | [CLP: Collocation-Length Prediction for Zero-Loss Adaptive Multi-Token Inference](2606.10935-clp-collocation-length-prediction-for-zero-loss-adaptive-multi-token-i.md) | 2026-06-09 | 0.0 | 0 | 0 |  |  | Backbone-as-Architect is proposed, a design principle where the backbone LM head always generates the first token, and MTP heads are responsible only for … |
| 21 | [EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction](2606.27550-entmtp-accelerating-llm-inference-with-entropy-guided-multi-token-pred.md) | 2026-06-25 | 0.0 | 0 | 0 |  |  | Entropy-guided Multi-Token Prediction (EntMTP), a training-free scheduler that toggles between tree-based attention topologies from a set of task-specific … |
| 22 | [Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context](2607.21535-windowed-mtp-removing-the-full-context-draft-kv-tax-at-million-token-c.md) | 2026-07-23 | 0.0 | 0 | 0 |  |  | Across three architecture families at 1M context on a single GPU in SGLang, windowing cuts the per-decode-step cost over the shipping native MTP draft by +28% … |
| 23 | [AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction](2608.00434-adamtp-an-adaptive-training-paradigm-for-multi-token-prediction.md) | 2026-08-01 | 0.0 | 0 | 0 |  |  | AdaMTP is proposed, an adaptive training paradigm that dynamically aligns the prediction horizon with the intrinsic predictability of the sequence, and … |
| 24 | [LoopMTP: A looped transformer guided by latent multi-token prediction](2608.03624-loopmtp-a-looped-transformer-guided-by-latent-multi-token-prediction.md) | 2026-08-04 | 0.0 | 0 | 0 |  |  | Multi-token prediction (MTP) supplies exactly the dense, forward-looking supervision the loop is missing, by softly aligning the hidden state of loop $t$ with … |
| 25 | [Hierarchical Latent Prediction for Language Models](2608.05806-hierarchical-latent-prediction-for-language-models.md) | 2026-08-06 | 0.0 | 0 | 0 |  |  | Hierarchical Latent Prediction (HiLP) is introduced, which introduces an auxiliary higher-level abstract latent to help reduce the error accumulation effect in … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [MiMo-V2-Flash Technical Report](../../models-and-architectures/technical-reports/2601.02780-mimo-v2-flash-technical-report.md) | Model technical reports (open & frontier models) | 15.85 |
| [Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters](../../models-and-architectures/technical-reports/2602.10604-step-3-5-flash-open-frontier-level-intelligence-with-11b-active-parame.md) | Model technical reports (open & frontier models) | 13.54 |
| [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation De](../../model-conversion/ar-to-diffusion/2607.05722-nemotron-labs-diffusion-a-tri-mode-language-model-unifying-autoregress.md) | AR → diffusion LM conversion | 8.34 |
| [DeepSeek: Paradigm Shifts and Technical Evolution in Large AI Models](../../models-and-architectures/technical-reports/2507.09955-deepseek-paradigm-shifts-and-technical-evolution-in-large-ai-models.md) | Model technical reports (open & frontier models) | 5.04 |
| [GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding](../../model-conversion/attention-conversion/2605.15250-gqla-group-query-latent-attention-for-hardware-adaptive-large-language.md) | Attention conversion (MHA/GQA → MLA, GQA uptraining, sparse retrofit) | 4.3 |
| [SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training](../../mixture-of-experts/compression/2605.08738-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre.md) | MoE compression (expert pruning, merging, quantization) | 4.27 |
| [Dynamic Multi-Byte Prediction With Hierarchical Language Models](../../training/tokenization/2608.15454-dynamic-multi-byte-prediction-with-hierarchical-language-models.md) | Tokenization & byte-level models | 3.85 |
| [KAT-V1: Kwai-AutoThink Technical Report](../../reasoning/efficient-reasoning/2507.08297-kat-v1-kwai-autothink-technical-report.md) | Efficient reasoning (CoT compression, overthinking, adaptive thinking) | 2.59 |
| [AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding](../speculative-decoding/2607.25852-angelspec-towards-real-world-high-performance-inference-with-speculati.md) | Speculative decoding (draft–verify) | 1.77 |
| [PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention](../../attention/sparse-attention/2607.24593-pivot-efficient-query-group-indexing-for-token-level-sparse-attention.md) | Sparse attention (trainable & training-free) | 1.73 |
| [P-EAGLE: Parallel-Drafting EAGLE with Scalable Training](../speculative-decoding/2602.01469-p-eagle-parallel-drafting-eagle-with-scalable-training.md) | Speculative decoding (draft–verify) | 1.61 |
| [JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency](../../models-and-architectures/technical-reports/2604.03044-joyai-llm-flash-advancing-mid-scale-llms-with-token-efficiency.md) | Model technical reports (open & frontier models) | 1.23 |
| [MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration](../../context-compression/prompt-and-context-compression/2604.14889-memosight-unifying-context-compression-and-multi-token-prediction-for.md) | Prompt & context compression | 0.67 |
| [HyperDFlash: Hyper-Connection-Aligned Block Speculative Decoding with Gated Residual Reduction](../speculative-decoding/2606.26744-hyperdflash-hyper-connection-aligned-block-speculative-decoding-with-g.md) | Speculative decoding (draft–verify) | 0.0 |
