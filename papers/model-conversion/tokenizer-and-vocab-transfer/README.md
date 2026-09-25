# Tokenizer / vocabulary transfer & cross-tokenizer distillation

Swapping or extending a pretrained model's tokenizer, cross-tokenizer distillation, zero-shot tokenizer transfer.

**27 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Changing a pretrained model's tokenizer is now cheap. The work splits into four jobs:

1. **Transplant a new vocabulary** (domain/language efficiency, or matching a teacher's vocabulary for KD). Initialize new
   embeddings training-free, then heal briefly:
   * [OMP](2506.06607-training-free-tokenizer-transplantation-via-orthogonal-matching-pursui.md): new tokens as sparse combinations of shared anchor tokens (in mergekit's tokensurgeon);
   * [TokAlign](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md): token co-occurrence alignment, <2 CPU-hours vs 661 GPU-h for ZeTT's hypernetwork;
   * [MATT](2510.21954-model-aware-tokenizer-transfer.md): model-aware, uses attention dynamics;
   * [Token Distillation](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md) (ICLR'26): new-token embeddings distilled from the attention behaviour of the
     original multi-token sequence, 2,500 tokens in <10 minutes.
2. **Domain vocabulary for speed.** [AdaptiVocab](2503.19693-adaptivocab-enhancing-llm-efficiency-in-focused-domains-through-lightw.md) cuts token usage >25% in focused domains with a
   lightweight fine-tune. Language-specific retokenization cuts fertility ([Italian](2504.17025-optimizing-llms-for-italian-reducing-token-fertility-and-enhancing-eff.md),
   [Bielik v3 Polish](2604.10799-advancing-polish-language-modeling-through-tokenizer-optimization-in-t.md)). Fewer tokens directly means fewer decode steps.
3. **Byteification** (subword → byte-level). [Bolmo](2512.15586-bolmo-byteifying-the-next-generation-of-language-models.md) converts OLMo into a byte-level LM with local
   mLSTM encoder/decoder layers using **39.3B tokens (<1% of pretraining)**, matching the subword source. Earlier,
   [ALM](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) showed byte-level transfer by approximate likelihood matching.
4. **Cross-tokenizer distillation** (teacher and student use different vocabularies):
   * [ALM](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) (NeurIPS'25): the principled, general method;
   * [byte-level distillation](2604.07466-cross-tokenizer-llm-distillation-through-a-byte-level-interface.md): a simple strong baseline;
   * [byte-prefix marginalization](2607.22334-cross-tokenizer-on-policy-distillation-via-byte-prefix-marginalization.md): for **on-policy** distillation across families;
   * [SRA](2605.01205-sra-span-representation-alignment-for-large-language-model-distillatio.md): span alignment.

   This is what makes "distill Qwen into Llama-vocab" or "Gemma → my model" possible.

### Cost table

| Method | Job | Budget | H100-h | Basis |
| --- | --- | --- | ---: | --- |
| [OMP transplant](2506.06607-training-free-tokenizer-transplantation-via-orthogonal-matching-pursui.md) | Vocabulary swap | training-free | **~0** | reported |
| [Token Distillation](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md) | Add new tokens | 2,500 tokens in <10 min on 1 GPU | **<0.2** | reported |
| [TokAlign](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md) (ACL'25) | Vocabulary replacement | <2 h CPU alignment + short fine-tune | **~1–20** | reported/estimate |
| [MATT](2510.21954-model-aware-tokenizer-transfer.md) | Model-aware transfer | ~250M tokens | **~5–10** (7B) | estimate |
| [AdaptiVocab](2503.19693-adaptivocab-enhancing-llm-efficiency-in-focused-domains-through-lightw.md) | Domain vocabulary | ~8M tokens, 3×A6000 | **~1** | estimate |
| [ALM](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) (NeurIPS'25) | Llama-3-8B → byte-level / cross-tokenizer KD | 32×TPU-v4 × 24 h | **~200** | reported (v4 ≈ 0.28 H100) |
| [Bolmo](2512.15586-bolmo-byteifying-the-next-generation-of-language-models.md) | OLMo-3 7B → byte-level | 39.3B tokens | **~1.2K** | estimate |

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [ALM: universal cross-tokenizer distillation](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) (NeurIPS'25) | Approximate likelihood matching over aligned chunks; works for any tokenizer pair, including subword → byte | First effective cross-tokenizer distillation; also transfers tokenizers (Llama3/Gemma2 → bytes) |
| 2 | [Bolmo: Byteifying LMs](2512.15586-bolmo-byteifying-the-next-generation-of-language-models.md) (AI2) | Byteify an existing LLM with local encoder/decoder + boundary prediction | Byte-level 1B/7B models matching the subword source with <1% of the pretraining budget |
| 3 | [AdaptiVocab](2503.19693-adaptivocab-enhancing-llm-efficiency-in-focused-domains-through-lightw.md) | Domain n-gram tokens + exponential embedding init + light fine-tune | >25% fewer tokens (faster decode) at equal quality in focused domains |
| 4 | [TokAlign](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md) (ACL'25) / [TokAlign++](2605.13429-tokalign-advancing-vocabulary-adaptation-via-better-token-alignment.md) | Align vocabularies via token co-occurrence, then progressive fine-tune | Fast replacement; enables token-level KD between model families |
| 5 | [Token Distillation](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md) (ICLR'26) | Distil new-token embeddings from the original multi-token representation's attention behaviour | Best training-free-ish initialization; minutes |
| 6 | [OMP tokenizer transplantation](2506.06607-training-free-tokenizer-transplantation-via-orthogonal-matching-pursui.md) | Sparse anchor reconstruction, training-free | Strong zero-shot preservation; production tooling |
| 7 | [MATT](2510.21954-model-aware-tokenizer-transfer.md) | Model-aware transfer using attention-influence modeling | Better transfer to distinct-script languages |
| 8 | [Byte-prefix marginalization](2607.22334-cross-tokenizer-on-policy-distillation-via-byte-prefix-marginalization.md) | Re-express teacher next-token distribution over the student's tokens exactly via byte prefixes | **Cross-tokenizer on-policy distillation** without dropping probability mass |
| 9 | [Byte-level distillation](2604.07466-cross-tokenizer-llm-distillation-through-a-byte-level-interface.md) | Distil at the common byte interface | Simple, strong cross-tokenizer baseline |
| 10 | [VocabTailor](2508.15229-vocabtailor-dynamic-vocabulary-selection-for-downstream-tasks-in-small.md) (ACL'26) | Dynamic vocabulary selection for SLMs (offload embeddings, prune LM head per task) | Big memory cut for edge models |

**Also useful.**
* Cross-tokenizer distillation: [DWA-KD](2602.21669-dwa-kd-dual-space-weighting-and-time-warped-alignment-for-cross-tokeni.md), [Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation](2512.14954-cross-tokenizer-likelihood-scoring-algorithms-for-language-model-disti.md) (likelihood scoring), [ACTD](2608.29662-actd-anchor-based-cross-tokenizer-distillation-with-residual-regulariz.md),
  [Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch](2603.22056-dual-space-knowledge-distillation-with-key-query-matching-for-large-la.md), [Distilling Token-Trained Models into Byte-Level Models](2602.01007-distilling-token-trained-models-into-byte-level-models.md) (token-trained → byte-level).
* Vocabulary extension: [Teaching Old Tokenizers New Words](2512.03989-teaching-old-tokenizers-new-words-efficient-tokenizer-adaptation-for-p.md) (continued BPE training + pruning), [in-place tokenizer expansion](2607.15232-in-place-tokenizer-expansion-for-pre-trained-llms.md),
  [Beyond Initialization Loss](2608.03494-beyond-initialization-loss-a-systematic-study-of-token-embedding-initi.md) (embedding init study), [Vocabulary Expansion of Large Language Models via Kullback-Leibler-Based Self-Distillation](2508.15807-vocabulary-expansion-of-large-language-models-via-kullback-leibler-bas.md), [Vocabulary Customization for Efficient Domain-Specific LLM Deployment](2509.26124-vocabulary-customization-for-efficient-domain-specific-llm-deployment.md).
* Language adaptation: [Franken-Adapter](2502.08037-franken-adapter-cross-lingual-adaptation-of-llms-by-embedding-surgery.md).

**Recommendation.**
* For **speed in a domain or language**: extend or re-fit the vocabulary, initialize with OMP or Token Distillation, and
  heal on ~0.1–1B tokens (≈2–50 H100-h at 7–8B).
* For **distilling across model families**: use ALM or byte-prefix OPD rather than forcing the student onto the teacher's
  tokenizer.
* For a runtime, the LM head is a large share of small-model decode cost at 150K+ vocabularies. Vocabulary trimming
  (VocabTailor) and certified sub-vocabulary heads are cheap wins. Byte-level models (Bolmo) need a runtime with
  **boundary-aware patch decoding**.

## 🏆 Best of the best by impact score (top 10)

1. **[AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation](2503.19693-adaptivocab-enhancing-llm-efficiency-in-focused-domains-through-lightw.md)** (2025-08) — This work introduces AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains, and shows that AdaptiVocab reduces token usage by over 25% without …  
   _score 8.71 · 17 cites · 76▲ HF · [code](https://github.com/itay-nakash/AdaptiVocab)_
2. **[Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md)** (2025-10) — This work develops a principled cross-tokenizer distillation method, which is the first to enable effective distillation across fundamentally different tokenizers, while also substantially outperforming prior methods in …  
   _score 7.56 · NeurIPS 2025 · 42 cites · ~48–192 H100-h_
3. **[Bolmo: Byteifying the Next Generation of Language Models](2512.15586-bolmo-byteifying-the-next-generation-of-language-models.md)** (2026-02) — Bolmo is introduced, a family of fully open byte-level LLMs that approach the capabilities of subword-based systems, demonstrating that models operating on raw text encodings can scale competitively while offering …  
   _score 7.48 · 8 cites · 17▲ HF · [code](https://github.com/allenai/bolmo-core)_
4. **[Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation](2504.17025-optimizing-llms-for-italian-reducing-token-fertility-and-enhancing-eff.md)** (2025-04) — This work thoroughly compares a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and puts forward Semantic Alignment Vocabulary Adaptation (SAVA), a novel method that …  
   _score 5.74 · North American Chapter of the Association for Computational  · 7 cites · 17▲ HF · [code](https://github.com/SapienzaNLP/sava)_
5. **[Token Distillation: Attention-aware Input Embeddings For New Tokens](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md)** (2026-03) — By distilling representations obtained using the original tokenization, this paper shows that by distilling representations obtained using the original tokenization, one can quickly learn high-quality input embeddings …  
   _score 4.1 · ICLR 2026 · 7 cites · [code](https://github.com/konstantinjdobler/token-distillation) · ~0.02–0.17 H100-h_
6. **[TokAlign: Efficient Vocabulary Adaptation via Token Alignment](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md)** (2025-06) — This paper proposes an efficient method named TokAlign to replace the vocabulary of LLM from the token co-occurrences view, and further transfer the token-level knowledge between models, which costs as few as 5k steps …  
   _score 3.93 · ACL 2025 · 12 cites · [code](https://github.com/ZNLP/TokAlign) · ~661 H100-h_
7. **[Cross-Tokenizer LLM Distillation through a Byte-Level Interface](2604.07466-cross-tokenizer-llm-distillation-through-a-byte-level-interface.md)** (2026-04) — A simple but effective baseline called Byte-Level Distillation (BLD) is proposed which enables CTD by operating at a common interface across tokenizers: the byte level, suggesting that the byte level is a natural common …  
   _score 3.8 · Proceedings of the Second Workshop on Customizable NLP: Prog · 4 cites · 6▲ HF_
8. **[Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning](2505.09738-achieving-tokenizer-flexibility-in-language-models-through-heuristic-a.md)** (2025-05) — Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including …  
   _score 3.77 · 6 cites · 5▲ HF · [code](https://github.com/Tinycompany-AI/tokenadapt)_
9. **[Model-Aware Tokenizer Transfer](2510.21954-model-aware-tokenizer-transfer.md)** (2026-05) — Experiments show that MATT recovers a large fraction of the original model's performance within a few GPU hours, outperforming heuristic baselines, and demonstrate that incorporating model-level signals offers a …  
   _score 3.56 · 6 cites · [code](https://github.com/CPJKU/wechsel)_
10. **[VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models](2508.15229-vocabtailor-dynamic-vocabulary-selection-for-downstream-tasks-in-small.md)** (2026-04) — VocabTailor is introduced, a novel decoupled dynamic vocabulary selection framework that addresses memory constraints through offloading embedding and implements a hybrid static-dynamic vocabulary selection strategy for …  
   _score 2.89 · Annual Meeting of the Association for Computational Linguist · 4 cites · [code](https://github.com/AwakenedInsects/VocabTailor)_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization](2607.22334-cross-tokenizer-on-policy-distillation-via-byte-prefix-marginalization.md)** (2026-07-24; 0▲, 2 cites) — The target exactly recovers the teacher-induced byte-prefix marginal when the relevant prefix does not span multiple teacher tokens (a condition satisfied at more than 99% of …
- **[ACTD: Anchor-Based Cross-Tokenizer Distillation with Residual Regularization](2608.29662-actd-anchor-based-cross-tokenizer-distillation-with-residual-regulariz.md)** (2026-08-30; 0▲, 0 cites) — An Anchor-Based Cross-Tokenizer Distillation with Residual Regularization with Residual Regularization (ACTD) bridges structural heterogeneity through vocabulary and sequence …
- **[In-Place Tokenizer Expansion for Pre-trained LLMs](2607.15232-in-place-tokenizer-expansion-for-pre-trained-llms.md)** (2026-07-16; 0▲, 0 cites) — Combining these reductions with the measured per-token cost of the larger vocabulary, we estimate a $2.2$-$3.7\times$ per-character decode speedup for these languages across our …
- **[Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension](2608.03494-beyond-initialization-loss-a-systematic-study-of-token-embedding-initi.md)** (2026-08-04; 0▲, 0 cites) — It is shown that initialization loss and initialization bits-per-byte (Init BPB) are unreliable predictors of downstream convergence, whereas lightweight CPT, as few as 50 steps, …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit](2506.06607-training-free-tokenizer-transplantation-via-orthogonal-matching-pursui.md) | 0.01 | 312 | H100 | OMP proves highly efficient: on a single H100 GPU, it required only 38 seconds (with k=8) and 74 seconds (with k=32).… |
| [Token Distillation: Attention-aware Input Embeddings For New Tokens](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md) | 0.02 | 0.17 | H100 | In return, ZeTT is faster at inference time (in our experiments on a single H100 80GB GPU, ZeTT took less than a minute).… |
| [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) | 48 | 192 | unspecified | The largest individual experiments run on a pod of 32 v4 TPU chips and take \approx 24 hours for transfer of Llama3 to byte-level tokenization, \approx 12 hours… |
| [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md) | 661 | 661 | unspecified | ZeTT requires more computation to train a hypernetwork for the parameters prediction, e.g., 661.2 GPU hours for Pythia{}_{\text{2.8B}}, while our method only co… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation](2503.19693-adaptivocab-enhancing-llm-efficiency-in-focused-domains-through-lightw.md) | 2025-08-01 | 8.71 | 17 | 76 |  | [✓](https://github.com/itay-nakash/AdaptiVocab) | This work introduces AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains, and shows that … |
| 2 | [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](2503.20083-universal-cross-tokenizer-distillation-via-approximate-likelihood-matc.md) | 2025-10-24 | 7.56 | 42 | 0 | NeurIPS 2025 |  | This work develops a principled cross-tokenizer distillation method, which is the first to enable effective distillation across fundamentally different … |
| 3 | [Bolmo: Byteifying the Next Generation of Language Models](2512.15586-bolmo-byteifying-the-next-generation-of-language-models.md) | 2026-02-09 | 7.48 | 8 | 17 |  | [✓](https://github.com/allenai/bolmo-core) | Bolmo is introduced, a family of fully open byte-level LLMs that approach the capabilities of subword-based systems, demonstrating that models operating on raw … |
| 4 | [Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation](2504.17025-optimizing-llms-for-italian-reducing-token-fertility-and-enhancing-eff.md) | 2025-04-23 | 5.74 | 7 | 17 | North American Chapter of the Associatio | [✓](https://github.com/SapienzaNLP/sava) | This work thoroughly compares a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and puts forward Semantic … |
| 5 | [Token Distillation: Attention-aware Input Embeddings For New Tokens](2505.20133-token-distillation-attention-aware-input-embeddings-for-new-tokens.md) | 2026-03-13 | 4.1 | 7 | 0 | ICLR 2026 | [✓](https://github.com/konstantinjdobler/token-distillation) | By distilling representations obtained using the original tokenization, this paper shows that by distilling representations obtained using the original … |
| 6 | [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](2506.03523-tokalign-efficient-vocabulary-adaptation-via-token-alignment.md) | 2025-06-04 | 3.93 | 12 | 0 | ACL 2025 | [✓](https://github.com/ZNLP/TokAlign) | This paper proposes an efficient method named TokAlign to replace the vocabulary of LLM from the token co-occurrences view, and further transfer the … |
| 7 | [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](2604.07466-cross-tokenizer-llm-distillation-through-a-byte-level-interface.md) | 2026-04-13 | 3.8 | 4 | 6 | Proceedings of the Second Workshop on Cu |  | A simple but effective baseline called Byte-Level Distillation (BLD) is proposed which enables CTD by operating at a common interface across tokenizers: the … |
| 8 | [Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning](2505.09738-achieving-tokenizer-flexibility-in-language-models-through-heuristic-a.md) | 2025-05-14 | 3.77 | 6 | 5 |  | [✓](https://github.com/Tinycompany-AI/tokenadapt) | Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional … |
| 9 | [Model-Aware Tokenizer Transfer](2510.21954-model-aware-tokenizer-transfer.md) | 2026-05-10 | 3.56 | 6 | 0 |  | [✓](https://github.com/CPJKU/wechsel) | Experiments show that MATT recovers a large fraction of the original model's performance within a few GPU hours, outperforming heuristic baselines, and … |
| 10 | [VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models](2508.15229-vocabtailor-dynamic-vocabulary-selection-for-downstream-tasks-in-small.md) | 2026-04-18 | 2.89 | 4 | 0 | Annual Meeting of the Association for Co | [✓](https://github.com/AwakenedInsects/VocabTailor) | VocabTailor is introduced, a novel decoupled dynamic vocabulary selection framework that addresses memory constraints through offloading embedding and … |
| 11 | [Advancing Polish Language Modeling through Tokenizer Optimization in the Bielik v3 7B and 11B Series](2604.10799-advancing-polish-language-modeling-through-tokenizer-optimization-in-t.md) | 2026-04-12 | 2.83 | 1 | 7 |  |  | This report details the transition from the universal Mistral-based tokenization to a dedicated Polish-optimized vocabulary for the Bielik v3 models, exploring … |
| 12 | [SRA: Span Representation Alignment for Large Language Model Distillation](2605.01205-sra-span-representation-alignment-for-large-language-model-distillatio.md) | 2026-06-02 | 2.69 | 4 | 0 | ACL 2026 |  | SRA is introduced, a novel framework that reframes CTKD through the physical lens of Multi-Particle Dynamical Systems and shifts the fundamental unit of … |
| 13 | [Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit](2506.06607-training-free-tokenizer-transplantation-via-orthogonal-matching-pursui.md) | 2025-06-07 | 2.43 | 4 | 3 |  |  | This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, … |
| 14 | [DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation](2602.21669-dwa-kd-dual-space-weighting-and-time-warped-alignment-for-cross-tokeni.md) | 2026-02-25 | 2.4 | 5 | 0 | Conference of the European Chapter of th |  | Extensive experiments across diverse NLP benchmarks demonstrate that DWA-KD outperforms state-of-the-art KD baselines, while ablation studies confirm the … |
| 15 | [Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization](2607.22334-cross-tokenizer-on-policy-distillation-via-byte-prefix-marginalization.md) | 2026-07-24 | 1.69 | 2 | 0 |  |  | The target exactly recovers the teacher-induced byte-prefix marginal when the relevant prefix does not span multiple teacher tokens (a condition satisfied at … |
| 16 | [Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery](2502.08037-franken-adapter-cross-lingual-adaptation-of-llms-by-embedding-surgery.md) | 2025-02-12 | 1.64 | 5 | 0 |  |  | The Franken-Adapter method is presented, offering a modular solution to transfer reasoning abilities across languages post hoc, and reveals the critical role … |
| 17 | [Cross-Tokenizer Likelihood Scoring Algorithms for Language Model Distillation](2512.14954-cross-tokenizer-likelihood-scoring-algorithms-for-language-model-disti.md) | 2026-05-06 | 1.07 | 2 | 0 |  |  | This work addresses vocabulary misalignment problem by uncovering an implicit recursive structure in the commonly deployed Byte-Pair Encoding (BPE) algorithm … |
| 18 | [Vocabulary Customization for Efficient Domain-Specific LLM Deployment](2509.26124-vocabulary-customization-for-efficient-domain-specific-llm-deployment.md) | 2025-09-30 | 1.02 | 1 | 0 | Accepted at NEURIPS 2025 CCFM W |  | This work designs an algorithm that extends an existing tokenizer while guaranteeing it never decreases tokenization efficiency: every input sequence is … |
| 19 | [Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models](2512.03989-teaching-old-tokenizers-new-words-efficient-tokenizer-adaptation-for-p.md) | 2026-03-23 | 0.91 | 2 | 0 |  |  | This work proposes continued BPE training that extends a pre-trained tokenizer by continuing the BPE merge learning process on new data and introduces … |
| 20 | [ACTD: Anchor-Based Cross-Tokenizer Distillation with Residual Regularization](2608.29662-actd-anchor-based-cross-tokenizer-distillation-with-residual-regulariz.md) | 2026-08-30 | 0.7 | 0 | 0 | Accepted by EMNLP 2026 main c |  | An Anchor-Based Cross-Tokenizer Distillation with Residual Regularization with Residual Regularization (ACTD) bridges structural heterogeneity through … |
| 21 | [Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch](2603.22056-dual-space-knowledge-distillation-with-key-query-matching-for-large-la.md) | 2026-05-17 | 0.63 | 1 | 0 | IEEE International Conference on Acousti |  | A novel method is introduced, DSKD-CMA-GA, based on Generative Adversarial (GA) learning, to address the mismatched distributions between the keys and queries … |
| 22 | [TokAlign++: Advancing Vocabulary Adaptation via Better Token Alignment](2605.13429-tokalign-advancing-vocabulary-adaptation-via-better-token-alignment.md) | 2026-05-13 | 0.62 | 1 | 0 |  |  | This work introduces a method named TokAlign++ to improve vocabulary adaptation performance by learning better token alignment lexicon and shows that this … |
| 23 | [Vocabulary Expansion of Large Language Models via Kullback-Leibler-Based Self-Distillation](2508.15807-vocabulary-expansion-of-large-language-models-via-kullback-leibler-bas.md) | 2026-01-12 | 0.41 | 1 | 0 |  |  | This work introduces a mathematically grounded method for knowledge distillation via KL divergence, even when the original and extended models use different … |
| 24 | [When the Same Coefficients Reach Different Places: Asymmetric Realizability in Transplanting Tokenizers across](2601.00065-when-the-same-coefficients-reach-different-places-asymmetric-realizabi.md) | 2026-05-28 | 0.0 | 0 | 0 |  |  | Tokenizer transplant in cross-vocabulary model composition reconstructs donor-only embedding rows as weighted combinations over shared lexical anchors and … |
| 25 | [Distilling Token-Trained Models into Byte-Level Models](2602.01007-distilling-token-trained-models-into-byte-level-models.md) | 2026-02-01 | 0.0 | 0 | 0 |  |  | This paper proposes an efficient distillation recipe that converts existing token-trained LLMs into BLMs while retaining comparable capabilities, and … |
| 26 | [In-Place Tokenizer Expansion for Pre-trained LLMs](2607.15232-in-place-tokenizer-expansion-for-pre-trained-llms.md) | 2026-07-16 | 0.0 | 0 | 0 |  |  | Combining these reductions with the measured per-token cost of the larger vocabulary, we estimate a $2.2$-$3.7\times$ per-character decode speedup for these … |
| 27 | [Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary](2608.03494-beyond-initialization-loss-a-systematic-study-of-token-embedding-initi.md) | 2026-08-04 | 0.0 | 0 | 0 |  |  | It is shown that initialization loss and initialization bits-per-byte (Init BPB) are unreliable predictors of downstream convergence, whereas lightweight CPT, … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families](../../compression/knowledge-distillation/2606.09456-breaking-the-tokenizer-barrier-on-policy-distillation-across-model-fam.md) | Knowledge distillation (LLM → smaller LLM) | 2.05 |
| [CTPD: Cross Tokenizer Preference Distillation](../../compression/knowledge-distillation/2601.11865-ctpd-cross-tokenizer-preference-distillation.md) | Knowledge distillation (LLM → smaller LLM) | 2.02 |
