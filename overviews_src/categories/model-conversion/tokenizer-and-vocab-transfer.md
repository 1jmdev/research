**Verdict.** Changing a pretrained model's tokenizer is now cheap. The work splits into four jobs:

1. **Transplant a new vocabulary** (domain/language efficiency, or matching a teacher's vocabulary for KD). Initialize new
   embeddings training-free, then heal briefly:
   * [[2506.06607|OMP]]: new tokens as sparse combinations of shared anchor tokens (in mergekit's tokensurgeon);
   * [[2506.03523|TokAlign]]: token co-occurrence alignment, <2 CPU-hours vs 661 GPU-h for ZeTT's hypernetwork;
   * [[2510.21954|MATT]]: model-aware, uses attention dynamics;
   * [[2505.20133|Token Distillation]] (ICLR'26): new-token embeddings distilled from the attention behaviour of the
     original multi-token sequence, 2,500 tokens in <10 minutes.
2. **Domain vocabulary for speed.** [[2503.19693|AdaptiVocab]] cuts token usage >25% in focused domains with a
   lightweight fine-tune. Language-specific retokenization cuts fertility ([[2504.17025|Italian]],
   [[2604.10799|Bielik v3 Polish]]). Fewer tokens directly means fewer decode steps.
3. **Byteification** (subword → byte-level). [[2512.15586|Bolmo]] converts OLMo into a byte-level LM with local
   mLSTM encoder/decoder layers using **39.3B tokens (<1% of pretraining)**, matching the subword source. Earlier,
   [[2503.20083|ALM]] showed byte-level transfer by approximate likelihood matching.
4. **Cross-tokenizer distillation** (teacher and student use different vocabularies):
   * [[2503.20083|ALM]] (NeurIPS'25): the principled, general method;
   * [[2604.07466|byte-level distillation]]: a simple strong baseline;
   * [[2607.22334|byte-prefix marginalization]]: for **on-policy** distillation across families;
   * [[2605.01205|SRA]]: span alignment.

   This is what makes "distill Qwen into Llama-vocab" or "Gemma → my model" possible.

### Cost table

| Method | Job | Budget | H100-h | Basis |
| --- | --- | --- | ---: | --- |
| [[2506.06607\|OMP transplant]] | Vocabulary swap | training-free | **~0** | reported |
| [[2505.20133\|Token Distillation]] | Add new tokens | 2,500 tokens in <10 min on 1 GPU | **<0.2** | reported |
| [[2506.03523\|TokAlign]] (ACL'25) | Vocabulary replacement | <2 h CPU alignment + short fine-tune | **~1–20** | reported/estimate |
| [[2510.21954\|MATT]] | Model-aware transfer | ~250M tokens | **~5–10** (7B) | estimate |
| [[2503.19693\|AdaptiVocab]] | Domain vocabulary | ~8M tokens, 3×A6000 | **~1** | estimate |
| [[2503.20083\|ALM]] (NeurIPS'25) | Llama-3-8B → byte-level / cross-tokenizer KD | 32×TPU-v4 × 24 h | **~200** | reported (v4 ≈ 0.28 H100) |
| [[2512.15586\|Bolmo]] | OLMo-3 7B → byte-level | 39.3B tokens | **~1.2K** | estimate |

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [[2503.20083\|ALM: universal cross-tokenizer distillation]] (NeurIPS'25) | Approximate likelihood matching over aligned chunks; works for any tokenizer pair, including subword → byte | First effective cross-tokenizer distillation; also transfers tokenizers (Llama3/Gemma2 → bytes) |
| 2 | [[2512.15586\|Bolmo: Byteifying LMs]] (AI2) | Byteify an existing LLM with local encoder/decoder + boundary prediction | Byte-level 1B/7B models matching the subword source with <1% of the pretraining budget |
| 3 | [[2503.19693\|AdaptiVocab]] | Domain n-gram tokens + exponential embedding init + light fine-tune | >25% fewer tokens (faster decode) at equal quality in focused domains |
| 4 | [[2506.03523\|TokAlign]] (ACL'25) / [[2605.13429\|TokAlign++]] | Align vocabularies via token co-occurrence, then progressive fine-tune | Fast replacement; enables token-level KD between model families |
| 5 | [[2505.20133\|Token Distillation]] (ICLR'26) | Distil new-token embeddings from the original multi-token representation's attention behaviour | Best training-free-ish initialization; minutes |
| 6 | [[2506.06607\|OMP tokenizer transplantation]] | Sparse anchor reconstruction, training-free | Strong zero-shot preservation; production tooling |
| 7 | [[2510.21954\|MATT]] | Model-aware transfer using attention-influence modeling | Better transfer to distinct-script languages |
| 8 | [[2607.22334\|Byte-prefix marginalization]] | Re-express teacher next-token distribution over the student's tokens exactly via byte prefixes | **Cross-tokenizer on-policy distillation** without dropping probability mass |
| 9 | [[2604.07466\|Byte-level distillation]] | Distil at the common byte interface | Simple, strong cross-tokenizer baseline |
| 10 | [[2508.15229\|VocabTailor]] (ACL'26) | Dynamic vocabulary selection for SLMs (offload embeddings, prune LM head per task) | Big memory cut for edge models |

**Also useful.**
* Cross-tokenizer distillation: [[2602.21669|DWA-KD]], [[2512.14954]] (likelihood scoring), [[2608.29662|ACTD]],
  [[2603.22056]], [[2602.01007]] (token-trained → byte-level).
* Vocabulary extension: [[2512.03989]] (continued BPE training + pruning), [[2607.15232|in-place tokenizer expansion]],
  [[2608.03494]] (embedding init study), [[2508.15807]], [[2509.26124]].
* Language adaptation: [[2502.08037|Franken-Adapter]].

**Recommendation.**
* For **speed in a domain or language**: extend or re-fit the vocabulary, initialize with OMP or Token Distillation, and
  heal on ~0.1–1B tokens (≈2–50 H100-h at 7–8B).
* For **distilling across model families**: use ALM or byte-prefix OPD rather than forcing the student onto the teacher's
  tokenizer.
* For a runtime, the LM head is a large share of small-model decode cost at 150K+ vocabularies. Vocabulary trimming
  (VocabTailor) and certified sub-vocabulary heads are cheap wins. Byte-level models (Bolmo) need a runtime with
  **boundary-aware patch decoding**.
