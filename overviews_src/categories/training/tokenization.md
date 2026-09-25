**Verdict.** Tokenization is back as a first-class design decision, with three directions.

1. **Bigger and better vocabularies are nearly free wins.**
   * [[2501.16975|Over-Tokenized Transformer]] (ICML'25): decouple the input vocabulary (huge, multi-gram) from the
     output vocabulary. Log-linear gains in input vocabulary size; comparable to a 2× larger model at no extra compute.
   * [[2503.13423|SuperBPE]] (COLM'25): a pretokenization curriculum that learns *superword* tokens crossing
     whitespace. Up to ~33% fewer tokens and better downstream results (+4% average at 8B) at equal compute, because
     per-token difficulty becomes more uniform. [[2508.11857|SupraTok]] refines it.
   * Numbers: [[2502.09741|FoNE]] encodes each number as one token with Fourier features; 64× less data for 99%
     addition accuracy.
2. **Tokenizer-free / learned chunking is getting competitive.**
   * [[2507.07955|H-Net: dynamic chunking]] (Goomba/CMU): end-to-end learned segmentation. A byte-level H-Net beats a
     BPE Transformer at matched compute and data; ~4× data efficiency on DNA; biggest wins for Chinese and code.
   * [[2506.14761|AU-Net]] (Meta, NeurIPS'25): autoregressive U-Net over bytes.
   * [[2507.12720|FLEXITOKENS]], [[2603.03583|ByteFlow]], [[2501.10322|hierarchical AR transformers]].
   * Byte-level decoding speed: [[2605.08044|Fast BLT]] (parallel bytes + self-speculation / diffusion, >50% lower
     bandwidth) and [[2608.15454|dynamic multi-byte prediction]].
   * Converting existing models: see Bolmo in
     [`model-conversion/tokenizer-and-vocab-transfer`](../../model-conversion/tokenizer-and-vocab-transfer/README.md).
3. **Inference-time tokenization tricks.**
   * [[2506.01084|zip2zip]] (NeurIPS'25): online LZW "hypertokens" learned at inference; 15–40% fewer input and output
     tokens after a 10 GPU-hour PEFT uptrain.
   * [[2506.14123|Byte-level sampling from BPE models]]: exact conversion to byte/character LMs. This fixes
     prompt-boundary problems and allows ensembling models with different tokenizers.
   * [[2506.19004|Non-canonical tokenizations]]: models tolerate them, and character-level segmentation can help
     string tasks.

Measurement and fairness: [[2512.20757|TokSuite]] (same model, many tokenizers), [[2506.03149|causal tokenization
bias]], [[2509.05486|The Token Tax]] (a 2× token blow-up means ~4× training cost for many languages),
[[2506.03101]] (Zipf-based intrinsic metrics beat compression rate).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2501.16975\|Over-Tokenized Transformer]] (ICML'25) | Vocabulary scaling | Separate large multi-gram input vocabulary from the output vocabulary | Log-linear loss gains; ≈ a 2× larger baseline at no extra cost |
| 2 | [[2507.07955\|H-Net: dynamic chunking]] | Tokenizer-free | Learned content-dependent chunking inside a hierarchical network, end to end | Byte-level H-Net beats BPE Transformers at matched compute; large gains on code, Chinese and DNA |
| 3 | [[2503.13423\|SuperBPE]] (COLM'25) | Tokenizer | Two-stage BPE: subwords, then superwords across spaces | Fewer tokens per byte and better downstream results at equal compute |
| 4 | [[2506.14761\|AU-Net: from bytes to ideas]] (NeurIPS'25) | Tokenizer-free | Autoregressive U-Net pooling bytes → words → multi-word | Ties strong BPE baselines; deeper hierarchies promising |
| 5 | [[2506.01084\|zip2zip]] (NeurIPS'25) | Inference-time | LZW hypertokens created on the fly; model uptrained to read and write them | 15–40% fewer tokens; 10 GPU-hours of PEFT |
| 6 | [[2502.09741\|FoNE]] | Numbers | Fourier-feature single-token number embeddings | 64× data efficiency on 6-digit addition; 100% arithmetic accuracy |
| 7 | [[2605.08044\|Fast Byte Latent Transformer]] | Byte decoding | Parallel multi-byte decoding + BLT self-speculation / diffusion + verification | >50% lower estimated bandwidth cost vs BLT |
| 8 | [[2506.14123\|Sampling one byte at a time]] | Inference | Exact byte-level sampling from any BPE model | Fixes prompt-boundary artefacts; ensembling across tokenizers |
| 9 | [[2512.20757\|TokSuite]] | Evaluation | Identical models trained with different tokenizers + multilingual robustness benchmark | Isolates tokenizer effects |
| 10 | [[2509.05486\|The Token Tax]] | Fairness/economics | Token fertility vs accuracy and cost across African languages | Fertility predicts accuracy drops; 2× tokens → 4× cost |

**Also useful.**
* Tokenizer design: [[2505.24689|SCRIPT-BPE]], [[2502.00894|MorphBPE]], [[2601.21665|AdaptBPE]],
  [[2609.15991|Functionalizer]], [[2608.00837|Pruned BPE]], [[2506.10766]] (multilingual tokenizers give plasticity).
* Robustness: [[2506.01687|StochasTok]], [[2511.05578]] (ill-formed UTF-8 from byte-level tokenizers).
* Proxy compression training: [[2602.04289]] (ICML'26).
* Pixel fallback: [[2504.02122]].
* Position paper: [[2601.13260]].

**Recommendation.**
* *Model builders:* use a SuperBPE-style tokenizer with a large (≥200K) input vocabulary (over-tokenization) and
  special handling for numbers. Watch H-Net/AU-Net for the next generation.
* *Runtime builders:* handle byte-level/hierarchical models (patch boundaries, multi-byte decode). Implement exact
  token-healing / byte-level sampling at prompt boundaries, and stateful incremental tokenization for agents (see
  serving `_general`, TokTier).
