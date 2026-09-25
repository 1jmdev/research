**Verdict.** Context compression is about sending fewer tokens *into* the model. It complements KV-cache compression, which
shrinks the state after prefill. The 2025–26 landscape has four practical families.

1. **Task-aware hard pruning for agents and RAG.** This is the most deployable family: it is model-agnostic and plain
   text.
   * [[2601.16746|SWE-Pruner]]: goal-conditioned skimming of code context; −23–54% tokens on SWE-Bench Verified with
     *higher* success, up to 14.8× on LongCodeQA. [[2607.18213|SWE-Pruner Pro]] uses the coder LLM itself as pruner.
   * [[2510.00446|LongCodeZip]]: function-level, then block-level, perplexity-based pruning; 5.6×.
   * [[2510.00615|ACON]] (Microsoft, ICML'26): optimizes compression guidelines for agent histories in natural
     language; −26–54% peak tokens, and small models get up to +46%.
   * [[2501.16214|Provence]]: a RAG context pruner that decides how much to prune.
2. **Optical / visual-text compression.** Render text as images and read it with a VLM.
   * [[2510.18234|DeepSeek-OCR]]: ~97% decoding precision at <10× compression, ~60% at 20×.
   * [[2510.17800|Glyph]] (Zhipu): 3–4× token compression; a 128K VLM handles ~1M-token tasks.
   * [[2601.20552|DeepSeek-OCR 2]]: causal visual flow.
   * A new "memory decay" direction: older context kept at lower visual resolution.
3. **Learned soft compression** (encoder → compact embeddings for the decoder):
   * [[2509.01092|REFRAG]] (Meta): compress retrieved chunks to embeddings and selectively expand; large TTFT speedups
     for RAG without accuracy loss;
   * [[2510.20535|ARC-Encoder]] (Kyutai): 4–8× fewer continuous tokens, portable across LLMs;
   * [[2606.09659|End-to-end compression at scale]]: encoder–decoder compressors that finally compete with KV
     compression and support skim-then-expand agents;
   * [[2503.10337|KV-Distill]].
4. **Context → parameters** ("read once, answer many"):
   * [[2506.06266|Cartridges]] (Hazy): train a small KV "cartridge" per corpus by self-study; ICL quality at 38.6× less
     memory and 26.4× higher throughput; composable;
   * [[2602.15902|Doc-to-LoRA]]: hypernetwork emits a LoRA per document instantly;
   * [[2603.13875|GradMem]]: test-time gradient writes into memory tokens.

**Prefill-side speedup without changing the prompt:** [[2502.02789|Speculative Prefill]] uses a small model to pick the
important prompt tokens and prefill only those (with positions kept) in the big model. Llama-3.1-405B-FP8 gets up to
7.66× TTFT.

### Hand ranking

| # | Paper | Family | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.18234\|DeepSeek-OCR]] | Optical | Render text as images; DeepEncoder + MoE decoder reads it | ~97% precision at <10×, ~60% at 20×; 200K+ pages/day on one A100 for data generation |
| 2 | [[2506.06266\|Cartridges]] | Context → KV params | Offline "self-study" training of a small KV cache per corpus | Matches ICL at 38.6× less memory, 26.4× throughput; extends effective context (128K → 484K) |
| 3 | [[2601.16746\|SWE-Pruner]] | Agent pruning | Goal-conditioned neural skimmer (0.6B) over code context | −23–54% tokens with higher SWE-Bench success; up to 14.84× single-turn |
| 4 | [[2510.00615\|ACON]] (ICML'26) | Agent history | Failure-driven optimization of compression guidelines; distil into small compressors | −26–54% peak tokens; +46% for small-LM agents |
| 5 | [[2502.02789\|Speculative Prefill]] | Prefill | Small model estimates token importance; big model prefills only selected tokens | Up to 7.66× TTFT and 7× QPS on 405B |
| 6 | [[2509.01092\|REFRAG]] (Meta) | Soft RAG | Chunk embeddings replace retrieved tokens; RL policy expands chunks when needed | Large TTFT speedups for RAG with no accuracy loss |
| 7 | [[2510.17800\|Glyph]] | Optical | LLM-driven search of rendering configs + VLM continual pretraining | 3–4× compression; ~4× faster prefill/decoding; 1M-token tasks with a 128K VLM |
| 8 | [[2510.00446\|LongCodeZip]] | Code pruning | Coarse (function) + fine (block) conditional-perplexity selection under a budget | Up to 5.6× compression without task loss |
| 9 | [[2602.15902\|Doc-to-LoRA]] | Context → LoRA | Hypernetwork maps a document to an adapter in one pass | Answer without re-reading context; lower KV memory and latency |
| 10 | [[2606.09659\|End-to-end context compression at scale]] | Soft | Architecture search over encoder–decoder compressors pretrained at scale | Improves the frontier vs KV compression; skim-and-expand agents |
| 11 | [[2510.20535\|ARC-Encoder]] | Soft | Portable encoder producing 4–8× fewer continuous tokens | Works across multiple decoder LLMs |
| 12 | [[2501.16214\|Provence]] | RAG pruning | Joint reranking + adaptive pruning of retrieved passages | Negligible loss at almost no cost |

**Also useful.**
* Agent memory: [[2601.07190|Active Context Compression]], [[2602.12108|Pensieve]] (models managing their own
  context).
* Long-context systems: [[2505.18092|QwenLong-CPRS]].
* Soft-token methods: [[2505.12215|GMSA]], [[2510.08907|SAC]], [[2502.11493|DAST]], [[2602.03784|ComprExIT]].
* Hard pruning: [[2603.19635|BEAVER]], [[2507.11942|DAC]], [[2602.01719|COMI]], [[2503.10720|AttentionRAG]].
* Evaluation: [[2505.00019]], [[2503.19114]].
* Position: [[2505.19147]] (data-centric compression).
* Text + vision + latent: [[2609.01507|LatentPress]], [[2511.15244|Context Cascade Compression]].

**Runtime checklist.**
* A pluggable context-compression stage before prefill: pruner model, optical renderer, or soft encoder.
* Speculative prefill with a small scorer model.
* **Cartridge/KV-artifact loading** as a first-class cache object, next to prefix caching.
* Adapter-from-document (D2L) as a request type.
* Skim-then-expand APIs for agents.
