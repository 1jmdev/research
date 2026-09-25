**Verdict.** Three separate problems are grouped here.

**1. Positional encoding and length extrapolation.** RoPE plus scaling (YaRN/LongRoPE2) is standard. The strongest
2025–26 ideas:
* **Hybrid RoPE/NoPE layers** ([[2501.18795|RoPE to NoPE]]; used in Llama-4-style iRoPE and Command-A).
* **Attention temperature scaling with log n** (Scalable-Softmax, critical scaling). Qwen and Llama already use log-n
  scaling.
* **Data-dependent position** (PaTH, the Forgetting Transformer's forget gate, RePo).
* **Dropping positional embeddings to extend context** ([[2512.12167|DroPE]]).

**2. Training for long context.**
* LongRoPE2 mixed-window training.
* Synthetic long data (NExtLong, LongMagpie).
* Short-to-long preference optimization (LongPO, SoLoPO).
* Restoring short-context quality (LongReD).
* Test-time training that turns the context into weights (TTT-E2E, PERK).

**3. The uncomfortable evaluation truth.**
* Models claiming 128K–1M fail **NoLiMa** (non-literal needles) at modest lengths.
* Performance falls **13.9–85% with length even when retrieval is perfect** ([[2510.05381]]).

This is why agentic/recursive approaches (**Recursive Language Models**, MemAgent) that *avoid* stuffing the context
are rising fast.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2512.24601\|Recursive Language Models]] | Inference paradigm | The prompt becomes an **environment variable in a REPL**; the model inspects, decomposes and recursively calls itself on snippets | 100× beyond the context window; beats compaction and CodeAct; post-trained RLM ~48 H100-h |
| 2 | [[2507.02259\|MemAgent]] (ICLR'26 oral) | RL memory agent | Read in segments, **overwrite a fixed memory**, trained with multi-conversation DAPO | 8K-context model trained at 32K extrapolates to 3.5M QA with <5% loss |
| 3 | [[2503.02130\|Forgetting Transformer (FoX)]] (ICLR'25) | Architecture | **Data-dependent forget gate on softmax attention logits**; no positional embeddings; FlashAttention-compatible | Beats the Transformer on long-context LM and extrapolation |
| 4 | [[2512.23675\|TTT-E2E]] | Test-time training | SWA Transformer that keeps learning on the context by next-token prediction; meta-learned init | Scales with context like full attention at constant per-token cost (3B/164B tokens) |
| 5 | [[2502.20082\|LongRoPE2]] (ICML'25) | RoPE extension | Under-trained high RoPE dims explain OOD; needle-driven evolutionary rescale + mixed-window training | Near-lossless 128K extension with 10B tokens, short quality kept |
| 6 | [[2501.19399\|Scalable-Softmax (SSMax)]] | Softmax fix | Scale logits by s·log n so attention does not flatten with length | Better pretraining loss and long-context retrieval |
| 7 | [[2505.16381\|PaTH]] (NeurIPS'25) | Data-dependent PE | Accumulated **Householder** transforms instead of fixed rotations | More expressive than RoPE; efficient parallel training |
| 8 | [[2501.18795\|RoPE to NoPE and back]] (NeurIPS'25) | Hybrid PE | Interleave RoPE (local) and NoPE (global) layers + QK-norm | Better long-context performance and efficiency |
| 9 | [[2502.05167\|NoLiMa]] (ICML'25) | Benchmark | Needles with minimal lexical overlap | Most "128K" models degrade sharply by 32K. **Use it** |
| 10 | [[2510.05381\|Context length alone hurts]] (EMNLP F'25) | Analysis | Accuracy drops with length even with perfect retrieval (and even with whitespace padding) | Motivates recite-then-solve prompting and context engineering |
| 11 | [[2502.13922\|LongPO]] (ICLR'25) | Alignment | Self-generated short-to-long preference pairs | Long-context alignment without annotation |
| 12 | [[2512.12167\|DroPE: Dropping positional embeddings]] | Extension | Remove PE after pretraining + short recalibration | Zero-shot context extension without long fine-tuning |
| 13 | [[2502.08910\|InfiniteHiP]] | Inference | Hierarchical token pruning + adaptive RoPE + KV offload | 3M tokens on one GPU |
| 14 | [[2505.07793\|Overflow prevention for recurrent LLMs]] | Recurrent inference | Chunk-based inference prevents memory overflow in SSM/linear models | Big long-context gains for recurrent models, training-free |
| 15 | [[2510.05554\|Critical attention scaling]] (ICLR'26) | Theory | Phase transition: **β_n ≍ log n** is the critical scale | Justifies log-n attention temperature |

**For a runtime.**
* Implement the RoPE scaling family (linear/NTK/YaRN/LongRoPE, per-layer NoPE, **log-n attention temperature**, iRoPE
  chunked attention) as config options. Get the numerics exactly right: pre- vs post-RoPE caching, FP32 angles.
* For >1M-token products, favour **agentic context management** (RLM/MemAgent-style tool loops) plus sparse attention
  and KV offloading over raw window size. See [`context-compression`](../../context-compression/README.md).
