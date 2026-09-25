**Verdict.** This bucket holds cross-cutting KV work: surveys, KV as a *communication medium* between models, streaming
video KV memory, lossless transfer codecs, and some sobering correctness results.

| # | Paper | Why it matters |
| ---: | --- | --- |
| 1 | [[2510.03215\|Cache-to-Cache (C2C)]] (ICLR'26) | LLMs **communicate by fusing KV caches** instead of text: a small projector and fuser (~9 GPU-hours to train) beats text-to-text multi-LLM pipelines in both quality and latency. Foundation for multi-model runtimes |
| 2 | [[2503.00540\|ReKV]] (ICLR'25) | Streaming video QA: sliding-window encoding, KV offloaded to RAM/disk, **in-context KV retrieval** on each question. Training-free |
| 3 | [[2603.19664\|The Residual Stream Is All You Need]] | K and V are deterministic projections of the residual stream. Caching one residual vector per token and recomputing K and V is **bit-exact**, a memory/compute trade for compute-rich GPUs (see also XQuant in KV quantization) |
| 4 | [[2604.15409\|The Illusion of Equivalence]] | **KV cache ON vs OFF gives different tokens in FP16** (100% divergence on GSM8K, greedy) because of accumulation order. Matters for reproducibility tests and speculative-decoding verification |
| 5 | [[2605.01708\|SplitZip]] | GPU-friendly **lossless** KV compressor for prefill→decode transfer that keeps up with prefill throughput |
| 6 | [[2505.13109\|FreeKV]] | Speculative KV retrieval off the critical path + fine-grained correction; hybrid CPU/GPU layouts |
| 7 | [[2511.00868\|FlexiCache]] (MLSys'26) | Classify KV heads as temporally stable or unstable: keep all pages for unstable heads and only top-k for stable ones |
| 8 | [[2506.05344\|SparseMM]] (ICCV'25) | <5% of heads are "visual heads", so give them the KV budget in MLLMs |
| 9 | [[2604.10044\|LoopGuard]] | Attention-based KV policies can **amplify repetition loops**; detect and intervene |
| 10 | [[2503.24000\|Rethinking KV compression for serving]] (MLSys) | Why KV compression rarely ships: throughput with real kernels, longer outputs after compression, and missing benchmarks |
| 11 | [[2607.08057\|System-aware KV survey]] (ACL'26 F) | Temporal/spatial/structural taxonomy of KV infrastructure |
| 12 | [[2512.01953\|KV Pareto]] | Joint Pareto of KV quantization, weight quantization and chunked prefill for edge deployment |

**Runtime notes.**
* Build a KV abstraction that can be **quantized, evicted, offloaded, transferred, translated (C2C) and recomputed from
  the residual stream** behind one page-table API.
* Expect numeric non-equivalence across paths, and make your tests tolerant of it (see also
  [[2506.09501|numerical nondeterminism]]).
