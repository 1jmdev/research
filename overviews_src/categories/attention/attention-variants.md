**Verdict.** Two practical changes to softmax attention have strong evidence.

1. **Output gating.** A head-specific sigmoid gate after SDPA ([[2505.06708|Gated Attention]], Qwen; NeurIPS'25 best
   paper). It was tested across 30 variants at 1.7B dense and 15B MoE on 3.5T tokens. It:
   * improves quality;
   * permits larger learning rates;
   * **removes the attention sink and massive activations**, so the model is easier to quantize and handles long
     context better.

   Qwen3-Next / Qwen3.5 ship it.
2. **KV-compact head designs** beyond GQA: MLA (DeepSeek), TPA (tensor product factorized Q/K/V), GTA, GVA. These
   match or beat GQA with 2–10× smaller caches. See also
   [`kv-cache/low-rank-and-latent`](../../kv-cache/low-rank-and-latent/README.md).

The large attention-sink literature has converged.
* **Why sinks exist.** Sinks and massive activations let deep Transformers **avoid over-mixing** and implement a
  "no-op" state ([[2504.02732]], [[2603.11487]], [[2601.22966]]).
* **What they cost.** They are the root of activation outliers that hurt quantization.
* **How to remove them safely.** Architectural fixes (gating, softpick, value-state gating) remove sinks without losing
  quality. Naive suppression does not.

### Hand ranking

| # | Paper | Change | Evidence | Why a builder cares |
| ---: | --- | --- | --- | --- |
| 1 | [[2505.06708\|Gated Attention]] (Qwen, NeurIPS'25 best paper) | Sigmoid gate on the SDPA output, per head | 30 variants, up to 15B MoE and 3.5T tokens | **Adopt in new models.** Quality, stability, no sinks, better long context |
| 2 | [[2501.06425\|Tensor Product Attention (T6)]] (NeurIPS'25 spotlight) | Contextual low-rank tensor factorization of Q/K/V; RoPE-compatible | Beats MHA/GQA/MLA at equal params with a **smaller KV cache** | Alternative to MLA; simple decode |
| 3 | [[2504.20966\|Softpick]] (ACL'26) | Rectified, not sum-to-one softmax replacement | 0% sink rate; lower kurtosis; **better low-bit quantization** | Quantization-friendly models |
| 4 | [[2504.02732\|Why do LLMs attend to the first token?]] | Theory + experiments: sinks prevent over-mixing | Depth, context and packing effects | Don't "fix" sinks blindly; keep a sink or gate |
| 5 | [[2504.00927\|Multi-Token Attention]] (Meta) | Convolutions over queries, keys and heads so weights depend on several tokens | Better long-context retrieval | Research-grade; kernel cost |
| 6 | [[2507.02754\|2-Simplicial Attention]] (Meta) | Trilinear attention in an efficient Triton kernel | **Better token efficiency** (steeper scaling exponent) for reasoning/math/code under a token budget | Relevant as data runs out |
| 7 | [[2609.13285\|Grouped Value Attention]] | Store grouped values; reconstruct keys with a linear map absorbed into the query; small decoupled RoPE key | −45–47% cache vs matched GQA | MLA-like savings with a simpler path |
| 8 | [[2503.09579\|Cost-optimal GQA]] (EMNLP'25) | Decouple head size from hidden size; jointly optimize model size and GQA config for a context length | Common GQA configs are far from optimal for long context | Use when designing a model for 128K+ |
| 9 | [[2606.20945\|Grouped Query Experts]] | MoE over query heads inside each GQA group; KV stays dense | Matches GQA with fewer active query heads | Compute savings without KV cost |
| 10 | [[2510.06477\|Sinks ↔ compression valleys]] | Massive activations cause both | 410M–120B models | Mix-Compress-Refine view of depth |
| 11 | [[2603.05498\|The Spike, the Sparse and the Sink]] | Massive activations act as implicit parameters; sinks modulate locally | Separates their roles | Guides mitigation |
| 12 | [[2606.04032\|Do Transformers need three projections?]] (ICML'26) | Q-K=V sharing etc. | Q-K=V gives **50% KV reduction** at par quality (1.2B) | Cheap KV saving |
| 13 | [[2505.21487\|Hardware-efficient attention for fast decoding]] (GTA/GLA) | Grouped-tied and grouped-latent attention designed for decode arithmetic intensity | ~2× faster decode than FlashMLA at parity | Serving-friendly MLA alternative |

**For model builders.**
* Default to **GQA or MLA + output gating + QK-norm**.
* Consider TPA/GVA/GTA if KV memory dominates your serving cost.
* Avoid architectures that create massive activations (see also [[2506.19697|Outlier-Safe Pre-Training]]) if you plan
  4-bit W/A/KV deployment.
