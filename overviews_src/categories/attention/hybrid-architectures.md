**Verdict.** Hybrid is the default architecture for new efficient LLMs in 2026: a few full or MLA attention layers plus
many linear/SSM/SWA layers. Production examples:

| Family | Recipe |
| --- | --- |
| Nemotron-H → Nemotron Nano 2 → Nemotron 3 | Mamba-2 + attention, MoE ([[2504.03624]], [[2508.14444]], [[2512.20856]]) |
| Falcon-H1 | Parallel attention + Mamba heads ([[2507.22448]]) |
| Jet-Nemotron | PostNAS search of the linear/full layout ([[2508.15884]]) |
| Hunyuan-TurboS | Mamba-Transformer MoE ([[2505.15431]]) |
| Qwen3-Next / Qwen3.5 | Gated DeltaNet + gated attention |
| Kimi Linear | KDA + MLA |
| MiniMax-01 / M1 | Lightning + softmax |
| Olmo Hybrid | GDN in place of SWA ([[2604.03444]]) |

Design findings that recur across the systematic studies:
* **Retrieval is carried almost entirely by the full-attention layers** ([[2510.19861]], [[2606.15378]]). The efficient
  layers mostly shape optimization and local modeling.
* A **1:3 to 1:6** full:efficient ratio is the sweet spot.
* **Intra-layer (parallel-head) hybrids** slightly beat inter-layer ones ([[2510.04800]]).
* **Smaller SWA windows can help long context** ([[2509.24552]]): they force the recurrent memory to learn.
* Hybrids have their own failure modes. CoT-SFT can destroy long-range recall
  ([[2606.11052|Attention Amnesia]], fixed by restoring W_Q/W_K). Massive activations cluster before full-attention
  layers.

### Hand ranking (architecture papers in this category)

| # | Paper | Contribution | Take-away |
| ---: | --- | --- | --- |
| 1 | [[2510.04800\|Hybrid Architectures: Systematic Analysis]] (Meta) | Inter- vs intra-layer fusion; scaling to 4T tokens; efficiency | Hybrids beat homogeneous models in both quality and speed. Intra-layer fusion is best |
| 2 | [[2604.03444\|Olmo Hybrid]] | Theory (hybrids can express tasks beyond both parents, e.g. code execution) + a **fully open 7B** hybrid (GDN replaces SWA) | Beats Olmo 3 7B. The best open, reproducible hybrid recipe |
| 3 | [[2507.06607\|SambaY / Gated Memory Unit]] (Microsoft, NeurIPS'25) | Decoder-hybrid-decoder: a cross-decoder shares SSM memory readouts via GMUs; no positional encoding | Up to 10× decode throughput for **long generation** (Phi-4-mini-flash-reasoning) |
| 4 | [[2510.07019\|Native Hybrid Attention]] (ACL'26) | One softmax over **linear-RNN KV slots + a sliding window**; a single hyperparameter moves between linear and full | Unified intra/inter-layer hybrid with a Triton kernel |
| 5 | [[2510.07318\|Artificial Hippocampus Networks]] | Sliding-window KV (lossless short-term) + an RNN compressing out-of-window context; train only the AHN by self-distillation | Retrofits open LLMs: ~10 h on 32 A100s for 7B (≈100 H100-h) |
| 6 | [[2606.15378\|Rethinking efficient attention in hybrids]] | Scaling + mechanism analysis | Long-range retrieval lives in full attention; hybrids converge given enough data |
| 7 | [[2602.11761\|MiniCPM-SALA]] | **Sparse (InfLLM-V2) + linear (Lightning)** at 1:3, hybrid positional encoding, continual-training conversion | 9B ultra-long-context hybrid from a pretrained dense model |
| 8 | [[2606.11052\|Attention Amnesia]] (EMNLP'26) | CoT-SFT breaks long-range recall in hybrids; **QK-Restore** reverts W_Q and W_K | Must-know post-training pitfall |
| 9 | [[2509.24552\|Short window attention → long-term memory]] | SWAX: short SWA windows force the xLSTM memory to learn | Train with small or stochastic windows |
| 10 | [[2503.01868\|StripedHyena 2 (multi-hybrid)]] | Convolutional multi-hybrids co-designed with kernels at 40B | 1.2–2.9× faster training than optimized Transformers |
| 11 | [[2506.00744\|Blending KV and fast-weight memory]] (NeurIPS'25) | Three ways to fuse softmax KV memory with fast weights | Principled hybrid memory designs |
| 12 | [[2603.26380\|Switch Attention]] (EMNLP'26) | Per-token, per-layer **routing** between full attention and SWA | Dynamic hybrids |

**For a runtime.**
1. Support heterogeneous layer types in one model: full/MLA, SWA, GDN/KDA/Mamba-2, cross-layer KV. Each needs its own
   cache object: KV pages, a ring buffer, or a fixed recurrent state.
2. Size the scheduler memory per request type (state size vs KV growth).
3. Prefix caching, speculative decoding and PD disaggregation must all handle recurrent-state snapshots. See
   [[2608.30386|DASC]] for compressing that state in serving.
