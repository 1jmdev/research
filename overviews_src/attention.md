# Attention & sequence mixers: state of the art, 2025–2026

> Synthesis of [`papers/attention`](../papers/attention/README.md). For converting an existing model to these
> architectures (MHA→MLA, Transformer→linear/hybrid), see [model conversion](model-conversion.md). For cache-level
> tricks, see [KV cache](kv-cache.md).

## TL;DR

The 2025–26 consensus in frontier open models:

* **Hybrid layers.** Most layers are linear/recurrent (Gated DeltaNet / KDA / Mamba-style) or sparse, with a few
  full-attention layers interleaved.
  * [[2507.06457|Systematic analysis of hybrid linear attention]] recommends GatedDeltaNet- or HGRN-2-style layers
    at a **linear:full ratio of 3:1 to 6:1**.
  * Production examples: Ling/Ring "Every Attention Matters" ([[2510.19338]], ~1/10 the inference cost of a 32B dense
    model), MiniMax-01 (lightning attention), Qwen3-Next / Qwen3.5 (Gated DeltaNet), Nemotron-H / Nemotron 3 (Mamba-2)
    and Kimi Linear (KDA). See the [technical reports](../papers/models-and-architectures/technical-reports/README.md).
* **Trainable block-sparse attention for the full-attention layers.**
  * [[2502.11089|NSA]] (DeepSeek): compression + selection + sliding window, trained end-to-end, with fast kernels
    for decode, forward *and* backward.
  * [[2502.13189|MoBA]] (Moonshot): MoE-style routing over KV blocks.
  * [[2606.13392|MiniMax Sparse Attention]]: GQA-based block sparsity. At 1M context it cuts attention compute
    28.4× and reaches 14.2× prefill / 7.6× decode wall-clock on H800, on par with GQA quality.
  * [[2603.12201|IndexCache]]: for DeepSeek-style DSA, reuse the top-k indices across layers to remove 75% of the
    indexer cost.
* **Gated attention output.** [[2505.06708|Gated Attention]] (Qwen team, NeurIPS 2025 best paper) adds a
  head-specific **sigmoid gate after SDPA**. It improves quality, stability and learning-rate tolerance, and
  **removes attention sinks and massive activations**, which directly helps quantization. It is cheap to adopt.
* **Low-precision attention kernels.**
  * [[2505.11594|SageAttention3]]: FP4 microscaling attention at 1038 TOPS on an RTX 5090, 5× over the fastest
    FlashAttention there. It also explores 8-bit training attention.
  * [[2603.05451|FlashAttention-4]]: Blackwell-specific pipelining because tensor cores doubled while the exp/SMEM
    units did not. Up to 1613 TFLOPs/s on B200 BF16, 1.3× over cuDNN.
  * [[2501.01005|FlashInfer]] (MLSys'25) is the attention engine to embed: block-sparse / paged layouts, JIT
    variants, load-balanced scheduling.

## Sub-areas

### Sparse attention ([folder](../papers/attention/sparse-attention/README.md))

| Use case | Pick | Why |
| --- | --- | --- |
| Training a new long-context model | [[2502.11089\|NSA]], [[2502.13189\|MoBA]], [[2606.13392\|MSA]], [[2607.02980\|HiLS]] | Natively trainable; HiLS extrapolates 64× beyond training length |
| Training-free prefill acceleration | [[2502.20766\|FlexPrefill]] (ICLR'25 oral), [[2504.16083\|MMInference]] (VLM, 8.3× at 1M) | Per-head adaptive patterns chosen from the prompt |
| Serving system | [[2502.14866\|LServe]] (MLSys'25) | Static streaming heads + dynamic page sparsity: 2.9× prefill, 1.3–2.1× decode over vLLM |
| 100M-token memory | [[2603.23516\|Memory Sparse Attention]] | Doc-wise RoPE + sparse attention; 100M tokens on 2×A800 |

### Linear attention / delta rule ([folder](../papers/attention/linear-attention/README.md))

* **Gated DeltaNet** is the workhorse, and [[2605.22791|Gated DeltaNet-2]] decouples erase and write gates. KDA
  (Kimi Delta Attention) is used in Kimi Linear.
* Test-time training (TTT) reframed: [[2505.23884|LaCT]] uses huge chunk updates to fix TTT's <5% FLOP
  utilization. [[2602.21204]] shows TTT with KV binding *is* linear attention, which yields parallel forms.
* Sequence parallelism for linear attention: [[2502.07563|LASP-2]] (36.6% faster than Ring Attention at 2M tokens).

### SSM / recurrent ([folder](../papers/attention/state-space-and-recurrent/README.md))

* [[2503.14456|RWKV-7 "Goose"]]: generalized delta rule; recognizes all regular languages; 3B SoTA multilingual.
* [[2603.15569|Mamba-3]] (ICLR'26): complex-valued state, MIMO formulation, better state tracking with no extra
  decode latency.
* [[2502.10297|DeltaProduct]]: Householder products for tunable state tracking.
* [[2504.13173|Miras]]: unifies attention and RNNs as associative memory with attentional bias.
* Reasoning with SSMs: [[2504.10449|M1]] (hybrid Mamba reasoning model; higher accuracy than R1-distilled
  transformers at a fixed *time* budget).

### Hybrid design ([folder](../papers/attention/hybrid-architectures/README.md))

* Design studies: [[2510.04800]] (inter- versus intra-layer fusion), [[2606.15378]] (efficient-attention choice
  mostly affects *how fast* long-context ability emerges) and [[2606.20097|HydraHead]] (hybridize along the *head*
  axis).
* Pitfall: [[2606.11052|Attention amnesia]]. CoT fine-tuning can break long-range recall in hybrids; QK-Restore
  fixes it at zero training cost.

### Attention variants and sinks ([folder](../papers/attention/attention-variants/README.md))

* KV-lean head designs: MLA (DeepSeek), [[2501.06425|Tensor Product Attention]] (factorized QKV that beats
  MHA/MQA/GQA/MLA at a smaller KV) and [[2502.07864|TransMLA]] (convert GQA to MLA to reuse DeepSeek kernels:
  93% KV compression, 10.6× at 8K).
* Why sinks exist: [[2504.02732]] (to avoid over-mixing), [[2510.06477]] (sinks and compression valleys both come
  from massive activations) and [[2603.05498]] (pre-norm enables both). Ways to remove sinks:
  [[2505.06708|gated attention]] and [[2504.20966|softpick]] (rectified, not-sum-to-one softmax).
  Removing sinks makes low-bit quantization of activations and KV much easier.

### Long context & position ([folder](../papers/attention/long-context-and-position/README.md))

* [[2503.02130|Forgetting Transformer]] adds a data-dependent forget gate on attention logits and beats RoPE-only
  transformers on long context.
* [[2512.23675|End-to-end TTT for long context]] keeps latency constant, 2.7× faster than full attention at 128K.
* Evaluation reality check: [[2502.05167|NoLiMa]]. At 32K, 11 of 13 "128K" models drop below 50% of their
  short-context baseline once literal matching is removed.
* System-level alternatives: [[2512.24601|Recursive Language Models]] (the model recursively calls itself over
  chunks, handling inputs 100× beyond its window) and [[2510.17800|Glyph]] (render text as images: 3–4× token
  compression).

## What to implement first (runtime)

1. FlashInfer-style paged, block-sparse attention with GQA/MLA decode kernels and FP8 (then FP4) attention paths.
2. A **hybrid-layer executor**: recurrent-state caching for linear/SSM layers next to paged KV for full-attention
   layers, including prefix caching of recurrent states (chunk checkpoints).
3. Top-k block selection kernels (NSA/DSA/MSA-style) with cross-layer index reuse ([[2603.12201|IndexCache]]).
4. Gate-after-SDPA and softmax variants as kernel epilogues.
