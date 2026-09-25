# Attention & sequence mixers: state of the art, 2025–2026

> Synthesis of [`papers/attention`](../papers/attention/README.md). For converting an existing model to these
> architectures (MHA→MLA, Transformer→linear/hybrid), see [model conversion](model-conversion.md). For cache-level
> tricks, see [KV cache](kv-cache.md).

## TL;DR

The 2025–26 consensus in frontier open models:

* **Hybrid layers.** Most layers are linear/recurrent (Gated DeltaNet / KDA / Mamba-style) or sparse, with a few
  full-attention layers interleaved.
  * [Systematic analysis of hybrid linear attention](../papers/attention/linear-attention/2507.06457-a-systematic-analysis-of-hybrid-linear-attention.md) recommends GatedDeltaNet- or HGRN-2-style layers
    at a **linear:full ratio of 3:1 to 6:1**.
  * Production examples: Ling/Ring "Every Attention Matters" ([Every Attention Matters](../papers/models-and-architectures/technical-reports/2510.19338-every-attention-matters-an-efficient-hybrid-architecture-for-long-cont.md), ~1/10 the inference cost of a 32B dense
    model), MiniMax-01 (lightning attention), Qwen3-Next / Qwen3.5 (Gated DeltaNet), Nemotron-H / Nemotron 3 (Mamba-2)
    and Kimi Linear (KDA). See the [technical reports](../papers/models-and-architectures/technical-reports/README.md).
* **Trainable block-sparse attention for the full-attention layers.**
  * [NSA](../papers/attention/sparse-attention/2502.11089-native-sparse-attention-hardware-aligned-and-natively-trainable-sparse.md) (DeepSeek): compression + selection + sliding window, trained end-to-end, with fast kernels
    for decode, forward *and* backward.
  * [MoBA](../papers/attention/sparse-attention/2502.13189-moba-mixture-of-block-attention-for-long-context-llms.md) (Moonshot): MoE-style routing over KV blocks.
  * [MiniMax Sparse Attention](../papers/attention/sparse-attention/2606.13392-minimax-sparse-attention.md): GQA-based block sparsity. At 1M context it cuts attention compute
    28.4× and reaches 14.2× prefill / 7.6× decode wall-clock on H800, on par with GQA quality.
  * [IndexCache](../papers/attention/sparse-attention/2603.12201-indexcache-accelerating-sparse-attention-via-cross-layer-index-reuse.md): for DeepSeek-style DSA, reuse the top-k indices across layers to remove 75% of the
    indexer cost.
* **Gated attention output.** [Gated Attention](../papers/attention/attention-variants/2505.06708-gated-attention-for-large-language-models-non-linearity-sparsity-and-a.md) (Qwen team, NeurIPS 2025 best paper) adds a
  head-specific **sigmoid gate after SDPA**. It improves quality, stability and learning-rate tolerance, and
  **removes attention sinks and massive activations**, which directly helps quantization. It is cheap to adopt.
* **Low-precision attention kernels.**
  * [SageAttention3](../papers/attention/kernels-and-io-aware/2505.11594-sageattention3-microscaling-fp4-attention-for-inference-and-an-explora.md): FP4 microscaling attention at 1038 TOPS on an RTX 5090, 5× over the fastest
    FlashAttention there. It also explores 8-bit training attention.
  * [FlashAttention-4](../papers/attention/kernels-and-io-aware/2603.05451-flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmet.md): Blackwell-specific pipelining because tensor cores doubled while the exp/SMEM
    units did not. Up to 1613 TFLOPs/s on B200 BF16, 1.3× over cuDNN.
  * [FlashInfer](../papers/attention/kernels-and-io-aware/2501.01005-flashinfer-efficient-and-customizable-attention-engine-for-llm-inferen.md) (MLSys'25) is the attention engine to embed: block-sparse / paged layouts, JIT
    variants, load-balanced scheduling.

## Sub-areas

### Sparse attention ([folder](../papers/attention/sparse-attention/README.md))

| Use case | Pick | Why |
| --- | --- | --- |
| Training a new long-context model | [NSA](../papers/attention/sparse-attention/2502.11089-native-sparse-attention-hardware-aligned-and-natively-trainable-sparse.md), [MoBA](../papers/attention/sparse-attention/2502.13189-moba-mixture-of-block-attention-for-long-context-llms.md), [MSA](../papers/attention/sparse-attention/2606.13392-minimax-sparse-attention.md), [HiLS](../papers/attention/sparse-attention/2607.02980-hierarchical-sparse-attention-done-right-toward-infinite-context-model.md) | Natively trainable; HiLS extrapolates 64× beyond training length |
| Training-free prefill acceleration | [FlexPrefill](../papers/attention/sparse-attention/2502.20766-flexprefill-a-context-aware-sparse-attention-mechanism-for-efficient-l.md) (ICLR'25 oral), [MMInference](../papers/attention/sparse-attention/2504.16083-mminference-accelerating-pre-filling-for-long-context-vlms-via-modalit.md) (VLM, 8.3× at 1M) | Per-head adaptive patterns chosen from the prompt |
| Serving system | [LServe](../papers/attention/sparse-attention/2502.14866-lserve-efficient-long-sequence-llm-serving-with-unified-sparse-attenti.md) (MLSys'25) | Static streaming heads + dynamic page sparsity: 2.9× prefill, 1.3–2.1× decode over vLLM |
| 100M-token memory | [Memory Sparse Attention](../papers/attention/sparse-attention/2603.23516-msa-memory-sparse-attention-for-efficient-end-to-end-memory-model-scal.md) | Doc-wise RoPE + sparse attention; 100M tokens on 2×A800 |

### Linear attention / delta rule ([folder](../papers/attention/linear-attention/README.md))

* **Gated DeltaNet** is the workhorse, and [Gated DeltaNet-2](../papers/attention/linear-attention/2605.22791-gated-deltanet-2-decoupling-erase-and-write-in-linear-attention.md) decouples erase and write gates. KDA
  (Kimi Delta Attention) is used in Kimi Linear.
* Test-time training (TTT) reframed: [LaCT](../papers/attention/linear-attention/2505.23884-test-time-training-done-right.md) uses huge chunk updates to fix TTT's <5% FLOP
  utilization. [Test-Time Training with KV Binding Is Secretly Linear Attention](../papers/attention/linear-attention/2602.21204-test-time-training-with-kv-binding-is-secretly-linear-attention.md) shows TTT with KV binding *is* linear attention, which yields parallel forms.
* Sequence parallelism for linear attention: [LASP-2](../papers/serving-systems/distributed-inference-and-parallelism/2502.07563-lasp-2-rethinking-sequence-parallelism-for-linear-attention-and-its-hy.md) (36.6% faster than Ring Attention at 2M tokens).

### SSM / recurrent ([folder](../papers/attention/state-space-and-recurrent/README.md))

* [RWKV-7 "Goose"](../papers/attention/state-space-and-recurrent/2503.14456-rwkv-7-goose-with-expressive-dynamic-state-evolution.md): generalized delta rule; recognizes all regular languages; 3B SoTA multilingual.
* [Mamba-3](../papers/attention/state-space-and-recurrent/2603.15569-mamba-3-improved-sequence-modeling-using-state-space-principles.md) (ICLR'26): complex-valued state, MIMO formulation, better state tracking with no extra
  decode latency.
* [DeltaProduct](../papers/attention/state-space-and-recurrent/2502.10297-deltaproduct-improving-state-tracking-in-linear-rnns-via-householder-p.md): Householder products for tunable state tracking.
* [Miras](../papers/attention/state-space-and-recurrent/2504.13173-it-s-all-connected-a-journey-through-test-time-memorization-attentiona.md): unifies attention and RNNs as associative memory with attentional bias.
* Reasoning with SSMs: [M1](../papers/attention/state-space-and-recurrent/2504.10449-m1-towards-scalable-test-time-compute-with-mamba-reasoning-models.md) (hybrid Mamba reasoning model; higher accuracy than R1-distilled
  transformers at a fixed *time* budget).

### Hybrid design ([folder](../papers/attention/hybrid-architectures/README.md))

* Design studies: [Hybrid Architectures for Language Models](../papers/attention/hybrid-architectures/2510.04800-hybrid-architectures-for-language-models-systematic-analysis-and-desig.md) (inter- versus intra-layer fusion), [Rethinking the Role of Efficient Attention in Hybrid Architectures](../papers/attention/hybrid-architectures/2606.15378-rethinking-the-role-of-efficient-attention-in-hybrid-architectures.md) (efficient-attention choice
  mostly affects *how fast* long-context ability emerges) and [HydraHead](../papers/attention/hybrid-architectures/2606.20097-hydrahead-from-head-level-functional-heterogeneity-to-specialized-atte.md) (hybridize along the *head*
  axis).
* Pitfall: [Attention amnesia](../papers/attention/hybrid-architectures/2606.11052-attention-amnesia-in-hybrid-llms-when-cot-fine-tuning-breaks-long-rang.md). CoT fine-tuning can break long-range recall in hybrids; QK-Restore
  fixes it at zero training cost.

### Attention variants and sinks ([folder](../papers/attention/attention-variants/README.md))

* KV-lean head designs: MLA (DeepSeek), [Tensor Product Attention](../papers/attention/attention-variants/2501.06425-tensor-product-attention-is-all-you-need.md) (factorized QKV that beats
  MHA/MQA/GQA/MLA at a smaller KV) and [TransMLA](../papers/model-conversion/attention-conversion/2502.07864-transmla-multi-head-latent-attention-is-all-you-need.md) (convert GQA to MLA to reuse DeepSeek kernels:
  93% KV compression, 10.6× at 8K).
* Why sinks exist: [Why do LLMs attend to the first token?](../papers/attention/attention-variants/2504.02732-why-do-llms-attend-to-the-first-token.md) (to avoid over-mixing), [Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin](../papers/attention/attention-variants/2510.06477-attention-sinks-and-compression-valleys-in-llms-are-two-sides-of-the-s.md) (sinks and compression valleys both come
  from massive activations) and [The Spike, the Sparse and the Sink](../papers/attention/attention-variants/2603.05498-the-spike-the-sparse-and-the-sink-anatomy-of-massive-activations-and-a.md) (pre-norm enables both). Ways to remove sinks:
  [gated attention](../papers/attention/attention-variants/2505.06708-gated-attention-for-large-language-models-non-linearity-sparsity-and-a.md) and [softpick](../papers/attention/attention-variants/2504.20966-softpick-no-attention-sink-no-massive-activations-with-rectified-softm.md) (rectified, not-sum-to-one softmax).
  Removing sinks makes low-bit quantization of activations and KV much easier.

### Long context & position ([folder](../papers/attention/long-context-and-position/README.md))

* [Forgetting Transformer](../papers/attention/long-context-and-position/2503.02130-forgetting-transformer-softmax-attention-with-a-forget-gate.md) adds a data-dependent forget gate on attention logits and beats RoPE-only
  transformers on long context.
* [End-to-end TTT for long context](../papers/attention/long-context-and-position/2512.23675-end-to-end-test-time-training-for-long-context.md) keeps latency constant, 2.7× faster than full attention at 128K.
* Evaluation reality check: [NoLiMa](../papers/attention/long-context-and-position/2502.05167-nolima-long-context-evaluation-beyond-literal-matching.md). At 32K, 11 of 13 "128K" models drop below 50% of their
  short-context baseline once literal matching is removed.
* System-level alternatives: [Recursive Language Models](../papers/attention/long-context-and-position/2512.24601-recursive-language-models.md) (the model recursively calls itself over
  chunks, handling inputs 100× beyond its window) and [Glyph](../papers/context-compression/prompt-and-context-compression/2510.17800-glyph-scaling-context-windows-via-visual-text-compression.md) (render text as images: 3–4× token
  compression).

## What to implement first (runtime)

1. FlashInfer-style paged, block-sparse attention with GQA/MLA decode kernels and FP8 (then FP4) attention paths.
2. A **hybrid-layer executor**: recurrent-state caching for linear/SSM layers next to paged KV for full-attention
   layers, including prefix caching of recurrent states (chunk checkpoints).
3. Top-k block selection kernels (NSA/DSA/MSA-style) with cross-layer index reuse ([IndexCache](../papers/attention/sparse-attention/2603.12201-indexcache-accelerating-sparse-attention-via-cross-layer-index-reuse.md)).
4. Gate-after-SDPA and softmax variants as kernel epilogues.
