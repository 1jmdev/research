**Verdict.** KV-cache quantization is the cheapest long-context win.

* **KV8 (FP8/INT8) is free.**
* **KV4 is near-lossless** with the right transform:
  * per-channel K / per-token V (KIVI rule), or
  * a Hadamard or random rotation followed by per-token quantization.
* **KV2 now works** with attention-aware rotations and INT2 kernels (OSCAR, RotateKV, CommVQ, KVarN). Keep a small
  full-precision window of sink and recent tokens.

The most important 2026 insight: under long **autoregressive decoding** (reasoning models), errors *accumulate across
timesteps*. Methods calibrated in prefill-like settings under-report the damage. KVarN, PM-KVQ and MixKVQ fix this.
TurboQuant (Google) is the theory backbone for many 2026 codecs:
1. a random rotation, so coordinates follow a known Beta distribution;
2. Lloyd-Max scalar quantizers;
3. a 1-bit QJL residual for unbiased inner products.

It is quality-neutral at 3.5 bits per channel.

### Hand ranking

| # | Paper | Bits | Key idea | Why it matters for a runtime |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.19874\|TurboQuant]] (Google) | 2.5–4 | Data-oblivious online VQ: random rotation → per-coordinate optimal scalar quantizer + 1-bit QJL residual; within ~2.7× of the Shannon bound | **Calibration-free, online, provably near-optimal.** The default design for a KV codec; many 2026 papers ([[2606.21448\|Fast-TurboQuant]], [[2605.21226\|OCTOPUS]], [[2604.16957\|Open-TQ-Metal]], [[2606.24033\|Block-GTQ]]) extend it |
| 2 | [[2606.03458\|KVarN]] | 2–4 | Hadamard + **dual-axis variance normalization**; shows decode-time errors come from wrong *token scales* and accumulate | Best on **generative reasoning** (AIME/MATH/HumanEval); calibration-free |
| 3 | [[2605.17757\|OSCAR]] | **INT2** | Offline attention-aware covariance → fixed rotations + clip thresholds; **fused INT2 attention kernel, paged-KV compatible (SGLang)** | Deployable INT2 KV today |
| 4 | [[2506.18879\|CommVQ]] (ICML'25) | 1–2 | Additive VQ with a **RoPE-commutative codebook**, so decoding folds into attention | Near-lossless 2-bit; 1-bit usable; 128K context on one RTX 4090 for 8B |
| 5 | [[2502.04420\|KVTuner]] (ICML'25) | mixed | Offline layer-wise K/V precision pairs; **keys matter more than values** | Near-lossless at 3.25 bits (Llama-3.1-8B); no online overhead |
| 6 | [[2505.18610\|PM-KVQ]] (ICLR'26) | mixed | **Progressive** precision lowering as the cache fills plus long-context calibration | Built for long-CoT reasoners |
| 7 | [[2605.19660\|OScaR]] | 2 | Finds **token norm imbalance** as the bottleneck of per-channel K quantization; canalized rotation + omni-token scaling; CUDA kernels | A simpler alternative to TurboQuant pipelines |
| 8 | [[2501.16383\|RotateKV]] (IJCAI'25) | 2 | Outlier-aware per-head rotation (channel reordering) + pre-RoPE grouped-head rotation + attention-sink protection | Robust 2-bit baseline |
| 9 | [[2508.10395\|XQuant (rematerialization)]] | 2–3 | Cache the quantized **layer input X** (one tensor) and recompute K and V on the fly; trades compute for memory | 2× over KV quantization at equal bits; fits compute-rich GPUs |
| 10 | [[2502.10424\|QuantSpec]] (ICML'25) | 4 | Self-speculative decoding: the draft uses **4-bit hierarchical KV and 4-bit weights** of the same model | ~2.5× speedups with >90% acceptance for long context |
| 11 | [[2502.15075\|Quantize What Counts]] (ACL'25) | K4V2 | Theorem: key projections have larger norms, so **give keys more bits** | Simple allocation rule |
| 12 | [[2508.04257\|KVSink]] | — | Predicts sink tokens beyond position 0 so they stay full-precision | Plug-in for any KV quantizer |
| 13 | [[2502.00527\|PolarQuant (keys)]] (NeurIPS'25) | ~4 | RoPE pairs in polar form (radius + angle), with a lookup-table QK product | Decoding speedup plus compression |
| 14 | [[2501.19392\|AQUA-KV]] (ICML'25) | 2–3 | Predict K and V from other layers' K and V with small adapters; quantize only the residual | 70B calibrated in 4 h on one GPU |
| 15 | [[2503.16257\|VidKV]] | 1.x | Mixed 1/2-bit K with FFT, 1.58/2-bit V for **video LLMs** | Visual KV tolerates <2 bits |

Also useful:
* [[2503.18773|BitDecoding]]: tensor-core low-bit KV decode kernels.
* [[2604.19157|SAW-INT4]]: system-aware INT4 KV in real serving.
* [[2510.05373|KVLinC]]: Hadamard for V plus linear correction adapters for K.
* [[2505.10938|OTT]]: outlier-token tracing.
* [[2605.05699]]: INT4 KV is *faster* than FP16 on Apple Silicon.
* [[2605.20868]]: runtime-certified error bounds.

**Runtime recommendations.**
1. **Paged KV format.** Use a per-block scale plus FP8 by default. Offer INT4 (K per-channel pre-RoPE or rotated; V
   per-token) and INT2 (with a rotation) as opt-in. Always keep **sink tokens plus a recent window of ~64–128 tokens in
   BF16**.
2. **Rotations.** Use a Hadamard or random orthogonal matrix on K/V, fused into the QKV projection epilogue. The query
   gets the same rotation, so no inverse is needed at attention time.
3. **Kernels.** Dequantize inside the attention kernel: FlashInfer/FlashAttention-style tiles plus LUT or bit-unpack.
   BitDecoding and OSCAR show tensor-core-friendly layouts.
4. **Validation.** Test on *long generation* (AIME, LongGenBench), not prefill-only benchmarks such as LongBench.
