# Sub-1-bit quantization (< 1 bit per weight)

Extreme compression below one bit per weight: binarization + structured sparsity, vector/codebook quantization with < 1 bpw, weight sharing.

**5 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Quantization](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/quantization.md](../../../overviews/quantization.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Below 1 bit per weight there is no per-weight code left to store, so every working method in this set
*factorizes* the weight matrix: low-rank binary factors, shared binary codebooks, or sketches. The field is small
(5 papers) but moved from "QAT only" to "PTQ on one GPU" in 2026. Sub-1-bit is interesting for fitting 70B-class models
into 8 GB, but none of it is near-lossless yet. Treat it as a research tier, not a production format.

| # | Paper | Kind | What it does | Headline result | Cost (hand-checked) |
| ---: | --- | --- | --- | --- | --- |
| 1 | [NanoQuant](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md) (ICML'26) | PTQ | Low-rank **binary** factorization `W ≈ s₁ ⊙ (U_b V_bᵀ) ⊙ s₂`; ADMM initialization, then block and model reconstruction | First PTQ reaching both 1-bit and sub-1-bit. Llama-2-70B compressed 24× runs on an 8 GB GPU. Custom binary CUDA kernels | **~13 H100-h for 70B**, under 3 H100-h for 7B, on one H100 |
| 2 | [LittleBit](2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md) (NeurIPS'25) | QAT | Latent factorization, binarized factors, and row/column/latent compensation scales; Dual-SVID initialization | Viable down to **0.1 BPW** (Llama-2-13B under 0.9 GB); sub-0.5 BPW beats prior 1-bit methods | QAT (multi-GPU days) |
| 3 | [BTC-LLM](2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md) (ACL'26) | PTQ | Learnable transform (scaling, sign flip, orthogonal) plus a **binary codebook** of recurring sign vectors; no sparse masks | State of the art at 0.7–1.11 bits and runs on standard hardware | 1×H800 |
| 4 | [UltraSketchLLM](2506.17255-ultrasketchllm-sub-1-bit-llm-compression-via-sketch-and-hardware-frien.md) (DAC'26) | PTQ + HW | Data-sketch compression of weights with hardware-friendly decode operators | 0.5 bpw, 14.9× faster than naive sketch decode | — |
| 5 | [Sign Lock-In](2602.17063-sign-lock-in-randomly-initialized-weight-signs-persist-and-bottleneck.md) (ICML'26) | Theory | Learned sign matrices are indistinguishable from random ±1, and most signs never change from initialization | Explains the **"one-bit wall"**. Per-weight sign codes cannot be compressed after the fact | — |

**For a runtime.**
* All three practical methods reduce to **binary GEMMs on factors** (`x → xV_b → ·s → U_b`). One kernel that does
  "packed ±1 matmul with per-row/col scales" (XNOR/popcount on CPU, bit-unpack to FP16 MMA on GPU) covers NanoQuant,
  LittleBit and most of the 1-bit papers in the neighbouring category.
* The rank sets the bit-rate, so a single checkpoint format with a variable latent rank supports any BPW between 0.1 and
  1.0.

**For model building.** Sign Lock-In implies that if you want sub-bit-compressible models, you have to control sign
structure **at initialization or during pretraining**. Nothing post-hoc will do it. This is an open research direction.

## 🏆 Best of the best by impact score (top 10)

1. **[LittleBit: Ultra Low-Bit Quantization via Latent Factorization](2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md)** (2026-02) — This paper introduces LittleBit, a novel framework for extreme LLM compression that targets quantization rates as low as $0.1$ bits per weight (BPW), achieving a memory reduction of approximately $31\times, which …  
   _score 5.38 · Accepted to NeurIPS 2025 · 14 cites · [code](https://github.com/SamsungLabs/LittleBit)_
2. **[NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md)** (2026-06) — NanoQuant is proposed, the first post-training quantization (PTQ) method to compress LLMs to both binary and sub-1-bit levels, and establishes a new Pareto frontier in low-memory post-training quantization, and enables …  
   _score 4.65 · Accepted to ICML 2026 · 0 cites · 22▲ HF · [code](https://github.com/SamsungLabs/NanoQuant) · ~13–768 H100-h_
3. **[BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook](2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md)** (2026-04) — Binary quantization represents the most extreme form of compression, reducing weights to +/-1 for maximal memory and computational efficiency, and a novel sub-1-bit LLM quantization framework that leverages binary …  
   _score 2.85 · Annual Meeting of the Association for Computational Linguist · 6 cites_
4. **[UltraSketchLLM: Sub-1-Bit LLM Compression via Sketch and Hardware-Friendly Operators](2506.17255-ultrasketchllm-sub-1-bit-llm-compression-via-sketch-and-hardware-frien.md)** (2026-06) — UltraSketchLLM is introduced, compressing LLMs with data sketch with a high compression rate down to 0.5 bit per weight, which reduces peak GPU memory footprint with a high compression rate down to 0.5 bit per weight.  
   _score 0.7 · Accepted by the 63rd ACM/IEEE The Chips to Systems Conferenc · 0 cites_
5. **[Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression](2602.17063-sign-lock-in-randomly-initialized-weight-signs-persist-and-bottleneck.md)** (2026-06) — This work introduces a from-scratch low-rank sign-template training method that prevents the emergence of the one-bit wall, and formalizes this behavior with sign lock-in theory, a stopping-time analysis of sign flips …  
   _score 0.7 · ICML 2026 · 0 cites_

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md) | 13 | 768 | H100 | For example, it compresses Llama-2-70B by 24\times in just 13 hours on a single H100, enabling a 70B model to operate on a consumer 8 GB GPU.… |

## Bit-width map

| Bits | # papers | Top papers |
| --- | ---: | --- |
| <1 | 5 | [LittleBit: Ultra Low-Bit Quantization via Latent F](2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md); [NanoQuant: Efficient Sub-1-Bit Quantization of Lar](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md); [BTC-LLM: Efficient Sub-1-Bit LLM Quantization via ](2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md); [UltraSketchLLM: Sub-1-Bit LLM Compression via Sket](2506.17255-ultrasketchllm-sub-1-bit-llm-compression-via-sketch-and-hardware-frien.md) |
| 1 | 5 | [LittleBit: Ultra Low-Bit Quantization via Latent F](2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md); [NanoQuant: Efficient Sub-1-Bit Quantization of Lar](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md); [BTC-LLM: Efficient Sub-1-Bit LLM Quantization via ](2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md); [UltraSketchLLM: Sub-1-Bit LLM Compression via Sket](2506.17255-ultrasketchllm-sub-1-bit-llm-compression-via-sketch-and-hardware-frien.md) |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [LittleBit: Ultra Low-Bit Quantization via Latent Factorization](2506.13771-littlebit-ultra-low-bit-quantization-via-latent-factorization.md) | 2026-02-05 | 5.38 | 14 | 0 | Accepted to NeurIPS 2025 | [✓](https://github.com/SamsungLabs/LittleBit) | This paper introduces LittleBit, a novel framework for extreme LLM compression that targets quantization rates as low as $0.1$ bits per weight (BPW), achieving … |
| 2 | [NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](2602.06694-nanoquant-efficient-sub-1-bit-quantization-of-large-language-models.md) | 2026-06-14 | 4.65 | 0 | 22 | Accepted to ICML 2026 | [✓](https://github.com/SamsungLabs/NanoQuant) | NanoQuant is proposed, the first post-training quantization (PTQ) method to compress LLMs to both binary and sub-1-bit levels, and establishes a new Pareto … |
| 3 | [BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook](2506.12040-btc-llm-efficient-sub-1-bit-llm-quantization-via-learnable-transformat.md) | 2026-04-09 | 2.85 | 6 | 0 | Annual Meeting of the Association for Co |  | Binary quantization represents the most extreme form of compression, reducing weights to +/-1 for maximal memory and computational efficiency, and a novel … |
| 4 | [UltraSketchLLM: Sub-1-Bit LLM Compression via Sketch and Hardware-Friendly Operators](2506.17255-ultrasketchllm-sub-1-bit-llm-compression-via-sketch-and-hardware-frien.md) | 2026-06-12 | 0.7 | 0 | 0 | Accepted by the 63rd ACM/IEEE The Chips  |  | UltraSketchLLM is introduced, compressing LLMs with data sketch with a high compression rate down to 0.5 bit per weight, which reduces peak GPU memory … |
| 5 | [Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression](2602.17063-sign-lock-in-randomly-initialized-weight-signs-persist-and-bottleneck.md) | 2026-06-02 | 0.7 | 0 | 0 | ICML 2026 |  | This work introduces a from-scratch low-rank sign-template training method that prevents the emergence of the one-bit wall, and formalizes this behavior with … |
