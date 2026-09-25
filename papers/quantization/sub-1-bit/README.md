# Sub-1-bit quantization (< 1 bit per weight)

Extreme compression below one bit per weight: binarization + structured sparsity, vector/codebook quantization with < 1 bpw, weight sharing.

**5 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Quantization](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/quantization.md](../../../overviews/quantization.md)

## 🏆 Best of the best (top 10)

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
