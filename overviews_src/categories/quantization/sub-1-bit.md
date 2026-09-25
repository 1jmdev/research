**Verdict.** Below 1 bit per weight there is no per-weight code left to store, so every working method in this set
*factorizes* the weight matrix: low-rank binary factors, shared binary codebooks, or sketches. The field is small
(5 papers) but moved from "QAT only" to "PTQ on one GPU" in 2026. Sub-1-bit is interesting for fitting 70B-class models
into 8 GB, but none of it is near-lossless yet. Treat it as a research tier, not a production format.

| # | Paper | Kind | What it does | Headline result | Cost (hand-checked) |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2602.06694\|NanoQuant]] (ICML'26) | PTQ | Low-rank **binary** factorization `W ≈ s₁ ⊙ (U_b V_bᵀ) ⊙ s₂`; ADMM initialization, then block and model reconstruction | First PTQ reaching both 1-bit and sub-1-bit. Llama-2-70B compressed 24× runs on an 8 GB GPU. Custom binary CUDA kernels | **~13 H100-h for 70B**, under 3 H100-h for 7B, on one H100 |
| 2 | [[2506.13771\|LittleBit]] (NeurIPS'25) | QAT | Latent factorization, binarized factors, and row/column/latent compensation scales; Dual-SVID initialization | Viable down to **0.1 BPW** (Llama-2-13B under 0.9 GB); sub-0.5 BPW beats prior 1-bit methods | QAT (multi-GPU days) |
| 3 | [[2506.12040\|BTC-LLM]] (ACL'26) | PTQ | Learnable transform (scaling, sign flip, orthogonal) plus a **binary codebook** of recurring sign vectors; no sparse masks | State of the art at 0.7–1.11 bits and runs on standard hardware | 1×H800 |
| 4 | [[2506.17255\|UltraSketchLLM]] (DAC'26) | PTQ + HW | Data-sketch compression of weights with hardware-friendly decode operators | 0.5 bpw, 14.9× faster than naive sketch decode | — |
| 5 | [[2602.17063\|Sign Lock-In]] (ICML'26) | Theory | Learned sign matrices are indistinguishable from random ±1, and most signs never change from initialization | Explains the **"one-bit wall"**. Per-weight sign codes cannot be compressed after the fact | — |

**For a runtime.**
* All three practical methods reduce to **binary GEMMs on factors** (`x → xV_b → ·s → U_b`). One kernel that does
  "packed ±1 matmul with per-row/col scales" (XNOR/popcount on CPU, bit-unpack to FP16 MMA on GPU) covers NanoQuant,
  LittleBit and most of the 1-bit papers in the neighbouring category.
* The rank sets the bit-rate, so a single checkpoint format with a variable latent rank supports any BPW between 0.1 and
  1.0.

**For model building.** Sign Lock-In implies that if you want sub-bit-compressible models, you have to control sign
structure **at initialization or during pretraining**. Nothing post-hoc will do it. This is an open research direction.
