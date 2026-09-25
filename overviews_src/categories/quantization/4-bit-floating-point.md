**Verdict.** On Blackwell (B200/GB200/RTX 50xx) and MI355, FP4 is the fastest 4-bit path because the tensor cores
consume it natively.

* **NVFP4** uses 16-element blocks with an E4M3 scale plus a per-tensor FP32 scale. It is clearly more accurate than
  **MXFP4**, which uses 32-element blocks and a power-of-two E8M0 scale.
* Classic INT4 tricks do not transfer. Global rotations break NVFP4's block isolation and interact badly with MXFP4's
  power-of-two scales.
* What works:
  * **block-local** transforms aligned to the format group (MR-GPTQ, BRQ, DuQuant++, BATQuant);
  * **smarter scale selection** (Four-over-Six, OAS/MBS, RaZeR, IF4);
  * **distillation** (QAD) for post-trained models.
* **A hybrid 27B (Gated DeltaNet + attention) runs at full NVFP4 W4A4 within seed noise of BF16** ([[2609.04098]]), so
  recurrent layers do not need higher precision.

### Hand ranking

| # | Paper | Kind | Key idea | Headline | Cost |
| ---: | --- | --- | --- | --- | --- |
| 1 | [[2509.23202\|MR-GPTQ + QuTLASS]] (IST/Red Hat) | PTQ + kernels | Proves NVFP4's small group neutralizes outlier mitigation and that MXFP4's power-of-two scales hurt; **block-wise Hadamard + format-aware GPTQ**; QuTLASS kernels | 3.6× layer-wise, 2.2× end-to-end over FP16 on B200; MXFP4 close to NVFP4 | GPTQ-level |
| 2 | [[2510.25602\|INT vs FP]] | Study | QSNR theory for INT vs FP at each granularity | **MXINT8 > MXFP8** in accuracy and area; at 4 bits FP wins, but **NVINT4 + Hadamard beats NVFP4** | 200B-token training runs |
| 3 | [[2512.02010\|Four Over Six]] | Format tweak | Adaptively scale some NVFP4 blocks so their max maps to 4 instead of 6, evening out the non-uniform grid | Better NVFP4 PTQ **and pretraining**; cheap on hardware | 1T-token pretrain ablations |
| 4 | [[2601.20088\|QAD for NVFP4]] (NVIDIA) | Recipe | KL distillation from the BF16 teacher into the NVFP4 student | Robust for SFT + RL + merged models where QAT is unstable; tolerant to data quality | Short FT |
| 5 | [[2609.04098\|Minima: GDN survives NVFP4]] | Study | All 496 linear layers of a Qwen3.x-27B hybrid in NVFP4 W4A4, Gated DeltaNet included | Within seed noise of BF16 up to 64K RULER; 17.5 GiB; +14–19% prefill | — |
| 6 | [[2601.07475\|ARCQuant]] (ACL'26) | PTQ | **Augmented residual channels**: add quantized error channels to the K dimension; keeps a pure NVFP4 GEMM | W4A8-level accuracy with an NVFP4 kernel | — |
| 7 | [[2605.20315\|Mix-Quant]] | Serving | **NVFP4 prefill, BF16 decode** for agentic, input-heavy workloads | Up to 3× prefill with near-zero task loss | Training-free |
| 8 | [[2603.08713\|OAS + MBS]] | Software scaling | Overflow-aware scaling plus macro-block scaling for MXFP4 | MXFP4–NVFP4 gap goes from ~10% to <1% | — |
| 9 | [[2511.04214\|BRQ: Block Rotation]] | PTQ | Benchmarks INT4 methods on MXFP4 and explains why rotations fail; block rotation fixes it | GPTQ strong; block rotation needed | — |
| 10 | [[2601.19213\|M2XFP]] (ASPLOS'26) | HW format | Metadata-augmented MX: extra bits per block | −70.6% loss vs MXFP4, −37.3% vs NVFP4; 1.91× | — |
| 11 | [[2603.28765\|IF4 (Adaptive Block-Scaled Types)]] | Format | Per 16-block choice of FP4 or INT4, flagged by the unused sign bit of the E4M3 scale | Beats NVFP4 at equal storage | — |
| 12 | [[2501.04052\|RaZeR]] | Format | Remap NVFP4's redundant −0 and the scale sign bit to extra values | Free accuracy at the same footprint | — |
| 13 | [[2604.17789\|DuQuant++]] | PTQ | Outlier-aware rotation with block = 32 = MX group | Halves online cost vs DuQuant | — |
| 14 | [[2601.19026\|Is Finer Better?]] (ICLR'26) | Analysis | Smaller MX blocks can be **worse**, because scale quantization kills low-magnitude blocks | Don't blindly shrink blocks | — |
| 15 | [[2603.22370\|FAAR]] | PTQ + FT | Learnable rounding on the non-uniform NVFP4 grid, then 2-stage fine-tuning | State of the art on Llama-3 and Qwen3 | ~4 GPU-h for a 1B model |

**Runtime checklist (Blackwell / MI355).**
1. **Kernels.** Implement **NVFP4 W4A4** (block-16, E4M3 scale) GEMMs with fused activation quantization in the
   previous op's epilogue. Also support MXFP4 for OCP/AMD compatibility; MXFP8 or MXINT8 are fine for attention and KV.
2. **Transforms.** Use only **block-local** transforms (16- or 32-wide Hadamard) with FP4. A full-width Hadamard
   destroys NVFP4's benefit.
3. **Reserved scale bits.** Adopt the sign-bit-of-scale tricks (Four-over-Six, IF4, RaZeR) when your own kernels decode
   the format. Keep the plain NVFP4 path for vendor kernels.
4. **Mix-Quant scheduling.** For agentic traffic, run prefill in NVFP4 and decode in BF16 or FP8. It is a scheduling
   decision, and you need both weight copies or on-the-fly quantization.
