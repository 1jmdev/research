**Verdict.** This is attention-to-attention retrofitting: the model stays a Transformer, but its KV cache and attention cost
shrink. There are two families.

1. **MHA/GQA → MLA** (DeepSeek-style latent KV). [[2502.07864|TransMLA]] proves MLA is strictly more expressive than GQA
   at equal KV size. It converts via **RoRoPE** (concentrate the RoPE signal into one head) + **FreqFold** + joint KV
   low-rank factorization. MHA2MLA does partial-RoPE removal + SVD. The payoff is **reusing DeepSeek's MLA kernels**
   (FlashMLA, absorbed MQA decode in vLLM/SGLang) and ~93–97% KV reduction.

   The cheap part is initialization: [[2603.17946|CARE]] (ICLR'26) makes it **activation-aware** (covariance-weighted
   SVD, per-layer rank allocation), which cuts the healing needed afterwards. Healing takes **1–6B tokens** at 7–8B.
2. **Full → sparse / sliding-window attention.** Full-attention models are intrinsically sparse. Only some heads are
   "retrieval heads"; the rest can become streaming/SWA heads.
   * [[2605.16928|RTPurbo]] converts in **a few hundred training steps**: head specialization + low-dimensional retrieval
     index + dynamic top-p.
   * [[2512.10411|SWAA]] shows naive SWA collapses. The fix is a recipe: keep some full-attention layers, sinks,
     interleave, and fine-tune briefly.
   * LongCat's [[2512.23966|LoZA]] and DeepSeek-V3.2's DSA show the same thing at frontier scale, applied during
     mid-training.

A new consideration is **hardware fit.** MLA's absorbed MQA path suits H100 compute/bandwidth ratios but loses head-axis
tensor parallelism and gains nothing from MTP on H20-class GPUs. [[2605.15250|GQLA]] fixes this with group-indexed
up-projections that allow two equivalent decode paths.

### Conversion cost table (hand-checked)

| Method | Conversion | Tokens / steps | H100-h | Basis |
| --- | --- | ---: | ---: | --- |
| [[2605.16928\|RTPurbo]] | Full → head-wise sparse (retrieval/streaming heads) | ~1M-token alignment + ~600 steps at 48K ctx | **~10–50** | estimate from steps (H20) |
| [[2512.10411\|SWAA]] | Full → SWA hybrid | short fine-tune | **~15** (Qwen3-4B), **~36** (30B-A3B) | reported: 8×H20 × 12 h / 30 h |
| [[2606.27791\|NLL-guided layer selection]] | Full → SWA hybrid, training-free | calibration only | **<1** | training-free |
| [[2603.17946\|CARE]] (ICLR'26) | GQA → MLA (activation-aware init) | 0B one-shot; 1–3B healing | **~0 / ~35–100** | estimate (8B) |
| [[2502.07864\|TransMLA]] | GQA → MLA (Llama-2-7B, Qwen) | 6B | **~180** | estimate |
| [[2502.14837\|MHA2MLA]] (ACL'25) | MHA/GQA → MLA | 0.6–1% of pretraining tokens (12K steps) | **~100–350** (7B) | estimate |
| [[2503.11132\|X-EcoMLA]] | Attention → MLA via KD from a larger teacher | 3.6–7B | **~70–140 MI300-h** | reported |
| [[2512.23966\|LoZA]] (LongCat-Flash) | Full → ZigZag sparse during mid-training | 500B + 40B long-context | part of the mid-training budget | frontier scale |

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [[2502.07864\|TransMLA]] | Any GQA → MLA; RoRoPE + FreqFold decouple RoPE; balanced KV factorization; direct compatibility with DeepSeek code | 93% KV compressed on Llama-2-7B, **10.6× speedup at 8K**; 6B tokens to restore quality |
| 2 | [[2605.16928\|RTPurbo: Full Attention Strikes Back]] | Retrieval vs streaming heads + low-dim retrieval index + dynamic top-p; a few hundred steps | Near-lossless long-context and reasoning at high sparsity |
| 3 | [[2502.14837\|MHA2MLA]] (ACL'25) | Contribution-aware partial-RoPE removal + SVD joint KV projection | Up to **96.87% KV reduction** (Llama-2-7B) with 0.6–1% of the data; composes with KV quantization |
| 4 | [[2603.17946\|CARE]] (ICLR'26) | Covariance-aware, rank-enhanced factorization + per-layer rank allocation | Much better one-shot MLA init; fewer healing tokens |
| 5 | [[2512.10411\|SWAA]] | Diagnoses SWA collapse (train/inference mismatch + no distant access); a recipe of FA layers + sinks + interleaving + light fine-tuning | Recovers long-context quality with SWA efficiency; ~12 h on 8×H20 for 4B |
| 6 | [[2605.15250\|GQLA]] | MLA variant whose weights admit **both** absorbed-MQA and GQA-style decode paths | Hardware-adaptive MLA: tensor parallelism + MTP gains on H20 |
| 7 | [[2512.23966\|LoZA]] (Meituan) | Convert full attention to ZigZag sparse during mid-training | 1M-token context; >50% prefill speedup and >30% decode savings at 256K |
| 8 | [[2503.11132\|X-EcoMLA]] | SVD init + distillation from a larger teacher ("dark knowledge") | 6.4× KV compression on Llama-3.2-1B, no loss, 70 MI300 GPU-h |
| 9 | [[2604.05688\|Attention Editing]] | General framework: any trained attention → MLA or hybrid SWA without strict structural requirements | Practical cross-architecture conversion |
| 10 | [[2505.17272\|Zebra-Llama]] (NeurIPS'25) | MLA + Mamba2 hybrid composed from a pretrained Transformer | Extreme KV reduction (see [transformer-to-linear-or-hybrid](../transformer-to-linear-or-hybrid/README.md)) |

**Also useful.**
* Other modalities: [[2601.11464|MHA2MLA-VLM]] (VLMs), [[2603.00563|Whisper-MLA]] (ASR).
* Domain case study: [[2606.05868|YouZhi]] (layer-adaptive FreqFold on Ascend).
* Small-model study: [[2506.09342]] (MLA + RoPE at half rank is a Pareto improvement at 30M).

**Recommendation.**
* For serving an existing GQA model at long context, **GQA → MLA (TransMLA + CARE init, 1–6B tokens, ≈50–200 H100-h)**
  pays for itself immediately. It gets 4–10× smaller KV plus DeepSeek's MLA kernels.
* For a lower-effort win, do head-level full → sparse/SWA conversion (RTPurbo/SWAA, ≈10–40 H100-h). This keeps GQA
  kernels.
* For a runtime, MLA support needs both the **absorbed decode path** (latent KV, MQA-style) and the **expanded prefill
  path**. Watch GQLA if you target non-H100 hardware.
