**Verdict.** Never pretrain a diffusion LM from scratch if you have a good AR checkpoint. AR→dLLM conversion reaches or
beats from-scratch dLLMs at **1/10 to 1/1000 of the tokens**. The field converged on a clear recipe:

1. **Target block diffusion, not full-sequence diffusion.** Keep attention causal *across* blocks and bidirectional
   *inside* the block. Exact KV cache, arbitrary length, and the AR weight distribution survives. Efficient-DLM's main
   finding is that preserving pretrained weight distributions is what matters.
2. **Grow the block size gradually from 1** (AR = block diffusion with block size 1): NBDiff, LLaDA 2.0's WSD
   (warm-up → stable → decay of block size), T\*, BARD's progressive block merging.
3. **Keep an AR loss alongside** (SDAR, NBDiff, Nemotron-Labs-Diffusion, FLARE's clean/noisy two-stream objective). The
   model then serves AR, diffusion *and* self-speculative modes from one checkpoint.
4. **Close the train/inference gap.** Position-dependent masking (Efficient-DLM), complementary masks (Fast-dLLM v2),
   **on-policy distillation from the AR teacher** (OPDLM: ~70M tokens), and representation alignment to the frozen AR
   model (Repr-Align).
5. **Data quality beats token count** (FLARE). A post-trained source + mismatched SFT data drifts.

**Cost tiers:**

| Target | Budget |
| --- | --- |
| Usable block-dLLM with AR-level quality | **~1–10B tokens (≈50–500 H100-h at 7–8B)** |
| Strong speed/quality frontier | **25–100B tokens (≈1–4K H100-h)** |
| A best-in-class family | 300–700B tokens (≈10–25K H100-h) |

Hybrid (linear-attention) AR sources convert *faster* than full-attention ones: dQwen3.5 reaches equal loss in about half
the tokens. FLARE shows how to handle recurrent state within blocks.

### Conversion cost table (hand-checked)

H100-hours are either **reported** (GPU count × wall-clock; A100 = 0.32 H100, H800 ≈ H100) or **estimated** as
6·N·D FLOPs at 40% MFU of H100 BF16 dense (≈1.42·10¹⁸ FLOP per H100-hour). N is the active parameter count and D the
training tokens. Estimates are ±2×.

| Method | Source → target | Tokens | H100-h | Basis |
| --- | --- | ---: | ---: | --- |
| [[2508.09192\|D2F]] | LLaDA/Dream → block-AR dLLM (distill) | Bespoke-17k | **~31** | reported: 8×A100-40G × 12 h |
| [[2606.06712\|OPDLM]] | Qwen3 4B/8B → DLM (on-policy distill) | 0.066–0.076B | **~10–50** | estimate; sub-4B fits on 2×A6000 |
| [[2509.26328\|Fast-dLLM v2]] | Qwen2.5-7B → block dLLM | ~1B | **~250** (1.5B: ~165) | reported: 64×A100 × 12 h (1.5B: × 8 h) |
| [[2606.01774\|FLARE]] | Qwen3.5 hybrid 2–9B → dLLM | ~10B | **~380** (9B) | estimate |
| [[2512.14067\|Efficient-DLM]] (recovery point) | Qwen3 8B → block dLLM | ~10B | **~340** | estimate |
| [[2510.06303\|SDAR]] | Qwen3 1.7B–30B-A3B → block dLLM | 50B | **~360** (1.7B), **~1.7K** (8B), **~0.6–1K** (30B-A3B) | estimate |
| [[2509.04185\|Set Block Decoding]] | Llama-3.1-8B / Qwen3-8B → NTP+MATP | 70B | **~2.4K** | estimate |
| [[2512.22737\|WeDLM]] | Qwen2.5-7B / Qwen3-8B → causal-attention DLM | 100B + 10B SFT | **~3.7K** | estimate |
| [[2510.03270\|CoDA]] | Qwen3-1.7B → code DLM | ~200B | **~1.4K** equivalent (TPU) | estimate |
| [[2512.14067\|Efficient-DLM]] (full) | Qwen2.5-1.5B / Qwen3-4B / 8B | 300B / 300B / 500B | **~1.9K / ~5K / ~17K** | estimate (128×H100) |
| [[2607.05722\|Nemotron-Labs-Diffusion]] 8B | AR (1T AR stage) → joint AR+diffusion | 300B + 45B SFT | **~12K** (excl. AR stage) | estimate (256×H100) |
| [[2508.15487\|Dream 7B]] | Qwen2.5-7B → full-sequence MDM | 580B | **~19–25K** | estimate; blog: 96×H800 × 256 h |
| [[2512.06776\|NBDiff]]-7B | AR → block diffusion (block-size growth) | 700B + 100B long-ctx | **~24K** | estimate |
| [[2606.26493\|Nemotron-Labs-TwoTower]] | Nemotron-3-Nano-30B-A3B + new denoiser tower | ~2.1T | **≳27K** | estimate (3B active) |
| [[2512.15745\|LLaDA 2.0]] | Ling 16B/100B MoE → block dLLM | undisclosed | n/a | 3-phase WSD block schedule |

For scale: LLaDA 8B from scratch used 2.3T tokens (~0.13M H800-h). Conversion is **5–500× cheaper**.

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [[2510.06303\|SDAR]] (ACL) | Brief block-diffusion CPT of an AR model; AR training and within-block parallel diffusion inference, decoupled | AR-level quality at **50B tokens** (vs 580B for Dream); dense and MoE up to 30B; SDAR-8B/30B released. Most-cited conversion baseline |
| 2 | [[2512.14067\|Efficient-DLM]] (NVIDIA) | Block-wise causal-across-blocks attention that *preserves AR weight distributions* + position-dependent masking | Recovers accuracy at ~10B tokens; Efficient-DLM 8B: +5.4% accuracy at 4.5× throughput vs Dream 7B, +2.7% at 2.7× vs Qwen3 4B |
| 3 | [[2512.06776\|NBDiff: Next-Token → Next-Block]] | Adaptation as **block-size growth from 1**; context-causal attention + parallel training with auxiliary AR loss | Principled path; NBDiff-7B is state of the art among 7B dLLMs |
| 4 | [[2607.05722\|Nemotron-Labs-Diffusion]] (NVIDIA) | Joint AR + diffusion objective → **tri-mode** model (AR / diffusion / self-speculation) | 8B: 6× tokens per forward vs Qwen3-8B, **4× throughput** on GB200 (SGLang). Diffusion drafting beats MTP heads |
| 5 | [[2606.06712\|OPDLM]] | **On-policy distillation** from the AR teacher on the student's own confidence-decoding trajectories | Competitive with SDAR at **15–7,000× fewer tokens** |
| 6 | [[2509.26328\|Fast-dLLM v2]] | Block diffusion + complementary masks; hierarchical caches | Lossless adaptation with ~1B tokens; 2.5× over AR decoding |
| 7 | [[2606.01774\|FLARE]] | Conversion for **hybrid (GDN) backbones**: clean/noisy two-stream objective, packed masks, recurrent-state scheduling, unified serving | ~10B tokens keeps source capability; handles CUDA graphs for mixed modes |
| 8 | [[2512.22737\|WeDLM]] | Diffusion decoding under **standard causal attention** (topological reordering), so the stock prefix KV cache and engines work | ~3× over **vLLM-served AR** on reasoning, up to 10× in low-entropy regimes; streaming commit avoids block stop-and-wait |
| 9 | [[2605.06885\|Repr-Align]] | Align student hidden states layer-wise to the frozen AR model during denoising training | Works even on a 0.8B-token subset. "dLLM training = relearning the decoding path, not the language" |
| 10 | [[2609.20751\|dQwen3.5]] | Adapting **hybrid attention+RNN** Qwen3.5 (0.8–9B) to DLMs | Reaches equal loss in ~½ the tokens of a full-attention control |
| 11 | [[2606.26493\|Nemotron-Labs-TwoTower]] | **Frozen AR context tower** + trainable bidirectional denoiser tower via cross-attention | 98.7% of AR quality at 2.42× wall-clock throughput |
| 12 | [[2512.15745\|LLaDA 2.0]] | 3-phase block-level WSD conversion of a 100B MoE; confidence loss; embedding noise at start | The only 100B-class converted dLLM |

**Multimodal.** [[2512.15713|DiffusionVL]], [[2604.06832|Fast-dVLM]] (direct conversion beats two-stage; 64×H100) and
[[2604.16514|BARD]] (progressive block merging + stage-wise distillation) apply the same recipe to VLMs.

**Other routes.** [[2607.24507|UNIFUSION]] targets *uniform-noise* diffusion, where every token stays editable.
[[2605.27387|FLUID]] uses strictly causal alignment with elastic horizons. [[2601.11214|T\*]] scales block size with
TraceRL. [[2601.14758]] is a mechanistic study of what changes inside the network.

**Recommendation for model builders.**
1. Start from the best AR checkpoint (hybrid is fine) and do block-diffusion CPT with an AR auxiliary loss. Grow the
   block size 1 → 4 → 16/32 over **10–50B tokens**.
2. Then run on-policy distillation from the AR teacher and a Fast-dLLM/D2F-style fine-tune for parallel decoding.
3. Validate on ParallelBench.
4. Budget **~0.5–2K H100-h for an 8B model**. Keep the AR head/mode so the same weights can self-speculate.
