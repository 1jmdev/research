# AR → diffusion LM conversion

Adapting pretrained autoregressive LMs into diffusion LMs (DiffuLLaMA, Dream-from-Qwen, block-diffusion adaptation). Compute cost is the key metric.

**18 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Model conversion (AR→diffusion, linearization, upcycling, …)](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/model-conversion.md](../../../overviews/model-conversion.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

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
| [D2F](../../decoding/diffusion-llm-inference/2508.09192-diffusion-llms-can-do-faster-than-ar-inference-via-discrete-diffusion.md) | LLaDA/Dream → block-AR dLLM (distill) | Bespoke-17k | **~31** | reported: 8×A100-40G × 12 h |
| [OPDLM](2606.06712-data-efficient-autoregressive-to-diffusion-language-models-via-on-poli.md) | Qwen3 4B/8B → DLM (on-policy distill) | 0.066–0.076B | **~10–50** | estimate; sub-4B fits on 2×A6000 |
| [Fast-dLLM v2](../../decoding/diffusion-llm-inference/2509.26328-fast-dllm-v2-efficient-block-diffusion-llm.md) | Qwen2.5-7B → block dLLM | ~1B | **~250** (1.5B: ~165) | reported: 64×A100 × 12 h (1.5B: × 8 h) |
| [FLARE](2606.01774-flare-diffusion-for-hybrid-language-model.md) | Qwen3.5 hybrid 2–9B → dLLM | ~10B | **~380** (9B) | estimate |
| [Efficient-DLM](2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md) (recovery point) | Qwen3 8B → block dLLM | ~10B | **~340** | estimate |
| [SDAR](2510.06303-sdar-a-synergistic-diffusion-autoregression-paradigm-for-scalable-sequ.md) | Qwen3 1.7B–30B-A3B → block dLLM | 50B | **~360** (1.7B), **~1.7K** (8B), **~0.6–1K** (30B-A3B) | estimate |
| [Set Block Decoding](../../decoding/jacobi-and-parallel-decoding/2509.04185-set-block-decoding-is-a-language-model-inference-accelerator.md) | Llama-3.1-8B / Qwen3-8B → NTP+MATP | 70B | **~2.4K** | estimate |
| [WeDLM](../../decoding/diffusion-llm-inference/2512.22737-wedlm-reconciling-diffusion-language-models-with-standard-causal-atten.md) | Qwen2.5-7B / Qwen3-8B → causal-attention DLM | 100B + 10B SFT | **~3.7K** | estimate |
| [CoDA](2510.03270-coda-coding-lm-via-diffusion-adaptation.md) | Qwen3-1.7B → code DLM | ~200B | **~1.4K** equivalent (TPU) | estimate |
| [Efficient-DLM](2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md) (full) | Qwen2.5-1.5B / Qwen3-4B / 8B | 300B / 300B / 500B | **~1.9K / ~5K / ~17K** | estimate (128×H100) |
| [Nemotron-Labs-Diffusion](2607.05722-nemotron-labs-diffusion-a-tri-mode-language-model-unifying-autoregress.md) 8B | AR (1T AR stage) → joint AR+diffusion | 300B + 45B SFT | **~12K** (excl. AR stage) | estimate (256×H100) |
| [Dream 7B](../../decoding/diffusion-language-models/2508.15487-dream-7b-diffusion-large-language-models.md) | Qwen2.5-7B → full-sequence MDM | 580B | **~19–25K** | estimate; blog: 96×H800 × 256 h |
| [NBDiff](2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md)-7B | AR → block diffusion (block-size growth) | 700B + 100B long-ctx | **~24K** | estimate |
| [Nemotron-Labs-TwoTower](2606.26493-nemotron-labs-twotower-diffusion-language-modeling-with-pretrained-aut.md) | Nemotron-3-Nano-30B-A3B + new denoiser tower | ~2.1T | **≳27K** | estimate (3B active) |
| [LLaDA 2.0](../../decoding/diffusion-language-models/2512.15745-llada2-0-scaling-up-diffusion-language-models-to-100b.md) | Ling 16B/100B MoE → block dLLM | undisclosed | n/a | 3-phase WSD block schedule |

For scale: LLaDA 8B from scratch used 2.3T tokens (~0.13M H800-h). Conversion is **5–500× cheaper**.

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [SDAR](2510.06303-sdar-a-synergistic-diffusion-autoregression-paradigm-for-scalable-sequ.md) (ACL) | Brief block-diffusion CPT of an AR model; AR training and within-block parallel diffusion inference, decoupled | AR-level quality at **50B tokens** (vs 580B for Dream); dense and MoE up to 30B; SDAR-8B/30B released. Most-cited conversion baseline |
| 2 | [Efficient-DLM](2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md) (NVIDIA) | Block-wise causal-across-blocks attention that *preserves AR weight distributions* + position-dependent masking | Recovers accuracy at ~10B tokens; Efficient-DLM 8B: +5.4% accuracy at 4.5× throughput vs Dream 7B, +2.7% at 2.7× vs Qwen3 4B |
| 3 | [NBDiff: Next-Token → Next-Block](2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md) | Adaptation as **block-size growth from 1**; context-causal attention + parallel training with auxiliary AR loss | Principled path; NBDiff-7B is state of the art among 7B dLLMs |
| 4 | [Nemotron-Labs-Diffusion](2607.05722-nemotron-labs-diffusion-a-tri-mode-language-model-unifying-autoregress.md) (NVIDIA) | Joint AR + diffusion objective → **tri-mode** model (AR / diffusion / self-speculation) | 8B: 6× tokens per forward vs Qwen3-8B, **4× throughput** on GB200 (SGLang). Diffusion drafting beats MTP heads |
| 5 | [OPDLM](2606.06712-data-efficient-autoregressive-to-diffusion-language-models-via-on-poli.md) | **On-policy distillation** from the AR teacher on the student's own confidence-decoding trajectories | Competitive with SDAR at **15–7,000× fewer tokens** |
| 6 | [Fast-dLLM v2](../../decoding/diffusion-llm-inference/2509.26328-fast-dllm-v2-efficient-block-diffusion-llm.md) | Block diffusion + complementary masks; hierarchical caches | Lossless adaptation with ~1B tokens; 2.5× over AR decoding |
| 7 | [FLARE](2606.01774-flare-diffusion-for-hybrid-language-model.md) | Conversion for **hybrid (GDN) backbones**: clean/noisy two-stream objective, packed masks, recurrent-state scheduling, unified serving | ~10B tokens keeps source capability; handles CUDA graphs for mixed modes |
| 8 | [WeDLM](../../decoding/diffusion-llm-inference/2512.22737-wedlm-reconciling-diffusion-language-models-with-standard-causal-atten.md) | Diffusion decoding under **standard causal attention** (topological reordering), so the stock prefix KV cache and engines work | ~3× over **vLLM-served AR** on reasoning, up to 10× in low-entropy regimes; streaming commit avoids block stop-and-wait |
| 9 | [Repr-Align](2605.06885-don-t-retrain-align-adapting-autoregressive-lms-to-diffusion-lms-via-r.md) | Align student hidden states layer-wise to the frozen AR model during denoising training | Works even on a 0.8B-token subset. "dLLM training = relearning the decoding path, not the language" |
| 10 | [dQwen3.5](2609.20751-dqwen3-5-hybrid-attention-diffusion-language-models.md) | Adapting **hybrid attention+RNN** Qwen3.5 (0.8–9B) to DLMs | Reaches equal loss in ~½ the tokens of a full-attention control |
| 11 | [Nemotron-Labs-TwoTower](2606.26493-nemotron-labs-twotower-diffusion-language-modeling-with-pretrained-aut.md) | **Frozen AR context tower** + trainable bidirectional denoiser tower via cross-attention | 98.7% of AR quality at 2.42× wall-clock throughput |
| 12 | [LLaDA 2.0](../../decoding/diffusion-language-models/2512.15745-llada2-0-scaling-up-diffusion-language-models-to-100b.md) | 3-phase block-level WSD conversion of a 100B MoE; confidence loss; embedding noise at start | The only 100B-class converted dLLM |

**Multimodal.** [DiffusionVL](2512.15713-diffusionvl-translating-any-autoregressive-models-into-diffusion-visio.md), [Fast-dVLM](2604.06832-fast-dvlm-efficient-block-diffusion-vlm-via-direct-conversion-from-aut.md) (direct conversion beats two-stage; 64×H100) and
[BARD](2604.16514-bard-bridging-autoregressive-and-diffusion-vision-language-models-via.md) (progressive block merging + stage-wise distillation) apply the same recipe to VLMs.

**Other routes.** [UNIFUSION](2607.24507-unifusion-adapting-autoregressive-language-models-into-discrete-diffus.md) targets *uniform-noise* diffusion, where every token stays editable.
[FLUID](2605.27387-from-ar-to-diffusion-efficiently-adapting-large-language-models-with-s.md) uses strictly causal alignment with elastic horizons. [T\*](2601.11214-t-star-progressive-block-scaling-for-masked-diffusion-language-models.md) scales block size with
TraceRL. [Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models](2601.14758-mechanism-shift-during-post-training-from-autoregressive-to-masked-dif.md) is a mechanistic study of what changes inside the network.

**Recommendation for model builders.**
1. Start from the best AR checkpoint (hybrid is fine) and do block-diffusion CPT with an AR auxiliary loss. Grow the
   block size 1 → 4 → 16/32 over **10–50B tokens**.
2. Then run on-policy distillation from the AR teacher and a Fast-dLLM/D2F-style fine-tune for parallel decoding.
3. Validate on ParallelBench.
4. Budget **~0.5–2K H100-h for an 8B model**. Keep the AR head/mode so the same weights can self-speculate.

## 🏆 Best of the best by impact score (top 10)

1. **[SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation](2510.06303-sdar-a-synergistic-diffusion-autoregression-paradigm-for-scalable-sequ.md)** (2025-10) — The SDAR model surpasses its AR counterpart on challenging scientific reasoning benchmarks such as GPQA and ChemBench, and gains further improvements under test-time scaling methods like majority voting and pass@k, …  
   _score 11.46 · Annual Meeting of the Association for Computational Linguist · 114 cites · [code](https://github.com/InternLM/lmdeploy)_
2. **[Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md)** (2026-04) — A continuous pretraining scheme with a block-wise attention pattern, which remains causal across blocks while enabling bidirectional modeling within each block is introduced, which can better preserve pretrained AR …  
   _score 9.5 · 24 cites · 18▲ HF_
3. **[From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs](2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md)** (2026-01) — This work reframe the whole AR-to-DLM adaptation under the Block-Diffusion paradigm, transitioning from block size 1 to the final Block-Diffusion state, and proposes NBDiff-7B that could inherit the long-context …  
   _score 9.05 · 22 cites · 26▲ HF · [code](https://github.com/YuchuanTian/NBDiff)_
4. **[Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](2607.05722-nemotron-labs-diffusion-a-tri-mode-language-model-unifying-autoregress.md)** (2026-07) — Nemotron-Labs-Diffusion, a tri-mode language model (LM) that unifies AR, diffusion, and self-speculation decoding within a single architecture, shows that AR and diffusion objectives are complementary.  
   _score 8.34 · 14 cites · 11▲ HF_
5. **[DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](2512.15713-diffusionvl-translating-any-autoregressive-models-into-diffusion-visio.md)** (2026-03) — DiffusionVL, a family of dVLMs obtained by translating pretrained AR models into the diffusion paradigm via an efficient diffusion finetuning procedure that changes the training objective and decoding process while …  
   _score 7.56 · 8 cites · 20▲ HF · [code](https://github.com/hustvl/DiffusionVL)_
6. **[CoDA: Coding LM via Diffusion Adaptation](2510.03270-coda-coding-lm-via-diffusion-adaptation.md)** (2025-09) — This release introduces CoDA, a 1.7B-parameter diffusion coder trained on TPU with a fully open-source training pipeline, enabling confidence-guided sampling that keeps inference latency competitive.  
   _score 6.23 · 5 cites · 43▲ HF · [code](https://github.com/SalesforceAIResearch/CoDA)_
7. **[Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation](2606.06712-data-efficient-autoregressive-to-diffusion-language-models-via-on-poli.md)** (2026-06) — An On-Policy Diffusion Language Model (OPDLM) in which On-Policy Distillation (OPD) is employed for ARLM-to-DLM transformation, which eliminates the train-inference mismatch in DLMs and enhances knowledge retention from …  
   _score 3.86 · 4 cites · 2▲ HF · [code](https://github.com/divelab/OPDLM)_
8. **[FLARE: Diffusion for Hybrid Language Model](2606.01774-flare-diffusion-for-hybrid-language-model.md)** (2026-08) — The results suggest that practical dLLMs are limited not only by decoding algorithms, but also by transfer data quality and the training inefficiency of current block-diffusion objectives, motivating joint design of …  
   _score 2.92 · 3 cites · [code](https://github.com/yuchen-zhu-zyc/FLARE)_
9. **[Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM](2604.06832-fast-dvlm-efficient-block-diffusion-vlm-via-direct-conversion-from-aut.md)** (2026-04) — A suite of multimodal diffusion adaptations, block size annealing, causal context attention, auto-truncation masking, and vision efficient concatenation, that collectively enable effective block diffusion in the VLM …  
   _score 2.85 · 6 cites_
10. **[Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment](2605.06885-don-t-retrain-align-adapting-autoregressive-lms-to-diffusion-lms-via-r.md)** (2026-05) — REPR-ALIGN is introduced, a representation alignment objective that adapts a bidirectional masked diffusion model to reuse representations from a pretrained AR model of identical architecture that suggests that …  
   _score 1.97 · 3 cites · [code](https://github.com/pengzhangzhi/Open-dLLM)_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models](2601.14758-mechanism-shift-during-post-training-from-autoregressive-to-masked-dif.md)** (2026-08-31; 0▲, 0 cites) — It is suggested that diffusion post-training selectively preserves or reorganizes inherited computation according to task structure, rather than uniformly replacing autoregressive …
- **[Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](2606.26493-nemotron-labs-twotower-diffusion-language-modeling-with-pretrained-aut.md)** (2026-06-29; 0▲, 1 cites) — TwoTower is proposed, a block-wise autoregressive diffusion model that decouples these roles into two towers: a frozen AR context tower that causally processes clean tokens, and a …
- **[BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merging and Stage-Wise Distillation](2604.16514-bard-bridging-autoregressive-and-diffusion-vision-language-models-via.md)** (2026-07-12; 0▲, 0 cites) — This work presents BARD, a simple and effective bridging framework that converts a pretrained autoregressive VLM into a same-architecture, decoding-efficient dVLM, and establishes …
- **[UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Objective](2607.24507-unifusion-adapting-autoregressive-language-models-into-discrete-diffus.md)** (2026-07-27; 0▲, 0 cites) — This work proposes a simple continual pre-training approach for directly adapting pretrained GPT2 checkpoints to uniform-noise diffusion, and establishes connections among SEDD, …
- **[PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention](2607.25157-prediff-lm-pretrained-discrete-masked-diffusion-language-modeling-with.md)** (2026-07-28; 0▲, 0 cites) — PreDiff-LM preserves causal attention within the observed prompt while allowing full bidirectional attention within the masked target, position hybrid attention as a complementary …
- **[dQwen3.5: Hybrid-Attention Diffusion Language Models](2609.20751-dqwen3-5-hybrid-attention-diffusion-language-models.md)** (2026-09-17; 0▲, 0 cites) — This work investigates whether hybrid backbones can become effective DLMs by adapting Qwen3.5 at 0.8B, 2B, 4B, and 9B scales, and finds that they can be efficient starting points …

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation](2510.06303-sdar-a-synergistic-diffusion-autoregression-paradigm-for-scalable-sequ.md) | 2025-10-18 | 11.46 | 114 | 0 | Annual Meeting of the Association for Co | [✓](https://github.com/InternLM/lmdeploy) | The SDAR model surpasses its AR counterpart on challenging scientific reasoning benchmarks such as GPQA and ChemBench, and gains further improvements under … |
| 2 | [Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](2512.14067-efficient-dlm-from-autoregressive-to-diffusion-language-models-and-bey.md) | 2026-04-29 | 9.5 | 24 | 18 |  |  | A continuous pretraining scheme with a block-wise attention pattern, which remains causal across blocks while enabling bidirectional modeling within each block … |
| 3 | [From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs](2512.06776-from-next-token-to-next-block-a-principled-adaptation-path-for-diffusi.md) | 2026-01-30 | 9.05 | 22 | 26 |  | [✓](https://github.com/YuchuanTian/NBDiff) | This work reframe the whole AR-to-DLM adaptation under the Block-Diffusion paradigm, transitioning from block size 1 to the final Block-Diffusion state, and … |
| 4 | [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation De](2607.05722-nemotron-labs-diffusion-a-tri-mode-language-model-unifying-autoregress.md) | 2026-07-07 | 8.34 | 14 | 11 |  |  | Nemotron-Labs-Diffusion, a tri-mode language model (LM) that unifies AR, diffusion, and self-speculation decoding within a single architecture, shows that AR … |
| 5 | [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](2512.15713-diffusionvl-translating-any-autoregressive-models-into-diffusion-visio.md) | 2026-03-31 | 7.56 | 8 | 20 |  | [✓](https://github.com/hustvl/DiffusionVL) | DiffusionVL, a family of dVLMs obtained by translating pretrained AR models into the diffusion paradigm via an efficient diffusion finetuning procedure that … |
| 6 | [CoDA: Coding LM via Diffusion Adaptation](2510.03270-coda-coding-lm-via-diffusion-adaptation.md) | 2025-09-27 | 6.23 | 5 | 43 |  | [✓](https://github.com/SalesforceAIResearch/CoDA) | This release introduces CoDA, a 1.7B-parameter diffusion coder trained on TPU with a fully open-source training pipeline, enabling confidence-guided sampling … |
| 7 | [Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation](2606.06712-data-efficient-autoregressive-to-diffusion-language-models-via-on-poli.md) | 2026-06-04 | 3.86 | 4 | 2 |  | [✓](https://github.com/divelab/OPDLM) | An On-Policy Diffusion Language Model (OPDLM) in which On-Policy Distillation (OPD) is employed for ARLM-to-DLM transformation, which eliminates the … |
| 8 | [FLARE: Diffusion for Hybrid Language Model](2606.01774-flare-diffusion-for-hybrid-language-model.md) | 2026-08-04 | 2.92 | 3 | 0 |  | [✓](https://github.com/yuchen-zhu-zyc/FLARE) | The results suggest that practical dLLMs are limited not only by decoding algorithms, but also by transfer data quality and the training inefficiency of … |
| 9 | [Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM](2604.06832-fast-dvlm-efficient-block-diffusion-vlm-via-direct-conversion-from-aut.md) | 2026-04-10 | 2.85 | 6 | 0 |  |  | A suite of multimodal diffusion adaptations, block size annealing, causal context attention, auto-truncation masking, and vision efficient concatenation, that … |
| 10 | [Don't Retrain, Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment](2605.06885-don-t-retrain-align-adapting-autoregressive-lms-to-diffusion-lms-via-r.md) | 2026-05-07 | 1.97 | 3 | 0 |  | [✓](https://github.com/pengzhangzhi/Open-dLLM) | REPR-ALIGN is introduced, a representation alignment objective that adapts a bidirectional masked diffusion model to reuse representations from a pretrained AR … |
| 11 | [T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforceme](2601.11214-t-star-progressive-block-scaling-for-masked-diffusion-language-models.md) | 2026-06-03 | 1.92 | 2 | 0 | Annual Meeting of the Association for Co |  | Starting from an AR-initialized small-block MDM, T$^\star$ transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance … |
| 12 | [Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context](2606.26493-nemotron-labs-twotower-diffusion-language-modeling-with-pretrained-aut.md) | 2026-06-29 | 1.32 | 1 | 0 |  | [✓](https://huggingface.co/collections/nvidia) | TwoTower is proposed, a block-wise autoregressive diffusion model that decouples these roles into two towers: a frozen AR context tower that causally processes … |
| 13 | [From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons](2605.27387-from-ar-to-diffusion-efficiently-adapting-large-language-models-with-s.md) | 2026-05-28 | 1.2 | 0 | 0 | Accepted by ACL 2026 | [✓](https://github.com/Oli-lab-nun/FLUID) | FLUID is proposed, a framework that efficiently adapts AR backbones to the diffusion paradigm, and enables seamless initialization from standard GPT-style … |
| 14 | [Mechanism Shift During Post-training from Autoregressive to Masked Diffusion Language Models](2601.14758-mechanism-shift-during-post-training-from-autoregressive-to-masked-dif.md) | 2026-08-31 | 0.7 | 0 | 0 | accepted for publication at EMNLP 2026 |  | It is suggested that diffusion post-training selectively preserves or reorganizes inherited computation according to task structure, rather than uniformly … |
| 15 | [BARD: Bridging AutoRegressive and Diffusion Vision-Language Models Via Highly Efficient Progressive Block Merg](2604.16514-bard-bridging-autoregressive-and-diffusion-vision-language-models-via.md) | 2026-07-12 | 0.5 | 0 | 0 |  | [✓](https://github.com/fudan-generative-vision/Bard-VL) | This work presents BARD, a simple and effective bridging framework that converts a pretrained autoregressive VLM into a same-architecture, decoding-efficient … |
| 16 | [UNIFUSION: Adapting Autoregressive Language Models into Discrete Diffusion under a Unified Reverse-Rate Object](2607.24507-unifusion-adapting-autoregressive-language-models-into-discrete-diffus.md) | 2026-07-27 | 0.0 | 0 | 0 |  |  | This work proposes a simple continual pre-training approach for directly adapting pretrained GPT2 checkpoints to uniform-noise diffusion, and establishes … |
| 17 | [PreDiff-LM: Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention](2607.25157-prediff-lm-pretrained-discrete-masked-diffusion-language-modeling-with.md) | 2026-07-28 | 0.0 | 0 | 0 |  |  | PreDiff-LM preserves causal attention within the observed prompt while allowing full bidirectional attention within the masked target, position hybrid … |
| 18 | [dQwen3.5: Hybrid-Attention Diffusion Language Models](2609.20751-dqwen3-5-hybrid-attention-diffusion-language-models.md) | 2026-09-17 | 0.0 | 0 | 0 |  |  | This work investigates whether hybrid backbones can become effective DLMs by adapting Qwen3.5 at 0.8B, 2B, 4B, and 9B scales, and finds that they can be … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Dream-Coder 7B: An Open Diffusion Language Model for Code](../../decoding/diffusion-language-models/2509.01142-dream-coder-7b-an-open-diffusion-language-model-for-code.md) | Diffusion language models (dLLMs) — models & training | 8.72 |
| [TESS 2: A Large-Scale Generalist Diffusion Language Model](../../decoding/diffusion-language-models/2502.13917-tess-2-a-large-scale-generalist-diffusion-language-model.md) | Diffusion language models (dLLMs) — models & training | 7.87 |
