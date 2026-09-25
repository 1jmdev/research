**Verdict.** Diffusion LLMs (dLLMs) went from curiosity to frontier-scale in 18 months.

**Scaling milestones:**
* **LLaDA 8B** from scratch, on par with Llama-3-8B;
* **Dream 7B**, AR-initialized from Qwen2.5;
* **LLaDA-MoE** (~20T tokens) and **LLaDA 2.0**, 100B MoE *converted from an AR model*;
* **DiffusionGemma**.

**Commercial speed:** Mercury (≈1,100 tok/s on H100), Gemini Diffusion, Seed Diffusion (2,146 tok/s on H20).

**Core architecture choice: block diffusion (BD3-LM).** Autoregressive across blocks, diffusion within a block. It fixes
the two big dLLM problems: variable length and **KV caching**. Nearly every serious 2026 dLLM is block-wise.

**Open problems:**
* **Real parallelism.** Fast dLLMs often decode almost left-to-right ([[2602.23225]], [[2601.15165|The Flexibility Trap]]).
* **The factorization barrier.** Parallel tokens are sampled independently.
* **RL post-training** under intractable likelihood: d1/diffu-GRPO, wd1, SPG, TraceRL, d2.

**Where dLLMs genuinely win:** data-constrained pretraining (**"super data learners"**: they beat AR when unique data is
scarce, by using many more epochs), infilling/editing, and code.

### Hand ranking

| # | Paper | Type | Key idea | Result / compute |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.09992\|LLaDA]] (NeurIPS'25) | From-scratch MDM, 8B | Masked diffusion with Transformer denoiser; pretrain + SFT | On par with Llama-3-8B; beats GPT-4o on reversal poem completion. 2.3T tokens, ~0.13M H800-h |
| 2 | [[2503.09573\|Block Diffusion (BD3-LM)]] (ICLR'25 oral) | Architecture | AR over blocks + diffusion within a block; variance-reduced training; data-driven noise schedules | Arbitrary length, KV cache, state of the art among diffusion LMs. **The design everyone builds on** |
| 3 | [[2512.15745\|LLaDA 2.0]] | AR→dLLM at 100B | 3-phase **block-level WSD conversion** (grow block → full-sequence diffusion → shrink block) from Ling MoE; SFT + DPO | 16B and 100B MoE dLLMs; frontier-scale deployment path. See [`ar-to-diffusion`](../../model-conversion/ar-to-diffusion/README.md) |
| 4 | [[2508.15487\|Dream 7B]] | AR-init dLLM | Init from Qwen2.5-7B + context-adaptive token-level noise rescheduling | Strongest open dLLM of its time; planning/infilling strengths. ~580B tokens |
| 5 | [[2508.02193\|Seed Diffusion Preview]] (ByteDance) | Code dLLM | Two-stage curriculum, constrained-order training, on-policy learning for speed | **2,146 tok/s on H20** at competitive code quality |
| 6 | [[2506.17298\|Mercury]] (Inception) | Commercial dLLM | Diffusion Transformer LLM for code | ~1,100 tok/s (H100) at GPT-4o-mini-class code quality |
| 7 | [[2511.03276\|dLLMs are Super Data Learners]] | Scaling | Controlled crossover: with limited unique data, dLLMs beat AR by training many more epochs | 1.7B dLLM on 10B unique Python tokens beats matched AR. Key for data-limited builders |
| 8 | [[2506.20639\|DiffuCoder]] (Apple) | Code + RL | Local/global "AR-ness" metrics; **coupled-GRPO** | 7B code dLLM on 130B tokens; ~1.5K H100-h for the RL stage (64 H100 × 24 h) |
| 9 | [[2505.19223\|LLaDA 1.5 / VRPO]] | Alignment | Variance-reduced ELBO preference optimization (optimal MC allocation, antithetic sampling) | Better math/code/alignment; **~405 H100-h** |
| 10 | [[2509.06949\|TraceRL / TraDo]] | RL | Trajectory-aware RL with a diffusion value model; also adapts block size | TraDo-4B beats 7B AR models on math |
| 11 | [[2506.10892\|The Diffusion Duality (Duo)]] (ICML'25) | Uniform-state diffusion | Uniform discrete diffusion as argmax of Gaussian diffusion → curriculum + **discrete consistency distillation** | 2× faster training; **two orders of magnitude fewer sampling steps** |
| 12 | [[2503.00307\|ReMDM]] (NeurIPS'25) | Sampler | **Remasking** backward process = inference-time scaling for pretrained MDMs | Approaches AR quality with more steps |
| 13 | [[2506.01928\|Esoteric LMs]] (ICML'26) | Hybrid AR/MDM | Causal-attention MDM ⇒ exact likelihood + **KV caching** with parallel generation | Interpolates AR ↔ MDM perplexity |
| 14 | [[2502.06768\|Train for the Worst, Plan for the Best]] (ICML'25 outstanding paper) | Theory | MDMs train on intractable infilling subproblems; **adaptive decoding order** sidesteps them | Sudoku 7%→90% with order planning |
| 15 | [[2602.22661\|dLLM framework]] (ACL'26) | Tooling | Unified training/inference/eval for LLaDA, Dream etc. | Start here to reproduce |

Also important:
* **Multimodal:** [[2505.15809|MMaDA]], [[2505.16933|LLaDA-V]], [[2505.16839|LaViDa]].
* **Long context:** [[2506.14429|LongLLaDA]], [[2510.10481|UltraLLaDA]].
* **Scaling laws:** [[2510.03280|Quokka]], [[2512.10858]].
* **Continuous and latent diffusion is catching up:** [[2604.11748|LangFlow]], [[2605.06548]], [[2605.18530]].
* **Reality checks:** [[2601.12979|Bitter lesson for agentic workflows]], [[2602.23225]].
* **RL for dLLMs:** [[2504.12216|d1]], [[2507.08838|wd1]], [[2510.09541|SPG]].

**For a runtime.** dLLM serving needs a different engine loop: block-wise denoising steps with **approximate or exact KV
caching** (Fast-dLLM, dKV-Cache, Esoteric/BD3 exact caching), confidence-threshold parallel unmasking, and remasking.
See [`diffusion-llm-inference`](../diffusion-llm-inference/README.md) for the inference toolbox.
