# Jacobi, lookahead & consistency-based parallel decoding

Fixed-point (Jacobi) iteration decoding, lookahead decoding, consistency LLMs, Jacobi forcing, n-gram parallel decoding.

**16 papers** (first submitted 2025-01-01 → 2026-09-25), ranked by impact score (see [METHODOLOGY.md](../../../METHODOLOGY.md)). Up: [Decoding: speculative, parallel, MTP, diffusion](../README.md) · [Index](../../../README.md)

📖 Written overview of this area: [../../../overviews/decoding.md](../../../overviews/decoding.md)

## 🔬 Analyst notes: hand ranking and verdict

_Written after reading the abstracts, and the full text where available, of this category's papers. The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just citations. The automatic impact ranking follows below._

**Verdict.** Parallel decoding *inside* an AR model now has three practical forms.

1. **Jacobi / consistency-trained decoding.** Jacobi Forcing is the 2025 state of the art. It uses progressive
   distillation on the model's own Jacobi trajectories with a noise schedule. The model stays **causal**, so exact KV
   reuse is kept (unlike dLLMs). It adds rejection-recycling and multi-block decoding: **up to 3.8× speedup** on code
   and math at near-AR quality.
2. **Masked-token prediction in an AR model.** Set Block Decoding (Meta) samples several *non-consecutive* future
   tokens in parallel with discrete-diffusion solvers. There are no architecture changes and KV caching stays exact
   (3–5× fewer forward passes).
3. **Semantic / structural parallelism.** The model decides to fork independent branches:
   * Multiverse: MapReduce-style reasoning with Multiverse Attention;
   * PASTA: learned asynchronous decoding;
   * ASPD, parallel reasoning within one sequence;
   * intra-prompt parallel QA (IPPD, HPD).

Before chasing speedups, read [How much parallelism is "free"?](2605.30851-how-much-parallelism-is-free-a-principle-of-near-free-parallelism-for.md). The hardware only offers a bounded number
of near-free positions per forward, set by memory-bound slack and kernel granularity.

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [Jacobi Forcing](2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md) | Causal parallel decoder | Progressive distillation on the model's own Jacobi trajectories (noise-aware, packed sequences) + rejection recycling + multi-block decoding | Up to **3.8×** with AR-level quality; beats AR→dLLM conversions on speed/quality |
| 2 | [Set Block Decoding](2509.04185-set-block-decoding-is-a-language-model-inference-accelerator.md) (Meta) | NTP + masked prediction | Fine-tune to fill **sets of future tokens** in parallel; use diffusion solvers (EB-sampler); exact KV cache | 3–5× fewer forward passes at equal accuracy (Llama-3.1-8B, Qwen-3-8B) |
| 3 | [Multiverse](2506.09991-multiverse-your-language-models-secretly-decide-how-to-parallelize-and.md) (NeurIPS'25) | Native parallel reasoning | Map → parallel Process → Reduce inside the model; Multiverse Attention + an engine; converts AR models with ~1K examples | Parallel reasoning with real wall-clock speedup at AR-level accuracy |
| 4 | [PASTA](2502.11517-learning-to-keep-a-promise-scaling-language-model-decoding-parallelism.md) (ICML'25) | Learned async decoding | The model annotates independent chunks (PASTA-LANG); an interpreter decodes them in parallel | Pareto-better speed/quality than heuristic parallel decoding |
| 5 | [One-step text generation](2505.21189-exploring-the-hidden-capacity-of-llms-for-one-step-text-generation.md) (EMNLP'25) | Analysis | Frozen LLMs can emit **hundreds of tokens in one forward pass** from two learned embeddings | Shows latent multi-token capacity |
| 6 | [PCCoT](2506.18582-parallel-continuous-chain-of-thought-with-jacobi-iteration.md) (EMNLP'25) | Latent CoT + Jacobi | Jacobi iteration over continuous thought tokens | ~50% less training and inference time for latent CoT |
| 7 | [Any-subset AR + ASSD](2504.20456-reviving-any-subset-autoregressive-models-with-principled-parallel-sam.md) | Correct parallel sampling | Any-subset AR models can **self-verify** parallel drafts, giving the exact joint distribution | Provable correctness, unlike dLLM parallel sampling |
| 8 | [ASPD](2508.08895-aspd-unlocking-adaptive-serial-parallel-decoding-by-exploring-intrinsi.md) | Intrinsic parallelism | Extract parallelizable branches from AR outputs; branch-invisible masks with shared position ids | Latency reduction without quality loss |
| 9 | [Near-Free Parallelism](2605.30851-how-much-parallelism-is-free-a-principle-of-near-free-parallelism-for.md) | Systems | Predicts how many positions per forward are ~free for dense, MoE and attention layers | Sets realistic speedup ceilings |

**Runtime requirements.**
* Support **multi-position decode steps**: a verify/accept loop with tree or block masks and KV commit/rollback, plus
  **fork/join of branches sharing a prefix** (Multiverse/PASTA). This is the same machinery speculative decoding needs,
  so build it once.

## 🏆 Best of the best by impact score (top 10)

1. **[Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation](2506.09991-multiverse-your-language-models-secretly-decide-how-to-parallelize-and.md)** (2025-06) — This work introduces Multiverse, a new generative model that enables natively parallel generation in sequential generation and open-sourced the entire Multiverse ecosystem, including data, model weights, engine, as well …  
   _score 11.81 · Neural Information Processing Systems (Neural Inf Process Sy · 35 cites · 55▲ HF · [code](https://github.com/Multiverse4FM/Multiverse) · ~53 H100-h_
2. **[Set Block Decoding is a Language Model Inference Accelerator](2509.04185-set-block-decoding-is-a-language-model-inference-accelerator.md)** (2025-09) — This work introduces Set Block Decoding (SBD), a simple and flexible paradigm that accelerates generation by integrating standard next token prediction (NTP) and masked token prediction (MATP) within a single …  
   _score 8.46 · 19 cites · 54▲ HF_
3. **[Parallel Continuous Chain-of-Thought with Jacobi Iteration](2506.18582-parallel-continuous-chain-of-thought-with-jacobi-iteration.md)** (2026-02) — This paper proposes Parallel Continuous Chain-of-Thought (PCCoT), which performs Jacobi iteration on the latent thought tokens, updating them iteratively in parallel instead of sequentially and thus improving both …  
   _score 7.59 · Accepted to EMNLP 2025 main c · 26 cites · [code](https://github.com/whyNLP/PCCoT)_
4. **[Exploring the Hidden Capacity of LLMs for One-Step Text Generation](2505.21189-exploring-the-hidden-capacity-of-llms-for-one-step-text-generation.md)** (2025-11) — It is shown that frozen LLMs can generate hundreds of accurate tokens in just one token-parallel forward pass, when provided with only two learned embeddings, revealing a surprising and underexplored multi-token …  
   _score 7.17 · Conference on Empirical Methods in Natural Language Processi · 5 cites · 62▲ HF · [code](https://github.com/Glebzok/OneStepLLMGeneration)_
5. **[Fast and Accurate Causal Parallel Decoding using Jacobi Forcing](2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md)** (2025-12) — Jacobi Forcing is introduced, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly shifting AR models into efficient parallel decoders while …  
   _score 6.96 · 7 cites · 44▲ HF · [code](https://github.com/hao-ai-lab/JacobiForcing)_
6. **[Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding](2502.11517-learning-to-keep-a-promise-scaling-language-model-decoding-parallelism.md)** (2025-02) — PASTA is a learning-based system that teaches LLMs to identify semantic independence and express parallel decoding opportunities in their own responses and PASTA-LANG is an annotation language that enables LLMs to …  
   _score 5.9 · International Conference on Machine Learning (ICML) · 31 cites_
7. **[Accelerate Parallelizable Reasoning via Parallel Decoding within One Sequence](2503.20533-accelerate-parallelizable-reasoning-via-parallel-decoding-within-one-s.md)** (2025-08) — This work leverages the inherent parallelizability of certain tasks to accelerate the reasoning process when multiple parallel reasoning steps exist, and decode multiple tokens per forward pass via a tree-like attention …  
   _score 5.28 · Conference on Empirical Methods in Natural Language Processi · 5 cites · 12▲ HF · [code](https://github.com/yuyijiong/parallel-decoding-in-one-sequence)_
8. **[A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](2508.08712-a-survey-on-parallel-text-generation-from-parallel-decoding-to-diffusi.md)** (2026-02) — A systematic survey of parallel text generation methods is presented, categorizing existing approaches into AR-based and Non-AR-based paradigms, and providing a detailed examination of the core techniques within each …  
   _score 4.77 · 27 cites · [code](https://github.com/zhanglingzhe0820/Awesome-Parallel-Text-Generation)_
9. **[ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs](2508.08895-aspd-unlocking-adaptive-serial-parallel-decoding-by-exploring-intrinsi.md)** (2025-08) — This paper introduces a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models, and proposes an Adaptive Serial-Parallel Decoding (ASPD), …  
   _score 2.92 · 9 cites_
10. **[Intra-Prompt Parallel Decoding for Common-Context Question Answering](2609.05707-intra-prompt-parallel-decoding-for-common-context-question-answering.md)** (2026-09) — Intra-Prompt Parallel Decoding (IPPD), a novel inference method that answers multiple common-context questions in parallel within a single prompt without requiring fine-tuning or any changes to the LLM architecture, is …  
   _score 2.72 · Accepted to EMNLP 2026 main c · 1 cites · [code](https://github.com/networkslab/IPPD) · ~0.00 H100-h_

## 🆕 Recent papers to watch (last 90 days)

Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the last three months.

- **[Gumbel Distillation for Parallel Text Generation](2603.22216-gumbel-distillation-for-parallel-text-generation.md)** (2026-07-23; 0▲, 1 cites) — Gumbel Distillation substantially improves the generation quality of parallel language models, achieving a 30.0% improvement in MAUVE score and 10.5% in generative perplexity over …

## 💻 Compute cost (estimated H100-hours, auto-extracted)

Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.

| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |
| --- | ---: | ---: | --- | --- |
| [Intra-Prompt Parallel Decoding for Common-Context Question Answering](2609.05707-intra-prompt-parallel-decoding-for-common-context-question-answering.md) | 0.00 | 0.00 | L40S/RTX6000Ada | We run all primary experiments on the Amazon EC2 g6e.8xlarge server using one Nvidia L40S 48GB GPU.… |
| [Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Ge](2506.09991-multiverse-your-language-models-secretly-decide-how-to-parallelize-and.md) | 53 | 53 | B200 | Fine-tuning took 3 hours on 8 NVIDIA B200 GPUs with PyTorch FSDP.… |

## Full ranking

| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | [Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation](2506.09991-multiverse-your-language-models-secretly-decide-how-to-parallelize-and.md) | 2025-06-13 | 11.81 | 35 | 55 | Neural Information Processing Systems (N | [✓](https://github.com/Multiverse4FM/Multiverse) | This work introduces Multiverse, a new generative model that enables natively parallel generation in sequential generation and open-sourced the entire … |
| 2 | [Set Block Decoding is a Language Model Inference Accelerator](2509.04185-set-block-decoding-is-a-language-model-inference-accelerator.md) | 2025-09-04 | 8.46 | 19 | 54 |  |  | This work introduces Set Block Decoding (SBD), a simple and flexible paradigm that accelerates generation by integrating standard next token prediction (NTP) … |
| 3 | [Parallel Continuous Chain-of-Thought with Jacobi Iteration](2506.18582-parallel-continuous-chain-of-thought-with-jacobi-iteration.md) | 2026-02-26 | 7.59 | 26 | 0 | Accepted to EMNLP 2025 main c | [✓](https://github.com/whyNLP/PCCoT) | This paper proposes Parallel Continuous Chain-of-Thought (PCCoT), which performs Jacobi iteration on the latent thought tokens, updating them iteratively in … |
| 4 | [Exploring the Hidden Capacity of LLMs for One-Step Text Generation](2505.21189-exploring-the-hidden-capacity-of-llms-for-one-step-text-generation.md) | 2025-11-01 | 7.17 | 5 | 62 | Conference on Empirical Methods in Natur | [✓](https://github.com/Glebzok/OneStepLLMGeneration) | It is shown that frozen LLMs can generate hundreds of accurate tokens in just one token-parallel forward pass, when provided with only two learned embeddings, … |
| 5 | [Fast and Accurate Causal Parallel Decoding using Jacobi Forcing](2512.14681-fast-and-accurate-causal-parallel-decoding-using-jacobi-forcing.md) | 2025-12-16 | 6.96 | 7 | 44 |  | [✓](https://github.com/hao-ai-lab/JacobiForcing) | Jacobi Forcing is introduced, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly … |
| 6 | [Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding](2502.11517-learning-to-keep-a-promise-scaling-language-model-decoding-parallelism.md) | 2025-02-21 | 5.9 | 31 | 0 | International Conference on Machine Lear |  | PASTA is a learning-based system that teaches LLMs to identify semantic independence and express parallel decoding opportunities in their own responses and … |
| 7 | [Accelerate Parallelizable Reasoning via Parallel Decoding within One Sequence](2503.20533-accelerate-parallelizable-reasoning-via-parallel-decoding-within-one-s.md) | 2025-08-26 | 5.28 | 5 | 12 | Conference on Empirical Methods in Natur | [✓](https://github.com/yuyijiong/parallel-decoding-in-one-sequence) | This work leverages the inherent parallelizability of certain tasks to accelerate the reasoning process when multiple parallel reasoning steps exist, and … |
| 8 | [A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models](2508.08712-a-survey-on-parallel-text-generation-from-parallel-decoding-to-diffusi.md) | 2026-02-10 | 4.77 | 27 | 0 |  | [✓](https://github.com/zhanglingzhe0820/Awesome-Parallel-Text-Generation) | A systematic survey of parallel text generation methods is presented, categorizing existing approaches into AR-based and Non-AR-based paradigms, and providing … |
| 9 | [ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs](2508.08895-aspd-unlocking-adaptive-serial-parallel-decoding-by-exploring-intrinsi.md) | 2025-08-14 | 2.92 | 9 | 0 |  |  | This paper introduces a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models, … |
| 10 | [Intra-Prompt Parallel Decoding for Common-Context Question Answering](2609.05707-intra-prompt-parallel-decoding-for-common-context-question-answering.md) | 2026-09-04 | 2.72 | 1 | 0 | Accepted to EMNLP 2026 main c | [✓](https://github.com/networkslab/IPPD) | Intra-Prompt Parallel Decoding (IPPD), a novel inference method that answers multiple common-context questions in parallel within a single prompt without … |
| 11 | [Reviving Any-Subset Autoregressive Models with Principled Parallel Sampling and Speculative Decoding](2504.20456-reviving-any-subset-autoregressive-models-with-principled-parallel-sam.md) | 2025-04-29 | 2.57 | 13 | 0 |  | [✓](https://github.com/gabeguo/any-order-speculative-decoding) | Furthermore, we provide a mathematically justified scheme for training AS-ARMs for generation, and show that AS-ARMs achieve state-of-the-art performance among … |
| 12 | [Gumbel Distillation for Parallel Text Generation](2603.22216-gumbel-distillation-for-parallel-text-generation.md) | 2026-07-23 | 2.2 | 1 | 0 | ICLR 2026 | [✓](https://github.com/hxixixh/gumbel-distill) | Gumbel Distillation substantially improves the generation quality of parallel language models, achieving a 30.0% improvement in MAUVE score and 10.5% in … |
| 13 | [Parallel Sampling via Autospeculation](2511.07869-parallel-sampling-via-autospeculation.md) | 2025-11-11 | 2.06 | 5 | 0 | Symposium on the Theory of Computing (Sy |  | This work presents parallel algorithms to accelerate sampling via counting in two settings: any-order autoregressive models and denoising diffusion models and … |
| 14 | [Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction](2604.26209-breaking-the-autoregressive-chain-hyper-parallel-decoding-for-efficien.md) | 2026-04-29 | 1.78 | 1 | 0 | Annual Meeting of the Association for Co | [✓](https://github.com/networkslab/HPD) | Hyper-Parallel Decoding is presented, a novel decoding algorithm that accelerates offline decoding by leveraging both shared memory and computation across … |
| 15 | [SimpleTool: Parallel Decoding for Real-Time LLM Function Calling](2603.00030-simpletool-parallel-decoding-for-real-time-llm-function-calling.md) | 2026-02-04 | 0.43 | 1 | 0 |  |  | SimpleTool is presented, which introduces special tokens that serve a dual role: compressing low-entropy tokens while acting as mode selectors that enable … |
| 16 | [How Much Parallelism Is "Free"? A Principle of Near-Free Parallelism for Parallel Decoding](2605.30851-how-much-parallelism-is-free-a-principle-of-near-free-parallelism-for.md) | 2026-05-29 | 0.0 | 0 | 0 |  |  | A Near-Free Parallelism principle is established that predicts the NFP boundary from hardware balance and implementation granularity, revealing that the … |

## Also relevant (primary category elsewhere)

| Paper | Primary category | Score |
| --- | --- | ---: |
| [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](../diffusion-llm-inference/2505.22618-fast-dllm-training-free-acceleration-of-diffusion-llm-by-enabling-kv-c.md) | Diffusion LLM inference acceleration | 17.7 |
| [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](../diffusion-llm-inference/2505.16990-dimple-discrete-diffusion-multimodal-large-language-model-with-paralle.md) | Diffusion LLM inference acceleration | 16.85 |
| [The Diffusion Duality](../diffusion-language-models/2506.10892-the-diffusion-duality.md) | Diffusion language models (dLLMs) — models & training | 16.19 |
| [Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing](../diffusion-llm-inference/2508.09192-diffusion-llms-can-do-faster-than-ar-inference-via-discrete-diffusion.md) | Diffusion LLM inference acceleration | 13.87 |
| [ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](../diffusion-llm-inference/2510.04767-parallelbench-understanding-the-trade-offs-of-parallel-decoding-in-dif.md) | Diffusion LLM inference acceleration | 13.32 |
| [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](../speculative-decoding/2607.05147-dspark-confidence-scheduled-speculative-decoding-with-semi-autoregress.md) | Speculative decoding (draft–verify) | 12.62 |
| [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](../diffusion-llm-inference/2506.00413-accelerating-diffusion-llms-via-adaptive-parallel-decoding.md) | Diffusion LLM inference acceleration | 12.54 |
| [dParallel: Learnable Parallel Decoding for dLLMs](../diffusion-llm-inference/2509.26488-dparallel-learnable-parallel-decoding-for-dllms.md) | Diffusion LLM inference acceleration | 11.75 |
| [DMax: Aggressive Parallel Decoding for dLLMs](../diffusion-llm-inference/2604.08302-dmax-aggressive-parallel-decoding-for-dllms.md) | Diffusion LLM inference acceleration | 9.39 |
| [Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall](../diffusion-language-models/2510.19304-loopholing-discrete-diffusion-deterministic-bypass-of-the-sampling-wal.md) | Diffusion language models (dLLMs) — models & training | 8.12 |
| [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](../diffusion-llm-inference/2509.26432-adablock-dllm-semantic-aware-diffusion-llm-inference-via-adaptive-bloc.md) | Diffusion LLM inference acceleration | 8.07 |
| [LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding](../diffusion-llm-inference/2512.16229-lopa-scaling-dllm-inference-via-lookahead-parallel-decoding.md) | Diffusion LLM inference acceleration | 7.44 |
| [dMoE: dLLMs with Learnable Block Experts](../diffusion-llm-inference/2605.30876-dmoe-dllms-with-learnable-block-experts.md) | Diffusion LLM inference acceleration | 6.71 |
| [Accelerating Diffusion LLM Inference via Local Determinism Propagation](../diffusion-llm-inference/2510.07081-accelerating-diffusion-llm-inference-via-local-determinism-propagation.md) | Diffusion LLM inference acceleration | 5.36 |
| [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](../diffusion-llm-inference/2509.25188-learning-to-parallel-accelerating-diffusion-large-language-models-via.md) | Diffusion LLM inference acceleration | 5.31 |
| [CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../diffusion-llm-inference/2510.06133-creditdecoding-accelerating-parallel-decoding-in-diffusion-large-langu.md) | Diffusion LLM inference acceleration | 5.29 |
| [On the Role of Discreteness in Diffusion LLMs](../diffusion-language-models/2512.22630-on-the-role-of-discreteness-in-diffusion-llms.md) | Diffusion language models (dLLMs) — models & training | 5.19 |
| [Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts](../diffusion-language-models/2510.05040-test-time-scaling-in-diffusion-llms-via-hidden-semi-autoregressive-exp.md) | Diffusion language models (dLLMs) — models & training | 5.04 |
| [Why Diffusion Language Models Struggle with Truly Parallel (Non-Autoregressive) Decoding?](../diffusion-language-models/2602.23225-why-diffusion-language-models-struggle-with-truly-parallel-non-autoreg.md) | Diffusion language models (dLLMs) — models & training | 4.83 |
| [DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](../diffusion-llm-inference/2603.12996-dapd-dependency-aware-parallel-decoding-via-attention-for-diffusion-ll.md) | Diffusion LLM inference acceleration | 4.78 |
| [DAWN: Dependency-Aware Fast Inference for Diffusion LLMs](../diffusion-llm-inference/2602.06953-dawn-dependency-aware-fast-inference-for-diffusion-llms.md) | Diffusion LLM inference acceleration | 4.58 |
| [From Bits to Rounds: Parallel Decoding with Exploration for Diffusion Language Models](../diffusion-llm-inference/2511.21103-from-bits-to-rounds-parallel-decoding-with-exploration-for-diffusion-l.md) | Diffusion LLM inference acceleration | 4.26 |
| [Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models](../diffusion-llm-inference/2510.00294-free-draft-and-verification-toward-lossless-parallel-decoding-for-diff.md) | Diffusion LLM inference acceleration | 4.09 |
| [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](../diffusion-llm-inference/2609.26796-flash-dllm-io-aware-kv-caching-and-parallel-decoding-for-fast-memory-e.md) | Diffusion LLM inference acceleration | 3.98 |
| [FourierSampler: Unlocking Non-Autoregressive Potential in Diffusion Language Models via Frequency-Guided Gener](../diffusion-language-models/2601.23182-fouriersampler-unlocking-non-autoregressive-potential-in-diffusion-lan.md) | Diffusion language models (dLLMs) — models & training | 3.96 |
| [DART: Distilling Autoregressive Reasoning to Silent Thought](../../reasoning/latent-and-looped/2506.11752-dart-distilling-autoregressive-reasoning-to-silent-thought.md) | Latent reasoning, looped & recurrent-depth models | 3.94 |
| [Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion ](../diffusion-llm-inference/2606.26120-dynamic-dllm-dynamic-cache-budget-and-adaptive-parallel-decoding-for-t.md) | Diffusion LLM inference acceleration | 3.7 |
| [Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language](../diffusion-language-models/2604.10567-early-decisions-matter-proximity-bias-and-initial-trajectory-shaping-i.md) | Diffusion language models (dLLMs) — models & training | 3.63 |
| [Reward-Weighted Sampling: Enhancing Non-Autoregressive Characteristics in Masked Diffusion LLMs](../diffusion-language-models/2509.00707-reward-weighted-sampling-enhancing-non-autoregressive-characteristics.md) | Diffusion language models (dLLMs) — models & training | 3.3 |
| [Corrective Diffusion Language Models](../diffusion-language-models/2512.15596-corrective-diffusion-language-models.md) | Diffusion language models (dLLMs) — models & training | 3.16 |
| [CD4LM: Consistency Distillation and aDaptive Decoding for Diffusion Language Models](../diffusion-llm-inference/2601.02236-cd4lm-consistency-distillation-and-adaptive-decoding-for-diffusion-lan.md) | Diffusion LLM inference acceleration | 2.75 |
| [When Confidence Misleads: Suffix Anchoring and Anchor-Proximity Confidence Modulation for Diffusion Language M](../diffusion-llm-inference/2605.28181-when-confidence-misleads-suffix-anchoring-and-anchor-proximity-confide.md) | Diffusion LLM inference acceleration | 2.67 |
| [Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models](../diffusion-llm-inference/2604.02560-dependency-guided-parallel-decoding-in-discrete-diffusion-language-mod.md) | Diffusion LLM inference acceleration | 2.11 |
| [Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](../speculative-decoding/2511.20340-scaling-llm-speculative-decoding-non-autoregressive-forecasting-in-lar.md) | Speculative decoding (draft–verify) | 2.1 |
| [Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](../diffusion-llm-inference/2604.08964-breaking-block-boundaries-anchor-based-history-stable-decoding-for-dif.md) | Diffusion LLM inference acceleration | 2.03 |
| [Generation Order and Parallel Decoding in Masked Diffusion Models: An Information-Theoretic Perspective](../diffusion-llm-inference/2602.00286-generation-order-and-parallel-decoding-in-masked-diffusion-models-an-i.md) | Diffusion LLM inference acceleration | 2.01 |
| [SpecFLASH: A Latent-Guided Semi-autoregressive Speculative Decoding Framework for Efficient Multimodal Generat](../speculative-decoding/2505.12728-specflash-a-latent-guided-semi-autoregressive-speculative-decoding-fra.md) | Speculative decoding (draft–verify) | 1.98 |
| [From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding](../speculative-decoding/2608.02123-from-chains-to-trees-parent-conditioned-drafting-for-semi-autoregressi.md) | Speculative decoding (draft–verify) | 1.85 |
| [CForce: Boosting Parallel Decoding for dLLMs via Consistency Forcing](../diffusion-llm-inference/2608.13925-cforce-boosting-parallel-decoding-for-dllms-via-consistency-forcing.md) | Diffusion LLM inference acceleration | 1.77 |
| [Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](../diffusion-llm-inference/2602.22868-rejection-mixing-fast-semantic-propagation-of-mask-tokens-for-efficien.md) | Diffusion LLM inference acceleration | 1.7 |
| [Time-Annealed Perturbation Sampling: Diverse Generation for Diffusion Language Models](../diffusion-language-models/2601.22629-time-annealed-perturbation-sampling-diverse-generation-for-diffusion-l.md) | Diffusion language models (dLLMs) — models & training | 1.53 |
| [Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](../diffusion-llm-inference/2602.00879-dynamic-expert-sharing-decoupling-memory-from-parallelism-in-mixture-o.md) | Diffusion LLM inference acceleration | 1.48 |
| [Locally Coherent Parallel Decoding in Diffusion Language Models](../diffusion-llm-inference/2603.20216-locally-coherent-parallel-decoding-in-diffusion-language-models.md) | Diffusion LLM inference acceleration | 1.34 |
| [Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models](../diffusion-llm-inference/2601.12247-plan-verify-and-fill-a-structured-parallel-decoding-approach-for-diffu.md) | Diffusion LLM inference acceleration | 1.21 |
| [Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge](../diffusion-llm-inference/2606.08048-diffusion-language-model-parallel-decoding-via-product-of-experts-brid.md) | Diffusion LLM inference acceleration | 1.2 |
| [DiLaDiff: Distilled Latent-Augmented Diffusion for Language Modeling](../diffusion-language-models/2605.23605-diladiff-distilled-latent-augmented-diffusion-for-language-modeling.md) | Diffusion language models (dLLMs) — models & training | 1.15 |
| [Consistent Diffusion Language Models](../diffusion-language-models/2605.00161-consistent-diffusion-language-models.md) | Diffusion language models (dLLMs) — models & training | 0.7 |
| [Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence Extrapolation](../diffusion-llm-inference/2605.30753-efficient-diffusion-llms-via-temporal-spatial-parallel-decoding-and-co.md) | Diffusion LLM inference acceleration | 0.68 |
| [NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding](../../mixture-of-experts/inference-and-serving/2605.21100-nanocp-request-level-dynamic-context-parallelism-for-data-expert-paral.md) | MoE inference & serving (expert offloading, EP) | 0.64 |
| [Block-R1: Rethinking the Role of Block Size in Multi-domain Reinforcement Learning for Diffusion Large Languag](../diffusion-language-models/2605.11726-block-r1-rethinking-the-role-of-block-size-in-multi-domain-reinforceme.md) | Diffusion language models (dLLMs) — models & training | 0.5 |
| [DC-Leap: Training-Free Acceleration of dLLMs via Draft-Guided Contiguous Leaping Decoding](../diffusion-llm-inference/2607.20467-dc-leap-training-free-acceleration-of-dllms-via-draft-guided-contiguou.md) | Diffusion LLM inference acceleration | 0.5 |
| [Retrofitting Linear Attention into Diffusion Language Models](../../model-conversion/transformer-to-linear-or-hybrid/2608.06628-retrofitting-linear-attention-into-diffusion-language-models.md) | Transformer → linear / SSM / hybrid conversion (linearization) | 0.5 |
| [Accelerating Diffusion Language Models via Structured Suffix Modeling](../diffusion-llm-inference/2608.23167-accelerating-diffusion-language-models-via-structured-suffix-modeling.md) | Diffusion LLM inference acceleration | 0.5 |
| [TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding](../speculative-decoding/2609.22098-treespark-calibrated-load-adaptive-draft-trees-for-semi-autoregressive.md) | Speculative decoding (draft–verify) | 0.5 |
| [Latent Shadows: The Gaussian-Discrete Duality in Masked Diffusion](../diffusion-language-models/2602.00792-latent-shadows-the-gaussian-discrete-duality-in-masked-diffusion.md) | Diffusion language models (dLLMs) — models & training | 0.0 |
| [Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding](../diffusion-llm-inference/2602.23792-divide-and-conquer-accelerating-diffusion-based-large-language-models.md) | Diffusion LLM inference acceleration | 0.0 |
| [LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection](../diffusion-llm-inference/2605.10980-leap-unlocking-dllm-parallelism-via-lookahead-early-convergence-token.md) | Diffusion LLM inference acceleration | 0.0 |
| [Cluster-Level Attention-Guided Parallel Decoding for Masked Diffusion Language Models](../diffusion-llm-inference/2605.29607-cluster-level-attention-guided-parallel-decoding-for-masked-diffusion.md) | Diffusion LLM inference acceleration | 0.0 |
| [Reconsidering Positional Supervision in Masked Diffusion Language Model Training](../diffusion-language-models/2601.22947-reconsidering-positional-supervision-in-masked-diffusion-language-mode.md) | Diffusion language models (dLLMs) — models & training | 0.0 |
| [Mean-Field Parallel Decoding for Discrete Diffusion Language Models](../diffusion-llm-inference/2606.15805-mean-field-parallel-decoding-for-discrete-diffusion-language-models.md) | Diffusion LLM inference acceleration | 0.0 |
