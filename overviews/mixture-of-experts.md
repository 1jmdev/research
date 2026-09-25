# Mixture of Experts (2025–2026)

> Synthesis of [`papers/mixture-of-experts`](../papers/mixture-of-experts/README.md). Upcycling (dense→MoE) is in
> [model conversion](model-conversion.md). Most flagship open models are MoE; see
> [technical reports](../papers/models-and-architectures/technical-reports/README.md) (DeepSeek-V3/V4, Qwen3,
> Kimi K2, GLM-4.5, Ling/Ring, LongCat, gpt-oss, MiniMax).

## TL;DR

* **Architecture defaults (2025–26):**
  * Fine-grained experts (64–384) with 1–2 shared experts, top-k of 4–8, and about 3–10% activation.
  * Aux-loss-free or **global-batch** load balancing: [Demons in the Detail](../papers/mixture-of-experts/training/2501.11873-demons-in-the-detail-on-implementing-load-balancing-loss-for-training.md) computes the balance loss
    over the global batch, which improves both perplexity and specialization.
  * Router–expert coupling losses: [ERC loss](../papers/mixture-of-experts/architecture-and-routing/2512.23447-coupling-experts-and-routers-in-mixture-of-experts-via-an-auxiliary-lo.md) (ICLR'26).
* **New sparsity axes beyond experts:**
  * [Engram / conditional memory](../papers/models-and-architectures/novel-architectures/2601.07372-conditional-memory-via-scalable-lookup-a-new-axis-of-sparsity-for-larg.md) (DeepSeek): O(1) N-gram lookup memory as a *second* sparse axis
    (+5 BBH, +3.4 MMLU).
  * [Scaling embeddings beats scaling experts](../papers/models-and-architectures/novel-architectures/2601.21204-scaling-embeddings-outperforms-scaling-experts-in-language-models.md) (LongCat-Flash-Lite, 68.5B total / 3B active).
  * Looped MoE: [SMELT](../papers/training/scaling-laws/2609.01343-smelt-scaling-laws-for-compute-matched-moe-looped-transformers.md) gives scaling laws for compute-matched looped MoE transformers.
* **RL on MoE is unstable** because training and inference engines route differently. [Rollout Routing
  Replay (R3)](../papers/training/rl-for-reasoning/2510.11370-stabilizing-moe-reinforcement-learning-by-aligning-training-and-infere.md) records the inference-time routing and replays it in training. Support this in the runtime: expose
  routing decisions.

## Serving MoE ([folder](../papers/mixture-of-experts/inference-and-serving/README.md))

| Setting | Technique | Papers |
| --- | --- | --- |
| Datacenter, large EP | wide expert parallelism with PD disaggregation, INT8 | [CloudMatrix384](../papers/serving-systems/hardware-accelerators/2506.12708-serving-large-language-models-on-huawei-cloudmatrix384.md) (EP320, 6.7k prefill tok/s/NPU on DeepSeek-R1), [ELDR](../papers/mixture-of-experts/inference-and-serving/2607.00466-eldr-expert-locality-aware-decode-routing-for-pd-disaggregated-moe-ser.md) (route decode by prefill expert signature, −6–14% TPOT) |
| Load imbalance | dynamic token/expert re-routing | [LLEP](../papers/mixture-of-experts/inference-and-serving/2601.17111-least-loaded-expert-parallelism-load-balancing-an-imbalanced-mixture-o.md) (up to 5× over standard EP, 1.9× on gpt-oss-120b), [MoETuner](../papers/mixture-of-experts/inference-and-serving/2502.06643-moetuner-optimized-mixture-of-expert-serving-with-balanced-expert-plac.md) (ILP expert placement) |
| Single GPU + CPU | hybrid CPU-GPU experts, expert cache | [HybriMoE](../papers/mixture-of-experts/inference-and-serving/2504.05897-hybrimoe-hybrid-cpu-gpu-scheduling-and-cache-management-for-efficient.md) (DAC'25), [Klotski](../papers/mixture-of-experts/inference-and-serving/2502.06888-klotski-efficient-mixture-of-expert-inference-via-expert-aware-multi-b.md) (expert-aware multi-batch pipeline) |
| Edge | bandwidth-adaptive execution | [FreeToken](../papers/mixture-of-experts/inference-and-serving/2608.16157-freetoken-efficient-edge-native-moe-serving-with-bandwidth-adaptive-ex.md), [SmallThinker](../papers/models-and-architectures/small-language-models/2507.20984-smallthinker-a-family-of-efficient-large-language-models-natively-trai.md) (pre-attention router hides expert loading; >20 tok/s on consumer CPUs) |
| Offloading-friendliness | measure *local routing consistency* first | [Not All Models Suit Expert Offloading](../papers/mixture-of-experts/inference-and-serving/2505.16056-not-all-models-suit-expert-offloading-on-local-routing-consistency-of.md) |

## Compressing MoE ([folder](../papers/mixture-of-experts/compression/README.md))

* **Pruning beats merging for one-shot compression**: [REAP](../papers/mixture-of-experts/compression/2510.13999-reap-the-experts-why-pruning-prevails-for-one-shot-moe-compression.md) uses router-weighted expert activation. It
  is near-lossless for code at 50% expert pruning on Qwen3-Coder-480B and Kimi-K2.
* Basis sharing: [MoBE](../papers/mixture-of-experts/compression/2508.05257-mobe-mixture-of-basis-experts-for-compressing-moe-based-llms.md) cuts DeepSeek-V3 / Kimi-K2 / Qwen3-235B parameters by 24–30% at 1–2% accuracy
  cost. [Sub-MoE](../papers/mixture-of-experts/compression/2506.23266-sub-moe-efficient-mixture-of-expert-llms-compression-via-subspace-expe.md) does joint SVD merging.
* Non-uniform allocation: [DiEP](../papers/mixture-of-experts/compression/2509.16105-diep-adaptive-mixture-of-experts-compression-through-differentiable-ex.md) and [EvoESAP](../papers/mixture-of-experts/compression/2603.06003-evoesap-non-uniform-expert-pruning-for-sparse-moe.md) (+19.6 MATH-500 at 50%).
* Domain pruning: [Domain-Specific Pruning of Large Mixture-of-Experts Models with Few-shot Demonstrations](../papers/mixture-of-experts/compression/2504.06792-domain-specific-pruning-of-large-mixture-of-experts-models-with-few-sh.md) keeps half the experts of DeepSeek-R1/V3 for about 3× throughput at equal memory.
* Runtime expert skipping for VLM-MoE: [MoDES](../papers/mixture-of-experts/compression/2511.15690-modes-accelerating-mixture-of-experts-multimodal-large-language-models.md).
* Pre-training scale: [SlimQwen](../papers/mixture-of-experts/compression/2605.08738-slimqwen-exploring-the-pruning-and-distillation-in-large-moe-model-pre.md) (prune + distill during MoE pre-training).

## Training systems ([folder](../papers/mixture-of-experts/training/README.md))

[FSMoE](../papers/mixture-of-experts/training/2501.10714-fsmoe-a-flexible-and-scalable-training-system-for-sparse-mixture-of-ex.md) (1.2–3× over DeepSpeed-MoE/Tutel), [VeOmni](../papers/training/pretraining-recipes-and-efficiency/2508.02317-veomni-scaling-any-modality-model-training-with-model-centric-distribu.md) (omni-modal MoE recipes) and
[MixNet](../papers/mixture-of-experts/training/2501.03905-mixnet-a-runtime-reconfigurable-optical-electrical-fabric-for-distribu.md) (reconfigurable optical fabric for all-to-all). Communication–compute overlap papers
([Comet](../papers/serving-systems/distributed-inference-and-parallelism/2502.19811-comet-fine-grained-computation-communication-overlapping-for-mixture-o.md)) are in [serving systems / parallelism](serving-systems.md).
