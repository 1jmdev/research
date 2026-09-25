# Mixture of Experts (2025–2026)

> Synthesis of [`papers/mixture-of-experts`](../papers/mixture-of-experts/README.md). Upcycling (dense→MoE) is in
> [model conversion](model-conversion.md). Most flagship open models are MoE; see
> [technical reports](../papers/models-and-architectures/technical-reports/README.md) (DeepSeek-V3/V4, Qwen3,
> Kimi K2, GLM-4.5, Ling/Ring, LongCat, gpt-oss, MiniMax).

## TL;DR

* **Architecture defaults (2025–26):**
  * Fine-grained experts (64–384) with 1–2 shared experts, top-k of 4–8, and about 3–10% activation.
  * Aux-loss-free or **global-batch** load balancing: [[2501.11873|Demons in the Detail]] computes the balance loss
    over the global batch, which improves both perplexity and specialization.
  * Router–expert coupling losses: [[2512.23447|ERC loss]] (ICLR'26).
* **New sparsity axes beyond experts:**
  * [[2601.07372|Engram / conditional memory]] (DeepSeek): O(1) N-gram lookup memory as a *second* sparse axis
    (+5 BBH, +3.4 MMLU).
  * [[2601.21204|Scaling embeddings beats scaling experts]] (LongCat-Flash-Lite, 68.5B total / 3B active).
  * Looped MoE: [[2609.01343|SMELT]] gives scaling laws for compute-matched looped MoE transformers.
* **RL on MoE is unstable** because training and inference engines route differently. [[2510.11370|Rollout Routing
  Replay (R3)]] records the inference-time routing and replays it in training. Support this in the runtime: expose
  routing decisions.

## Serving MoE ([folder](../papers/mixture-of-experts/inference-and-serving/README.md))

| Setting | Technique | Papers |
| --- | --- | --- |
| Datacenter, large EP | wide expert parallelism with PD disaggregation, INT8 | [[2506.12708\|CloudMatrix384]] (EP320, 6.7k prefill tok/s/NPU on DeepSeek-R1), [[2607.00466\|ELDR]] (route decode by prefill expert signature, −6–14% TPOT) |
| Load imbalance | dynamic token/expert re-routing | [[2601.17111\|LLEP]] (up to 5× over standard EP, 1.9× on gpt-oss-120b), [[2502.06643\|MoETuner]] (ILP expert placement) |
| Single GPU + CPU | hybrid CPU-GPU experts, expert cache | [[2504.05897\|HybriMoE]] (DAC'25), [[2502.06888\|Klotski]] (expert-aware multi-batch pipeline) |
| Edge | bandwidth-adaptive execution | [[2608.16157\|FreeToken]], [[2507.20984\|SmallThinker]] (pre-attention router hides expert loading; >20 tok/s on consumer CPUs) |
| Offloading-friendliness | measure *local routing consistency* first | [[2505.16056]] |

## Compressing MoE ([folder](../papers/mixture-of-experts/compression/README.md))

* **Pruning beats merging for one-shot compression**: [[2510.13999|REAP]] uses router-weighted expert activation. It
  is near-lossless for code at 50% expert pruning on Qwen3-Coder-480B and Kimi-K2.
* Basis sharing: [[2508.05257|MoBE]] cuts DeepSeek-V3 / Kimi-K2 / Qwen3-235B parameters by 24–30% at 1–2% accuracy
  cost. [[2506.23266|Sub-MoE]] does joint SVD merging.
* Non-uniform allocation: [[2509.16105|DiEP]] and [[2603.06003|EvoESAP]] (+19.6 MATH-500 at 50%).
* Domain pruning: [[2504.06792]] keeps half the experts of DeepSeek-R1/V3 for about 3× throughput at equal memory.
* Runtime expert skipping for VLM-MoE: [[2511.15690|MoDES]].
* Pre-training scale: [[2605.08738|SlimQwen]] (prune + distill during MoE pre-training).

## Training systems ([folder](../papers/mixture-of-experts/training/README.md))

[[2501.10714|FSMoE]] (1.2–3× over DeepSpeed-MoE/Tutel), [[2508.02317|VeOmni]] (omni-modal MoE recipes) and
[[2501.03905|MixNet]] (reconfigurable optical fabric for all-to-all). Communication–compute overlap papers
([[2502.19811|Comet]]) are in [serving systems / parallelism](serving-systems.md).
