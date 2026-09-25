**Verdict.** Token-level early exit and layer skipping mostly **fail to give wall-clock speedups** in batched serving.
Three reasons:
* **missing KV** for skipped layers;
* **batch divergence**, where different tokens exit at different depths;
* modern LLMs being "deep all the way" ([[2603.23701|The Diminishing Returns of Early-Exit Decoding in Modern LLMs]]).

FlexiDepth skips 8 of 32 layers in Llama-3-8B at full quality but reports **no throughput gain on GPUs**.

What does work:
1. **Per-input (prompt-level) depth programs.** Skip or repeat whole layers: Dr.LLM routers, PoLaR/CoLa. These *improve
   accuracy* at lower depth for many inputs and are batch-friendly per request.
2. **Early exit as self-speculative drafting.** Exit early to draft, then verify with the full model: SpecEE, LayerSkip
   lineage, KNN-SSD, CLaSp. Lossless and compatible with KV.
3. **Training-time layer dropout.** It makes models robust to later depth pruning ([[2609.05275|Don't Drop Dropout]],
   ICML'26).

| # | Paper | Granularity | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.12773\|Dr.LLM]] (ICLR'26) | Per-input, per-layer | Retrofittable routers decide **skip / execute / repeat** each block; supervised by MCTS-found configurations | Better accuracy **and** fewer layers on reasoning; generalizes out of domain |
| 2 | [[2606.06574\|PoLaR: program-of-layers]] (ICML'26) / [[2507.07996\|CoLa]] | Per-input | Pretrained layers can be **skipped or looped** into per-input programs; a small predictor generates them | Shorter programs keep or improve accuracy; can fix wrong answers |
| 3 | [[2504.08850\|SpecEE]] (ISCA'25) | Token, speculative | Lightweight exit predictors over *speculative* token candidates; two-level predictor scheduling; context-aware merged mapping | 2.25–2.43× on Llama-2 (cloud and PC) |
| 4 | [[2609.05275\|Don't Drop Dropout]] (ICML'26) | Training | Layer dropout with the right layer distribution and schedule gives **lower loss at equal FLOPs** + robustness to depth pruning | Re-adopt stochastic depth in pretraining |
| 5 | [[2503.23798\|FlexiDepth]] | Token | Plug-in router + adapter; skips 8/32 layers of Llama-3-8B | Shows *which* tokens need depth (arithmetic vs boilerplate); no GPU speedup |
| 6 | [[2506.21103\|Skip the Middle Layers]] | Architecture | Learned gate skips a symmetric span of middle blocks; gated attention hides skipped positions | Principled conditional depth (at small scale, gains are modest) |
| 7 | [[2501.02336\|AdaSkip]] (AAAI'25) | Sublayer, long context | On-the-fly similarity to skip attention/FFN sublayers in **prefill and decode** | Long-context acceleration |
| 8 | [[2604.18396\|River-LLM]] (ACL'26) | Token | **KV-shared exit river**: exited tokens get KV for skipped layers cheaply | Seamless token-level exit with real speedup |
| 9 | [[2506.03700\|AdaDecode]] (ICML'25) | Token | Predict a token early from an intermediate layer, start the next token, and complete the deferred layers in parallel later (keeps KV exact) | Up to 1.73× with output parity |
| 10 | [[2510.13876\|GateSkip]] | Token | Residual-stream sigmoid gates fine-tuned stably; rank tokens by gate and skip under a budget | Near-baseline quality at ~50% compute (instruct models) |
| 11 | [[2504.06949\|ACP for Forgetting Transformer]] | Attention FLOPs | Provably safe pruning of decayed attention in FoX | −70% attention FLOPs in pretraining |

**Runtime guidance.**
* Prefer **self-speculative early exit**: verification makes it lossless and it reuses the KV of the full model. Or use
  **request-level depth programs** (Dr.LLM), which batch cleanly.
* Only use token-level skipping if the engine can **rebatch by depth** ([[2512.15705|DREX]]) and fill skipped-layer KV
  (River-LLM / KV copy from the exit layer).
