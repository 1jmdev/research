**Verdict.** This is the largest decoding category (318 papers).

* **Production default (early 2026): EAGLE-3.** It has a feature-fusion drafter trained with "training-time test"
  and dynamic trees, and is in vLLM, SGLang and TensorRT-LLM.
* **The 2026 shift: block-diffusion drafters.** DFlash drafts a whole block in one forward pass conditioned on target
  features. Descendants (DDTree, JetSpec, DSpark, Domino, DFlare, PARD) add trees, semi-AR heads and confidence
  scheduling to fix suffix decay. This beats EAGLE-3 at larger draft budgets.
* **Draft training** is moving from KL-SFT to **acceptance-rate objectives** (LK losses) and **on-policy distillation**
  (Draft-OPD, AdaSPEC).
* **Lossy verification** (Judge Decoding, reward-guided SD, R2R token routing, Lookahead Reasoning at the step level)
  breaks the acceptance ceiling for reasoning models.

Production caution:
* [[2510.22876|Correctness forensics]]: several batched SD implementations **silently corrupt output** (the
  ragged-tensor problem).
* [[2601.11580|Speculative Decoding: Performance or Illusion?]]: gains shrink at high batch sizes.
* Benchmark on [[2604.09557|SPEED-Bench]] with throughput splits.

### Hand ranking

| # | Paper | Drafter | Key idea | Headline |
| ---: | --- | --- | --- | --- |
| 1 | [[2503.01840\|EAGLE-3]] (NeurIPS'25) | 1-layer feature drafter | Drop feature regression; **direct token prediction with multi-layer (low/mid/high) feature fusion**; training-time test (simulate multi-step drafting in training); scales with data | Up to ~6.5× latency speedup; ~1.4× throughput at batch 64 in SGLang. The industry baseline |
| 2 | [[2602.06036\|DFlash]] (ICML'26) | **Block diffusion** drafter | Draft a whole block in **one forward pass** conditioned on target hidden features | New state of the art over EAGLE-3 (~6× lossless on Qwen3-8B); cheap drafting at long blocks |
| 3 | [[2602.23881\|LK Losses]] (ICML'26) | Training objective | Losses that directly maximize **acceptance rate** instead of KL | Consistent gains for 4 drafter types, targets 8B–685B |
| 4 | [[2606.18394\|JetSpec]] | Parallel tree drafting | One-forward drafting + **path-consistent trees** (fixes diffusion drafters' inconsistent marginals) | Breaks the draft-budget scaling ceiling |
| 5 | [[2607.05147\|DSpark]] | Semi-AR | Parallel backbone + small sequential head; **confidence- and load-aware verification** | Keeps throughput at high concurrency |
| 6 | [[2505.21600\|R2R: Roads to Rome]] (NeurIPS'25) | SLM + LLM routing | Route only **path-divergent tokens** to the large model | 1.5B+32B router reaches R1-32B-level accuracy with ~5.6B average active params |
| 7 | [[2501.19324\|Reward-Guided SD]] (ICML'25) | Lossy | A PRM decides when to call the target; threshold mixture is provably optimal | Up to 4.4× fewer FLOPs on reasoning with *better* accuracy |
| 8 | [[2506.19830\|Lookahead Reasoning]] (NeurIPS'25) | Step-level | Draft several future **reasoning steps**; accept semantically correct ones | Adds step-level parallelism on top of token SD (1.4→2.1×) |
| 9 | [[2501.19309\|Judge Decoding]] (ICLR'25) | Lossy verify | A small judge head accepts "valid but different" tokens | Breaks the alignment ceiling of exact SD |
| 10 | [[2605.29343\|Draft-OPD]] / [[2510.19779\|AdaSPEC]] | Training | On-policy distillation on draft-induced states / filter hard tokens in KD | SFT plateaus; these push acceptance further |
| 11 | [[2502.14856\|FR-Spec]] (ACL'25) + [[2506.22694\|VocabTrim]] | Large-vocab | Draft over a **frequency-ranked vocabulary subset** | −75% draft LM-head cost for 128K+ vocabularies |
| 12 | [[2604.12989\|DDTree]] | Diffusion + tree | Best-first draft tree from block-diffusion per-position distributions | Longer acceptance than single-path DFlash |
| 13 | [[2502.10424\|QuantSpec]] / [[2503.13565\|ML-SpecQD]] | Self-spec, quantized | The draft = the same model with 4-bit weights and KV | No extra model; good for long context |
| 14 | [[2603.18567\|SpecForge]] (SGLang) | Tooling | Open, scalable training framework for EAGLE-3/DFlash drafters | Train drafters for your model |
| 15 | [[2510.22876\|Correctness Forensics for Batch SD]] (EMNLP'26) | Systems | Ragged tensors desynchronize position ids, masks and KV across a batch; EQSPEC/EXSPEC fixes | **Test your implementation** |

Also useful:
* **MoE targets:** [[2505.19645|MoESD]] (SD helps MoE more at medium batch), [[2602.16052|MoE-Spec]],
  [[2506.20675]].
* **Hybrid/SSM targets:** [[2505.14969|STree]], [[2608.01651|Bole]], [[2607.16673|SpecLA]].
* **Long context:** [[2502.17421|LongSpec]], [[2502.20330|RAPID]], [[2510.07535|OWL]].
* **Model-free (n-gram/retrieval):** [[2502.08923|CopySpec]], [[2507.01449|LogitSpec]],
  [[2606.05742|AdaPLD]].
* **Theory:** [[2505.07858|Scaling laws for SD]], [[2512.11718|Speed-of-light bounds]].
* **Production at scale:** [[2508.08192|Llama at scale]] (Meta).

**Runtime checklist.**
1. **Tree verification kernel** (tree attention mask + KV rollback) and **EAGLE-3 drafter support**. Add a **block
   drafter path** (DFlash-style: one forward, then tree or chain).
2. **Adaptive speculation length per batch load.** Turn SD off at high batch (DSpark, TETRIS, Nightjar). SD is a
   latency tool at low batch.
3. **Exactness test suite.** Greedy equivalence with and without SD, per batch shape. Use FP32 accumulation for the
   verification logits ([[2609.15504]]).
4. **Model builders.** Ship an MTP head or EAGLE-3/DFlash drafter with the model (Qwen/DeepSeek/GLM already do).
