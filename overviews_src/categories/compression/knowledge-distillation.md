**Verdict.** Distillation is the largest compression bucket (~250 papers), and in 2026 it is dominated by one idea:
**on-policy distillation (OPD)**. The student samples its own trajectories and the teacher gives **dense per-token
supervision** (reverse KL on student-visited states). This removes the exposure bias of SFT and sequence-level KD, gives
RL-like generalization at SFT-like cost, and is now used in frontier post-training (Qwen3 strong-to-weak, MiMo-V2-Flash's
multi-teacher OPD). The main threads:

1. **Making OPD work reliably.**
   * [[2604.13016|Rethinking OPD]]: success = progressive alignment on a small shared high-probability token set (97–99%
     of mass). Fix failures with an off-policy cold start + teacher-aligned prompts.
   * [[2603.25562|Revisiting OPD]]: sampled-token OPD has three failure modes; distilling over the teacher-supported
     token set gives +19.8%.
   * [[2603.07079|Entropy-aware OPD]] (ICML'26): add forward KL where the teacher is uncertain.
   * [[2604.13010|Lightning OPD]]: *offline* OPD with teacher consistency; Qwen3-8B to 69.9% AIME'24 in **30 GPU-hours**,
     no live teacher server.
   * [[2609.04172|OPD II]]: a single prompt already covers 71.5% of the states full-data OPD visits.
2. **Multi-teacher and cross-scale.**
   * [[2606.30406|MOPD]]: domain teachers developed independently, merged into one student on its own rollouts. Beats
     mixed RL, cascaded RL and parameter merging on Qwen3-30B-A3B.
   * [[2607.05394|Direct-OPD]]: transfer the teacher's *RL-induced shift* (log-ratio to its pre-RL reference) → weak
     teachers improve strong students (Qwen3-1.7B 48.3 → 58.3 AIME'24 in 4 h on 8×A100).
   * [[2511.10643|GAD]]: black-box OPD with a discriminator; Qwen2.5-14B ≈ GPT-5-Chat on LMSYS-Chat.
3. **Self-distillation** (teacher = the same model with privileged context: the answer, a demonstration, a
   self-revision):
   * [[2601.19897|SDFT]]: continual learning without forgetting;
   * [[2601.18734|OPSD]], [[2604.12002|SD-Zero]], [[2604.03128|RLSD]].

   It has sharp failure modes: it suppresses uncertainty expression, with up to −40% OOD
   ([[2603.24472|why self-distillation degrades reasoning]]), and privileged teachers leak.
4. **Classic logit KD at scale, for pretraining and pruning recovery.**
   * [[2502.08606|Distillation Scaling Laws]] (Apple, ICML'25): distillation beats supervised pretraining **only if the
     teacher already exists or serves many students**, up to a predictable compute level.
   * [[2509.01649|Distilled pretraining]]: better test-time scaling, but hurts in-context learning (induction heads).
   * Cheap logits: [[2503.16870|sparse logit sampling]] (unbiased, <10% overhead vs CE), [[2608.03796]] (offline top-K +
     fused chunked KL).
   * [[2505.12781|Low-Rank Clone]]: soft-prune teacher weights + activation cloning → 3–7B students matching
     trillion-token models with **20B tokens**.
5. **Reasoning distillation data.** [[2502.12143|Small models struggle to learn from strong reasoners]] (ACL'25): mix
   long and short CoT for small students. See also [[2502.18001]] and [[2509.22193|Scale or Reason?]] (compute-equivalent
   analysis).

### Cost table (post-training distillation)

| Method | Setting | H100-h | Basis |
| --- | --- | ---: | --- |
| [[2607.05394\|Direct-OPD]] | Qwen3-1.7B, weak teacher, +10 AIME pts | **~10** | reported: 8×A100 × 4 h |
| [[2604.13010\|Lightning OPD]] | Qwen3-8B SFT-init → 69.9% AIME'24 | **~30** | reported: 30 GPU-h |
| [[2604.13010\|Lightning OPD]] | Qwen3-30B-A3B → 71.0% AIME'24 | **one 8×H100 node** (≈ tens–hundreds) | reported |
| [[2505.12781\|Low-Rank Clone]] | 3–7B student from Llama/Qwen teacher, 20B tokens | **~1–3K** | estimate |
| [[2503.16870\|Sparse logit sampling]] | Pretraining KD, 300M–3B | **+<10%** over CE training | reported |

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2604.13016\|Rethinking OPD]] | OPD science + recipe | Mechanism (shared high-probability token alignment), failure diagnosis, fixes | The reference for making OPD work; questions long-horizon scaling |
| 2 | [[2606.30406\|MOPD]] (Xiaomi) | Multi-teacher OPD | Distil several domain teachers into the student on its own rollouts | Inherits nearly all teacher capability; used for MiMo-V2-Flash |
| 3 | [[2502.08606\|Distillation Scaling Laws]] (ICML'25) | Pretraining KD | Scaling law for student loss vs teacher, student and data; when KD beats supervised | **Decide whether to distil at all** |
| 4 | [[2604.13010\|Lightning OPD]] | Efficient OPD | Offline OPD with teacher-consistency condition (same teacher for SFT and OPD) | 69.9% AIME'24 on Qwen3-8B in 30 GPU-h |
| 5 | [[2603.25562\|Revisiting OPD: failure modes and fixes]] | OPD recipe | Teacher-supported-set objective + top-p rollouts + special-token masking | +19.8% over sampled-token OPD across reasoning and agentic tasks |
| 6 | [[2607.05394\|Direct-OPD]] | Weak-to-strong | Distil the teacher's RL policy *shift*, not its policy | Reuse RL results across scales; composable shifts |
| 7 | [[2601.19897\|SDFT: self-distillation enables continual learning]] | Self-distillation | Demonstration-conditioned model as its own teacher, on-policy | Learns new skills with much less forgetting than SFT |
| 8 | [[2505.12781\|Low-Rank Clone]] | Pretraining-scale KD | Low-rank projections compress teacher weights + clone FFN/attention activations | >1,000× token efficiency for 3–7B students |
| 9 | [[2502.12143\|Small models struggle to learn from strong reasoners]] (ACL'25) | Data | Learnability gap; Mix Distillation (long+short CoT, large+small teachers) | Much better small-model reasoning |
| 10 | [[2511.10643\|GAD: black-box OPD]] | Black-box | Generator-discriminator on-policy distillation from API teachers | Beats SeqKD; 14B student ≈ GPT-5-Chat on LMSYS |
| 11 | [[2503.16870\|Sparse Logit Sampling]] | Efficiency | Importance-sampled sparse logits (unbiased) instead of top-K caching | Pretraining KD at <10% overhead |
| 12 | [[2603.24472\|Why self-distillation degrades reasoning]] | Failure analysis | Self-distillation suppresses uncertainty expression | Up to −40% OOD on Qwen3/Olmo3; **check before adopting OPSD** |

**Also useful.**
* OPD objectives: [[2601.07155]], [[2606.01249|trust-region OPD]], [[2605.07865]] (control variates),
  [[2606.17199|PowerOPD]], [[2604.08527]] (length inflation), [[2603.11137]] (relaxed OPD).
* Agentic OPD: [[2607.04763]] (multi-turn with prefix replay), [[2605.07725|SOD]], [[2505.17612]] (agent distillation,
  NeurIPS'25).
* Combining with RL: [[2506.02208|KDRL]], [[2602.22495]], [[2608.24696]].
* Classic KD objectives: [[2503.07067|DistiLLM-2]] (ICML'25), [[2501.16937|TAID]], [[2505.16297|ToDi]].
* Teacher hacking: [[2502.02671]].
* Boomerang distillation (zero-shot size interpolation): [[2510.05064]].
* Tooling: [[2603.01875|KDFlow]]. Surveys: [[2604.00626]], [[2504.14772]].

**Recommendation.**
* *Model builders.* Post-train small models with **SFT cold start → OPD from the strongest same-tokenizer teacher**
  (cross-tokenizer: see [`tokenizer-and-vocab-transfer`](../../model-conversion/tokenizer-and-vocab-transfer/README.md)).
  Budget tens to hundreds of H100-h per 8B student. For several domains, train specialists and combine them with MOPD
  instead of mixing RL.
* *Runtime / infra.* OPD needs a fast **student rollout engine plus teacher log-prob scoring** on the same tokens (a
  prefill-only teacher pass). Batch the teacher scoring as prefix-cached prefill. Offline OPD (Lightning) removes the
  live teacher entirely.
