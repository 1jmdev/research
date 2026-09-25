**Verdict.** Pure recurrent models (Mamba-3, RWKV-7, xLSTM 7B) are now competitive at small and medium scale. They have
provably **better state tracking** than Transformers: RWKV-7 recognizes all regular languages, while Transformers are
limited to TC⁰. They still trail on **associative recall** at long context, the gather-and-aggregate skill. Two lines
of work address that gap:

* **Richer state updates.** Diagonal → diagonal + low-rank (DeltaProduct, PD-SSM) → non-linear or implicit RNNs
  (ParaRNN, M²RNN).
* **Bigger or smarter memory.** Test-time memorization (Titans, ATLAS, the Miras framework); mixture-of-memories;
  sparse delta memory; memory caching.

In production the answer is **hybrids**: see [`hybrid-architectures`](../hybrid-architectures/README.md) and
[`linear-attention`](../linear-attention/README.md).

### Hand ranking

| # | Paper | Key idea | Why it matters |
| ---: | --- | --- | --- |
| 1 | [[2603.15569\|Mamba-3]] (ICLR'26) | Inference-first SSM: a more expressive (trapezoidal) discretization, **complex-valued states** (rotations → state tracking) and **MIMO** updates for higher arithmetic intensity at decode | Best pure SSM for decode efficiency and quality; the reference for SSM kernels |
| 2 | [[2503.14456\|RWKV-7 "Goose"]] | Generalized delta rule with vector gating and in-context learning rates; 2.9B trained on 3.1T open tokens | 3B multilingual state of the art at constant memory; provable state tracking. Open data |
| 3 | [[2501.00663\|Titans]] (Google, NeurIPS'25) | **Neural long-term memory** (MLP) updated at test time by surprise-gated gradient descent, combined with attention as short-term memory | Scales to >2M context; the anchor of the test-time-memorization line |
| 4 | [[2504.13173\|Miras: It's All Connected]] | Unifies Transformers, Titans and linear RNNs as associative memory with an **attentional-bias objective + retention gate**; introduces Moneta/Yaad/Memora | Design framework for new recurrent layers |
| 5 | [[2505.23735\|ATLAS]] | Sliding-window (Omega) memory learning, higher-order feature maps, Muon-style memory optimizer | Beats Titans and Transformers on long-context recall |
| 6 | [[2502.10297\|DeltaProduct]] (NeurIPS'25) | Multiple delta-rule steps per token = **products of Householder matrices**; tunable expressivity | Cleanly trades compute for state tracking |
| 7 | [[2502.13685\|MoM: Mixture-of-Memories]] | Router sends tokens to **multiple independent memory states** | Large recall gains at constant memory |
| 8 | [[2506.05233\|MesaNet]] (ICLR'26) | Chunkwise-parallel **locally optimal** TTT (solves in-context regression exactly with CG) | Strong recall; principled |
| 9 | [[2504.10449\|M1]] | Hybrid Mamba **reasoning model** distilled from R1 then RL'd | Matches R1-distill at the same scale with >3× generation throughput, so more samples fit a fixed test-time budget |
| 10 | [[2607.07386\|Sparse Delta Memory]] | GDN with **sparse reads/writes to a large explicit memory** | Orders-of-magnitude bigger state at iso-FLOPs |
| 11 | [[2602.24281\|Memory Caching]] | Cache checkpoints of RNN states so memory grows with length | Interpolates between RNN and Transformer memory |
| 12 | [[2510.21450\|ParaRNN]] (Apple) | Newton-based **parallel training of non-linear RNNs** at 7B | Revives non-linear RNNs for LLMs |
| 13 | [[2503.13427\|xLSTM 7B]] | mLSTM-based 7B LLM optimized for fast inference | Competitive 7B recurrent model |
| 14 | [[2504.18574\|Gather-and-Aggregate skill gap]] | Recall failures trace to a few G&A heads, which SSMs implement poorly | Explains why a few attention layers fix hybrids |
| 15 | [[2507.02782\|Length generalization in recurrent models]] (ICML'25) | "Unexplored states" hypothesis; state-init interventions fix length generalization | Cheap training fix |

**Runtime notes.**
* SSM decode is **state-update bound**, not bandwidth bound on KV. Mamba-3 MIMO raises arithmetic intensity.
  Implement the fused selective-scan decode step and **state snapshots** for speculative decoding and prefix caching.
* For RWKV-7 and xLSTM, specialized kernels (WKV7, mLSTM chunkwise) matter more than for Transformers. Watch the
  numerical-precision sensitivity the RWKV-7 authors report.
