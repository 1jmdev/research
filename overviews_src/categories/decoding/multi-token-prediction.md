**Verdict.** Multi-token prediction (MTP) has two uses.

1. **A pretraining auxiliary loss.** DeepSeek-V3-style sequential MTP modules are now standard in DeepSeek, Qwen3-Next,
   GLM-4.5, MiMo and others. The quality gains from *exact* future-token MTP are inconsistent at small scale, so softer
   targets work better:
   * token order prediction (TOP);
   * future summaries (FSP);
   * register tokens (MuToR) for fine-tuning.
2. **A built-in speculative drafter.** The MTP head drafts and the main model verifies. The main levers:
   * **self-distillation**: FastMTP lifts 3rd-token acceptance from 2% to 36%, and MTP-D adds 7.5%;
   * **recursive/shared heads**;
   * **windowed draft attention** at million-token context ([[2607.21535|Windowed-MTP]]; otherwise the draft's full-KV
     read dominates).

A third, emerging use is converting a pretrained AR model into a **standalone multi-token generator** with no separate
verifier (MTP via self-distillation, MARS, K-Forcing). It is a cousin of diffusion conversion (see
[`model-conversion/ar-to-diffusion`](../../model-conversion/ar-to-diffusion/README.md)).

### Hand ranking

| # | Paper | Use | Key idea | Result / cost |
| ---: | --- | --- | --- | --- |
| 1 | [[2509.18362\|FastMTP]] | MTP → SD | Fine-tune **one shared-weight MTP head** on self-distilled data for recursive drafting + language-aware vocabulary compression | 2.03× over NTP (+82% vs vanilla MTP); <1 day on one H20 node (~4 H100-h) |
| 2 | [[2507.11851\|Your LLM Knows the Future]] (Apple) | Retrofit MTP | Mask-token inputs + **gated LoRA** (keeps NTP exact) + learnable sampler head | ~5× speedup on code/math with no quality loss |
| 3 | [[2508.19228\|Token Order Prediction (TOP)]] | Pretraining aux | Rank upcoming tokens by proximity (learning-to-rank) instead of exact MTP; one extra unembedding | Beats NTP, MTP and DeepSeek-MTP at 340M/1.8B/7B; 7B run ≈ 3K H100-h |
| 4 | [[2505.10518\|MuToR: MTP needs registers]] (NeurIPS'25) | FT / pretrain aux | Interleave **register tokens** that predict future targets; no architecture change | MTP benefits carried to SFT |
| 5 | [[2510.14751\|Future Summary Prediction]] (Meta) | Pretraining aux | Predict a **summary** (bag-of-words or learned reverse-LM embedding) of the long-term future | Gains on reasoning/planning at 3B/8B with 1T tokens |
| 6 | [[2603.23911\|MTP-D]] | Pretraining + SD | Self-distillation for MTP heads + **looped extension** to 8–16 heads | +7.5% acceptance; +35% speed from head extension |
| 7 | [[2602.06019\|MTP via self-distillation]] | Standalone MTP | Online distillation turns an AR model into a multi-token model with the **same implementation**, no verifier | >3× faster, <5% GSM8K drop |
| 8 | [[2604.07023\|MARS]] | Standalone MTP | Mask-autoregression fine-tuning on instruction data; no new params; still callable as NTP | Multiple tokens per pass with no NTP degradation |
| 9 | [[2603.17942\|ESP: embedding-space probing]] (ICML'26) | Training-free MTP | Probe with mask tokens from the embedding space + dynamic tree | Beats training-free baselines (LADE); vs EAGLE-3's 200–300 GPU-h training |
| 10 | [[2505.17505\|L-MTP]] (NeurIPS'25) | Leap MTP | Predict **non-adjacent** future tokens + a matching decoding strategy | Better long-range dependency and speed |
| 11 | [[2607.21535\|Windowed-MTP]] | Serving | Sliding window + sink for the **draft head's** attention only | Keeps MTP speculation profitable at 1M context |
| 12 | [[2604.11912\|How transformers learn to plan via MTP]] (COLM'26) | Theory | MTP induces reverse reasoning (attend to goal, trace back) via gradient decoupling | Explains the planning benefit |

**For a runtime.**
* Treat a model's MTP head as an EAGLE-like drafter: chain or tree drafting with the MTP module reusing the main
  model's KV.
* Window the draft attention at long context.
* Choose speculation depth by entropy or load ([[2606.27550|EntMTP]]).

**For model builders.** Train with 1–2 MTP modules (DeepSeek-V3 recipe) or TOP/FSP-style soft auxiliaries. Then
**self-distill the MTP head** before release (FastMTP) to maximize acceptance.
