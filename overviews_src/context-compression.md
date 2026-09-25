# Context & token compression (2025–2026)

> Synthesis of [`papers/context-compression`](../papers/context-compression/README.md). This is the cheapest speedup
> for agent and multimodal workloads: do not process tokens you do not need.

## Text / agent context ([folder](../papers/context-compression/prompt-and-context-compression/README.md))

* **Agents:** [[2510.00615|ACON]] (ICML'26) optimises compression of observations and history. It cuts peak tokens
  by 26–54% while *improving* success, and lets small LMs act as long-horizon agents.
  [[2601.16746|SWE-Pruner]] (coding agents: −23–54% tokens on SWE-Bench Verified) and
  [[2510.00446|LongCodeZip]] (function-level, then block-level, code compression).
* **Learned soft compression:** [[2606.09659|Latent Context LMs]] (end-to-end compression at scale),
  [[2510.20535|ARC-Encoder]] (one encoder serves many decoders), [[2505.12215|GMSA]] and
  [[2510.08907|Semantic anchors]].
* **Training-free selection:** [[2602.01719|COMI]] (marginal information gain) and [[2603.19635|BEAVER]]
  (page-level hierarchical selection, 26.4× lower latency at 128K).
* **Gist tokens as sparse attention:** [[2604.20920|Simplified Sparse Attention]] turns gist tokens into a
  log-linear decode hierarchy.
* **Optical compression:** DeepSeek-OCR (render text as image tokens, ~10× compression) and
  [[2510.17800|Glyph]] (3–4×).

## Visual tokens ([folder](../papers/context-compression/visual-token-reduction/README.md))

Visual tokens are 70–95% redundant. Training-free pruning is mature:

| Method | Idea | Result |
| --- | --- | --- |
| [[2502.11494\|DART]] | prune by *duplication*, not importance | −88.9% vision tokens, 2.99× prefill; works with FlashAttention |
| [[2503.02175\|DivPrune]] (CVPR'25) | maximise diversity | SOTA over 16 datasets, no fine-tuning |
| [[2506.10967\|CDPruner]] (NeurIPS'25) | instruction-conditioned DPP | −95% FLOPs, −78% latency, 94% accuracy on LLaVA |
| [[2508.18264\|MMTok]] (ICLR'26) | coverage of text + vision | 1.87× at 98.7%; 4 tokens keep 87.7% |
| [[2505.21334\|HoliTom]] (video) | global temporal segmentation + merging | −90% video tokens |
| [[2507.07990\|STTM]] (video) | multi-granular spatio-temporal merging | 2× at −0.5%, 3× at −2% |
| [[2602.04804\|OmniSIFT]] (omni) | modality-asymmetric compression | beats full context at 25% of tokens |

Runtime note: prefer methods compatible with FlashAttention, i.e. no attention-map dependence (DART, DivPrune,
[[2508.00367|Representation Shift]]), and apply them before the LLM or in early layers.
