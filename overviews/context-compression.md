# Context & token compression (2025–2026)

> Synthesis of [`papers/context-compression`](../papers/context-compression/README.md). This is the cheapest speedup
> for agent and multimodal workloads: do not process tokens you do not need.

## Text / agent context ([folder](../papers/context-compression/prompt-and-context-compression/README.md))

* **Agents:** [ACON](../papers/context-compression/prompt-and-context-compression/2510.00615-acon-optimizing-context-compression-for-long-horizon-llm-agents.md) (ICML'26) optimises compression of observations and history. It cuts peak tokens
  by 26–54% while *improving* success, and lets small LMs act as long-horizon agents.
  [SWE-Pruner](../papers/context-compression/prompt-and-context-compression/2601.16746-swe-pruner-self-adaptive-context-pruning-for-coding-agents.md) (coding agents: −23–54% tokens on SWE-Bench Verified) and
  [LongCodeZip](../papers/context-compression/prompt-and-context-compression/2510.00446-longcodezip-compress-long-context-for-code-language-models.md) (function-level, then block-level, code compression).
* **Learned soft compression:** [Latent Context LMs](../papers/context-compression/prompt-and-context-compression/2606.09659-end-to-end-context-compression-at-scale.md) (end-to-end compression at scale),
  [ARC-Encoder](../papers/context-compression/prompt-and-context-compression/2510.20535-arc-encoder-learning-compressed-text-representations-for-large-languag.md) (one encoder serves many decoders), [GMSA](../papers/context-compression/prompt-and-context-compression/2505.12215-gmsa-enhancing-context-compression-via-group-merging-and-layer-semanti.md) and
  [Semantic anchors](../papers/context-compression/prompt-and-context-compression/2510.08907-autoencoding-free-context-compression-for-llms-via-contextual-semantic.md).
* **Training-free selection:** [COMI](../papers/context-compression/prompt-and-context-compression/2602.01719-comi-coarse-to-fine-context-compression-via-marginal-information-gain.md) (marginal information gain) and [BEAVER](../papers/context-compression/prompt-and-context-compression/2603.19635-beaver-a-training-free-hierarchical-prompt-compression-method-via-stru.md)
  (page-level hierarchical selection, 26.4× lower latency at 128K).
* **Gist tokens as sparse attention:** [Simplified Sparse Attention](../papers/attention/sparse-attention/2604.20920-simplified-sparse-attention-via-gist-tokens.md) turns gist tokens into a
  log-linear decode hierarchy.
* **Optical compression:** DeepSeek-OCR (render text as image tokens, ~10× compression) and
  [Glyph](../papers/context-compression/prompt-and-context-compression/2510.17800-glyph-scaling-context-windows-via-visual-text-compression.md) (3–4×).

## Visual tokens ([folder](../papers/context-compression/visual-token-reduction/README.md))

Visual tokens are 70–95% redundant. Training-free pruning is mature:

| Method | Idea | Result |
| --- | --- | --- |
| [DART](../papers/context-compression/visual-token-reduction/2502.11494-stop-looking-for-important-tokens-in-multimodal-language-models-duplic.md) | prune by *duplication*, not importance | −88.9% vision tokens, 2.99× prefill; works with FlashAttention |
| [DivPrune](../papers/context-compression/visual-token-reduction/2503.02175-divprune-diversity-based-visual-token-pruning-for-large-multimodal-mod.md) (CVPR'25) | maximise diversity | SOTA over 16 datasets, no fine-tuning |
| [CDPruner](../papers/context-compression/visual-token-reduction/2506.10967-beyond-attention-or-similarity-maximizing-conditional-diversity-for-to.md) (NeurIPS'25) | instruction-conditioned DPP | −95% FLOPs, −78% latency, 94% accuracy on LLaVA |
| [MMTok](../papers/context-compression/visual-token-reduction/2508.18264-mmtok-multimodal-coverage-maximization-for-efficient-inference-of-vlms.md) (ICLR'26) | coverage of text + vision | 1.87× at 98.7%; 4 tokens keep 87.7% |
| [HoliTom](../papers/context-compression/visual-token-reduction/2505.21334-holitom-holistic-token-merging-for-fast-video-large-language-models.md) (video) | global temporal segmentation + merging | −90% video tokens |
| [STTM](../papers/context-compression/visual-token-reduction/2507.07990-multi-granular-spatio-temporal-token-merging-for-training-free-acceler.md) (video) | multi-granular spatio-temporal merging | 2× at −0.5%, 3× at −2% |
| [OmniSIFT](../papers/context-compression/visual-token-reduction/2602.04804-omnisift-modality-asymmetric-token-compression-for-efficient-omni-moda.md) (omni) | modality-asymmetric compression | beats full context at 25% of tokens |

Runtime note: prefer methods compatible with FlashAttention, i.e. no attention-map dependence (DART, DivPrune,
[Representation Shift](../papers/context-compression/visual-token-reduction/2508.00367-representation-shift-unifying-token-compression-with-flashattention.md)), and apply them before the LLM or in early layers.
