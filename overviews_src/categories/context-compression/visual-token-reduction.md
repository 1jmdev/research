**Verdict.** VLMs spend most prefill on visual tokens: 576–2,880 per image and far more for video. With 300+ papers, the
field has converged on some lessons.

1. **Benchmarks flatter pruning methods.** [[2510.07143|Are we using the right benchmark?]] finds plain **image
   downsampling beats many "advanced" token-compression methods** on standard benchmarks, because most samples don't
   need fine detail. Use downsampling as the baseline and VTC-Bench-style hard subsets. [[2502.11501|Are we solving the
   right problem?]] makes the same point.
2. **Duplication and diversity beat "importance".** Attention-score importance (FastV lineage) is biased and incompatible
   with FlashAttention.
   * [[2502.11494|DART]]: prune duplicates of pivot tokens; keeps 11% of tokens, 2× end-to-end.
   * [[2503.02175|DivPrune]]: max-min diversity.
   * [[2506.10967|CDPruner]]: instruction-conditional DPP; −95% FLOPs at 94% accuracy.
   * [[2508.18264|MMTok]]: multimodal coverage.
   These are training-free, kernel-compatible, and select *before* the LLM.
3. **Video is where the big wins are.**
   * Temporal redundancy: [[2504.17343|TimeChat-Online]] (80% of streaming-video tokens are redundant; differential token
     drop).
   * Holistic merging: [[2505.21334|HoliTom]] (6.9% of FLOPs at 99.1% performance; 2.28× TTFT).
   * [[2507.07990|STTM]], [[2503.11187|FastVID]], [[2501.01986|FrameFusion]].
   * Streaming: [[2510.09608|StreamingVLM]] (sink + recent-vision + longer-text windows; 8 FPS on one H100) and
     [[2512.00891|STC]].
4. **Let the model ask for resolution.** [[2507.13348|VisionThink]] starts low-resolution and uses RL to request
   high-resolution only when needed. Crop retrieval: [[2603.16932]], [[2512.03794|AdaptVision]].
5. **Architectural reduction.** [[2501.03895|LLaVA-Mini]] (one vision token after modality pre-fusion; −77% FLOPs, 40 ms
   responses), [[2504.00502|ShortV]] (freeze visual tokens in the ~60% of layers where they don't help),
   [[2503.04130|STORM]] (Mamba temporal encoder before the LLM).

Omni-modal: audio-guided video pruning ([[2511.14582|OmniZip]], [[2602.04804|OmniSIFT]]). Text-as-image compression (the
reverse direction) is in [`prompt-and-context-compression`](../prompt-and-context-compression/README.md)
(DeepSeek-OCR, Glyph).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2510.07143\|Are we using the right benchmark?]] | Evaluation | Downsampling as a discriminator; VTC-Bench denoises existing benchmarks | Downsampling beats many methods. **Required baseline** |
| 2 | [[2502.11494\|DART: duplication matters more]] | Training-free pruning | Keep pivot tokens and drop tokens duplicating them; FlashAttention-compatible | 88.9% pruned at comparable accuracy; 1.99× total, 2.99× prefill |
| 3 | [[2505.21334\|HoliTom]] (NeurIPS'25) | Video merging | Outer-LLM global temporal segmentation + spatio-temporal merging + inner-LLM merging | 6.9% of FLOPs at 99.1% performance; 2.28× TTFT |
| 4 | [[2510.09608\|StreamingVLM]] (MIT) | Streaming video | Training aligned with streaming KV (sinks + recent vision + long text window) | Stable real-time understanding of infinite streams at 8 FPS on one H100 |
| 5 | [[2506.10967\|CDPruner]] | Training-free pruning | Instruction-conditioned DPP for conditional diversity | −95% FLOPs, −78% CUDA latency at 94% accuracy (LLaVA) |
| 6 | [[2507.13348\|VisionThink]] | Adaptive resolution | RL-trained decision to request high-resolution images | Most queries answered at low resolution with little loss on OCR-heavy tasks |
| 7 | [[2501.03895\|LLaVA-Mini]] (ICLR'25) | Architecture | Modality pre-fusion into text, then one vision token per image/frame | Beats LLaVA-1.5 with 1 token instead of 576; 10K+ frames on 24 GB |
| 8 | [[2504.17343\|TimeChat-Online]] | Streaming | Differential token drop based on frame-to-frame change | ~80% of visual tokens removed with strong streaming results |
| 9 | [[2503.02175\|DivPrune]] (CVPR'25) | Pruning | Max-min diversity subset selection | State of the art across 16 image/video datasets; lower latency and memory |
| 10 | [[2504.00502\|ShortV]] | Layer-wise | Freeze visual-token updates in ineffective layers (layer contribution metric) | −50% FLOPs on LLaVA-NeXT-13B |
| 11 | [[2508.18264\|MMTok]] | Pruning | Multimodal max-coverage selection using both text and vision | 98.7% performance with large speedup; 87.7% with only 4 tokens |
| 12 | [[2503.04130\|STORM]] | Long video | Mamba temporal encoder injects dynamics, then aggressive token reduction | Better long-video reasoning at lower token cost |

**Also useful.**
* Training-free video: [[2507.07990|STTM]], [[2503.11187|FastVID]], [[2505.14454|Video Compression Commander]],
  [[2605.30010|EarlyTom]], [[2506.21862|LLaVA-Scissor]].
* Distillation for compressed tokens: [[2510.00515|EPIC]].
* Holistic context retention: [[2510.02912]].
* Hybrid: [[2512.08240]].
* Codec-aligned encoders: [[2602.08683|OneVision-Encoder]].
* Survey: [[2507.20198]].

**Runtime checklist.**
* Pre-LLM token selection hooks (duplication/diversity based, instruction-conditioned).
* Image-level prefix caching by content hash.
* Streaming KV policy for video (sinks + recent frames + text).
* Dynamic-resolution requests in the loop (VisionThink-style tool call).
* Report against a downsampling baseline.
