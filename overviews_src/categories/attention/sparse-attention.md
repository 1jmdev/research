**Verdict.** Sparse attention went from research to **production** in 2025–26.

* **NSA** (DeepSeek) established *natively trainable*, hardware-aligned sparse attention: compressed + selected +
  sliding branches.
* **DSA** in DeepSeek-V3.2 ([[2512.02556]]) shipped a token-level "lightning indexer" + top-k.
* MiniMax, Kimi (MoBA) and others followed with block-level variants; MSA converts pretrained GQA checkpoints
  near-losslessly.

There are two families.

1. **Trainable / native** (NSA, MoBA, DSA, MSA, InfLLM-V2, HiLS, SeerAttention-R). These match or beat full attention
   and cut both prefill and decode cost. The new bottleneck is the **indexer itself**, which is O(L²). IndexCache
   (cross-layer index reuse, −75% indexer cost) and HISA (hierarchical indexing) attack it.
2. **Training-free** for existing models (XAttention, FlexPrefill, MInference lineage, Twilight, SpargeAttention).
   Mainly for **prefill** acceleration. [[2504.17768|The Sparse Frontier]] (largest study) finds:
   * larger sparse models beat smaller dense ones at equal cost;
   * decoding tolerates more sparsity than prefill;
   * there is no universal method.

### Hand ranking

| # | Paper | Type | Key idea | Headline |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.11089\|NSA]] (DeepSeek, ACL'25 best paper) | Native, trainable | Three branches: **compressed tokens (global) + selected blocks (top-n) + sliding window**, with a gate; arithmetic-intensity-balanced kernel for GQA | Matches or beats full attention at 27B; large prefill, decode and backward speedups at 64K |
| 2 | [[2512.02556\|DSA (DeepSeek-V3.2)]] | Native, token-level | FP8 **lightning indexer** scores all tokens; attention on the top-2048 per query; continued pretraining from V3.1 | Near-flat cost with context length in production; reference design |
| 3 | [[2606.13392\|MiniMax Sparse Attention (MSA)]] | Native, block | GQA-group-wise **Index Branch** (KL-aligned) selects top-k blocks; **near-lossless conversion from GQA checkpoints** | Scales to a 109B MoE; simple, deployable |
| 4 | [[2502.13189\|MoBA]] (Kimi, NeurIPS'25) | Native, block | MoE-style gating over KV blocks with a "less structure" principle; can switch between full and sparse | Deployed for Kimi long context |
| 5 | [[2603.12201\|IndexCache]] | Indexer optimization | Top-k selections are similar across layers, so a few Full layers run indexers and the rest reuse them | −75% indexer cost with no loss (DSA/GLM) |
| 6 | [[2503.16428\|XAttention]] (ICML'25) | Training-free prefill | **Antidiagonal sums** as a cheap block-importance proxy | Up to 13.5× attention speedup, near full accuracy (text + video) |
| 7 | [[2509.24663\|InfLLM-V2]] | Trainable, switchable | Dense-sparse switchable attention reusing dense parameters; fits the pretrain-short, finetune-long workflow | Seamless short-to-long adaptation (MiniCPM4) |
| 8 | [[2506.08889\|SeerAttention-R]] | Plug-in gate for decode | Self-distilled gating for **reasoning decode**; TileLang kernel | 0.4B tokens of training; near-lossless AIME at a 4K budget; up to 9× over FA3 |
| 9 | [[2502.20766\|FlexPrefill]] (ICLR'25 oral) | Training-free prefill | Per-head, per-input pattern choice (JS divergence) + cumulative-attention budget | Adaptive sparse prefill |
| 10 | [[2502.14866\|LServe]] (MLSys'25) | System | Unified block-sparse prefill + decode, streaming heads + hierarchical page selection | Multiplicative speedups on long-sequence serving |
| 11 | [[2603.28458\|HISA]] | Indexer optimization | Block-level coarse filter, then token-level indexer only inside candidates | Removes the O(L²) scan of DSA-style indexers |
| 12 | [[2502.02770\|Twilight]] (NeurIPS'25 spotlight) | Adaptive budget | **Top-p** instead of top-k budgeting for any sparse method | Up to 98% pruning, 15.4× attention speedup |
| 13 | [[2607.02980\|HiLS]] | Native, chunk | Chunk retrieval learned end-to-end through attention fusion | Toward infinite context with good extrapolation |
| 14 | [[2504.17768\|The Sparse Frontier]] (ACL'26) | Study | Six training-free methods, up to 128K and 95% sparsity | Actionable trade-offs; read before choosing |
| 15 | [[2502.18137\|SpargeAttention]] (ICML'25) + [[2602.13515\|SpargeAttention2]] | Universal, training-free / trainable | Block-sparse + online softmax-aware filter on top of SageAttention quantization | Works for LLMs, image and video models |

Also useful:
* [[2603.23516|Memory Sparse Attention]]: end-to-end memory to 100M tokens.
* [[2509.24006|SLA]]: sparse + linear attention for DiTs.
* [[2505.13389|VSA]]: video.
* [[2508.18224|FSA]]: an alternative NSA kernel.
* [[2511.11571|Optimizing MoBA]].
* [[2510.13602|NOSA]]: offloadable native sparse attention.
* [[2603.02227]]: random gates are hard to beat. A sobering control.

**Runtime recommendations.**
1. **Model-native sparse attention (NSA/DSA/MSA/MoBA).** Implement **block top-k gather attention** (FlashMLA/FlashInfer
   sparse kernels, TileLang) plus the indexer kernel (FP8 dot products + top-k). Reuse indices across layers when the
   model allows it (IndexCache).
2. **Dense models.** Offer training-free **sparse prefill** (XAttention / FlexPrefill) for >32K prompts, and keep decode
   dense or top-p sparse (Twilight). Always keep sinks and a local window.
3. **Pair with KV offloading.** Sparse decode lets the full KV live in CPU/CXL memory; see
   [`kv-cache/offloading`](../../kv-cache/offloading-and-hierarchical-storage/README.md).
