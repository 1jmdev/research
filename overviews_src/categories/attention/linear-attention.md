**Verdict.** Linear attention won its place **inside hybrids**. The lineage runs:

**DeltaNet → Gated DeltaNet (GDN) → Kimi Delta Attention (KDA) → Gated DeltaNet-2.** The **delta rule** with gating is
the consensus recurrent primitive. It is used in Qwen3-Next / Qwen3.5+, Kimi Linear, MiniMax (lightning attention in
MiniMax-01/M1) and many 2026 hybrids.

Kimi Linear (KDA : full MLA at 3:1) is the first to **beat full attention in fair comparisons**, including RL scaling,
while cutting the KV cache by up to 75% and giving up to 6× decode throughput at 1M context.
[[2507.06457|The Systematic Analysis of Hybrid Linear Attention]] (72 open models) gives the recipe:
* use gating plus controlled forgetting;
* a **3:1–6:1 linear:full ratio** matches Transformer recall;
* standalone linear quality does **not** predict hybrid quality.

Test-time training (TTT/LaCT/In-Place TTT) is the same family. [[2602.21204]] shows that TTT with KV binding **is**
learned linear attention.

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [[2510.26692\|Kimi Linear / KDA]] | GDN with **channel-wise (fine-grained) gating** + a specialized DPLR chunkwise kernel; 3:1 KDA:MLA hybrid (48B-A3B) | Beats full MLA on short, long and RL; −75% KV; up to 6× decode at 1M. Open kernels in FLA |
| 2 | [[2507.06457\|Systematic Analysis of Hybrid Linear Attention]] | 72 models, 6 linear variants × 5 ratios | **Selective gating + hierarchical recurrence + forgetting**; 3:1–6:1 ratio. A design guide |
| 3 | [[2605.22791\|Gated DeltaNet-2]] | Decouples the **erase** (key-side) and **write** (value-side) gates that GDN/KDA tie to one scalar β | Generalizes GDN and KDA; better recall |
| 4 | [[2505.23884\|LaCT: Test-Time Training Done Right]] | **Large-chunk** TTT (2K–1M tokens per update) + window attention; high FLOPs use | Scales TTT to 14B video and language; state up to 40% of params |
| 5 | [[2604.06169\|In-Place TTT]] (ICLR'26) | Use each MLP's **final projection as fast weights**, adapted in place; drop-in for existing LLMs | TTT without re-architecting |
| 6 | [[2506.04761\|Log-Linear Attention]] | A **logarithmically growing set of states** (Fenwick-tree style); applied to Mamba-2 and GDN | Between linear and softmax in expressivity at O(T log T) |
| 7 | [[2602.21204\|TTT with KV binding = linear attention]] (ICML'26) | Reduces TTT variants to learned linear attention operators | Simplifications and fully parallel forms |
| 8 | [[2501.12352\|Test-time regression]] | Unifies attention, SSMs, fast weights and online learners as regression over keys and values | The best conceptual map of the design space |
| 9 | [[2601.07832\|MHLA]] | Token-level multi-head linear attention to avoid global context collapse | Recovers softmax-like expressivity at linear cost |
| 10 | [[2512.12602\|EFLA]] | Exact closed-form flow instead of the Euler step of the delta rule | Stability under corrupted/high-energy input, no extra params |
| 11 | [[2609.14320\|SpectralShift]] | Spectral reparameterization of GDN decay for **context extension** | Principled long-context CPT for linear layers |
| 12 | [[2602.10743\|Kalman Linear Attention]] (ICML'26) / [[2609.07816\|Kalman Delta Networks]] | Exact Bayesian filtering as an associative scan; uncertainty-aware writes | Better state tracking |
| 13 | [[2608.12149\|Massive activations in hybrid LA]] | **Pre-attention spikes** before full-attention layers in hybrids (1.2B–397B) | Matters for quantizing hybrids |

Also useful:
* [[2608.28444|Sliding-window beats linear attention]]: a contrarian control. Always compare against SWA.
* [[2605.19049|KVBuffer]]: IO-aware serving for linear attention.
* [[2608.15533|DeltaLog]]: deferred state materialization for decode.
* [[2608.20961|TreeWY]]: speculative verification for GDN hybrids.
* [[2605.21325]] / [[2606.06034]]: fast triangular inversion in chunkwise kernels.

**Runtime checklist for hybrid models (Qwen3-Next style GDN, Kimi Linear KDA, MiniMax).**
1. **Kernels.** Chunkwise-parallel prefill (WY/UT transform) + a recurrent decode kernel with an in-register state
   update. The FLA library has reference Triton kernels.
2. **Memory manager.** Treat the **per-request recurrent state** as a fixed-size "KV page". Checkpoint it for prefix
   caching; see [`kv-cache/prefix-caching`](../../kv-cache/prefix-caching-and-reuse/README.md).
3. **Speculative decoding.** Needs state rollback: snapshot the state per draft position or use tree-WY verification
   (TreeWY).
4. **Quantization.** GDN layers survive NVFP4 W4A4 ([[2609.04098]]). Watch massive activations before the full-attention
   layers.
