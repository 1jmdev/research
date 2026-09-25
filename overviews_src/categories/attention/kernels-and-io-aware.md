**Verdict.** The attention-kernel landscape in 2026:

* **FlashAttention-4** is the Blackwell dense baseline.
* **FlashInfer** is the serving engine: paged/ragged KV, JIT variants, CUDA-graph-friendly scheduling.
* **FlashMLA**-family kernels serve MLA.
* **SageAttention3** covers FP4 and 8-bit attention.

On Blackwell, tensor cores got 2× faster but **shared memory and the exponential unit did not**. Kernels are now bound
by softmax and SMEM, hence FA4's software exp2 and conditional rescaling, and FP4 attention's softmax bottleneck
([[2609.04105]], [[2609.09721|EFQ-Softmax]]). For serving, the big wins come from **shared-prefix-aware decode**
(PAT, CoDec, TyphoonMLA, DualKV for RL) and from portable Triton/Pallas kernels (AMD, TPU).

### Hand ranking

| # | Paper | Scope | Key idea | Headline |
| ---: | --- | --- | --- | --- |
| 1 | [[2603.05451\|FlashAttention-4]] | Dense, Blackwell | Fully async MMA pipelines with larger tiles; **software-emulated exp** + conditional softmax rescaling; 2-CTA MMA for backward; written in CuTe-DSL (Python) | Up to 1.3× over cuDNN 9.13 and 2.7× over Triton on B200; ~71% utilization |
| 2 | [[2501.01005\|FlashInfer]] (MLSys'25 best paper) | Serving engine | Block-sparse and composable KV formats, **JIT attention templates**, load-balanced scheduler compatible with CUDA graphs | Powers vLLM, SGLang and MLC. The reference for serving attention |
| 3 | [[2505.11594\|SageAttention3]] (NeurIPS'25) | FP4 inference + INT8 training | **Microscaling FP4 attention** (1038 TOPS on RTX 5090, 5× FA) + trainable 8-bit attention (SageBwd) | Plug-and-play FP4 attention |
| 4 | [[2603.02170\|SageBwd]] | Low-bit training | INT8 in 6 of 7 attention matmuls; needs **QK-norm**; the error lives in the dS gradient | Matches full-precision pretraining at moderate tokens per step |
| 5 | [[2605.23081\|ThriftAttention]] | FP4 long context | Keep a few important Q-K block pairs in FP16, the rest FP4 | Near-FP16 long-context quality at FP4 speed |
| 6 | [[2511.22333\|PAT]] (ASPLOS'26) | Decode, shared prefix | **Pack queries by shared prefix** and a multi-tile kernel (pack-forward-merge) | Cuts redundant prefix KV reads in serving |
| 7 | [[2503.14376\|Tiled Flash Linear Attention]] (NeurIPS'25) | Linear RNN kernels | Two-level tiling within chunks → fewer materialized states | Faster than FA and FLA for mLSTM/linear RNNs |
| 8 | [[2506.01969\|FlashMLA-ETAP]] | MLA decode on H20 | Transposed pipeline aligning KV length with the WGMMA M-dimension | 2.78× over FlashMLA at 64K |
| 9 | [[2605.15422\|DualKV]] | RL training | Process the **shared prompt once** across N rollouts in fused forward and backward | Large savings for GRPO/DAPO with long prompts |
| 10 | [[2511.11581\|Anatomy of a Triton attention kernel]] (IBM/vLLM) | Portability | State-of-the-art paged attention in **pure Triton** on NVIDIA and AMD | In vLLM; shows portability is achievable |
| 11 | [[2604.15464\|Ragged Paged Attention]] (Google) | TPU | Pallas/Mosaic ragged paged attention with fused KV update | TPU serving reference |
| 12 | [[2601.21824\|DASH]] | Deterministic training | DAG scheduling of the deterministic backward | Recovers most of the ~38% determinism penalty |
| 13 | [[2604.22312\|Guess-Verify-Refine]] | Sparse decode top-k | Exact top-k reusing the previous step's selection (temporal correlation) on Blackwell | Speeds up the DSA indexer's top-k stage |
| 14 | [[2509.21081\|TyphoonMLA]] | MLA shared prefix | Mixed naive/absorb MLA: naive for the shared prefix, absorb for the rest | Faster MLA with prefix sharing |
| 15 | [[2505.12044\|FlashBias]] (NeurIPS'25) | Attention with bias | Low-rank bias decomposition keeps fusion | Fast biased attention (ALiBi-like, science models) |

Also relevant:
* Compiler and DSL approaches: [[2511.02043|Flashlight]], [[2609.13612|AttnFuse]], [[2506.07311]] (PagedAttention in
  FlexAttention).
* LLM-generated kernels: [[2506.12355|QiMeng-Attention]], [[2605.05023|CuBridge]].
* NPUs: [[2607.04302|HiFA4 (Ascend)]], [[2609.21264|AMD XDNA]].

**Runtime recommendations.**
1. Adopt FlashInfer (or FA3/FA4 + FlashMLA) as the kernel layer. Your runtime's job is paging, scheduling and variant
   selection.
2. Add **cascade/shared-prefix decode** (FlashInfer cascade, PAT) for agents and RL.
3. On Blackwell, prefer FP8 attention by default. Use FP4 attention (SageAttention3) only with a mixed-precision guard
   such as ThriftAttention for long context.
