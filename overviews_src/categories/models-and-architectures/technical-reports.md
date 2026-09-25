**Verdict.** For a runtime builder, the technical reports are the **spec sheet** of what must be supported. The table
below distills the inference-relevant architecture choices of the major 2025–26 open(-weight) models, from each report's
abstract and paper. The trends:

* **MoE everywhere, with activation ratios of 2–6%.** Fine-grained experts + shared experts; LatentMoE (Nemotron 3,
  Kimi K3); zero-computation experts (LongCat).
* **Attention is hybrid or sparse by default.** Options:
  * SWA:global interleaves (gpt-oss, Gemma 3, MiMo-V2-Flash at 5:1 with 128-token windows, Step 3.5 at 3:1);
  * linear/SSM + attention (Qwen3-Next / Qwen3.8-Next with Gated DeltaNet 3:1, Kimi Linear/K3 with KDA, Nemotron-H/3
    with Mamba2, MiniMax-M1 with Lightning);
  * learned sparse attention (DeepSeek-V3.2 DSA, GLM-5 DSA, DeepSeek-V4 CSA+HCA, Qwen QSA).
* **MTP heads ship with the model** (DeepSeek-V3, GLM-4.5, MiMo, Step 3.5 with MTP-3, Nemotron 3 Super/Ultra, Ling 2.0).
  The runtime should use them for speculative decoding.
* **Residual-stream redesign reached production**: mHC (DeepSeek-V4), Attention Residuals (Kimi K3), Gated Residual
  (Qwen3.8-Next).
* **Memory tables off-accelerator**: 51B n-gram embeddings in Qwen3.8-Next, Engram (DeepSeek), LongCat-Flash-Lite. The
  runtime needs host-memory lookup with prefetch.
* **Low-precision native training/serving**: FP8 (DeepSeek-V3/V3.2, Ling), NVFP4 pretraining (Nemotron 3 Super/Ultra),
  MXFP4 MoE weights (gpt-oss).
* **Prefill-cheap designs for agentic, input-heavy workloads**: DeepSeek-V4.1-Flash's causal encoder–decoder activates
  8B parameters in prefill vs 16B in decode.

### Inference-relevant architecture table

| Model | Report | Params (total / active) | Attention | Other features |
| --- | --- | --- | --- | --- |
| DeepSeek-R1 | [[2501.12948]] (Nature) | 671B / 37B (V3 base) | MLA | MTP; FP8; pure-RL reasoning recipe |
| DeepSeek-V3.2 | [[2512.02556]] | 671B / 37B | MLA + **DSA** (lightning indexer, top-k) | Scaled RL; Speciale variant at IMO/IOI gold level |
| DeepSeek-V4 (Pro / Flash) | [[2606.19348]] | 1.6T / 49B; 284B / 13B | **CSA + HCA** hybrid compressed/sparse attention, 1M ctx | **mHC** residuals; Muon; 32T+ tokens |
| DeepSeek-V4.1-Flash | [[2609.19969]] | 552B backbone; 8B prefill / 16B decode | Causal encoder–decoder; aggressive KV compression | Multimodal; built for input-heavy agents |
| Qwen3 | [[2505.09388]] | 0.6B–32B dense; 30B-A3B, 235B-A22B MoE | GQA | Hybrid thinking/non-thinking modes with a thinking budget |
| Qwen3.8-Next (Flash) | [[2608.30320]] | 125B / 6B (+51B off-accelerator n-gram tables) | Gated DeltaNet : attention 3:1; later **QSA** sparse | 4-branch Gated Residual; ~1/9 the training FLOPs of the 397B-A17B predecessor |
| Kimi K2 | [[2507.20534]] | 1T / 32B | MLA | MuonClip (QK-clip); 15.5T tokens with zero loss spikes; agentic RL |
| Kimi K3 | [[2607.24653]] | 2.8T / 104B, 1M ctx | **Kimi Delta Attention** hybrid | Attention Residuals; Stable LatentMoE (16 of 896 experts) |
| GLM-4.5 / GLM-5 | [[2508.06471]] / [[2602.15763]] | 355B / 32B (4.5) | GQA (4.5); **DSA** (5) | MTP; asynchronous agent RL infrastructure (5) |
| MiniMax-M1 | [[2506.13585]] | 456B / 45.9B | **Lightning attention** hybrid (7 linear : 1 softmax) | CISPO RL; 1M context |
| MiniMax-M2 | [[2605.26494]] | 229.9B / 9.8B | Full attention | Agent-native RL system (Forge) |
| MiMo-V2-Flash | [[2601.02780]] | 309B / 15B | **SWA (128) : global 5:1** | MTP; 27T tokens; multi-teacher OPD |
| Step 3.5 Flash | [[2602.10604]] | 196B / 11B | **SWA : full 3:1** | MTP-3 |
| LongCat-Flash | [[2509.01322]] | 560B / 18.6–31.3B (dynamic) | MLA | **Zero-computation experts**; shortcut-connected MoE for overlap |
| Ling 2.0 / Ring | [[2510.22115]] | 16B–1T, high sparsity | GQA | MTP; FP8 training; efficiency-leverage laws |
| gpt-oss-120b / 20b | [[2508.10925]] | 117B / 5.1B; 21B / 3.6B | Alternating **banded (128) / dense**, attention sinks | **MXFP4** MoE weights; harmony chat format |
| Gemma 3 | [[2503.19786]] | 1B–27B dense | **Local (1024) : global 5:1** | 128K context; QAT checkpoints |
| Gemma 4 | [[2607.02770]] | 2.3B–31B, dense + MoE | Efficiency-oriented design | Encoder-free 12B (raw audio/image patches); thinking mode |
| Nemotron-H / Nano 2 / Nemotron 3 | [[2504.03624]] / [[2508.14444]] / [[2512.20856]] | up to Ultra (MoE) | **Mamba-2 hybrid** | LatentMoE + MTP + NVFP4 training (Super/Ultra); 1M context; reasoning budget control |
| Falcon-H1 | [[2507.22448]] | 0.5B–34B | **Parallel attention + Mamba heads** | Hybrid-head design |
| LFM2 | [[2511.23404]] | 350M–8.3B (MoE 8.3B-A1.5B) | Gated short conv + few GQA blocks | 2× CPU prefill/decode vs peers |
| MiniCPM4 | [[2506.07900]] | 0.5B / 8B | InfLLM v2 trainable sparse attention | End-device focus; speculative decoding + quantization |
| OLMo 2 / Olmo 3 | [[2501.00656]] / [[2512.13961]] | 1B–32B dense | GQA-family (details in reports) | Fully open data, code and checkpoints |

### Hand ranking (most useful reports to read in full)

| # | Report | Why read it |
| ---: | --- | --- |
| 1 | [[2501.12948\|DeepSeek-R1]] | The reasoning-RL recipe everyone copied (GRPO, cold start, distillation to small models) |
| 2 | [[2505.09388\|Qwen3]] | Hybrid thinking modes, thinking budget, strong-to-weak distillation, full family |
| 3 | [[2606.19348\|DeepSeek-V4]] | 1M-context CSA/HCA attention + mHC + Muon at 1.6T |
| 4 | [[2512.02556\|DeepSeek-V3.2]] | DSA sparse attention in production + large-scale agentic RL |
| 5 | [[2507.20534\|Kimi K2]] | MuonClip at 1T scale; agentic data synthesis |
| 6 | [[2601.02780\|MiMo-V2-Flash]] | Aggressive SWA hybrid (128-token windows) + MTP + multi-teacher OPD |
| 7 | [[2512.20856\|Nemotron 3]] | Hybrid Mamba MoE with LatentMoE, NVFP4 pretraining, MTP, open data |
| 8 | [[2608.30320\|Qwen3.8-Next architecture]] | Detailed ablations: GDN hybrid, QSA, gated residual, off-accelerator n-gram tables |
| 9 | [[2508.10925\|gpt-oss]] | MXFP4 MoE, sinks, banded attention; reference open reasoning models |
| 10 | [[2509.01322\|LongCat-Flash]] | Dynamic-compute MoE (zero experts) and shortcut MoE for overlap |

**Also useful.**
* VLM/omni reports: [[2502.13923|Qwen2.5-VL]], [[2511.21631|Qwen3-VL]], [[2504.10479|InternVL3]],
  [[2508.18265|InternVL3.5]], [[2509.17765|Qwen3-Omni]], [[2602.02276|Kimi K2.5]], [[2507.01006|GLM-4.1V]],
  [[2505.07062|Seed1.5-VL]].
* Reasoning reports: [[2501.12599|Kimi k1.5]], [[2504.13914|Seed1.5-Thinking]], [[2506.10910|Magistral]],
  [[2504.21318|Phi-4-reasoning]], [[2505.07608|MiMo]], [[2505.00949|Llama-Nemotron]].
* Closed models: [[2507.06261|Gemini 2.5]], [[2506.12103|Amazon Nova]], [[2507.13575|Apple Foundation Models 2025]].
* Long context: [[2501.15383|Qwen2.5-1M]], [[2501.08313|MiniMax-01]].
* Embeddings: [[2506.05176|Qwen3 Embedding]], [[2509.20354|EmbeddingGemma]].

**Runtime feature matrix to implement:** MLA + DSA/CSA-style sparse attention with an indexer; SWA:global interleaves
with small windows; GDN/KDA/Mamba2/Lightning recurrent layers with state caching; attention sinks; MTP-head
speculation; LatentMoE and zero-compute experts; mHC/AttnRes residuals; MXFP4/NVFP4/FP8 weights; host-memory n-gram
tables; thinking-budget control; 1M-token context.
