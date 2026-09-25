**Verdict.** With ~500 papers, this is the busiest area of 2025–26. The practical consensus after DeepSeek-R1:

**1. The algorithm is GRPO-family with specific fixes.**
* [[2503.14476|DAPO]] (ByteDance): clip-higher, dynamic sampling (drop all-correct and all-wrong groups), token-level
  loss, overlong shaping. 50 AIME'24 on Qwen2.5-32B base.
* [[2503.20783|Dr. GRPO]]: remove length and std normalization biases.
* [[2507.18071|GSPO]] (Qwen3): **sequence-level** importance ratios, which stabilize MoE RL.
* Also: CISPO (MiniMax-M1), [[2511.20347|SAPO]], [[2507.20673|GMPO]], [[2501.03262|REINFORCE++]],
  [[2504.05118|VAPO]] (value-based).
* [[2510.13786|The Art of Scaling RL Compute]] (ScaleRL, Meta): a 400K GPU-hour study. RL follows **sigmoidal
  compute–performance curves** that can be fitted early. Recipe choices change the *efficiency*; only some change the
  *ceiling*. Validated on a single 100K GPU-h run.

**2. What RL actually does (keep expectations calibrated).**
* [[2504.13837|Does RL really incentivize reasoning beyond the base?]] (NeurIPS'25 oral): RLVR improves pass@1 but the
  base model wins at large k. RL *sharpens* the distribution. Prolonged, diverse RL ([[2505.24864|ProRL]]) and
  [[2508.10751|pass@k training]] push the boundary somewhat.
* Only **high-entropy "fork" tokens** matter: training on the top 20% entropy tokens matches or beats full RL
  ([[2506.01939|Beyond the 80/20 rule]]).
* Entropy collapse sets a predictable ceiling ([[2505.22617|The Entropy Mechanism]]; Clip-Cov / KL-Cov fix it).
* RL updates **small subnetworks** ([[2505.11711]]).
* **1–2 training examples** can give most of the math gain on Qwen ([[2504.20571|1-shot RLVR]]). Spurious and random
  rewards also help Qwen ([[2506.10947|Spurious Rewards]]): be careful about data contamination
  ([[2507.10532]]) and model-family effects.

**3. Systems are the bottleneck.** Rollout generation is 70–90% of RL time.
* **Asynchronous RL** with staleness control: [[2505.24298|AReaL]] (2.77×), [[2507.01663|AsyncFlow]],
  [[2510.12633|Laminar]], [[2505.24034|LlamaRL]], [[2504.15930|StreamRL]] (disaggregated generation);
  [[2510.01161|stale-data limits]]; [[2607.18722|staleness-adaptive trust regions]].
* **Long-tail rollouts**: [[2509.21009|RollPacker]], [[2511.14617|Seer]] (online context learning for synchronous RL).
* **Speculative decoding inside rollouts**: [[2604.26779]], [[2606.18967|EfficientRollout]], [[2609.07108]] (online
  draft co-training).
* **Training–inference mismatch** (vLLM vs FSDP numerics) destabilizes RL. Fixes:
  * FP16 instead of BF16 ([[2510.26788]]);
  * deterministic TP-invariant kernels ([[2511.17826]]);
  * MoE routing replay ([[2510.11370|R3]]);
  * LR scheduling ([[2602.01826]]); diagnosis in [[2605.14220]].
* Quantized RL rollouts: [[2510.11696|QeRL]], [[2604.07853|QaRL]].

**4. Beyond verifiable math and code.**
* Rubrics as rewards ([[2507.17746|RaR]]).
* Verifier-free probability rewards ([[2506.18254|RLPR]]).
* Generative reward models ([[2505.02387|RM-R1]], [[2507.01352|Skywork-Reward-V2]]).
* Self-rewarding / unsupervised: [[2504.16084|TTRL]], [[2505.19590|Intuitor]]; limits in [[2603.08660]].
* Self-play: [[2505.03335|Absolute Zero]], [[2506.24119|SPIRAL]].
* Agentic RL: [[2503.09516|Search-R1]], [[2504.11536|ReTool]], [[2509.02479|SimpleTIR]], [[2505.10978|GiGPO]],
  [[2507.19849|ARPO]], [[2508.20722|rStar2-Agent]], [[2502.18449|SWE-RL]].
* Self-distillation as RL: [[2601.20802|SDPO]].

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2503.14476\|DAPO]] (NeurIPS'25) | Algorithm + open system | Clip-higher, dynamic sampling, token-level loss, overlong reward shaping (verl) | 50 on AIME'24 from Qwen2.5-32B base in half DeepSeek's steps; fully open |
| 2 | [[2510.13786\|The Art of Scaling RL Compute (ScaleRL)]] | RL scaling | Sigmoidal compute–performance fits; ablate what changes the asymptote vs efficiency | Predicts a 100K GPU-h run from early points; best-practice recipe |
| 3 | [[2507.18071\|GSPO]] (Qwen) | Algorithm | Sequence-level ratio, clipping and optimization | Stabilizes MoE RL; used for Qwen3 |
| 4 | [[2504.13837\|Does RL really incentivize reasoning?]] | Science | Pass@k at large k: base ≥ RLVR model | RLVR sharpens rather than expands; sets expectations |
| 5 | [[2506.01939\|High-entropy minority tokens]] | Science + method | Policy-gradient only on the top-20% entropy "fork" tokens | Matches or beats full-token RL, larger gains at 32B |
| 6 | [[2505.22617\|The Entropy Mechanism of RL]] | Science + method | Performance ≈ −a·exp(entropy)+b; entropy change driven by covariance | Clip-Cov / KL-Cov prevent collapse and raise the ceiling |
| 7 | [[2503.20783\|Understanding R1-Zero-like training / Dr. GRPO]] | Algorithm | Remove GRPO length and std normalization biases; base-model template effects | 43.3% AIME'24 at 7B; shorter wrong answers |
| 8 | [[2505.24298\|AReaL]] | System | Fully asynchronous rollouts and training with staleness-aware PPO | Up to 2.77× training speedup at equal or better accuracy |
| 9 | [[2510.26788\|Defeating training–inference mismatch via FP16]] | Stability | BF16 rounding causes rollout/trainer divergence; FP16 removes it | Stable RL without importance-sampling hacks |
| 10 | [[2504.20571\|1-shot RLVR]] (NeurIPS'25) | Data efficiency | RL on a single example | MATH500 36% → 73.6% on Qwen2.5-Math-1.5B. Beware family effects |
| 11 | [[2505.24864\|ProRL]] (NVIDIA) | Long RL | KL control + reference resets + diverse tasks over 2K+ steps | Expands the reasoning boundary on tasks where the base fails |
| 12 | [[2507.17746\|Rubrics as Rewards]] | Reward design | Checklist rubrics scored by a judge as dense structured rewards | RL beyond verifiable domains (medicine, science) |
| 13 | [[2503.09516\|Search-R1]] | Agentic RL | Interleaved search calls with retrieved-token masking and outcome reward | +41% over RAG baselines (Qwen2.5-7B) |
| 14 | [[2507.19457\|GEPA]] | Alternative to RL | Reflective prompt evolution with Pareto selection | Beats GRPO with up to 35× fewer rollouts on several tasks |

**Also useful.**
* Off-policy guidance and SFT+RL: [[2504.14945|LUFFY]], [[2508.11408]], [[2506.07527]], [[2509.04419]],
  [[2508.05629|DFT]] (one-line SFT fix).
* Reward models and judges: [[2507.08794|One token to fool LLM-as-a-judge]].
* Rollout efficiency: [[2504.13818]] (down-sampling), [[2506.02177]] (selective rollouts), [[2510.01180|BroRL]].
* Recipes: [[2503.24290|Open-Reasoner-Zero]], [[2503.18892|SimpleRL-Zoo]], [[2505.16400|AceReason-Nemotron]],
  [[2512.13607|Nemotron-Cascade]], [[2503.10460|Light-R1]].
* Agent frameworks: [[2508.03680|Agent Lightning]], [[2509.01055|VerlTool]], [[2509.08755|AgentGym-RL]],
  [[2510.04206|AgentRL]], [[2603.18815|ProRL Agent]] (rollout-as-a-service).
* Evolution strategies: [[2509.24372]].
* PEFT for RLVR: [[2512.23165]].
* Surveys: [[2509.08827]], [[2509.02547]].

**Recommendation.**
* *Model builders:* strong mid-trained base → cold-start SFT on long CoT → DAPO/GSPO-style RL with clip-higher,
  dynamic sampling, entropy management and no KL. Fit a ScaleRL sigmoid early to decide whether to keep going.
  Evaluate at pass@k too.
* *Runtime builders (rollout engines):*
  * weight hot-swap from the trainer;
  * deterministic / TP-invariant mode, and optionally FP16;
  * routing capture for MoE (R3);
  * async generation with staleness tags;
  * speculative decoding with online-updated drafters;
  * long-tail rollout packing;
  * multi-turn tool environments with KV kept across tool calls.
