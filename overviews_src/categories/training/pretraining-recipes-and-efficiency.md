**Verdict.** The pretraining recipe of 2025–26 is **"pretrain → mid-train (reasoning/agentic data, long context) → RL"**
designed as one pipeline. The most consequential findings:

1. **Mid-training decides RL headroom.**
   * [[2506.20512|OctoThinker]]: Llama-family models become Qwen-level RL learners after math-heavy mid-training
     (MegaMath-Web-Pro-Max, QA-style long CoT, then instruction data). Releases a 70B-token corpus.
   * [[2510.03264|Front-loading reasoning]] (NVIDIA): reasoning data in *pretraining* gives durable gains that SFT can't
     reproduce, while naively scaling SFT data washes them out.
   * [[2512.07783|Pre-, mid-training and RL interplay]] (CMU): controlled study. RL only extends capabilities the base
     can almost do; mid-training is central.
   * [[2509.13310|Agentic continual pretraining]] (AgentFounder): the same idea for agents.
   * The [[2510.15020|coverage principle]] gives theory: pretraining must *cover* good behaviours for post-training to
     find them.
2. **Don't overtrain the base you will fine-tune.** [[2503.19206|Overtrained LMs are harder to fine-tune]] (ICML'25):
   past a token budget, more pretraining *hurts* post-fine-tuning performance ("catastrophic overtraining") because
   parameter sensitivity grows.
3. **Cheap, drop-in training accelerators:**
   * [[2501.01956|Metadata conditioning (MeCo)]]: prepend source URLs for most of training, then cool down without them;
     equal quality with 33% less data;
   * [[2503.15450|SkyLadder]]: short-to-long context-window schedule; +3.7% and 22% faster;
   * [[2605.06546|Token Superposition]]: up to 2.5× less pretraining time at equal loss (10B-A1B);
   * [[2507.17634|WSM]]: checkpoint merging replaces LR decay; +3.5 MATH, +5.5 MMLU-Pro over WSD;
   * [[2502.10940|CoLA]]: low-rank activations, 2× less compute;
   * [[2603.05369|progressive residual warmup]].
4. **New objectives beyond NTP:**
   * [[2506.08007|Reinforcement Pre-Training]] (Microsoft): next-token prediction as an RL task with verifiable reward,
     scaling with compute;
   * [[2502.08524|CoCoMix]] (Meta): predict SAE concepts and interleave them;
   * [[2605.24956|NITP]] (ICML'26): next implicit token prediction;
   * [[2512.03442|PretrainZero]].
   * See also MTP / TOP in [`decoding/multi-token-prediction`](../../decoding/multi-token-prediction/README.md).
5. **Architecture-level training bottlenecks**: [[2603.10145|the LM head suppresses 95–99% of gradient norm]]
   (COLM'26). This motivates better output layers.
6. **Synthetic pre-pretraining**: [[2603.10055|neural cellular automata]]. 164M NCA tokens improve LM by up to 6% and
   converge 1.6× faster, beating 1.6B natural tokens.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2506.20512\|OctoThinker]] | Mid-training | Stable-then-decay mid-training on high-quality math + long-CoT QA makes Llama RL-scalable | Closes the RL gap to Qwen; open 70B-token math corpus |
| 2 | [[2503.19206\|Overtrained LMs are harder to fine-tune]] (ICML'25) | Pretraining budget | Parameter sensitivity grows with pretraining tokens | Instruction-tuned OLMo-1B pretrained on 3T tokens is >2% worse than its 2.3T counterpart. Budget for adaptability |
| 3 | [[2510.03264\|Front-Loading Reasoning]] (NVIDIA) | Data allocation | Reasoning data early (pretraining) vs late (SFT) | Early injection gives durable gains; over-scaling SFT erases them |
| 4 | [[2506.08007\|Reinforcement Pre-Training]] | Objective | Reason about the next token and get a verifiable reward from the corpus | Better NTP accuracy with compute; stronger RL starting point |
| 5 | [[2501.01956\|MeCo: metadata conditioning]] | Efficiency | URL/metadata prefix in pretraining, removed in cooldown | Same performance with 33% less data; steerable |
| 6 | [[2507.17634\|WSM: checkpoint-merging schedule]] | LR schedule | Constant LR + merge checkpoints instead of decay | +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro vs WSD |
| 7 | [[2605.06546\|Token Superposition]] | Efficiency | Train on superposed token sequences early in training | Up to 2.5× less pretraining time at equal loss (10B-A1B MoE) |
| 8 | [[2503.15450\|SkyLadder]] (NeurIPS'25) | Context schedule | Grow context window during pretraining | +3.7% and 22% faster training |
| 9 | [[2512.07783\|Interplay of pre-/mid-training and RL]] | Science | Synthetic reasoning tasks with controlled distributions | RL extends only near-competence skills; mid-training matters; process rewards reduce hacking |
| 10 | [[2509.13310\|Agentic continual pre-training]] (Tongyi) | Agent CPT | Large-scale agentic trajectories in CPT before post-training | AgentFounder-30B state of the art on BrowseComp/HLE |
| 11 | [[2603.10145\|LM head is a gradient bottleneck]] (COLM'26) | Analysis | Low-rank LM head suppresses most of the gradient | Trivial patterns become unlearnable; motivates new heads |
| 12 | [[2502.08524\|CoCoMix: continuous concepts]] (Meta) | Objective | Predict SAE concepts and mix them into the hidden state | More sample-efficient than NTP, KD and pause tokens |

**Also useful.**
* Training dynamics: [[2506.16029|EvoLM]], [[2503.09543|PolyPythias]], [[2506.21551]] (grokking in MoE pretraining).
* Reflection appears in pretraining: [[2504.04022]].
* Curriculum and reweighting: [[2506.11300]], [[2502.06733]].
* Multimodal pretraining: [[2603.03276]], [[2608.05000]], [[2607.22043]].
* Recovery without checkpoints: [[2506.15461|CheckFree]].
* Long-sequence memory: [[2506.03077|StreamBP]].
* Distributed: [[2501.18512|Streaming DiLoCo]].
* Budget case study: [[2608.27370|Puro-2B]] (Qwen2-1.5B-class on one RTX 5090 for $5,090).

**Recommendation.**
* Plan the base for **post-training**: don't overtrain small bases far beyond what the fine-tune can use.
* Front-load reasoning and code data, then run a dedicated mid-training phase before RL.
* Use MeCo + short-to-long context + WSM-style merging.
* Add an MTP/TOP auxiliary loss; they are cheap.
