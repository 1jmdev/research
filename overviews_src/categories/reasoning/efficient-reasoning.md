**Verdict.** Reasoning models overthink: long CoT on easy questions, "Wait" loops, switching between thoughts.
Efficiency methods fall into four families, in order of practicality.

1. **Training-free, runtime-level controls.** These are the cheapest and deploy with any model.
   * **Early exit when the answer converges**: [[2504.15895|DEER]] probes a trial answer at reasoning transition
     points and stops when confident; 19–80% shorter CoT with *higher* accuracy across 11 models.
   * **Suppress reflection tokens**: [[2506.08343|NoWait]], −27–51% length.
   * **Skip thinking** ([[2504.09858|NoThinking]]) plus parallel sampling; strong at low budgets.
   * **Shortest-first voting**: [[2505.17813|short-m@k]] takes the first m finished of k parallel chains. Shorter chains
     are more often correct.
   * **Prompting**: [[2502.18600|Chain of Draft]] (as little as 7.6% of the tokens), [[2503.05179|Sketch-of-Thought]]
     (up to −84%).
   * **Activation steering**: [[2504.07986|SEAL]].
2. **Length-aware RL** (the model learns *how much* to think):
   * [[2503.04697|L1 / LCPO]]: obey a length budget given in the prompt; 1.5B beats GPT-4o at equal length;
   * [[2505.15612|LASER]], [[2504.01296|ThinkPrune]] (halve length for −2%), [[2508.09726|GFPO]] (sample more, keep
     short correct ones), [[2505.13438|BRPO]] (anytime reasoning), [[2503.04472|DAST]], [[2501.12570|O1-Pruner]].
3. **Hybrid / adaptive thinking** (decide *whether* to think):
   * [[2505.13379|Thinkless]] (DeGRPO, −50–90% long-thinking use);
   * [[2505.14631|Large Hybrid-Reasoning Models]];
   * [[2505.11896|AdaCoT]], [[2505.20258|ARM]].
   * Production models now ship thinking on/off and budget controls.
4. **Compress the CoT itself**:
   * [[2502.12067|TokenSkip]]: skip unimportant tokens; −40% at <0.4% loss;
   * [[2502.15589|LightThinker]]: compress thoughts into gist tokens mid-generation, cutting KV and memory;
   * [[2603.05433|CRISP]]: iterative self-policy distillation; −56% on Qwen3-14B;
   * merging long and short models: [[2503.20641]].
   * Latent CoT is in [`latent-and-looped`](../latent-and-looped/README.md).

**Failure analyses to know:** [[2501.18585|underthinking]] (premature thought switching), [[2504.06514|missing premise
causes runaway length]], [[2502.08235|overthinking in agentic tasks]], and [[2502.07266|optimal CoT length is
task-dependent and shrinks with capability]].

### Hand ranking

| # | Paper | Family | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.15895\|DEER: dynamic early exit]] | Runtime, training-free | At "Wait"/transition points, induce a trial answer; exit if its confidence is high | −19–80% CoT length with +0.3–5% accuracy on 11 models |
| 2 | [[2503.04697\|L1 (LCPO)]] | RL length control | RL with target-length-in-prompt reward | Precise length control; 1.5B beats GPT-4o at equal reasoning length |
| 3 | [[2505.13379\|Thinkless]] (NeurIPS'25) | Hybrid | `<short>` / `<think>` control tokens learned with decoupled GRPO | −50–90% long-thinking use |
| 4 | [[2505.17813\|Don't overthink: short-m@k]] | Runtime | Run k chains in parallel; stop when the first m finish; majority vote | Better accuracy at lower wall-clock; shorter chains are more often correct |
| 5 | [[2502.12067\|TokenSkip]] (EMNLP'25) | CoT compression | Train on importance-pruned CoTs at controllable ratios | −40% tokens at <0.4% loss (Qwen2.5-14B GSM8K) |
| 6 | [[2508.09726\|GFPO]] (Microsoft) | RL | Sample larger groups, train only on the shortest / most token-efficient correct responses | Big length cuts; training compute buys test-time savings |
| 7 | [[2504.09858\|Reasoning models can be effective without thinking]] | Runtime | Prefill an empty thinking block; parallel NoThinking + best-of-N | Beats thinking at low budgets (51.3 vs 28.9 at 700 tokens) |
| 8 | [[2502.18600\|Chain of Draft]] | Prompting | Minimal "draft" intermediate steps | CoT-level accuracy with as little as 7.6% of tokens |
| 9 | [[2502.15589\|LightThinker]] (EMNLP'25) | Compression | Learn to compress past thoughts into gist tokens during generation | Lower peak KV memory and time at competitive accuracy |
| 10 | [[2504.01296\|ThinkPrune]] | RL | Hard token limit during RL with iterative tightening | R1-Distill-1.5B: length halved at −2% |
| 11 | [[2603.05433\|CRISP]] | Self-distillation | Iterative on-policy self-distillation from concise-prompted self (reverse KL) | −56% length on Qwen3-14B with accuracy preserved |
| 12 | [[2506.08343\|NoWait]] | Runtime | Suppress "Wait/Hmm" tokens by logit masking | −27–51% length across R1-style families |

**Also useful.**
* Adaptive budgets: [[2505.18822|AdaCtrl]], [[2507.14958|MUR]], [[2602.08354|SAGE]] (models implicitly know when to stop),
  [[2505.14604]] (self-braking).
* Verification-based: [[2505.17941|VeriThinker]].
* Structure over content: [[2502.07374]] (long-CoT structure is what's learned).
* Surveys: [[2503.16419|Stop Overthinking]], [[2503.21614]], [[2504.10903]], [[2503.23077]].

**Runtime checklist.**
* Thinking-budget parameters: max thinking tokens, and forcing `</think>` with an answer prefix when the budget ends.
* DEER-style confidence probes at transition tokens (needs cheap forked answer probes on a shared prefix).
* Logit suppression lists for reflection tokens.
* Parallel sample-and-stop-early (short-m@k) sharing the prompt KV.
* Hybrid-mode routing: think vs no-think per request.
