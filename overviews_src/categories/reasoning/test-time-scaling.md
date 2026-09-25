**Verdict.** Test-time scaling (TTS) has two axes, and the runtime has to support both.

* **Sequential** (think longer): [[2501.19393|s1]] budget forcing, with "Wait" appended to extend and forced end to
  stop. s1-32B goes from 50% to 57% AIME'24 by extrapolation.
* **Parallel** (think wider): best-of-N / self-consistency with smarter selection.

**The 2025–26 consensus:**
1. **Parallel beats sequential per unit of latency** once the model's single-chain length saturates. Longer CoTs
   contain more harmful self-revisions ([[2502.12215]]). There is even [[2507.14417|inverse scaling in test-time
   compute]] on some tasks.
2. **Confidence-filtered parallel thinking is the best training-free recipe.** [[2508.15260|DeepConf]] (Meta) uses
   local token-confidence signals to kill low-confidence traces early and weight votes. It reaches **99.9% AIME'25 with
   −84.7% tokens** vs full parallel thinking. It integrates into vLLM with a few lines. Related: [[2502.18581|self-certainty
   BoN]], [[2502.06233|confidence-weighted SC]].
3. **Learned parallelism is the next step.** The model spawns and joins threads itself:
   * [[2504.15466|APR]] (spawn/join, RL-trained; 80.1% vs 66.6% at 20K tokens);
   * [[2509.07980|Parallel-R1]], [[2512.07461|Native Parallel Reasoner]];
   * [[2601.05593|PaCoRe]]: an 8B model reaches 94.5% on HMMT'25 by coordinating ~2M tokens of parallel reasoning;
   * [[2504.06261|Hogwild! Inference]]: parallel workers share one KV cache, with no fine-tuning needed.
   * Also see [`decoding/jacobi-and-parallel-decoding`](../../decoding/jacobi-and-parallel-decoding/README.md)
     (Multiverse).
4. **Verifiers got generative and scalable**: [[2504.16828|ThinkPRM]] (long-CoT PRM from 1% of PRM800K labels),
   [[2504.00891|GenPRM]], [[2504.02495|inference-time scaling for generalist RMs]] (DeepSeek GRM),
   [[2501.07301|lessons from developing PRMs]] (Qwen). *When to verify vs solve*: [[2504.01005]].
5. **Compute-optimal TTS depends on model, task and verifier**: [[2502.06703|a 1B model can beat a 405B one]] with the
   right strategy.
6. **Beyond single queries.**
   * [[2504.13171|Sleep-time compute]]: pre-compute over context before the query arrives; 2.5× lower cost per query.
   * [[2601.16175|TTT-Discover]]: test-time *training* with RL for discovery problems (math, kernels, algorithms) at a
     few hundred dollars per problem.
   * [[2510.14901|Power sampling]]: base models match RL-trained ones via MCMC over p^α.
7. **Speculative reasoning**: [[2504.07891|SpecReason]] lets a small model do the easy reasoning steps while the big
   model judges them; 1.4–3× faster with better accuracy.

### Hand ranking

| # | Paper | Axis | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2508.15260\|DeepConf: Deep Think with Confidence]] (Meta) | Parallel, training-free | Group/tail token confidence to filter traces online (early stop) and weight votes | 99.9% AIME'25 (GPT-OSS-120B) with up to −84.7% tokens |
| 2 | [[2501.19393\|s1: Simple test-time scaling]] | Sequential | 1K curated examples + budget forcing (append "Wait" / force end) | s1-32B beats o1-preview by up to 27%; extrapolates 50 → 57% AIME'24 |
| 3 | [[2504.15466\|APR: adaptive parallel reasoning]] (COLM'25) | Learned parallel | spawn()/join() threads trained end to end with RL | 80.1% vs 66.6% at 20K tokens; better accuracy at equal latency |
| 4 | [[2601.05593\|PaCoRe]] | Learned parallel | Parallel coordinated reasoning rounds with message passing, trained with RL | 8B reaches 94.5% HMMT'25 (> GPT-5's 93.2%) at ~2M effective tokens |
| 5 | [[2504.16828\|ThinkPRM]] | Verifier | Generative long-CoT PRM fine-tuned on few labels | Beats discriminative PRMs and LLM-as-judge with 1% of PRM800K labels |
| 6 | [[2502.06703\|Can 1B surpass 405B?]] | Compute-optimal | Reward-aware compute-optimal TTS across policies, PRMs and difficulty | 3B > 405B, 7B > o1 on MATH-500/AIME with the right strategy |
| 7 | [[2504.07891\|SpecReason]] | Speculative | Small model drafts reasoning steps; large model scores and accepts or regenerates | 1.4–3× faster, +0.4–9% accuracy; composes with speculative decoding |
| 8 | [[2504.06261\|Hogwild! Inference]] (NeurIPS'25) | Parallel system | Workers attend to each other's KV via RoPE-shifted shared cache | Parallel collaboration with existing reasoning models, no training |
| 9 | [[2504.13171\|Sleep-time compute]] | Offline | Anticipate queries and pre-reason over context | 2.5× lower cost per query at equal accuracy |
| 10 | [[2601.16175\|TTT-Discover]] | Test-time training | RL on a single hard problem at test time with continuous rewards | New state of the art on several discovery tasks with gpt-oss-120b |
| 11 | [[2504.02495\|Generalist reward modeling at inference time]] (DeepSeek) | Verifier scaling | Pointwise generative RM + self-principled critique tuning; scale by sampling | Inference-time scaling of the RM itself beats bigger RMs |
| 12 | [[2507.14417\|Inverse scaling in test-time compute]] | Failure analysis | Longer reasoning hurts on distractor/regression/deduction tasks | Evaluate across reasoning lengths |

**Also useful.**
* Search: [[2501.04519|rStar-Math]] (MCTS + process preference model), [[2502.12018|Atom of Thoughts]],
  [[2501.09891|Mind Evolution]].
* Code: [[2502.14382|S*]], [[2501.14723|CodeMonkeys]], [[2504.00810|Z1]].
* Agents: [[2506.12928]].
* Theory: [[2503.21878]] (is best-of-N optimal?), [[2510.15444]].
* Meta-RL view: [[2503.07572|MRT]] (cumulative regret over tokens).
* Surveys: [[2503.24235]], [[2501.11223|RLM blueprint]].

**Runtime checklist.**
* **n-way parallel sampling with a shared prefix KV** and *per-trace online early termination* (DeepConf needs token
  log-prob windows).
* Budget forcing (thinking-token caps, forced end, "Wait" injection).
* spawn/join primitives with KV sharing (APR/Hogwild/Multiverse).
* A verifier/PRM co-scheduled with the policy.
* Speculative reasoning hooks (small model + judge).
