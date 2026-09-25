**Verdict.** Grammar-constrained decoding (GCD) is now a solved *engineering* problem. XGrammar / llguidance-style engines give
near-zero-overhead token masks for JSON Schema and CFGs, and [[2601.04426|XGrammar-2]] extends this to *dynamic*, agentic
structures (tag-triggered tool calls, cross-grammar caching). The open problems are about **quality**, not speed:

* **The constraint/format tax.** Masking + renormalizing pushes the model onto locally valid but semantically wrong paths.
  Restrictive grammars provably reduce reasoning power ([[2502.09061|CRANE]], ICML'25). The fix is always the same shape:
  **let the model think unconstrained, then constrain** (CRANE's augmented grammar, [[2601.07525|In-Writing]] trigger
  tokens, [[2603.03305|DCCD]] draft-then-constrain).
* **Sampling bias.** Myopic per-token masking does not sample from *p(x | constraint)*. SMC with tractable
  proposals ([[2606.01926]], ICML'26) and bias-corrected variants fix it where the distribution matters.
* **Tool suppression.** Enabling JSON Schema *and* tool calling together can make tool-call tokens unreachable under the
  compiled mask ([[2606.25605|Constraint Tax]]). The engine must compose grammars (tool-call branch ∪ schema branch),
  not intersect them.
* **dLLMs need new algorithms.** Left-to-right masks do not apply to parallel/any-order decoding: [[2505.23061|DINGO]]
  (regular, distribution-preserving DP), [[2508.10111|CFG for dLLMs]] and [[2602.00612|LAVE]] (CFG with lookahead that
  guarantees extendability).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2601.04426\|XGrammar-2]] | Engine | **TagDispatch** (switch grammar on tags, e.g. `<tool_call>`), **cross-grammar cache** of sub-structures, Earley-based adaptive mask cache, JIT compile, repetition-state compression | >6× faster than prior engines on dynamic agentic workloads. **Reference design for a runtime** |
| 2 | [[2502.09061\|CRANE]] (ICML'25) | Theory + method | Proof that constant-depth transformers under a *restrictive* output grammar lose expressivity; **augment the grammar** with a free-reasoning region, then constrain the answer | Up to +10 pts over constrained and unconstrained baselines on GSM-symbolic/FOLIO |
| 3 | [[2501.10868\|JSONSchemaBench]] | Benchmark | 10K real-world schemas; measures efficiency, declared and empirical coverage, compliance, quality across 6 engines (Guidance, Outlines, llama.cpp, XGrammar, OpenAI, Gemini) | **The standard test for a structured-output engine** |
| 4 | [[2502.05111\|Flexible and Efficient GCD]] (ICML'25) | Engine algorithm | New tokenizer/lexer alignment for CFGs | **17.71× faster offline preprocessing** with state-of-the-art online masking. Matters for per-request grammars |
| 5 | [[2603.03305\|DCCD]] | Training-free | Unconstrained draft → constrained decode **conditioned on the draft**; KL-projection analysis of the "projection tax" | Recovers much of the accuracy lost to masking; costs one extra (unconstrained) draft pass |
| 6 | [[2505.23061\|DINGO]] (NeurIPS'25) | dLLM | Dynamic programming over a DFA for **block-parallel** tokens; provably distribution-preserving | Strict regex/JSON adherence for dLLMs with large accuracy gains over unconstrained |
| 7 | [[2504.09246\|Type-constrained code generation]] (PLDI'25) | Semantic constraint | Prefix automata + search over inhabitable types (TypeScript) | **Halves compile errors**; better functional correctness in synthesis, translation, repair |
| 8 | [[2606.25605\|Constraint Tax: tool suppression]] | Failure analysis | Joint tool-calling + JSON-Schema masks make tool-call tokens unreachable | Reproducible in production across model families. **Test your engine for it** |
| 9 | [[2606.01926\|Tractable proposals for SMC]] (ICML'26) | Unbiased sampling | Tensorize finite automata on GPU → globally constrained proposals + circuit-based potentials for SMC | Removes locally-constrained-decoding bias at practical cost |
| 10 | [[2602.00612\|LAVE]] / [[2508.10111\|CFG for dLLMs]] | dLLM | Use the parallel per-position distributions to *look ahead* and verify each accepted token can still complete to a valid sentence | Reliable CFG compliance for LLaDA/Dream-class models |
| 11 | [[2601.07525\|In-Writing: think before constraining]] | Decoding policy | Free-form reasoning until a trigger token, then structured decoding; prevents premature triggering | Accuracy of free generation + format guarantees in one call |
| 12 | [[2502.18878\|SchemaBench + schema RL]] (ACL'25) / [[2512.00319\|RL-Struct]] | Training | RL with a fine-grained schema validator as reward | Models that emit valid JSON *without* masks (cheap: GRPO on small models) |
| 13 | [[2602.22647\|STATIC]] (KDD'26) | Engine (retrieval) | Flatten the item trie to a **CSR sparse transition matrix** → vectorized constrained decoding on TPU/GPU | Removes trie latency penalty for generative retrieval over millions of IDs |
| 14 | [[2604.14862\|Schema-key wording]] | Prompting | Schema keys are an **instruction channel** (they enter the context) | Changing key names alone moves GSM8K/Math500 substantially |
| 15 | [[2510.14703\|ToolPRM]] (ACL'26) | Inference scaling | Fine-grained beam search + process reward over function name/argument decisions | "Explore more, retain less": early JSON errors are unrecoverable |

**Also useful.**
* Engine speed: [[2506.01151|Earley-driven dynamic pruning]], [[2605.29986|token-space compression]],
  [[2608.03065|parser-stack classification]], [[2608.12574|trie automata for large finite sets]],
  [[2603.05540|attention meets reachability]].
* Bias/intent preservation: [[2504.09135]], [[2510.17376|AdapTrack]], [[2503.18050|(G)I-DLE]],
  [[2608.10137|The Parser Already Knows]].
* Tax measurements: [[2604.03616|The Format Tax]], [[2605.26128]], [[2606.09410|Capacity, Not Format]] (the tax shrinks with
  spare capacity), [[2607.18476]] (JSON collapses answer diversity), [[2609.23742]].
* dLLM constrained decoding: [[2606.00722|EPIC]], [[2607.07026]], [[2605.16829]], [[2606.04535|dynamic infilling anchors]],
  [[2503.09790|Constrained Discrete Diffusion]] (NeurIPS'25).
* Systems cross-over: [[2604.18170|Copy-as-Decode]] (grammar-constrained *parallel prefill* for edits: copy spans verbatim
  from the input) and [[2608.28276]] (structure-conditioned KV persistence).
* Serialization format: [[2603.03306|TOON vs JSON]].

**For a runtime.**
1. Integrate an XGrammar-2-class engine: masks computed on CPU **overlapped** with the GPU forward, adaptive token-mask
   cache, per-request JIT grammar compile with a cross-request cache.
2. Support **tag-triggered grammars** (free text → `<tool_call>` → JSON args → free text). This is both the agent use case
   and the practical fix for the reasoning tax. Never compile "tools + response schema" as an intersection.
3. Make masks compose with **speculative decoding** (mask drafted tokens; roll the grammar state back on reject) and
   with **jump-forward** decoding: deterministic grammar spans are appended in one prefill step (see Copy-as-Decode).
4. Expose an optional "draft-then-constrain" mode (DCCD) for high-stakes structured reasoning.
5. For dLLM backends, ship a DFA/CFG checker that works per block (DINGO/LAVE), not per token.

**For model builders.** Train format adherence in (schema-validator RL is cheap), keep reasoning *outside* the schema,
and evaluate on JSONSchemaBench + StructEval with the mask **off** and **on** to measure the tax.
