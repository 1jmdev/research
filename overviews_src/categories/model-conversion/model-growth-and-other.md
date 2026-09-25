**Verdict.** This small bucket covers the remaining ways to reuse a trained model's weights: **grow** it (depth/width
up-scaling, warm-starting), **re-architect** it (decoder-only → encoder-decoder or bidirectional encoder), or **add recurrent
depth**. The practical results:

* **Model families via progressive growth.** [[2504.00623|Progressive model expansion]] (COLM'25) trains the small
  model first, then expands, and cuts the cost of a whole 1B–8B family by ~25%.
* **Warm-starting works if you stop insisting on function preservation.** [[2605.13405|When is warmstarting
  effective?]] shows function-preserving growth operators over-constrain training. Hyperparameter transfer (µP-style)
  matters more.
* **Decoder → encoder-decoder** ([[2504.06225|Encoder-Decoder Gemma]], the T5Gemma line): better quality per inference
  FLOP for input-heavy tasks (summarization, RAG, classification).
* **Decoder → bidirectional encoder** ([[2604.02045|BidirLM]]) for embeddings/retrieval, composing specialized
  generative models by merging.
* **Retrofitted recurrent depth.** Split a pretrained model into prelude / weight-tied recurrent block / coda and train
  it to loop ([[2511.07384|Teaching LMs to think deeper]], [[2608.11233]]). You get test-time compute without more
  parameters.
* **KV-invariant expansion** ([[2609.27294|KITE]]): grow the model without growing KV-cache cost, which targets
  agentic long-context decode.

### Hand ranking

| # | Paper | Conversion | Key idea | Result / cost |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.06225\|Encoder-Decoder Gemma]] (Google) | Decoder-only → encoder-decoder | Initialize both stacks from a pretrained decoder; pretrain with PrefixLM/UL2 | At similar inference budget: comparable pretraining, clearly better fine-tuning (2B-2B beats Gemma 2B by ~7% after IT); unbalanced 9B-2B pairs beat 2B-2B by >3% |
| 2 | [[2504.00623\|Progressive model-family construction]] (COLM'25) | Growth | Train small → expand → continue, yielding every size in the family | ~25% less total compute for a 1B–8B family at comparable quality |
| 3 | [[2511.07384\|Retrofitted recurrence]] | Dense → recurrent-depth | Convert to prelude/recurrent/coda + curriculum on the number of loops | Better math than post-training the original model at the same compute; decouples test-time compute from size |
| 4 | [[2609.27294\|KITE]] | Expansion | KV-invariant transformer expansion: new capacity that adds no KV cache | Step Scale Transformer (67B MoE, 2.15B active): lower loss than 47B/63B MoEs at equal training compute, with 6.7–31.6% lower inference cost |
| 5 | [[2605.13405\|When is warmstarting effective?]] | Growth study | Growth operators × hyperparameters × scale | Warm-starting works when paired with the right HP transfer; ~40–50K GPU-h study |
| 6 | [[2604.02045\|BidirLM]] | Causal → bidirectional encoder | Adaptation objectives + merging to compose specialized generative models into an omnimodal encoder | 250 MI250X-h extra for the omnimodal version (≈90 H100-h) |
| 7 | [[2508.08011\|OpT-DeUS]] | Depth up-scaling | Optimal-transport neuron alignment before merging adjacent layers into new layers | Better than copy/average depth up-scaling |
| 8 | [[2507.07129\|Growing Transformers on a frozen substrate]] | Layer-wise growth | Stack and train only new blocks + head; LoRA phases for global readjustment | Feasibility and trade-offs of constant-active-parameter growth |

**Recommendation.** If you ship a family of sizes, grow rather than train each size separately. If you want test-time
compute without extra parameters, retrofit recurrence. For runtimes, recurrent-depth models need **loop-count control
per request** and KV handling for weight-tied blocks. Encoder-decoder support is cheap to add and pays off for
input-heavy workloads.
