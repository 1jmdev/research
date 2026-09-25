# Curation guide (manual review of every candidate)

Every candidate paper's title and abstract was read by the reviewer (Claude). Each paper is **kept** (K) and filed
under a category code, or **dropped** (D). Dropped papers are removed from `papers/`, but their ids stay in
`data/curation.tsv`.

## Keep if the paper is useful for building a top LLM runtime or for building / optimising LLMs

* It proposes or evaluates a technique, kernel, system, format, architecture, training recipe, conversion method or
  decoding method that transfers to LLMs or VLMs.
* It is an open-model technical report with architecture or training details.
* It gives a strong empirical or theoretical insight that changes design decisions (scaling laws, failure analyses,
  careful benchmarks of efficiency methods).
* It is a high-quality survey of a runtime- or model-building topic.

## Drop if

* It is an **application** of LLMs to a domain (medicine, law, finance, education, recommendation, code for a niche
  task, agents for a specific product) with no transferable efficiency or modelling technique.
* It is **off-topic**: non-LLM vision, speech, robotics or time series; generic ML theory with no LLM relevance;
  networking or IoT papers that only mention LLMs.
* It is safety, alignment, bias or jailbreak work without a runtime or modelling contribution.
* It uses a technique only as a tool (e.g. "we fine-tune with LoRA to do X").
* It is a trivial or incremental benchmark with no actionable finding.

## Category codes

Q0 sub-1-bit · Q1 1-bit/ternary · Q2 2-bit · Q3 3-bit · Q4 INT4 · QF FP4/MX · Q8 5–8 bit · QM mixed precision ·
QT low-precision training · QG quant general ·
KQ KV quant · KE KV eviction · KL KV low-rank · KX KV cross-layer · KO KV offload · KP prefix/KV reuse · KG KV general ·
AS sparse attn · AL linear attn · AR SSM/RNN · AH hybrid · AK attention kernels · AV attention variants/sinks ·
AP long context/position ·
DS speculative · DM MTP · DJ Jacobi/parallel · DD diffusion LMs · DI dLLM inference · DE early exit · DC constrained ·
DP sampling ·
CA AR→diffusion · CL →linear/hybrid · CM →MLA/GQA · CU →MoE · CT tokenizer transfer · CG merging · CO other ·
PU 2:4/unstructured · PS structured · PA activation sparsity · PL low-rank · PD distillation · PG compression general ·
EA MoE arch · EI MoE serving · EC MoE compression · ET MoE training ·
SB scheduling · SD PD-disagg · SP parallelism · SK kernels/compilers · SH hardware · SE edge · SM memory/offload ·
SN energy · SG serving general ·
TO optimizers · TS scaling laws · TP pre-training · TF PEFT · TR RL · TD data · TK tokenization ·
RE efficient reasoning · RT test-time scaling · RL latent/looped ·
XP prompt compression · XV visual tokens · MR tech reports · MS small LMs · MA architectures
