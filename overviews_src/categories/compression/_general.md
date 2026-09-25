**Verdict.** This bucket holds cross-cutting compression work. Two parts are directly actionable for a runtime.

1. **Lossless weight compression is real and cheap to adopt.** BF16 weights carry only ~11 bits of entropy (the
   exponent is highly skewed), so entropy coding gives **~30% smaller models with bit-identical outputs**:
   * [[2504.11651|DFloat11]]: Huffman-coded exponents + a GPU decompression kernel; Llama-3.1-405B on one 8×80GB node;
     2.3–46× faster than CPU offload;
   * [[2603.17435|ZipServ]]: fixed-length tensor-core-aware encoding decompressed **inside the GEMM**; *faster* than
     cuBLAS (up to 2.21× kernel, 1.22× end-to-end over vLLM);
   * [[2606.15789|Shannon-bound ANS]]: tile-aligned ANS decoding in SGLang; bigger batches, up to 1.6× throughput on
     Mixtral, 11× over DFloat11/NeuZip;
   * [[2502.00922|Huff-LLM]] (hardware) and [[2604.03298|ENEC]] (Ascend).

   This is the **free lunch for BF16 serving** and composes with KV/activation tricks. It does not stack on top of
   4-bit quantization, which is already near-entropy.
2. **Evaluate compressed models on generation and safety, not multiple choice.**
   * [[2606.17609|The Benchmark Illusion]]: pruned models pass MC but cannot *produce* the answer.
   * [[2602.09130|UniComp]]: factual recall survives while multi-step reasoning, multilingual and instruction following
     degrade; retained accuracy does not imply retained safety.
   * [[2607.28196|Fidelity Is Not Safety]]: gently compressed low-rank builds pass perplexity/MMLU guards yet invent
     procedure steps in agentic use. It proposes a data-free screen.
   * [[2504.02010|When reasoning meets compression]]: quantizers over-compress the final-layer MLP and gate
     projections; protecting 2% of weights gives +6.57%.

Also here: outlier science ([[2502.06415|Systematic Outliers]], ICLR'25: outliers act as implicit attention scaling
factors, and removing them structurally helps compression), any-size compression without recomputation
([[2502.01717|ACIP]]), rate-distortion bit allocation ([[2505.03031|Radio]]), delta compression of fine-tunes
([[2504.13237|ImPart]], [[2505.13563|UltraDelta]]), and dLLM-specific pruning ([[2602.17664|sink-aware pruning for
DLMs]]).

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2504.11651\|DFloat11]] | Lossless | Entropy-code BF16 exponents; hierarchical-LUT GPU decoder | 70% size, bit-exact; 405B on a single node; 5.7–14.9× longer generation at fixed memory |
| 2 | [[2603.17435\|ZipServ]] | Lossless + kernel | Triple-bitmap fixed-length encoding decoded straight into tensor-core registers (ZipGEMM) | −30% size **and** 1.22× faster end to end than vLLM |
| 3 | [[2606.15789\|Approaching the Shannon bound]] | Lossless | Tile-level ANS decoding aligned with GEMM tiling; SGLang integration | Batch 20 → 95 on Mixtral-176B (1.6× throughput); up to 11× over prior lossless |
| 4 | [[2602.09130\|UniComp]] | Evaluation | 7 compression methods × 40+ datasets incl. safety/fairness | Knowledge bias + performance-reliability decoupling; task calibration +50% for pruned reasoning |
| 5 | [[2504.02010\|When reasoning meets compression]] | Analysis | Mechanistic view of compressed R1-distilled models | Final-layer MLP-up/gate are critical; protect 2% → +6.57% |
| 6 | [[2606.17609\|The Benchmark Illusion]] | Evaluation | Recognition-only errors in pruned models | MC benchmarks overstate usability |
| 7 | [[2502.06415\|Systematic Outliers]] (ICLR'25) | Analysis | Activation, weight and attention outliers share a cause (softmax); they act as context-aware scaling | Structural removal speeds convergence and improves compressibility |
| 8 | [[2502.01717\|ACIP]] | Any-size compression | Sparsity-penalized SVD pruning order gives a global score map → any target size, no recomputation | One run, every size |
| 9 | [[2607.28196\|Fidelity Is Not Safety]] | Safety eval | Gently compressed models pass all data-free guards but fail agentic procedures | Data-free two-axis screen of compression error |
| 10 | [[2505.03031\|Radio]] | Bit allocation | Rate-distortion optimization for post-training quantization at 100B+ scale | Size- or accuracy-targeted compression |

**Also useful.**
* Composition order: [[2511.19495]] (compression ordering), [[2608.24070|Compression Trinity]].
* Recovery: [[2510.08600|Recover-LoRA]], [[2602.23795|GRAIL]].
* Tooling: [[2603.28845|OneComp]].
* Domain calibration: [[2502.18424|MixCal]].
* Edge lossless: [[2505.02380|EntroLLM]].

**Recommendation for a runtime.** Ship a lossless BF16 path (ZipServ/DFloat11-style fused decompress-GEMM) for "no
accuracy risk" deployments. Put a **generation-based and agentic regression suite** in front of every lossy compression
recipe, because perplexity, MMLU and weight fidelity are not sufficient acceptance tests.
