**Verdict.** This bucket holds the *bit-width-agnostic* core of quantization:

* solver theory (GPTQ = Babai; error bounds);
* cross-layer error propagation (GPTAQ, QEP, Qronos);
* rotation and transform calibration (DartQuant, FPTQuant);
* calibration-free quantization (SINQ);
* "prevent outliers during pretraining" (OSP);
* architecture-specific studies: Mamba, RWKV, diffusion LLMs, MLLMs.

If you implement one PTQ solver, make it **GPTQ with asymmetric calibration (GPTAQ) and error propagation (QEP)**,
fed by a rotation. That combination is the 2025–26 consensus baseline.

### Hand ranking

| # | Paper | Contribution | Why you care |
| ---: | --- | --- | --- |
| 1 | [[2507.18553\|GPTQ = Babai's nearest plane]] (ICLR'26) | Back-to-front GPTQ is exactly Babai's CVP algorithm on the Hessian lattice | Gives error bounds, principled column ordering, and a link to lattice decoding |
| 2 | [[2504.02692\|GPTAQ]] (ICML'25) | Calibrate each layer to the **full-precision** input (asymmetric), closed form | About 20 extra lines of code; quantizes a 405B model on one GPU; strictly better than GPTQ |
| 3 | [[2504.09629\|QEP]] (NeurIPS'25) | Propagate and compensate accumulated cross-layer error, with a tunable strength | Drop-in for GPTQ/AWQ/QuIP; biggest gains at low bits |
| 4 | [[2509.22944\|SINQ]] (Huawei) | Sinkhorn-normalized **dual (row + column) scales**; column std predicts activation scale | **Calibration-free**, near-zero cost: ideal for "quantize on load" in a runtime |
| 5 | [[2506.19697\|Outlier-Safe Pre-Training]] (ACL'25) | Muon + single-scale RMSNorm + learnable embedding projection → **no outliers** | 1.4B model on 1T tokens that quantizes to 4 bits with minimal loss. The model-builder's fix |
| 6 | [[2505.11695\|Qronos]] | Sequential rounding that corrects past (weight + activation + previous-layer) error and diffuses future error; Cholesky implementation | Beats GPTQ/GPTAQ; composes with Hadamard |
| 7 | [[2505.07004\|GuidedQuant]] (ICML'25) | End-loss gradient-weighted objective keeping intra-channel dependencies, plus the LNQ non-uniform quantizer | Improves scalar, vector and W&A; Llama-2-70B in <3 h on 8×RTX 6000 Ada |
| 8 | [[2511.04063\|DartQuant]] (NeurIPS'25) | Distribution-aware rotation calibration ("Whip" loss) + QR-Orth | 47× faster, 10× less memory than SpinQuant/OSTQuant; **70B on one RTX 3090 in ~3 h** |
| 9 | [[2506.04985\|FPTQuant]] (ICML'26) | Mergeable pre-RoPE Q/K transform, V transform, and dynamic per-token scaling | **Static INT4** with no custom kernels, up to 3.9× over FP |
| 10 | [[2508.14896\|PTQ for diffusion LLMs]] | First systematic dLLM PTQ study: outliers everywhere; GPTQ and DuQuant are best | Read before quantizing LLaDA/Dream |
| 11 | [[2505.02214\|Qwen3 quantization study]] | 1–8-bit sweeps on Qwen3 | Qwen3 is *more* sensitive than Llama-3 at ≤3 bits |
| 12 | [[2501.13484\|MambaQuant]] (ICLR'25) | Variance-aligned rotations for Mamba (parallel scan amplifies outliers) | SSM quantization reference |
| 13 | [[2608.27875\|HyQuant]] (EMNLP'26) | Low-bit attention except vertical-line tokens and the local window in high precision | Hybrid-precision attention and KV |
| 14 | [[2601.14277\|Which llama.cpp quant?]] | Controlled GGUF K-quant comparison on Llama-3.1-8B | **5-bit K-quants** are the sweet spot on CPU |
| 15 | [[2510.06213\|Training dynamics & PTQ robustness]] | Quantization error diverges from val loss **once LR decays**; LR schedule controls robustness | Model builders: choose schedules with quantization in mind |

Theory worth knowing:
* [[2508.04853]]: OPTQ/Qronos bounds.
* [[2601.17187]] / [[2605.13768]]: high-rate quantized matmul I/II.
* [[2602.05790]]: the price of VQ metric universality is ≤0.11 bit.
* [[2609.11716|Why Does PTQ Work?]]

Serving-relevant:
* [[2609.26333|Disaggregated Quantization]]: separate formats for prefill and decode.
* [[2603.19296|TTQ]]: test-time activation-aware quantization.
* [[2606.02288]]: massive spikes are bias vectors, and removing them makes activations spike-free.
