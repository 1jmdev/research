"""Category taxonomy + rule-based classifier for LLM efficiency / modelling papers.

Every leaf category has regex patterns with weights. A paper's score for a
category = sum(weight * (3 if hit in title else 1 if hit in abstract)).
The highest-scoring leaf above MIN_SCORE becomes the paper's primary folder;
other leaves above SECONDARY_SCORE become cross-reference tags.
"""
import re

MIN_SCORE = 3.0
SECONDARY_SCORE = 3.0

# (folder, human title, description, [(regex, weight), ...])
CATEGORIES = [
    # ------------------------------------------------------------ quantization
    ("quantization/_general", "Quantization — general / analysis / surveys",
     "Quantization papers without a single dominant bit-width: surveys, benchmarks, scaling laws for quantization, sensitivity analysis.",
     [(r"\bquantiz", 3), (r"\blow-bit\b", 2), (r"\bpost-training quantization|\bPTQ\b", 2),
      (r"quantization-aware training|\bQAT\b", 2), (r"\bW\d+A\d+", 2), (r"\bINT[2-8]\b", 1),
      (r"\bbits? per (weight|parameter)|\bbpw\b", 2), (r"\bGPTQ\b|\bAWQ\b|\bSmoothQuant\b|\bQuaRot\b|\bSpinQuant\b|\bOmniQuant\b", 2)]),
    ("quantization/low-precision-training", "Low-precision training (FP8 / FP4 / INT8 training)",
     "Training (pre-training or fine-tuning) with low-precision arithmetic: FP8/MXFP8/FP4/NVFP4 training recipes, quantized optimizers and gradients.",
     [(r"(FP8|FP4|MXFP[468]|NVFP4|INT8|low[- ]precision|8-bit|4-bit)[^.]{0,40}\b(training|pre-?training|pretrain)", 5),
      (r"\b(training|pre-?training)[^.]{0,30}\b(in|with|using) (FP8|FP4|MXFP[468]|NVFP4|low[- ]precision)", 5),
      (r"quantized (training|gradients|optimizer states?)|low-bit optimizer|8-bit optimizer|4-bit optimizer", 4),
      (r"fully quantized training|\bFQT\b", 5)]),
    # bit-width leaves are filled in by assign_quant_bucket(); patterns empty
    ("quantization/sub-1-bit", "Sub-1-bit quantization (< 1 bit per weight)",
     "Extreme compression below one bit per weight: binarization + structured sparsity, vector/codebook quantization with < 1 bpw, weight sharing.", []),
    ("quantization/1-bit-and-ternary", "1-bit and ternary (1 – 1.58 bit) quantization",
     "Binary and ternary weights (BitNet b1.58 family, binarized LLMs, ternary QAT, 1-bit kernels).", []),
    ("quantization/2-bit", "~2-bit quantization (2 – 2.x bits)",
     "Weight quantization around 2 bits per weight: vector/codebook (AQLM, QuIP#, QTIP), 2-bit QAT, mixed 2/3-bit.", []),
    ("quantization/3-bit", "~3-bit quantization",
     "Weight quantization around 3 bits per weight.", []),
    ("quantization/4-bit-integer", "4-bit integer quantization (INT4, W4A16, W4A8, W4A4)",
     "INT4 weight-only and weight-activation quantization, rotation/outlier methods targeting W4A4/W4A8.", []),
    ("quantization/4-bit-floating-point", "4-bit floating point / microscaling (FP4, MXFP4, NVFP4)",
     "FP4-family formats (MXFP4, NVFP4, E2M1) for inference; microscaling block formats.", []),
    ("quantization/5-to-8-bit", "5 – 8 bit quantization (INT8, FP8, W8A8, FP6)",
     "INT8/FP8 inference quantization, W8A8, FP6/FP5 formats.", []),
    ("quantization/mixed-precision", "Mixed-precision & bit allocation",
     "Per-layer / per-channel / per-expert bit allocation, any-precision and elastic-bit models.", []),
    # ------------------------------------------------------------ KV cache
    ("kv-cache/quantization", "KV cache quantization",
     "Low-bit (2–4 bit, sub-2 bit) quantization of keys/values.",
     [(r"(KV|key-value)[- ]cache[^.]{0,40}quantiz|quantiz[^.]{0,40}(KV|key-value)[- ]cache", 6),
      (r"\bKV[^.]{0,15}\b[1-4](\.\d+)?-bit|\b[1-4](\.\d+)?-bit[^.]{0,20}\bKV", 5), (r"\bKIVI\b|\bKVQuant\b", 4)]),
    ("kv-cache/eviction-and-token-selection", "KV cache eviction / token selection / sparse retrieval",
     "Dropping or selecting tokens from the cache (H2O, SnapKV, StreamingLLM-style, query-aware retrieval of KV, budget allocation across heads/layers).",
     [(r"\b(KV|key-value)[- ]cache[^.]{0,60}(evict|prun|drop|selection|select|budget|compress)", 4),
      (r"\b(evict|eviction|token dropping|token eviction)\b", 4), (r"\bSnapKV\b|\bH2O\b|\bPyramidKV\b|\bAdaKV\b|\bStreamingLLM\b", 4),
      (r"\bheavy[- ]hitter", 3), (r"\b(important|critical|salient) tokens?\b[^.]{0,40}\b(KV|cache)", 3)]),
    ("kv-cache/low-rank-and-latent", "KV cache low-rank / latent / head compression",
     "Compressing KV along the hidden/head dimension: low-rank projection, latent KV (MLA-style), head merging.",
     [(r"(KV|key-value)[- ]cache[^.]{0,50}(low-rank|low rank|SVD|latent|projection|dimension)", 5),
      (r"(low-rank|low rank|SVD)[^.]{0,50}(KV|key-value)[- ]cache", 5)]),
    ("kv-cache/cross-layer-sharing", "KV cache cross-layer sharing & merging",
     "Sharing or merging KV across layers or heads (CLA, YOCO-style, MiniCache).",
     [(r"cross-layer[^.]{0,40}(KV|cache|attention)|(KV|cache)[^.]{0,40}(across|between) (adjacent )?layers", 5),
      (r"\b(KV|key-value)[^.]{0,30}(shar|merg)", 3)]),
    ("kv-cache/offloading-and-hierarchical-storage", "KV cache offloading & hierarchical storage",
     "Moving KV to CPU/SSD/remote memory, KV transfer, hierarchical caching tiers.",
     [(r"(KV|key-value)[- ]cache[^.]{0,60}(offload|CPU memory|host memory|SSD|disk|storage|transfer|network)", 5),
      (r"(offload|CPU|SSD|disk)[^.]{0,40}(KV|key-value)[- ]cache", 4)]),
    ("kv-cache/prefix-caching-and-reuse", "Prefix caching & KV reuse (RAG, multi-turn, agents)",
     "Reusing precomputed KV across requests: prefix caching, position-independent caching, CacheBlend-style fusion.",
     [(r"prefix[- ]cach|\bprompt cach|context cach|KV[- ]cache reuse|reus\w*[^.]{0,30}(KV|key-value)[- ]cache|cache blending|position-independent", 6),
      (r"(KV|key-value)[- ]cache[^.]{0,40}(RAG|retrieval-augmented|multi-turn|agent)", 3)]),
    ("kv-cache/_general", "KV cache — general, systems & analysis",
     "KV cache management, memory allocation, surveys and analysis that do not fit a narrower bucket.",
     [(r"\b(KV|key-value)[- ]?caches?\b", 4), (r"\bPagedAttention\b|paged KV|KV block", 3)]),
    # ------------------------------------------------------------ attention / sequence mixers
    ("attention/sparse-attention", "Sparse attention (trainable & training-free)",
     "Block/top-k/dynamic sparse attention for prefill and decode (NSA, MoBA, DSA, MInference, sparse kernels).",
     [(r"\bsparse attention|attention sparsity|\bsparse (prefill|decoding)|block[- ]sparse|\bNSA\b|\bMoBA\b|\bMInference\b|top-k attention|dynamic sparse", 6)]),
    ("attention/linear-attention", "Linear attention & kernelized attention",
     "Linear/kernel attention, gated linear attention, DeltaNet family, test-time-training layers.",
     [(r"\blinear attention|linear-time attention|\bGLA\b|\bDeltaNet\b|gated delta|linear transformer|kernelized attention|test-time training layer|\bTTT\b", 6),
      (r"\blinear (complexity|time) (sequence|token) mix", 3)]),
    ("attention/state-space-and-recurrent", "State-space models & recurrent LMs (Mamba, RWKV, xLSTM)",
     "SSM/RNN language models, selective state spaces, recurrent memory LMs.",
     [(r"\bMamba|state[- ]space models?|\bSSMs?\b|\bRWKV|\bxLSTM|\bS4\b|recurrent (language )?models?|\bRNNs?\b|\bLRU\b", 6)]),
    ("attention/hybrid-architectures", "Hybrid architectures (attention + SSM/linear layers)",
     "Models interleaving full attention with linear/SSM/sliding-window layers (Jamba, Nemotron-H, Qwen3-Next style), layer-ratio studies.",
     [(r"\bhybrid (model|architecture|attention|transformer|LLM|layers?)|interleav\w+[^.]{0,40}(attention|Mamba|SSM|linear)", 6),
      (r"(attention|transformer)[^.]{0,20}(and|with|\+)[^.]{0,20}(Mamba|SSM|linear attention|state[- ]space)", 3)]),
    ("attention/kernels-and-io-aware", "Attention kernels & IO-aware implementations",
     "FlashAttention-style kernels, FP8/FP4 attention kernels (SageAttention), decode kernels (FlashDecoding, FlashInfer), attention on new hardware.",
     [(r"FlashAttention|Flash-?Decoding|FlashInfer|FlashMLA|SageAttention|attention kernel|IO-aware|fused attention|attention (operator|implementation)", 6)]),
    ("attention/attention-variants", "Attention variants (MLA, GQA/MQA, differential, gated, sinks)",
     "Changes to the attention operator/head layout: MLA, GQA, MQA, differential attention, gated attention, attention sinks, softmax alternatives.",
     [(r"multi-head latent attention|\bMLA\b|grouped-query attention|\bGQA\b|multi-query attention|\bMQA\b|differential attention|gated attention|attention sinks?|softmax[- ](alternative|free)|sigmoid attention|attention heads? (design|pruning)|\bQK-?norm|attention logits", 5)]),
    ("attention/long-context-and-position", "Long context, context extension & positional encoding",
     "RoPE scaling / extrapolation, length generalization, long-context training & evaluation, context-window extension.",
     [(r"\bRoPE\b|rotary position|positional (encoding|embedding)|position (encoding|embedding)|\bNoPE\b|\bALiBi\b|\bYaRN\b", 5),
      (r"context (window )?extension|extend\w*[^.]{0,30}context|length (extrapolation|generalization)|long[- ]context", 4)]),
    # ------------------------------------------------------------ decoding
    ("decoding/speculative-decoding", "Speculative decoding (draft–verify)",
     "Draft-then-verify decoding: draft models, EAGLE/Medusa heads, tree verification, self-speculation, speculative decoding systems.",
     [(r"speculative (decoding|sampling|inference|generation|execution)|draft(er|ing)? (model|head|tokens?)|draft-(then-)?verify|\bEAGLE-?\d?\b|\bMedusa\b|self-speculative|token tree verification|acceptance (rate|length)", 7)]),
    ("decoding/multi-token-prediction", "Multi-token prediction (MTP)",
     "Training/inference with multiple future-token heads or objectives (DeepSeek-V3 MTP, future-token prediction, register tokens for MTP).",
     [(r"multi-token prediction|\bMTP\b|multiple future tokens|predict\w* multiple (future |next )?tokens|next-k|future token prediction|multi-token (heads?|objective|training)", 7)]),
    ("decoding/jacobi-and-parallel-decoding", "Jacobi, lookahead & consistency-based parallel decoding",
     "Fixed-point (Jacobi) iteration decoding, lookahead decoding, consistency LLMs, Jacobi forcing, n-gram parallel decoding.",
     [(r"\bJacobi\b|lookahead decoding|consistency (LLMs?|models?|distillation|training)|\bCLLMs?\b|fixed-point iteration|parallel decoding|non-autoregressive|semi-autoregressive", 6)]),
    ("decoding/diffusion-language-models", "Diffusion language models (dLLMs) — models & training",
     "Masked/discrete/continuous diffusion LMs: LLaDA, Dream, Mercury, Gemini Diffusion-style models, block diffusion, training recipes, RL for dLLMs.",
     [(r"diffusion (large )?language models?|language diffusion models?|\bdLLMs?\b|diffusion LLMs?|masked diffusion|discrete diffusion|\bLLaDA\b|\bDream-?\d|block diffusion|diffusion-based (language|text)|text diffusion", 7)]),
    ("decoding/diffusion-llm-inference", "Diffusion LLM inference acceleration",
     "Making dLLM inference fast: KV/feature caching for dLLMs, parallel unmasking schedules, early commit, speculative/AR-hybrid dLLM decoding.",
     []),  # assigned by rule: dLLM + (cache|acceleration|speedup|parallel)
    ("decoding/early-exit-and-layer-skipping", "Early exit, layer skipping & dynamic depth",
     "Adaptive computation per token: early exit, layer skipping, mixture-of-depths, router-based depth.",
     [(r"early[- ]exit|layer[- ]skip|skip\w* layers|mixture[- ]of[- ]depths|dynamic depth|adaptive (depth|computation)|token-level routing across layers", 6)]),
    ("decoding/constrained-and-structured", "Constrained & structured decoding",
     "Grammar/JSON-constrained generation, structured output engines (XGrammar-style).",
     [(r"constrained decoding|grammar-constrained|structured (output|generation|decoding)|JSON (schema|mode)|context-free grammar", 6)]),
    ("decoding/sampling-and-strategies", "Sampling & decoding strategies",
     "Temperature/min-p/top-p style samplers, contrastive decoding, decoding-time interventions.",
     [(r"decoding (strategy|strategies|method|algorithm)|sampling (method|strategy|strategies)|\bmin-p\b|top-p|nucleus sampling|contrastive decoding|temperature (scaling|sampling)", 4)]),
    # ------------------------------------------------------------ model conversion
    ("model-conversion/ar-to-diffusion", "AR → diffusion LM conversion",
     "Adapting pretrained autoregressive LMs into diffusion LMs (DiffuLLaMA, Dream-from-Qwen, block-diffusion adaptation). Compute cost is the key metric.",
     []),  # rule-based
    ("model-conversion/transformer-to-linear-or-hybrid", "Transformer → linear / SSM / hybrid conversion (linearization)",
     "Distilling or converting pretrained attention models into linear-attention, Mamba or hybrid models (LoLCATs, Mamba-in-Llama, Liger, RAD).",
     []),  # rule-based
    ("model-conversion/attention-conversion", "Attention conversion (MHA/GQA → MLA, GQA uptraining, sparse retrofit)",
     "Retrofitting the attention of a pretrained model: MHA2MLA, TransMLA, X-EcoMLA, GQA conversion, retrofitting sparse/sliding attention.",
     []),  # rule-based
    ("model-conversion/dense-to-moe-upcycling", "Dense → MoE upcycling & MoE-fication",
     "Sparse upcycling, MoE-fication of FFNs, expert construction from dense checkpoints.",
     [(r"upcycl|MoE-?fication|moefication|dense[- ]to[- ](sparse|MoE)|convert\w*[^.]{0,30}(into|to) (a )?(mixture[- ]of[- ]experts|MoE)", 7)]),
    ("model-conversion/tokenizer-and-vocab-transfer", "Tokenizer / vocabulary transfer & cross-tokenizer distillation",
     "Swapping or extending a pretrained model's tokenizer, cross-tokenizer distillation, zero-shot tokenizer transfer.",
     [(r"tokenizer (transfer|adaptation|swap|replacement)|vocabulary (transfer|expansion|adaptation|extension)|cross-tokenizer|zero-shot tokenizer", 7)]),
    ("model-conversion/model-merging", "Model merging",
     "Merging fine-tuned checkpoints (task arithmetic, TIES, DARE, model soups, merging for long-to-short reasoning).",
     [(r"model merging|merg\w+ (models|checkpoints|LLMs|weights)|task arithmetic|task vectors?|model soups?|(?-i:\bTIES\b)|(?-i:\bDARE\b)|weight averaging", 6)]),
    ("model-conversion/model-growth-and-other", "Model growth & other architecture conversion",
     "Depth/width growth, layer stacking, re-using checkpoints across architectures, converting modality or precision paradigm.",
     [(r"model growth|depth up-?scaling|layer stacking|progressive (training|stacking)|width expansion|function-preserving", 5)]),
    # ------------------------------------------------------------ compression
    ("compression/unstructured-and-semi-structured-pruning", "Unstructured & N:M (2:4) pruning",
     "Weight-level pruning of LLMs (SparseGPT, Wanda successors), semi-structured 2:4 sparsity and sparse kernels.",
     [(r"\b[24]:[48]\b|N:M sparsity|semi-structured (sparsity|pruning|sparse)|unstructured (pruning|sparsity)|\bSparseGPT\b|\bWanda\b|weight pruning|sparse weights", 6)]),
    ("compression/structured-pruning", "Structured pruning (layers, heads, width, experts)",
     "Removing layers/blocks/heads/channels, depth pruning, width pruning (Minitron-style prune + distill).",
     [(r"structured pruning|layer pruning|depth pruning|width pruning|block pruning|prun\w+ (layers|heads|channels|blocks)|remov\w+ (layers|blocks)|\bMinitron\b", 4), (r"\bpruning\b", 1)]),
    ("compression/activation-sparsity", "Activation / contextual sparsity",
     "Exploiting sparse activations at inference (ReLU-fication, TEAL, contextual sparsity, neuron-level skipping).",
     [(r"activation sparsity|contextual sparsity|sparse activations?|ReLU-?fication|\bTEAL\b|neuron (sparsity|skipping|activation)|dynamic sparsity|\bDejaVu\b|\bPowerInfer", 7)]),
    ("compression/low-rank-decomposition", "Low-rank decomposition & weight factorization",
     "SVD / low-rank / tensor-train compression of weights (SVD-LLM, ASVD), weight sharing, basis sharing.",
     [(r"(low-rank|low rank|SVD|singular value|tensor(-train)? decomposition|matrix (factorization|decomposition))[^.]{0,60}(compress|approximat|factoriz)", 6),
      (r"\bSVD-LLM\b|\bASVD\b|weight sharing|basis sharing", 5)]),
    ("compression/knowledge-distillation", "Knowledge distillation (LLM → smaller LLM)",
     "Logit/feature/on-policy distillation, reasoning-trace distillation, distillation scaling laws.",
     [(r"knowledge distillation|\bdistill\w*|teacher[- ]student|student model|on-policy distillation", 5)]),
    ("compression/_general", "Model compression — general & surveys",
     "Compression surveys, benchmarks and combined pipelines (prune + quantize + distill).",
     [(r"model compression|compress\w* (LLMs|large language models|the model)|compression (techniques|methods|pipeline|benchmark)", 4)]),
    # ------------------------------------------------------------ MoE
    ("mixture-of-experts/architecture-and-routing", "MoE architecture & routing",
     "Router design, load balancing, expert granularity, shared experts, MoE scaling laws.",
     [(r"mixture[- ]of[- ]experts|\bMoEs?\b", 4), (r"\brout(er|ing)\b|load[- ]balanc|expert (specialization|granularity|capacity)|fine-grained experts|shared experts?", 3)]),
    ("mixture-of-experts/inference-and-serving", "MoE inference & serving (expert offloading, EP)",
     "Expert offloading/prefetching, expert parallel serving, MoE kernels, expert caching.",
     [(r"expert (offload|prefetch|cach|placement|parallel)|(?-i:\bEP\b)|MoE (inference|serving|kernel)|all-to-all", 5)]),
    ("mixture-of-experts/compression", "MoE compression (expert pruning, merging, quantization)",
     "Pruning/merging/quantizing experts, expert skipping, MoE-specific compression.",
     [(r"expert (pruning|merging|skipping|dropping|compression|quantization)|prun\w+ experts|merg\w+ experts|compress\w* (MoE|experts)", 6)]),
    ("mixture-of-experts/training", "MoE training systems",
     "Efficient MoE training, communication, stability.", []),
    # ------------------------------------------------------------ serving systems
    ("serving-systems/scheduling-and-batching", "Request scheduling, batching & SLO serving",
     "Continuous batching, request scheduling, preemption, SLO-aware serving, LLM routing between models.",
     [(r"continuous batching|request scheduling|\bscheduler\b|\bSLOs?\b|goodput|preemption|head-of-line|queueing|\bTTFT\b|time-to-first-token|tail latency|batching", 4),
      (r"\b(LLM )?serving (system|framework|engine)|inference (engine|serving|system)|\bvLLM\b|\bSGLang\b|TensorRT-LLM", 3)]),
    ("serving-systems/disaggregated-prefill-decode", "Prefill/decode disaggregation & phase-aware serving",
     "Splitting prefill and decode across instances/hardware, chunked prefill, attention–FFN disaggregation.",
     [(r"disaggregat|prefill[- ]decode|prefill and decode|chunked prefill|decode phase|prefill phase|attention[- ]FFN disaggregation", 6)]),
    ("serving-systems/distributed-inference-and-parallelism", "Distributed inference & parallelism (TP/PP/SP/CP/EP)",
     "Tensor/pipeline/sequence/context parallelism, communication overlap, multi-node inference and training parallelism.",
     [(r"tensor parallel|pipeline parallel|sequence parallel|context parallel|expert parallel|\bFSDP\b|(?-i:\bZeRO\b)|Megatron|all-reduce|collective communication|communication overlap", 5),
      (r"data parallel|distributed (inference|training)|multi-node|multi-GPU", 2)]),
    ("serving-systems/gpu-kernels-and-compilers", "GPU kernels, compilers & operator optimization",
     "GEMM/GEMV kernels, low-bit kernels, kernel fusion, Triton/CUDA generation, compilers, LLM-generated kernels.",
     [(r"\bkernels?\b|\bCUDA\b|\bTriton\b|\bGEMM\b|\bGEMV\b|kernel fusion|operator fusion|compiler|tensor cores?|\bCUTLASS\b|megakernel", 4)]),
    ("serving-systems/hardware-accelerators", "Hardware accelerators (ASIC, FPGA, PIM, NPU, photonic)",
     "Custom silicon and near-memory computing for LLMs, hardware/software co-design.",
     [(r"\bFPGA|\bASIC|processing[- ]in[- ]memory|(?-i:\bPIM\b)|compute[- ]in[- ]memory|(?-i:\bCIM\b)|\bNPUs?\b|accelerator design|hardware accelerator|systolic|photonic|analog (in-memory|computing)|chiplet|co-design", 6)]),
    ("serving-systems/edge-and-on-device", "Edge, mobile & on-device inference",
     "LLMs on phones, laptops, embedded devices, consumer GPUs and CPUs.",
     [(r"on-device|edge (devices?|deployment|computing|inference)|mobile (devices?|phones?|NPU)|smartphones?|consumer[- ](grade )?GPUs?|resource-constrained|microcontroller|\bCPU inference|laptops?|embedded", 5)]),
    ("serving-systems/memory-and-offloading", "Memory management & offloading (weights, activations)",
     "Weight/activation offloading, memory-bandwidth-bound optimizations, SSD/flash-based inference.",
     [(r"offload|GPU memory|memory (bandwidth|footprint|wall|hierarchy|capacity)|flash memory|\bSSD\b|unified memory|\bHBM\b|memory-bound", 4)]),
    ("serving-systems/energy-and-cost", "Energy, carbon & cost of LLM inference/training",
     "Measuring and reducing energy/carbon/$ cost of LLMs.",
     [(r"energy (consumption|efficiency|cost|usage)|energy-efficient|power (consumption|draw|capping)|carbon (footprint|emissions?)|joules?|\bDVFS\b", 6)]),
    ("serving-systems/_general", "Serving & inference systems — general",
     "Inference-system papers that do not fit narrower buckets; benchmarks of inference engines.",
     [(r"\binference (efficiency|optimization|acceleration|latency|throughput)|\bthroughput\b|\blatency\b", 2)]),
    # ------------------------------------------------------------ training
    ("training/optimizers", "Optimizers & training dynamics (Muon, AdamW, schedules)",
     "New optimizers, second-order methods, learning-rate schedules, hyper-parameter transfer (muP), loss spikes.",
     [(r"\boptimi[sz]ers?\b|\bMuon\b|\bAdamW?\b|\bShampoo\b|\bSOAP\b|second-order|learning[- ]rate (schedule|warmup|decay)|\bWSD\b|\bmuP\b|μP|weight decay|loss spikes?|training (stability|instability|dynamics)", 5)]),
    ("training/scaling-laws", "Scaling laws",
     "Compute/data/parameter scaling laws, compute-optimal training, scaling of inference and RL.",
     [(r"scaling laws?|compute-optimal|Chinchilla|power[- ]law", 6)]),
    ("training/pretraining-recipes-and-efficiency", "Pre-training recipes & efficiency",
     "Pre-training methodology, efficiency, curriculum, mid-training, architecture ablations at scale.",
     [(r"\bpre-?training\b|\bpretrain\w*|mid-training|annealing|training (efficiency|cost|recipe)|GPU[- ]hours|tokens? budget", 3)]),
    ("training/parameter-efficient-finetuning", "Parameter-efficient fine-tuning (LoRA & friends)",
     "LoRA/DoRA/QLoRA variants, adapters, PEFT for LLMs, serving many LoRAs.",
     [(r"\bLoRA\b|\bQLoRA\b|\bDoRA\b|low-rank adaptation|parameter[- ]efficient|\bPEFT\b|adapters?\b|prefix[- ]tuning|prompt[- ]tuning", 6)]),
    ("training/rl-for-reasoning", "Reinforcement learning for LLM reasoning (GRPO, RLVR)",
     "RL post-training methods, RL training efficiency (rollout systems, async RL), reward design for reasoning.",
     [(r"\bGRPO\b|\bRLVR\b|verifiable rewards?|\bPPO\b|\bDAPO\b|policy optimization|reinforcement learning|\bRLHF\b|rollouts?", 5)]),
    ("training/data-curation-and-synthetic-data", "Data curation, mixtures & synthetic data",
     "Pre-/post-training data selection, filtering, mixture optimization, synthetic data generation.",
     [(r"data (curation|selection|filtering|mixture|mixing|quality|pruning)|synthetic data|pre-?training (data|corpus|corpora)|dataset (construction|curation)", 6)]),
    ("training/tokenization", "Tokenization & byte-level models",
     "Tokenizer design, vocabulary size, byte-level / tokenizer-free models, dynamic patching.",
     [(r"\btokeni[sz](er|ation)|byte-level|tokenizer-free|vocabulary size|\bBPE\b|byte latent|patch(es|ing) of bytes|subword", 6)]),
    # ------------------------------------------------------------ reasoning efficiency
    ("reasoning/efficient-reasoning", "Efficient reasoning (CoT compression, overthinking, adaptive thinking)",
     "Shortening reasoning traces, adaptive think/no-think, budget control, early stopping of reasoning.",
     [(r"efficient reasoning|overthinking|underthinking|reasoning length|(CoT|chain-of-thought) (compression|length)|concise reasoning|adaptive (thinking|reasoning)|thinking budget|token budget|long-to-short|early stopping of reasoning|reasoning efficiency|length penalty", 7)]),
    ("reasoning/test-time-scaling", "Test-time scaling & inference-time compute",
     "Best-of-N, self-consistency, parallel thinking, verifier-guided search, efficient test-time compute allocation.",
     [(r"test-time (scaling|compute)|inference-time (scaling|compute)|best-of-N|self-consistency|parallel (thinking|reasoning|sampling)|tree search|\bMCTS\b|beam search|process reward model|\bPRMs?\b", 6)]),
    ("reasoning/latent-and-looped", "Latent reasoning, looped & recurrent-depth models",
     "Continuous-thought / latent CoT, looped transformers, recurrent-depth (depth-recurrence) LMs, pause/filler tokens.",
     [(r"latent reasoning|continuous thought|latent (chain|space|thoughts?|CoT)|looped (transformers?|models?|language models?)|recurrent[- ]depth|depth[- ]recurren|implicit reasoning|\bCoconut\b|pause tokens?|filler tokens?", 7)]),
    # ------------------------------------------------------------ context / token compression
    ("context-compression/prompt-and-context-compression", "Prompt & context compression",
     "Compressing long prompts/contexts into fewer tokens (LLMLingua-style, gist/memory tokens, soft compression).",
     [(r"prompt compression|context compression|compress\w* (the )?(prompt|context|long context)|gist tokens?|memory tokens?|soft (prompt|context) compression|LLMLingua", 7)]),
    ("context-compression/visual-token-reduction", "Visual token pruning / merging for VLMs",
     "Reducing image/video tokens in multimodal LLMs for faster inference.",
     [(r"visual tokens?|vision tokens?|image tokens?|video tokens?", 5), (r"token (pruning|merging|reduction|compression|dropping)", 4)]),
    # ------------------------------------------------------------ models
    ("models-and-architectures/technical-reports", "Model technical reports (open & frontier models)",
     "Technical reports of released LLMs/VLMs/dLLMs with architecture and training details.",
     [(r"technical report", 6), (r"\bopen[- ](source|weights?)\b[^.]{0,60}\b(model|LLM)s?", 2)]),
    ("models-and-architectures/small-language-models", "Small language models (≤ ~4B)",
     "Designing and training small LMs, SLM recipes, sub-billion models.",
     [(r"small language models?|\bSLMs?\b|sub-billion|tiny (language )?models?|compact (language )?models?", 6)]),
    ("models-and-architectures/novel-architectures", "Novel architectures & architecture analysis",
     "New layer types, normalization, residual/skip designs, memory layers, architecture search for LLMs.",
     [(r"\bnew architecture|novel architecture|architectur(e|al) (design|search|modification)|normalization layer|\bLayerNorm\b|\bRMSNorm\b|residual (stream|connections?)|memory layers?|hyper-connections|\bMLP\b layer|feed-forward (network|layer)", 3)]),
]

CAT_INDEX = {c[0]: c for c in CATEGORIES}

# ---------------------------------------------------------------------------
LLM_RE = re.compile(
    r"\b(LLMs?|large language models?|language models?|LMs\b|GPT-?\d|ChatGPT|LLaMA|Llama|Qwen|Mistral|Mixtral|DeepSeek|Gemma|Phi-\d|"
    r"\bOPT-|Falcon|OLMo|Pythia|Vicuna|InternLM|GLM-?\d|Kimi|MiniMax|VLMs?|MLLMs?|LVLMs?|LMMs?|vision[- ]language models?|"
    r"multimodal large|reasoning models?|LRMs?\b|dLLMs?|diffusion language|text generation|chatbots?|foundation models?|"
    r"autoregressive (language|generation|decoding|models?)|next-token prediction|transformer-based (language|models?)|decoder-only)",
    re.I)

COMPILED = [(c[0], [(re.compile(p, re.I), w) for p, w in c[3]]) for c in CATEGORIES]

DLLM_RE = re.compile(r"diffusion (large )?language models?|language diffusion models?|\bdLLMs?\b|diffusion LLMs?|masked diffusion (language )?models?|discrete diffusion|\bLLaDA|block diffusion|diffusion-based (language|text|LLM)", re.I)
DLLM_ACCEL_RE = re.compile(r"\b(accelerat|speed-?up|faster|cach\w+|KV[- ]cache|parallel (decoding|sampling|unmasking)|inference (efficiency|speed|acceleration)|throughput|early (commit|stopping|exit)|training-free)", re.I)
AR2DIFF_RE = re.compile(r"(adapt|convert|initializ|continual(ly)? pre-?train|transfer)\w*[^.]{0,80}(autoregressive|\bAR\b|pretrained (LLM|language model))[^.]{0,80}(diffusion)|(autoregressive|\bAR\b)[- ](to|into|→)[- ](diffusion|dLLM)|from (pretrained )?autoregressive (models?|LLMs?)|AR-initialized|\bDiffuLLaMA\b|\bDiffuGPT\b", re.I)
LINEARIZE_RE = re.compile(r"lineariz\w*[^.]{0,60}(LLMs?|large language models?|transformers?|attention|pretrained)|(LLMs?|transformers?|attention)[^.]{0,40}lineariz|(distill|convert|transform(?!er))\w*[^.]{0,80}(pretrained|existing)?[^.]{0,40}(transformers?|attention|LLMs?)[^.]{0,60}(into|to) (a )?(linear|Mamba|SSM|RNN|recurrent|hybrid|subquadratic)|\bLoLCATs\b|Mamba-?in-?(the-?)?Llama|\bLiger\b|\bMOHAWK\b|attention transfer|(convert|distill|transform(?!er))\w*[^.]{0,80}(softmax attention|transformers?|Qwen|Llama|LLMs?|MLLMs?|teacher)[^.]{0,80}\b(linear[- ]attention|RWKV|Mamba|SSMs?|state[- ]space|recurrent|hybrid)|(linear[- ]attention|RWKV|Mamba|SSM|hybrid)[^.]{0,60}(distilled|converted) from|quadratic[- ]to[- ]linear distillation|born from transformer", re.I)
LINEAR_TARGET_RE = re.compile(r"linear[- ]attention|recurren|\bRNNs?\b|subquadratic|sub-quadratic|Mamba|\bSSMs?\b|state[- ]space|hybrid|gated (linear|recurrent|delta)|RWKV|DeltaNet|linear[- ]time|linear complexity", re.I)
ATTNCONV_RE = re.compile(r"\bMHA2MLA\b|\bTransMLA\b|X-EcoMLA|(convert|transform(?!er)|migrat|retrofit|adapt)\w*[^.]{0,60}(MHA|GQA|multi-head attention|grouped-query attention|pretrained (models?|LLMs?))[^.]{0,60}(to|into) (MLA|multi-head latent|GQA|sparse attention|sliding[- ]window)|GQA (conversion|uptraining)|uptrain", re.I)
QUANT_RE = re.compile(r"quantiz|low-bit|\bbinari[sz]|\bternary|\b1\.58|\bW\d+A\d+|\bINT[2-8]\b|\bFP[468]\b|\bMXFP|\bNVFP4|\bbits? per weight|\bbpw\b|\b[1-4](\.\d+)?-bit", re.I)
KV_RE = re.compile(r"\b(KV|key-value)[- ]?caches?\b", re.I)
KV_TITLE_RE = re.compile(r"\bKV\b|key-value|\bkeys?\b[^.]{0,25}\bvalues?\b", re.I)


EVIDENCE = {}


def score(title, abstract):
    """Weighted pattern score per leaf. EVIDENCE[key] = (title_hit, abstract_occurrences)."""
    scores = {}
    EVIDENCE.clear()
    for key, pats in COMPILED:
        s, th, ab = 0.0, False, 0
        for rx, w in pats:
            if rx.search(title):
                s += 3 * w
                th = True
            else:
                n = len(rx.findall(abstract))
                if n:
                    s += w * (1 + 0.25 * min(n - 1, 4))
                    ab += n
        if s:
            scores[key] = s
            EVIDENCE[key] = (th, ab)
    return scores


# ---------------------------------------------------------------- bit buckets
BIT_PATTERNS = [
    (re.compile(r"sub-?1-?bit|sub-one-bit|below (one|1)[- ]bit|less than (one|1)[- ]bit( per)?|<\s*1[- ]bits?\b|\b0\.\d+[- ]bits? ?(per|/) ?(weight|parameter)|\b0\.\d+ ?(bpw|BPW|bits per weight)|\b0\.\d+-bit (LLM|model|quantiz)", re.I), 0.7),
    (re.compile(r"\b1\.58|\bternary|1\.5[- ]bit|\b1\.[0-9]+[- ]bits?\b|\btrit", re.I), 1.58),
    (re.compile(r"\bbinar(y|ized|ization|isation)\b[^.]{0,30}(weights?|LLM|network|quantiz)|\b1-bit\b|\bone-bit\b|\bBitNet\b|\bW1A", re.I), 1.0),
    (re.compile(r"\b2(\.\d+)?[- ]bits?\b|\bW2A|\bINT2\b|\b2 bits\b|\b2-?bpw|\btwo-bit", re.I), 2.0),
    (re.compile(r"\b3(\.\d+)?[- ]bits?\b|\bW3A|\bINT3\b|\bthree-bit", re.I), 3.0),
    (re.compile(r"\bFP4\b|\bMXFP4\b|\bNVFP4\b|\bE2M1\b|4-bit float|\bFP4 ", re.I), 4.1),
    (re.compile(r"\b4(\.\d+)?[- ]bits?\b|\bW4A|\bINT4\b|\bfour-bit", re.I), 4.0),
    (re.compile(r"\b[56]-bits?\b|\bFP6\b|\bFP5\b|\bW[56]A", re.I), 6.0),
    (re.compile(r"\b8-bits?\b|\bW8A8\b|\bINT8\b|\bFP8\b|\bMXFP8\b|\bE4M3\b|\bE5M2\b", re.I), 8.0),
]
MIXED_RE = re.compile(r"mixed[- ]precision|bit (allocation|width search)|any-precision|elastic (bit|precision)|variable bit|per-layer bit|matryoshka quantization", re.I)


OVERHEAD_RX = re.compile(r"(extra|additional|overhead|only|storage of|metadata|index|indices|side information)[^.]{0,25}$", re.I)


def detect_bits(title, abstract):
    """Return (bits_in_title, bits_in_abstract) as sorted lists of bucket values."""
    def find(s):
        out = set()
        for rx, v in BIT_PATTERNS:
            for m in rx.finditer(s):
                # "0.02 bits per weight of overhead" is not a sub-1-bit model
                if v == 0.7 and (OVERHEAD_RX.search(s[max(0, m.start() - 40):m.start()]) or
                                 re.match(r"[^.]{0,25}(overhead|extra|additional)", s[m.end():], re.I)):
                    continue
                out.add(v)
                break
        return sorted(out)
    return find(title), find(abstract)


def quant_bucket(title, abstract):
    t, a = detect_bits(title, abstract)
    text = title + " " + abstract
    if re.search(r"(FP8|FP4|MXFP|NVFP4|low[- ]precision|8-bit|4-bit)[^.]{0,40}(training|pre-?train)", text, re.I) and \
            re.search(r"\btrain", title, re.I):
        return "quantization/low-precision-training"
    bits = t or a
    if not bits:
        if MIXED_RE.search(text):
            return "quantization/mixed-precision"
        return "quantization/_general"
    if MIXED_RE.search(title):
        return "quantization/mixed-precision"
    b = min(bits)
    # ignore 8-bit when it's only mentioned as a baseline alongside lower bits in the abstract
    if b < 1:
        return "quantization/sub-1-bit"
    if b <= 1.58:
        return "quantization/1-bit-and-ternary"
    if b <= 2.0:
        return "quantization/2-bit"
    if b <= 3.0:
        return "quantization/3-bit"
    if b == 4.0:
        if 4.1 in bits and not re.search(r"\bINT4\b|\bW4A", text):
            return "quantization/4-bit-floating-point"
        return "quantization/4-bit-integer"
    if b == 4.1:
        return "quantization/4-bit-floating-point"
    return "quantization/5-to-8-bit"


WEAK_REL_RE = re.compile(r"\b(transformers?|attention|tokens?|language|decod\w+|autoregressive|sequence model)", re.I)
OFFTOPIC_RE = re.compile(r"point clouds?|recommend(er|ation)|forecasting|graph neural|\bGNNs?\b|image (generation|synthesis|super-resolution)|video generation|jet tagging|medical imag|remote sensing|speech enhancement|protein|molecul", re.I)
OFFTOPIC_TITLE_RE = re.compile(r"image generation|text-to-image|text-to-video|video generation|image editing|image synthesis|\b[34]D\b|point clouds?|gaussian splatting|diffusion transformers?|\bDiTs?\b|super-resolution|segmentation|object detection|image classification|\bViTs?\b|vision transformers?|speech synthesis|\bTTS\b|recommend|world models?|\bVLNs?\b|autonomous driving|trajector|vision-language-action|\bVLAs?\b|robot|embodied|manipulation|speech recognition|\bASR\b|\baudio\b|\bvideo\b|one-step diffusion|\bvision\b|\bimages?\b|visual geometry|forecasting|time series|speech|codec|\bTTS\b|acoustic|music|\bEEG\b|\bECG\b|segment anything|track anything|\bCTR\b|click-through", re.I)
LLM_TITLE_RE = re.compile(r"(LLMs?|VLMs?|LMMs?)\b|\b(large language models?|language models?|VLMs?|MLLMs?|LVLMs?|LMMs?|multimodal large|vision-language models?|reasoning models?|KV[- ]cache)\b", re.I)
TECH_TITLE_RE = re.compile(r"technical report|tech report|model card|system card", re.I)
RELEASE_RE = re.compile(r"\b[Ww]e (introduce|present|release|open-source|unveil)\s+[A-Z][\w\-\.]*(?:[\s\-][\w\.\-]+){0,3},?\s+(a|an|the|our)\s+[^.]{0,100}\b(open(-source|-weight)?|foundation|frontier|flagship|series|family|suite|large language|language|multimodal|reasoning|MoE|Mixture-of-Experts|vision-language|base|instruct|chat)\b[^.]{0,60}\b(models?|LLMs?|MLLMs?|VLMs?)\b")
FAMILY_TITLE_RE = re.compile(r"^(DeepSeek|Qwen|Llama|Kimi|GLM|MiniMax|Gemma|Phi|Mistral|Magistral|Devstral|Codestral|InternVL|InternLM|Intern-S|Hunyuan|HY-|ERNIE|Seed|Step|Nemotron|NVIDIA Nemotron|OLMo|Olmo|SmolLM|Falcon|Granite|EXAONE|Pangu|MiMo|Ling|Ring|LongCat|Apertus|Aya|Command|Baichuan|Yi-|Skywork|Ovis|gpt-oss|Gemini|Claude|GPT-|Grok|Mercury|LLaDA|Dream|Jamba|Zamba|Hymba|Llama-Nemotron|Trinity|Arcee|Motif|Kanana|K2|Marin|Tulu|OpenThinker|AReaL|Moxin|Megrez|MiniCPM|BlueLM|Xmodel|Youtu|dots|Tele|Sarvam|Hermes|Instella|AFM|Apriel|Pleias|SmolVLM|Molmo|Eagle|Cosmos|Emu)[\w\-\.]*(\s[\w\-\.]+){0,3}\s*(:|Technical Report|Model Card)")
NAMED_TITLE_RE = re.compile(r"^[A-Z][\w\-\.]*(?:[\s\-][\w\.\-]+){0,2}\s*:")
TOOL_LIKE = ("quantization/", "compression/knowledge-distillation", "compression/structured-pruning",
             "training/parameter-efficient-finetuning", "training/rl-for-reasoning", "reasoning/test-time-scaling",
             "serving-systems/edge-and-on-device", "serving-systems/_general", "training/pretraining-recipes-and-efficiency")
CORE_PREFIXES = ("attention/", "kv-cache/", "decoding/", "quantization/", "model-conversion/", "compression/", "mixture-of-experts/")


def classify(title, abstract):
    """Return (primary_folder, [secondary folders], scores) or (None, [], scores)."""
    text = title + " " + abstract
    strong = bool(LLM_RE.search(text))
    if not strong and not WEAK_REL_RE.search(text):
        return None, [], {}
    if not strong and OFFTOPIC_RE.search(text):
        return None, [], {}
    if OFFTOPIC_TITLE_RE.search(title) and not LLM_TITLE_RE.search(title):
        return None, [], {}
    sc = score(title, abstract)
    # require real evidence: a title hit or >= 2 abstract occurrences (>= 3 for topics that are
    # often only *used* as a tool, e.g. "we run a 4-bit model" or "fine-tuned with LoRA")
    sc = {k: v for k, v in sc.items()
          if EVIDENCE[k][0] or EVIDENCE[k][1] >= (3 if k.startswith(TOOL_LIKE) else 2)}
    if not strong:
        # keep only core efficiency topics with strong evidence
        sc = {k: v for k, v in sc.items() if k.startswith(CORE_PREFIXES) and v >= 6}

    # ---- rule-based special categories (override scores)
    is_dllm = bool(DLLM_RE.search(text))
    if is_dllm:
        if AR2DIFF_RE.search(text):
            sc["model-conversion/ar-to-diffusion"] = max(sc.values(), default=0) + 12
        elif DLLM_ACCEL_RE.search(title) or len(DLLM_ACCEL_RE.findall(abstract)) >= 3:
            sc["decoding/diffusion-llm-inference"] = sc.get("decoding/diffusion-language-models", 0) + 6
    # conversion papers are primarily about the conversion, whatever architecture they target
    if LINEARIZE_RE.search(text) and LINEAR_TARGET_RE.search(text) and not re.search(r"neural tangent|\bNTK\b", text, re.I):
        sc["model-conversion/transformer-to-linear-or-hybrid"] = max(max(sc.values(), default=0) + 2, 14)
    if ATTNCONV_RE.search(text):
        sc["model-conversion/attention-conversion"] = max(max(sc.values(), default=0) + 2, 14)

    # quantization: merge into a bit bucket
    q = sc.get("quantization/_general", 0)
    if q and QUANT_RE.search(text):
        if KV_TITLE_RE.search(title) or (sc.get("kv-cache/quantization", 0) >= 10 and
                                   not re.search(r"weight(-only| quantization| and activation)|W\d+A\d+", text, re.I)):
            sc["kv-cache/quantization"] = max(sc.get("kv-cache/quantization", 0), q) + 4
            sc.pop("quantization/_general", None)
        else:
            bucket = quant_bucket(title, abstract)
            lpt = sc.pop("quantization/low-precision-training", 0)
            sc.pop("quantization/_general", None)
            sc[bucket] = q + (lpt if bucket == "quantization/low-precision-training" else 0)
            if lpt and bucket != "quantization/low-precision-training":
                sc["quantization/low-precision-training"] = lpt

    # MoE: training-system papers
    if sc.get("mixture-of-experts/architecture-and-routing", 0) and re.search(r"\btrain", title, re.I) and \
            re.search(r"system|communication|throughput|efficien|scal", text, re.I):
        sc["mixture-of-experts/training"] = sc["mixture-of-experts/architecture-and-routing"] + 1
    moe = sc.get("mixture-of-experts/architecture-and-routing", 0)
    if moe:
        for k in ("mixture-of-experts/inference-and-serving", "mixture-of-experts/compression"):
            if k in sc:
                sc[k] += moe

    # KV-cache leaves inherit general KV evidence
    kvg = sc.get("kv-cache/_general", 0)
    if kvg:
        for k in list(sc):
            if k.startswith("kv-cache/") and k != "kv-cache/_general":
                sc[k] += kvg

    # general buckets are fallbacks: damp them
    for k in list(sc):
        if k.endswith("/_general"):
            sc[k] *= 0.6

    # model releases / technical reports win over the topic they happen to mention
    if strong and (TECH_TITLE_RE.search(title) or FAMILY_TITLE_RE.search(title) or
                   (RELEASE_RE.search(abstract) and NAMED_TITLE_RE.search(title))):
        sc["models-and-architectures/technical-reports"] = max(sc.values(), default=0) + 5

    if not sc:
        return None, [], sc
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    primary, top = ranked[0]
    if top < MIN_SCORE:
        return None, [], sc
    secondary = [k for k, v in ranked[1:6] if v >= SECONDARY_SCORE]
    return primary, secondary, sc
