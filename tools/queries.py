"""arXiv abstract-field search queries used to harvest candidate papers.

Each value is a list of terms that are ANDed together (each term may use OR and
quoted phrases, as supported by arxiv.org advanced search).
"""

LLM = ('LLM OR LLMs OR "language model" OR "language models" OR transformer OR '
       'transformers OR "foundation model" OR "foundation models"')

QUERIES = {
    # --- quantization / numerics ---
    "quantization": ['quantization OR quantized OR quantize OR "low-bit" OR binarization OR binarized OR ternary OR "1-bit" OR "2-bit" OR "4-bit"', LLM],
    "low_precision_formats": ['FP8 OR FP4 OR MXFP4 OR NVFP4 OR MXFP8 OR "microscaling" OR "low-precision training" OR "low precision training" OR INT4 OR INT8'],
    # --- KV cache ---
    "kv_cache": ['"KV cache" OR "key-value cache" OR "KV-cache" OR "KV caches" OR "prefix caching" OR "prefix cache"'],
    # --- attention & architectures ---
    "attention": ['"sparse attention" OR "linear attention" OR "efficient attention" OR FlashAttention OR "attention kernel" OR "multi-head latent attention" OR "grouped-query attention" OR "attention sink" OR "sliding window attention" OR "native sparse attention"'],
    "attention_general": ['attention', '"quadratic complexity" OR "quadratic cost" OR "subquadratic" OR "sub-quadratic" OR "attention mechanism"', LLM],
    "ssm_hybrid": ['"state space model" OR "state space models" OR "state-space model" OR Mamba OR Mamba2 OR RWKV OR "gated delta" OR DeltaNet OR "linear recurrent" OR "hybrid architecture" OR "hybrid model" OR "recurrent neural network"', LLM],
    "long_context": ['"long context" OR "long-context" OR "context extension" OR "context window" OR "length extrapolation" OR "length generalization" OR RoPE OR "rotary position" OR "positional encoding" OR "position embedding"', LLM],
    # --- decoding ---
    "speculative": ['"speculative decoding" OR "speculative sampling" OR "speculative inference" OR "draft model" OR "self-speculative" OR "draft tokens" OR "drafter"'],
    "parallel_decoding": ['"multi-token prediction" OR "multi-token" OR "parallel decoding" OR Jacobi OR "lookahead decoding" OR "non-autoregressive" OR "semi-autoregressive" OR "consistency model" OR "any-order"', LLM],
    "diffusion_lm": ['"diffusion language model" OR "diffusion language models" OR "masked diffusion" OR "discrete diffusion" OR dLLM OR dLLMs OR "diffusion LLM" OR "diffusion LLMs" OR "diffusion-based language" OR "block diffusion"'],
    "diffusion_lm_broad": ['"language diffusion" OR "diffusion model" OR "diffusion models" OR "masked diffusion" OR "denoising"', '"large language" OR LLM OR LLMs OR "autoregressive models" OR ARMs OR "text generation" OR "language modeling"'],
    "early_exit": ['"early exit" OR "early-exit" OR "early exiting" OR "layer skipping" OR "layer skip" OR "dynamic depth" OR "mixture of depths" OR "mixture-of-depths" OR "adaptive computation"', LLM],
    "decoding_misc": ['"constrained decoding" OR "structured output" OR "structured generation" OR "grammar-constrained" OR "decoding strategy" OR "decoding strategies" OR "min-p" OR "sampling method" OR "token sampling"', LLM],
    # --- compression ---
    "pruning": ['pruning OR pruned OR "N:M sparsity" OR "2:4 sparsity" OR "semi-structured sparsity" OR "structured sparsity" OR "activation sparsity" OR "contextual sparsity" OR "weight sparsity" OR "sparse LLM"', LLM],
    "low_rank": ['"low-rank" OR "low rank" OR SVD OR "singular value decomposition" OR "tensor decomposition" OR "model compression" OR "weight sharing"', 'compression OR compress OR inference OR efficient', LLM],
    "distillation": ['distillation OR distill OR distilled OR distilling', LLM],
    # --- MoE ---
    "moe": ['"mixture of experts" OR "mixture-of-experts" OR MoE OR "sparse experts" OR "expert routing" OR "expert parallelism"'],
    # --- conversion ---
    "conversion": ['upcycling OR upcycled OR linearization OR linearize OR linearizing OR linearized OR retrofit OR retrofitting OR "convert pretrained" OR "converting pretrained" OR "from a pretrained" OR "adapting autoregressive" OR "autoregressive to diffusion" OR "AR-to-diffusion" OR "cross-architecture" OR "architecture conversion" OR MHA2MLA OR TransMLA OR "distill into"', LLM],
    # --- systems ---
    "serving": ['serving OR "inference engine" OR "inference system" OR "inference framework" OR prefill OR "decode phase" OR "continuous batching" OR disaggregated OR disaggregation OR "request scheduling" OR "time-to-first-token" OR TTFT OR "tail latency" OR goodput', LLM],
    "kernels_hardware": ['"GPU kernel" OR "GPU kernels" OR CUDA OR Triton OR "kernel fusion" OR GEMM OR "tensor core" OR "tensor cores" OR FPGA OR ASIC OR NPU OR "hardware accelerator" OR "processing-in-memory" OR "compute-in-memory" OR TPU OR "roofline"', LLM],
    "edge": ['"on-device" OR "edge device" OR "edge devices" OR "edge deployment" OR smartphone OR "mobile device" OR "mobile devices" OR "consumer GPU" OR "consumer-grade" OR "CPU inference" OR "resource-constrained" OR "microcontroller"', LLM],
    "memory_offload": ['offloading OR offload OR "memory-efficient" OR "memory efficient" OR "memory bandwidth" OR "memory footprint" OR "memory wall" OR "HBM" OR "unified memory"', LLM, 'inference OR training OR serving'],
    "parallelism": ['"tensor parallelism" OR "pipeline parallelism" OR "sequence parallelism" OR "context parallelism" OR "data parallelism" OR "distributed training" OR "distributed inference" OR FSDP OR ZeRO OR "collective communication" OR "all-reduce" OR "all-to-all" OR Megatron'],
    "energy": ['"energy consumption" OR "energy efficiency" OR "energy-efficient" OR "power consumption" OR "carbon footprint" OR "joules"', LLM, 'inference OR training'],
    # --- training ---
    "optimizers": ['optimizer OR optimizers OR Muon OR AdamW OR "second-order" OR Shampoo OR SOAP OR "learning rate" OR "weight decay"', 'pretraining OR "pre-training" OR training', LLM],
    "scaling_laws": ['"scaling law" OR "scaling laws" OR "compute-optimal" OR "compute optimal"', LLM],
    "pretraining": ['pretraining OR "pre-training" OR pretrain', 'efficient OR efficiency OR "training cost" OR "GPU hours" OR "compute budget" OR "data mixture" OR "data selection" OR "curriculum" OR "stability"', LLM],
    "peft": ['LoRA OR QLoRA OR DoRA OR "parameter-efficient" OR "parameter efficient" OR "low-rank adaptation" OR "adapter tuning" OR "prefix tuning"', LLM],
    "rl_reasoning": ['GRPO OR RLVR OR "verifiable rewards" OR "policy optimization" OR PPO OR "reinforcement learning"', 'reasoning OR "large reasoning model" OR "reasoning models" OR LLM OR LLMs'],
    "data": ['"synthetic data" OR "data curation" OR "data selection" OR "data mixture" OR "data mixing" OR "pretraining data" OR "data filtering" OR "data quality"', LLM],
    "tokenization": ['tokenizer OR tokenizers OR tokenization OR "byte-level" OR "tokenizer-free" OR "vocabulary size" OR "vocabulary expansion" OR "byte latent"', LLM],
    "model_merging": ['"model merging" OR "merging models" OR "merge models" OR "weight averaging" OR "task arithmetic" OR "model soup" OR "model soups"'],
    "tech_reports": ['"technical report"', 'open-source OR "open-weight" OR "open weights" OR "open-sourced" OR release OR "we introduce" OR "we present"', LLM],
    "model_release": ['"technical report" OR "model card" OR "open-weight" OR "open-source" OR "open-sourced" OR "we release" OR "we open-source"', LLM, '"pre-training" OR pretraining OR "pre-trained on" OR "trillion tokens" OR "activated parameters" OR "active parameters" OR "total parameters"'],
    "architecture": ['"residual connection" OR "residual connections" OR "residual stream" OR "hyper-connection" OR "hyper-connections" OR normalization OR "architecture design" OR "architectural"', '"language model" OR "language models" OR LLM OR LLMs', 'pretraining OR "pre-training" OR scaling OR "training stability"'],
    "small_models": ['"small language model" OR "small language models" OR SLM OR SLMs OR "sub-billion" OR "tiny language"'],
    # --- reasoning efficiency / test-time ---
    "efficient_reasoning": ['"efficient reasoning" OR overthinking OR "reasoning length" OR "chain-of-thought compression" OR "CoT compression" OR "token budget" OR "concise reasoning" OR "adaptive thinking" OR "long-to-short" OR "reasoning efficiency" OR "thinking budget" OR "underthinking"'],
    "test_time_scaling": ['"test-time scaling" OR "test-time compute" OR "inference-time scaling" OR "inference-time compute" OR "best-of-N" OR "self-consistency" OR "parallel thinking" OR "parallel reasoning"'],
    "latent_reasoning": ['"latent reasoning" OR "continuous thought" OR "continuous latent" OR "latent chain-of-thought" OR "looped transformer" OR "looped transformers" OR "recurrent depth" OR "recurrent-depth" OR "implicit reasoning" OR "latent space reasoning"'],
    # --- context / token compression ---
    "context_compression": ['"prompt compression" OR "context compression" OR "token pruning" OR "token merging" OR "token compression" OR "visual token" OR "visual tokens" OR "token reduction" OR "gist tokens"', LLM + ' OR VLM OR VLMs OR MLLM OR MLLMs OR "vision-language"'],
}
