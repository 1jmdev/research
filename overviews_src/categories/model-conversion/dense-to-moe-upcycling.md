**Verdict.** There are two different jobs under "dense → MoE", and they should not be confused.

1. **Sparse upcycling for capacity** (copy the FFN into E experts, add a router, keep pretraining). This buys a
   **head start**, but the advantage **shrinks as the budget grows**. [[2502.03009|Scaling laws for upcycling]] (ICML'25)
   finds an interaction term between dense and upcycled tokens that caps upcycling efficiency at large budgets.
   [[2502.19261|Drop-Upcycling]] (ICLR'25) shows naive copies specialize slowly. **Partially re-initializing expert
   weights** (drop ratio ~0.5) fixes long-run training: 5.9B active matches a 13B dense model at ~1/4 of the training
   FLOPs. Newer ideas:
   * [[2604.19835|Expert Upcycling]]: grow the expert count *during* training with utility-based duplication, saving
     32% of GPU hours at 7→13B total;
   * [[2603.13364|FineRMoE]]: finer-grained experts in two dimensions;
   * [[2509.18542|Symphony-MoE]]: experts from *different* pretrained models;
   * [[2510.01185|Dirichlet-prior routing loss]]: fixes weak specialization.
2. **MoEfication for inference speed** (partition an existing FFN into experts so each token activates fewer neurons).
   This is closer to activation sparsity or structured pruning:
   * [[2502.04416|analytical FFN→MoE]] (ACL'26): 4.5 minutes of analysis + 2K-sample fine-tune;
   * [[2501.15316|ToMoE]]: dynamic structural pruning;
   * [[2506.09351|DIVE]] (ACL'25): diversity-aware reconstruction, 1–5B tokens;
   * [[2602.15521|ExpertWeaver]]: GLU activation patterns;
   * [[2606.01666|DOT-MoE]] (ICML'26): optimal-transport partitioning.

   Real speedups need a grouped-GEMM MoE kernel and moderate top-k.

**Rule of thumb from the scaling-law work.** Upcycle when your remaining budget is **small relative to the dense
model's pretraining tokens**. With a budget comparable to pretraining, a from-scratch MoE (or Drop-Upcycling) wins.

### Cost table

| Method | Job | Budget | H100-h | Basis |
| --- | --- | --- | ---: | --- |
| [[2502.04416\|Analytical FFN→MoE]] | MoEfication (Llama-2-7B) | 4M tokens; 46 min end-to-end | **~1** | reported |
| [[2506.09351\|DIVE]] | MoE reconstruction (Llama-2-7B-class) | 0.5B router + 1–5B sparse retraining | **~25–150** | estimate |
| [[2502.19261\|Drop-Upcycling]] | 8×3.7B MoE (5.9B active) from dense | 500B tokens | **~12–20K** per model (study total >200K) | estimate; paper reports >200K H100-h for all experiments |
| [[2604.19835\|Expert Upcycling]] | Grow 7B → 13B-total MoE mid-training | CPT | **−32%** GPU-h vs fixed-size MoE | reported |
| [[2604.25578\|Marco-MoE]] | Multilingual MoE (~5% active) upcycled from dense | 5.1T tokens | pretraining-scale | reported token count |

### Hand ranking

| # | Paper | Key idea | Result |
| ---: | --- | --- | --- |
| 1 | [[2502.19261\|Drop-Upcycling]] (ICLR'25) | Copy FFN weights into experts, then **re-initialize a random subset of dimensions** per expert | Beats naive upcycling and from-scratch at hundreds of billions of tokens; 5.9B-active ≈ 13B dense at ~¼ FLOPs; fully open logs/checkpoints |
| 2 | [[2502.03009\|Scaling laws for upcycling MoE]] (ICML'25) | Joint laws in dense tokens, upcycled tokens and configuration | Tells you **when** upcycling beats from-scratch; the benefit shrinks at large budgets |
| 3 | [[2604.19835\|Expert Upcycling]] | Increase expert count mid-training with gradient-utility-based duplication | Matches fixed-size MoE loss at **32% fewer GPU-hours** |
| 4 | [[2502.04416\|Analytical FFN-to-MoE restructuring]] (ACL'26) | Activation-frequency analysis → shared + routed experts + a router from neuron statistics | Minutes of compute; up to 1.17× speedup in compute-bound settings |
| 5 | [[2506.09351\|DIVE]] (ACL'25) | Domain-affinity mining → pruning-based expert reconstruction → efficient retraining | Beats LLaMA-MoE with 1–5B tokens |
| 6 | [[2501.15316\|ToMoE]] (TMLR) | Dense → MoE via dynamic structural pruning (experts found without fine-tuning) | Beats structured pruning at equal active params |
| 7 | [[2603.13364\|FineRMoE]] | Fine-grained experts along intermediate **and** output dims + bi-level sparse compute + upcycling path | Pushes past the fine-granularity ceiling |
| 8 | [[2509.18542\|Symphony-MoE]] (AAAI'26) | Build an MoE from **disparate** pretrained models (align, then route) | More expert diversity than single-source upcycling |
| 9 | [[2510.01185\|Dirichlet-Prior Shaping]] | Router regularizer matching a Dirichlet prior on routing probabilities | Sharper routing and specialization in upcycled MoEs |
| 10 | [[2604.25578\|Marco-MoE]] | Highly sparse (≈5% active) multilingual MoEs upcycled from dense, 5.1T tokens, open data | Best-in-class performance/compute in the class |

**Also useful.**
* MoEfication variants: [[2602.15521|ExpertWeaver]], [[2511.21089|MLPMoE]] (zero-shot), [[2606.01666|DOT-MoE]],
  [[2609.21672|L0-MoE]], [[2605.26496|Dense2MoE]] (on-device), [[2502.12325|DynaMoE]] (token-difficulty routing).
* Fine-tuning-time upcycling: [[2503.11144|MoLEx]], [[2506.12597|SIMoE]], [[2603.29765]] (training-free dynamic
  upcycling of experts).
* Other: [[2509.00679|Router Upcycling]], [[2507.18671|Innovator]] (science CPT without forgetting),
  [[2606.16456|SPRI]] (SVD-partitioned init for data-constrained upcycling).

**Recommendation.**
* *Model builders:* upcycle with Drop-Upcycling-style partial re-init, fine-grained experts and a shared expert. Budget
  ≥100B tokens so the experts actually specialize. If your budget approaches the dense pretraining size, train the MoE
  from scratch.
* *Runtime builders:* MoEfied dense models help only with a fast grouped-GEMM or expert-parallel path and small
  top-k. At batch size 1 they behave like activation sparsity, so see
  [`compression/activation-sparsity`](../../compression/activation-sparsity/README.md).
