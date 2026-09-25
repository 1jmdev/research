**Verdict.** 2025 was the year of **Muon** (orthogonalized momentum via Newton–Schulz).
[[2502.16982|Muon is Scalable]] (Moonshot) added weight decay and per-shape update-RMS matching, trained Moonlight
(16B-A3B, 5.7T tokens) at ~2× AdamW compute efficiency, and Kimi K2 followed (MuonClip). The careful benchmarks then
tempered the hype:

* [[2509.02046|Fantastic Pretraining Optimizers]] (Stanford/Marin): with *fair* per-optimizer tuning, the fastest
  optimizers are all **matrix-preconditioned** (Muon, SOAP, Kron). The speedup over AdamW **shrinks with scale**: 1.4×
  at 0.1B, 1.1× at 1.2B.
* [[2512.05620|HP transfer for matrix optimizers]]: with *correct* hyperparameter scaling (µP-style LR transfer +
  independent weight decay), the ~1.4× gain holds from 190M to 1.4B. Much of the "shrinking gain" is mis-scaled
  hyperparameters.
* [[2509.01440|Benchmarking optimizers for LLM pretraining]] (EPFL): per-scenario guidance across model size, batch
  size and duration.

**Refinements worth adopting:**
* [[2510.05491|NorMuon]]: Muon + neuron-wise second moment; +11% over Muon at 1.1B.
* [[2601.08393|Spectral Sphere Optimizer]]: µP-aligned; beats AdamW and Muon on dense, MoE and 200-layer models, with
  better router balance and bounded activations.
* [[2510.12402|Cautious Weight Decay]]: drop-in, no new hyperparameters.
* Weight decay drives LR transfer ([[2510.19093]], [[2606.16899|Hyperball]]).
* **Distributed Muon** at scale: [[2504.05295|Dion]] (low-rank orthonormalized updates with error feedback),
  [[2602.06079|Canzona]], [[2510.16981|MuonBP]].
* Muon for DiLoCo: [[2505.23725|MuLoCo]].

**Stability** (loss/gradient spikes are the practical killer): [[2501.06842|SPAM]] (ICLR'25, momentum reset on
spikes), [[2504.02507|ZClip]] (z-score adaptive clipping), [[2502.17055|GradientStabilizer]], [[2602.01734|MSign]].

**Surprises:**
* [[2507.07101|Small batch sizes]]: batch size 1 with vanilla SGD/Adam trains stably, is more hyperparameter-robust and
  is equal or better per FLOP. Scale β₂ by token half-life; avoid gradient accumulation.
* [[2505.21829|Adam's secret sauce]] (NeurIPS'25): β₁ = β₂ keeps near-optimal performance.
* [[2509.26030|Muon learns tail associations]] better than Adam on heavy-tailed data.

### Hand ranking

| # | Paper | Kind | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2502.16982\|Muon is Scalable for LLM Training]] (Moonshot) | Optimizer at scale | Muon + weight decay + per-parameter update-scale matching; distributed ZeRO-1 implementation | ~2× compute efficiency vs AdamW; Moonlight 16B-A3B on 5.7T tokens |
| 2 | [[2509.02046\|Fantastic Pretraining Optimizers]] | Benchmark | Rigorous per-optimizer tuning at 0.1–1.2B and multiple data ratios | Matrix preconditioners win; speedup 1.4× → 1.1× with scale under standard tuning |
| 3 | [[2512.05620\|HP transfer for matrix-preconditioned optimizers]] | Scaling | Correct LR/WD transfer rules for Muon/Shampoo/SOAP | Consistent ~1.4× over AdamW from 190M to 1.4B |
| 4 | [[2510.05491\|NorMuon]] | Optimizer | Orthogonalization + neuron-level adaptive LR (second moment per row) | +21.7% over Adam, +11.3% over Muon (1.1B); Muon-level memory |
| 5 | [[2601.08393\|Spectral Sphere Optimizer]] | Optimizer | Steepest descent on the spectral sphere; fully µP-aligned; Megatron implementation | Beats AdamW and Muon on dense 1.7B, MoE 8B-A1B and 200-layer DeepNet; bounded activations |
| 6 | [[2504.05295\|Dion]] (Microsoft) | Distributed | Low-rank orthonormalized updates with error feedback, compatible with sharding | Muon-quality updates with much lower wall-clock at scale |
| 7 | [[2501.06842\|SPAM]] (ICLR'25) | Stability | Momentum reset + spike-aware clipping | Beats Adam and memory-efficient optimizers; removes spike damage |
| 8 | [[2507.07101\|Small batch size training]] | Recipe | Scale Adam β₂ by token half-life; small batches are stable and efficient | Batch-size-1 training works; gradient accumulation is wasteful |
| 9 | [[2510.12402\|Cautious Weight Decay]] | Regularization | Apply decay only where it agrees with the update direction | Consistent loss gains for AdamW/Lion/Muon, zero new hyperparameters |
| 10 | [[2509.01440\|Benchmarking optimizers]] (EPFL) | Benchmark | 11+ optimizers across size, batch and duration | Practitioner guidance per regime |
| 11 | [[2510.19093\|Weight decay > µP for LR transfer]] | Theory/empirics | µP mostly acts as implicit warmup; independent weight decay is what transfers | Simplifies HP transfer recipes |
| 12 | [[2504.02507\|ZClip]] | Stability | Z-score anomaly detection on gradient norms for adaptive clipping | Prevents loss spikes without manual thresholds |

**Also useful.**
* Muon theory and variants: [[2503.12645]] (non-Euclidean trust region), [[2505.13416|Gluon]], [[2505.21799|PolarGrad]],
  [[2507.01598]] (critical batch size), [[2511.20626|ROOT]], [[2602.17080|AdaMuon-style]], [[2510.21800|MARS-M]].
* Muon with MLA and MoE: [[2509.24406]].
* Second-order: [[2510.09378]] (full Gauss-Newton upper bound), [[2506.07254|SPlus]].
* Memory-efficient: [[2502.17410|COSMOS]], [[2502.01586|SubTrack++]], [[2506.16659]].
* Update masking: [[2602.15322]]. Learnable multipliers: [[2601.04890]]. Symmetry principle: [[2605.18106]].
* Benchmark taxonomy: [[2607.04033|OmniOpt]].

**Recommendation for model builders.**
* Use Muon (or NorMuon/SSO) for 2D matrices and AdamW for embeddings, norms and heads.
* Use µP-style width scaling with **independent weight decay**.
* Add spike protection (ZClip/SPAM-style).
* Distribute Muon with a Dion/Canzona-style implementation.
* Tune per optimizer; don't transfer AdamW hyperparameters.
