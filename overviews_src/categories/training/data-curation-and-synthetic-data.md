**Verdict.** Data is still the highest-leverage knob (see [[2502.12120|LLMs on the Line]]). The 2025–26 lessons are:

1. **Pretraining data: model-based selection + mixture search + rephrasing.**
   * *Selection*: [[2503.00808|PreSelect]] keeps data whose loss predicts downstream ability, beating DCLM/FineWeb-Edu
     filters at 3B/100B. [[2602.05400|OPUS]] does per-iteration selection at 4.7% overhead. [[2502.10341|Organize the
     Web]] builds topic×format domains.
   * *Mixtures*: [[2504.13161|Nemotron-CLIMB]] (cluster + iterative proxy search), mixture laws (see
     [`scaling-laws`](../scaling-laws/README.md)).
   * *Rephrasing/synthetic*: [[2508.10975|BeyondWeb]] (DatologyAI) beats Cosmopedia and Nemotron-Synth by up to 5.1 pp
     and trains up to 7.7× faster; [[2506.04689|Recycling the Web]], [[2510.10681|RePro]],
     [[2507.03253|RefineX]]. Synthetic data follows its own scaling laws ([[2503.19551]]).
   * *Small-scale prediction*: [[2504.11393|DataDecide]]: rankings at 150M predict 1B about 80% of the time; continuous
     likelihood proxies make benchmarks >80% predictable at 0.01% of the compute.
   * *Open corpora*: [[2506.20920|FineWeb2]] (multilingual pipeline), [[2506.05209|Common Pile]] (8 TB openly
     licensed), [[2504.02807|MegaMath]], [[2502.18443|olmOCR]] (PDF → trillions of tokens),
     [[2506.08300|Institutional Books]].
2. **Post-training data: small and hard beats large.**
   * [[2502.03387|LIMO]] (COLM'25): ~800 curated examples (1% of prior data) give 63.3% AIME'24. Also s1's 1K ([[2501.19393]]).
   * [[2602.11149|Data repetition beats data scaling]]: many epochs on a small long-CoT set beat more data.
   * **Distillation source matters** ([[2505.14464]]).
   * Automated instruction selection often loses to random at scale ([[2503.01807]]).
   * Diversity measured by gradients predicts OOD generalization: [[2505.20161|Prismatic Synthesis / G-Vendi]].
3. **RL data at pretraining scale** (the new frontier):
   * [[2510.06499|Webscale-RL]]: 1.2M verifiable QA pairs from pretraining documents; RL matches continual pretraining
     with up to 100× fewer tokens.
   * [[2601.22975|Golden Goose]]: unlimited RLVR tasks from unverifiable text via masked-span multiple choice.
   * Procedural environments: [[2505.24760|Reasoning Gym]].
   * Verified datasets: [[2504.11456|DeepMath-103K]], [[2502.17387|Big-Math]], [[2503.02951|KodCode]],
     [[2504.01943|OpenCodeReasoning]].
4. **Safety through data.** [[2508.06601|Deep Ignorance]]: filtering biothreat-proxy content from pretraining gives
   tamper-resistant safeguards that survive fine-tuning attacks. Token-level filtering: [[2601.21571]].

### Hand ranking

| # | Paper | Stage | Key idea | Result |
| ---: | --- | --- | --- | --- |
| 1 | [[2508.10975\|BeyondWeb]] | Pretraining synthetic | Targeted rephrasing of web data at trillion scale (format, style, diversity jointly tuned) | +5.1 pp over Cosmopedia, +2.6 over Nemotron-Synth; up to 7.7× faster training |
| 2 | [[2504.13161\|Nemotron-CLIMB]] | Mixture | Embed + cluster the corpus; iterative proxy-model + predictor search over cluster weights | 1B model beats Llama-3.2-1B by 2.0% on 400B tokens; ClimbMix released |
| 3 | [[2502.03387\|LIMO]] (COLM'25) | SFT data | ~800 carefully chosen reasoning demonstrations as "cognitive templates" | 63.3% AIME'24, 95.6% MATH500; +45.8% absolute OOD vs models trained on 100× more data |
| 4 | [[2503.00808\|PreSelect]] | Selection | Score documents by how predictive their loss is of downstream ability (fastText scorer) | Beats DCLM and FineWeb-Edu filters at 3B / 100B tokens |
| 5 | [[2510.06499\|Webscale-RL]] | RL data | Convert pretraining documents to verifiable QA for RL | Matches continual pretraining with up to 100× fewer tokens |
| 6 | [[2504.11393\|DataDecide]] | Methodology | 25 corpora × 14 sizes × 3 seeds | Small-scale rankings predict 1B ~80%; continuous metrics >80% predictable at 0.01% compute |
| 7 | [[2602.11149\|Data repetition beats data scaling (long-CoT SFT)]] | SFT recipe | Many epochs on fewer examples, stop by token accuracy | Better than scaling unique data at equal compute |
| 8 | [[2505.24760\|Reasoning Gym]] (NeurIPS'25) | RL environments | 100+ procedurally generated verifiable tasks with adjustable difficulty | Unlimited curriculum data for RLVR |
| 9 | [[2506.20920\|FineWeb2]] | Multilingual pretraining | Language-adaptive filtering and dedup pipeline for 1,000+ languages | Better non-English corpora than prior pipelines |
| 10 | [[2508.06601\|Deep Ignorance]] | Safety | Multi-stage filtering of dual-use knowledge in pretraining | 6.9B models resist up to 10K steps / 300M tokens of adversarial fine-tuning; knowledge still usable in-context |
| 11 | [[2505.20161\|Prismatic Synthesis]] | Diversity | G-Vendi gradient-entropy diversity metric + targeted synthesis | Diversity predicts OOD reasoning; beats larger datasets |
| 12 | [[2601.22975\|Golden Goose]] | RL data | Mask key spans of reasoning-rich text → multiple-choice RLVR tasks | Scales RLVR to unverifiable domains (e.g. cybersecurity) |

**Also useful.**
* Instruction data: [[2506.11116|Infinity Instruct]], [[2501.12273|Condor]], [[2502.01968|Token Cleaning]],
  [[2502.09650]] (difficult examples can hurt alignment).
* Reasoning distillation datasets: [[2503.19633]] (AM-1.4M), [[2509.03059|Loong]].
* Crawling for LLMs: [[2502.13347|Craw4LLM]].
* Data quality via gradients: [[2504.10766]].
* Data-mixing frameworks: [[2603.26164|DataFlex]]; benchmark of post-training data: [[2512.14051|OpenDataArena]].
* Multilingual selection: [[2502.10361]].
* Survey: [[2503.14023]].

**Recommendation.**
* **Pretraining:** FineWeb/DCLM-style base + model-based selection (PreSelect/OPUS) + a mixture fitted with
  CLIMB-style search + 20–40% well-designed rephrased synthetic data. Validate choices at 150M–1B with DataDecide-style
  continuous metrics.
* **Post-training:** a few thousand hard, diverse, verified examples trained for multiple epochs.
* **RL data:** procedural generators + Webscale-RL-style conversion from pretraining text.
