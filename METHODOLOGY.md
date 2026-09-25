# Methodology

This page explains how the knowledge base was built. Read it before trusting the rankings or the compute numbers.

## 1. Collecting papers

* **Source:** arXiv's **OAI-PMH** interface (`oaipmh.arxiv.org`), which arXiv provides for bulk metadata
  harvesting, run by [`tools/oai_harvest.py`](tools/oai_harvest.py). It collects title, abstract, authors, categories,
  dates, comments and journal-ref for **every paper in the `cs` set** (cross-lists included) updated since
  2025-01-01. Only papers whose arXiv ID is `25xx.*` or `26xx.*` are kept, i.e. **first submitted in 2025 or later**.
  (arxiv.org/search is disallowed for bots by robots.txt, and the Atom API returned HTTP 406 from this environment,
  so neither was used for the final dataset.)
* **Topic filter:** 40 topic queries in arXiv search syntax ([`tools/queries.py`](tools/queries.py)) are compiled to
  regular expressions ([`tools/prefilter.py`](tools/prefilter.py)) and matched locally against title + abstract.
  Because this runs over the full CS set, there is no 10,000-results-per-query cap.
* **Scope:** 40 topic queries cover LLM inference and runtime efficiency (quantization, KV cache, attention, speculative,
  parallel and diffusion decoding, MoE, serving, kernels, hardware), model conversion (AR→diffusion,
  linearization, MHA→MLA, upcycling, merging), compression, and model building (optimizers, scaling laws, pre-training,
  RL for reasoning, data, tokenization, technical reports, small models). Topics outside this scope were not harvested,
  e.g. general LLM safety, agents, or RAG applications. To extend coverage, add a query to `tools/queries.py` and a leaf
  category to `tools/taxonomy.py`.
* Papers that only loosely match a query and have no LLM / language-model context are dropped by the relevance filter
  in `taxonomy.classify`.

## 2. Categorising

[`tools/taxonomy.py`](tools/taxonomy.py) holds a two-level taxonomy with about 75 leaf folders. Each leaf has weighted
regular expressions. A pattern that hits in the **title** counts 3×; a hit in the abstract counts 1×, plus a small bonus
for repeated hits. The highest-scoring leaf becomes the paper's folder. Other strong leaves are listed as
"Also relevant to" links, and the paper appears in the "Also relevant" table of those leaves.

Special rules:

* **Quantization bit buckets.** Bit-widths are detected from phrases such as `2-bit`, `W4A8`, `INT4`, `1.58`, `ternary`,
  `sub-1-bit`, `0.8 bits per weight`, `FP4`/`MXFP4`/`NVFP4`, `FP8`. Bit-widths in the title win over the abstract.
  The paper goes into the **lowest** bit bucket it targets: `sub-1-bit`, `1-bit-and-ternary`, `2-bit`, `3-bit`,
  `4-bit-integer`, `4-bit-floating-point`, `5-to-8-bit`, `mixed-precision`, `low-precision-training` or `_general`.
  KV-cache quantization goes to `kv-cache/quantization`.
* **Conversion.** Adapting AR models into diffusion LMs, linearizing transformers into linear/SSM/hybrid models,
  MHA/GQA→MLA conversion, and dense→MoE upcycling are matched by dedicated rules. These papers go to
  `model-conversion/…` even when they are also diffusion or attention papers.
* **Diffusion LMs.** Papers about accelerating dLLMs (caching, parallel unmasking, speedups) go to
  `decoding/diffusion-llm-inference`. Papers about models and training go to `decoding/diffusion-language-models`.

The keyword classifier only produces a first guess.

### 2b. Manual review and recall audit

* **Every candidate was reviewed by hand.** The reviewer (Claude) read the title and abstract of all 21,353 prefiltered
  candidates. Each paper was kept with a category code or dropped: **8,146 kept, 13,207 dropped**. The decisions,
  including the final category, are in [`data/curation.tsv`](data/curation.tsv) (the last line for an id wins). The
  keep/drop rules and codes are in [`data/CURATION-GUIDE.md`](data/CURATION-GUIDE.md). The hand-assigned category
  overrides the keyword classifier.
* **Recall audit.** The prefilter was checked against Hugging Face daily-papers popularity and against landmark papers
  in each area. About 1,900 popular titles that were missing from the candidate set were reviewed by hand. **130
  relevant papers were rescued** and categorised manually. About 12 more were added later because an analyst note needed
  them, e.g. *Distillation Scaling Laws* (2502.08606) and *Joint MoE Scaling Laws* (2502.05172). The rescued ids are in
  `curation.tsv`, and their candidate records carry the query tag `rescue`.
* The remaining error is mostly about **placement, not inclusion**. Borderline papers can reasonably live in two
  leaves; "Also relevant to" links cover that.

## 3. Ranking: the impact score

Each leaf README ranks its papers by an **impact score**:

```
age_months   = max(1, months since first submission)
cite_rate    = citations / age_months^0.75
score = 2.2·ln(1+cite_rate) + 1.0·ln(1+influential_citations) + 1.1·ln(1+HF_upvotes)
      + 0.5·[code released] + 0.7·[top venue] + 0.35·ln(1+github_stars/50)
```

* **Citations and influential citations** come from the Semantic Scholar batch API, as of 2026-09-25.
* **HF upvotes, organisation and GitHub repo** come from Hugging Face Daily Papers. Only papers featured there have
  upvotes.
* **Top venue** is detected from arXiv comments and journal refs (e.g. "Accepted to ICML 2026") or the Semantic
  Scholar venue: ICML, ICLR, NeurIPS, ACL, EMNLP, NAACL, COLM, MLSys, OSDI, SOSP, NSDI, EuroSys, ASPLOS, ISCA, MICRO,
  HPCA, ATC, SC, PPoPP, CVPR, ICCV, ECCV, AAAI, IJCAI, KDD, DAC, TMLR, JMLR, Nature, Science and others.

The score measures **community impact**, not correctness or benchmark SOTA. Very recent papers (from the last 1–2
months) have not had time to collect citations, so they are under-ranked. Always read the "Key results" and results
tables of the top few papers yourself. The hand-written [overviews](overviews/) give an opinionated "what is actually
best" view per area.

## 4. Full-text extraction (compute, hardware, tables)

[`tools/fulltext.py`](tools/fulltext.py) uses the [`arxiv` CLI](https://github.com/1jmdev/arxiv) (`arxiv read <id>
--format json`) to fetch each paper's full text. The CLI rate-limits itself to one request every 3 s. From the text it
extracts:

* **Compute sentences.** Sentences that mention a GPU/TPU/NPU and a duration or GPU-hours, for example
  "quantizing Llama-3-70B takes 4.5 hours on a single A100" or "0.13 million H800 GPU hours".
* **Hardware setup sentences**, **largest token count** mentioned, **models used** (frequency-ranked), **contributions**
  paragraph, **conclusion**, **limitations**, **section outline** and up to **two results tables**. Tables whose caption
  mentions results, accuracy, perplexity, speed or throughput are preferred.

### H100-hour normalisation

Each parsed figure is `count × hours × factor`, or `GPU-hours × factor`. The factor is roughly the device's dense
BF16 tensor throughput relative to an H100 SXM:

| Device | Factor | Device | Factor |
| --- | ---: | --- | ---: |
| H100 / H800 / GH200 | 1.0 | A100 / A800 | 0.32 |
| H200 | 1.1 | V100 | 0.13 |
| B200 / B100 | 2.2 | RTX 4090 | 0.17 |
| GB200 | 2.5 | RTX 5090 | 0.21 |
| H20 | 0.15 | RTX 3090 | 0.07 |
| MI300X / MI325X | 1.3 | A6000 | 0.16 |
| MI355X | 2.3 | L40S / RTX 6000 Ada | 0.37 |
| MI250 | 0.37 | L40 | 0.18 |
| TPU v6e | 0.93 | L4 | 0.12 |
| TPU v5p | 0.46 | A10 | 0.13 |
| TPU v5e | 0.2 | T4 | 0.07 |
| TPU v4 | 0.28 | Ascend 910B | 0.35 |
| Gaudi 3 | 1.9 | Gaudi 2 | 0.43 |

Caveats:

* Memory-bound workloads, such as most quantization calibration and decoding, run relatively faster on older GPUs than
  these FLOP ratios suggest. For those workloads, treat the A100→H100 conversion (×0.32) as a lower bound; ×0.5 is
  often more realistic.
* The parser cannot tell which experiment a sentence belongs to. One paper may report both "quantize 7B in 10 min" and
  "train 1.3B from scratch in 2k GPU-h", so every paper page lists **each** figure with its source sentence. The
  category tables show the min and max.
* A figure with an unspecified GPU is taken 1:1 as H100-hours.
* Papers without an HTML version fall back to PDF text, which is less reliable.

## 5. Analyst notes (hand ranking and verdicts)

Every leaf README opens with **🔬 Analyst notes**, written by hand from the papers rather than generated. They live in
[`overviews_src/categories/<area>/<leaf>.md`](overviews_src/categories/) and are rendered into the README by
`tools/build.py`. `[[arxiv-id]]` references are resolved into links to the paper pages. Each note has:

* a **verdict**: what is state of the art, what failed, and what the consensus is;
* a **hand ranking table**. The rank reflects usefulness for building a runtime or model, **not** the citation-based
  impact score, so it often differs from the automatic top-10 below it;
* an **"Also useful"** list for the long tail;
* **recommendations** for runtime builders and for model builders.

How the notes were written: for each leaf, the reviewer read the abstracts, contributions, conclusions and limitations
of the top ~24 papers by impact score plus the titles and snippets of the rest of the leaf. For the papers used in a
ranking or a cost table, the full text was fetched with the `arxiv` CLI to check claims and numbers. Headline numbers in
the tables come from each paper's abstract or text. They are the authors' claims, not independent reproductions.

**Compute in the notes is hand-checked.** The automatic compute parser (section 4) often picks up sentences that
describe *other* papers' compute, so the notes do not reuse its figures blindly. Conversion, quantization and
distillation costs were recomputed in one of two ways:

* **reported**: GPU count × wall-clock from the paper, normalised to H100 with the table above;
* **estimated**: 6·N·D FLOPs at 40% MFU of H100 BF16 dense (≈1.42·10¹⁸ FLOP per H100-hour), with N = active parameters
  and D = training tokens.

Every table says which basis it uses. Treat estimates as ±2×.

## 6. Reproducing / updating

See [`tools/README.md`](tools/README.md). In short:

```sh
cargo install --git https://github.com/1jmdev/arxiv --locked arxiv
pip install beautifulsoup4 lxml
python3 tools/oai_harvest.py --from 2025-01-01   # -> data/raw/oai_cs.jsonl   (~1 h, all CS papers)
python3 tools/candidates.py                      # -> data/candidates.jsonl  (filter + classify)
python3 tools/enrich.py all                      # -> data/s2.json, data/hf.json
python3 tools/select_ids.py > ids.txt
python3 tools/fulltext.py ids.txt                # -> data/digests/*.json    (CLI rate limit: 3 s/request)
python3 tools/build.py                           # -> papers/**, overviews/**, data/papers.csv
```

To add papers submitted after the snapshot, run `oai_harvest.py --from <last date>` (it appends new records), then
re-run the remaining steps. Enrichment and full-text steps skip papers they have already processed.
