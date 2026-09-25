# Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1. Harvest metadata of every arXiv CS paper since 2025-01-01 (OAI-PMH, arXiv's bulk-metadata endpoint) | `oai_harvest.py` | `data/raw/oai_cs.jsonl` |
| 2. Topic pre-filter (40 queries in `queries.py`, compiled to regexes by `prefilter.py`) + classification (`taxonomy.py`) | `candidates.py` | `data/candidates.jsonl` (committed gzipped as `data/candidates.jsonl.gz`) |
| 3. Impact signals: Semantic Scholar citations, Hugging Face Daily Papers upvotes/code | `enrich.py all` | `data/s2.json`, `data/hf.json` |
| 4. Choose papers for full-text reading (compute-critical categories first, then top-N per category) | `select_ids.py` | id list |
| 5. Full text via the [`arxiv` CLI](https://github.com/1jmdev/arxiv): compute → H100-hours, hardware, tables, conclusion | `fulltext.py ids.txt` | `data/digests/<id>.json` |
| 6. Render markdown knowledge base + overviews (`overviews_src/` → `overviews/`, resolving `[[arxiv-id]]` links) | `build.py` | `papers/**`, `overviews/**`, `README.md`, `data/papers.csv` |

```sh
cargo install --git https://github.com/1jmdev/arxiv --locked arxiv
pip install beautifulsoup4 lxml
python3 tools/oai_harvest.py --from 2025-01-01
python3 tools/candidates.py
python3 tools/enrich.py all
python3 tools/select_ids.py --top 25 > /tmp/ids.txt && python3 tools/fulltext.py /tmp/ids.txt
python3 tools/build.py
```

To extend coverage, add a query to `queries.py` and (optionally) a leaf category with patterns to `taxonomy.py`,
then re-run steps 2–6. Steps 3 and 5 are incremental (already-fetched papers are skipped).
