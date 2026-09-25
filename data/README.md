# Data files

| File | Contents |
| --- | --- |
| `papers.csv` | One row per paper: id, title, first-submission date, primary + secondary categories, rank in category, impact score, citations, influential citations, HF upvotes, venue, code URL, bit-widths (quantization), estimated compute min/max in H100-hours, path to the paper's `.md` |
| `candidates.jsonl.gz` | Full metadata (title, authors, abstract, categories, dates, comments, journal-ref) for every in-scope paper, with the topic queries it matched and the classifier's category scores |
| `s2.json` | Semantic Scholar: citations, influential citations, venue, TLDR (keyed by arXiv id) |
| `hf.json` | Hugging Face Daily Papers: upvotes, organisation, GitHub repo, AI summary/keywords (keyed by arXiv id) |
| `digests/<id>.json` | Full-text digest from the `arxiv` CLI: section outline, contributions, conclusion, limitations, compute sentences, hardware, token counts, models used, results tables |
| `top_by_category.json` | Top 40 papers per category with key fields, handy for scripts |
| `stats.json` | Counts per category and coverage of full-text / compute extraction |

```python
import pandas as pd
df = pd.read_csv("data/papers.csv")
# cheapest AR->diffusion conversions with an extracted compute figure
df[df.primary_category == "model-conversion/ar-to-diffusion"].dropna(subset=["compute_h100_hours_min"]) \
  .sort_values("compute_h100_hours_min")[["title", "compute_h100_hours_min", "path"]]
```
