#!/usr/bin/env python3
"""Build the categorized markdown knowledge base from harvested + enriched data.

Inputs  (env overridable):  CANDIDATES (data/candidates.jsonl), CURATION (data/curation.tsv),
                             ENRICH_DIR/{s2,hf}.json, DIGEST_DIR/<id>.json
When CURATION exists, only papers kept in manual review are published, under the reviewed category.
Outputs: papers/<category>/<leaf>/<id>-<slug>.md, README.md per folder, data/papers.csv
"""
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy import CATEGORIES, CAT_INDEX, classify, detect_bits  # noqa: E402
from fulltext import compute_from_sentence  # noqa: E402
from curate import CODES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.environ.get("CANDIDATES", os.path.join(ROOT, "data", "candidates.jsonl"))
CURATION = os.environ.get("CURATION", os.path.join(ROOT, "data", "curation.tsv"))
ENR = os.environ.get("ENRICH_DIR", os.path.join(ROOT, "data"))
DIG = os.environ.get("DIGEST_DIR", os.path.join(ROOT, "data", "digests"))
OUT = os.path.join(ROOT, "papers")
TODAY = dt.date(2026, 9, 25)

TOP_VENUES = r"\b(ICML|ICLR|NeurIPS|NIPS|ACL|EMNLP|NAACL|COLM|MLSys|OSDI|SOSP|NSDI|EuroSys|ASPLOS|ISCA|MICRO|HPCA|ATC|SC'?\d{2}|PPoPP|CVPR|ICCV|ECCV|AAAI|IJCAI|KDD|DAC|TMLR|JMLR|Nature|Science|FAST|VLDB|SIGMOD|ICCAD|FPGA|TPAMI)\b"
MAJORS = {
    "quantization": "Quantization",
    "kv-cache": "KV cache",
    "attention": "Attention & sequence mixers",
    "decoding": "Decoding: speculative, parallel, MTP, diffusion",
    "model-conversion": "Model conversion (AR→diffusion, linearization, upcycling, …)",
    "compression": "Compression: pruning, sparsity, low-rank, distillation",
    "mixture-of-experts": "Mixture of Experts",
    "serving-systems": "Serving systems, kernels & hardware",
    "training": "Training: optimizers, scaling, RL, data, PEFT, tokenization",
    "reasoning": "Reasoning efficiency & test-time compute",
    "context-compression": "Context & token compression",
    "models-and-architectures": "Models & architectures",
}
COMPUTE_FOCUS = ("quantization/", "model-conversion/", "decoding/diffusion-language-models",
                 "decoding/multi-token-prediction", "decoding/jacobi-and-parallel-decoding",
                 "decoding/speculative-decoding", "compression/", "mixture-of-experts/training",
                 "training/", "attention/hybrid-architectures", "attention/sparse-attention",
                 "attention/linear-attention", "reasoning/latent-and-looped", "models-and-architectures/")

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}


def parse_date(s, pid):
    if re.match(r"\d{4}-\d{2}-\d{2}$", s or ""):
        return dt.date.fromisoformat(s)
    m = re.match(r"(\d+) (\w+), (\d{4})", s or "")
    if m and m.group(2) in MONTHS:
        return dt.date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    yy, mm = int(pid[:2]), int(pid[2:4])
    return dt.date(2000 + yy, mm, 15)


def slug(t, n=70):
    s = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return s[:n].rstrip("-")


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------------------ extraction from abstract
CLAIM_RX = re.compile(r"(\d+(\.\d+)?\s*(×|x|X|times|%|-fold|fold)|\bperplexity\b|\bPPL\b|\bspeed-?up|\bthroughput|\blatency|\bmemory|\baccuracy|\boutperform|\bstate-of-the-art|\bSOTA\b|\blossless|\bnear-lossless|\bbits?\b)", re.I)
HW_RX = re.compile(r"\b(H100|H800|H200|H20|B200|GB200|A100|A800|A6000|RTX ?\d{4}|L40S?|MI300X?|MI250|TPU ?v\d\w?|Ascend ?910\w?|Jetson[\w ]*|Snapdragon[\w ]*|Apple M\d\w*|iPhone|Raspberry Pi|Gaudi ?\d)\b")
FREE_RX = re.compile(r"training-free|without (any )?(re)?training|no (re)?training|tuning-free|plug-and-play|post-training (quantization|pruning|compression)|\bPTQ\b|zero-shot (compression|pruning)|calibration-free|data-free|retraining-free|fine-tuning-free|drop-in", re.I)
TRAIN_RX = re.compile(r"\b(fine-tun|pre-?train|continued pre-?training|quantization-aware training|\bQAT\b|distill|trained from scratch|train\w* (a|the|our))", re.I)
GH_RX = re.compile(r"https?://(?:www\.)?github\.com/[\w\-\.]+/[\w\-\.]+|https?://huggingface\.co/[\w\-\.]+/[\w\-\.]+")
MODEL_RX = re.compile(r"\b(Llama[- ]?[234](?:\.\d)?(?:[- ]\d+B)?|LLaMA[- ]?[123]?(?:[- ]\d+B)?|Qwen[- ]?[23](?:\.\d)?(?:[- ]?(?:VL|Coder|Math|MoE))?(?:[- ]\d+(?:\.\d)?B)?|Mistral(?:[- ]\d+B)?|Mixtral(?:[- ]\d+x\d+B)?|DeepSeek[- ](?:V[23](?:\.\d)?|R1|Coder|MoE)|Gemma[- ]?[123]?(?:[- ]\d+B)?|Phi[- ][234](?:\.\d)?|OPT(?:[- ]\d+B)|GPT-(?:2|3|4o?|OSS)|gpt-oss|Falcon|OLMo[- ]?2?|Pythia|Vicuna|InternLM\d?|GLM[- ]?4(?:\.\d)?|Kimi[- ]?K2|LLaDA(?:[- ]\d\.\d)?|Dream[- ]\d+B|Mamba[- ]?2?|RWKV[- ]?\d?|Jamba|Nemotron[\w\-]*|LLaVA[\w\-\.]*|InternVL[\d\.]*)", re.I)


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if s.strip()]


RESULT_RX = re.compile(r"\b(achiev|outperform|improv|reduc|accelerat|speed|faster|up to|surpass|match|retain|preserv|recover|boost|yield|deliver|attain|enabl|lower|higher|saving|compress|lossless|only)\w*", re.I)
CONTEXT_RX = re.compile(r"\b(however|existing|prior|previous|typically|usually|commonly|popular|widely|remains? (a )?challeng)\w*", re.I)


def key_claims(abstract, k=4):
    out = []
    sents = split_sents(abstract)
    for i, s in enumerate(sents):
        n = len(CLAIM_RX.findall(s))
        has_num = bool(re.search(r"\d", s))
        if not (n and has_num):
            continue
        sc = n + 2 + 2 * len(RESULT_RX.findall(s)) - 3 * bool(CONTEXT_RX.search(s)) + (i / max(1, len(sents))) * 2
        if re.search(r"\b(we|our|ours)\b", s, re.I):
            sc += 1.5
        if sc > 3:
            out.append((sc, s))
    out.sort(key=lambda x: -x[0])
    keep = [s for _, s in out[:k]]
    order = {s: i for i, s in enumerate(split_sents(abstract))}
    return sorted(keep, key=lambda s: order.get(s, 0))


def venue_from(p, s2):
    txt = " ".join([p.get("comments", ""), p.get("journal_ref", "")])
    m = re.search(r"(accepted|published|appear|to appear|camera[- ]ready|oral|spotlight|poster)[^.;]{0,60}?" + TOP_VENUES + r"[^.;,)]{0,12}", txt, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip(), True
    m = re.search(TOP_VENUES + r"\s*'?\s*(20)?2[5-7]", txt)
    if m:
        return m.group(0).strip(), True
    if s2 and s2.get("venue") and "arxiv" not in s2["venue"].lower():
        v = s2["venue"]
        short = s2.get("venue_short") or ""
        top = bool(re.search(TOP_VENUES, v + " " + short)) or bool(re.search(
            r"Neural Information Processing|International Conference on Machine Learning|Learning Representations|Association for Computational Linguistics|Empirical Methods|Operating Systems|Computer Architecture|Machine Learning and Systems", v))
        return v + (f" ({short})" if short and short not in v else ""), top
    return "", False


def fmt_hours(h):
    if h is None:
        return "—"
    if h >= 1e6:
        return f"{h/1e6:.2f}M"
    if h >= 1e4:
        return f"{h/1e3:.0f}k"
    if h >= 1e3:
        return f"{h/1e3:.1f}k"
    if h >= 10:
        return f"{h:.0f}"
    if h >= 1:
        return f"{h:.1f}"
    return f"{h:.2f}"


def fmt_tokens(t):
    if not t:
        return "—"
    if t >= 1e12:
        return f"{t/1e12:.2g}T"
    if t >= 1e9:
        return f"{t/1e9:.3g}B"
    return f"{t/1e6:.3g}M"


def compute_summary(dig):
    """Return (min_h, max_h, [(h, gpu, snippet, parsed)]) from digest compute sentences.

    Snippets are re-parsed with the current parser; figures that describe other models
    ("pre-training DeepSeek-V3 required 2.66M GPU hours") are kept for display but
    excluded from min/max."""
    ests = []
    for c in (dig or {}).get("compute", []):
        for p in compute_from_sentence(c["snippet"]):
            h = p["h100_hours"]
            if h and 0.001 <= h <= 5e7:
                ests.append((h, p["gpu"], c["snippet"], p))
    own = [e[0] for e in ests if not e[3].get("cited")]
    if not own:
        return None, None, ests
    return min(own), max(own), ests


def impact(p):
    age = max(1.0, (TODAY - p["date"]).days / 30.4)
    cit, inf, up = p["citations"], p["influential"], p["upvotes"]
    rate = cit / (age ** 0.75)
    s = 2.2 * math.log1p(rate) + 1.0 * math.log1p(inf) + 1.1 * math.log1p(up) \
        + 0.5 * bool(p["code"]) + 0.7 * p["top_venue"] + 0.35 * math.log1p(p.get("github_stars", 0) / 50)
    return round(s, 2)


# ------------------------------------------------------------------ load
def load_curation():
    """Manual review decisions (tools/curate.py): id -> category folder, or None if dropped."""
    if not os.path.exists(CURATION):
        return {}
    dec = {}
    for line in open(CURATION):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            dec[parts[0]] = CODES[parts[2]] if parts[1] == "K" else None
    return dec


def load():
    s2 = load_json(os.path.join(ENR, "s2.json"))
    hf = load_json(os.path.join(ENR, "hf.json"))
    cur = load_curation()
    out = []
    for line in open(CANDIDATES):
        r = json.loads(line)
        pid = r["id"]
        primary, secondary, sc = r["primary"], r["secondary"], r["scores"]
        if cur:
            if not cur.get(pid):  # dropped in review, or never reviewed
                continue
            primary = cur[pid]
            secondary = [c for c in secondary if c != primary]
        s = s2.get(pid) or {}
        h = hf.get(pid) or {}
        dig = None
        dp = os.path.join(DIG, f"{pid}.json")
        if os.path.exists(dp):
            dig = json.load(open(dp))
            if dig.get("error"):
                dig = None
        venue, top = venue_from(r, s)
        code = ""
        m = GH_RX.search(r["abstract"] + " " + r.get("comments", ""))
        if m:
            code = m.group(0).rstrip(".")
        elif h.get("github"):
            code = h["github"]
        elif dig and dig.get("github"):
            code = dig["github"][0]
        text = r["title"] + " " + r["abstract"]
        p = dict(
            id=pid, title=r["title"], authors=r["authors"], abstract=r["abstract"],
            categories=r["categories"], date=parse_date(r.get("created", ""), pid), comments=r.get("comments", ""),
            journal_ref=r.get("journal_ref", ""), primary=primary, secondary=secondary, scores=sc,
            citations=s.get("citations", 0) or 0, influential=s.get("influential", 0) or 0,
            tldr=s.get("tldr", "") or h.get("ai_summary", ""), venue=venue, top_venue=top,
            upvotes=h.get("upvotes", 0) or 0, org=h.get("org", ""), code=code,
            github_stars=h.get("github_stars", 0) or 0, keywords=h.get("ai_keywords", []),
            dig=dig, queries=r.get("queries", []),
            training_free=bool(FREE_RX.search(text)), needs_training=bool(TRAIN_RX.search(text)),
            models=sorted({re.sub(r"\s+", "-", m) for m in MODEL_RX.findall(text)})[:10],
            hardware=sorted(set(HW_RX.findall(text)))[:6],
            bits=detect_bits(r["title"], r["abstract"]),
            claims=key_claims(r["abstract"]),
        )
        p["cmin"], p["cmax"], p["cests"] = compute_summary(dig)
        p["score"] = impact(p)
        out.append(p)
    return out


# ------------------------------------------------------------------ rendering
def rel(from_dir, to_path):
    return os.path.relpath(to_path, from_dir).replace(os.sep, "/")


def cat_title(key):
    return CAT_INDEX[key][1]


def paper_path(p):
    return os.path.join(OUT, p["primary"], f"{p['id']}-{slug(p['title'])}.md")


BIT_LABEL = {0.7: "<1", 1.0: "1", 1.58: "1.58 (ternary)", 2.0: "2", 3.0: "3", 4.0: "4", 4.1: "FP4", 6.0: "5–6", 8.0: "8"}


def md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ")


def authors_str(a, n=10):
    return ", ".join(a[:n]) + (f", … (+{len(a)-n})" if len(a) > n else "")


def render_paper(p, rank, total, by_id):
    d = os.path.dirname(paper_path(p))
    L = []
    L.append(f"# {p['title']}\n")
    L.append(f"**arXiv:** [{p['id']}](https://arxiv.org/abs/{p['id']}) · [PDF](https://arxiv.org/pdf/{p['id']}) · "
             f"[HTML](https://arxiv.org/html/{p['id']}) · **First submitted:** {p['date'].isoformat()}  ")
    L.append(f"**Category:** [{cat_title(p['primary'])}](README.md) — **rank #{rank} of {total}** by impact score  ")
    if p["secondary"]:
        links = [f"[{cat_title(s)}]({rel(d, os.path.join(OUT, s, 'README.md'))})" for s in p["secondary"]]
        L.append(f"**Also relevant to:** " + " · ".join(links) + "  ")
    L.append("")
    if p["tldr"]:
        L.append(f"> **TL;DR** — {p['tldr'].strip()}\n")

    L.append("## At a glance\n")
    L.append("| Field | Value |\n| --- | --- |")
    L.append(f"| Authors | {md_escape(authors_str(p['authors']))} |")
    if p["org"]:
        L.append(f"| Organisation (HF) | {md_escape(p['org'])} |")
    L.append(f"| arXiv categories | {', '.join(p['categories'])} |")
    if p["venue"]:
        L.append(f"| Venue | {md_escape(p['venue'])} |")
    if p["comments"]:
        L.append(f"| arXiv comments | {md_escape(p['comments'][:300])} |")
    L.append(f"| Citations (Semantic Scholar, {TODAY.isoformat()}) | {p['citations']} ({p['influential']} influential) |")
    L.append(f"| Hugging Face upvotes | {p['upvotes']} |")
    L.append(f"| Code | {p['code'] if p['code'] else '—'} |")
    tb, ab = p["bits"]
    if p["primary"].startswith(("quantization/", "kv-cache/quantization")) and (tb or ab):
        L.append(f"| Bit-widths mentioned | {', '.join(BIT_LABEL[b] for b in (tb or ab))}-bit |")
    if p["training_free"] or p["needs_training"]:
        tr = []
        if p["training_free"]:
            tr.append("training-free / post-training")
        if p["needs_training"]:
            tr.append("involves (re)training / fine-tuning")
        L.append(f"| Training requirement (from abstract) | {'; '.join(tr)} |")
    models = p["models"]
    if p["dig"] and p["dig"].get("models"):
        models, seen = [], set()
        for m, _ in p["dig"]["models"]:
            m = m.rstrip("-.")
            if len(m) < 3 or m.lower() in seen:
                continue
            seen.add(m.lower())
            models.append(m)
        models = models[:10]
    if models:
        L.append(f"| Models used | {md_escape(', '.join(models))} |")
    hw = list(p["hardware"])
    if p["dig"] and p["dig"].get("gpus"):
        hw = [f"{k} (×{v} mentions)" for k, v in sorted(p["dig"]["gpus"].items(), key=lambda kv: -kv[1])[:5]]
    if hw:
        L.append(f"| Hardware | {md_escape(', '.join(hw))} |")
    if p["cmin"] is not None:
        rng = fmt_hours(p["cmin"]) if p["cmin"] == p["cmax"] else f"{fmt_hours(p['cmin'])} – {fmt_hours(p['cmax'])}"
        L.append(f"| **Compute (est. H100-hours)** | **{rng}** (auto-extracted, see below) |")
    if p["dig"] and p["dig"].get("max_train_tokens"):
        L.append(f"| Largest token count mentioned | {fmt_tokens(p['dig']['max_train_tokens'])} tokens |")
    L.append(f"| Impact score | {p['score']} |")
    L.append("")

    if p["claims"]:
        L.append("## Key results (claimed in abstract)\n")
        for c in p["claims"]:
            L.append(f"- {c}")
        L.append("")

    L.append("## Abstract\n")
    L.append(p["abstract"] + "\n")

    dig = p["dig"]
    if dig and dig.get("contributions"):
        L.append("## Contributions (from the paper)\n")
        L.append(dig["contributions"] + "\n")

    if p["cests"] or (dig and dig.get("hardware_sentences")):
        L.append("## Compute & hardware (extracted from full text)\n")
        if p["cests"]:
            L.append("Estimates are normalised to H100-hours with the factors in "
                     f"[METHODOLOGY.md]({rel(d, os.path.join(ROOT, 'METHODOLOGY.md'))}). "
                     "Always check the quoted sentence: the parser can't tell which experiment a number belongs to.\n")
            L.append("| Est. H100-h | Parsed | Source sentence |\n| --- | --- | --- |")
            seen = set()
            for h, gpu, snip, pr in p["cests"][:8]:
                k = (round(h, 3), snip[:80])
                if k in seen:
                    continue
                seen.add(k)
                how = f"{pr['count']:g}× {gpu} × {pr['hours']:.3g} h" if pr.get("count") else f"{pr['hours']:.3g} {gpu} GPU-h"
                if pr.get("cited"):
                    how += " _(cited: another model, excluded from range)_"
                L.append(f"| {fmt_hours(h)} | {md_escape(how)} | {md_escape(snip[:420])} |")
            L.append("")
        if dig and dig.get("hardware_sentences"):
            L.append("Hardware setup mentions:\n")
            for s in dig["hardware_sentences"][:4]:
                L.append(f"- {s}")
            L.append("")

    if dig and dig.get("tables"):
        L.append("## Main results tables (from the paper)\n")
        for t in dig["tables"][:2]:
            if t["caption"]:
                L.append(f"**{md_escape(t['caption'][:400])}**\n")
            L.append(t["markdown"] + "\n")

    if dig and dig.get("conclusion"):
        L.append("## Conclusion (from the paper)\n")
        L.append(dig["conclusion"] + "\n")
    if dig and dig.get("limitations"):
        L.append("## Limitations (from the paper)\n")
        L.append(dig["limitations"] + "\n")
    if dig and dig.get("headings"):
        L.append("<details><summary>Paper outline</summary>\n")
        for h in dig["headings"]:
            if h.lower() not in ("abstract",):
                L.append(f"- {h}")
        L.append("\n</details>\n")
    if p["keywords"]:
        L.append(f"**Keywords:** {', '.join(p['keywords'][:12])}\n")
    L.append("## Reproduce / read\n")
    L.append("```sh\n"
             f"arxiv read {p['id']}            # full paper as Markdown (github.com/1jmdev/arxiv)\n"
             f"arxiv section {p['id']} \"method\"\n"
             f"arxiv bibtex {p['id']}\n"
             "```\n")
    return "\n".join(L)


def short_claim(p, n=220):
    s = p["tldr"] or (p["claims"][0] if p["claims"] else split_sents(p["abstract"])[0] if p["abstract"] else "")
    s = s.strip()
    return md_escape(s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …")


def render_leaf_readme(key, items, cross, by_id):
    d = os.path.join(OUT, key)
    _, title, desc, _ = CAT_INDEX[key]
    L = [f"# {title}\n", f"{desc}\n",
         f"**{len(items)} papers** (first submitted 2025-01-01 → {TODAY.isoformat()}), ranked by impact score "
         f"(see [METHODOLOGY.md]({rel(d, os.path.join(ROOT, 'METHODOLOGY.md'))})). "
         f"Up: [{MAJORS[key.split('/')[0]]}](../README.md) · [Index]({rel(d, os.path.join(ROOT, 'README.md'))})\n"]
    ov = os.path.join(ROOT, "overviews", key.split("/")[0] + ".md")
    if os.path.exists(ov):
        L.append(f"📖 Written overview of this area: [{rel(d, ov)}]({rel(d, ov)})\n")
    notes = category_notes(key, by_id)
    if notes:
        L.append("## 🔬 Analyst notes: hand ranking and verdict\n")
        L.append("_Written after reading the abstracts, and the full text where available, of this category's papers. "
                 "The hand ranking weighs technical merit and usefulness for a runtime or model builder, not just "
                 "citations. The automatic impact ranking follows below._\n")
        L.append(notes + "\n")
    L.append("## 🏆 Best of the best by impact score (top 10)\n")
    for i, p in enumerate(items[:10], 1):
        extras = []
        if p["venue"]:
            extras.append(p["venue"][:60])
        extras.append(f"{p['citations']} cites")
        if p["upvotes"]:
            extras.append(f"{p['upvotes']}▲ HF")
        if p["code"]:
            extras.append(f"[code]({p['code']})")
        if p["cmin"] is not None and key.startswith(COMPUTE_FOCUS):
            extras.append(f"~{fmt_hours(p['cmin'])}–{fmt_hours(p['cmax'])} H100-h" if p["cmin"] != p["cmax"] else f"~{fmt_hours(p['cmin'])} H100-h")
        L.append(f"{i}. **[{md_escape(p['title'])}]({rel(d, paper_path(p))})** ({p['date'].strftime('%Y-%m')}) — "
                 f"{short_claim(p)}  \n   _score {p['score']} · {' · '.join(extras)}_")
    L.append("")

    recent = [p for p in items if (TODAY - p["date"]).days <= 90 and p not in items[:10]]
    recent.sort(key=lambda p: (-(p["upvotes"] + 3 * p["citations"] + 5 * p["top_venue"] + 2 * bool(p["code"])), p["id"]))
    if recent:
        L.append("## 🆕 Recent papers to watch (last 90 days)\n")
        L.append("Citations lag, so new work is under-ranked above. These are the most-upvoted or most-cited papers from the "
                 "last three months.\n")
        for p in recent[:8]:
            L.append(f"- **[{md_escape(p['title'])}]({rel(d, paper_path(p))})** ({p['date'].isoformat()}; "
                     f"{p['upvotes']}▲, {p['citations']} cites) — {short_claim(p, 180)}")
        L.append("")

    if key.startswith(COMPUTE_FOCUS):
        comp = [p for p in items if p["cmin"] is not None]
        if comp:
            L.append("## 💻 Compute cost (estimated H100-hours, auto-extracted)\n")
            L.append("Sorted cheapest first by the *smallest* compute figure found in the paper. Ranges cover all figures "
                     "the parser found (e.g. per-model-size runs). Verify against the quoted sentence in each paper file.\n")
            L.append("| Paper | Min H100-h | Max H100-h | GPU seen | Evidence |\n| --- | ---: | ---: | --- | --- |")
            for p in sorted(comp, key=lambda p: p["cmin"])[:150]:
                h, gpu, snip, _ = min((e for e in p["cests"] if not e[3].get("cited")), key=lambda e: e[0])
                L.append(f"| [{md_escape(p['title'][:80])}]({rel(d, paper_path(p))}) | {fmt_hours(p['cmin'])} | {fmt_hours(p['cmax'])} | {gpu} | {md_escape(snip[:160])}… |")
            L.append("")

    if key.startswith("quantization/") or key == "kv-cache/quantization":
        L.append("## Bit-width map\n")
        buckets = defaultdict(list)
        for p in items:
            tb, ab = p["bits"]
            for b in (tb or ab):
                buckets[b].append(p)
        if buckets:
            L.append("| Bits | # papers | Top papers |\n| --- | ---: | --- |")
            for b in sorted(buckets):
                ps = buckets[b][:4]
                L.append(f"| {BIT_LABEL[b]} | {len(buckets[b])} | " + "; ".join(f"[{md_escape(x['title'][:50])}]({rel(d, paper_path(x))})" for x in ps) + " |")
            L.append("")

    L.append("## Full ranking\n")
    L.append("| # | Paper | Date | Score | Cites | HF▲ | Venue | Code | Headline |\n| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |")
    for i, p in enumerate(items, 1):
        L.append(f"| {i} | [{md_escape(p['title'][:110])}]({rel(d, paper_path(p))}) | {p['date'].strftime('%Y-%m-%d')} | {p['score']} | "
                 f"{p['citations']} | {p['upvotes']} | {md_escape(p['venue'][:40]) if p['venue'] else ''} | {'[✓](' + p['code'] + ')' if p['code'] else ''} | {short_claim(p, 160)} |")
    L.append("")
    if cross:
        L.append("## Also relevant (primary category elsewhere)\n")
        L.append("| Paper | Primary category | Score |\n| --- | --- | ---: |")
        for p in cross[:60]:
            L.append(f"| [{md_escape(p['title'][:110])}]({rel(d, paper_path(p))}) | {cat_title(p['primary'])} | {p['score']} |")
        L.append("")
    return "\n".join(L)


def render_major_readme(major, leaves, cat_items):
    d = os.path.join(OUT, major)
    total = sum(len(cat_items[k]) for k in leaves)
    L = [f"# {MAJORS[major]}\n", f"**{total} papers** across {len(leaves)} sub-categories. [Back to index](../../README.md)\n"]
    ov = os.path.join(ROOT, "overviews", major + ".md")
    if os.path.exists(ov):
        L.append(f"📖 **Read first:** [written overview & recommendations]({rel(d, ov)})\n")
    L.append("| Sub-category | Papers | #1 paper |\n| --- | ---: | --- |")
    for k in leaves:
        it = cat_items[k]
        top = f"[{md_escape(it[0]['title'][:90])}]({rel(d, paper_path(it[0]))})" if it else "—"
        L.append(f"| [{cat_title(k)}]({k.split('/', 1)[1]}/README.md) | {len(it)} | {top} |")
    L.append("")
    allp = sorted([p for k in leaves for p in cat_items[k]], key=lambda p: -p["score"])
    L.append("## Top 25 across the whole area\n")
    for i, p in enumerate(allp[:25], 1):
        L.append(f"{i}. [{md_escape(p['title'])}]({rel(d, paper_path(p))}) — _{cat_title(p['primary'])}_ · score {p['score']} · {p['citations']} cites · {p['upvotes']}▲  \n   {short_claim(p, 200)}")
    L.append("")
    return "\n".join(L)


def render_root_readme(papers, cat_items, majors):
    L = ["# LLM Research Atlas — arXiv 2025 → 2026\n",
         f"A categorised, ranked map of **{len(papers):,} arXiv papers** on LLM inference, runtime optimisation and "
         f"model building, all first submitted between **2025-01-01 and {TODAY.isoformat()}**. Each paper has its own "
         "Markdown file with abstract, key claimed results, venue, citations, code link, models and hardware used, and where "
         "available: **compute cost in estimated H100-hours**, results tables, contributions, conclusion and limitations "
         "taken from the full text. Every paper was kept or dropped by hand from ~21K keyword-matched candidates, plus a recall "
         "audit that rescued popular papers the keyword filter missed (see METHODOLOGY.md).\n",
         "Built for engineers writing an LLM runtime (quantization, KV cache, attention kernels, speculative/parallel/"
         "diffusion decoding, serving) and for people training or converting models.\n",
         "* 📖 **Start with the [overviews](overviews/)**: hand-written syntheses of what is state of the art in each area and what to "
         "implement first.",
         f"* 🔬 Every one of the {sum(len(v) for v in majors.values())} category READMEs opens with **analyst notes**. These are hand-written: a verdict, a "
         "**hand ranking** of the papers that matter, hand-checked H100-hour cost tables where relevant, and concrete "
         "recommendations for runtime builders and model builders.",
         "* 🏆 Below the notes, each category README has an automatic **Best of the best** top-10 by impact score and a "
         "full ranking table.",
         "* 💻 Compute-heavy categories (quantization, conversion, diffusion LMs, MTP, training, …) have a **compute-cost table** "
         "sorted by H100-hours.",
         "* 🧮 [`data/papers.csv`](data/papers.csv) has every paper with category, rank, score, citations, bits and compute, ready "
         "for pandas or a spreadsheet.",
         "* ⚙️ [METHODOLOGY.md](METHODOLOGY.md) explains how papers were collected, classified, ranked and how compute was "
         "estimated. Read it before trusting a number.\n",
         "## Categories\n"]
    for m, leaves in majors.items():
        n = sum(len(cat_items[k]) for k in leaves)
        ov = os.path.join(ROOT, "overviews", m + ".md")
        ovl = f" · [overview](overviews/{m}.md)" if os.path.exists(ov) else ""
        L.append(f"### [{MAJORS[m]}](papers/{m}/README.md) — {n:,} papers{ovl}\n")
        L.append("| Sub-category | Papers | #1 by impact | #2 | #3 |\n| --- | ---: | --- | --- | --- |")
        for k in leaves:
            it = cat_items[k]
            tops = [f"[{md_escape(p['title'][:60])}{'…' if len(p['title']) > 60 else ''}]({rel(ROOT, paper_path(p))})" for p in it[:3]]
            tops += ["—"] * (3 - len(tops))
            L.append(f"| [{cat_title(k)}](papers/{k}/README.md) | {len(it)} | " + " | ".join(tops) + " |")
        L.append("")
    allp = sorted(papers, key=lambda p: -p["score"])
    L.append("## Top 50 papers overall (impact score)\n")
    L.append("| # | Paper | Category | Cites | HF▲ |\n| ---: | --- | --- | ---: | ---: |")
    for i, p in enumerate(allp[:50], 1):
        L.append(f"| {i} | [{md_escape(p['title'][:100])}]({rel(ROOT, paper_path(p))}) | {cat_title(p['primary'])} | {p['citations']} | {p['upvotes']} |")
    L.append("")
    L.append("## Repository layout\n")
    L.append("```\npapers/<area>/<sub-category>/README.md      ranked list, best-of, compute table\n"
             "papers/<area>/<sub-category>/<arxiv-id>-<slug>.md   one file per paper\n"
             "overviews/<area>.md                         hand-written state-of-the-art summaries\n"
             "data/papers.csv                             machine-readable index\n"
             "tools/                                      the pipeline that builds all of the above\n```\n")
    L.append("Paper metadata © their authors, obtained through arXiv's OAI-PMH interface. Citation counts from Semantic Scholar, "
             "upvotes from Hugging Face Daily Papers. Full text was read with the "
             "[`arxiv` CLI](https://github.com/1jmdev/arxiv). Thank you to arXiv for use of its open access interoperability.\n")
    return "\n".join(L)


REF_RX = re.compile(r"\[\[(\d{4}\.\d{4,5})(\\?\|[^\]]*)?\]\]")


MISSING_REFS = set()


def resolve_refs(text, from_dir, by_id):
    """Turn [[arxiv-id]] / [[id|label]] into links to the per-paper files (or arXiv if not in the corpus)."""
    def sub(m):
        pid, label = m.group(1), (m.group(2) or "").lstrip("\\")[1:]
        p = by_id.get(pid)
        if not p:
            MISSING_REFS.add(pid)
            return f"[{label or 'arXiv:' + pid}](https://arxiv.org/abs/{pid})"
        name = label or re.split(r"[:—]", p["title"])[0].strip()
        return f"[{name}]({rel(from_dir, paper_path(p))})"
    return REF_RX.sub(sub, text)


def category_notes(key, by_id):
    """Hand-written analyst notes for one sub-category: overviews_src/categories/<area>/<leaf>.md"""
    fn = os.path.join(ROOT, "overviews_src", "categories", key + ".md")
    if not os.path.exists(fn):
        return ""
    return resolve_refs(open(fn).read().strip(), os.path.join(OUT, key), by_id)


def render_overviews(by_id):
    """overviews_src/<area>.md -> overviews/<area>.md, resolving [[arxiv-id]] / [[id|label]] links."""
    src = os.path.join(ROOT, "overviews_src")
    dst = os.path.join(ROOT, "overviews")
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(dst, fn), "w") as f:
            f.write(resolve_refs(open(os.path.join(src, fn)).read(), dst, by_id))


def main():
    papers = load()
    by_id = {p["id"]: p for p in papers}
    cat_items = defaultdict(list)
    cross = defaultdict(list)
    for p in papers:
        cat_items[p["primary"]].append(p)
        for s in p["secondary"]:
            cross[s].append(p)
    for k in cat_items:
        cat_items[k].sort(key=lambda p: (-p["score"], -p["citations"], p["id"]))
    for k in cross:
        cross[k].sort(key=lambda p: -p["score"])

    render_overviews(by_id)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for key in CAT_INDEX:
        items = cat_items.get(key, [])
        if not items:
            continue
        os.makedirs(os.path.join(OUT, key), exist_ok=True)
        for i, p in enumerate(items, 1):
            p["rank"] = i
            with open(paper_path(p), "w") as f:
                f.write(render_paper(p, i, len(items), by_id))
        with open(os.path.join(OUT, key, "README.md"), "w") as f:
            f.write(render_leaf_readme(key, items, cross.get(key, []), by_id))
    majors = defaultdict(list)
    for key in CAT_INDEX:
        if cat_items.get(key):
            majors[key.split("/")[0]].append(key)
    for m, leaves in majors.items():
        with open(os.path.join(OUT, m, "README.md"), "w") as f:
            f.write(render_major_readme(m, leaves, cat_items))
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(render_root_readme(papers, cat_items, majors))

    # index csv + json for programmatic use
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "papers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "date", "primary_category", "secondary_categories", "rank_in_category", "impact_score",
                    "citations", "influential_citations", "hf_upvotes", "venue", "code", "bits", "compute_h100_hours_min",
                    "compute_h100_hours_max", "path"])
        for p in sorted(papers, key=lambda p: (p["primary"], p["rank"])):
            tb, ab = p["bits"]
            w.writerow([p["id"], p["title"], p["date"].isoformat(), p["primary"], ";".join(p["secondary"]), p["rank"],
                        p["score"], p["citations"], p["influential"], p["upvotes"], p["venue"], p["code"],
                        ";".join(BIT_LABEL[b] for b in (tb or ab)),
                        "" if p["cmin"] is None else round(p["cmin"], 3), "" if p["cmax"] is None else round(p["cmax"], 3),
                        rel(ROOT, paper_path(p))])
    stats = dict(total=len(papers), categories={k: len(v) for k, v in sorted(cat_items.items())},
                 with_fulltext=sum(1 for p in papers if p["dig"]), with_compute=sum(1 for p in papers if p["cmin"] is not None))
    json.dump(stats, open(os.path.join(ROOT, "data", "stats.json"), "w"), indent=1)
    json.dump({k: [dict(id=p["id"], title=p["title"], score=p["score"], cites=p["citations"], up=p["upvotes"],
                        tldr=p["tldr"], claims=p["claims"][:3], venue=p["venue"], date=p["date"].isoformat(),
                        cmin=p["cmin"], cmax=p["cmax"], path=rel(ROOT, paper_path(p)))
                   for p in v[:40]] for k, v in cat_items.items()},
              open(os.environ.get("TOP_JSON", os.path.join(ROOT, "data", "top_by_category.json")), "w"), indent=0, ensure_ascii=False)
    print(json.dumps(stats, indent=1))
    if MISSING_REFS:
        print("refs not in the corpus (linked to arXiv):", sorted(MISSING_REFS))


if __name__ == "__main__":
    main()
