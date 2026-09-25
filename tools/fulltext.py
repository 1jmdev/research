#!/usr/bin/env python3
"""Fetch full text of papers with the `arxiv` CLI (github.com/1jmdev/arxiv) and
distil a compact digest per paper: compute used (normalised to H100-hours),
hardware, training tokens, contributions, conclusion, limitations, key tables.

Usage: python3 tools/fulltext.py ids.txt [--workers 4]
Digests are written to $DIGEST_DIR/<id>.json (default data/digests).
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGEST = os.environ.get("DIGEST_DIR", os.path.join(ROOT, "data", "digests"))
CACHE = os.path.expanduser("~/.cache/arxiv")

# H100-equivalence factors (≈ dense BF16 tensor throughput relative to H100 SXM).
# These are coarse; memory-bound workloads narrow the gap. Documented in METHODOLOGY.md.
GPU_FACTORS = [
    (r"GB200", 2.5, "GB200"), (r"B200|B100|Blackwell", 2.2, "B200"), (r"H200", 1.1, "H200"),
    (r"H800", 1.0, "H800"), (r"H100", 1.0, "H100"), (r"GH200", 1.0, "GH200"), (r"H20\b", 0.15, "H20"),
    (r"A800", 0.32, "A800"), (r"A100", 0.32, "A100"), (r"MI355X?", 2.3, "MI355X"), (r"MI325X?", 1.3, "MI325X"),
    (r"MI300X?|MI300A", 1.3, "MI300X"), (r"MI250X?", 0.37, "MI250"), (r"RTX\s?PRO\s?6000", 0.5, "RTX PRO 6000"),
    (r"RTX\s?6000\s?Ada|L40S", 0.37, "L40S/RTX6000Ada"), (r"L40\b", 0.18, "L40"), (r"RTX\s?5090", 0.21, "RTX 5090"),
    (r"RTX\s?4090", 0.17, "RTX 4090"), (r"RTX\s?3090", 0.07, "RTX 3090"), (r"RTX\s?A6000|\bA6000\b", 0.16, "A6000"),
    (r"\bA40\b", 0.15, "A40"), (r"\bA10G?\b", 0.13, "A10"), (r"\bL4\b", 0.12, "L4"), (r"V100", 0.13, "V100"),
    (r"\bT4\b", 0.07, "T4"), (r"TPU\s?v6e|Trillium", 0.93, "TPU v6e"), (r"TPU\s?v5p", 0.46, "TPU v5p"),
    (r"TPU\s?v5e|TPU\s?v5 lite", 0.2, "TPU v5e"), (r"TPU\s?v4", 0.28, "TPU v4"), (r"TPU\s?v3", 0.12, "TPU v3"),
    (r"Ascend\s?910[BC]?", 0.35, "Ascend 910"), (r"Gaudi\s?3", 1.9, "Gaudi 3"), (r"Gaudi\s?2", 0.43, "Gaudi 2"),
]
GPU_RX = [(re.compile(p), f, n) for p, f, n in GPU_FACTORS]
GPU_ANY = re.compile("|".join(p for p, _, _ in GPU_FACTORS) + r"|\bGPUs?\b|\bTPUs?\b|\bNPUs?\b")
NUMW = {"a": 1, "one": 1, "single": 1, "two": 2, "four": 4, "eight": 8, "sixteen": 16, "a single": 1}
TIME_RX = re.compile(r"(\d+(?:\.\d+)?|a|one|two|three|several)\s*(?:~\s*)?(hours?|hrs?|h\b|days?|minutes?|mins?|weeks?|seconds?|secs?|s\b)", re.I)
GPUH_RX = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k|K|thousand|M|million)?\s*(?:(?!tokens?\b|samples?\b|steps?\b|and\b|or\b)(?:NVIDIA|AMD|Nvidia|Google|[A-Za-z][\w\-]*)\s+){0,2}(?:GPU|H100|H800|A100|A800|H200|B200|TPU|NPU|chip|accelerator|node)[- ]?(hours?|hrs?|days?)\b", re.I)
OWN_RX = re.compile(r"\b(we|our|ours|us|this work|this paper|the proposed)\b", re.I)
CITED_RX = re.compile(r"(pre-?train\w*|train\w*)\s+(of\s+|the\s+)?(original\s+)?(DeepSeek|Llama|LLaMA|GPT|Qwen|Gemma|Mistral|OPT|Falcon|PaLM|Chinchilla|the (original|base|teacher) model)|(DeepSeek|Llama|LLaMA|GPT|Qwen|Gemma)[\w\-\.]*\s+(model\s+)?(required|requires|took|used|consumed|cost)", re.I)
COUNT_RX = re.compile(r"(\d+|a single|one|single|two|four|eight|sixteen)\s*(?:×|x|\*)?\s*(?:NVIDIA\s+|AMD\s+|Nvidia\s+|Google\s+|Huawei\s+|Intel\s+)?(?:Tesla\s+|GeForce\s+)?(" +
                      "|".join(p for p, _, _ in GPU_FACTORS) + r"|GPUs?|TPUs?|NPUs?)", re.I)
TOKENS_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(B|T|M|billion|trillion|million)\s+(?:training\s+|high-quality\s+|pre-?training\s+)?tokens", re.I)
MODEL_RX = re.compile(r"\b(Llama[- ]?[234](?:\.\d)?(?:[- ]\d+B)?|LLaMA[- ]?[123]?(?:[- ]\d+B)?|Qwen[- ]?[23](?:\.\d)?(?:[- ]?(?:VL|Coder|Math|MoE))?(?:[- ]\d+(?:\.\d)?B)?|Qwen3[- ]?Next|Mistral[- ]\d+B|Mixtral[- ]\d+x\d+B|DeepSeek[- ](?:V[23](?:\.\d)?|R1|Coder|MoE)(?:[- ]Distill)?|Gemma[- ]?[123]?(?:[- ]\d+B)?|Phi[- ][234](?:\.\d)?(?:[- ]mini)?|OPT[- ]\d+(?:\.\d)?B|GPT[- ]?(?:2|3|4o?|OSS|5)(?:[- ]\d+B)?|gpt-oss[- ]\d+b|Falcon[- ]?\w*|OLMo[- ]?2?|Pythia[- ]\d+[MB]?|Vicuna|InternLM\d?|GLM[- ]?4(?:\.\d)?|Kimi[- ]?K2|MiniMax[- ]?M\d|LLaDA(?:[- ]\d\.\d)?|Dream[- ]\d|SDAR|Mamba[- ]?2?|RWKV[- ]?\d?|Jamba|Nemotron[- ]?H?|Llava[- ]?\w*|LLaVA[- ]?\w*|Qwen2\.5[- ]VL|InternVL\d?(?:\.\d)?)", re.I)
GITHUB_RX = re.compile(r"https?://(?:www\.)?github\.com/[\w\-\.]+/[\w\-\.]+")


def sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)


def gpu_type(s):
    for rx, f, n in GPU_RX:
        if rx.search(s):
            return n, f
    return None, None


def to_hours(num, unit):
    u = unit.lower()
    if u.startswith("min"):
        return num / 60
    if u.startswith("day"):
        return num * 24
    if u.startswith("week"):
        return num * 24 * 7
    if u.startswith("s"):
        return num / 3600
    return num


def parse_num(x):
    x = x.lower()
    if x in NUMW:
        return NUMW[x]
    if x in ("three",):
        return 3
    if x == "several":
        return None
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


COMPUTE_CUE = re.compile(r"\b(takes?|took|taking|requir\w+|within|in (only|just|about|around|approximately|roughly)?|train\w*|fine-?tun\w*|quantiz\w*|convert\w*|distill\w*|complet\w*|cost\w*|consum\w*|spend|spent|run\w*|last\w*|budget|process\w*|calibrat\w*|compress\w*|adapt\w*|pre-?train\w*)\b", re.I)


def compute_from_sentence(s):
    """Return list of dicts {h100_hours, gpu, count, hours, kind, cited} parsed from one sentence.

    `cited` marks figures that describe *someone else's* model (e.g. "pre-training DeepSeek-V3
    required 2.66M GPU hours") rather than the paper's own method."""
    out = []
    for clause in re.split(r",\s*(?=while\b|whereas\b|compared\b|in contrast\b|versus\b|vs\.?\s)|;\s+|\(\s*vs\.?", s):
        cited = bool(CITED_RX.search(clause)) and not OWN_RX.search(clause)
        for r in _compute_from_sentence(clause):
            r["cited"] = cited
            out.append(r)
    return out


def _compute_from_sentence(s):
    out = []
    typ, fac = gpu_type(s)
    # explicit GPU-hours / GPU-days
    for m in GPUH_RX.finditer(s):
        n = parse_num(m.group(1))
        if n is None:
            continue
        mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get((m.group(2) or "").lower(), 1)
        hrs = to_hours(n * mult, m.group(3))
        t, f = gpu_type(m.group(0))
        t, f = (t, f) if t else (typ, fac)
        out.append(dict(kind="gpu-hours", gpu=t or "unspecified", count=None, hours=hrs,
                        h100_hours=hrs * (f if f else 1.0)))
    if out:
        return out
    if not COMPUTE_CUE.search(s):
        return out
    cm = COUNT_RX.search(s)
    tm = [m for m in TIME_RX.finditer(s)
          if not re.match(r"s\b|h\b", m.group(2)) or re.match(r"\d", m.group(1))]
    if cm and tm:
        cnt = parse_num(cm.group(1))
        t, f = gpu_type(cm.group(0))
        t, f = (t, f) if t else (typ, fac)
        if cnt and cnt <= 100000:
            for m in tm[:3]:
                n = parse_num(m.group(1))
                if n is None:
                    continue
                hrs = to_hours(n, m.group(2))
                if hrs <= 0 or hrs > 24 * 365:
                    continue
                out.append(dict(kind="count×time", gpu=t or "unspecified", count=cnt, hours=hrs,
                                h100_hours=cnt * hrs * (f if f else 1.0)))
    return out


def text_of(blocks):
    return [b for b in blocks if b.get("kind") in ("paragraph", "heading", "table", "list", "list_item")]


def section_text(blocks, name_rx, limit):
    out, on, lvl = [], False, None
    for b in blocks:
        if b["kind"] == "heading":
            if on and b["level"] <= lvl:
                break
            if re.search(name_rx, b["text"], re.I):
                on, lvl = True, b["level"]
                continue
        elif on and b["kind"] in ("paragraph", "list", "list_item"):
            out.append(b["text"])
            if sum(map(len, out)) > limit:
                break
    s = " ".join(out)
    return (s[:limit].rsplit(" ", 1)[0] + " …") if len(s) > limit else s


def contributions(blocks):
    for i, b in enumerate(blocks):
        if b["kind"] == "paragraph" and re.search(r"\b(our|main|key|primary) contributions?\b|we (make|summarize) the following contributions|contributions (are|of this|can be)", b["text"], re.I):
            parts = [b["text"]]
            for nb in blocks[i + 1:i + 6]:
                if nb["kind"] == "heading":
                    break
                if nb["kind"] in ("list", "list_item") or len(nb["text"]) < 700:
                    parts.append(nb["text"])
                else:
                    break
            s = " ".join(parts)
            return s[:1600] + (" …" if len(s) > 1600 else "")
    return ""


def pick_tables(blocks, n=2, max_rows=18):
    tabs = []
    for b in blocks:
        if b["kind"] != "table":
            continue
        md = b.get("markdown") or ""
        lines = md.splitlines()
        rows = [l for l in lines if l.startswith("|")]
        if len(rows) < 3:
            continue
        cap = next((l for l in lines if l.startswith("###")), "")
        body = rows[:max_rows]
        if len(rows) > max_rows:
            body.append("| … |")
        # keep tables narrow enough to read
        if max(len(r) for r in body) > 700:
            continue
        tabs.append((cap, "\n".join(body)))
    # prefer tables whose caption talks about results / comparison
    def pri(t):
        c = t[0].lower()
        return 0 if re.search(r"result|comparison|perplexity|accuracy|speed|throughput|latency|performance|benchmark", c) else 1
    tabs.sort(key=pri)
    return [dict(caption=c.lstrip("# ").strip(), markdown=m) for c, m in tabs[:n]]


def digest(pid):
    out = os.path.join(DIGEST, f"{pid}.json")
    if os.path.exists(out):
        return "cached"
    try:
        r = subprocess.run(["arxiv", "read", pid, "--format", "json", "--no-references", "--no-figures"],
                           capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        json.dump({"id": pid, "error": "timeout"}, open(out, "w"))
        return "timeout"
    if r.returncode != 0 or not r.stdout.strip():
        json.dump({"id": pid, "error": (r.stderr or "")[-300:]}, open(out, "w"))
        return "error"
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        json.dump({"id": pid, "error": "bad json"}, open(out, "w"))
        return "error"
    blocks = d.get("blocks", [])
    paras = [b["text"] for b in blocks if b.get("kind") in ("paragraph", "list", "list_item")]
    full = "\n".join(paras)

    compute, hw_sent = [], []
    seen = set()
    for s in sentences(full):
        if len(s) > 900 or not GPU_ANY.search(s) and not re.search(r"GPU[- ]hours|GPU[- ]days", s, re.I):
            continue
        c = compute_from_sentence(s)
        key = s[:120]
        if c and key not in seen:
            seen.add(key)
            compute.append(dict(snippet=s.strip()[:600], parsed=c))
        elif key not in seen and len(hw_sent) < 6 and re.search(r"(conduct|run|perform|train|implement|evaluat|experiment|measur|benchmark)\w*", s, re.I) \
                and COUNT_RX.search(s) and gpu_type(s)[0]:
            seen.add(key)
            hw_sent.append(s.strip()[:400])
    gpus = {}
    for rx, f, n in GPU_RX:
        k = len(rx.findall(full))
        if k:
            gpus[n] = k
    toks = []
    for m in TOKENS_RX.finditer(full):
        v = float(m.group(1)) * {"b": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12, "m": 1e6, "million": 1e6}[m.group(2).lower()]
        toks.append(v)
    models = {}
    for m in MODEL_RX.finditer(full):
        k = re.sub(r"\s+", "-", m.group(1))
        models[k] = models.get(k, 0) + 1
    gh = sorted(set(u.rstrip(".") for u in GITHUB_RX.findall(full + " " + r.stdout[:0])))
    dig = dict(
        id=pid, version=d.get("metadata", {}).get("version", ""), origin=d.get("origin", ""),
        headings=[b["text"] for b in blocks if b["kind"] == "heading" and b["level"] <= 3][:60],
        contributions=contributions(blocks),
        conclusion=section_text(blocks, r"^\s*(\d+\.?\s*)?(conclusions?|concluding remarks|summary|discussion and conclusion)", 1400),
        limitations=section_text(blocks, r"limitations?", 900),
        compute=compute[:12], hardware_sentences=hw_sent, gpus=gpus,
        max_train_tokens=max(toks) if toks else None,
        models=sorted(models.items(), key=lambda kv: -kv[1])[:12],
        github=gh[:5], tables=pick_tables(blocks), n_chars=len(full),
    )
    json.dump(dig, open(out, "w"), ensure_ascii=False)
    # free the HTML cache to keep disk usage bounded
    for p in glob.glob(os.path.join(CACHE, pid + "v*")):
        shutil.rmtree(p, ignore_errors=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(DIGEST, exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    ids = [i for i in ids if not os.path.exists(os.path.join(DIGEST, f"{i}.json"))]
    print(f"{len(ids)} to fetch", flush=True)
    stats = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(digest, ids)):
            stats[res] = stats.get(res, 0) + 1
            if k % 50 == 0:
                print(k, stats, flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
