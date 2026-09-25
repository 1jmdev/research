#!/usr/bin/env python3
"""Manual curation helper: a human/LLM reviewer reads every candidate (title + abstract)
and records keep/drop + category decisions.

  python3 tools/curate.py next [N]      print the next N undecided papers (grouped by suggested category)
  python3 tools/curate.py apply         read decisions for the last printed batch from stdin:
                                          D: <id> <id> ...          -> drop (not useful)
                                          K: <id> <id> ...          -> keep only these, drop the rest
                                          R: <id>:<CODE> ...        -> keep, re-categorised
                                        every other paper of the batch -> keep with suggested code
  python3 tools/curate.py stats         progress summary

Decisions are stored in $CURATION (default data/curation.tsv): id <TAB> K|D <TAB> CODE
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.environ.get("CANDIDATES", os.path.join(ROOT, "data", "candidates.jsonl"))
CURATION = os.environ.get("CURATION", os.path.join(ROOT, "data", "curation.tsv"))
LAST = CURATION + ".lastbatch"

CODES = {
    "Q0": "quantization/sub-1-bit", "Q1": "quantization/1-bit-and-ternary", "Q2": "quantization/2-bit",
    "Q3": "quantization/3-bit", "Q4": "quantization/4-bit-integer", "QF": "quantization/4-bit-floating-point",
    "Q8": "quantization/5-to-8-bit", "QM": "quantization/mixed-precision", "QT": "quantization/low-precision-training",
    "QG": "quantization/_general",
    "KQ": "kv-cache/quantization", "KE": "kv-cache/eviction-and-token-selection", "KL": "kv-cache/low-rank-and-latent",
    "KX": "kv-cache/cross-layer-sharing", "KO": "kv-cache/offloading-and-hierarchical-storage",
    "KP": "kv-cache/prefix-caching-and-reuse", "KG": "kv-cache/_general",
    "AS": "attention/sparse-attention", "AL": "attention/linear-attention", "AR": "attention/state-space-and-recurrent",
    "AH": "attention/hybrid-architectures", "AK": "attention/kernels-and-io-aware", "AV": "attention/attention-variants",
    "AP": "attention/long-context-and-position",
    "DS": "decoding/speculative-decoding", "DM": "decoding/multi-token-prediction",
    "DJ": "decoding/jacobi-and-parallel-decoding", "DD": "decoding/diffusion-language-models",
    "DI": "decoding/diffusion-llm-inference", "DE": "decoding/early-exit-and-layer-skipping",
    "DC": "decoding/constrained-and-structured", "DP": "decoding/sampling-and-strategies",
    "CA": "model-conversion/ar-to-diffusion", "CL": "model-conversion/transformer-to-linear-or-hybrid",
    "CM": "model-conversion/attention-conversion", "CU": "model-conversion/dense-to-moe-upcycling",
    "CT": "model-conversion/tokenizer-and-vocab-transfer", "CG": "model-conversion/model-merging",
    "CO": "model-conversion/model-growth-and-other",
    "PU": "compression/unstructured-and-semi-structured-pruning", "PS": "compression/structured-pruning",
    "PA": "compression/activation-sparsity", "PL": "compression/low-rank-decomposition",
    "PD": "compression/knowledge-distillation", "PG": "compression/_general",
    "EA": "mixture-of-experts/architecture-and-routing", "EI": "mixture-of-experts/inference-and-serving",
    "EC": "mixture-of-experts/compression", "ET": "mixture-of-experts/training",
    "SB": "serving-systems/scheduling-and-batching", "SD": "serving-systems/disaggregated-prefill-decode",
    "SP": "serving-systems/distributed-inference-and-parallelism", "SK": "serving-systems/gpu-kernels-and-compilers",
    "SH": "serving-systems/hardware-accelerators", "SE": "serving-systems/edge-and-on-device",
    "SM": "serving-systems/memory-and-offloading", "SN": "serving-systems/energy-and-cost",
    "SG": "serving-systems/_general",
    "TO": "training/optimizers", "TS": "training/scaling-laws", "TP": "training/pretraining-recipes-and-efficiency",
    "TF": "training/parameter-efficient-finetuning", "TR": "training/rl-for-reasoning",
    "TD": "training/data-curation-and-synthetic-data", "TK": "training/tokenization",
    "RE": "reasoning/efficient-reasoning", "RT": "reasoning/test-time-scaling", "RL": "reasoning/latent-and-looped",
    "XP": "context-compression/prompt-and-context-compression", "XV": "context-compression/visual-token-reduction",
    "MR": "models-and-architectures/technical-reports", "MS": "models-and-architectures/small-language-models",
    "MA": "models-and-architectures/novel-architectures",
}
FOLDER2CODE = {v: k for k, v in CODES.items()}
ORDER = list(CODES)


def load_decisions():
    d = {}
    if os.path.exists(CURATION):
        for line in open(CURATION):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                d[parts[0]] = (parts[1], parts[2])
    return d


def load_candidates():
    out = []
    for line in open(CANDIDATES):
        r = json.loads(line)
        r["code"] = FOLDER2CODE.get(r["primary"], "QG")
        out.append(r)
    return out


def cmd_next(n):
    dec = load_decisions()
    todo = [r for r in load_candidates() if r["id"] not in dec]
    todo.sort(key=lambda r: (ORDER.index(r["code"]), r["id"]))
    batch = todo[:n]
    with open(LAST, "w") as f:
        for r in batch:
            f.write(f"{r['id']}\t{r['code']}\n")
    print(f"# {len(todo)} undecided; showing {len(batch)}")
    for r in batch:
        ab = r["abstract"][:int(os.environ.get("ABSTRACT_CHARS", 300))].rsplit(" ", 1)[0]
        print(f"{r['id']} [{r['code']}] {r['title']} :: {ab}")


def cmd_apply(text):
    batch = [l.split("\t") for l in open(LAST).read().splitlines() if l]
    ids = {i for i, _ in batch}
    drops, recodes = set(), {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("K:"):  # keep-only list: every other paper in the batch is dropped
            keep = set(line[2:].split())
            drops.update(ids - keep)
        elif line.startswith("D:"):
            drops.update(line[2:].split())
        elif line.startswith("R:"):
            for tok in line[2:].split():
                i, c = tok.split(":")
                assert c in CODES, f"unknown code {c}"
                recodes[i] = c
    unknown = (drops | set(recodes)) - ids
    assert not unknown, f"ids not in batch: {sorted(unknown)}"
    with open(CURATION, "a") as f:
        for i, code in batch:
            if i in drops:
                f.write(f"{i}\tD\t-\n")
            else:
                f.write(f"{i}\tK\t{recodes.get(i, code)}\n")
    print(f"applied: {len(batch)} papers, {len(drops)} dropped, {len(recodes)} re-categorised")
    os.remove(LAST)


def cmd_stats():
    dec = load_decisions()
    c = Counter(v for v, _ in dec.values())
    cand = load_candidates()
    print(f"candidates {len(cand)}; decided {len(dec)} (keep {c['K']}, drop {c['D']}); "
          f"remaining {sum(1 for r in cand if r['id'] not in dec)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "next":
        cmd_next(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    elif cmd == "apply":
        cmd_apply(sys.stdin.read())
    else:
        cmd_stats()
