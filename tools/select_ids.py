#!/usr/bin/env python3
"""List candidate ids for full-text digestion.

Order: every paper in compute-critical leaves first (conversion, quantization,
diffusion/MTP/parallel decoding, low-precision training, upcycling), then the
top-N papers (by impact score, if data/s2.json + data/hf.json exist) of every
other leaf. Usage: python3 tools/select_ids.py [--top 25] > ids.txt
"""
import argparse
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.environ.get("CANDIDATES", os.path.join(ROOT, "data", "candidates.jsonl"))
ENR = os.environ.get("ENRICH_DIR", os.path.join(ROOT, "data"))
FULL = ("model-conversion/", "quantization/", "decoding/diffusion-language-models", "decoding/multi-token-prediction",
        "decoding/jacobi-and-parallel-decoding", "kv-cache/quantization")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
s2, hf = load(os.path.join(ENR, "s2.json")), load(os.path.join(ENR, "hf.json"))
by_leaf = defaultdict(list)
for line in open(CANDIDATES):
    r = json.loads(line)
    s = s2.get(r["id"]) or {}
    h = hf.get(r["id"]) or {}
    score = math.log1p(s.get("citations", 0) or 0) + math.log1p(s.get("influential", 0) or 0) + math.log1p(h.get("upvotes", 0) or 0)
    by_leaf[r["primary"]].append((score, r["id"]))
first, rest = [], []
for leaf, items in by_leaf.items():
    items.sort(reverse=True)
    if leaf.startswith(FULL):
        first += items
    else:
        rest += items[: a.top]
first.sort(reverse=True)
rest.sort(reverse=True)
for _, i in first + rest:
    print(i)
