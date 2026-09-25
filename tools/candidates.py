#!/usr/bin/env python3
"""Filter OAI-PMH records down to in-scope LLM papers and classify them.

Reads  $RAW_DIR/oai_cs.jsonl   (tools/oai_harvest.py)
Writes $DATA_DIR/candidates.jsonl  — one record per kept paper with
       `queries` (topic queries it matches) and `primary`/`secondary` categories.
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from prefilter import matching_queries  # noqa: E402
from taxonomy import classify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("RAW_DIR", os.path.join(ROOT, "data", "raw"))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))


def work(line):
    r = json.loads(line)
    q = matching_queries(r["title"], r["abstract"])
    if not q:
        return None
    prim, sec, sc = classify(r["title"], r["abstract"])
    if not prim:
        return None
    r.update(queries=q, primary=prim, secondary=sec, scores={k: round(v, 2) for k, v in sc.items()})
    return json.dumps(r, ensure_ascii=False)


def main():
    latest = {}  # later records (re-harvests / updates) win
    with open(os.path.join(RAW, "oai_cs.jsonl")) as f:
        for line in f:
            try:
                latest[json.loads(line)["id"]] = line
            except json.JSONDecodeError:
                continue  # partial trailing line while a harvest is still running
    lines = list(latest.values())
    print(len(lines), "unique records", flush=True)
    kept = 0
    with mp.Pool(max(1, os.cpu_count())) as pool, open(os.path.join(DATA, "candidates.jsonl"), "w") as out:
        for res in pool.imap(work, lines, chunksize=200):
            if res:
                out.write(res + "\n")
                kept += 1
    print("kept", kept)


if __name__ == "__main__":
    main()
