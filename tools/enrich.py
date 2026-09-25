#!/usr/bin/env python3
"""Enrich harvested papers with impact signals used for ranking.

* Semantic Scholar batch API -> citationCount, influentialCitationCount, venue, tldr
* Hugging Face Daily Papers   -> community upvotes, organisation, github repo/stars

Outputs data/s2.json and data/hf.json keyed by arXiv id.
Usage: python3 tools/enrich.py [s2|hf|all]
"""
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = os.environ.get("CANDIDATES", os.path.join(ROOT, "data", "candidates.jsonl"))
OUT = os.environ.get("ENRICH_DIR", os.path.join(ROOT, "data"))


def all_ids():
    ids = set()
    for line in open(CANDIDATES):
        ids.add(json.loads(line)["id"])
    return sorted(ids)


def curl_json(args, tries=6):
    for i in range(tries):
        r = subprocess.run(["curl", "-sS", "--max-time", "120", "-w", "\n%{http_code}"] + args,
                           capture_output=True, text=True)
        body, _, code = r.stdout.rpartition("\n")
        if code == "200":
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
        wait = 3 * (2 ** i)
        print(f"  http {code}; retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    return None


def enrich_s2():
    path = os.path.join(OUT, "s2.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    todo = [i for i in all_ids() if i not in data]
    fields = "citationCount,influentialCitationCount,venue,publicationVenue,tldr,publicationDate,externalIds"
    for k in range(0, len(todo), 400):
        chunk = todo[k:k + 400]
        res = curl_json(["-X", "POST",
                         f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={fields}",
                         "-H", "Content-Type: application/json",
                         "-d", json.dumps({"ids": [f"arXiv:{i}" for i in chunk]})])
        if res is None:
            print("giving up on chunk", k, file=sys.stderr)
            continue
        for pid, item in zip(chunk, res):
            if not item:
                data[pid] = None
                continue
            pv = item.get("publicationVenue") or {}
            tl = item.get("tldr") or {}
            data[pid] = dict(
                citations=item.get("citationCount") or 0,
                influential=item.get("influentialCitationCount") or 0,
                venue=item.get("venue") or "",
                venue_short=(pv.get("alternate_names") or [""])[0] if pv else "",
                venue_type=pv.get("type", "") if pv else "",
                tldr=tl.get("text") or "",
                dblp=(item.get("externalIds") or {}).get("DBLP", ""),
            )
        print(f"s2 {k + len(chunk)}/{len(todo)}")
        json.dump(data, open(path, "w"))
        time.sleep(1.5)


def enrich_hf():
    path = os.path.join(OUT, "hf.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    done_path = os.path.join(OUT, "hf_dates.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    d = dt.date(2025, 1, 1)
    end = dt.date.today()
    while d <= end:
        ds = d.isoformat()
        if ds not in done:
            res = curl_json([f"https://huggingface.co/api/daily_papers?date={ds}&limit=100"])
            if res is not None:
                for x in res:
                    p = x.get("paper", {})
                    pid = p.get("id")
                    if not pid:
                        continue
                    org = p.get("organization") or x.get("organization") or {}
                    data[pid] = dict(
                        upvotes=p.get("upvotes", 0),
                        org=org.get("fullname") or org.get("name") or "",
                        github=p.get("githubRepo") or "",
                        github_stars=p.get("githubStars") or 0,
                        project_page=p.get("projectPage") or "",
                        ai_summary=p.get("ai_summary") or "",
                        ai_keywords=p.get("ai_keywords") or [],
                    )
                done.add(ds)
            time.sleep(0.4)
            if len(done) % 30 == 0:
                json.dump(data, open(path, "w"))
                json.dump(sorted(done), open(done_path, "w"))
                print("hf", ds, len(data))
        d += dt.timedelta(days=1)
    json.dump(data, open(path, "w"))
    json.dump(sorted(done), open(done_path, "w"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("hf", "all"):
        enrich_hf()
    if what in ("s2", "all"):
        enrich_s2()
