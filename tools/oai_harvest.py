#!/usr/bin/env python3
"""Harvest arXiv metadata (title, abstract, authors, categories, dates, comments)
through arXiv's OAI-PMH interface — the endpoint arXiv provides for bulk metadata
harvesting (https://info.arxiv.org/help/oa/index.html).

Harvests set=cs from --from (default 2025-01-01) and keeps only papers whose
arXiv id is 25xx/26xx (i.e. first submitted in 2025 or later).
Output: $RAW_DIR/oai_cs.jsonl  (resumable: the last resumption token is saved).

Usage: python3 tools/oai_harvest.py [--from 2025-01-01] [--set cs]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("RAW_DIR", os.path.join(ROOT, "data", "raw"))
BASE = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
ID_RX = re.compile(r"^(25|26)\d{2}\.\d{4,5}$")


def get(url, tries=8):
    tmp_h, tmp_b = RAW + "/.oai_head", RAW + "/.oai_body"
    for i in range(tries):
        subprocess.run(["curl", "-sSL", "--max-time", "300", "-D", tmp_h, "-o", tmp_b, url])
        head = open(tmp_h, errors="replace").read() if os.path.exists(tmp_h) else ""
        body = open(tmp_b, errors="replace").read() if os.path.exists(tmp_b) else ""
        codes = re.findall(r"HTTP/\S+ (\d+)", head)
        code = codes[-1] if codes else "?"
        if code == "200" and "<OAI-PMH" in body:
            return body
        ra = re.search(r"Retry-After:\s*(\d+)", head, re.I)
        wait = int(ra.group(1)) if ra else 10 * (i + 1)
        print(f"  http {code}; waiting {wait}s", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise SystemExit("OAI request failed repeatedly: " + url)


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def parse(xml):
    root = ET.fromstring(xml)
    out = []
    for rec in root.iterfind(".//o:record", NS):
        md = rec.find("o:metadata/a:arXiv", NS)
        if md is None:
            continue
        pid = clean(md.findtext("a:id", default="", namespaces=NS))
        if not ID_RX.match(pid):
            continue
        authors = []
        for a in md.iterfind("a:authors/a:author", NS):
            fn = clean(a.findtext("a:forenames", default="", namespaces=NS))
            kn = clean(a.findtext("a:keyname", default="", namespaces=NS))
            authors.append((fn + " " + kn).strip())
        out.append(dict(
            id=pid,
            title=clean(md.findtext("a:title", default="", namespaces=NS)),
            authors=authors,
            abstract=clean(md.findtext("a:abstract", default="", namespaces=NS)),
            categories=clean(md.findtext("a:categories", default="", namespaces=NS)).split(),
            created=clean(md.findtext("a:created", default="", namespaces=NS)),
            updated=clean(md.findtext("a:updated", default="", namespaces=NS)),
            comments=clean(md.findtext("a:comments", default="", namespaces=NS)),
            journal_ref=clean(md.findtext("a:journal-ref", default="", namespaces=NS)),
            doi=clean(md.findtext("a:doi", default="", namespaces=NS)),
            license=clean(md.findtext("a:license", default="", namespaces=NS)),
        ))
    tok = root.find(".//o:resumptionToken", NS)
    token = tok.text.strip() if tok is not None and tok.text else None
    total = tok.get("completeListSize") if tok is not None else None
    return out, token, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2025-01-01")
    ap.add_argument("--until", default="")
    ap.add_argument("--set", default="cs")
    ap.add_argument("--fresh", action="store_true", help="truncate the output file instead of appending")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, f"oai_{a.set}.jsonl")
    state = out + ".token"
    token = open(state).read().strip() if os.path.exists(state) else None
    if token:
        url = f"{BASE}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
    else:
        url = f"{BASE}?verb=ListRecords&set={a.set}&metadataPrefix=arXiv&from={a.frm}" + (f"&until={a.until}" if a.until else "")
        if a.fresh:
            open(out, "w").close()
    n = 0
    while url:
        body = get(url)
        recs, token, total = parse(body)
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += len(recs)
        print(f"kept {n} (page {len(recs)}) total_list={total} token={token}", flush=True)
        if token:
            open(state, "w").write(token)
            url = f"{BASE}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
            time.sleep(1)
        else:
            url = None
            if os.path.exists(state):
                os.remove(state)
    print("done", n)


if __name__ == "__main__":
    main()
