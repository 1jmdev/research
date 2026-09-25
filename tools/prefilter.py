"""Local re-implementation of the topic queries in queries.py.

arXiv advanced-search syntax (quoted phrases, OR) is compiled to regexes and
matched against title + abstract of OAI-PMH records, so that the full set of
cs papers can be filtered offline without hitting arxiv.org/search.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from queries import QUERIES  # noqa: E402


def term_regex(tok):
    tok = tok.strip()
    if tok.startswith('"') and tok.endswith('"'):
        words = tok[1:-1].split()
        body = r"(?:[\s\-]+|\s*\([^)]{1,12}\)\s*)".join(re.escape(w) for w in words)
        # allow simple plural / inflection on the last word
        return r"(?i:\b" + body + r"(s|es|ed|ing)?\b)"
    if re.fullmatch(r"[A-Z0-9][A-Z0-9:\-]*[A-Za-z0-9]*", tok) and any(c.isupper() for c in tok) and len(tok) <= 8:
        # acronym: case-sensitive, optional plural s
        return r"\b" + re.escape(tok) + r"s?\b"
    # bare word: prefix match approximates arXiv's stemming
    stem = tok.lower()
    for suf in ("ization", "ation", "ing", "ed", "es", "s", "e"):
        if stem.endswith(suf) and len(stem) - len(suf) >= 4:
            stem = stem[: -len(suf)]
            break
    return r"(?i:\b" + re.escape(stem) + r"\w*)"


def split_or(term):
    parts, cur, q = [], "", False
    for tok in re.findall(r'"[^"]*"|\S+', term):
        if tok == "OR":
            parts.append(cur.strip())
            cur = ""
        else:
            cur += " " + tok
    parts.append(cur.strip())
    return [p for p in parts if p]


def compile_query(terms):
    groups = []
    for term in terms:
        alts = []
        for p in split_or(term):
            # un-quoted multi-word alternatives are ANDed words; treat as phrase
            if not p.startswith('"') and " " in p:
                p = '"' + p + '"'
            alts.append(term_regex(p))
        groups.append(re.compile("|".join(alts)))
    return groups


COMPILED = {k: compile_query(v) for k, v in QUERIES.items()}


def matching_queries(title, abstract):
    text = title + " " + abstract
    hits = []
    for k, groups in COMPILED.items():
        if all(rx.search(text) for rx in groups):
            hits.append(k)
    return hits
