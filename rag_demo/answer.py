# rag_demo/answer.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .utils import load_yaml

ROOT = Path(__file__).parent
CFG = load_yaml(ROOT / "config.yaml")
CHROMA_DIR = (ROOT / CFG["paths"]["chroma_dir"]).resolve()

_EMB = SentenceTransformer("all-MiniLM-L6-v2")

CANON = {
    "tei": "Teaching Effectiveness Initiative",
    "tef": "Teaching Effectiveness Framework",
    "oura": "Office for Undergraduate Research and Artistry",
    "la": "Learning Assistants",
    "writing center": "Writing Center",
    "tutoring": "TILT Tutoring",
    "GStcc" : "Graduate Student Teaching Certificate of Completion",
}

NOISY_PATH_HINTS = ["/wp-content/uploads/"]  # demote scans and OCR dumps


@dataclass
class Hit:
    text: str
    url: str
    distance: float
    score: float
    tokens: int | None = None


def normalize_query(q: str) -> str:
    ql = q.lower().strip()
    for k, v in CANON.items():
        if re.search(rf"\b{k}\b", ql, flags=re.I):
            ql = re.sub(rf"\b{k}\b", v, ql, flags=re.I)
    ql = re.sub(r"\s+", " ", ql).strip()
    return ql


def is_noisy(text: str) -> bool:
    if not text:
        return True
    non_ascii = sum(ord(ch) > 127 for ch in text)
    return non_ascii / max(1, len(text)) > 0.35  # relaxed per review


def demote_url(url: str) -> float:
    url = url or ""
    for bad in NOISY_PATH_HINTS:
        if bad in url:
            return 0.5
    return 1.0


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def jaccard(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def chroma_query(col, query: str, k: int = 15) -> List[Hit]:
    emb = _EMB.encode([query], normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        url = meta.get("url") or meta.get("source") or ""
        sim = 1.0 - float(dist)
        sim *= demote_url(url)
        out.append(Hit(text=doc or "", url=url, distance=float(dist), score=sim, tokens=meta.get("tokens")))

    out.sort(key=lambda h: (h.score, h.url.startswith("https://")), reverse=True)

    seen = set()
    uniq = []
    for h in out:
        key = h.url or h.text[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq


def mmr_rerank(items: List[Hit], lambda_weight: float = 0.65, top_k: int = 5) -> List[Hit]:
    if not items:
        return []
    chosen = [items[0]]
    candidates = items[1:]
    while candidates and len(chosen) < top_k:
        best = None
        best_val = -1e9
        for c in candidates:
            div = max(jaccard(ci.text, c.text) for ci in chosen)
            val = lambda_weight * c.score - (1 - lambda_weight) * div
            if val > best_val:
                best_val = val
                best = c
        chosen.append(best)
        candidates = [x for x in candidates if x is not best]
    return chosen


def bm25_snippet(query: str, body: str, max_chars: int = 500) -> str:
    sents = re.split(r"(?<=[\.\!\?])\s+", body.strip())
    sents = [s for s in sents if s and not is_noisy(s)]
    if not sents:
        return body[:max_chars].strip()

    tok_sents = [tokenize(s) for s in sents]
    bm25 = BM25Okapi(tok_sents)
    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)

    if not any(scores):
        return " ".join(sents[:2])[:max_chars].strip()

    idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    window = " ".join(sents[idx: idx + 2])
    if len(window) < 200 and idx + 2 < len(sents):
        window = " ".join(sents[idx: idx + 3])
    return window[:max_chars].strip()


def format_answer(query: str, hits: List[Hit], min_score: float = 0.15) -> Tuple[str, List[str]]:
    good = [h for h in hits if h.score >= min_score and not is_noisy(h.text)]
    if not good:
        return "No strong matches in TILT content. Be more specific", []

    top = good[0]
    snippet = bm25_snippet(query, top.text)
    snippet = re.sub(r"\s+", " ", snippet).strip()

    srcs = []
    for h in good[:5]:
        if h.url:
            srcs.append(h.url)
    srcs = list(dict.fromkeys(srcs))
    if not srcs and top.url:
        srcs = [top.url]

    return snippet, srcs


class RAG:
    def __init__(self):
        self.client = PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_or_create_collection("tilt_docs")

    def retrieve(self, query: str, k: int = 15) -> List[Hit]:
        q = normalize_query(query)
        candidates = chroma_query(self.col, q, k=k)
        return mmr_rerank(candidates, lambda_weight=0.65, top_k=7)

    def answer(self, query: str) -> Dict:
        hits = self.retrieve(query)
        text, sources = format_answer(query, hits)
        return {
            "answer": text,
            "sources": sources,
            "debug": [
                {"url": h.url, "score": round(h.score, 3), "distance": round(h.distance, 3)}
                for h in hits[:5]
            ],
        }


def chat_cli():
    rag = RAG()
    print("TILT RAG CLI — type 'exit' to quit")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        out = rag.answer(q)
        print("\nBot:")
        print(out["answer"])
        if out["sources"]:
            print("\nSources")
            for i, s in enumerate(out["sources"], 1):
                print(f"{i}. {s}")
        else:
            print("\nSources\nnone")
        print()


def one_off(q: str):
    out = RAG().answer(q)
    print(out["answer"])
    if out["sources"]:
        print("\nSources")
        for i, s in enumerate(out["sources"], 1):
            print(f"{i}. {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, default="")
    args = parser.parse_args()
    if args.q:
        one_off(args.q)
    else:
        chat_cli()


if __name__ == "__main__":
    main()
