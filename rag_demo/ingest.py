# rag_demo/ingest.py
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

from .utils import load_yaml, ensure_dir

# Optional parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None  # handled later

try:
    import trafilatura
except Exception:
    trafilatura = None  # handled later

from bs4 import BeautifulSoup

import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient


# ----------------------------
# Config and paths
# ----------------------------
ROOT = Path(__file__).parent
CFG = load_yaml(ROOT / "config.yaml")

DOCS_DIR = (ROOT.parent / CFG["paths"]["docs_dir"]).resolve()
CHROMA_DIR = (ROOT / CFG["paths"]["chroma_dir"]).resolve()
ensure_dir(CHROMA_DIR)

URL_MANIFEST_PATH = ROOT / "meta" / "url_manifest.json"
URL_MAP: Dict[str, str] = {}
if URL_MANIFEST_PATH.exists():
    with open(URL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    URL_MAP = {row["file"]: row["url"] for row in rows}
else:
    print("WARN url_manifest.json not found so chunks will not have URLs")

# Chunking params
CHUNK_TOKENS = int(CFG["ingest"].get("chunk_tokens", 500))
OVERLAP = int(CFG["ingest"].get("chunk_overlap", 50))


# ----------------------------
# Utilities
# ----------------------------
def guess_url_for_file(fname: str) -> str:
    return URL_MAP.get(fname, "")


def list_candidate_files(base: Path, limit: int | None = None) -> List[Path]:
    exts = {".txt", ".md", ".html", ".htm", ".pdf"}
    files = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
            if limit and len(files) >= limit:
                break
    return files


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if trafilatura:
        extracted = trafilatura.extract(raw) or ""
        if extracted.strip():
            return extracted
    # fallback
    soup = BeautifulSoup(raw, "lxml")
    return soup.get_text(" ", strip=True)


def read_pdf_file(path: Path) -> str:
    if not pdfplumber:
        return ""
    try:
        text_chunks = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    text_chunks.append(txt)
        return "\n".join(text_chunks)
    except Exception:
        return ""


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix in {".html", ".htm"}:
        return read_html_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    return ""


# ----------------------------
# Tokenizer and chunking
# ----------------------------
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(s: str) -> int:
    return len(_enc.encode(s))


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> List[str]:
    toks = _enc.encode(text)
    n = len(toks)
    if n == 0:
        return []
    chunks = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, n, step):
        end = start + chunk_tokens
        piece = _enc.decode(toks[start:end])
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
    return chunks


# ----------------------------
# Embedding and DB
# ----------------------------
@dataclass
class IngestItem:
    doc_id: str
    chunk_id: int
    text: str
    metadata: Dict


def embed_and_write(
    items: List[IngestItem],
    client: PersistentClient,
    collection_name: str = "tilt_docs",
    batch_size: int = 128,
    reset: bool = False,
) -> Tuple[int, int]:
    # wipe if requested
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(collection_name)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    total = len(items)
    added = 0

    use_upsert = hasattr(col, "upsert")

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        texts = [it.text for it in batch]
        ids = [f"{it.doc_id}::chunk-{it.chunk_id}" for it in batch]
        metas = [it.metadata for it in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()

        if use_upsert:
            col.upsert(documents=texts, metadatas=metas, ids=ids, embeddings=embs)
        else:
            # older API has only add which fails on duplicate ids
            try:
                col.add(documents=texts, metadatas=metas, ids=ids, embeddings=embs)
            except Exception as e:
                # best effort skip duplicates
                print(f"skip batch {i} due to {type(e).__name__}: {e}")

        added += len(batch)
        print(f"added {added}/{total}")

    # no client.persist on modern chromadb
    if hasattr(col, "persist"):
        try:
            col.persist()
        except Exception:
            pass

    return added, total


# ----------------------------
# Main pipeline
# ----------------------------
def build_items(files: List[Path]) -> List[IngestItem]:
    items = []
    for f in files:
        text = load_document(f)
        if not text or not text.strip():
            continue

        fname = f.name
        url = guess_url_for_file(fname)

        chunks = chunk_text(text, CHUNK_TOKENS, OVERLAP)
        if not chunks:
            continue

        base_id = fname
        for idx, ch in enumerate(chunks):
            meta = {
                "file": fname,
                "url": url,
                "source": url or f"file://{f.resolve()}",
                "chunk_index": idx,
                "n_chunks": len(chunks),
                "tokens": count_tokens(ch),
            }
            items.append(IngestItem(doc_id=base_id, chunk_id=idx, text=ch, metadata=meta))
    return items


def main():
    parser = argparse.ArgumentParser(description="Ingest local docs into Chroma with URL metadata")
    parser.add_argument("--docs", type=str, default=str(DOCS_DIR), help="Docs directory to scan")
    parser.add_argument("--chroma", type=str, default=str(CHROMA_DIR), help="Chroma persist dir")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files during testing")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()

    docs_dir = Path(args.docs).resolve()
    chroma_dir = Path(args.chroma).resolve()
    ensure_dir(chroma_dir)

    if not docs_dir.exists():
        raise SystemExit(f"Docs dir does not exist at {docs_dir}")

    files = list_candidate_files(docs_dir, limit=args.limit if args.limit and args.limit > 0 else None)
    if not files:
        raise SystemExit(f"No supported files found under {docs_dir}")

    print(f"Scanning {len(files)} files from {docs_dir}")
    items = build_items(files)
    if not items:
        raise SystemExit("No chunks produced. Check scraper outputs and parsers")

    client = PersistentClient(path=str(chroma_dir))
    added, total = embed_and_write(items, client, collection_name="tilt_docs", reset=args.reset)
    print(f"Ingest complete. Added {added} chunks from {len(files)} files into {chroma_dir}")


if __name__ == "__main__":
    main()
