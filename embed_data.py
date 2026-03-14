# Efficient, idempotent embedding pipeline for your scraped data
# Works with .txt and .pdf, persists to Chroma, attaches source URLs, skips duplicates

import os
import re
import uuid
import time
import json
import hashlib
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DATA_DIR = "data"                # your scraped folder
DB_PATH = "vector_store"         # chroma persistence dir
COLLECTION = "tilt_knowledgebase"
EMBED_MODEL = "all-MiniLM-L6-v2" # fast, accurate on CPU
TARGET_CHARS = 1000              # chunk target size
OVERLAP_CHARS = 200              # chunk overlap
BASE_URL = "https://tilt.colostate.edu"
VISITED_FILE = "visited.txt"     # produced by your scraper

# Compile once for speed
BOILERPLATE_PATTERNS = [
    r"^The Institute for Learning and Teaching",
    r"^TILT Quick Links",
    r"^Departments & Programs",
    r"^Pressbooks and OER",
    r"^Best Practices in Teaching",
    r"^Courses$",
    r"^Contact CSU",
    r"^Disclaimer",
    r"^Equal Opportunity",
    r"^Privacy Statement",
    r"^Accessibility Statement",
    r"WP ADA Compliance Check",
    r"^Phone:",
    r"^Email:",
    r"^© \d{4} Colorado State University",
    r"^\s*$"
]
BP_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def chunk_md5(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip()).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"TXT read error {path} -> {e}")
        return ""


def read_pdf(path: str) -> str:
    txt = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            t = p.extract_text() or ""
            txt += t + "\n"
    except Exception as e:
        print(f"PDF read error {path} -> {e}")
    return txt


def clean_text(s: str) -> str:
    # remove boilerplate lines
    lines = []
    for line in s.splitlines():
        if not BP_REGEX.search(line):
            lines.append(line.strip())
    s = "\n".join(lines)
    # fix common pdf hyphenation and join broken words
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()


def paragraphs(s: str) -> List[str]:
    # split on blank lines, keep order
    return [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]


def pack_chunks(paras: List[str], target: int, overlap: int) -> List[str]:
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= target:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
                # create overlap by taking the tail of previous chunk
                tail = buf[-overlap:] if len(buf) > overlap else buf
                buf = f"{tail}\n{p}".strip()
            else:
                # very long paragraph, hard cut
                for i in range(0, len(p), target):
                    seg = p[i:i+target]
                    if seg.strip():
                        chunks.append(seg.strip())
                buf = ""
    if buf:
        chunks.append(buf)
    # final cleanup
    return [re.sub(r"\s+", " ", c).strip() for c in chunks if c.strip()]


def load_visited_urls(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u.startswith("http"):
                urls.append(u)
    return urls


def slug_from_url(u: str) -> str:
    try:
        p = urlparse(u).path
        return re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_").lower()
    except:
        return ""


def map_filename_to_url(fname: str, visited: List[str]) -> Optional[str]:
    # your scraper names files from path-with-slashes to underscores
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    # exact match first
    for u in visited:
        if slug_from_url(u) == stem:
            return u
    # fallback reconstruction
    return f"{BASE_URL}/{stem.replace('_', '/')}"


def ensure_collection(client: chromadb.PersistentClient, name: str):
    try:
        return client.get_collection(name)
    except:
        return client.create_collection(name)


def main():
    t0 = time.time()
    print("Init model and vector store")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = ensure_collection(client, COLLECTION)

    visited = load_visited_urls(VISITED_FILE)
    files_count = 0
    chunks_count = 0

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.lower().endswith((".txt", ".pdf")):
                continue
            fpath = os.path.join(root, f)
            fhash = file_md5(fpath)

            # skip whole file if exact same hash already present
            existing = coll.get(where={"hash": fhash})
            if existing and existing.get("ids"):
                print(f"Skip unchanged {f}")
                continue

            raw = read_pdf(fpath) if f.lower().endswith(".pdf") else read_txt(fpath)
            if not raw.strip():
                print(f"Empty file skipped {f}")
                continue

            cleaned = clean_text(raw)
            if not cleaned:
                print(f"No content after cleaning {f}")
                continue

            paras = paragraphs(cleaned)
            chunks = pack_chunks(paras, TARGET_CHARS, OVERLAP_CHARS)
            if not chunks:
                print(f"No chunks produced {f}")
                continue

            # dedup chunk level by hash against existing collection
            docs, metas, ids, to_embed = [], [], [], []
            url = map_filename_to_url(f, visited)

            # prefetch any existing chunk hashes for this file hash to avoid get spam
            # minimal index: request small batch to see if any exist, then per chunk check
            for idx, ch in enumerate(chunks):
                chash = chunk_md5(ch)
                existed = coll.get(where={"chunk_hash": chash})
                if existed and existed.get("ids"):
                    continue
                docs.append(ch)
                metas.append({
                    "file": f,
                    "hash": fhash,
                    "chunk_index": idx,
                    "chunk_hash": chash,
                    "source": url
                })
                ids.append(f"{os.path.splitext(f)[0]}_{idx}_{uuid.uuid4().hex[:6]}")
                to_embed.append(ch)

            if not to_embed:
                print(f"All chunks already present for {f}")
                continue

            embs = model.encode(to_embed, show_progress_bar=True)
            coll.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)

            files_count += 1
            chunks_count += len(to_embed)
            print(f"Embedded {f} new chunks {len(to_embed)}")

    dt = round(time.time() - t0, 2)
    print(json.dumps({
        "files_processed": files_count,
        "chunks_added": chunks_count,
        "db_path": DB_PATH,
        "collection": COLLECTION,
        "seconds": dt
    }, indent=2))


if __name__ == "__main__":
    main()
