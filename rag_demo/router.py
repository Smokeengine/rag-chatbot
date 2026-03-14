# rag_demo/router.py
from __future__ import annotations
import json
from pathlib import Path
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from .utils import load_yaml

try:
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None  # handled below

ROOT = Path(__file__).parent
CFG = load_yaml(ROOT / "config.yaml")
CHROMA_DIR = (ROOT / CFG["paths"]["chroma_dir"]).resolve()

class RouterLLM:
    """Lightweight orchestrator LLM before retrieval."""
    def __init__(self, local_model="mistral", retriever=None):
        self.local_model = local_model
        self.retriever = retriever
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_or_create_collection("tilt_docs")

    def _query_llm(self, prompt: str) -> str:
        if not ollama_chat:
            return "RAG fallback"
        resp = ollama_chat(model=self.local_model, messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]

    def _is_smalltalk(self, text: str) -> bool:
        smalltalk = ["hi", "hello", "hey", "thank", "weather", "who are you", "how are you"]
        return any(s in text.lower() for s in smalltalk)

    def _retrieve(self, query: str, k=5):
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=k, include=["documents", "metadatas", "distances"])
        hits = []
        for i in range(len(res["documents"][0])):
            meta = res["metadatas"][0][i]
            hits.append({
                "text": res["documents"][0][i],
                "url": meta.get("url") or meta.get("source") or "",
                "distance": float(res["distances"][0][i])
            })
        # distance is smaller is better
        hits.sort(key=lambda h: h["distance"])
        return hits

    def ask(self, query: str):
        if self._is_smalltalk(query):
            return {"answer": "Hey there. Ask me about tutoring, learning, or academic support", "sources": []}

        # simple intent rephrase step
        prompt = f"Rephrase this into a concise search query for CSU TILT website: '{query}'"
        refined = self._query_llm(prompt)
        refined = refined.strip() if refined else query

        hits = self._retrieve(refined)
        if not hits:
            return {"answer": "No relevant content found on TILT site", "sources": []}

        top = hits[0]
        snippet = top["text"][:600].strip()
        return {
            "answer": snippet,
            "sources": [h["url"] for h in hits if h["url"]]
        }

def main():
    print("Router LLM Chat — type 'exit' to quit.\n")
    router = RouterLLM()
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        out = router.ask(q)
        print("\nBot:", out["answer"])
        if out["sources"]:
            print("Sources:")
            for s in out["sources"]:
                print("-", s)
        print()

if __name__ == "__main__":
    main()
