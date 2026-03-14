# rag_demo/chat.py
from __future__ import annotations

from .answer import retrieve, format_answer

def main():
    print("CLI RAG chat. Type 'exit' to quit")
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        hits = retrieve(q, k=4)  # raise or lower k if needed
        out = format_answer(q, hits)
        print("\nBot:\n" + out + "\n")

if __name__ == "__main__":
    main()
