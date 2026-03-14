# rag_demo/serve.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from wsgiref.simple_server import make_server

from .answer import RAGAnswerer
from .utils import load_yaml

ROOT = Path(__file__).parent
CFG = load_yaml(ROOT / "config.yaml")
CHROMA_DIR = (ROOT / CFG["paths"]["chroma_dir"]).resolve()

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="Ask a question and print an answer")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port")
    ap.add_argument("--http", action="store_true", help="start tiny HTTP server")
    args = ap.parse_args()

    rag = RAGAnswerer(CHROMA_DIR)

    if args.q:
        out = rag.answer(args.q)
        print(json.dumps(out, indent=2))
        return

    if args.http:
        def app(environ, start_response):
            try:
                if environ["REQUEST_METHOD"] == "POST" and environ.get("PATH_INFO") == "/ask":
                    length = int(environ.get("CONTENT_LENGTH") or 0)
                    body = environ["wsgi.input"].read(length)
                    payload = json.loads(body.decode("utf-8"))
                    q = payload.get("q") or ""
                    out = rag.answer(q)
                    data = json.dumps(out).encode("utf-8")
                    start_response("200 OK", [("Content-Type", "application/json")])
                    return [data]
                start_response("404 Not Found", [("Content-Type", "text/plain")])
                return [b"not found"]
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
                return [str(e).encode("utf-8")]

        with make_server("", args.port, app) as httpd:
            print(f"listening on http://127.0.0.1:{args.port}")
            httpd.serve_forever()

if __name__ == "__main__":
    cli()
