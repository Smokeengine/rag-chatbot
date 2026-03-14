# rag_demo/api.py
from __future__ import annotations
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from .router import build_router

app = FastAPI(title="TILT RAG API")

router = build_router()

class AskIn(BaseModel):
    query: str
    k: Optional[int] = 5
    min_sim: Optional[float] = 0.58

class AskOut(BaseModel):
    answer: str
    sources: List[str]
    mode: str

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<meta charset="utf-8">
<title>TILT RAG</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto }
#out { white-space: pre-wrap; margin-top: 1rem }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 4px }
</style>
<h1>TILT RAG chat</h1>
<p>Type a question about tutoring, writing center, workshops, study groups, policies</p>
<input id="q" style="width:100%" placeholder="eg writing center hours">
<button id="ask">Ask</button>
<pre id="out"></pre>
<script>
const q = document.getElementById("q")
const out = document.getElementById("out")
document.getElementById("ask").onclick = async () => {
  const body = { query: q.value, k: 5, min_sim: 0.6 }
  const r = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  const j = await r.json()
  out.textContent = "Mode: " + j.mode + "\\n\\n" + j.answer + "\\n\\nSources\\n" + (j.sources || []).join("\\n")
}
</script>
"""

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    out = router.ask(payload.query, k=payload.k or 5, min_sim=payload.min_sim or 0.58)
    return AskOut(answer=out["answer"], sources=out.get("sources", []), mode=out.get("mode", "retrieval"))
