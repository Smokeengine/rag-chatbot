from fastapi import FastAPI
from query import ask

app = FastAPI()

@app.get("/chat")
def chat(query: str):
    return {"response": ask(query)}
