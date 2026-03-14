# 🤖 RAG Chatbot — Retrieval-Augmented Generation with Ollama + Mistral + ChromaDB

A local, privacy-first RAG (Retrieval-Augmented Generation) chatbot that scrapes website content and ingests documents, embeds them into a vector store, and answers natural language questions grounded entirely in that data — no hallucinations, no external API calls.

---

## 🧠 How It Works

```
Website / PDFs
      ↓
  scraper.py / ingest.py        ← scrape & chunk content
      ↓
  embed_data.py                 ← embed chunks via Ollama
      ↓
  ChromaDB (vector store)       ← store embeddings locally
      ↓
  query.py                      ← semantic search on user question
      ↓
  Mistral (via Ollama)          ← generate answer from retrieved context
      ↓
  answer.py                     ← return grounded response to user
```

---

## ✨ Features

- 🕷️ **Web scraping** — crawls and extracts content from target websites
- 📄 **Document ingestion** — supports PDF and local file ingestion
- 🔍 **Semantic search** — finds the most relevant chunks using vector similarity
- 💬 **Natural language Q&A** — answers questions based strictly on ingested data
- 🔒 **Fully local** — runs entirely on your machine, no OpenAI or external APIs
- ⚡ **Fast retrieval** — ChromaDB vector store with persistent SQLite backend

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | [Mistral](https://mistral.ai/) via [Ollama](https://ollama.com/) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | Ollama embedding models |
| Scraping | Python (BeautifulSoup / requests) |
| Backend | Python, FastAPI |
| Storage | ChromaDB + SQLite |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Mistral model pulled:
```bash
ollama pull mistral
```

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Step 1 — Scrape the website:**
```bash
python scraper.py
```

**Step 2 — Ingest & embed the data:**
```bash
python embed_data.py
```

**Step 3 — Start chatting:**
```bash
python app.py
```

Then type your question in the terminal and get answers grounded in the scraped content.

---

## 📁 Project Structure

```
rag-chatbot/
├── rag_demo/
│   ├── answer.py           # Formats and returns final answer
│   ├── api.py              # API layer
│   ├── chat.py             # Chat loop logic
│   ├── ingest.py           # Document ingestion pipeline
│   ├── query.py            # Semantic search against vector store
│   ├── router.py           # Request routing
│   ├── serve.py            # Server entry point
│   └── utils.py            # Helper utilities
├── scraper.py              # Website crawler & content extractor
├── embed_data.py           # Embedding pipeline
├── app.py                  # Main entry point
└── requirements.txt
```

---

## 💡 Motivation

I built this to understand how RAG pipelines actually work under the hood — not just as an API consumer, but by owning every layer from scraping and chunking to embedding and retrieval. Running everything locally with Ollama meant I could experiment freely without API costs or data privacy concerns.

---

## 📌 Roadmap

- [ ] Add a web UI (React frontend)
- [ ] Support multi-document uploads
- [ ] Add chat history / conversation memory
- [ ] Streaming responses

