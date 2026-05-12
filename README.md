# 🧠 DocMind AI — Document Q&A Bot

> RAG-based intelligent query system powered by **Google Gemini** + **ChromaDB**

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 PDF Ingestion | Upload multiple PDFs at once |
| 🧩 Smart Chunking | Sentence-aware overlapping chunks |
| 🔢 Semantic Embeddings | Google `embedding-001` model |
| 🗄️ Vector Store | ChromaDB with cosine similarity |
| 🤖 LLM Generation | Gemini 1.5 Flash / Pro / 2.0 Flash |
| 🎨 Beautiful UI | Dark-theme Streamlit app |
| 📎 Source Attribution | Every answer shows retrieved chunks + scores |
| 🕓 Query History | Session-level conversation tracking |

---

## 🏗️ Architecture

```
PDF Files
   │
   ▼
Text Extraction (PyMuPDF)
   │
   ▼
Chunking (sentence-aware, overlapping)
   │
   ▼
Embedding (Gemini embedding-001)
   │
   ▼
ChromaDB Vector Store
   │
   ▼
User Query ──► Embed Query ──► Cosine Similarity Search ──► Top-K Chunks
                                                                │
                                                                ▼
                                                    Prompt + Context
                                                                │
                                                                ▼
                                                    Gemini LLM Generation
                                                                │
                                                                ▼
                                                    Structured Answer + Sources
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/ai-doc-qa-bot.git
cd ai-doc-qa-bot
pip install -r requirements.txt
```

### 2. Get a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a free API key
3. Either enter it in the sidebar at runtime, or:

```bash
cp .env.example .env
# Edit .env and add your key
export GEMINI_API_KEY=your_key_here
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
ai-doc-qa-bot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG logic
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .streamlit/
│   └── config.toml     # Streamlit dark theme
└── README.md
```

---

## ⚙️ Configuration (Sidebar)

| Parameter | Default | Description |
|---|---|---|
| Gemini API Key | — | Required for embeddings + generation |
| Model | `gemini-1.5-flash` | LLM model selection |
| Chunk Size | 512 tokens | Words per chunk |
| Chunk Overlap | 64 tokens | Overlap between consecutive chunks |
| Top-K Retrieval | 4 | Number of chunks retrieved per query |

---

## 🔧 Key Modules

### `rag_pipeline.py` — `RAGPipeline`

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline(
    gemini_api_key="your_key",
    model_name="gemini-1.5-flash",
    chunk_size=512,
    chunk_overlap=64,
)

# Ingest PDFs
total_chunks = rag.ingest_documents(["report.pdf", "manual.pdf"])

# Query
result = rag.query("What is the refund policy?", top_k=4)
print(result["answer"])
for src in result["sources"]:
    print(src["source"], src["score"])
```

---

## 📦 Dependencies

```
streamlit          — Web UI framework
google-generativeai — Gemini API (embeddings + LLM)
chromadb           — Local vector store
pymupdf            — PDF text extraction
```

---

## 💡 How It Works

1. **Ingestion** — PDFs are parsed page by page using PyMuPDF
2. **Chunking** — Text is split into overlapping windows at sentence boundaries
3. **Embedding** — Each chunk is embedded using Gemini's `embedding-001`
4. **Indexing** — Embeddings are stored in ChromaDB with cosine similarity index
5. **Retrieval** — User query is embedded; top-K chunks retrieved by similarity score
6. **Generation** — Chunks + query are passed to Gemini with a grounded prompt
7. **Response** — Answer and source attribution returned to the UI

---

## 🛠️ Extending the Project

- **Persistent storage** — Replace `chromadb.Client()` with `chromadb.PersistentClient(path=...)`
- **More file types** — Add `.docx`, `.txt`, `.md` extractors in `rag_pipeline.py`
- **Reranking** — Add a cross-encoder reranker after initial retrieval
- **Chat mode** — Maintain conversation history in session state for multi-turn Q&A
- **Authentication** — Add Streamlit authenticator for multi-user deployment

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
