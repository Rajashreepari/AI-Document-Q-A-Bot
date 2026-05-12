"""
RAG Pipeline — PDF ingestion, chunking, embedding, retrieval, and generation.
Uses ChromaDB for vector storage and Google Gemini for embeddings + generation.
"""

from __future__ import annotations

import os
import uuid
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import google.generativeai as genai


# ─── Helpers ────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        )


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks by approximate word count.
    Uses sentence boundaries where possible.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_count + len(words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap
            overlap_words = current_words[-overlap:] if overlap else []
            current_words = overlap_words + words
            current_count = len(current_words)
        else:
            current_words.extend(words)
            current_count += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c.strip() for c in chunks if c.strip()]


# ─── RAG Pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Workflow:
        1. ingest_documents() → extract text, chunk, embed, store in ChromaDB
        2. query()            → embed query, retrieve top-K chunks, generate answer
    """

    EMBED_MODEL = "models/text-embedding-004"
    COLLECTION  = "docmind_collection"

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-1.5-flash",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        persist_dir: str = "/tmp/chromadb_docmind",
    ):
        self.model_name    = model_name
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel(model_name)

        # ChromaDB (ephemeral per session; swap to PersistentClient for disk storage)
        self._chroma = chromadb.Client(Settings(anonymized_telemetry=False))

        # Drop old collection if exists to avoid duplicate IDs across re-indexing
        try:
            self._chroma.delete_collection(self.COLLECTION)
        except Exception:
            pass

        self._collection = self._chroma.create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_documents(self, pdf_paths: list[str]) -> int:
        """
        Extract, chunk, embed, and store all documents.
        Returns total chunk count.
        """
        total_chunks = 0
        for path in pdf_paths:
            source_name = Path(path).name
            raw_text = _extract_text_from_pdf(path)
            chunks = _chunk_text(raw_text, self.chunk_size, self.chunk_overlap)

            documents, embeddings, metadatas, ids = [], [], [], []

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                emb = self._embed(chunk)
                doc_id = f"{source_name}_{idx}_{uuid.uuid4().hex[:8]}"

                documents.append(chunk)
                embeddings.append(emb)
                metadatas.append({"source": source_name, "chunk_id": idx})
                ids.append(doc_id)

            # Batch add to ChromaDB
            if documents:
                self._collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_chunks += len(documents)

        return total_chunks

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Embed query and return top-K most relevant chunks."""
        q_emb = self._embed(query)
        results = self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "score":    1 - dist,   # cosine similarity
            })
        return chunks

    # ── Generation ─────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 4) -> dict[str, Any]:
        """Full RAG: retrieve relevant chunks, then generate a grounded answer."""
        sources = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"[Source: {s['source']} | Chunk #{s['chunk_id']}]\n{s['text']}"
            for s in sources
        )

        prompt = f"""You are DocMind, an expert document assistant. 
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."
Be concise, accurate, and cite the source document when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = self.llm.generate_content(prompt)
        answer   = response.text.strip()

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }

    # ── Embedding Helper ───────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Generate embedding vector using Gemini embedding model."""
        result = genai.embed_content(
            model=self.EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
