from __future__ import annotations

import uuid
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types


def _extract_text_from_pdf(pdf_path: str) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_count + len(words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            overlap_words = current_words[-overlap:] if overlap else []
            current_words = overlap_words + words
            current_count = len(current_words)
        else:
            current_words.extend(words)
            current_count += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c.strip() for c in chunks if c.strip()]


class RAGPipeline:

    EMBED_MODEL = "models/gemini-embedding-001"
    COLLECTION = "docmind_collection"

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-1.5-flash",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        persist_dir: str = "/tmp/chromadb_docmind",
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = genai.Client(api_key=gemini_api_key)
        self.llm_model = model_name

        self._chroma = chromadb.Client(Settings(anonymized_telemetry=False))

        try:
            self._chroma.delete_collection(self.COLLECTION)
        except Exception:
            pass

        self._collection = self._chroma.create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_documents(self, pdf_paths: list[str]) -> int:
        total_chunks = 0
        for path in pdf_paths:
            source_name = Path(path).name
            raw_text = _extract_text_from_pdf(path)
            chunks = _chunk_text(raw_text, self.chunk_size, self.chunk_overlap)

            documents = []
            embeddings = []
            metadatas = []
            ids = []

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                emb = self._embed(chunk)
                doc_id = f"{source_name}_{idx}_{uuid.uuid4().hex[:8]}"
                documents.append(chunk)
                embeddings.append(emb)
                metadatas.append({"source": source_name, "chunk_id": idx})
                ids.append(doc_id)

            if documents:
                self._collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_chunks += len(documents)

        return total_chunks

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
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
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", 0),
                "score": 1 - dist,
            })
        return chunks

    def query(self, question: str, top_k: int = 4) -> dict[str, Any]:
        sources = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"[Source: {s['source']} | Chunk #{s['chunk_id']}]\n{s['text']}"
            for s in sources
        )

        prompt = f"""You are DocMind, an expert document assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: I could not find relevant information in the uploaded documents.
Be concise, accurate, and cite the source document when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )
        answer = response.text.strip()

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    def _embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.EMBED_MODEL,
            contents=text,
        )
        return result.embeddings[0].values