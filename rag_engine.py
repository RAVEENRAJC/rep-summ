"""Core document processing and RAG services for the Streamlit application."""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from dotenv import load_dotenv

load_dotenv()

StatusCallback = Callable[[int, str, str], None]


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    section: str = "Document content"

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class EmbeddingService:
    """Creates local normalized text vectors without needing a separate model download."""

    def __init__(self) -> None:
        self._vectorizer = None

    def fit_encode(self, texts: Iterable[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=8192, ngram_range=(1, 2))
        matrix = self._vectorizer.fit_transform(list(texts)).astype("float32")
        return normalize(matrix, norm="l2").toarray().astype("float32")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise ValueError("The embedding index has not been initialized.")
        from sklearn.preprocessing import normalize

        matrix = self._vectorizer.transform(list(texts)).astype("float32")
        return normalize(matrix, norm="l2").toarray().astype("float32")


class VectorStore:
    """In-memory semantic index with FAISS when it is available."""

    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self._index = None
        try:
            import faiss

            self._index = faiss.IndexFlatIP(embeddings.shape[1])
            self._index.add(embeddings)
        except ImportError:
            # The NumPy implementation preserves functionality if FAISS is unavailable.
            self._index = None

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        limit = min(top_k, len(self.chunks))
        if not limit:
            return []
        if self._index is not None:
            scores, indices = self._index.search(query_vector.reshape(1, -1), limit)
            pairs = zip(indices[0], scores[0])
        else:
            scores = self.embeddings @ query_vector
            indices = np.argsort(scores)[::-1][:limit]
            pairs = ((int(index), float(scores[index])) for index in indices)
        return [RetrievedChunk(self.chunks[int(index)], float(score)) for index, score in pairs]


class RAGPipeline:
    def __init__(self, status_callback: StatusCallback | None = None) -> None:
        self.status_callback = status_callback or (lambda *_: None)
        self.embedder = EmbeddingService()
        self.store: VectorStore | None = None
        self.chunks: list[Chunk] = []
        self.document_stats: dict[str, int] = {}
        configured_keys = os.getenv("OPENROUTER_API_KEYS", "")
        self.api_keys = [key.strip() for key in configured_keys.replace("\n", ",").split(",") if key.strip()]
        if not self.api_keys and os.getenv("OPENROUTER_API_KEY"):
            self.api_keys = [os.environ["OPENROUTER_API_KEY"].strip()]
        self.active_key_index = 0

    def _status(self, module: int, state: str, detail: str) -> None:
        self.status_callback(module, state, detail)

    @staticmethod
    def _extract_pdf(uploaded_file) -> list[tuple[int, str]]:
        import fitz

        document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        return [(page.number + 1, page.get_text("text")) for page in document]

    @staticmethod
    def _extract_docx(uploaded_file) -> list[tuple[int, str]]:
        from docx import Document

        doc = Document(io.BytesIO(uploaded_file.getvalue()))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        return [(1, text)]

    @staticmethod
    def _extract_txt(uploaded_file) -> list[tuple[int, str]]:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        return [(1, raw)]

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if len(line) > 1]
        counts = Counter(lines)
        # Repeated short lines usually represent page headers or footers.
        lines = [line for line in lines if not (counts[line] > 2 and len(line) < 100)]
        return "\n".join(lines)

    @staticmethod
    def _section_for(text: str, current: str) -> str:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line.isupper() and 3 < len(first_line) < 90:
            return first_line.title()
        return current

    @staticmethod
    def _split_chunks(text: str, size: int = 900, overlap: int = 150) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                breakpoint = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if breakpoint > start + 300:
                    end = breakpoint + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def ingest(self, uploaded_files) -> dict[str, int]:
        if not uploaded_files:
            raise ValueError("Upload at least one PDF, DOCX, or TXT file.")

        self._status(1, "active", "Validating uploaded files")
        accepted = {"pdf", "docx", "txt"}
        for file in uploaded_files:
            extension = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
            if extension not in accepted:
                raise ValueError(f"{file.name} is not a supported file type.")
            if file.size > 25 * 1024 * 1024:
                raise ValueError(f"{file.name} exceeds the 25 MB limit.")
        self._status(1, "done", f"Validated and stored {len(uploaded_files)} document(s)")

        self._status(2, "active", "Extracting text and page information")
        extracted: list[tuple[str, int, str]] = []
        for file in uploaded_files:
            extension = file.name.rsplit(".", 1)[-1].lower()
            if extension == "pdf":
                pages = self._extract_pdf(file)
            elif extension == "docx":
                pages = self._extract_docx(file)
            else:
                pages = self._extract_txt(file)
            extracted.extend((file.name, page, text) for page, text in pages if text.strip())
        if not extracted:
            raise ValueError("No readable text was found in the uploaded document(s).")
        self.document_stats["pages"] = len(extracted)
        self._status(2, "done", f"Extracted text from {len(extracted)} page(s)")

        self._status(3, "active", "Cleaning text and identifying sections")
        cleaned = [(name, page, self._clean_text(text)) for name, page, text in extracted]
        cleaned = [(name, page, text) for name, page, text in cleaned if text]
        self._status(3, "done", f"Prepared {len(cleaned)} text segment(s)")

        self._status(4, "active", "Creating chunks, embeddings, and vector index")
        self.chunks = []
        section = "Document content"
        for name, page, text in cleaned:
            section = self._section_for(text, section)
            self.chunks.extend(
                Chunk(chunk_text, name, page, section)
                for chunk_text in self._split_chunks(text)
            )
        if not self.chunks:
            raise ValueError("The extracted text could not be converted into searchable chunks.")
        embeddings = self.embedder.fit_encode(chunk.text for chunk in self.chunks)
        self.store = VectorStore(embeddings, self.chunks)
        self.document_stats["chunks"] = len(self.chunks)
        self._status(4, "done", f"Indexed {len(self.chunks)} chunks in the vector database")
        return self.document_stats

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.store is None:
            raise ValueError("Upload and process documents before asking a question.")
        if not query.strip():
            raise ValueError("Enter a question or report request.")
        self._status(5, "active", "Searching the vector database for relevant evidence")
        results = self.store.search(self.embedder.encode([query])[0], top_k)
        self._status(5, "done", f"Retrieved {len(results)} relevant evidence chunk(s)")
        return results

    @staticmethod
    def _context(results: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[Evidence {index} | Source: {result.chunk.citation} | Section: {result.chunk.section}]\n{result.chunk.text}"
            for index, result in enumerate(results, 1)
        )

    @staticmethod
    def _openrouter_client(api_key: str):
        from openai import OpenAI

        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    @staticmethod
    def _can_rotate_key(error: Exception) -> bool:
        """Only switch keys for quota, authentication, or rate-limit failures."""
        message = str(error).lower()
        retry_signals = ("429", "rate limit", "rate_limit", "quota", "credit", "insufficient", "401", "invalid api key")
        return any(signal in message for signal in retry_signals)

    def _generate(self, prompt: str) -> str:
        if not self.api_keys:
            raise ValueError("No OpenRouter API key found. Add OPENROUTER_API_KEYS or OPENROUTER_API_KEY to .env and restart Streamlit.")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        last_error: Exception | None = None
        for attempt in range(len(self.api_keys)):
            key_index = (self.active_key_index + attempt) % len(self.api_keys)
            api_key = self.api_keys[key_index]
            try:
                client = self._openrouter_client(api_key)
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence-grounded document analyst. Use only supplied evidence. "
                                "For every factual claim, add a citation using exactly this format: "
                                "[Source: filename, page N]. If evidence is insufficient, say so clearly."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                self.active_key_index = key_index
                return response.choices[0].message.content or "No response was generated."
            except Exception as error:
                last_error = error
                if attempt < len(self.api_keys) - 1 and self._can_rotate_key(error):
                    self.active_key_index = (key_index + 1) % len(self.api_keys)
                    continue
                raise
        raise RuntimeError(f"All configured OpenRouter API keys failed: {last_error}")

    def answer(self, query: str) -> tuple[str, list[RetrievedChunk], int]:
        evidence = self.retrieve(query)
        self._status(6, "active", "Generating a grounded answer from retrieved evidence")
        answer = self._generate(f"Question: {query}\n\nEvidence:\n{self._context(evidence)}")
        self._status(6, "done", "Generated grounded answer")
        confidence = self.verify(answer, evidence)
        return answer, evidence, confidence

    def report(self, request: str) -> tuple[str, list[RetrievedChunk], int]:
        evidence = self.retrieve(request, top_k=8)
        self._status(6, "active", "Generating a structured report from retrieved evidence")
        prompt = (
            f"Create a concise structured report for: {request}\n\n"
            "Use these exact headings when relevant: Executive Summary, Key Findings, Important Metrics, "
            "Risks and Issues, Recommendations. Do not invent facts. Cite claims.\n\n"
            f"Evidence:\n{self._context(evidence)}"
        )
        report = self._generate(prompt)
        self._status(6, "done", "Generated structured report")
        confidence = self.verify(report, evidence)
        return report, evidence, confidence

    def verify(self, generated_text: str, evidence: list[RetrievedChunk]) -> int:
        self._status(7, "active", "Checking evidence coverage and confidence")
        cited_sources = sum(1 for item in evidence if item.chunk.source in generated_text)
        citation_coverage = cited_sources / max(len(evidence), 1)
        relevance = float(np.mean([max(item.score, 0) for item in evidence]))
        score = round(min(98, max(35, (0.65 * relevance + 0.35 * citation_coverage) * 100)))
        self._status(7, "done", f"Evidence verified with {score}% confidence")
        return score
