"""Unit tests for RAGPipeline in src/rag.py.

The LLM (Ollama) is mocked so tests run without a running model server.
The embedding model and FAISS index are exercised with real in-memory data
to verify the retrieval logic.
"""
import os
import types
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Prevent @traceable from hitting LangSmith during tests
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from src.rag import RAGPipeline, TOP_K


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_tmp_pdf(tmp_path, name="test.pdf"):
    """Write a minimal PDF to a pytest tmp_path directory and return the path."""
    pdf_path = str(tmp_path / name)
    text = "Hello world test document."
    content = (
        "%PDF-1.4\n"
        "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        f"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        f"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        f"4 0 obj<</Length {len(text) + 20}>>\nstream\nBT /F1 12 Tf 72 720 Td ({text}) Tj ET\nendstream\nendobj\n"
        "5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        "xref\n0 6\n0000000000 65535 f \n"
        "trailer<</Size 6/Root 1 0 R>>\n"
        "startxref\n0\n%%EOF\n"
    )
    with open(pdf_path, "w") as f:
        f.write(content)
    return pdf_path


def _fake_pages_property(text: str):
    """Return a property that makes PdfReader.pages yield one page with the given text."""
    return property(lambda self: [types.SimpleNamespace(extract_text=lambda: text)])


def _build_rag_with_text(text: str, tmp_path):
    """Build a RAGPipeline with a real FAISS index built from `text`."""
    from pypdf import PdfReader
    with patch("src.rag.ChatOllama"):
        rag = RAGPipeline()
    pdf_path = _write_tmp_pdf(tmp_path)
    with patch.object(PdfReader, "pages", new_callable=lambda: _fake_pages_property(text)):
        rag.load_pdf(pdf_path)
    return rag


# ── RAGPipeline.is_loaded ─────────────────────────────────────────────────

class TestIsLoaded:
    def test_false_before_loading(self):
        with patch("src.rag.ChatOllama"):
            rag = RAGPipeline()
        assert rag.is_loaded() is False

    def test_true_after_index_set(self):
        with patch("src.rag.ChatOllama"):
            rag = RAGPipeline()
        rag.index = MagicMock()
        assert rag.is_loaded() is True


# ── RAGPipeline.load_pdf ──────────────────────────────────────────────────

class TestLoadPdf:
    def test_load_pdf_populates_state(self, tmp_path):
        from pypdf import PdfReader
        long_text = "Data engineering is the practice of building data pipelines. " * 10
        with patch("src.rag.ChatOllama"):
            rag = RAGPipeline()
        pdf_path = _write_tmp_pdf(tmp_path)
        with patch.object(PdfReader, "pages", new_callable=lambda: _fake_pages_property(long_text)):
            info = rag.load_pdf(pdf_path)

        assert rag.is_loaded()
        assert len(rag.chunks) > 0
        assert rag.current_pdf == "test.pdf"
        assert info["pages"] == 1
        assert info["chunks"] == len(rag.chunks)
        assert info["filename"] == "test.pdf"

    def test_load_pdf_returns_metadata_keys(self, tmp_path):
        from pypdf import PdfReader
        with patch("src.rag.ChatOllama"):
            rag = RAGPipeline()
        pdf_path = _write_tmp_pdf(tmp_path)
        with patch.object(PdfReader, "pages", new_callable=lambda: _fake_pages_property("Some text " * 20)):
            info = rag.load_pdf(pdf_path)

        assert set(info.keys()) == {"filename", "pages", "chunks", "characters"}


# ── RAGPipeline._retrieve ─────────────────────────────────────────────────

class TestRetrieve:
    def test_retrieve_returns_k_chunks(self, tmp_path):
        text = (
            "Machine learning is a subset of artificial intelligence. " * 5 +
            "Python is widely used for data science and machine learning. " * 5 +
            "Neural networks are inspired by biological brains. " * 5
        )
        rag = _build_rag_with_text(text, tmp_path)
        result = rag._retrieve("What is machine learning?", k=2)
        assert len(result["chunks"]) == 2
        assert len(result["distances"]) == 2

    def test_retrieve_result_keys(self, tmp_path):
        text = "Neural networks learn representations from data. " * 10
        rag = _build_rag_with_text(text, tmp_path)
        result = rag._retrieve("neural networks", k=1)
        assert set(result.keys()) == {"question", "chunks", "distances"}
        assert result["question"] == "neural networks"

    def test_retrieve_distances_are_floats(self, tmp_path):
        text = "Python programming is popular for data science. " * 10
        rag = _build_rag_with_text(text, tmp_path)
        result = rag._retrieve("Python programming", k=3)
        assert all(isinstance(d, float) for d in result["distances"])


# ── RAGPipeline.query ─────────────────────────────────────────────────────

class TestQuery:
    def test_query_raises_if_no_pdf_loaded(self):
        with patch("src.rag.ChatOllama"):
            rag = RAGPipeline()
        with pytest.raises(ValueError, match="No PDF loaded"):
            rag.query("What is AI?")

    def test_query_returns_expected_keys(self, tmp_path):
        text = "Artificial intelligence enables machines to learn from data. " * 10
        rag = _build_rag_with_text(text, tmp_path)
        rag.chain = MagicMock()
        rag.chain.invoke.return_value = "Mocked answer about AI."

        result = rag.query("What is AI?")
        assert set(result.keys()) == {"question", "answer", "sources", "distances"}
        assert result["question"] == "What is AI?"
        assert result["answer"] == "Mocked answer about AI."
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) == TOP_K

    def test_query_passes_k_to_retrieve(self, tmp_path):
        text = "Deep learning uses neural networks with many layers. " * 10
        rag = _build_rag_with_text(text, tmp_path)
        rag.chain = MagicMock(return_value="Answer.")

        result = rag.query("neural networks", k=2)
        assert len(result["sources"]) == 2
        assert len(result["distances"]) == 2
