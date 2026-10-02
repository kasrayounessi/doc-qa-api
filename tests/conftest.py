"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Two-page in-memory PDF with extractable text."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 750, "Page one: The company Acme Corp was founded in 2010.")
    c.showPage()
    c.drawString(72, 750, "Page two: The CEO of Acme Corp is Jane Smith.")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_json_bytes() -> bytes:
    """Nested JSON document for ingestion tests."""
    data = {
        "company": {"name": "Acme Corp", "founded": 2010},
        "security": {
            "authentication": {"mfa_required": True, "provider": "Okta"}
        },
    }
    return json.dumps(data).encode("utf-8")


# ---------------------------------------------------------------------------
# OpenAI mocks — no real API calls ever made in tests
# ---------------------------------------------------------------------------


class _FakeEmbeddings(Embeddings):
    """
    Deterministic fake embeddings that pass FAISS's isinstance(Embeddings) check.

    Each document gets a unique vector seeded by a counter.  The query always
    gets the same vector (seed=999), which is fine for structural tests that
    verify count, metadata, and pipeline flow rather than ranking quality.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._counter = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for _ in texts:
            self._counter += 1
            v = np.random.default_rng(self._counter).random(self._dim).astype(np.float32)
            result.append(v.tolist())
        return result

    def embed_query(self, text: str) -> list[float]:
        return np.random.default_rng(999).random(self._dim).astype(np.float32).tolist()


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Patch OpenAIEmbeddings in vector_store with a fast, deterministic fake."""
    fake = _FakeEmbeddings(dim=16)
    monkeypatch.setattr(
        "app.retrieval.vector_store.OpenAIEmbeddings",
        lambda **kwargs: fake,
    )
    return fake


@pytest.fixture
def mock_llm_instance() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mocked answer from the document.")
    return llm


@pytest.fixture
def mock_llm(monkeypatch, mock_llm_instance: MagicMock) -> MagicMock:
    """Patch ChatOpenAI factory so no real LLM calls are made."""
    monkeypatch.setattr(
        "app.generation.llm.ChatOpenAI",
        lambda **kwargs: mock_llm_instance,
    )
    return mock_llm_instance


# ---------------------------------------------------------------------------
# API test client
# ---------------------------------------------------------------------------


@pytest.fixture
def client(mock_embeddings, mock_llm):
    """FastAPI TestClient with OpenAI fully mocked."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
