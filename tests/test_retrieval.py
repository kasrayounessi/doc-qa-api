"""Tests for retrieval (FAISS vector store + DocumentRetriever)."""
from __future__ import annotations

from langchain_core.documents import Document as LCDocument

from app.retrieval.retriever import DocumentRetriever
from app.retrieval.vector_store import build_vector_store


def _make_chunks(texts: list[str], source: str = "test.pdf") -> list[LCDocument]:
    return [
        LCDocument(page_content=t, metadata={"source": source, "page": i + 1})
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Vector store construction
# ---------------------------------------------------------------------------


def test_build_vector_store_returns_faiss(mock_embeddings):
    from langchain_community.vectorstores import FAISS

    chunks = _make_chunks(["alpha content", "beta content", "gamma content"])
    vs = build_vector_store(chunks)
    assert isinstance(vs, FAISS)


# ---------------------------------------------------------------------------
# Retriever top_k behavior
# ---------------------------------------------------------------------------


def test_retrieve_respects_default_top_k(mock_embeddings):
    # Build a corpus larger than the default top_k (4)
    chunks = _make_chunks([f"document chunk number {i}" for i in range(10)])
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs)

    results = retriever.retrieve("some query")
    assert len(results) == 4


def test_retrieve_respects_custom_top_k(mock_embeddings):
    chunks = _make_chunks([f"chunk {i}" for i in range(8)])
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs, top_k=2)

    results = retriever.retrieve("some query")
    assert len(results) == 2


def test_retrieve_top_k_capped_at_corpus_size(mock_embeddings):
    # Corpus smaller than top_k — should return all available
    chunks = _make_chunks(["only one chunk"])
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs, top_k=10)

    results = retriever.retrieve("some query")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_retrieved_chunks_have_metadata(mock_embeddings):
    chunks = _make_chunks(["content about security policy"], source="policy.pdf")
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs, top_k=1)

    results = retriever.retrieve("security")
    assert results[0].metadata["source"] == "policy.pdf"
    assert "page" in results[0].metadata


def test_retrieved_chunks_are_langchain_documents(mock_embeddings):
    chunks = _make_chunks(["some document text"])
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs, top_k=1)

    results = retriever.retrieve("query")
    assert all(isinstance(r, LCDocument) for r in results)
