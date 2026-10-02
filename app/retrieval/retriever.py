from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

from app.core.config import settings


class DocumentRetriever:
    """
    Thin wrapper around a FAISS vector store.

    Keeping retrieval explicit (rather than using .as_retriever()) makes
    each RAG stage independently testable and the retrieval call transparent.
    Swap the vector store implementation by changing only this class.
    """

    def __init__(self, vector_store: FAISS, top_k: int | None = None) -> None:
        self._vs = vector_store
        self._top_k = top_k if top_k is not None else settings.retrieval_top_k

    def retrieve(self, query: str) -> list[LCDocument]:
        """Return the top_k most relevant chunks for the query."""
        return self._vs.similarity_search(query, k=self._top_k)
