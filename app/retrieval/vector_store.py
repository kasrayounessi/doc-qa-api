from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def build_vector_store(chunks: list[LCDocument]) -> FAISS:
    """
    Build an in-memory FAISS index from document chunks.

    Embeddings are computed once here; all questions in the request reuse
    the returned index.  The index is ephemeral — it exists only for the
    lifetime of the request.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return FAISS.from_documents(chunks, embeddings)
