from __future__ import annotations

from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.base import BaseLoader
from app.ingestion.json_loader import JSONLoader
from app.ingestion.pdf_loader import PDFLoader

logger = get_logger(__name__)

_LOADERS: dict[str, BaseLoader] = {
    "pdf": PDFLoader(),
    "json": JSONLoader(),
}


def ingest_file(file_bytes: bytes, filename: str) -> list[LCDocument]:
    """
    Full ingestion pipeline: raw bytes → parsed sections → text chunks.

    The returned LangChain Documents carry source metadata through every chunk,
    ready for embedding and FAISS indexing.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    loader = _LOADERS.get(ext)
    if loader is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Supported formats: pdf, json.",
        )

    internal_docs = loader.load(file_bytes, filename)
    lc_docs = [doc.to_langchain() for doc in internal_docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # split_documents copies metadata from each source Document onto all its chunks
    chunks: list[LCDocument] = splitter.split_documents(lc_docs)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document produced no text chunks after splitting.",
        )

    logger.info(
        "document_ingested",
        extra={
            "doc": filename,
            "source_sections": len(lc_docs),
            "chunks": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )
    return chunks
