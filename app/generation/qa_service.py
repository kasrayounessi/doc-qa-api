from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger
from app.generation.prompts import build_context, build_messages
from app.models.qa import QAResult, SourceReference
from app.retrieval.retriever import DocumentRetriever

logger = get_logger(__name__)


class QAService:
    """
    Orchestrates the four explicit RAG stages for a single question:
      1. Retrieval  — fetch relevant chunks from the vector store
      2. Context    — format chunks into a labeled context block
      3. Generation — send context + question to the LLM
      4. Sources    — extract and deduplicate provenance from retrieved chunks
    """

    def __init__(self, retriever: DocumentRetriever, llm: ChatOpenAI) -> None:
        self._retriever = retriever
        self._llm = llm

    def answer_question(self, question_id: str, question: str) -> QAResult:
        # Stage 1: Retrieval
        retrieved_chunks: list[LCDocument] = self._retriever.retrieve(question)

        # Stage 2: Context assembly
        context: str = build_context(retrieved_chunks)

        # Stage 3: Generation
        messages = build_messages(question=question, context=context)
        response = self._llm.invoke(messages)
        answer: str = response.content.strip()

        # Stage 4: Source extraction + deduplication
        sources = self._extract_sources(retrieved_chunks)

        logger.info(
            "question_answered",
            extra={
                "question_id": question_id,
                "chunks_retrieved": len(retrieved_chunks),
                "sources": len(sources),
            },
        )

        return QAResult(
            id=question_id,
            question=question,
            answer=answer,
            sources=sources,
        )

    def _extract_sources(self, chunks: list[LCDocument]) -> list[SourceReference]:
        seen: set[tuple] = set()
        sources: list[SourceReference] = []
        for chunk in chunks:
            meta = chunk.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page")
            json_path = meta.get("json_path")
            key = (source, page, json_path)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceReference(source=source, page=page, json_path=json_path)
                )
        return sources
