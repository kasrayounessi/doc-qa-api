"""Tests for the QA service — explicit RAG stage verification."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage

from app.generation.qa_service import QAService


def _make_retriever(chunks: list[LCDocument]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    return mock


def _make_llm(answer: str = "Mocked answer.") -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=answer)
    return mock


def _make_chunk(text: str, source: str = "doc.pdf", page: int = 1) -> LCDocument:
    return LCDocument(page_content=text, metadata={"source": source, "page": page})


# ---------------------------------------------------------------------------
# Stage 1: Retrieval called correctly
# ---------------------------------------------------------------------------


def test_retriever_called_with_question_text():
    retriever = _make_retriever([_make_chunk("Some context")])
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    svc.answer_question("q1", "What is the policy?")

    retriever.retrieve.assert_called_once_with("What is the policy?")


# ---------------------------------------------------------------------------
# Stage 2: Context flows from retrieved chunks to the LLM
# ---------------------------------------------------------------------------


def test_retrieved_content_appears_in_llm_context():
    chunk_text = "MFA is mandatory for all administrative users."
    retriever = _make_retriever([_make_chunk(chunk_text)])
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    svc.answer_question("q1", "Is MFA required?")

    call_args = llm.invoke.call_args
    messages = call_args[0][0]
    # The human message should embed the chunk text in the context
    human_message_content = messages[1].content
    assert chunk_text in human_message_content


def test_grounding_system_prompt_is_present():
    retriever = _make_retriever([_make_chunk("Some text")])
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    svc.answer_question("q1", "Any question?")

    messages = llm.invoke.call_args[0][0]
    system_content = messages[0].content
    assert "ONLY" in system_content
    assert "CONTEXT" in system_content


# ---------------------------------------------------------------------------
# Stage 3: Answer comes from LLM response
# ---------------------------------------------------------------------------


def test_answer_is_llm_response_content():
    retriever = _make_retriever([_make_chunk("context")])
    llm = _make_llm("Passwords must be 12+ characters.")
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q1", "What is the password policy?")

    assert result.answer == "Passwords must be 12+ characters."


def test_result_fields_are_populated():
    retriever = _make_retriever([_make_chunk("context")])
    llm = _make_llm("Some answer.")
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q42", "A question?")

    assert result.id == "q42"
    assert result.question == "A question?"
    assert result.answer == "Some answer."


# ---------------------------------------------------------------------------
# Stage 4: Source extraction and deduplication
# ---------------------------------------------------------------------------


def test_sources_contain_chunk_metadata():
    retriever = _make_retriever([_make_chunk("text", source="policy.pdf", page=3)])
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q1", "Q?")

    assert len(result.sources) == 1
    assert result.sources[0].source == "policy.pdf"
    assert result.sources[0].page == 3


def test_sources_are_deduplicated():
    # Two chunks from the same page → only one source reference
    chunks = [
        _make_chunk("first chunk on page 4", source="report.pdf", page=4),
        _make_chunk("second chunk on page 4", source="report.pdf", page=4),
    ]
    retriever = _make_retriever(chunks)
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q1", "Q?")

    assert len(result.sources) == 1
    assert result.sources[0].page == 4


def test_sources_from_different_pages_are_distinct():
    chunks = [
        _make_chunk("chunk on page 1", source="doc.pdf", page=1),
        _make_chunk("chunk on page 2", source="doc.pdf", page=2),
    ]
    retriever = _make_retriever(chunks)
    llm = _make_llm()
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q1", "Q?")

    assert len(result.sources) == 2


def test_no_retrieved_chunks_returns_empty_sources():
    retriever = _make_retriever([])
    llm = _make_llm("Cannot answer.")
    svc = QAService(retriever=retriever, llm=llm)

    result = svc.answer_question("q1", "Q?")

    assert result.sources == []
