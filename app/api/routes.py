from __future__ import annotations

import json

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import normalize_questions
from app.core.config import settings
from app.core.logging import get_logger
from app.generation.llm import get_llm
from app.generation.qa_service import QAService
from app.ingestion.service import ingest_file
from app.models.qa import QAResponse
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.vector_store import build_vector_store

router = APIRouter()
logger = get_logger(__name__)


@router.post("/qa", response_model=QAResponse)
async def qa_endpoint(
    document_file: UploadFile = File(..., description="PDF or JSON document to query"),
    questions_file: UploadFile = File(..., description="JSON file containing questions"),
) -> QAResponse:
    """
    Answer a list of questions grounded in the uploaded document.

    Both files are required. The document may be a PDF or JSON file.
    The questions file must be a JSON array in Format A or Format B.
    """
    logger.info(
        "request_received",
        extra={
            "document": document_file.filename,
            "questions_file": questions_file.filename,
        },
    )

    # --- 1. Read and size-check the document ---
    doc_bytes = await document_file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(doc_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Document exceeds the {settings.max_upload_size_mb} MB upload limit.",
        )

    # --- 2. Parse questions ---
    questions_bytes = await questions_file.read()
    try:
        raw_questions = json.loads(questions_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Questions file is not valid JSON: {exc}",
        )

    normalized_questions = normalize_questions(raw_questions)
    logger.info("questions_parsed", extra={"count": len(normalized_questions)})

    # --- 3. Ingest document (parse → chunk) ---
    doc_filename = document_file.filename or "upload"
    chunks = ingest_file(doc_bytes, doc_filename)

    # --- 4. Build FAISS index once for all questions ---
    try:
        vector_store = build_vector_store(chunks)
    except Exception as exc:
        logger.error("vector_store_build_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500,
            detail="Failed to build the vector index. Check that OPENAI_API_KEY is set.",
        )

    retriever = DocumentRetriever(vector_store)
    llm = get_llm()
    qa_service = QAService(retriever=retriever, llm=llm)

    # --- 5. Answer each question, reusing the same index ---
    results = []
    for q in normalized_questions:
        try:
            result = qa_service.answer_question(q.id, q.question)
        except Exception as exc:
            logger.error(
                "question_failed",
                extra={"question_id": q.id, "error": str(exc)},
            )
            raise HTTPException(
                status_code=500,
                detail=f"LLM error while answering question '{q.id}'.",
            )
        results.append(result)

    return QAResponse(results=results)
