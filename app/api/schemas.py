from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.models.qa import QuestionInput


def normalize_questions(raw: Any) -> list[QuestionInput]:
    """
    Accept two question list formats and return a normalized list.

    Format A: [{"id": "q1", "question": "..."}, ...]
    Format B: ["question text", ...]

    Mixed formats in the same array are supported.
    """
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=400,
            detail="'questions' must be a JSON array.",
        )

    if len(raw) == 0:
        raise HTTPException(
            status_code=400,
            detail="'questions' array must not be empty.",
        )

    normalized: list[QuestionInput] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            if not item.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Question at index {i} is an empty string.",
                )
            normalized.append(QuestionInput(id=f"q{i + 1}", question=item.strip()))
        elif isinstance(item, dict):
            if "question" not in item:
                raise HTTPException(
                    status_code=400,
                    detail=f"Question at index {i} is missing the required 'question' field.",
                )
            if not str(item["question"]).strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Question at index {i} has an empty 'question' value.",
                )
            q_id = str(item["id"]) if "id" in item and item["id"] is not None else f"q{i + 1}"
            normalized.append(QuestionInput(id=q_id, question=str(item["question"]).strip()))
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Question at index {i} must be a string or an object, "
                    f"got {type(item).__name__}."
                ),
            )

    return normalized
