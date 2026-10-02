from __future__ import annotations

from pydantic import BaseModel


class QuestionInput(BaseModel):
    id: str
    question: str


class SourceReference(BaseModel):
    source: str
    page: int | None = None
    json_path: str | None = None


class QAResult(BaseModel):
    id: str
    question: str
    answer: str
    sources: list[SourceReference]


class QAResponse(BaseModel):
    results: list[QAResult]
