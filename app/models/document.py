from __future__ import annotations

from langchain_core.documents import Document
from pydantic import BaseModel


class InternalDocument(BaseModel):
    """Normalized representation of a parsed document section."""

    content: str
    source: str
    metadata: dict = {}

    def to_langchain(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={"source": self.source, **self.metadata},
        )
