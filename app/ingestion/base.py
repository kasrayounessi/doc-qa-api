from abc import ABC, abstractmethod

from app.models.document import InternalDocument


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_bytes: bytes, filename: str) -> list[InternalDocument]:
        """Parse raw file bytes into a list of normalized InternalDocuments."""
        ...
