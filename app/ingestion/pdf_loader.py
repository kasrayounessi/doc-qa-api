from __future__ import annotations

import io

from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.logging import get_logger
from app.ingestion.base import BaseLoader
from app.models.document import InternalDocument

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    def load(self, file_bytes: bytes, filename: str) -> list[InternalDocument]:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except PdfReadError as exc:
            logger.warning("malformed_pdf", extra={"doc": filename, "error": str(exc)})
            raise HTTPException(status_code=400, detail=f"Malformed or unreadable PDF: {exc}")
        except Exception as exc:
            logger.warning("pdf_read_error", extra={"doc": filename, "error": str(exc)})
            raise HTTPException(status_code=400, detail=f"Could not open PDF: {exc}")

        if len(reader.pages) == 0:
            raise HTTPException(status_code=400, detail="PDF contains no pages.")

        docs: list[InternalDocument] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                logger.debug("page_extraction_failed", extra={"doc": filename, "page": page_num})
                text = ""

            text = text.strip()
            if not text:
                logger.debug("empty_page_skipped", extra={"doc": filename, "page": page_num})
                continue

            docs.append(
                InternalDocument(
                    content=text,
                    source=filename,
                    metadata={"page": page_num},
                )
            )

        if not docs:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PDF contains no extractable text. "
                    "Scanned or image-only PDFs require an OCR preprocessing stage "
                    "which is outside the scope of this service."
                ),
            )

        logger.info("pdf_loaded", extra={"doc": filename, "pages_with_text": len(docs)})
        return docs
