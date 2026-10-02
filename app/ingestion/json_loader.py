from __future__ import annotations

import json

from fastapi import HTTPException

from app.core.logging import get_logger
from app.ingestion.base import BaseLoader
from app.models.document import InternalDocument

logger = get_logger(__name__)


class JSONLoader(BaseLoader):
    def load(self, file_bytes: bytes, filename: str) -> list[InternalDocument]:
        try:
            raw = json.loads(file_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON document: {exc}")

        if not raw and raw != 0:
            raise HTTPException(status_code=400, detail="JSON document is empty.")

        flat_lines = self._flatten(raw)
        if not flat_lines:
            raise HTTPException(status_code=400, detail="JSON document produced no content after flattening.")

        blocks = self._group_into_blocks(flat_lines)
        docs = [
            InternalDocument(
                content=block_text,
                source=filename,
                metadata={"json_path": json_path},
            )
            for json_path, block_text in blocks
        ]

        logger.info(
            "json_loaded",
            extra={
                "doc": filename,
                "top_level_sections": len(docs),
                "total_lines": len(flat_lines),
            },
        )
        return docs

    def _flatten(self, obj: object, prefix: str = "") -> list[str]:
        """
        Recursively flatten a JSON value into 'dotted.path: value' lines.

        - dicts: recurse with dot-joined key
        - lists: recurse with bracket-index notation
        - None: emit 'prefix: null'
        - bool: emit lowercased ('true'/'false')
        - everything else: emit str(value)
        """
        lines: list[str] = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                lines.extend(self._flatten(value, child_prefix))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                child_prefix = f"{prefix}[{i}]"
                lines.extend(self._flatten(item, child_prefix))
        elif obj is None:
            lines.append(f"{prefix}: null")
        elif isinstance(obj, bool):
            # Must be before int check — bool is a subclass of int in Python
            lines.append(f"{prefix}: {str(obj).lower()}")
        else:
            lines.append(f"{prefix}: {obj}")

        return lines

    def _group_into_blocks(self, flat_lines: list[str]) -> list[tuple[str, str]]:
        """
        Group flattened lines by their top-level key so that logical JSON
        sections are kept together before chunking.

        Returns list of (top_level_key, block_text).
        """
        groups: dict[str, list[str]] = {}
        for line in flat_lines:
            # Top-level key ends at the first '.' or '['
            top_key = line.split(".")[0].split("[")[0]
            groups.setdefault(top_key, []).append(line)

        return [(key, "\n".join(lines)) for key, lines in groups.items()]
