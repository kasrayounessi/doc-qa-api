from __future__ import annotations

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """\
You are a precise document question-answering assistant.

Rules you MUST follow without exception:
1. Answer ONLY using information explicitly present in the CONTEXT provided below.
2. Do NOT use any prior knowledge, training data, assumptions, or information outside the CONTEXT.
3. If the CONTEXT does not contain enough information to answer the question, respond with exactly:
   "The document does not contain sufficient information to answer this question."
4. Be concise and factual. Do not speculate or infer beyond what is stated.
5. When referencing specific details, use the exact wording from the CONTEXT.\
"""


def build_context(chunks: list[LCDocument]) -> str:
    """
    Format retrieved chunks into a labeled context block.

    Each chunk is clearly delimited with its provenance so the model can
    reference sources, and so humans debugging prompts can identify origins.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page")
        json_path = meta.get("json_path")

        if page is not None:
            header = f"[Chunk {i} | Source: {source} | Page: {page}]"
        elif json_path:
            header = f"[Chunk {i} | Source: {source} | Section: {json_path}]"
        else:
            header = f"[Chunk {i} | Source: {source}]"

        parts.append(f"{header}\n{chunk.page_content}")

    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str) -> list:
    human_content = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer based solely on the CONTEXT above:"
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
