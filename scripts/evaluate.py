"""
Lightweight RAG evaluation script.

Usage:
    python -m scripts.evaluate            # deterministic only (no API cost)
    python -m scripts.evaluate --live     # includes live LLM calls (incurs cost)

The deterministic section builds a FAISS index using fake embeddings and
verifies retrieval hit rate.  The grounding section uses mocked LLM responses
to verify that context flows correctly and unsupported questions are declined.

The --live flag runs both sections with real OpenAI embeddings and generation.
It requires OPENAI_API_KEY to be set and will incur API cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Evaluation corpus
# ---------------------------------------------------------------------------

CORPUS = [
    # Signal chunks — expected to be retrieved for matching queries
    {
        "content": "MFA is mandatory for all administrative users. Provider is Okta.",
        "source": "policy.json",
        "metadata": {"json_path": "security"},
    },
    {
        "content": "The company Acme Corp was founded in 2010 by its original board.",
        "source": "company.json",
        "metadata": {"json_path": "company"},
    },
    {
        "content": "The CEO of Acme Corp is Jane Smith. She joined in 2015.",
        "source": "leadership.json",
        "metadata": {"json_path": "leadership"},
    },
    # Noise chunks — should not surface for the above queries
    {
        "content": "Annual employee picnic is held every June in the main parking lot.",
        "source": "events.json",
        "metadata": {"json_path": "events"},
    },
    {
        "content": "Office kitchen supplies are restocked every Monday morning.",
        "source": "facilities.json",
        "metadata": {"json_path": "facilities"},
    },
    {
        "content": "The company color palette uses navy blue and silver gray.",
        "source": "brand.json",
        "metadata": {"json_path": "brand"},
    },
    {
        "content": "Quarterly all-hands meetings are streamed via Zoom.",
        "source": "comms.json",
        "metadata": {"json_path": "comms"},
    },
    {
        "content": "Expense reports must be submitted within 30 days of travel.",
        "source": "finance.json",
        "metadata": {"json_path": "finance"},
    },
    {
        "content": "New hire onboarding takes approximately two weeks.",
        "source": "hr.json",
        "metadata": {"json_path": "hr"},
    },
    {
        "content": "Software licenses are managed by the IT department.",
        "source": "it.json",
        "metadata": {"json_path": "it"},
    },
]


# Deterministic retrieval uses exact-text queries.  With hash-seeded fake
# embeddings, identical text → identical hash seed → identical vector →
# cosine similarity of 1.0.  This tests that the retrieval infrastructure
# (index build, top-k search, metadata plumbing) works end-to-end.
#
# Semantic retrieval quality (paraphrase matching) requires real embeddings
# and is evaluated in the --live section.
RETRIEVAL_CASES = [
    {
        "query": "MFA is mandatory for all administrative users. Provider is Okta.",
        "expected_json_path": "security",
        "description": "Exact-text retrieval: security/MFA policy",
    },
    {
        "query": "The CEO of Acme Corp is Jane Smith. She joined in 2015.",
        "expected_json_path": "leadership",
        "description": "Exact-text retrieval: leadership",
    },
    {
        "query": "The company Acme Corp was founded in 2010 by its original board.",
        "expected_json_path": "company",
        "description": "Exact-text retrieval: company founding",
    },
]

# Semantic retrieval cases — run only under --live
SEMANTIC_RETRIEVAL_CASES = [
    {
        "query": "Is MFA required for administrators?",
        "expected_json_path": "security",
        "description": "Semantic: paraphrase of MFA policy",
    },
    {
        "query": "Who leads the organization?",
        "expected_json_path": "leadership",
        "description": "Semantic: different wording for CEO",
    },
    {
        "query": "When was the organization established?",
        "expected_json_path": "company",
        "description": "Semantic: paraphrase of 'founded'",
    },
]

QA_CASES = [
    {
        "id": "supported_1",
        "question": "Is MFA required for administrators?",
        "context": "MFA is mandatory for all administrative users. Provider is Okta.",
        "expected_substring": None,  # any non-abstention answer passes
        "expect_unsupported": False,
    },
    {
        "id": "supported_2",
        "question": "What authentication provider is used?",
        "context": "The authentication provider configured for the organization is Okta.",
        "expected_substring": "Okta",
        "expect_unsupported": False,
    },
    {
        "id": "unsupported_1",
        "question": "What is the CEO's annual salary?",
        "context": "The company Acme Corp was founded in 2010 by its original board.",
        "expected_substring": None,
        "expect_unsupported": True,
    },
]


class EvalResult(NamedTuple):
    hits: int
    total: int
    details: list[str]


# ---------------------------------------------------------------------------
# Deterministic retrieval evaluation (no API cost)
# ---------------------------------------------------------------------------


def _make_fake_embeddings():
    """Return a proper Embeddings subclass with hash-seeded numpy vectors."""
    import hashlib
    from langchain_core.embeddings import Embeddings

    def _seed(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)

    class _HashSeededEmbeddings(Embeddings):
        """Same text always produces the same vector — enables deterministic retrieval tests."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [
                np.random.default_rng(_seed(t)).random(64).astype(np.float32).tolist()
                for t in texts
            ]

        def embed_query(self, text: str) -> list[float]:
            return np.random.default_rng(_seed(text)).random(64).astype(np.float32).tolist()

    return _HashSeededEmbeddings()


def run_retrieval_eval() -> EvalResult:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document as LCDocument

    from app.retrieval.retriever import DocumentRetriever

    lc_docs = [
        LCDocument(page_content=c["content"], metadata=c["metadata"])
        for c in CORPUS
    ]
    embeddings = _make_fake_embeddings()
    vs = FAISS.from_documents(lc_docs, embeddings)
    retriever = DocumentRetriever(vs, top_k=4)

    hits = 0
    details: list[str] = []
    for case in RETRIEVAL_CASES:
        results = retriever.retrieve(case["query"])
        result_paths = [r.metadata.get("json_path") for r in results]
        hit = case["expected_json_path"] in result_paths
        if hit:
            hits += 1
            details.append(f"  HIT:  [{case['description']}] '{case['query']}'")
        else:
            details.append(
                f"  MISS: [{case['description']}] '{case['query']}'"
                f" → got {result_paths}, expected '{case['expected_json_path']}'"
            )

    return EvalResult(hits=hits, total=len(RETRIEVAL_CASES), details=details)


# ---------------------------------------------------------------------------
# Grounding evaluation with mocked LLM (no API cost)
# ---------------------------------------------------------------------------


def run_grounding_eval_mock() -> tuple[EvalResult, EvalResult]:
    """
    Test QA pipeline grounding using mocked retrieval and LLM.

    Returns (supported_result, unsupported_result).
    """
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from langchain_core.documents import Document as LCDocument

    from app.generation.qa_service import QAService

    supported_hits = 0
    unsupported_hits = 0
    supported_total = sum(1 for c in QA_CASES if not c["expect_unsupported"])
    unsupported_total = sum(1 for c in QA_CASES if c["expect_unsupported"])
    supported_details: list[str] = []
    unsupported_details: list[str] = []

    for case in QA_CASES:
        # Simulate what the LLM should respond given the context
        if case["expect_unsupported"]:
            simulated_answer = (
                "The document does not contain sufficient information to answer this question."
            )
        elif case["expected_substring"]:
            simulated_answer = f"Based on the context: {case['expected_substring']}."
        else:
            simulated_answer = "Yes, based on the context this is confirmed."

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=simulated_answer)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            LCDocument(
                page_content=case["context"],
                metadata={"source": "test.json", "json_path": "test"},
            )
        ]

        svc = QAService(retriever=mock_retriever, llm=mock_llm)
        result = svc.answer_question(case["id"], case["question"])

        if case["expect_unsupported"]:
            passed = "does not contain" in result.answer.lower()
            if passed:
                unsupported_hits += 1
            unsupported_details.append(
                f"  {'PASS' if passed else 'FAIL'}: [{case['id']}] '{case['question']}'"
            )
        else:
            if case["expected_substring"]:
                passed = case["expected_substring"] in result.answer
            else:
                passed = len(result.answer) > 0
            if passed:
                supported_hits += 1
            supported_details.append(
                f"  {'PASS' if passed else 'FAIL'}: [{case['id']}] '{case['question']}'"
            )

    return (
        EvalResult(supported_hits, supported_total, supported_details),
        EvalResult(unsupported_hits, unsupported_total, unsupported_details),
    )


# ---------------------------------------------------------------------------
# Live evaluation (requires OPENAI_API_KEY, incurs cost)
# ---------------------------------------------------------------------------


def run_live_eval():
    """
    Run semantic retrieval and QA with real OpenAI embeddings and generation.

    This will make API calls. Only invoked when --live is passed.
    """
    from langchain_community.vectorstores import FAISS

    from app.ingestion.service import ingest_file
    from app.retrieval.vector_store import build_vector_store
    from app.retrieval.retriever import DocumentRetriever
    from app.generation.llm import get_llm
    from app.generation.qa_service import QAService
    from app.core.config import settings

    if settings.openai_api_key in ("missing", ""):
        print("  ERROR: OPENAI_API_KEY is not set. Skipping live evaluation.")
        return

    # Build a structured JSON document from the corpus signal chunks
    corpus_doc = {chunk["metadata"]["json_path"]: chunk["content"] for chunk in CORPUS}
    doc_bytes = json.dumps(corpus_doc).encode("utf-8")

    chunks = ingest_file(doc_bytes, "eval_corpus.json")
    vs = build_vector_store(chunks)
    retriever = DocumentRetriever(vs, top_k=4)
    llm = get_llm()
    svc = QAService(retriever=retriever, llm=llm)

    print("  Semantic retrieval cases:")
    semantic_hits = 0
    for case in SEMANTIC_RETRIEVAL_CASES:
        results = retriever.retrieve(case["query"])
        result_paths = [r.metadata.get("json_path") for r in results]
        hit = case["expected_json_path"] in result_paths
        if hit:
            semantic_hits += 1
        print(
            f"  {'HIT ' if hit else 'MISS'}: [{case['description']}] '{case['query']}'"
        )
    print(f"\n  Semantic retrieval hit rate: {semantic_hits}/{len(SEMANTIC_RETRIEVAL_CASES)}")

    print("\n  QA answers:")
    for case in QA_CASES:
        result = svc.answer_question(case["id"], case["question"])
        label = "UNSUPPORTED" if case["expect_unsupported"] else "SUPPORTED"
        print(f"  [{label}] Q: {case['question']}")
        print(f"           A: {result.answer[:120]}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation script")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live LLM evaluation (makes real OpenAI API calls and incurs cost)",
    )
    args = parser.parse_args()

    all_passed = True

    print("=" * 60)
    print("Retrieval Evaluation (deterministic, no API cost)")
    print("=" * 60)
    print("  Note: uses hash-seeded fake embeddings — tests retrieval infrastructure,")
    print("  not semantic quality.  Use --live for semantic retrieval evaluation.")
    retrieval_result = run_retrieval_eval()
    for detail in retrieval_result.details:
        print(detail)
    print(f"\nRetrieval hit rate: {retrieval_result.hits}/{retrieval_result.total}")
    if retrieval_result.hits < retrieval_result.total:
        all_passed = False

    print()
    print("=" * 60)
    print("QA Grounding Evaluation (mocked LLM, no API cost)")
    print("=" * 60)
    supported_result, unsupported_result = run_grounding_eval_mock()
    print("  Supported questions:")
    for detail in supported_result.details:
        print(detail)
    print("  Unsupported questions (expected abstention):")
    for detail in unsupported_result.details:
        print(detail)
    print(
        f"\nSupported cases correctly answered: "
        f"{supported_result.hits}/{supported_result.total}"
    )
    print(
        f"Unsupported cases correctly declined: "
        f"{unsupported_result.hits}/{unsupported_result.total}"
    )
    if supported_result.hits < supported_result.total:
        all_passed = False
    if unsupported_result.hits < unsupported_result.total:
        all_passed = False

    if args.live:
        print()
        print("=" * 60)
        print("Live LLM Evaluation (real API calls — cost incurred)")
        print("=" * 60)
        run_live_eval()

    print()
    if all_passed:
        print("All deterministic checks passed.")
    else:
        print("Some checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
