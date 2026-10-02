"""Integration tests for the /v1/qa API endpoint."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_qa(client: TestClient, doc_bytes: bytes, doc_name: str, questions: list) -> object:
    questions_bytes = json.dumps(questions).encode("utf-8")
    return client.post(
        "/v1/qa",
        files={
            "document_file": (doc_name, doc_bytes, "application/octet-stream"),
            "questions_file": ("questions.json", questions_bytes, "application/json"),
        },
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_json_document_returns_200(client, sample_json_bytes):
    questions = [{"id": "q1", "question": "What is the company name?"}]
    resp = _post_qa(client, sample_json_bytes, "data.json", questions)
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 1
    assert body["results"][0]["id"] == "q1"


def test_valid_pdf_document_returns_200(client, sample_pdf_bytes):
    questions = [{"id": "q1", "question": "Who founded the company?"}]
    resp = _post_qa(client, sample_pdf_bytes, "report.pdf", questions)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1


def test_format_b_string_questions_returns_200(client, sample_json_bytes):
    questions = ["What is the company name?", "Is MFA required?"]
    resp = _post_qa(client, sample_json_bytes, "data.json", questions)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    # IDs auto-assigned
    assert results[0]["id"] == "q1"
    assert results[1]["id"] == "q2"


def test_multiple_questions_all_answered(client, sample_json_bytes):
    questions = [
        {"id": "a", "question": "What is the company?"},
        {"id": "b", "question": "Who provides MFA?"},
    ]
    resp = _post_qa(client, sample_json_bytes, "data.json", questions)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert {r["id"] for r in results} == {"a", "b"}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Document validation errors
# ---------------------------------------------------------------------------


def test_unsupported_file_extension_returns_400(client):
    txt_bytes = b"Some plain text content"
    questions = [{"id": "q1", "question": "What is this?"}]
    questions_bytes = json.dumps(questions).encode("utf-8")
    resp = client.post(
        "/v1/qa",
        files={
            "document_file": ("document.txt", txt_bytes, "text/plain"),
            "questions_file": ("questions.json", questions_bytes, "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_malformed_pdf_returns_400(client):
    questions = [{"id": "q1", "question": "What?"}]
    resp = _post_qa(client, b"not a pdf at all", "bad.pdf", questions)
    assert resp.status_code == 400


def test_invalid_json_document_returns_400(client):
    questions = [{"id": "q1", "question": "What?"}]
    resp = _post_qa(client, b"{ invalid json", "data.json", questions)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Questions validation errors
# ---------------------------------------------------------------------------


def test_malformed_questions_json_returns_400(client, sample_json_bytes):
    malformed = b"not valid json at all"
    resp = client.post(
        "/v1/qa",
        files={
            "document_file": ("data.json", sample_json_bytes, "application/json"),
            "questions_file": ("questions.json", malformed, "application/json"),
        },
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"]


def test_empty_questions_array_returns_400(client, sample_json_bytes):
    resp = _post_qa(client, sample_json_bytes, "data.json", [])
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_questions_not_array_returns_400(client, sample_json_bytes):
    questions_bytes = b'{"question": "something"}'
    resp = client.post(
        "/v1/qa",
        files={
            "document_file": ("data.json", sample_json_bytes, "application/json"),
            "questions_file": ("questions.json", questions_bytes, "application/json"),
        },
    )
    assert resp.status_code == 400


def test_invalid_question_item_type_returns_400(client, sample_json_bytes):
    # Numbers in the array are invalid
    resp = _post_qa(client, sample_json_bytes, "data.json", [123, 456])
    assert resp.status_code == 400


def test_question_missing_question_field_returns_400(client, sample_json_bytes):
    questions = [{"id": "q1", "text": "Where is the policy?"}]
    resp = _post_qa(client, sample_json_bytes, "data.json", questions)
    assert resp.status_code == 400
    assert "missing" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def test_response_contains_expected_fields(client, sample_json_bytes):
    questions = [{"id": "q1", "question": "What is the MFA provider?"}]
    resp = _post_qa(client, sample_json_bytes, "data.json", questions)
    result = resp.json()["results"][0]
    assert "id" in result
    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
