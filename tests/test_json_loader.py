"""Tests for the JSON ingestion loader."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from app.ingestion.json_loader import JSONLoader


@pytest.fixture
def loader() -> JSONLoader:
    return JSONLoader()


# ---------------------------------------------------------------------------
# Flattening correctness
# ---------------------------------------------------------------------------


def test_flatten_nested_dict_preserves_hierarchy(loader: JSONLoader):
    data = {"security": {"authentication": {"mfa_required": True, "provider": "Okta"}}}
    lines = loader._flatten(data)
    assert "security.authentication.mfa_required: true" in lines
    assert "security.authentication.provider: Okta" in lines


def test_flatten_array_uses_bracket_notation(loader: JSONLoader):
    data = {"tags": ["python", "ml"]}
    lines = loader._flatten(data)
    assert "tags[0]: python" in lines
    assert "tags[1]: ml" in lines


def test_flatten_null_value(loader: JSONLoader):
    data = {"key": None}
    lines = loader._flatten(data)
    assert "key: null" in lines


def test_flatten_bool_is_lowercased(loader: JSONLoader):
    data = {"active": True, "disabled": False}
    lines = loader._flatten(data)
    assert "active: true" in lines
    assert "disabled: false" in lines
    # Must NOT contain Python's Title-Case booleans
    assert "active: True" not in lines
    assert "disabled: False" not in lines


def test_flatten_integer_and_float(loader: JSONLoader):
    data = {"count": 42, "ratio": 3.14}
    lines = loader._flatten(data)
    assert "count: 42" in lines
    assert "ratio: 3.14" in lines


def test_flatten_nested_array_of_objects(loader: JSONLoader):
    data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
    lines = loader._flatten(data)
    assert "users[0].name: Alice" in lines
    assert "users[1].name: Bob" in lines


# ---------------------------------------------------------------------------
# Grouping by top-level key
# ---------------------------------------------------------------------------


def test_groups_into_separate_documents_per_top_level_key(loader: JSONLoader):
    data = {"company": {"name": "Acme"}, "security": {"mfa": True}}
    raw = json.dumps(data).encode("utf-8")
    docs = loader.load(raw, "test.json")

    keys = {d.metadata["json_path"] for d in docs}
    assert "company" in keys
    assert "security" in keys
    assert len(docs) == 2


def test_metadata_json_path_is_set(loader: JSONLoader):
    data = {"employees": [{"name": "Alice"}]}
    raw = json.dumps(data).encode("utf-8")
    docs = loader.load(raw, "data.json")
    assert docs[0].metadata["json_path"] == "employees"


def test_source_is_filename(loader: JSONLoader):
    raw = json.dumps({"x": 1}).encode("utf-8")
    docs = loader.load(raw, "myfile.json")
    assert all(d.source == "myfile.json" for d in docs)


def test_block_text_contains_all_child_lines(loader: JSONLoader):
    data = {"company": {"name": "Acme", "founded": 2010}}
    raw = json.dumps(data).encode("utf-8")
    docs = loader.load(raw, "test.json")
    company_doc = next(d for d in docs if d.metadata["json_path"] == "company")
    assert "company.name: Acme" in company_doc.content
    assert "company.founded: 2010" in company_doc.content


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_json_raises_400(loader: JSONLoader):
    with pytest.raises(HTTPException) as exc_info:
        loader.load(b"not valid json", "bad.json")
    assert exc_info.value.status_code == 400
    assert "Invalid JSON" in exc_info.value.detail


def test_empty_object_raises_400(loader: JSONLoader):
    with pytest.raises(HTTPException) as exc_info:
        loader.load(b"{}", "empty.json")
    assert exc_info.value.status_code == 400


def test_empty_array_raises_400(loader: JSONLoader):
    with pytest.raises(HTTPException) as exc_info:
        loader.load(b"[]", "empty.json")
    assert exc_info.value.status_code == 400
