"""
test_chat_endpoint.py — req-1: POST /v1/chat returns 200 with correct schema.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_chat_returns_200_and_schema():
    resp = client.post("/v1/chat", json={"session_id": "req1_test", "message": "Hello, I need IT help."})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "response" in data, "Missing 'response' field"
    assert "active_topic" in data, "Missing 'active_topic' field"
    assert isinstance(data["response"], str) and data["response"], "response must be non-empty string"
    assert isinstance(data["active_topic"], str) and data["active_topic"], "active_topic must be non-empty string"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_missing_session_id_returns_422():
    resp = client.post("/v1/chat", json={"message": "Hello"})
    assert resp.status_code == 422


def test_missing_message_returns_422():
    resp = client.post("/v1/chat", json={"session_id": "req1_no_msg"})
    assert resp.status_code == 422
