"""
test_error_handling.py — req-10: LLM failure → HTTP 500 + no context corruption.

Strategy: patch the GROQ_API_KEY to a deliberately invalid value so the Groq
API rejects the call, then verify:
  1. The endpoint returns 500.
  2. The context file on disk is identical to what it was before the failed turn
     (or does not exist if no prior successful turn has run).
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, _sessions
from src import llm_client

client = TestClient(app)
LOGS_DIR = Path("logs")


def _clear_session(session_id: str):
    _sessions.pop(session_id, None)
    log_file = LOGS_DIR / f"context_{session_id}.json"
    if log_file.exists():
        log_file.unlink()


def _read_context_file(sid: str):
    log_file = LOGS_DIR / f"context_{sid}.json"
    if not log_file.exists():
        return None
    return json.loads(log_file.read_text())


def test_llm_failure_returns_500(monkeypatch):
    sid = "req10_error_test"
    _clear_session(sid)

    # First turn succeeds so there's a baseline on disk
    resp = client.post("/v1/chat", json={"session_id": sid, "message": "Hello."})
    assert resp.status_code == 200
    baseline = _read_context_file(sid)

    # Break the API key and force a client reset
    monkeypatch.setenv("GROQ_API_KEY", "invalid-key-that-will-be-rejected")
    llm_client._reset_client()

    resp_fail = client.post("/v1/chat", json={"session_id": sid, "message": "This call will fail."})
    assert resp_fail.status_code == 500, f"Expected 500, got {resp_fail.status_code}"

    # Restore real key and client for subsequent tests
    monkeypatch.undo()
    llm_client._reset_client()


def test_context_file_unchanged_after_llm_failure(monkeypatch):
    sid = "req10_no_corruption"
    _clear_session(sid)

    # Establish a good baseline
    resp = client.post("/v1/chat", json={"session_id": sid, "message": "My printer is jammed."})
    assert resp.status_code == 200
    before = _read_context_file(sid)
    assert before is not None

    # Now break the key
    monkeypatch.setenv("GROQ_API_KEY", "invalid-key-that-will-be-rejected")
    llm_client._reset_client()

    client.post("/v1/chat", json={"session_id": sid, "message": "Another message that will fail."})

    after = _read_context_file(sid)

    # Restore
    monkeypatch.undo()
    llm_client._reset_client()

    # File on disk must be unchanged — same number of messages, same content
    assert after is not None
    assert after["recent"] == before["recent"], (
        "Context file was modified despite LLM failure — req-10 violated"
    )
    assert after["total_recent_tokens"] == before["total_recent_tokens"]


def test_no_context_file_corruption_on_first_call_failure(monkeypatch):
    """If the very first call to a session fails, no file should be created."""
    sid = "req10_first_call_fail"
    _clear_session(sid)

    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
    llm_client._reset_client()

    resp = client.post("/v1/chat", json={"session_id": sid, "message": "Hello."})
    assert resp.status_code == 500

    monkeypatch.undo()
    llm_client._reset_client()

    log_file = LOGS_DIR / f"context_{sid}.json"
    assert not log_file.exists(), "No context file should be written on a failed first turn"
