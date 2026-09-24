"""
test_context_persistence.py — req-2: context written to disk after each turn;
                               req-9: GET /v1/context/{session_id} returns 404/200.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, _sessions

client = TestClient(app)
LOGS_DIR = Path("logs")


def _clear_session(session_id: str):
    _sessions.pop(session_id, None)
    log_file = LOGS_DIR / f"context_{session_id}.json"
    if log_file.exists():
        log_file.unlink()


def test_context_written_to_disk_after_turn():
    sid = "req2_persist_test"
    _clear_session(sid)

    resp = client.post("/v1/chat", json={"session_id": sid, "message": "I need help with my laptop."})
    assert resp.status_code == 200

    log_file = LOGS_DIR / f"context_{sid}.json"
    assert log_file.exists(), f"Expected {log_file} to exist after a successful turn"

    data = json.loads(log_file.read_text())
    assert data["session_id"] == sid
    assert "active_topic" in data
    assert "pinned" in data and "ticket_id" in data["pinned"]
    assert isinstance(data["recent"], list)
    assert "total_recent_tokens" in data
    # Check recent messages have role and content
    for msg in data["recent"]:
        assert "role" in msg
        assert "content" in msg


def test_context_updates_across_turns():
    sid = "req2_multi_turn"
    _clear_session(sid)

    client.post("/v1/chat", json={"session_id": sid, "message": "Hello"})
    client.post("/v1/chat", json={"session_id": sid, "message": "My WiFi is broken."})

    log_file = LOGS_DIR / f"context_{sid}.json"
    data = json.loads(log_file.read_text())
    # After two turns there should be at least 2 messages (user + assistant each)
    assert len(data["recent"]) >= 2


# ── req-9 ──────────────────────────────────────────────────────────────────────

def test_get_context_404_for_unknown_session():
    resp = client.get("/v1/context/nonexistent_session_xyz_999")
    assert resp.status_code == 404


def test_get_context_200_for_known_session():
    sid = "req9_known_session"
    _clear_session(sid)

    client.post("/v1/chat", json={"session_id": sid, "message": "I need printer help."})

    resp = client.get(f"/v1/context/{sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert "active_topic" in data
    assert "pinned" in data
    assert "recent" in data
    assert "total_recent_tokens" in data
