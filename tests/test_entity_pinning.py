"""
test_entity_pinning.py - req-3: ticket ID extracted via regex and pinned in context.

The regex IT-[0-9]{4} runs unconditionally -- the LLM extraction is for topic only.
We test that ticket IDs are pinned regardless of what topic the LLM assigns.
"""

import json
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


def test_ticket_id_pinned_after_mention():
    sid = "req3_ticket_pin"
    _clear_session(sid)

    resp = client.post("/v1/chat", json={"session_id": sid, "message": "Please check my ticket IT-4921."})
    assert resp.status_code == 200

    log_file = LOGS_DIR / f"context_{sid}.json"
    data = json.loads(log_file.read_text())
    assert data["pinned"]["ticket_id"] == "IT-4921", (
        f"Expected 'IT-4921', got {data['pinned']['ticket_id']!r}"
    )


def test_ticket_id_stays_pinned_after_topic_change():
    sid = "req3_pin_persists"
    _clear_session(sid)

    client.post("/v1/chat", json={"session_id": sid, "message": "My ticket is IT-7777."})
    client.post("/v1/chat", json={"session_id": sid, "message": "Now my WiFi is dropping too."})

    log_file = LOGS_DIR / f"context_{sid}.json"
    data = json.loads(log_file.read_text())
    # ticket_id must survive a topic shift to wifi_issue
    assert data["pinned"]["ticket_id"] == "IT-7777"


def test_ticket_id_updated_when_new_one_mentioned():
    sid = "req3_ticket_update"
    _clear_session(sid)

    client.post("/v1/chat", json={"session_id": sid, "message": "Check ticket IT-1001."})
    client.post("/v1/chat", json={"session_id": sid, "message": "Actually, refer to IT-2002 instead."})

    log_file = LOGS_DIR / f"context_{sid}.json"
    data = json.loads(log_file.read_text())
    assert data["pinned"]["ticket_id"] == "IT-2002"


def test_no_ticket_id_when_none_mentioned():
    sid = "req3_no_ticket"
    _clear_session(sid)

    client.post("/v1/chat", json={"session_id": sid, "message": "My keyboard is broken."})

    log_file = LOGS_DIR / f"context_{sid}.json"
    data = json.loads(log_file.read_text())
    assert data["pinned"]["ticket_id"] is None
