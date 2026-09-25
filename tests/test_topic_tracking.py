"""
test_topic_tracking.py — req-5: active_topic updated per turn via LLM extraction.

These tests make real HTTP calls through FastAPI TestClient and read the
resulting context file to confirm the topic reflects the user's message.
The exact label is LLM-determined, so we check membership in the known label
set rather than hardcoding a single expected string.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, _sessions
from src.llm_client import TOPIC_LABELS

client = TestClient(app)
LOGS_DIR = Path("logs")

WIFI_LABELS = {"wifi_issue", "network_issue"}
TICKET_LABELS = {"ticket_inquiry"}
GENERAL_LABELS = {"general"}


def _clear_session(session_id: str):
    _sessions.pop(session_id, None)
    log_file = LOGS_DIR / f"context_{session_id}.json"
    if log_file.exists():
        log_file.unlink()


def _read_topic(sid: str) -> str:
    log_file = LOGS_DIR / f"context_{sid}.json"
    return json.loads(log_file.read_text())["active_topic"]


def test_greeting_produces_valid_topic():
    sid = "req5_greeting"
    _clear_session(sid)
    resp = client.post("/v1/chat", json={"session_id": sid, "message": "Hi, I need some help."})
    assert resp.status_code == 200
    topic = _read_topic(sid)
    assert topic in TOPIC_LABELS, f"Unknown topic label: {topic!r}"


def test_wifi_message_shifts_topic():
    sid = "req5_wifi"
    _clear_session(sid)
    resp = client.post(
        "/v1/chat",
        json={"session_id": sid, "message": "My WiFi keeps dropping, I can't connect to the internet."},
    )
    assert resp.status_code == 200
    topic = _read_topic(sid)
    assert topic in WIFI_LABELS, f"Expected wifi/network topic, got {topic!r}"


def test_ticket_message_shifts_topic():
    sid = "req5_ticket"
    _clear_session(sid)
    resp = client.post(
        "/v1/chat",
        json={"session_id": sid, "message": "I want to check the status of my support ticket IT-3344."},
    )
    assert resp.status_code == 200
    topic = _read_topic(sid)
    # "general" is excluded on purpose — a message that explicitly names a ticket
    # must produce a ticket-related label; accepting general would make this
    # test unable to catch a wrong classification.
    assert topic in TICKET_LABELS, f"Expected a ticket-related topic, got {topic!r}"


def test_topic_in_api_response_matches_file():
    sid = "req5_match"
    _clear_session(sid)
    resp = client.post("/v1/chat", json={"session_id": sid, "message": "My email won't load."})
    assert resp.status_code == 200
    api_topic = resp.json()["active_topic"]
    file_topic = _read_topic(sid)
    assert api_topic == file_topic, "active_topic in API response must match what was written to disk"
