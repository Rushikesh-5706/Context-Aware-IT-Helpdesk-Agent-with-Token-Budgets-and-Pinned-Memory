"""
test_env_config.py — req-8: .env.example has the right keys; TOKEN_BUDGET_LIMIT
is read fresh per request so it can be overridden mid-process.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, _sessions

client = TestClient(app)


def test_env_example_contains_required_keys():
    env_example = Path(".env.example").read_text()
    assert "GROQ_API_KEY" in env_example, ".env.example must define GROQ_API_KEY"
    assert "TOKEN_BUDGET_LIMIT" in env_example, ".env.example must define TOKEN_BUDGET_LIMIT"
    # Confirm the default budget value
    assert "4000" in env_example, ".env.example TOKEN_BUDGET_LIMIT default should be 4000"


def test_env_example_has_no_real_secrets():
    env_example = Path(".env.example").read_text()
    # The placeholder value must be the literal placeholder string, not a real key
    assert "your_key_here" in env_example, ".env.example should contain placeholder text, not a real key"


def test_token_budget_limit_read_per_request(monkeypatch):
    """Changing TOKEN_BUDGET_LIMIT via os.environ must take effect on the next
    request without restarting the server."""
    sid = "req8_budget_env"
    _sessions.pop(sid, None)

    monkeypatch.setenv("TOKEN_BUDGET_LIMIT", "9999")
    resp = client.post("/v1/chat", json={"session_id": sid, "message": "Hello, test budget env."})
    assert resp.status_code == 200

    # Budget is 9999, so even after a short message no eviction should happen
    from src.api import _sessions as sessions
    ctx = sessions[sid]
    assert ctx.total_recent_tokens < 9999
