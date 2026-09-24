"""
llm_client.py — Groq-backed LLM calls using the openai SDK's OpenAI-compatible interface.

Two public functions:
  generate_reply(messages)       — production generation call
  extract_topic(message, current_topic) — structured JSON extraction for topic classification

Both share one lazy-initialized client so we're not re-reading env vars on every call —
but the client is recreated if GROQ_API_KEY changes (test isolation via os.environ patches
is done at the process level, so re-instantiating per-call is unnecessary).
"""

from __future__ import annotations

import json
import os
import re
import time

from openai import OpenAI, RateLimitError

# ── client ─────────────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY", "")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


# Module-level client; tests that swap GROQ_API_KEY call _reset_client() below.
_client: OpenAI = _make_client()


def _reset_client() -> None:
    """Force client reconstruction — called by tests that patch GROQ_API_KEY."""
    global _client
    _client = _make_client()


# ── retry helper ───────────────────────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    """
    Retry once on 429 (rate limit) after a short backoff.
    Any other error propagates immediately. One retry is enough for Groq's
    free-tier limits in interactive use; a full retry queue is out of scope.
    """
    try:
        return fn(*args, **kwargs)
    except RateLimitError:
        time.sleep(1.5)
        return fn(*args, **kwargs)


# ── generation ─────────────────────────────────────────────────────────────────

MODEL = "openai/gpt-oss-120b"

# Topic labels kept small and stable. Adding labels here requires a matching
# update to the extraction prompt so the model stays consistent.
TOPIC_LABELS = [
    "general",
    "wifi_issue",
    "ticket_inquiry",
    "hardware_issue",
    "software_issue",
    "account_access",
    "network_issue",
    "email_issue",
]


def generate_reply(messages: list[dict]) -> str:
    """
    Call Groq for a chat completion. `messages` is the fully assembled list
    including the system message — caller owns the ordering.

    Raises on API errors; the caller (api.py) catches and returns HTTP 500.
    """
    response = _call_with_retry(
        _client.chat.completions.create,
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── topic extraction ───────────────────────────────────────────────────────────

_TOPIC_SYSTEM = (
    "You are a topic classifier for an IT helpdesk. "
    "Given the user's message and the current topic, decide what the active topic is. "
    "Reply with ONLY a valid JSON object on a single line, nothing else, no markdown. "
    "Format: {\"topic\": \"<label>\"}. "
    f"Valid labels: {', '.join(TOPIC_LABELS)}. "
    "If the message is a greeting or unrelated, use 'general'. "
    "If the current topic is still relevant, keep it."
)


def extract_topic(message: str, current_topic: str) -> str:
    """
    Return the new active_topic label for this turn.

    The model is instructed to reply with a bare JSON object. If it wraps it in
    markdown or adds text, we scan for the first {...} block and try to parse
    that. On any failure, keep the previous topic so the turn does not crash.
    """
    prompt = (
        f"Current topic: {current_topic}\n"
        f"User message: {message}\n"
        "Classify the topic."
    )
    try:
        response = _call_with_retry(
            _client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": _TOPIC_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            # No response_format — openai/gpt-oss-120b on Groq returns 400
            # for json_object mode. We parse JSON from free-form text instead.
            temperature=0.0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content.strip()
        # The model sometimes wraps the JSON in markdown fences or adds prose.
        # Grab the first {...} block and try to parse it.
        match = re.search(r'\{[^}]+\}', raw)
        candidate = match.group(0) if match else raw
        data = json.loads(candidate)
        topic = data.get("topic", current_topic)
        if topic not in TOPIC_LABELS:
            return current_topic
        return topic
    except Exception:
        # Malformed JSON, network error, etc. — keep previous topic.
        return current_topic
