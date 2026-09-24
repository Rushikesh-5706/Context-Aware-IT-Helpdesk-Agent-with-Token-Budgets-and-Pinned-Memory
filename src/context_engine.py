"""
context_engine.py — AgentContext model, token counting, eviction, and prompt assembly.

The eviction loop pops the oldest message one at a time (pop(0)) rather than by pairs.
This keeps the budget accounting exact even when user and assistant turn lengths differ
significantly — a pair-based approach could undercount or overcount by one turn's tokens.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import tiktoken
from pydantic import BaseModel, Field

# ── tokenizer ──────────────────────────────────────────────────────────────────
# cl100k_base is mandated by the spec regardless of which LLM generates replies.
# It's the encoding used by gpt-4/gpt-3.5, and tiktoken ships it offline — no
# network call required, so this works even when Groq is the generation backend.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for a plain string."""
    return len(_ENCODING.encode(text))


# ── data model ─────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str          # "user" | "assistant"
    content: str
    tokens: int = 0    # kept in-memory for eviction math; excluded from the minimal disk schema if desired


class PinnedMemory(BaseModel):
    ticket_id: Optional[str] = None


class AgentContext(BaseModel):
    session_id: str
    active_topic: str = "general"
    pinned: PinnedMemory = Field(default_factory=PinnedMemory)
    recent: list[Message] = Field(default_factory=list)
    total_recent_tokens: int = 0


# ── eviction ───────────────────────────────────────────────────────────────────

def apply_token_budget(recent: list[Message], max_tokens: int) -> list[Message]:
    """
    Evict oldest messages one at a time until total tokens fit within max_tokens.

    Operates on a copy so the caller decides when to commit the result back.
    Pinned memory is never passed into this function — it lives separately in
    AgentContext.pinned and never counts toward the rolling budget.
    """
    messages = list(recent)  # shallow copy; Message objects are Pydantic and immutable enough
    while messages and sum(m.tokens for m in messages) > max_tokens:
        messages.pop(0)   # oldest first — matches reference skeleton
    return messages


# ── extraction helpers ─────────────────────────────────────────────────────────

# Ticket-ID regex runs unconditionally on every incoming message.
# The LLM extraction is for topic classification, not for ticket IDs.
_TICKET_RE = re.compile(r"IT-\d{4}", re.IGNORECASE)


def extract_ticket_id(text: str) -> Optional[str]:
    """Return the first IT-NNNN match found, uppercased, or None."""
    match = _TICKET_RE.search(text)
    if match:
        return match.group(0).upper()
    return None


# ── prompt assembly ────────────────────────────────────────────────────────────

def build_system_prompt(ctx: AgentContext) -> str:
    """
    Assemble the system prompt from three parts:
    1. Persona line.
    2. Current active topic.
    3. Pinned memory block — omitted entirely when ticket_id is None so the
       model never sees a "None" or empty line.
    """
    parts = [
        "You are a helpful IT helpdesk assistant. Answer clearly and concisely.",
        f"The user is currently discussing: {ctx.active_topic}",
    ]
    if ctx.pinned.ticket_id:
        parts.append(f"Known information about this user: Ticket ID: {ctx.pinned.ticket_id}")
    return "\n".join(parts)


# ── persistence ────────────────────────────────────────────────────────────────

LOGS_DIR = Path("logs")


def ensure_logs_dir() -> None:
    """Create ./logs at startup. The Docker volume mount may not create it."""
    LOGS_DIR.mkdir(exist_ok=True)


def persist_context(ctx: AgentContext) -> None:
    """Write context to ./logs/context_{session_id}.json after each successful turn."""
    ensure_logs_dir()
    path = LOGS_DIR / f"context_{ctx.session_id}.json"

    # Minimal disk schema: role + content per message (tokens omitted from file)
    disk_messages = [{"role": m.role, "content": m.content} for m in ctx.recent]

    payload = {
        "session_id": ctx.session_id,
        "active_topic": ctx.active_topic,
        "pinned": {"ticket_id": ctx.pinned.ticket_id},
        "recent": disk_messages,
        "total_recent_tokens": ctx.total_recent_tokens,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_context_from_disk(session_id: str) -> Optional[dict]:
    """Read a persisted context file. Returns raw dict or None if not found."""
    path = LOGS_DIR / f"context_{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
