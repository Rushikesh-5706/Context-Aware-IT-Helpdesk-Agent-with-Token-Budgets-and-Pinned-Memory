"""
api.py — FastAPI application: routes, session store, and per-turn orchestration.

Turn lifecycle:
  1. Load or create AgentContext for the session.
  2. Run regex ticket-ID extraction (always).
  3. Run topic classification (LLM, with fallback).
  4. Append user message to recent with token count.
  5. Apply token budget eviction.
  6. Assemble prompt and call LLM.
  7. Append assistant reply to recent.
  8. Persist context to disk.
  9. Return response + active_topic.

On any LLM error (step 6), we abort before steps 7–8 and return HTTP 500.
The context on disk is therefore guaranteed to match the state before the turn.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.context_engine import (
    AgentContext,
    Message,
    PinnedMemory,
    apply_token_budget,
    build_system_prompt,
    count_tokens,
    ensure_logs_dir,
    extract_ticket_id,
    persist_context,
    load_context_from_disk,
)
from src.llm_client import extract_topic, generate_reply, _reset_client

load_dotenv()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_logs_dir()
    # Repopulate in-memory sessions from any context files that survived a
    # previous run. This means a server restart does not silently lose state
    # for sessions that already have a context file on disk.
    _restore_sessions_from_disk()
    yield


app = FastAPI(title="IT Helpdesk Agent", lifespan=lifespan)

# In-memory session store. Each entry is a live AgentContext.
_sessions: dict[str, AgentContext] = {}


def _restore_sessions_from_disk() -> None:
    """
    Scan logs/context_*.json and rebuild AgentContext objects so sessions
    survive a server restart. Messages are rebuilt without token counts (we
    re-count them with tiktoken) so the eviction logic stays correct.
    """
    from src.context_engine import LOGS_DIR, count_tokens
    for log_file in sorted(LOGS_DIR.glob("context_*.json")):
        try:
            raw = load_context_from_disk(log_file.stem.removeprefix("context_"))
            if raw is None:
                continue
            messages = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    tokens=count_tokens(m["content"]),
                )
                for m in raw.get("recent", [])
            ]
            ctx = AgentContext(
                session_id=raw["session_id"],
                active_topic=raw.get("active_topic", "general"),
                pinned=PinnedMemory(ticket_id=raw.get("pinned", {}).get("ticket_id")),
                recent=messages,
                total_recent_tokens=sum(m.tokens for m in messages),
            )
            _sessions[ctx.session_id] = ctx
        except Exception:
            # A corrupt or partially-written file must not crash startup.
            continue


# ── schemas ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    active_topic: str


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: str) -> AgentContext:
    if session_id not in _sessions:
        _sessions[session_id] = AgentContext(
            session_id=session_id,
            pinned=PinnedMemory(),
        )
    return _sessions[session_id]


def _token_budget() -> int:
    """Read TOKEN_BUDGET_LIMIT fresh on every request so tests can override it
    via os.environ without restarting the process."""
    return int(os.environ.get("TOKEN_BUDGET_LIMIT", "4000"))


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    ctx = _get_or_create_session(req.session_id)

    # ── 1. Ticket-ID regex — always runs, regardless of LLM extraction ─────────
    ticket_id = extract_ticket_id(req.message)
    if ticket_id:
        ctx.pinned.ticket_id = ticket_id

    # ── 2. Topic classification ─────────────────────────────────────────────────
    # _reset_client() is called here so that tests which patch GROQ_API_KEY in
    # os.environ before calling this endpoint get a fresh client pointed at the
    # right key.
    _reset_client()
    ctx.active_topic = extract_topic(req.message, ctx.active_topic)

    # ── 3. Append user message to recent ───────────────────────────────────────
    user_tokens = count_tokens(req.message)
    user_msg = Message(role="user", content=req.message, tokens=user_tokens)
    ctx.recent.append(user_msg)

    # ── 4. Eviction ────────────────────────────────────────────────────────────
    budget = _token_budget()
    ctx.recent = apply_token_budget(ctx.recent, budget)
    ctx.total_recent_tokens = sum(m.tokens for m in ctx.recent)

    # ── 5. Assemble prompt ─────────────────────────────────────────────────────
    system_content = build_system_prompt(ctx)
    messages_for_llm = [{"role": "system", "content": system_content}]
    messages_for_llm += [{"role": m.role, "content": m.content} for m in ctx.recent]

    # ── 6. Generate reply — any exception here triggers req-10 path ───────────
    try:
        reply = generate_reply(messages_for_llm)
    except Exception as exc:
        # Roll back: user message was appended in step 3, remove it.
        # recent may have been evicted already so we pop the last appended element.
        if ctx.recent and ctx.recent[-1].role == "user" and ctx.recent[-1].content == req.message:
            ctx.recent.pop()
            ctx.total_recent_tokens = sum(m.tokens for m in ctx.recent)
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}") from exc

    # ── 7. Append assistant reply ──────────────────────────────────────────────
    assistant_tokens = count_tokens(reply)
    assistant_msg = Message(role="assistant", content=reply, tokens=assistant_tokens)
    ctx.recent.append(assistant_msg)

    # Re-run eviction to account for the assistant message potentially pushing
    # recent over budget again.
    ctx.recent = apply_token_budget(ctx.recent, budget)
    ctx.total_recent_tokens = sum(m.tokens for m in ctx.recent)

    # ── 8. Persist ─────────────────────────────────────────────────────────────
    persist_context(ctx)

    return ChatResponse(response=reply, active_topic=ctx.active_topic)


@app.get("/v1/context/{session_id}")
async def get_context(session_id: str):
    ctx = _sessions.get(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    disk_messages = [{"role": m.role, "content": m.content} for m in ctx.recent]
    return {
        "session_id": ctx.session_id,
        "active_topic": ctx.active_topic,
        "pinned": {"ticket_id": ctx.pinned.ticket_id},
        "recent": disk_messages,
        "total_recent_tokens": ctx.total_recent_tokens,
    }
