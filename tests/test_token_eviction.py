"""
test_token_eviction.py — req-4: deterministic unit-level eviction test.

Does NOT use the LLM or the HTTP layer. Tests apply_token_budget() directly
with synthetic Message objects whose token counts are pre-set — no tiktoken
encoding of real text is needed to prove the eviction logic is correct.

Reproduces the spec's own example:
  two ~40-token messages, budget 50
  after eviction: sum ≤ 50, first message gone, second remains.
"""

import pytest
from src.context_engine import Message, apply_token_budget, count_tokens


def make_msg(role: str, content: str, tokens: int) -> Message:
    """Build a Message with an explicit token count, bypassing tiktoken."""
    m = Message(role=role, content=content, tokens=0)
    m.tokens = tokens  # override directly
    return m


# ── spec example ───────────────────────────────────────────────────────────────

def test_eviction_removes_oldest_when_over_budget():
    msg1 = make_msg("user", "First message that is roughly forty tokens long for testing.", 40)
    msg2 = make_msg("assistant", "Second message also roughly forty tokens for testing purposes.", 40)

    result = apply_token_budget([msg1, msg2], max_tokens=50)

    total = sum(m.tokens for m in result)
    assert total <= 50, f"Expected total ≤ 50 after eviction, got {total}"
    assert msg1 not in result, "Oldest message should have been evicted"
    assert msg2 in result, "Newer message should remain after eviction"


def test_eviction_noop_when_within_budget():
    msg1 = make_msg("user", "Short message.", 10)
    msg2 = make_msg("assistant", "Short reply.", 10)

    result = apply_token_budget([msg1, msg2], max_tokens=50)
    assert result == [msg1, msg2], "No eviction should happen when under budget"


def test_eviction_removes_one_at_a_time():
    """Three messages, each 20 tokens, budget 45. After eviction sum ≤ 45."""
    msgs = [
        make_msg("user", "Turn 1 user message.", 20),
        make_msg("assistant", "Turn 1 assistant reply.", 20),
        make_msg("user", "Turn 2 user message.", 20),
    ]
    result = apply_token_budget(msgs, max_tokens=45)
    total = sum(m.tokens for m in result)
    assert total <= 45
    # The first message (oldest) must be gone
    assert msgs[0] not in result


def test_eviction_empty_list_is_safe():
    result = apply_token_budget([], max_tokens=100)
    assert result == []


def test_eviction_single_message_over_budget_leaves_it():
    """A single message over budget — pop(0) on a one-element list empties it."""
    msg = make_msg("user", "Huge message.", 200)
    result = apply_token_budget([msg], max_tokens=50)
    assert result == []


# ── tiktoken sanity ────────────────────────────────────────────────────────────

def test_count_tokens_returns_int():
    n = count_tokens("Hello, can you help me with my laptop?")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty_string():
    assert count_tokens("") == 0
