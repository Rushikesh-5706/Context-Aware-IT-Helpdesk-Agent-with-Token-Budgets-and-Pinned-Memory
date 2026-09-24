#!/usr/bin/env python3
"""
eval_5_turns.py — 5-turn end-to-end evaluation of the IT helpdesk agent.

Run:  python eval_5_turns.py
Exit: 0 on full pass, 1 on first assertion failure.

This script starts its own uvicorn server on port 8001 with TOKEN_BUDGET_LIMIT=100
so eviction is forced by turn 3 without touching the user's running server on 8000.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

LOGS_DIR = Path("logs")
PORT = 8001
BASE_URL = f"http://localhost:{PORT}"
SESSION_ID = f"eval_5turns_{int(time.time())}"
BUDGET = "100"


def start_server():
    env = os.environ.copy()
    env["TOKEN_BUDGET_LIMIT"] = BUDGET
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    for _ in range(20):
        time.sleep(0.5)
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
    proc.terminate()
    print("ERROR: Server failed to start within 10 seconds.")
    sys.exit(1)


def post_chat(message: str) -> dict:
    resp = httpx.post(
        f"{BASE_URL}/v1/chat",
        json={"session_id": SESSION_ID, "message": message},
        timeout=60.0,
    )
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()


def read_context() -> dict:
    log_file = LOGS_DIR / f"context_{SESSION_ID}.json"
    return json.loads(log_file.read_text())


def fail(turn: int, reason: str):
    print(f"  FAIL  Turn {turn}: {reason}")
    sys.exit(1)


def check(condition: bool, turn: int, reason: str):
    if not condition:
        fail(turn, reason)


def main():
    print("=" * 60)
    print("5-Turn Evaluation — IT Helpdesk Agent")
    print(f"Session ID : {SESSION_ID}")
    print(f"Budget     : {BUDGET} tokens")
    print(f"Server     : {BASE_URL}")
    print("=" * 60)

    proc = start_server()
    print(f"Server started (pid {proc.pid})\n")

    try:
        # ── Turn 1 ─────────────────────────────────────────────────────────────
        print("Turn 1: Greeting")
        data1 = post_chat("Hi, I need help.")
        ctx1 = read_context()

        print(f"  active_topic : {data1['active_topic']}")
        print(f"  response     : {data1['response'][:80]}...")

        valid_labels = {"general", "wifi_issue", "ticket_inquiry", "hardware_issue",
                        "software_issue", "account_access", "network_issue", "email_issue"}
        check(data1["active_topic"] in valid_labels, 1,
              f"active_topic {data1['active_topic']!r} is not a recognised label")
        print("  PASS\n")

        # ── Turn 2 ─────────────────────────────────────────────────────────────
        print("Turn 2: Ticket mention — IT-8812")
        data2 = post_chat("My ticket is IT-8812.")
        ctx2 = read_context()

        print(f"  active_topic : {data2['active_topic']}")
        print(f"  pinned       : {ctx2['pinned']}")

        check(ctx2["pinned"]["ticket_id"] == "IT-8812", 2,
              f"Expected pinned.ticket_id='IT-8812', got {ctx2['pinned']['ticket_id']!r}")
        check(data2["active_topic"] in {"ticket_inquiry", "general"}, 2,
              f"Expected ticket_inquiry or general, got {data2['active_topic']!r}")
        print("  PASS\n")

        # ── Turn 3 ─────────────────────────────────────────────────────────────
        print("Turn 3: WiFi problem — should push recent over 100-token budget")
        wifi_msg = (
            "Actually, before we do that, my WiFi in the library just dropped. "
            "I am using a Mac. It says 'connected without internet'. "
            "I have already tried turning the WiFi off and on again and restarting the router. "
            "The issue started after a macOS software update last night."
        )
        data3 = post_chat(wifi_msg)
        ctx3 = read_context()

        print(f"  active_topic         : {data3['active_topic']}")
        print(f"  total_recent_tokens  : {ctx3['total_recent_tokens']}")
        print(f"  pinned               : {ctx3['pinned']}")

        check(data3["active_topic"] in {"wifi_issue", "network_issue"}, 3,
              f"Expected wifi/network topic, got {data3['active_topic']!r}")
        check(ctx3["pinned"]["ticket_id"] == "IT-8812", 3,
              f"Pinned ticket changed — expected 'IT-8812', got {ctx3['pinned']['ticket_id']!r}")
        # Budget is 100 — after this long turn, eviction must have kicked in
        check(ctx3["total_recent_tokens"] <= int(BUDGET), 3,
              f"recent tokens {ctx3['total_recent_tokens']} still exceeds budget {BUDGET}")
        print("  PASS\n")

        # ── Turn 4 ─────────────────────────────────────────────────────────────
        print("Turn 4: Mac troubleshooting request — expect eviction continues")
        data4 = post_chat("Can you give me the Mac troubleshooting steps for 'connected without internet'?")
        ctx4 = read_context()

        print(f"  active_topic         : {data4['active_topic']}")
        print(f"  total_recent_tokens  : {ctx4['total_recent_tokens']}")
        print(f"  recent message count : {len(ctx4['recent'])}")

        check(ctx4["total_recent_tokens"] <= int(BUDGET), 4,
              f"recent tokens {ctx4['total_recent_tokens']} still exceeds budget {BUDGET} after eviction")
        check(ctx4["pinned"]["ticket_id"] == "IT-8812", 4,
              "Pinned ticket must survive eviction")
        print("  PASS\n")

        # ── Turn 5 ─────────────────────────────────────────────────────────────
        print("Turn 5: Back to ticket — assert IT-8812 still pinned AND in response")
        data5 = post_chat(
            "Okay, I fixed the WiFi. Let's go back to my original ticket. "
            "Can you confirm the status of the ticket I mentioned earlier?"
        )
        ctx5 = read_context()

        print(f"  active_topic         : {data5['active_topic']}")
        print(f"  pinned               : {ctx5['pinned']}")
        print(f"  response snippet     : {data5['response'][:120]}")

        check(ctx5["pinned"]["ticket_id"] == "IT-8812", 5,
              f"Final pinned.ticket_id should be 'IT-8812', got {ctx5['pinned']['ticket_id']!r}")
        # The model sometimes formats "IT-8812" with a non-breaking hyphen (U+2011).
        # Normalise the response before checking.
        response_normalised = data5["response"].replace("\u2011", "-")
        check("IT-8812" in response_normalised, 5,
              f"Final response must contain 'IT-8812'. Got: {data5['response'][:200]}")
        print("  PASS\n")

        # ── Summary ────────────────────────────────────────────────────────────
        print("=" * 60)
        print("All 5 turns passed.")
        print(f"Final context file: logs/context_{SESSION_ID}.json")
        print("=" * 60)

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    sys.exit(0)


if __name__ == "__main__":
    main()
