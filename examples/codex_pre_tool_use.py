"""Experimental Codex PreToolUse hook for shell calls.

Reads Codex's hook JSON from stdin and runs ClawKeeper's existing pre-tool
guards. Safe calls return no permission override; blocked calls return the
documented deny response. This example does not install or activate itself.
"""

# Imports inside evaluate are intentional: dependency failures must deny.
# ruff: noqa: PLC0415

from __future__ import annotations

import json
import os
import sys
from typing import Any


def deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"[ClawKeeper] {reason}",
        }
    }


def evaluate(payload: Any) -> dict[str, Any]:
    # Imports remain inside the error boundary so a missing installation also
    # produces a deny response instead of an ordinary non-blocking hook error.
    from clawkeeper_core.adapters.base import make_event, run_pre_tool_guards

    if not isinstance(payload, dict) or payload.get("hook_event_name") != "PreToolUse":
        raise ValueError("expected a PreToolUse event")
    if payload.get("tool_name") != "Bash":
        raise ValueError("this prototype supports the Bash hook only")
    params = payload.get("tool_input")
    if not isinstance(params, dict):
        raise ValueError("expected object tool_input")
    command = params.get("command")
    if not isinstance(command, str):
        raise ValueError("expected string tool_input.command")
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        raise ValueError("expected absolute cwd")
    os.chdir(cwd)
    event = make_event(
        tool_name="Bash",
        command=command,
        params=params,
        session_id=payload.get("session_id"),
    )
    decision, _ = run_pre_tool_guards(event)
    if decision is not None:
        return deny(f"{decision['source']}: {decision.get('reason', 'blocked')}")
    return {}


def main() -> None:
    try:
        result = evaluate(json.load(sys.stdin))
    except Exception as exc:
        # Exception text can contain tool arguments; report only its type.
        result = deny(f"hook failed ({type(exc).__name__})")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
