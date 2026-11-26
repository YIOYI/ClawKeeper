"""Agent-agnostic adapter base class.

A *guard* is a callable that takes a `dict` event (the same shape that
`guard_execution`, `guard_before_tool_call`, and the Judge already consume)
and returns a decision dict, or `None` to abstain.

An *adapter* knows how to:
  1. translate a host's native callbacks (Hermes `tool_start_callback`,
     OpenClaw watcher WebSocket message, Claude Code hook payload, MCP
     `tools/call` request, LangGraph middleware event, …) into the
     standard event dict;
  2. dispatch that event through a chain of guards;
  3. translate the resulting decision back into the host's native
     accept/deny vocabulary (Hermes's `"once"`/`"deny"` strings,
     OpenClaw's `continue`/`stop`, MCP's `isError`, …).

The decision dict shape (compatible with existing CK code):

    {
        "block": bool,                  # True if the guard wants to stop the call
        "outcome": "allow"|"deny"|"ask",
        "source": str,                  # which guard / rule
        "severity": "low"|"medium"|"high"|"critical"|"hardline",
        "reason": str,
        "evidence": dict,
    }

`None` from a guard means "no opinion — let the next one decide".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from clawkeeper_core.execution import analysis_scope
from clawkeeper_core.guards.exec_gate import guard_execution
from clawkeeper_core.guards.path_guard import guard_before_tool_call
from clawkeeper_core.guards.script_body_scan import script_body_scan_guard
from clawkeeper_core.guards.url_safety import url_safety_guard
from clawkeeper_core.learned_rules import pin_snapshot

GuardEvent = dict[str, Any]
GuardDecision = dict[str, Any]
GuardCallable = Callable[[GuardEvent], GuardDecision | None]

DEFAULT_PRE_TOOL_GUARDS: tuple[GuardCallable, ...] = (
    guard_execution, guard_before_tool_call, url_safety_guard, script_body_scan_guard,
)


def make_event(
    *,
    tool_name: str,
    command: str = "",
    params: dict | None = None,
    content: str = "",
    description: str = "",
    session_id: str | None = None,
    direction: str = "pre_tool_call",  # "pre_tool_call" | "post_tool_call" | "return_content"
) -> GuardEvent:
    """Build the standard event dict the guards consume.

    `direction` lets guards opt in/out of phases.  Pre-tool-call guards
    (exec_gate, path_guard) inspect upcoming actions; post-tool-call and
    return-content guards (credential_redact, return_content_scan) inspect
    things the host already produced.
    """
    base: GuardEvent = {
        "toolName": tool_name,
        "direction": direction,
        "params": {"command": command, "description": description, **(params or {})},
    }
    if content:
        base["content"] = content
    if session_id is not None:
        base["session_id"] = session_id
    return base


def run_guard_chain(
    guards: list[GuardCallable],
    event: GuardEvent,
    *,
    stop_on_block: bool = True,
) -> tuple[GuardDecision | None, list[GuardDecision]]:
    """Walk a guard chain.

    Returns ``(first_blocking_decision_or_None, all_decisions_emitted)``.
    """
    with pin_snapshot(), analysis_scope():
        return _run_guard_chain(guards, event, stop_on_block=stop_on_block)


def _run_guard_chain(guards, event, *, stop_on_block):
    seen: list[GuardDecision] = []
    blocking: GuardDecision | None = None
    for guard in guards:
        try:
            d = guard(event)
        except Exception as exc:
            d = {
                "block": False,
                "outcome": "allow",
                "source": getattr(guard, "__name__", "anon_guard"),
                "severity": "low",
                "reason": f"guard raised: {exc!r} (failing open by convention; "
                f"a guard with `failure_policy=fail-closed` should set block=True itself)",
                "evidence": {},
            }
        if d is None:
            continue
        seen.append(d)
        if d.get("block") and blocking is None:
            blocking = d
            if stop_on_block:
                break
    return blocking, seen


def _check_pre_tool_guard(guard: GuardCallable, event: GuardEvent) -> GuardDecision | None:
    source = getattr(guard, "__name__", "unknown_guard")
    try:
        finding = guard(event)
    except Exception:
        finding = {"error": True}
    if finding is None:
        return None
    if (not isinstance(finding, dict) or not isinstance(finding.get("block"), bool)
            or (not finding["block"] and finding.get("error"))):
        return {"block": True, "outcome": "ask", "source": source, "severity": "high",
                "reason": "The guard could not complete the check."}
    if finding["block"] and not finding.get("source"):
        return {**finding, "source": source}
    return finding


def run_pre_tool_guards(
    event: GuardEvent, guards: Sequence[GuardCallable] = DEFAULT_PRE_TOOL_GUARDS,
) -> tuple[GuardDecision | None, list[GuardDecision]]:
    """Apply the pre-execution checks, preserving a deny over a request for review."""
    findings = []
    review = None
    with pin_snapshot(), analysis_scope():
        for guard in guards:
            finding = _check_pre_tool_guard(guard, event)
            if finding is None:
                continue
            findings.append(finding)
            if finding["block"]:
                if finding.get("outcome") != "ask":
                    return finding, findings
                if review is None:
                    review = finding
    return review, findings


class BaseAdapter(ABC):
    """Abstract adapter — every per-host integration subclasses this.

    Subclasses MUST implement `install(host)` / `uninstall()`.  They
    typically wire host-specific callbacks to call `_dispatch_pre_tool_call`
    or `_dispatch_return_content` on themselves.
    """

    def __init__(
        self,
        *,
        guards: list[GuardCallable] | None = None,
        post_guards: list[GuardCallable] | None = None,
        return_content_guards: list[GuardCallable] | None = None,
    ):
        self.pre_guards: list[GuardCallable] = list(guards or [])
        self.post_guards: list[GuardCallable] = list(post_guards or [])
        self.return_content_guards: list[GuardCallable] = list(return_content_guards or [])
        self._decision_log: list[GuardDecision] = []

    @abstractmethod
    def install(self, host: Any) -> None:
        """Wire into the host's native callback system."""

    @abstractmethod
    def uninstall(self) -> None:
        """Detach from the host (idempotent)."""

    # ─── Dispatch entrypoints — host adapters call these from their bound callbacks ───

    def _dispatch_pre_tool_call(self, event: GuardEvent) -> GuardDecision | None:
        blocking, decisions = run_pre_tool_guards(event, self.pre_guards)
        self._decision_log.extend(decisions)
        return blocking

    def _dispatch_post_tool_call(self, event: GuardEvent) -> None:
        _blocking, decisions = run_guard_chain(self.post_guards, event, stop_on_block=False)
        self._decision_log.extend(decisions)

    def _dispatch_return_content(self, event: GuardEvent) -> GuardDecision | None:
        blocking, decisions = run_guard_chain(self.return_content_guards, event, stop_on_block=True)
        self._decision_log.extend(decisions)
        return blocking

    # ─── Introspection ───

    @property
    def decisions(self) -> list[GuardDecision]:
        return list(self._decision_log)

    def clear_decisions(self) -> None:
        self._decision_log.clear()
