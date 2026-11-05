"""Hermes Agent adapter — wires ClawKeeper into NousResearch/hermes-agent.

Two interception surfaces (both are documented Hermes integration points,
no host internals patched for behaviour):

1.  `tools.terminal_tool.set_approval_callback(cb)` and
    `tools.computer_use.tool.set_approval_callback(cb)` — Hermes calls
    these when its own regex flags a command. ClawKeeper's Judge
    decides allow/ask/deny here.

2.  `tool_start_callback` / `tool_complete_callback` on `AIAgent` —
    these fire for EVERY tool call, regardless of whether Hermes
    flagged it. ClawKeeper runs its proactive guard chain here
    (exec_gate, path_guard, url_safety) so attacks that bypass
    Hermes's regex (Python heredocs, multi-turn drift) still get
    inspected. Closes the structural bypass surfaced by red-team
    scenario 07.a.

`install(judge, agent)` registers both. Hermes does not need any patches.

Compatibility: the existing public surface (`install(judge, agent)`,
behaviour of the approval callback) is preserved. The proactive
guard chain is additive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from clawkeeper_core.adapters.base import (
    GuardCallable,
    make_event,
    run_guard_chain,
)
from clawkeeper_core.guards.credential_redact import credential_redact_guard
from clawkeeper_core.guards.exec_gate import guard_execution
from clawkeeper_core.guards.path_guard import guard_before_tool_call
from clawkeeper_core.guards.return_content_scan import return_content_scan_guard
from clawkeeper_core.guards.script_body_scan import script_body_scan_guard
from clawkeeper_core.guards.url_safety import url_safety_guard
from clawkeeper_core.judge import Judge
from clawkeeper_core.learned_rules import match_learned, pin_snapshot
from clawkeeper_core.schemas import AgentEvent

if TYPE_CHECKING:
    from run_agent import AIAgent  # noqa: F401


# ─── Decision → Hermes-return mapping ──────────────────────────────────────

_HERMES_RETURN_FOR_JS_DECISION: dict[str, str] = {
    "continue": "once",
    "stop": "deny",
    "ask_user": "deny",
}

# Backwards-compat alias kept so consumers that imported this constant directly
# don't break.
_HERMES_RETURN_FOR_OUTCOME: dict = {}
try:
    from clawkeeper_core.schemas import Outcome, Scope
    _HERMES_RETURN_FOR_OUTCOME = {
        Outcome.ALLOW: {
            Scope.ONCE.value: "once",
            Scope.SESSION.value: "session",
            Scope.ALWAYS.value: "always",
        },
        Outcome.ASK: {Scope.ONCE.value: "deny", Scope.SESSION.value: "deny", Scope.ALWAYS.value: "deny"},
        Outcome.DENY: {Scope.ONCE.value: "deny", Scope.SESSION.value: "deny", Scope.ALWAYS.value: "deny"},
    }
except ImportError:
    pass


# ─── Blocking exception (raised from tool_start when a guard wants to stop) ──


class ClawKeeperBlock(Exception):
    """Raised by ClawKeeper's tool_start_callback to abort a tool call.

    Hermes's main loop catches callback exceptions and surfaces them to the
    LLM as a tool-execution failure. The decision dict on this exception is
    what gets logged for the audit trail.
    """

    def __init__(self, decision: dict):
        self.decision = decision
        super().__init__(decision.get("reason", "blocked by ClawKeeper"))


# ─── Default proactive guard chains ────────────────────────────────────────


def _default_pre_guards() -> list[GuardCallable]:
    """Pre-execution guards that inspect upcoming commands.

    Order matters: cheap pattern matches first, then path resolution,
    then URL parsing.
    """
    return [
        guard_execution,            # dangerous-command regex (CK's broader set than Hermes's)
        guard_before_tool_call,     # protected file accesses from shared command/path facts
        url_safety_guard,           # homoglyph + SSRF + blocked-hostname
        script_body_scan_guard,     # sensitive accesses through bounded path associations
    ]


def _default_post_guards() -> list[GuardCallable]:
    """Post-execution observers — scan tool results before the LLM consumes them.

    These are observation-and-log only on the Hermes side (we can't mutate the
    tool result that Hermes passes back to the LLM through `tool_complete_callback`).
    On the MCP gateway adapter, the same guards block at the result boundary.
    """
    return [
        return_content_scan_guard,
        credential_redact_guard,
    ]


# ─── Public API: install ────────────────────────────────────────────────────


def install(
    judge: Judge,
    agent: "AIAgent",
    *,
    pre_guards: list[GuardCallable] | None = None,
    post_guards: list[GuardCallable] | None = None,
    watcher_url: str | None = None,
) -> None:
    """Wire a Judge + a chain of guards into a constructed AIAgent.

    Call after `AIAgent(...)`, before `agent.run(...)`.

    `pre_guards` is the proactive chain run on every tool call via
    `tool_start_callback`. Defaults to the standard CK guard set.

    `post_guards` is run on every tool result via `tool_complete_callback`
    (observation only on Hermes — see module docstring).

    Both default chains are conservative: passing your own list lets a
    deployment opt in/out per guard.
    """
    pre = pre_guards if pre_guards is not None else _default_pre_guards()
    post = post_guards if post_guards is not None else _default_post_guards()
    watcher_client = None
    if watcher_url:
        from clawkeeper_core.watcher.client import WatcherClient  # noqa: PLC0415 -- optional Watcher
        watcher_client = WatcherClient(base_url=watcher_url, on_unavailable="ask")

    # 1) Approval-callback path (Hermes-flagged commands → Judge decides).
    import tools.terminal_tool as terminal_tool
    from tools.computer_use import tool as computer_use_tool

    def approval_cb(command: str, description: str = "") -> str:
        event = {"toolName": "bash", "params": {"command": command, "description": description}}

        eg = guard_execution(event)
        if eg.get("block"):
            _record_decision(agent, "deny", source="exec_gate",
                             reason=eg.get("reason"), evidence=eg)
            return "deny"

        pg = guard_before_tool_call(event)
        if pg.get("block"):
            _record_decision(agent, "deny", source="path_guard",
                             reason=pg.get("reason"), evidence=pg)
            return "deny"

        # Honor synchronized script rules on the approval surface as well.
        learned_script = match_learned("script_body_scan", event)
        if learned_script:
            _record_decision(agent, "deny", source=learned_script["source"],
                             reason=learned_script["reason"], evidence=learned_script["evidence"])
            return "deny"

        # Build the Judge's forwarded-context payload.
        recent = _recent_messages(agent)
        forwarded_messages = [
            *(recent if isinstance(recent, list) else []),
            {"role": "tool", "toolName": "bash", "raw": command, "content": description},
        ]
        result = judge.evaluate({
            "mode": "local",
            "forwardedContext": {
                "messages": forwarded_messages,
                "metadata": {"sessionKey": getattr(agent, "session_id", None)},
            },
            "requestId": getattr(agent, "session_id", None),
        })
        js_decision = result.get("decision", "ask_user") if isinstance(result, dict) else "ask_user"
        hermes_string = _HERMES_RETURN_FOR_JS_DECISION.get(js_decision, "deny")
        _record_decision(agent, hermes_string, source="judge",
                         reason=result.get("stopReason") if isinstance(result, dict) else None,
                         evidence=result if isinstance(result, dict) else {})
        return hermes_string

    guarded_approval = _with_rule_sync(approval_cb, watcher_client, agent, "deny")
    terminal_tool.set_approval_callback(guarded_approval)
    computer_use_tool.set_approval_callback(guarded_approval)

    # NEW: register a Hermes `pre_tool_call` plugin hook.  This is the
    # documented Hermes API for blocking tool execution from outside the
    # built-in approval path — fires for every tool call (not just regex-
    # flagged ones), and Hermes honours the returned block message.
    _register_pre_tool_call_hook(judge, agent, pre_guards=pre, watcher_client=watcher_client)

    # 2) Observation + proactive-guard hooks on the agent instance.
    _register_observers(judge, agent, pre_guards=pre, post_guards=post)


# ─── helpers ────────────────────────────────────────────────────────────────


def _record_decision(
    agent: Any,
    hermes_return: str,
    *,
    source: str,
    reason: Any = None,
    evidence: Any = None,
) -> None:
    log = getattr(agent, "_clawkeeper_decisions", None)
    if log is None:
        log = []
        try:
            agent._clawkeeper_decisions = log  # type: ignore[attr-defined]
        except AttributeError:
            return
    log.append({
        "ts": _now().isoformat(),
        "hermes_return": hermes_return,
        "source": source,
        "reason": reason,
        "evidence": evidence,
    })


def _recent_messages(agent: Any, n: int = 20) -> list:
    history = getattr(agent, "conversation_history", None) or []
    return history[-n:]


def _user_text(content: Any) -> str:
    if isinstance(content, list):
        content = "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return content.strip() if isinstance(content, str) else ""


def _stated_user_intent(agent: Any) -> str | None:
    """Read user turns in order, retaining the original task and later revisions.

    Use the full host-owned history rather than the Judge's recent-message
    window: long tool trajectories must not discard the original constraints.
    Assistant/tool text and non-text content blocks are not user authorization.
    """
    turns = []
    for message in getattr(agent, "conversation_history", None) or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _user_text(message.get("content"))
        if text:
            turns.append(text)
    return "\n\n".join(turns) or None


def _with_rule_sync(callback, watcher_client, agent, blocked_result):
    if watcher_client is None:
        return callback

    def synchronized(*args, **kwargs):
        try:
            snapshot = watcher_client.sync_rules()
        except Exception:
            _record_decision(agent, "deny", source="watcher:rule_sync",
                             reason="Rule synchronization failed; ask operator", evidence={"decision": "ask"})
            return dict(blocked_result) if isinstance(blocked_result, dict) else blocked_result
        with pin_snapshot(snapshot):
            return callback(*args, **kwargs)

    return synchronized


def _same_agent_session(callback: Any, agent: Any) -> bool:
    owner = getattr(callback, "_clawkeeper_agent", None)
    return owner is agent or (
        owner is not None
        and str(getattr(owner, "session_id", "default")) == str(getattr(agent, "session_id", "default"))
    )


def _replace_session_hook(manager: Any, name: str, agent: Any, callback: Any = None) -> None:
    hooks = manager._hooks.setdefault(name, [])
    hooks[:] = [existing for existing in hooks if not _same_agent_session(existing, agent)]
    if callback is not None:
        callback._clawkeeper_agent = agent
        hooks.append(callback)


def _register_pre_tool_call_hook(judge, agent, *, pre_guards, watcher_client=None):

    """Register a Hermes plugin `pre_tool_call` hook that runs CK's guard chain.

    Returning {"action": "block", "message": ...} from this hook causes Hermes
    to skip tool execution and surface the message to the LLM as a tool error
    (the documented Hermes plugin API — `tools/approval.py::check_all_command_guards`
    is unrelated and not patched).
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered
    except ImportError:  # pragma: no cover
        return
    manager = _ensure_plugins_discovered()

    # Hermes owns a process-global hook registry. Capture only this agent's
    # original user input, before model/tool output can change its meaning.
    # Keep it independently of Hermes's model-facing history/compression.
    intents: dict[str, list[str]] = {}
    session_key = str(getattr(agent, "session_id", "default"))

    # Gateways can construct a fresh agent for each turn of the same session.
    # Hand over captured input and replace its old hooks instead of consulting
    # the Watcher once for every agent ever installed in that session.
    for callback in manager._hooks.get("pre_llm_call", []):
        if _same_agent_session(callback, agent):
            previous = getattr(callback, "_clawkeeper_intents", {})
            intents.update({
                key: list(turns) for key, turns in previous.items()
                if key == session_key or getattr(callback, "_clawkeeper_agent", None) is agent
            })

    def capture_user_intent(*, user_message, session_id="", **_kwargs):
        if not session_id or str(session_id) != str(getattr(agent, "session_id", "default")):
            return None
        text = _user_text(user_message)
        if text:
            key = str(session_id)
            if key not in intents:
                previous = _stated_user_intent(agent)
                intents[key] = [previous] if previous else []
            if not intents[key] or intents[key][-1] != text:
                intents[key].append(text)

    capture_user_intent._clawkeeper_intents = intents
    _replace_session_hook(
        manager, "pre_llm_call", agent, capture_user_intent if watcher_client is not None else None
    )

    def ck_pre_tool_call(*, tool_name, args, task_id="", session_id="", tool_call_id="", **_kwargs):
        command = ""
        if isinstance(args, dict):
            command = str(args.get("command") or args.get("url") or "")
        event = make_event(
            tool_name=str(tool_name),
            command=command,
            params=args if isinstance(args, dict) else {},
            session_id=getattr(agent, "session_id", None),
            direction="pre_tool_call",
        )
        blocking, decisions = run_guard_chain(pre_guards, event, stop_on_block=True)
        for d in decisions:
            if d is None:
                continue
            _record_decision(
                agent,
                "deny" if d.get("block") else "once",
                source=d.get("source", "?"),
                reason=d.get("reason"),
                evidence=d.get("evidence"),
            )
        if blocking is not None:
            src = blocking.get("source", "?")
            reason = blocking.get("reason", "blocked")
            return {"action": "block", "message": f"[ClawKeeper:{src}] {reason}"}

        if watcher_client is not None:
            try:
                key = str(getattr(agent, "session_id", "default"))
                d = watcher_client.evaluate(
                    session_id=key,
                    tool_name=str(tool_name),
                    args=args if isinstance(args, dict) else {},
                    stated_intent="\n\n".join(intents[key]) if key in intents else _stated_user_intent(agent),
                )
                _record_decision(
                    agent,
                    "deny" if d.get("decision") in {"deny", "ask"} else "once",
                    source=f"watcher:{d.get('decision','?')}",
                    reason=d.get("reason"),
                    evidence=d,
                )
                if d.get("decision") in {"deny", "ask"}:
                    return {
                        "action": "block",
                        "message": f"[ClawKeeper:watcher] {d.get('reason','blocked by Watcher')}",
                    }
            except Exception as exc:
                import sys
                print(f"[clawkeeper] watcher consult error: {exc!r}", file=sys.stderr)

        return None

    synchronized = _with_rule_sync(
        ck_pre_tool_call, watcher_client, agent,
        {"action": "block", "message": "[ClawKeeper:watcher] Rule synchronization failed; ask operator"},
    )

    def scoped_pre_tool_call(**kwargs):
        session_id = kwargs.get("session_id", "")
        if session_id and str(session_id) != str(getattr(agent, "session_id", "default")):
            return None
        return synchronized(**kwargs)

    _replace_session_hook(manager, "pre_tool_call", agent, scoped_pre_tool_call)


def _register_observers(
    judge: Judge,
    agent: Any,
    *,
    pre_guards: list[GuardCallable],
    post_guards: list[GuardCallable],
) -> None:
    """Wrap `tool_start_callback` / `tool_complete_callback` on the agent.

    Preserves any previously-set callback so multiple wrappers stack.
    """
    prev_tool_start = agent.tool_start_callback
    prev_tool_complete = agent.tool_complete_callback

    def on_tool_start(*args: Any) -> None:
        if len(args) == 3:
            tool_call_id, name, tool_args = args
        elif len(args) == 2:
            tool_call_id, name, tool_args = None, args[0], args[1]
        else:
            return

        # Build the standard event the guards consume.
        command = ""
        params_dict = {}
        if isinstance(tool_args, dict):
            params_dict = dict(tool_args)
            command = str(tool_args.get("command") or tool_args.get("url") or "")
        else:
            command = str(tool_args)
        event = make_event(
            tool_name=str(name),
            command=command,
            params=params_dict,
            session_id=getattr(agent, "session_id", None),
            direction="pre_tool_call",
        )

        # Proactive guard chain — closes scenario 07.a (Hermes regex missed
        # the heredoc, so approval_cb was never invoked; now we run the same
        # guards on every tool start).
        blocking, decisions = run_guard_chain(pre_guards, event, stop_on_block=True)
        for d in decisions:
            if d is None:
                continue
            _record_decision(
                agent,
                "deny" if d.get("block") else "once",
                source=d.get("source", "proactive_guard"),
                reason=d.get("reason"),
                evidence=d.get("evidence"),
            )

        # Existing observer event (for profiler / drift detector).
        _emit(judge, AgentEvent(
            ts=_now(),
            agent_id=getattr(agent, "session_id", "unknown"),
            session_id=getattr(agent, "session_id", None),
            kind="tool_start",
            payload={
                "tool_call_id": tool_call_id,
                "name": name,
                "args_preview": _stringify(tool_args),
            },
        ))

        if prev_tool_start:
            prev_tool_start(*args)
        # NB: we do NOT raise here. Hermes wraps tool_start_callback in a
        # try/except that swallows exceptions; raising is purely advisory.
        # Real blocking happens in `_register_pre_tool_call_hook` below.

    def on_tool_complete(*args: Any) -> None:
        if len(args) == 4:
            tool_call_id, name, _tool_args, result = args
        elif len(args) == 3:
            tool_call_id, name, result = None, args[0], args[2]
        else:
            return

        # Build event for post-guards. Result text is what the LLM is about to see.
        event = make_event(
            tool_name=str(name),
            content=_stringify(result),
            params={"result": _stringify(result)},
            session_id=getattr(agent, "session_id", None),
            direction="post_tool_call",
        )
        # Post-guard chain is observation+log on Hermes (we can't mutate the
        # result inline). Decisions are still recorded for audit.
        _blocking, decisions = run_guard_chain(post_guards, event, stop_on_block=False)
        for d in decisions:
            if d is None:
                continue
            _record_decision(
                agent,
                "deny" if d.get("block") else "once",
                source=d.get("source", "post_guard"),
                reason=d.get("reason"),
                evidence=d.get("evidence"),
            )

        # Existing observer
        _emit(judge, AgentEvent(
            ts=_now(),
            agent_id=getattr(agent, "session_id", "unknown"),
            session_id=getattr(agent, "session_id", None),
            kind="tool_complete",
            payload={
                "tool_call_id": tool_call_id,
                "name": name,
                "result_preview": _stringify(result)[:500],
            },
        ))

        if prev_tool_complete:
            prev_tool_complete(*args)

    agent.tool_start_callback = on_tool_start
    agent.tool_complete_callback = on_tool_complete


def _emit(judge: Judge, event: AgentEvent) -> None:
    profiler = getattr(judge, "profiler", None)
    if profiler is not None:
        profiler.update(event)


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _stringify(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return "<unprintable>"
