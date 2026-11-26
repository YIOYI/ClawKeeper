"""Context judge — the central decision engine.

Ported from legacy/clawkeeper-watcher/plugins/clawkeeper-watcher/src/core/context-judge.js.

`judge_forwarded_context(input)` is the single entry point. It evaluates a
forwarded conversation (`forwardedContext.messages`) against a policy and
returns a Decision describing whether to continue, ask the user, or stop.

DESIGN.md exposes this through the `Judge` class — that wrapper lives at
the bottom of this module and is what adapters import.
"""

from __future__ import annotations

import re
from typing import Any

# ── Policy & rank constants ────────────────────────────────────────────────


DEFAULT_POLICY: dict[str, Any] = {
    "maxRiskBeforeStop": "critical",
    "requireUserConfirmationFor": [],
    "autoContinueAllowed": False,
    "maxToolStepsWithoutUserTurn": 3,
}

RISK_ORDER: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Word boundaries (\b) are intentionally omitted so CJK phrases still match
# alongside English keywords.
_USER_STOP_RE = re.compile(
    r"^(?:(?:please\s+)?(?:stop|cancel|abort)\b(?!\s+(?:by|at)\b)[^,;.!?\n]*"
    r"|(?:请\s*)?(?:停止|取消|终止)(?:执行|运行|任务|构建|当前任务)?|不要继续(?:了)?|算了)"
    r"[。\uFF01\uFF1F.!?\s]*$",
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _normalize_policy(policy: Any) -> dict[str, Any]:
    p = policy if isinstance(policy, dict) else {}
    merged = {**DEFAULT_POLICY, **p}
    if not isinstance(p.get("requireUserConfirmationFor"), list):
        merged["requireUserConfirmationFor"] = DEFAULT_POLICY["requireUserConfirmationFor"]
    return merged


def _normalize_message(message: Any) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role") if isinstance(message.get("role"), str) else "unknown"
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    tool_name = message.get("toolName") if isinstance(message.get("toolName"), str) else (
        message.get("name") if isinstance(message.get("name"), str) else ""
    )
    error = message.get("error") if isinstance(message.get("error"), str) else ""
    result = message.get("result") if isinstance(message.get("result"), str) else ""
    return {
        "role": role,
        "content": content,
        "toolName": tool_name,
        "error": error,
        "result": result,
    }


def _last_message_by_role(messages: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == role:
            return m
    return None


def _last_message_index_by_role(messages: list[dict[str, Any]], role: str) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == role:
            return i
    return -1


_EVIDENCE_RESULT_RE = re.compile(r"CTX_PROBE_|CMD_OUT_|exitCode|signal|stderr|stdout", re.IGNORECASE)


def _summarize_evidence(messages: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for message in messages:
        if not message:
            continue
        tool_name = message.get("toolName")
        if tool_name:
            evidence.append(f"tool={tool_name}")
        err = message.get("error")
        if err:
            evidence.append(f"error={err[:120]}")
        result = message.get("result")
        if result and _EVIDENCE_RESULT_RE.search(result):
            evidence.append(f"result={result[:120]}")
    return evidence[:8]


def _build_base_decision(**overrides: Any) -> dict[str, Any]:
    base = {
        "version": 1,
        "decision": "continue",
        "stopReason": "completed",
        "shouldContinue": True,
        "needsUserDecision": False,
        "userQuestion": None,
        "summary": "The current context does not require any additional confirmation.",
        "riskLevel": "low",
        "evidence": [],
        "nextAction": "continue_run",
        "continueHint": None,
    }
    base.update(overrides)
    return base


def decision_from_guard(
    finding: dict[str, Any], mode: str = "local", *, can_ask: bool = True,
) -> dict[str, Any]:
    """Translate a blocking guard finding into the existing adapter decision format."""
    review = finding.get("outcome") == "ask"
    ask = review and can_ask
    reason = finding.get("reason") or "The tool call was blocked by a pre-execution check."
    severity = str(finding.get("severity") or "high").lower()
    if severity == "hardline":
        severity = "critical"
    evidence = [f"guard={finding.get('source', 'unknown')}"]
    if finding.get("matched"):
        evidence.append(f"rule={finding['matched']}")
    return _build_base_decision(
        mode=mode, localEnhanced=mode == "local",
        decision="ask_user" if ask else "stop",
        stopReason="guard_review" if review else "guard_blocked",
        shouldContinue=False, needsUserDecision=ask,
        summary=reason, userQuestion=reason if ask else None,
        riskLevel=severity if severity in RISK_ORDER else "high",
        evidence=evidence, nextAction="ask_user" if ask else "stop_run",
    )


# ── Main entry ─────────────────────────────────────────────────────────────


def judge_forwarded_context(input: Any = None) -> dict[str, Any]:
    """Decide allow / ask / stop for one forwarded conversation."""
    inp = input if isinstance(input, dict) else {}
    mode = inp.get("mode") if isinstance(inp.get("mode"), str) else "local"
    local_enhanced = mode == "local"

    def decide(**overrides: Any) -> dict[str, Any]:
        return _build_base_decision(localEnhanced=local_enhanced, mode=mode, **overrides)

    request_id = inp.get("requestId") if isinstance(inp.get("requestId"), str) else None
    forwarded = inp.get("forwardedContext") if isinstance(inp.get("forwardedContext"), dict) else None
    policy = _normalize_policy(inp.get("policy"))

    if (
        not forwarded
        or not isinstance(forwarded.get("messages"), list)
        or len(forwarded["messages"]) == 0
    ):
        return decide(
            decision="stop",
            stopReason="missing_input",
            shouldContinue=False,
            summary="Missing forwardedContext.messages. Context judgment cannot be completed.",
            riskLevel="medium",
            evidence=[f"requestId={request_id}"] if request_id else [],
            nextAction="stop_run",
        )

    normalized: list[dict[str, str]] = []
    for m in forwarded["messages"]:
        norm = _normalize_message(m)
        if norm is not None:
            normalized.append(norm)

    metadata = forwarded.get("metadata") if isinstance(forwarded.get("metadata"), dict) else {}

    last_user_idx = _last_message_index_by_role(normalized, "user")
    active_messages = normalized[last_user_idx:] if last_user_idx >= 0 else normalized
    last_user = _last_message_by_role(normalized, "user")

    tool_messages = [
        m for m in active_messages
        if (m.get("toolName") or m["role"] == "tool")
        and m["toolName"] != "clawbands_respond"
        and m["toolName"] != "clawkeeper_bands_respond"
    ]
    tool_count = len(tool_messages)
    tool_names = {m["toolName"].strip().lower() for m in tool_messages if m.get("toolName")}
    has_tool_error = any(m.get("error") for m in active_messages)
    last_user_content = last_user.get("content", "") if last_user else ""

    evidence: list[str] = []
    if request_id:
        evidence.append(f"requestId={request_id}")
    if metadata.get("sessionKey"):
        evidence.append(f"sessionKey={metadata['sessionKey']}")
    evidence.extend([
        f"messageCount={len(normalized)}",
        f"activeMessageCount={len(active_messages)}",
        f"toolCount={tool_count}",
    ])
    evidence.extend(_summarize_evidence(active_messages))

    if metadata.get("cancelled") is True or _USER_STOP_RE.fullmatch(last_user_content.strip()):
        return decide(
            decision="stop",
            stopReason="user_requested_stop",
            shouldContinue=False,
            summary="The run received a cancellation signal or matched a stop instruction.",
            riskLevel="low",
            evidence=evidence,
            nextAction="stop_run",
        )

    if has_tool_error or metadata.get("success") is False:
        upstream_error = metadata.get("error")
        return decide(
            decision="stop",
            stopReason="upstream_error" if upstream_error else "unknown",
            shouldContinue=False,
            summary=(
                f"The upstream run failed: {str(upstream_error)[:160]}"
                if upstream_error
                else "An error context was detected in the upstream run. "
                     "Stop first and inspect the failure before continuing."
            ),
            riskLevel="high",
            evidence=evidence,
            nextAction="stop_run",
        )

    if tool_count > policy["maxToolStepsWithoutUserTurn"]:
        return decide(
            decision="ask_user",
            stopReason="tool_loop_limit",
            shouldContinue=False,
            needsUserDecision=True,
            userQuestion=f"{tool_count} tool calls have already run in this turn. "
                         f"Do you want to continue with the next step?",
            summary="The consecutive tool-call limit has been reached. Return to the user for confirmation.",
            riskLevel="high",
            evidence=evidence,
            nextAction="ask_user",
            continueHint="Continue only after explicit user confirmation, and reduce the number "
                         "of tool calls in the next turn if possible.",
        )

    needs_confirmation = any(str(k).strip().lower() in tool_names for k in policy["requireUserConfirmationFor"])
    if needs_confirmation and not policy["autoContinueAllowed"]:
        return decide(
            decision="ask_user",
            stopReason="waiting_user_confirmation",
            shouldContinue=False,
            needsUserDecision=True,
            userQuestion="The configured policy requires confirmation. Do you want to continue?",
            summary="A named tool matched the configured confirmation policy.",
            riskLevel="high",
            evidence=evidence,
            nextAction="ask_user",
            continueHint="If the user confirms, execution may continue in the next turn, "
                         "but avoid asking the same confirmation again.",
        )

    highest_risk = "low"
    stop_at = str(policy.get("maxRiskBeforeStop") or "critical").lower()
    if RISK_ORDER[highest_risk] >= (RISK_ORDER.get(stop_at, RISK_ORDER["critical"])):
        return decide(
            decision="stop",
            stopReason="high_risk_action",
            shouldContinue=False,
            summary=f"The risk level {highest_risk} has reached the policy threshold {stop_at}. "
                    f"This turn should stop.",
            riskLevel=highest_risk,
            evidence=evidence,
            nextAction="stop_run",
        )

    return decide(
        decision="continue",
        stopReason="completed",
        shouldContinue=True,
        needsUserDecision=False,
        summary="The current context may continue without additional user confirmation.",
        riskLevel=highest_risk,
        evidence=evidence,
        nextAction="continue_run",
    )


# ── OO façade matching DESIGN.md surface ────────────────────────────────────


class Judge:
    """Convenience wrapper. DESIGN.md's public entry point."""

    def __init__(self, policy: dict[str, Any] | None = None, **_unused: Any) -> None:
        self.policy = policy

    def evaluate(self, input: Any = None) -> dict[str, Any]:
        if isinstance(input, dict) and self.policy is not None and "policy" not in input:
            input = {**input, "policy": self.policy}
        return judge_forwarded_context(input)

    async def evaluate_async(self, input: Any = None) -> dict[str, Any]:
        return self.evaluate(input)
