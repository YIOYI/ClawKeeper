"""FastAPI HTTP surface — the bridge to non-Python adapters.

The OpenClaw JS plugin (and any other non-Python adapter) consumes this
to delegate decisions to the Python core. All endpoints accept the same
shape the JS modules already produce / consume, so adapters can forward
their existing payloads with minimal translation.

Endpoints:
  GET  /v1/health                — liveness + version
  POST /v1/judge                 — judge one forwarded context
  POST /v1/event                 — fire-and-forget telemetry
  POST /v1/audit                 — startup / on-demand audit
  POST /v1/scan/logs             — log scan
  POST /v1/scan/skill            — skill scan
  POST /v1/maintenance/harden    — backup + auto-fix
  POST /v1/maintenance/rollback  — restore latest (or named) backup
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from clawkeeper_core import __version__
from clawkeeper_core.adapters.base import make_event, run_pre_tool_guards
from clawkeeper_core.audit import create_audit_context, run_audit
from clawkeeper_core.judge import Judge, decision_from_guard
from clawkeeper_core.maintenance import harden, rollback
from clawkeeper_core.scanner import scan_logs_for_security_risks, scan_skill

app = FastAPI(title="clawkeeper-core", version=__version__)
_judge = Judge()


# ── Health ─────────────────────────────────────────────────────────────────


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


# ── Judge ──────────────────────────────────────────────────────────────────


class ProposedToolCall(BaseModel):
    toolName: str = ""
    params: dict[str, Any]
    cwd: str | None = None


class JudgePayload(BaseModel):
    """Flexible JSON shape mirroring the JS `judgeForwardedContext` input.

    The JS plugin already builds this dict — we forward it verbatim. The
    Pydantic model is permissive (`extra = "allow"`) so adapters can add
    fields without breaking us.
    """

    model_config = {"extra": "allow"}

    mode: str | None = None
    requestId: str | None = None
    forwardedContext: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    toolCall: ProposedToolCall | None = None


def _legacy_tool_call(messages: Any) -> ProposedToolCall | None:
    """Older adapters append the proposed call as a terminal tool message with raw arguments."""
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    message = messages[-1]
    if message.get("role") != "tool" or "result" in message or "error" in message:
        return None
    raw = message.get("raw")
    if isinstance(raw, str) and raw.strip():
        # Legacy OpenClaw sent a JSON object, while other callers sent a shell string.
        try:
            decoded = json.loads(raw)
        except ValueError:
            decoded = None
        raw = decoded if isinstance(decoded, dict) else {"command": raw}
    if not isinstance(raw, dict):
        return None
    name = message.get("toolName", message.get("name"))
    return ProposedToolCall(toolName=name if isinstance(name, str) else "", params=raw)


@app.post("/v1/judge")
async def judge(payload: JudgePayload) -> dict[str, Any]:
    request = payload.model_dump(exclude_none=True)
    context = payload.forwardedContext or {}
    messages = context.get("messages")
    call = payload.toolCall or _legacy_tool_call(messages)
    if call is not None:
        event = make_event(tool_name=call.toolName, params=call.params)
        if call.cwd is not None:
            event["cwd"] = call.cwd
        blocking, _ = await run_in_threadpool(run_pre_tool_guards, event)
        if blocking is not None:
            return decision_from_guard(blocking, payload.mode or "local")
    if payload.toolCall is not None:
        history = messages if isinstance(messages, list) else []
        request["forwardedContext"] = {
            **context, "messages": [*history, {"role": "tool", "toolName": payload.toolCall.toolName}],
        }
    return await _judge.evaluate_async(request)


# ── Event (fire-and-forget telemetry) ──────────────────────────────────────


class EventPayload(BaseModel):
    model_config = {"extra": "allow"}

    kind: str
    agent_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] | None = None


@app.post("/v1/event")
async def event(_event: EventPayload) -> dict[str, bool]:
    # v0.2.0: just acknowledge. Profiler / decision-memory wiring lands
    # when the server-side AgentProfiler is wired up.
    return {"ok": True}


# ── Audit ──────────────────────────────────────────────────────────────────


class AuditPayload(BaseModel):
    stateDir: str
    strictMode: bool = False


@app.post("/v1/audit")
async def audit(payload: AuditPayload) -> dict[str, Any]:
    ctx = await create_audit_context(Path(payload.stateDir), {"strictMode": payload.strictMode})
    return await run_audit(ctx)


# ── Maintenance ────────────────────────────────────────────────────────────


class HardenPayload(BaseModel):
    stateDir: str


@app.post("/v1/maintenance/harden")
async def maintenance_harden(payload: HardenPayload) -> dict[str, Any]:
    return await harden(Path(payload.stateDir))


class RollbackPayload(BaseModel):
    stateDir: str
    backupName: str | None = None


@app.post("/v1/maintenance/rollback")
async def maintenance_rollback(payload: RollbackPayload) -> dict[str, Any]:
    return await rollback(Path(payload.stateDir), payload.backupName)


# ── Scanners ───────────────────────────────────────────────────────────────


class LogScanPayload(BaseModel):
    records: list[dict[str, Any]]


@app.post("/v1/scan/logs")
async def scan_logs(payload: LogScanPayload) -> dict[str, Any]:
    return await scan_logs_for_security_risks(payload.records)


class SkillScanPayload(BaseModel):
    path: str
    stateDir: str | None = None
    rulesPath: str | None = None


@app.post("/v1/scan/skill")
async def scan_skill_endpoint(payload: SkillScanPayload) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if payload.stateDir:
        opts["stateDir"] = payload.stateDir
    if payload.rulesPath:
        opts["rulesPath"] = payload.rulesPath
    return await scan_skill(payload.path, opts)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    uvicorn.run("clawkeeper_core.server:app", host="127.0.0.1", port=7474, reload=False)


if __name__ == "__main__":
    main()
