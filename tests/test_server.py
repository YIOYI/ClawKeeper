"""HTTP server tests — exercise every endpoint via FastAPI's TestClient.

These tests live in the clawkeeper env (no hermes needed). They prove
that the server contract the JS shim relies on is stable.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from clawkeeper_core.server import app


@pytest.fixture
def client():
    return TestClient(app)


# ── /v1/health ────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


# ── /v1/judge ─────────────────────────────────────────────────────────────


def test_judge_js_style_payload_returns_dict(client):
    r = client.post("/v1/judge", json={
        "mode": "local",
        "requestId": "req-1",
        "forwardedContext": {
            "messages": [
                {"role": "user", "content": "stop the build"},
            ],
        },
    })
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "stop"
    assert body["stopReason"] == "user_requested_stop"


def test_judge_missing_input(client):
    r = client.post("/v1/judge", json={"mode": "local"})
    assert r.status_code == 200
    assert r.json()["decision"] == "stop"
    assert r.json()["stopReason"] == "missing_input"


@pytest.mark.parametrize("name", ["bash", "terminal", "custom_runner", "read_file", "write"])
@pytest.mark.parametrize(("params", "decision", "guard"), [
    ({"command": "ls -la"}, "continue", None),
    ({"input": "What's on the agenda?"}, "continue", None),
    ({"request": {"cmd": "whoami"}}, "stop", "exec_gate"),
    ({"input": "whoami"}, "stop", "exec_gate"),
    ({"command": "whoami"}, "stop", "exec_gate"),
    ({"path": "/etc/shadow"}, "stop", "path_guard"),
    ({"code": "open('/tmp/example/.docker/config.json').read()"}, "stop", "script_body_scan"),
    ({"url": "http://127.0.0.1/"}, "stop", "url_safety"),
    ({"command": "printf 'unfinished"}, "ask_user", "exec_gate"),
    ({"path": "/tmp/example", "content": {"command": "whoami"}}, "continue", None),
])
def test_judge_http_checks_content_independent_of_tool_name(client, name, params, decision, guard):
    response = client.post("/v1/judge", json={
        "toolCall": {"toolName": name, "params": params},
    })
    body = response.json()
    assert body["decision"] == decision
    assert body["shouldContinue"] is (decision == "continue")
    if decision == "continue":
        assert body["toolCallChecked"] is True
    if guard:
        assert any(item.startswith(f"guard={guard}") for item in body["evidence"])


@pytest.mark.parametrize("raw", ["whoami", '{"command":"whoami"}'])
def test_legacy_proposed_tool_raw_is_checked(client, raw):
    response = client.post("/v1/judge", json={"forwardedContext": {"messages": [
        {"role": "tool", "toolName": "custom_runner", "raw": raw},
    ]}})
    assert response.json()["stopReason"] == "guard_blocked"


def test_legacy_client_receives_stop_for_incomplete_guard_check(client):
    response = client.post("/v1/judge", json={"forwardedContext": {"messages": [
        {"role": "tool", "toolName": "terminal", "raw": "printf 'unfinished"},
    ]}})
    assert response.json()["decision"] == "stop"
    assert response.json()["stopReason"] == "guard_review"


@pytest.mark.parametrize(("cwd", "workdir"), [("/tmp", "/etc"), ("/", "etc")])
def test_guard_resolves_relative_paths_from_the_calls_working_directory(client, cwd, workdir):
    response = client.post("/v1/judge", json={"toolCall": {
        "toolName": "read_file", "cwd": cwd, "params": {"workdir": workdir, "path": "shadow"},
    }})
    assert response.json()["evidence"][0] == "guard=path_guard"


def test_structured_call_does_not_recheck_history(client):
    response = client.post("/v1/judge", json={
        "forwardedContext": {"messages": [{"role": "tool", "toolName": "bash", "raw": "whoami"}]},
        "toolCall": {"toolName": "terminal", "params": {"command": "ls -la"}},
    })
    assert response.json()["decision"] == "continue"


def test_judge_accepts_extra_fields(client):
    """JS clients may include fields we don't model (decision-memory keys,
    profiler hints). The endpoint should accept them without 422-ing.
    """
    r = client.post("/v1/judge", json={
        "mode": "local",
        "forwardedContext": {"messages": [{"role": "user", "content": "ok"}]},
        "extraField": "should be ignored not rejected",
    })
    assert r.status_code == 200


# ── /v1/event ─────────────────────────────────────────────────────────────


def test_event_acknowledges(client):
    r = client.post("/v1/event", json={
        "kind": "tool_start",
        "agent_id": "agent-1",
        "payload": {"name": "bash"},
    })
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── /v1/scan/logs ─────────────────────────────────────────────────────────


def test_scan_logs_empty(client):
    r = client.post("/v1/scan/logs", json={"records": []})
    assert r.status_code == 200
    assert r.json()["totalEvents"] == 0


def test_scan_logs_detects_dangerous_command(client):
    r = client.post("/v1/scan/logs", json={"records": [
        {"type": "before_tool_call", "toolName": "bash", "params": {"command": "rm -rf /"}},
    ]})
    body = r.json()
    titles = [risk["title"] for risk in body["risks"]]
    assert any("Dangerous Command" in t for t in titles)


# ── /v1/scan/skill ────────────────────────────────────────────────────────


def test_scan_skill_clean(client, tmp_path):
    skill = tmp_path / "clean"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# clean")
    (skill / "skill.json").write_text("{}")
    r = client.post("/v1/scan/skill", json={"path": str(skill)})
    assert r.status_code == 200
    assert r.json()["score"] == 100


# ── /v1/audit + /v1/maintenance/harden + /v1/maintenance/rollback ─────────


def test_audit_harden_rollback_via_http(client, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "openclaw.json").write_text(json.dumps({
        "gateway": {"bind": "lan", "auth": {"token": "x"}},
        "agents": {"defaults": {"sandbox": {"mode": "off"}}},
        "tools": {"exec": {"security": "full"}},
    }))

    # Audit identifies issues
    r = client.post("/v1/audit", json={"stateDir": str(state)})
    assert r.status_code == 200
    finding_ids = {f["id"] for f in r.json()["findings"]}
    assert "network.local-gateway" in finding_ids

    # Harden fixes them
    r = client.post("/v1/maintenance/harden", json={"stateDir": str(state)})
    assert r.status_code == 200
    actions = r.json()["actions"]
    assert any("gateway.bind" in a for a in actions)

    # Rollback restores
    backup_name = r.json()["backupDir"].rsplit("/", 1)[-1]
    r = client.post("/v1/maintenance/rollback", json={"stateDir": str(state), "backupName": backup_name})
    assert r.status_code == 200
    assert "openclaw.json" in r.json()["restoredFiles"]
