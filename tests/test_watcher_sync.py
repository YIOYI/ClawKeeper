"""Protocol rejection, Hermes interception, and real process separation."""

# Numeric assertions and fixture-local imports keep scenario setup explicit.
# ruff: noqa: PLR2004, PLC0415

import hashlib
import json
import select
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from clawkeeper_core import learned_rules as rules
from clawkeeper_core.watcher.client import RuleSyncError, WatcherClient
from tests.watcher_process import fake_hermes_modules


def payload():
    return rules.make_snapshot([rules.validate_rule(r"CK_CANARY_\d+", "exec_gate")]).as_dict()


def response(data):
    return httpx.Response(200, json=data, headers={"ETag": f'"{data["version"]}"'})


def test_client_conditional_sync_and_single_authority():
    body = payload()
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return response(body)
        return httpx.Response(
            304,
            headers={"ETag": f'"{body["version"]}"', "X-ClawKeeper-Instance": "restarted-server"},
        )

    with WatcherClient("http://watcher") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        snapshot = client.sync_rules()
        assert client.sync_rules() is snapshot
        assert requests[1].headers["if-none-match"] == f'"{snapshot.version}"'
        assert client._server_instance == "restarted-server"
    with WatcherClient("http://different-watcher") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(lambda _: response(body)))
        with pytest.raises(RuleSyncError):
            client.sync_rules()
    assert rules.current_snapshot() is snapshot


@pytest.mark.parametrize(
    "failure",
    ["version", "target", "regex", "broad", "duplicate", "validator", "old_validator", "previous_validator", "disconnect", "bad304"],
)
def test_invalid_sync_preserves_previous_snapshot(failure):
    body = payload()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(body)
        bad = json.loads(json.dumps(body))
        if failure == "disconnect":
            raise httpx.ConnectError("offline", request=request)
        if failure == "bad304":
            return httpx.Response(304, headers={"ETag": '"wrong"'})
        if failure == "version":
            bad["version"] = "wrong"
        elif failure == "target":
            bad["rules"][0]["guard_target"] = "unknown"
        elif failure in {"regex", "broad"}:
            bad["rules"][0]["pattern"] = "[" if failure == "regex" else ".*"
        elif failure == "duplicate":
            bad["rules"].append(bad["rules"][0])
        elif failure == "old_validator":
            bad["validation_version"] = 2
        elif failure == "previous_validator":
            bad["validation_version"] = 3
        else:
            bad["validation_version"] = 999
        return response(bad)

    with WatcherClient("http://watcher") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        previous = client.sync_rules()
        with pytest.raises(RuleSyncError):
            client.sync_rules()
        assert client._rules is previous and rules.current_snapshot() is previous


@pytest.mark.parametrize(
    "target,command",
    [
        ("exec_gate", "CK_CANARY_8421"),
        ("path_guard", "cat /tmp/CK_CANARY_8421"),
        ("script_body_scan", "CK_CANARY_8421"),
    ],
)
def test_hermes_interceptors_share_client_and_preserve_host_callbacks(monkeypatch, target, command):
    from clawkeeper_core.adapters.hermes import install
    from clawkeeper_core.judge import Judge

    terminal, computer, manager = fake_hermes_modules(
        lambda k, v: monkeypatch.setitem(sys.modules, k, v)
    )
    native_computer_approval = Mock(return_value="session")
    computer.set_approval_callback(native_computer_approval)
    client = Mock()
    snapshot = rules.make_snapshot([rules.validate_rule(r"CK_CANARY_\d+", target)])
    client.sync_rules.return_value = snapshot
    factory = Mock(return_value=client)
    monkeypatch.setattr("clawkeeper_core.watcher.client.WatcherClient", factory)
    agent = SimpleNamespace(
        session_id="s",
        conversation_history=[],
        tool_start_callback=None,
        tool_complete_callback=None,
    )
    install(Judge(), agent, watcher_url="http://watcher")
    hook = manager._hooks["pre_tool_call"][0]
    assert f"learned:{target}" in hook(tool_name="terminal", args={"command": command})["message"]
    assert terminal.callback(command) == "deny"
    assert computer._approval_callback(command) == "deny"
    native_computer_approval.assert_not_called()
    factory.assert_called_once()
    assert client.sync_rules.call_count == 3
    assert terminal.callback("ls -la") == "once"
    assert computer._approval_callback("ls -la") == "session"
    native_computer_approval.assert_called_once_with("ls -la", "")
    client.sync_rules.side_effect = RuleSyncError("offline")
    assert hook(tool_name="terminal", args={"command": "ls -la"})["action"] == "block"
    assert terminal.callback("ls -la") == "deny"
    assert agent._clawkeeper_decisions[-1]["source"] == "watcher:rule_sync"


def test_rules_endpoint_and_legacy_delete(isolated_learning):
    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher.daemon import build_app
    from tests.test_learned_rules import candidate

    assert isolated_learning.add(candidate())
    watcher = SimpleNamespace(
        model=SimpleNamespace(model_id="fake"), history=SimpleNamespace(_calls={})
    )
    with TestClient(build_app(watcher=watcher)) as client:
        first = client.get("/watcher/rules")
        data = first.json()
        assert len(data["rules"]) == 1 and "source_command" not in str(data)
        assert (
            client.get(
                "/watcher/rules", headers={"If-None-Match": first.headers["etag"]}
            ).status_code
            == 304
        )
        assert client.get("/watcher/health").json()["version"] == data["version"]
        legacy_hash = hashlib.sha256(candidate().pattern.encode()).hexdigest()
        deleted = client.delete(f"/watcher/learned/{legacy_hash}")
        assert deleted.status_code == 200
        after = client.get("/watcher/rules")
        assert deleted.json() == {"removed": True, "watcher_runtime": "synchronized",
                                  "version": after.json()["version"], "error": None}
        assert not after.json()["rules"] and after.headers["etag"] != first.headers["etag"]


def test_delete_only_selected_rule_when_legacy_identifiers_overlap(isolated_learning):
    from fastapi.testclient import TestClient

    from clawkeeper_core.guards.script_body_scan import script_body_scan_guard
    from clawkeeper_core.watcher.daemon import build_app
    from tests.test_learned_rules import candidate

    compat = candidate(target="script_body_scan")
    old_id = rules.rule_id(compat.guard_target, compat.pattern)
    migrated = candidate(target=compat.guard_target, subject="program.name", legacy_ids=[old_id])
    assert isolated_learning.add(compat) and isolated_learning.add(migrated)
    assert isolated_learning.add(candidate())
    with TestClient(build_app(watcher=Mock())) as client:
        listed = client.get("/watcher/learned").json()
        ids = {item["id"] for item in listed}
        raw_hash = hashlib.sha256(compat.pattern.encode()).hexdigest()
        for ambiguous in (raw_hash, old_id):
            assert client.delete(f"/watcher/learned/{ambiguous}").status_code == 409
        assert {r.id for r in rules.current_snapshot().rules} == ids
        for expected in (True, False):
            result = client.delete(f"/watcher/learned/{old_id}?exact=true")
            assert result.status_code == 200 and result.json()["removed"] is expected
        assert {r.id for r in rules.current_snapshot().rules} == ids - {old_id}
        assert {item["id"] for item in client.get("/watcher/learned").json()} == ids - {old_id}
        migrated_id = rules.rule_id(migrated.guard_target, migrated.pattern, migrated.subject)
        event = {"toolName": "terminal", "params": {"command": migrated.source_command}}
        assert script_body_scan_guard(event)["matched"] == migrated_id
        assert client.delete(f"/watcher/learned/{migrated_id}?exact=true").json()["removed"]
        assert script_body_scan_guard(event) is None
        builtin_event = {"toolName": "terminal", "params": {"command": "cat /tmp/.ssh/example"}}
        assert script_body_scan_guard(builtin_event)["source"] == "script_body_scan:path_access"


def test_delete_reports_failures_and_retry_reconciles_runtime(isolated_learning, monkeypatch):
    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher import daemon, learner
    from tests.test_learned_rules import candidate

    assert isolated_learning.add(candidate())
    with TestClient(daemon.build_app(watcher=Mock())) as client:
        previous = rules.current_snapshot()
        url = f"/watcher/learned/{previous.rules[0].id}?exact=true"
        with monkeypatch.context() as patch:
            patch.setattr(isolated_learning, "_save", Mock(side_effect=OSError()))
            failure = client.delete(url)
        assert failure.status_code == 503 and failure.json() == {
            "removed": False, "watcher_runtime": "retained", "version": previous.version,
            "error": "learned_store_write_failed",
        }
        assert isolated_learning.all() and rules.current_snapshot() is previous

        with monkeypatch.context() as patch:
            patch.setattr(isolated_learning, "all", Mock(side_effect=learner.StoreUnavailableError()))
            failure = client.delete(url)
        assert failure.status_code == 503 and failure.json() == {
            "removed": True, "watcher_runtime": "retained", "version": previous.version,
            "error": "learned_store_unavailable",
        }
        assert isolated_learning.all() == [] and rules.current_snapshot() is previous
        assert client.get("/watcher/rules").status_code == 503
        retry = client.delete(url)
        assert retry.status_code == 200 and retry.json() == {
            "removed": False, "watcher_runtime": "synchronized",
            "version": rules.make_snapshot([]).version, "error": None,
        }
        assert client.get("/watcher/rules").json()["rules"] == []


def test_delete_from_restored_v3_store_before_first_list(isolated_learning):
    from dataclasses import asdict

    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher.daemon import build_app
    from tests.test_learned_rules import candidate

    compat = candidate()
    v3 = candidate(source_texts=["CK_CANARY_8421"], source_view_version=3)
    invocation_id = rules.rule_id(compat.guard_target, compat.pattern)
    migrated_id = rules.rule_id(v3.guard_target, v3.pattern, "program.name")
    with TestClient(build_app(watcher=Mock())) as client:
        restored = json.dumps([asdict(compat), asdict(v3)])
        isolated_learning._path.write_text(restored)
        assert client.delete(f"/watcher/learned/{invocation_id}").status_code == 409
        assert isolated_learning._path.read_text() == restored
        response = client.delete(f"/watcher/learned/{invocation_id}?exact=true")
        assert response.status_code == 200 and response.json()["removed"]
        assert [r.id for r in rules.current_snapshot().rules] == [migrated_id]
        isolated_learning._path.write_text(restored)
        response = client.delete(f"/watcher/learned/{migrated_id}?exact=true")
        assert response.status_code == 200 and response.json()["removed"]
        assert [r.id for r in rules.current_snapshot().rules] == [invocation_id]


def test_invalid_historical_rows_do_not_prevent_targeted_deletion(isolated_learning):
    from dataclasses import asdict

    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher.daemon import build_app
    from tests.test_learned_rules import candidate

    good = candidate()
    records = [asdict(good), dict(asdict(good), pattern=chr(0xd800)),
               dict(asdict(good), pattern=17), dict(asdict(good), confidence=float("nan"), pattern="CK_NAN"),
               dict(asdict(good), confidence=10**400, pattern="CK_RANGE")]
    isolated_learning._path.write_text(json.dumps(records))
    with TestClient(build_app(watcher=Mock())) as client:
        listed = client.get("/watcher/learned")
        assert listed.status_code == 200 and len(listed.json()) == len(records)
        for row in listed.json():
            assert row["id"]
            response = client.delete("/watcher/learned/" + row["id"], params={"exact": True})
            assert response.status_code == 200 and response.json()["removed"]
        assert not client.get("/watcher/learned").json()
        assert not rules.current_snapshot().rules


def test_embedded_separator_preserves_both_rule_identities(isolated_learning):
    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher.daemon import build_app
    from tests.test_learned_rules import candidate

    typed = candidate(pattern="CK_MARKER_A|CK_MARKER_B", command="CK_MARKER_B", subject="program.name")
    invocation = candidate(pattern="program.name\0" + typed.pattern, command="CK_MARKER_B")
    assert isolated_learning.add(typed) and isolated_learning.add(invocation)
    historical = rules.rule_id(typed.guard_target, typed.pattern, typed.subject)
    assert historical == rules.rule_id(invocation.guard_target, invocation.pattern, legacy=True)
    with TestClient(build_app(watcher=Mock())) as client:
        listed = {row["subject"]: row["id"] for row in client.get("/watcher/learned").json()}
        assert listed["invocation"] != listed["program.name"] == historical
        assert len(rules.snapshot_from_wire(client.get("/watcher/rules").json()).rules) == 2
        assert client.delete(f"/watcher/learned/{historical}").status_code == 409
        assert client.delete(f"/watcher/learned/{historical}?exact=true").json()["removed"]
        assert [r.id for r in rules.current_snapshot().rules] == [listed["invocation"]]
        assert client.delete(f"/watcher/learned/{historical}").json()["removed"]
        assert not rules.current_snapshot().rules


@pytest.mark.parametrize("failure", ["store_unavailable", "record_rejected"])
def test_failed_activation_does_not_announce_relearned_rule(isolated_learning, monkeypatch, failure):
    from clawkeeper_core.watcher import agent, daemon, learner, reload
    from tests.test_learned_rules import candidate

    rule = candidate()
    assert reload.store_learned_pattern(rule)
    deleted, _, error = reload.remove_learned_pattern(rules.current_snapshot().rules[0].id, exact=True)
    assert deleted and error is None
    broadcast, push = Mock(), Mock()
    monkeypatch.setattr(agent, "synthesize_pattern", Mock(return_value=rule))
    monkeypatch.setattr(agent, "maybe_push_upstream", push)
    monkeypatch.setattr(daemon, "_broadcast", broadcast)
    with monkeypatch.context() as patch:
        if failure == "store_unavailable":
            patch.setattr(isolated_learning, "all", Mock(side_effect=learner.StoreUnavailableError()))
        else:
            patch.setattr(reload, "validate_candidate", Mock(side_effect=ValueError("record rejected")))
        agent._maybe_learn("terminal", {}, {}, Mock())
    assert isolated_learning.all() and not rules.current_snapshot().rules
    assert bool(rules.policy_error()) == (failure == "store_unavailable")
    broadcast.assert_not_called()
    push.assert_not_called()
    assert reload.apply_learned_patterns() == 1 and rules.policy_error() is None


def test_noop_delete_preserves_healthy_snapshot_but_reconciles_stale_rule(isolated_learning, monkeypatch):
    from fastapi.testclient import TestClient

    from clawkeeper_core.watcher import daemon, reload
    from tests.test_learned_rules import candidate

    assert isolated_learning.add(candidate())
    with TestClient(daemon.build_app(watcher=Mock())) as client:
        previous = rules.current_snapshot()
        with monkeypatch.context() as patch:
            patch.setattr(reload, "validate_candidate", Mock(side_effect=AssertionError("unexpected revalidation")))
            for exact in (False, True):
                reply = client.delete("/watcher/learned/missing", params={"exact": exact})
                assert reply.status_code == 200 and not reply.json()["removed"]
            assert rules.current_snapshot() is previous
        # Another tool restored an empty store while the daemon was running.
        isolated_learning._path.write_text("[]")
        reply = client.delete("/watcher/learned/" + previous.rules[0].id, params={"exact": True})
        assert reply.status_code == 200 and reply.json()["version"] == rules.make_snapshot([]).version


def test_concurrent_relearning_waits_for_delete_snapshot(isolated_learning, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from clawkeeper_core.watcher import reload
    from tests.test_learned_rules import candidate

    rule = candidate()
    assert reload.store_learned_pattern(rule)
    original = rules.current_snapshot()
    remove = isolated_learning.remove
    removed_from_store, add_started = Event(), Event()

    def wait_for_add(*args, **kwargs):
        removed = remove(*args, **kwargs)
        removed_from_store.set()
        assert add_started.wait(timeout=5)
        return removed

    def relearn():
        assert removed_from_store.wait(timeout=5)
        add_started.set()
        return reload.store_learned_pattern(rule)

    monkeypatch.setattr(isolated_learning, "remove", wait_for_add)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(reload.remove_learned_pattern, original.rules[0].id, exact=True)
        addition = pool.submit(relearn)
        assert deletion.result(timeout=10) == (True, rules.make_snapshot([]).version, None)
        assert addition.result(timeout=10)
    assert rules.current_snapshot() == original
    assert [c.pattern for c in isolated_learning.all()] == [rule.pattern]


def start_server(store_path, port=0):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    port = listener.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.watcher_process",
            "server",
            str(store_path),
            str(listener.fileno()),
        ],
        pass_fds=(listener.fileno(),),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    listener.close()
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(proc.stderr.read())
        try:
            if httpx.get(url + "/watcher/health", timeout=0.3).status_code == 200:
                return proc, url, port
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=5)
    raise AssertionError("Watcher did not become ready")


def stop(proc):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream:
                stream.close()


def ask_adapter(proc, action, command="CK_CANARY_8421"):
    proc.stdin.write(json.dumps({"action": action, "command": command}) + "\n")
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], 15)
    assert ready, "Adapter did not respond"
    line = proc.stdout.readline()
    assert line, proc.stderr.read()
    return json.loads(line)


@pytest.mark.parametrize("legacy_record", [False, True])
@pytest.mark.parametrize("command", ["CK_CANARY_8421", "python -m CK_CANARY_8421", "CK_FIELD_8421", "python -m CK_FIELD_8421"])
def test_separate_process_learning_sync_restart_delete_and_disconnect(tmp_path, legacy_record, command):  # noqa: PLR0915 -- complete lifecycle across real processes
    server = adapter = None
    try:
        if legacy_record:
            from dataclasses import asdict

            from tests.test_learned_rules import candidate

            pattern = r"python\s+-m\s+CK_CANARY_\d+" if command.startswith("python") else r"CK_CANARY_\d+"
            record = asdict(candidate(pattern=pattern, command="", source_texts=["echo " + command]))
            record.pop("source_view_version")
            (tmp_path / "learned.json").write_text(json.dumps([record]))
        server, url, port = start_server(tmp_path / "learned.json")
        adapter = subprocess.Popen(
            [sys.executable, "-m", "tests.watcher_process", "adapter", url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with httpx.Client(base_url=url, timeout=3) as client:
            assert ask_adapter(adapter, "pre", "ls -la")["result"] is None
            empty_version = client.get("/watcher/rules").json()["version"]
            result = client.post(
                "/watcher/evaluate",
                json={
                    "session_id": "learn",
                    "tool_name": "terminal",
                    "args": {"command": command},
                },
            )
            assert result.status_code == 200 and result.json()["decision"] == "deny"
            assert result.json()["deterministic_findings"] == []
            snapshot = client.get("/watcher/rules").json()
            assert snapshot["pid"] == server.pid and snapshot["pid"] != adapter.pid
            assert snapshot["version"] != empty_version and len(snapshot["rules"]) == 1
            if "CK_FIELD" in command:
                assert snapshot["rules"][0]["subject"] == ("python.module" if command.startswith("python") else "program.name")
            synced = ask_adapter(adapter, "pre", command)
            assert "learned:exec_gate" in synced["result"]["message"]
            assert synced["version"] == snapshot["version"]
            assert ask_adapter(adapter, "approval", command)["result"] == "deny"
            assert not ask_adapter(adapter, "inspect", "echo " + command)["result"]["block"]
            old_instance = snapshot["instance_id"]
        stop(server)
        server, url, _ = start_server(tmp_path / "learned.json", port)
        with httpx.Client(base_url=url, timeout=3) as client:
            restored = client.get("/watcher/rules").json()
            assert restored["version"] == snapshot["version"]
            assert restored["instance_id"] != old_instance
            assert "learned:exec_gate" in ask_adapter(adapter, "pre", command)["result"]["message"]
            assert not ask_adapter(adapter, "inspect", "echo " + command)["result"]["block"]
            deleted = client.delete("/watcher/learned/" + snapshot["rules"][0]["id"]).json()
            assert deleted["removed"] and deleted["version"] == empty_version
            refreshed = ask_adapter(adapter, "pre", "ls -la")
            assert refreshed["version"] == empty_version and refreshed["result"] is None
            assert not ask_adapter(adapter, "inspect", command)["result"]["block"]
        stop(server)
        server, url, _ = start_server(tmp_path / "learned.json", port)
        with httpx.Client(base_url=url, timeout=3) as client:
            assert client.get("/watcher/rules").json()["rules"] == []
            # Restore a last-known rule, then disconnect. Failed synchronization
            # must retain that rule and stop an otherwise benign tool call.
            client.post(
                "/watcher/evaluate",
                json={
                    "session_id": "learn-again",
                    "tool_name": "terminal",
                    "args": {"command": command},
                },
            ).raise_for_status()
            assert "learned:exec_gate" in ask_adapter(adapter, "pre", command)["result"]["message"]
        stop(server)
        server = None
        assert ask_adapter(adapter, "pre", "ls -la")["result"]["action"] == "block"
        assert ask_adapter(adapter, "approval", "ls -la")["result"] == "deny"
        assert ask_adapter(adapter, "inspect", command)["result"]["block"]
    finally:
        stop(adapter)
        stop(server)
