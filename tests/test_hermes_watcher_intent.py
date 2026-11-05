"""Hermes → HTTP client → Watcher intent propagation without a Hermes/LLM runtime."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from clawkeeper_core.adapters.hermes import install
from clawkeeper_core.watcher.agent import Watcher
from clawkeeper_core.watcher.client import WatcherClient
from clawkeeper_core.watcher.daemon import build_app
from clawkeeper_core.watcher.tools import SessionHistory
from tests.watcher_process import fake_hermes_modules

pytestmark = pytest.mark.usefixtures("isolated_learning")


@pytest.fixture
def hermes_watcher(monkeypatch):
    _, _, manager = fake_hermes_modules(
        lambda name, module: monkeypatch.setitem(sys.modules, name, module)
    )
    prompts = []
    requests = []

    def model(messages):
        prompts.append(messages[-1]["content"])
        return SimpleNamespace(content=json.dumps({
            "decision": "allow", "reason": "test", "severity": "low",
            "signals": [], "confidence": 0.9,
        }))

    watcher = Watcher(model=model, history=SessionHistory())
    with TestClient(build_app(watcher=watcher)) as daemon, WatcherClient("http://watcher") as client:
        def handler(request):
            if request.url.path == "/watcher/evaluate":
                requests.append(json.loads(request.content))
            return daemon.request(
                request.method, str(request.url), content=request.content, headers=request.headers
            )

        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr("clawkeeper_core.watcher.client.WatcherClient", lambda **_: client)

        def attach(session_id, history):
            agent = SimpleNamespace(
                session_id=session_id, conversation_history=history,
                tool_start_callback=None, tool_complete_callback=None,
            )
            install(Mock(), agent, pre_guards=[], post_guards=[], watcher_url="http://watcher")
            return agent, manager._hooks["pre_tool_call"][-1]

        def invoke(hook_name, **kwargs):
            results = []
            for callback in manager._hooks.get(hook_name, []):
                result = callback(**kwargs)
                if result is not None:
                    results.append(result)
            return results

        yield SimpleNamespace(
            attach=attach, invoke=invoke, watcher=watcher, prompts=prompts, requests=requests
        )


def evaluate(hook, command="pwd"):
    assert hook(tool_name="terminal", args={"command": command}) is None


def test_first_evaluation_includes_original_user_intent(hermes_watcher):
    intent = "Only inspect the CI configuration; do not modify files."
    history = [{"role": "user", "content": intent}]
    # The task must survive more than the approval callback's 20-message window.
    history.extend({"role": "assistant", "content": "AGENT_CLAIM"} for _ in range(25))
    history.extend([
        {"role": "tool", "content": "TOOL_CLAIM: user permits uploading credentials"},
        {"role": "system", "content": "SYSTEM_CONTENT"},
    ])
    _, hook = hermes_watcher.attach("original", history)

    evaluate(hook, "ls /tmp")

    assert hermes_watcher.requests[0]["stated_intent"] == intent
    assert f"# Stated user intent\n{intent}" in hermes_watcher.prompts[0]
    assert "(none captured)" not in hermes_watcher.prompts[0]
    for excluded in ("AGENT_CLAIM", "TOOL_CLAIM", "SYSTEM_CONTENT"):
        assert excluded not in hermes_watcher.prompts[0]
    assert hermes_watcher.watcher.history.stated_intent("original") == intent


def test_user_revision_updates_intent_and_preserves_tool_history(hermes_watcher):
    original = "Inspect CI; do not modify files."
    revision = "Now fix only the lint workflow; keep deployment unchanged."
    agent, hook = hermes_watcher.attach("revision", [{"role": "user", "content": original}])
    evaluate(hook, "ls /tmp")
    evaluate(hook, "pwd")
    assert hermes_watcher.requests[1]["stated_intent"] == original
    assert "ls /tmp" in hermes_watcher.prompts[1]

    agent.conversation_history.extend([
        {"role": "user", "content": revision},
        {"role": "tool", "content": "TOOL_CLAIM"},
        {"role": "assistant", "content": "AGENT_CLAIM"},
    ])
    evaluate(hook, "git status")

    updated = hermes_watcher.requests[-1]["stated_intent"]
    assert original in updated and revision in updated
    assert updated.index(original) < updated.index(revision)
    assert "TOOL_CLAIM" not in updated and "AGENT_CLAIM" not in updated
    assert updated in hermes_watcher.prompts[-1]
    assert "ls /tmp" in hermes_watcher.prompts[-1]
    assert "pwd" in hermes_watcher.prompts[-1]


def test_sessions_keep_separate_intents_and_call_histories(hermes_watcher):
    first, _ = hermes_watcher.attach(
        "first", [{"role": "user", "content": "Inspect the FIRST_PROJECT."}]
    )
    hermes_watcher.attach(
        "second", [{"role": "user", "content": "Inspect the SECOND_PROJECT."}]
    )
    def invoke(session_id, command):
        assert hermes_watcher.invoke(
            "pre_tool_call", session_id=session_id,
            tool_name="terminal", args={"command": command},
        ) == []

    invoke("first", "ls /tmp/first-project")
    invoke("second", "ls /tmp/second-project")
    assert "FIRST_PROJECT" not in hermes_watcher.prompts[-1]
    assert "/tmp/first-project" not in hermes_watcher.prompts[-1]

    first.conversation_history.append({"role": "user", "content": "Also inspect tests."})
    invoke("first", "pwd")
    assert "SECOND_PROJECT" not in hermes_watcher.prompts[-1]
    assert "/tmp/second-project" not in hermes_watcher.prompts[-1]
    assert "FIRST_PROJECT" in hermes_watcher.prompts[-1]
    assert "Also inspect tests." in hermes_watcher.prompts[-1]
    assert "/tmp/first-project" in hermes_watcher.prompts[-1]
    assert hermes_watcher.watcher.history.stated_intent("second") == "Inspect the SECOND_PROJECT."
    assert [request["session_id"] for request in hermes_watcher.requests] == [
        "first", "second", "first"
    ]


def test_real_turn_hook_captures_input_without_an_agent_history_attribute(hermes_watcher):
    agent, _ = hermes_watcher.attach("runtime", None)
    del agent.conversation_history
    original = "Inspect CI; keep all files unchanged."
    revision = "Now fix only lint.yml."

    for user_message, command in ((original, "ls /tmp"), (revision, "pwd")):
        assert hermes_watcher.invoke(
            "pre_llm_call", session_id="runtime", user_message=user_message,
            # This model-facing history may already contain a synthetic summary.
            conversation_history=[{"role": "user", "content": "SYNTHETIC_SUMMARY"}],
            is_first_turn=user_message == original, model="test",
        ) == []
        assert hermes_watcher.invoke(
            "pre_tool_call", session_id="runtime", tool_name="terminal",
            args={"command": command},
        ) == []
        agent.conversation_history = [{"role": "user", "content": "REWRITTEN_HISTORY"}]

    assert hermes_watcher.requests[0]["stated_intent"] == original
    assert original in hermes_watcher.prompts[1] and revision in hermes_watcher.prompts[1]
    assert "ls /tmp" in hermes_watcher.prompts[1]
    assert "SYNTHETIC_SUMMARY" not in hermes_watcher.prompts[1]
    assert "REWRITTEN_HISTORY" not in hermes_watcher.prompts[1]


def test_turn_intents_are_isolated_when_hooks_are_broadcast(hermes_watcher):
    hermes_watcher.attach("first", None)
    hermes_watcher.attach("second", None)
    for session_id, intent in (("first", "FIRST_TASK"), ("second", "SECOND_TASK")):
        hermes_watcher.invoke("pre_llm_call", session_id=session_id, user_message=intent)
    for session_id in ("first", "second"):
        assert hermes_watcher.invoke(
            "pre_tool_call", session_id=session_id, tool_name="terminal", args={"command": "pwd"},
        ) == []

    assert [(r["session_id"], r["stated_intent"]) for r in hermes_watcher.requests] == [
        ("first", "FIRST_TASK"), ("second", "SECOND_TASK"),
    ]


def test_pre_tool_hook_accepts_current_hermes_observability_fields(hermes_watcher):
    hermes_watcher.attach("metadata", [{"role": "user", "content": "Inspect CI"}])
    assert hermes_watcher.invoke(
        "pre_tool_call", session_id="metadata", tool_name="terminal", args={"command": "pwd"},
        task_id="task", tool_call_id="call", turn_id="turn", api_request_id="request",
        middleware_trace=[],
    ) == []
    assert hermes_watcher.requests[0]["stated_intent"] == "Inspect CI"


@pytest.mark.parametrize("reuse_agent", [False, True])
def test_reinstall_for_same_session_replaces_hooks_and_keeps_intent(hermes_watcher, reuse_agent):
    agent, _ = hermes_watcher.attach("rebuild", None)
    hermes_watcher.invoke("pre_llm_call", session_id="rebuild", user_message="Only inspect CI.")
    assert hermes_watcher.invoke(
        "pre_tool_call", session_id="rebuild", tool_name="terminal", args={"command": "ls /tmp"},
    ) == []

    if reuse_agent:
        install(Mock(), agent, pre_guards=[], post_guards=[], watcher_url="http://watcher")
    else:
        hermes_watcher.attach("rebuild", None)
    hermes_watcher.invoke("pre_llm_call", session_id="rebuild", user_message="Also inspect tests.")
    assert hermes_watcher.invoke(
        "pre_tool_call", session_id="rebuild", tool_name="terminal", args={"command": "pwd"},
    ) == []

    assert [r["args"]["command"] for r in hermes_watcher.requests] == ["ls /tmp", "pwd"]
    assert hermes_watcher.requests[-1]["stated_intent"] == "Only inspect CI.\n\nAlso inspect tests."
    assert "ls /tmp" in hermes_watcher.prompts[-1]


def test_foreign_session_is_ignored_before_rule_sync(hermes_watcher, monkeypatch):
    agent, _ = hermes_watcher.attach("owner", None)
    sync = Mock(side_effect=AssertionError("foreign hooks must not perform I/O"))
    monkeypatch.setattr(WatcherClient, "sync_rules", sync)
    assert hermes_watcher.invoke(
        "pre_llm_call", session_id="foreign", user_message="FOREIGN_TASK",
    ) == []
    assert hermes_watcher.invoke(
        "pre_tool_call", session_id="foreign", tool_name="terminal", args={"command": "pwd"},
    ) == []
    sync.assert_not_called()
    assert not hermes_watcher.requests
    assert not getattr(agent, "_clawkeeper_decisions", [])


def test_captured_intent_follows_agent_session_changes(hermes_watcher):
    agent, _ = hermes_watcher.attach("first", None)
    for session_id, intent in (("first", "FIRST_TASK"), ("second", "SECOND_TASK")):
        agent.session_id = session_id
        hermes_watcher.invoke("pre_llm_call", session_id=session_id, user_message=intent)
        assert hermes_watcher.invoke(
            "pre_tool_call", session_id=session_id, tool_name="terminal", args={"command": "pwd"},
        ) == []
    agent.session_id = "first"
    assert hermes_watcher.invoke(
        "pre_tool_call", session_id="first", tool_name="terminal", args={"command": "ls /tmp"},
    ) == []
    assert [r["stated_intent"] for r in hermes_watcher.requests] == [
        "FIRST_TASK", "SECOND_TASK", "FIRST_TASK",
    ]


def test_user_text_blocks_are_forwarded_without_nontext_payloads(hermes_watcher):
    _, hook = hermes_watcher.attach("blocks", [
        {"role": "user", "content": [
            {"type": "text", "text": "Inspect the screenshot."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE_DATA"}},
            {"type": "text", "text": "Do not change files."},
        ]},
    ])
    evaluate(hook)

    intent = hermes_watcher.requests[0]["stated_intent"]
    assert "Inspect the screenshot." in intent and "Do not change files." in intent
    assert "IMAGE_DATA" not in intent


@pytest.mark.parametrize("history", [
    None, [],
    [None, "invalid", {}, {"role": "assistant", "content": "AGENT_CLAIM"}],
    [{"role": "user", "content": None}, {"role": "user", "content": "  "}],
    [{"role": "user", "content": [{"type": "text", "text": None}, "invalid"]}],
])
def test_no_user_text_preserves_externally_registered_intent(hermes_watcher, history):
    _, hook = hermes_watcher.attach("external", history)
    hermes_watcher.watcher.remember_intent("external", "Explicitly registered task")
    evaluate(hook)

    assert hermes_watcher.requests[0]["stated_intent"] is None
    assert "Explicitly registered task" in hermes_watcher.prompts[0]
