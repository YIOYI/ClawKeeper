"""Replay Hermes observation callbacks through the installed adapter."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clawkeeper_core.adapters.hermes import install
from clawkeeper_core.judge import Judge
from clawkeeper_core.profile import AgentProfiler
from tests.watcher_process import fake_hermes_modules


@pytest.fixture
def attach(monkeypatch):
    fake_hermes_modules(lambda name, module: monkeypatch.setitem(sys.modules, name, module))

    def attach_agent(judge, session, *, agent_id=None, start=None, complete=None):
        agent = SimpleNamespace(
            session_id=session, agent_id=agent_id, tool_start_callback=start, tool_complete_callback=complete,
        )
        install(judge, agent, pre_guards=[], post_guards=[])
        return agent

    return attach_agent


def test_profiler_aggregates_hermes_sessions_and_counts_only_starts(attach):
    judge = Judge()
    judge.profiler = AgentProfiler()
    first = attach(judge, "one", agent_id="worker")
    second = attach(judge, "two", agent_id="worker")
    other = attach(judge, "three")
    first.tool_start_callback("call-1", "read", {})
    first.tool_complete_callback("call-1", "read", {}, "ok")
    second.tool_start_callback("write", {})
    second.tool_complete_callback("write", {}, "ok")
    other.tool_start_callback("list", {})

    baseline = judge.profiler.baseline("worker")
    assert (baseline["dataPointCount"], baseline["toolCallCount"], baseline["sessionCount"]) == (4, 2, 2)
    assert baseline["toolDistribution"] == {"read": 0.5, "write": 0.5}
    assert judge.profiler.baseline("three")["knownTools"] == {"list"}


def test_failing_profiler_preserves_original_callbacks(attach, caplog):
    judge = Judge()
    judge.profiler = Mock()
    judge.profiler.update.side_effect = RuntimeError("observer unavailable")
    start, complete = Mock(), Mock()
    agent = attach(judge, "one", start=start, complete=complete)
    agent.tool_start_callback("read", {})
    agent.tool_complete_callback("read", {}, "ok")
    start.assert_called_once_with("read", {})
    complete.assert_called_once_with("read", {}, "ok")
    assert "Profiler failed to observe Hermes tool_start" in caplog.text
    assert "Profiler failed to observe Hermes tool_complete" in caplog.text


@pytest.mark.parametrize("interposed", [False, True])
def test_reinstall_replaces_observers_and_preserves_host_chain(attach, interposed):
    first, latest = Judge(), Judge()
    first.profiler, latest.profiler = AgentProfiler(), AgentProfiler()
    start, complete = Mock(), Mock()
    agent = attach(first, "one", start=start, complete=complete)
    if interposed:
        host_wrapper = Mock(wraps=agent.tool_start_callback)
        agent.tool_start_callback = host_wrapper
    install(latest, agent, pre_guards=[], post_guards=[])
    install(latest, agent, pre_guards=[], post_guards=[])
    agent.session_id = "two"
    agent.tool_start_callback("read", {})
    agent.tool_complete_callback("read", {}, "ok")
    profile = latest.profiler.baseline("two")
    assert (profile["toolCallCount"], profile["dataPointCount"]) == (1, 2)
    assert first.profiler.baseline("two") is None
    start.assert_called_once_with("read", {})
    complete.assert_called_once_with("read", {}, "ok")
    if interposed:
        host_wrapper.assert_called_once_with("read", {})
