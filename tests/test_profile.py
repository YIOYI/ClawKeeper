"""Behavior-parity tests ported from
legacy/clawkeeper-watcher/plugins/clawkeeper-watcher/src/core/agent-profiler.test.js.
"""

from __future__ import annotations

import time

from clawkeeper_core.profile import (
    AgentProfiler,
    _cache,
    build_agent_profiles,
    compute_deviation,
    detect_anomalies,
    invalidate_profile_cache,
)
from clawkeeper_core.schemas import AgentEvent, Decision, Outcome, Risk


def make_event(**overrides):
    base = {
        "timestamp": "2026-03-24T10:00:00.000Z",
        "type": "before_tool_call",
        "toolName": "read",
        "agentId": "agent-abc123",
        "sessionKey": "agent:agent-abc123:main",
    }
    base.update(overrides)
    return base


def make_decision(**overrides):
    base = {
        "timestamp": "2026-03-24T10:10:00.000Z",
        "sessionKey": "agent:agent-abc123:main",
        "decision": "ask_user",
        "riskLevel": "high",
    }
    base.update(overrides)
    return base


class TestBuildAgentProfiles:
    def test_aggregates_event_logs_per_agent(self):
        records = [
            make_event(toolName="read"),
            make_event(toolName="write"),
            make_event(
                type="llm_output", toolName=None,
                inputTokens=300, outputTokens=150, totalTokens=450,
            ),
            make_event(
                agentId="agent-def456",
                sessionKey="agent:agent-def456:main",
                toolName="glob",
            ),
        ]
        decisions = [
            make_decision(decision="ask_user"),
            make_decision(decision="continue"),
        ]
        profiles = build_agent_profiles(records, decision_records=decisions)
        p = profiles["agent-abc123"]
        assert p["toolCallCount"] == 2
        assert p["sessionCount"] == 1
        assert p["avgToolCallsPerSession"] == 2
        assert p["avgInputTokensPerCall"] == 300
        assert p["avgOutputTokensPerCall"] == 150
        assert p["avgTotalTokensPerCall"] == 450
        assert p["totalTokens"] == 450
        assert p["riskDecisionCount"] == 1
        assert p["judgeDecisionCount"] == 2
        assert p["riskRatio"] == 0.5
        assert sorted(p["knownTools"]) == ["read", "write"]
        assert p["toolDistribution"]["read"] == 0.5
        assert p["toolDistribution"]["write"] == 0.5


def test_streaming_baseline_matches_batch_and_can_be_replayed_after_restart():
    records = [
        make_event(toolName="Read"),
        make_event(type="after_tool_call"),
        make_event(type="llm_output", inputTokens=12, outputTokens=4, cacheReadTokens=2),
        make_event(sessionKey="agent:agent-abc123:second", toolName="write"),
    ]
    decisions = [make_decision(), make_decision(decision="continue")]
    expected = build_agent_profiles(records, decision_records=decisions)["agent-abc123"]
    profiler = AgentProfiler()
    for record in [*records, *decisions]:
        profiler.update(record)
    snapshot = profiler.baseline(" AGENT-ABC123 ")
    assert snapshot == expected
    snapshot["knownTools"].clear()
    snapshot["toolDistribution"].clear()
    assert profiler.baseline("agent-abc123") == expected

    restarted = AgentProfiler()
    restarted.update({})
    assert restarted.baseline("agent-abc123") is None
    assert not restarted.deviation("agent-abc123", records)["hasBaseline"]
    for record in [*records, *decisions]:
        restarted.update(record)
    assert restarted.baseline("agent-abc123") == expected


def test_streaming_event_window_deviation_leaves_baseline_intact():
    def event(kind, payload, agent="worker"):
        return AgentEvent(ts="2026-03-24T10:00:00Z", agent_id=agent, session_id="one",
                          kind=kind, payload=payload)

    profiler = AgentProfiler()
    profiler.update(event("tool_start", {"name": "read"}))
    profiler.update(event("llm_output", {"inputTokens": 10}).model_dump(mode="json"))
    profiler.update(event("decision", {"decision": "ask_user"}))
    baseline = profiler.baseline("worker")
    assert baseline["dataPointCount"] == 2
    assert baseline["riskRatio"] == 1
    assert baseline["judgeDecisionCount"] == 1
    recent = [
        event("tool_start", {"name": "write"}),
        event("tool_start", {"name": "list"}, agent="other"),
        event("llm_output", {"inputTokens": 50}),
    ]
    diff = profiler.deviation("worker", iter(recent))
    assert diff["novelTools"] == ["write"]
    assert diff["tokenDeviation"]["input"]["multiplier"] == 5
    assert detect_anomalies(diff)["detected"]
    assert profiler.deviation("worker", baseline)["novelTools"] == []
    assert profiler.baseline("worker") == baseline


def test_replay_orders_mixed_timestamp_formats_by_instant():
    profiler = AgentProfiler()
    profiler.update(AgentEvent(ts="2026-03-24T09:00:00+08:00", agent_id="worker", kind="message"))
    later = "2026-03-24T02:00:00Z"
    profiler.update(make_event(timestamp=later, agentId="worker"))
    profile = profiler.baseline("worker")
    assert profile["firstSeen"] == "2026-03-24T09:00:00+08:00"
    assert profile["lastSeen"] == later


def test_foreign_summary_cannot_be_attributed_to_requested_agent():
    profiler = AgentProfiler()
    profiler.update(make_event(agentId="worker"))
    profiler.update(make_event(agentId="other", toolName="list"))
    foreign = profiler.baseline("other")
    result = profiler.deviation("worker", foreign)
    assert result["current"]["toolCallCount"] == 0
    assert result["toolDistributionDivergence"] == 0
    assert not detect_anomalies(result)["detected"]


def test_native_decision_payload_updates_risk_counts():
    profiler = AgentProfiler()
    for outcome in (Outcome.ALLOW, Outcome.ASK, Outcome.DENY):
        decision = Decision(outcome=outcome, risk=Risk.LOW, reason="test")
        profiler.update(AgentEvent(ts="2026-03-24T00:00:00Z", agent_id="worker",
                                   kind="decision", payload=decision.model_dump(mode="json")))
    baseline = profiler.baseline("worker")
    assert (baseline["riskDecisionCount"], baseline["judgeDecisionCount"]) == (2, 3)


def test_single_recent_event_matches_window_and_reports_observed_baseline():
    profiler = AgentProfiler()
    profiler.update(make_event(agentId="worker"))
    event = AgentEvent(ts="2026-04-24T00:00:00Z", agent_id="worker", kind="tool_start",
                       payload={"name": "list"})
    expected = profiler.deviation("worker", [event])
    assert profiler.deviation("worker", event.model_dump(mode="json")) == expected
    assert profiler.deviation("worker", event) == expected
    assert profiler.deviation("worker", make_event(agentId="worker", toolName="list"))["novelTools"] == ["list"]
    report = detect_anomalies(expected)
    assert report["detected"]
    assert report["baselineSummary"]["lookbackDays"] is None
    assert "7-day" not in report["warning"]
    assert "7-day" not in report["deviations"][0]["message"]


def test_unrepresentable_token_sample_keeps_valid_metrics_and_completes_update():
    profiler = AgentProfiler()
    profiler.update(make_event(type="llm_output", inputTokens=2, outputTokens=10**400))
    profile = profiler.baseline("agent-abc123")
    assert (profile["dataPointCount"], profile["avgInputTokensPerCall"], profile["totalTokens"]) == (1, 2, 2)
    assert profile["avgOutputTokensPerCall"] == 0


class TestComputeDeviation:
    def test_detects_tool_distribution_shift(self):
        baseline = {
            "agentId": "agent-abc123",
            "toolDistribution": {"bash": 0.05, "read": 0.95},
            "toolCallCount": 100,
            "knownTools": {"bash", "read"},
        }
        current = {
            "agentId": "agent-abc123",
            "toolDistribution": {"bash": 0.6, "read": 0.4},
            "toolCallCount": 5,
            "knownTools": {"bash", "read"},
        }
        deviations = compute_deviation(current, baseline)
        bash_change = next(c for c in deviations["toolFrequencyChanges"] if c["tool"] == "bash")
        assert round(bash_change["multiplier"] * 10) / 10 == 12.0
        assert deviations["toolDistributionDivergence"] > 0

    def test_detects_token_spike(self):
        baseline = {
            "agentId": "agent-abc123",
            "toolDistribution": {"read": 1},
            "toolCallCount": 10,
            "avgInputTokensPerCall": 100,
            "avgOutputTokensPerCall": 80,
            "avgTotalTokensPerCall": 180,
            "knownTools": {"read"},
        }
        current = {
            "agentId": "agent-abc123",
            "toolDistribution": {"read": 1},
            "toolCallCount": 1,
            "avgInputTokensPerCall": 320,
            "avgOutputTokensPerCall": 240,
            "avgTotalTokensPerCall": 560,
            "knownTools": {"read"},
        }
        deviations = compute_deviation(current, baseline)
        assert round(deviations["tokenDeviation"]["input"]["multiplier"] * 10) / 10 == 3.2
        assert round(deviations["tokenDeviation"]["output"]["multiplier"] * 10) / 10 == 3.0
        assert round(deviations["tokenDeviation"]["total"]["multiplier"] * 10) / 10 == 3.1

    def test_detects_novel_tools(self):
        baseline = {
            "agentId": "agent-abc123",
            "toolDistribution": {"read": 1},
            "toolCallCount": 10,
            "knownTools": {"read"},
        }
        current = {
            "agentId": "agent-abc123",
            "toolDistribution": {"exec": 1},
            "toolCallCount": 1,
            "knownTools": {"exec"},
        }
        deviations = compute_deviation(current, baseline)
        assert deviations["novelTools"] == ["exec"]

    def test_handles_missing_baseline(self):
        deviations = compute_deviation({"agentId": "agent-abc123"}, None)
        assert deviations["hasBaseline"] is False
        assert deviations["novelTools"] == []
        assert deviations["toolFrequencyChanges"] == []


class TestDetectAnomalies:
    def test_respects_threshold_configuration(self):
        report = detect_anomalies(
            {
                "agentId": "agent-abc123",
                "hasBaseline": True,
                "novelTools": [],
                "toolFrequencyChanges": [
                    {"tool": "bash", "baseline": 0.1, "current": 0.29, "multiplier": 2.9, "delta": 0.19},
                ],
                "tokenDeviation": {
                    "total": {"baseline": 100, "current": 240, "multiplier": 2.4},
                },
                "baseline": {
                    "sessionCount": 3,
                    "toolCallCount": 20,
                    "knownTools": {"bash", "read"},
                },
            },
            {"lookbackDays": 7, "toolDeviationThreshold": 3, "tokenDeviationThreshold": 2.5},
        )
        assert report["detected"] is False
        assert report["deviations"] == []

    def test_flags_novel_tools_with_configured_severity(self):
        report = detect_anomalies(
            {
                "agentId": "agent-abc123",
                "hasBaseline": True,
                "novelTools": ["exec"],
                "toolFrequencyChanges": [],
                "tokenDeviation": {},
                "baseline": {
                    "sessionCount": 3,
                    "toolCallCount": 20,
                    "knownTools": {"read", "write"},
                },
            },
            {"lookbackDays": 7, "novelToolSeverity": "high"},
        )
        assert report["detected"] is True
        assert report["severity"] == "high"
        assert report["deviations"][0]["type"] == "novel_tool"


class TestCache:
    def setup_method(self):
        invalidate_profile_cache()

    def test_invalidates_cached_profiles(self):
        _cache["entries"][7] = {
            "loadedAt": int(time.time() * 1000),
            "profileMap": {"agent-abc123": {"agentId": "agent-abc123"}},
        }
        assert len(_cache["entries"]) == 1
        invalidate_profile_cache()
        assert len(_cache["entries"]) == 0
