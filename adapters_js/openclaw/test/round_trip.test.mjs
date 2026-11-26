/**
 * Round-trip test for the shim. Boots the Python server externally
 * (the test runner spawns it), then exercises judge() against it for
 * every relevant decision branch + the failure policies.
 *
 * Run with the Python server already listening on $CLAWKEEPER_TEST_URL
 * (default http://127.0.0.1:7474). Returns non-zero on assertion failure.
 */

import assert from "node:assert/strict";
import { clawkeeperShimPlugin } from "../src/index.js";
import { judge } from "../src/judge.js";

const SERVER_URL = process.env.CLAWKEEPER_TEST_URL || "http://127.0.0.1:7474";


async function suite() {
  console.log(`[round_trip] using server: ${SERVER_URL}\n`);
  let passed = 0;
  let failed = 0;

  const it = async (name, fn) => {
    try {
      await fn();
      console.log(`  PASS  ${name}`);
      passed++;
    } catch (err) {
      console.log(`  FAIL  ${name}`);
      console.log(`        ${err.message}`);
      failed++;
    }
  };

  await it("user_requested_stop", async () => {
    const decision = await judge({
      mode: "local",
      forwardedContext: { messages: [{ role: "user", content: "stop" }] },
    }, { serverUrl: SERVER_URL });
    assert.equal(decision.decision, "stop");
    assert.equal(decision.stopReason, "user_requested_stop");
  });

  await it("hook enforces guard results for any tool name", async () => {
    let beforeToolCall;
    clawkeeperShimPlugin.register({
      config: { serverUrl: SERVER_URL },
      on(name, handler) {
        assert.equal(name, "before_tool_call");
        beforeToolCall = handler;
      },
    });
    for (const toolName of ["bash", "terminal", "read_file", "custom_runner"]) {
      const invoke = (params) => beforeToolCall({ toolName, params });
      assert.deepEqual(await invoke({ command: "ls -la" }), { block: false });
      const blocked = await invoke({ request: { command: "whoami" } });
      assert.equal(blocked.block, true);
      assert.match(blocked.blockReason, /destructive command pattern/);
      const review = await invoke({ command: "printf 'unfinished" });
      assert.equal(review.block, true);
      assert.match(review.blockReason, /review required/);
    }
  });

  await it("missing forwardedContext", async () => {
    const decision = await judge({ mode: "local" }, { serverUrl: SERVER_URL });
    assert.equal(decision.decision, "stop");
    assert.equal(decision.stopReason, "missing_input");
  });

  await it("a legacy core cannot allow an unchecked structured call", async () => {
    const fetch = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ decision: "continue" }) });
    try {
      const decision = await judge({ toolCall: { toolName: "terminal", params: { command: "ls -la" } } });
      assert.equal(decision.decision, "stop");
      assert.match(decision.summary, /did not confirm the pre-execution checks/);
    } finally {
      globalThis.fetch = fetch;
    }
  });

  await it("benign read -> continue", async () => {
    const decision = await judge({
      mode: "local",
      forwardedContext: {
        messages: [
          { role: "user", content: "look at this" },
          { role: "tool", toolName: "read" },
        ],
      },
    }, { serverUrl: SERVER_URL });
    assert.equal(decision.decision, "continue");
  });

  await it("fail-closed on unreachable server", async () => {
    const decision = await judge(
      { mode: "local", forwardedContext: { messages: [{ role: "user", content: "x" }] } },
      { serverUrl: "http://127.0.0.1:1", failurePolicy: "fail-closed", timeoutMs: 500 },
    );
    assert.equal(decision.decision, "stop");
    assert.equal(decision.stopReason, "transport_failure_fail_closed");
  });

  await it("fail-open on unreachable server", async () => {
    const decision = await judge(
      { mode: "local", forwardedContext: { messages: [{ role: "user", content: "x" }] } },
      { serverUrl: "http://127.0.0.1:1", failurePolicy: "fail-open", timeoutMs: 500 },
    );
    assert.equal(decision.decision, "continue");
    assert.equal(decision.stopReason, "transport_failure_fail_open");
  });

  console.log(`\n[round_trip] ${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

suite();
