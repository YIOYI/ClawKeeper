/**
 * OpenClaw plugin entry — wires before_tool_call (and a handful of
 * observation hooks) to the Python core via HTTP.
 *
 * The plugin is intentionally thin: no local decision logic, no
 * pattern caches, no log scanners. Everything substantive lives in
 * clawkeeper-core. If you need to change a rule, edit Python; do not
 * fork this JS file.
 */

import { judge } from "./judge.js";

const PLUGIN_ID = "clawkeeper-shim";
const PLUGIN_NAME = "ClawKeeper Shim";
const PLUGIN_VERSION = "0.2.0";

export const clawkeeperShimPlugin = {
  id: PLUGIN_ID,
  name: PLUGIN_NAME,
  version: PLUGIN_VERSION,
  configSchema: {
    parse(value) {
      if (value && typeof value === "object" && !Array.isArray(value)) return value;
      return {};
    },
  },
  register(api) {
    const config = api.config || {};
    const options = {
      serverUrl: config.serverUrl,
      failurePolicy: config.failurePolicy,
      timeoutMs: config.timeoutMs,
    };
    const mode = config.mode || "local";

    api.on("before_tool_call", async (event, context) => {
      const toolName = event?.toolName || event?.name || "";
      const params = typeof event?.params === "string"
        ? { command: event.params }
        : event?.params ?? {};
      const payload = {
        mode,
        requestId: event?.requestId || event?.toolCallId || event?.callId,
        toolCall: { toolName, params, cwd: context?.cwd ?? event?.cwd },
        forwardedContext: {
          messages: _recentMessages(event),
          metadata: { sessionKey: context?.sessionKey ?? event?.sessionKey },
        },
      };
      const decision = await judge(payload, options);

      if (decision?.decision === "stop" || decision?.decision === "ask_user") {
        return {
          block: true,
          blockReason: decision.userQuestion || decision.summary || decision.stopReason,
        };
      }
      return { block: false };
    });

    api.logger?.info?.(`[${PLUGIN_ID}] registered — forwarding to ${options.serverUrl || "default server"}`);
  },
};


/**
 * Build a forwardedContext.messages slice from whatever shape OpenClaw
 * gave us. The plugin SDK has evolved over time; defensively support
 * both `event.messages` and `event.history`.
 */
function _recentMessages(event) {
  if (!event) return [];
  const history = event.messages || event.history || event.conversationHistory || [];
  return Array.isArray(history) ? history.slice(-20) : [];
}


export default clawkeeperShimPlugin;
