#!/usr/bin/env bash
# Add ClawKeeper only to this invocation; keep global/project config untouched.
set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$example_dir/.." && pwd)"
hook_config="$example_dir/codex_hook_override.toml"

if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI is not on PATH." >&2
  exit 1
fi
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "Missing project Python. Run: uv sync --python 3.12 --extra dev" >&2
  exit 1
fi
if [[ ! -r "$hook_config" ]]; then
  echo "Missing hook configuration: $hook_config" >&2
  exit 1
fi

# One TOML key=value argument, read as data without evaluating shell code.
hook_override="$(<"$hook_config")"

# Pass invocation settings to the selected subcommand as well as new chats.
# exec needs its own overrides when using --ignore-user-config in fixtures.
cli_command=()
if [[ "${1:-}" == "exec" || "${1:-}" == "resume" ]]; then
  cli_command=("$1")
  shift
fi

exec codex \
  --no-daemon \
  --ask-for-approval on-request \
  "${cli_command[@]}" \
  --enable hooks \
  --sandbox workspace-write \
  --cd "$repo_root" \
  --config "$hook_override" \
  "$@"
