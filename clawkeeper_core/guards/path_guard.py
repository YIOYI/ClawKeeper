"""Protected-path guard. Ported from
legacy/clawkeeper-plugin/src/core/path-guard.js.

Hard-blocks any tool that would read/write/exec against a path on the
protected list (e.g. ~/.ssh, /etc/shadow, ~/.aws/credentials).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from clawkeeper_core.command_facts import PathAccess
from clawkeeper_core.execution import analysis_scope, analyze_event
from clawkeeper_core.learned_rules import match_learned

_DEFAULT_FAILURE_POLICY = "fail-closed"

_DEFAULT_RULES = [
    {"pattern": "~/.ssh/**", "severity": "CRITICAL", "reason": "SSH key material"},
    {"pattern": "~/.aws/**", "severity": "CRITICAL", "reason": "AWS credentials"},
    {"pattern": "~/.gnupg/**", "severity": "CRITICAL", "reason": "GPG key material"},
    {"pattern": "~/.env", "severity": "HIGH", "reason": "Local secrets"},
    {"pattern": "/etc/passwd", "severity": "HIGH", "reason": "System users"},
    {"pattern": "/etc/shadow", "severity": "CRITICAL", "reason": "System password hashes"},
    {"pattern": "/etc/sudoers", "severity": "CRITICAL", "reason": "Sudoer rules"},
]

_cached: dict[str, Any] | None = None


def expand_home(p: str | None) -> str | None:
    if not p:
        return p
    if p == "~":
        return str(Path.home())
    if p.startswith("~/") or p.startswith("~\\"):
        return str(Path.home() / p[2:])
    return p


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """Convert simple glob to anchored regex. ** -> .* ; * -> [^/]* ; ? -> [^/]"""
    out = ["^"]
    i = 0
    while i < len(glob):
        c = glob[i]
        if c == "*":
            if i + 1 < len(glob) and glob[i + 1] == "*":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c in ".+^${}()|[]\\":
            out.append("\\" + c)
        else:
            out.append(c)
        i += 1
    out.append("$")
    return re.compile("".join(out))


def load_protected_paths(rules_path: str | Path | None = None) -> dict[str, Any]:
    global _cached  # noqa: PLW0603 -- process-local rule cache
    if rules_path is None:
        rules = [{**r, "regex": glob_to_regex(expand_home(r["pattern"])),
                  "expanded": expand_home(r["pattern"])}
                 for r in _DEFAULT_RULES]
        _cached = {"rules": rules,
                   "config": {"enabled": True, "failurePolicy": _DEFAULT_FAILURE_POLICY, "bashLikeTools": []},
                   "_source": None}
        return _cached
    if _cached and _cached.get("_source") == str(rules_path):
        return _cached
    raw = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    rules = [
        {**r, "regex": glob_to_regex(expand_home(r["pattern"])),
         "expanded": expand_home(r["pattern"])}
        for r in (raw.get("protectedPaths") or [])
    ]
    cfg = raw.get("pathGuard") or {"enabled": True, "failurePolicy": "fail-closed", "bashLikeTools": []}
    _cached = {"rules": rules, "config": cfg, "_source": str(rules_path)}
    return _cached


def reset_path_guard_cache() -> None:
    global _cached  # noqa: PLW0603 -- process-local rule cache
    _cached = None


def normalize_path(input_path: Any, cwd: str | Path | None = None) -> str | None:
    if not isinstance(input_path, str) or not input_path:
        return None
    p = input_path.strip()
    # Strip surrounding quotes
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    p = expand_home(p) or ""
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(str(cwd or os.getcwd()), p))
    try:
        return os.path.realpath(p)
    except OSError:
        return os.path.abspath(p)


def match_protected(abs_path: str | None, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not abs_path:
        return None
    for rule in rules:
        if rule["regex"].match(abs_path):
            return rule
        if rule["pattern"].endswith("/**"):
            prefix = rule["expanded"][:-3]
            if abs_path == prefix or abs_path.startswith(prefix + os.sep) or abs_path.startswith(prefix + "/"):
                return rule
    return None


class PathAnalysisIncomplete(ValueError):
    """A visible file access could not be resolved by bounded analysis."""


_PATH_FIELDS = {"path", "file", "file_path", "filePath", "filename", "file_name",
                "directory", "dir_path", "source", "destination", "target", "paths", "location"}
_MAX_ENVELOPE_DEPTH = 64
_DATA_FIELDS = {"content", "body", "description", "metadata", "env", "environment", "headers"}


def path_facts(event: dict, *, bash_like_tools=None):
    facts = analyze_event(event, bash_like_tools=bash_like_tools)
    paths = list(facts.paths)
    changing_cwd = {c.origin.source for c in facts.calls
                    if c.executable in {"cd", "pushd", "popd", "python:os.chdir"}
                    or (c.executable in {"env", "sudo"} and any(
                        a.display.split("=", 1)[0] in {"-C", "--chdir", "-D", "-R"} for a in c.arguments))}
    if changing_cwd:
        for source_id, source in enumerate(facts.sources):
            if source.parent is not None and source.parent.source in changing_cwd:
                changing_cwd.add(source_id)
        paths = [replace(p, value=None) if p.origin is not None and p.origin.source in changing_cwd
                 and p.value is not None and not p.value.startswith(("/", "~/")) else p
                 for p in paths]
    remaining = 10_000

    def collect(obj, field=()):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or len(field) > _MAX_ENVELOPE_DEPTH:
            raise PathAnalysisIncomplete("path envelope budget exhausted")
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in _DATA_FIELDS:
                    continue
                if key in _PATH_FIELDS:
                    values = value if isinstance(value, list) else [value]
                    for item in values:
                        if isinstance(item, str) and item:
                            paths.append(PathAccess(item, "file-tool", None, field=("params", *field, key)))
                elif isinstance(value, (dict, list)):
                    collect(value, (*field, key))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                collect(value, (*field, str(i)))

    collect(event.get("params") or {})
    return facts, paths


def event_cwd(event):
    params = event.get("params") or {}
    return event.get("cwd") or params.get("cwd") or params.get("workdir") or params.get("working_directory")


@analysis_scope()
def extract_paths_from_command(command: str | None) -> list[str]:
    return extract_paths_from_params("terminal", {"command": command})


@analysis_scope()
def extract_paths_from_params(tool_name: Any, params: Any, *, bash_like_tools: list[str] | None = None) -> list[str]:
    event = {"toolName": tool_name, "params": params}
    facts, paths = path_facts(event, bash_like_tools=bash_like_tools)
    if not facts.complete or any(p.value is None or p.certainty != "access" for p in paths):
        raise PathAnalysisIncomplete("path context analysis incomplete")
    return sorted({p.value for p in paths})


@analysis_scope()
def guard_before_tool_call(event: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0911 -- deny, review and configured failure decisions
    try:
        loaded = load_protected_paths()
    except (OSError, json.JSONDecodeError) as err:
        policy = (_cached or {}).get("config", {}).get("failurePolicy", _DEFAULT_FAILURE_POLICY)
        if policy == "fail-closed":
            return {"block": True, "error": str(err), "reason": "path-guard rule load failed (fail-closed policy)"}
        return {"block": False, "error": str(err)}
    rules, cfg = loaded["rules"], loaded["config"]
    if not cfg["enabled"]:
        return {"block": False}

    try:
        facts, paths = path_facts(event, bash_like_tools=cfg.get("bashLikeTools") or [])
    except PathAnalysisIncomplete as exc:
        return {"block": True, "outcome": "ask", "source": "path_guard:analysis",
                "analysis": "indeterminate", "reason": str(exc)}
    uncertain = not facts.complete
    for path in paths:
        candidate = path.value
        if candidate is None:
            uncertain = True
            continue
        resolved = normalize_path(candidate, event_cwd(event))
        hit = match_protected(resolved, rules) or match_protected(expand_home(candidate), rules)
        if path.certainty != "access":
            uncertain = True
            continue
        if hit:
            return {
                "block": True,
                "outcome": "deny",
                "source": "path_guard",
                "evidence": {"origin": asdict(path.origin) if path.origin is not None else None,
                             "field": list(path.field), "operation": path.operation},
                "matched": hit["pattern"],
                "candidate": candidate,
                "resolved": resolved,
                "severity": hit.get("severity"),
                "reason": hit.get("reason"),
            }
    if uncertain:
        return {"block": True, "outcome": "ask", "source": "path_guard:analysis",
                "analysis": "indeterminate", "reason": "Path access analysis incomplete",
                "evidence": {"diagnostics": list(facts.errors)}}
    return match_learned("path_guard", event) or {"block": False}
