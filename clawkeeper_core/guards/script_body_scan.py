"""Sensitive-path checks over parsed access operands, without distance windows.

Python path facts carry bounded assignment/join/loop associations. Comments,
whitespace and unrelated strings cannot change these associations. Unknown
file operands request review; arbitrary program semantics are outside the model.
"""

from __future__ import annotations

from clawkeeper_core.execution import analysis_scope, analyze_event
from clawkeeper_core.learned_rules import match_learned

_SENSITIVE_MARKERS = (
    '.aws', '.ssh', '.gnupg', '.kube', '.docker/config.json', '.pypirc', '.netrc',
    '/etc/shadow', '/etc/sudoers', '/etc/passwd',
)


@analysis_scope()
def script_body_scan_guard(event: dict) -> dict | None:
    learned = match_learned('script_body_scan', event)
    if learned is not None:
        return learned
    facts = analyze_event(event)
    unresolved = not facts.complete
    for path in facts.paths:
        if path.value is None:
            unresolved = True
            continue
        markers = [m for m in _SENSITIVE_MARKERS if m in path.value.lower()]
        if markers:
            if path.certainty != 'access':
                unresolved = True
                continue
            return {
                'block': True, 'outcome': 'deny', 'source': 'script_body_scan:path_access',
                'severity': 'high',
                'reason': f'Sensitive path in parsed file operation {path.operation!r}',
                'evidence': {'sensitive_markers': markers, 'file_op_markers': [path.operation],
                             'source_id': path.origin.source,
                             'origin': {'source': path.origin.source, 'start': path.origin.start, 'end': path.origin.end}},
            }
    if unresolved:
        return {'block': True, 'outcome': 'ask', 'source': 'script_body_scan:analysis',
                'reason': 'File access or execution context analysis incomplete', 'analysis': 'indeterminate'}
    return None
