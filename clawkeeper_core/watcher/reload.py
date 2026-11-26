"""Revalidate persisted candidates and atomically replace process-local policy."""
from __future__ import annotations

import hashlib
import logging
import threading
import time

from clawkeeper_core.bounded_regex import RegexResourceError
from clawkeeper_core.execution import analysis_scope
from clawkeeper_core.learned_rules import (
    audit,
    current_snapshot,
    install_snapshot,
    make_snapshot,
    mark_policy_unavailable,
    policy_error,
    rule_id,
)
from clawkeeper_core.watcher.learner import (
    STORE,
    PatternCandidate,
    StoreCapacityError,
    StoreUnavailableError,
    validate_candidate,
)

_reload_lock = threading.RLock()


def store_learned_pattern(candidate: PatternCandidate) -> bool:
    with _reload_lock:
        identity = STORE.add_with_id(candidate)
        if identity is None:
            return False
        apply_learned_patterns()
        return policy_error() is None and any(r.id == identity for r in current_snapshot().rules)


def _runtime_may_contain(identifier: str, *, exact: bool) -> bool:
    for rule in current_snapshot().rules:
        if identifier == rule.id:
            return True
        if not exact:
            pattern_hash = hashlib.sha256(rule.pattern.encode()).hexdigest()
            # Snapshots omit historical aliases. Include the possible pre-migration ID.
            if identifier in (pattern_hash, rule_id(rule.guard_target, rule.pattern),
                              rule_id(rule.guard_target, rule.pattern, legacy=True),
                              rule_id(rule.guard_target, rule.pattern, rule.subject, legacy=True)):
                return True
    return False


def remove_learned_pattern(pattern_hash: str, *, exact: bool) -> tuple[bool, str, str | None]:
    """Report the store mutation and resulting local policy under the same lock."""
    with _reload_lock:
        try:
            removed = STORE.remove(pattern_hash, exact=exact)
        except StoreUnavailableError:
            mark_policy_unavailable()
            return False, current_snapshot().version, policy_error()
        except (OSError, StoreCapacityError):
            return False, current_snapshot().version, "learned_store_write_failed"
        # An already absent rule needs no worker validation. Reconcile if a
        # failed reload or an externally restored store left a stale snapshot.
        if removed or policy_error() or _runtime_may_contain(pattern_hash, exact=exact):
            apply_learned_patterns()
        return removed, current_snapshot().version, policy_error()


@analysis_scope()
def apply_learned_patterns() -> int:
    """Newly installed count, local to this process. Also applies removals.

    Invalid historical records remain stored for inspection. This is not an
    acknowledgement of Adapter synchronization.
    """
    with _reload_lock:
        try:
            candidates = STORE.all()
        except StoreUnavailableError:
            mark_policy_unavailable()
            return 0
        deadline = time.monotonic() + 5
        rules = []
        for candidate in candidates:
            if time.monotonic() > deadline:
                mark_policy_unavailable("learned_validation_unavailable")
                return 0
            try:
                rule = validate_candidate(candidate)
            except RegexResourceError:
                mark_policy_unavailable("learned_validation_unavailable")
                return 0
            except (TypeError, ValueError) as exc:
                audit("rejected", level=logging.WARNING, guard_target=candidate.guard_target,
                      skip_reason=str(exc))
                continue
            rules.append(rule)
            audit("validated", rule_id=rule.id, guard_target=rule.guard_target)
        try:
            return install_snapshot(make_snapshot(rules))
        except ValueError:
            mark_policy_unavailable()
            return 0
