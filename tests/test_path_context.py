"""Access/data pairs exercise the default chain and the learned path lifecycle."""

import json
import shlex
from dataclasses import asdict
from pathlib import Path

import pytest

from clawkeeper_core import learned_rules as rules
from clawkeeper_core.execution import analyze_event, explain
from clawkeeper_core.guards.path_guard import guard_before_tool_call
from clawkeeper_core.guards.script_body_scan import script_body_scan_guard
from clawkeeper_core.watcher import reload
from clawkeeper_core.watcher.learner import PatternCandidate


def event(command, **params):
    return {'toolName': 'terminal', 'params': {'command': command, **params}}


def python(code):
    return 'python -c ' + shlex.quote(code)


@pytest.mark.parametrize('data,access', [
    ("echo '~/.aws/credentials'", 'cat ~/.aws/credentials'),
    ("printf '%s' '~/.aws/credentials'", 'head ~/.aws/credentials'),
    (python("print('~/.aws/credentials')"), python("open('~/.aws/credentials').read()")),
    (python("open('/tmp/guide.txt', 'w').write('~/.aws/credentials')"),
     python("open('~/.aws/credentials', 'w').write('example')")),
    (python("from pathlib import Path; Path('/tmp/guide.txt').write_text('~/.aws/credentials')"),
     python("from pathlib import Path; Path('~/.aws/credentials').read_text()")),
    ("cat <<'DOC' > /tmp/guide.txt\n~/.aws/credentials\nDOC", "printf example > ~/.aws/credentials"),
    ("cat <<< '~/.aws/credentials'", "cat < ~/.aws/credentials"),
    ("echo 'bash -c \"cat ~/.aws/credentials\"'", "bash -c 'cat ~/.aws/credentials'"),
    ("echo '$(cat ~/.aws/credentials)'", 'echo "$(cat ~/.aws/credentials)"'),
    ("git commit -m '~/.aws/credentials'", "env MODE=test cat ~/.aws/credentials"),
])
def test_path_data_and_access_in_default_chain(data, access):
    assert explain(event(data))['decision'] == 'allow'
    assert explain(event(access))['decision'] == 'deny'
    assert rules.target_texts('path_guard', event(data)) == [] or all(
        '.aws' not in p for p in rules.target_texts('path_guard', event(data)))


@pytest.mark.parametrize('padding', ['', '#' + '.' * 600 + '\n', '\n' * 800, '# 中文\n' * 200])
@pytest.mark.parametrize('code', [
    "from pathlib import Path\ns='.aws'\n{padding}p=Path.home()/s/'credentials'\np.read_text()",
    "from pathlib import Path as P\ns='credentials'\np=P.home()/'.aws'/s\n{padding}p.read_bytes()",
    "from pathlib import Path\np=Path.home()/'.aws'/'credentials'\npaths=[p]\n{padding}for f in paths:\n f.read_text()",
])
def test_semantic_path_association_survives_padding(code, padding):
    sample = event(python(code.format(padding=padding)))
    assert script_body_scan_guard(sample)['outcome'] == 'deny'
    assert explain(sample)['decision'] == 'deny'
    facts = analyze_event(sample)
    access = next(p for p in facts.paths if p.operation in {'.read_text', '.read_bytes'})
    source = facts.sources[access.origin.source].text.encode()
    assert b'.read_' in source[access.origin.start:access.origin.end]


@pytest.mark.parametrize('code', [
    "secret='.aws'\n" + '# filler\n' * 100 + "open('/tmp/guide.txt').read()",
    "from pathlib import Path\np=Path('/tmp/guide.txt')\np.write_text('~/.aws/credentials')",
    "from pathlib import Path\ns='.aws'\ns='guide'\n(Path('/tmp')/s).read_text()",
    "from pathlib import Path\np=Path.home()/'.aws'/'credentials'\nprint(p)",
])
def test_unrelated_sensitive_strings_are_not_accesses(code):
    sample = event(python(code))
    assert script_body_scan_guard(sample) is None
    assert explain(sample)['decision'] == 'allow'


@pytest.mark.parametrize('command', [
    'cp --target-directory=' + str(Path.home() / '.aws') + ' /tmp/example',
    'curl https://example.com/ --output=' + str(Path.home() / '.aws' / 'credentials'),
])
def test_option_file_operands_are_not_discarded(command):
    assert guard_before_tool_call(event(command))['outcome'] == 'deny'


def test_relative_paths_and_explicit_fields_keep_context():
    assert explain(event('cat credentials', cwd=str(Path.home() / '.aws')))['decision'] == 'deny'
    assert explain(event('cat credentials', workdir=str(Path.home() / '.aws')))['decision'] == 'deny'
    sample = {'toolName': 'write_file', 'params': {'path': '/tmp/guide.txt', 'content': '~/.aws/credentials'}}
    assert not guard_before_tool_call(sample)['block']
    sample['params']['path'] = '~/.aws/credentials'
    assert guard_before_tool_call(sample)['block']


@pytest.mark.parametrize('command', [
    python('open(filename).read()'), 'cat "$PATH_TO_READ"', 'bash -c "$SCRIPT"',
    'custom_program ~/.aws/credentials', 'cd /etc; cat shadow',
    'cd /etc; ' + python("open('shadow').read()"),
    python("import os; os.chdir('/etc'); open('shadow').read()"),
    'env -C /etc cat shadow', 'sudo -D /etc cat shadow',
    'cat --unknown-option=/etc/shadow', 'head -c/etc/shadow',
])
def test_unresolved_accesses_report_boundary(command):
    sample = event(command)
    assert guard_before_tool_call(sample)['outcome'] == 'ask'
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)


def candidate(command, **kw):
    return PatternCandidate(pattern=r'CK_PATH_\d+', guard_target='path_guard',
                            source_command=command, confidence=.99,
                            created_at='2026-11-02T00:00:00Z', **kw)


def test_learning_path_provenance_reload_and_delete(isolated_learning):
    source = 'cat /tmp/CK_PATH_8421'
    data = 'echo /tmp/CK_PATH_8421'
    with pytest.raises(ValueError, match='does not match the source'):
        rules.validate_rule(r'CK_PATH_\d+', 'path_guard', source_event=event(data))
    assert not isolated_learning.add(candidate(data))
    assert isolated_learning.add(candidate(source))
    assert reload.apply_learned_patterns() == 1
    assert guard_before_tool_call(event(source))['source'] == 'learned:path_guard'
    assert not guard_before_tool_call(event(data))['block']
    saved = isolated_learning._path.read_bytes()
    rules.install_snapshot(rules.make_snapshot([]))
    assert reload.apply_learned_patterns() == 1
    assert isolated_learning._path.read_bytes() == saved
    assert isolated_learning.remove(rules.rule_id('path_guard', r'CK_PATH_\d+'))
    reload.apply_learned_patterns()
    assert not guard_before_tool_call(event(source))['block']


def test_legacy_path_samples_without_provenance_are_retired(isolated_learning):
    legacy = candidate('', source_texts=['/tmp/CK_PATH_8421'], source_view_version=5)
    isolated_learning._path.write_text(json.dumps([asdict(legacy)]))
    assert reload.apply_learned_patterns() == 0
    stored = isolated_learning.all()[0]
    assert stored.inactive_reason and 'provenance' in stored.inactive_reason
    assert isolated_learning.remove(rules.rule_id('path_guard', legacy.pattern))
