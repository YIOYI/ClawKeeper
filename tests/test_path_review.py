"""Regressions found while reviewing the path operand/flow changes."""

import json
import shlex

import pytest

from clawkeeper_core import learned_rules as rules
from clawkeeper_core import path_analysis
from clawkeeper_core.execution import explain
from clawkeeper_core.guards import path_guard
from clawkeeper_core.guards.path_guard import guard_before_tool_call


def event(code):
    return {'toolName': 'terminal', 'params': {'command': 'python -c ' + shlex.quote(code)}}


@pytest.mark.parametrize('code', [
    "import shutil; shutil.copy('/tmp/source', dst='/etc/shadow')",
    "import shutil; shutil.copy(src='/tmp/source', dst='/etc/shadow')",
    "import os; os.rename('/tmp/source', dst='/etc/shadow')",
    "import os; os.replace(src='/tmp/source', dst='/etc/shadow')",
    "from pathlib import Path; Path('/tmp/source').rename(target='/etc/shadow')",
    "from pathlib import Path; Path('/tmp/source').replace(target='/etc/shadow')",
    "x: open('/etc/shadow').read() = 1",
    "def f(p=open('/etc/shadow').read()):\n pass",
    "def f(*, p=open('/etc/shadow').read()):\n pass",
    "@open('/etc/shadow').read()\ndef f():\n pass",
    "def f(p: open('/etc/shadow').read()):\n pass",
    "def f() -> open('/etc/shadow').read():\n pass",
    "p='/tmp/a'\n[open(p).read() for p in ['/etc/shadow']]",
    "p='/tmp/a'\n{p:open(p).read() for p in ['/etc/shadow']}",
])
def test_visible_access_cannot_hide_in_operands_or_expression_scopes(code):
    assert explain(event(code))['decision'] == 'deny'


@pytest.mark.parametrize('code', [
    "p='/tmp/a'\nfor _ in range(2):\n open(p).read()\n p='/etc/shadow'",
    "p='/tmp/a'\ni=0\nwhile i<2:\n open(p).read()\n p='/etc/shadow'\n i+=1",
    "p='/tmp/a'\ndef f():\n return open(p).read()\np='/etc/shadow'\nf()",
    "paths=['/tmp/a']\npaths.append('/etc/shadow')\nfor p in paths:\n open(p).read()",
    "paths=['/tmp/a']\nother=paths\nother.append('/etc/shadow')\nfor p in paths:\n open(p).read()",
    "p='/tmp/a'\ntry:\n p='/etc/shadow'\n something()\n p='/tmp/a'\nexcept Exception:\n open(p).read()",
    "p='/tmp/a'\nfor _ in range(2):\n if condition:\n  p='/etc/shadow'\n  break\n p='/tmp/a'\nopen(p).read()",
    "p='/tmp/a'\n[(p:='/etc/shadow') for _ in [1]]\nopen(p).read()",
    "import shutil; shutil.copy('/tmp/source', **options)",
])
def test_unresolved_flow_never_becomes_a_clean_path_sample(code):
    sample = event(code)
    assert guard_before_tool_call(sample)['outcome'] in {'ask', 'deny'}
    assert explain(sample)['decision'] in {'ask', 'deny'}
    with pytest.raises(ValueError):
        rules.target_texts('path_guard', sample)


@pytest.mark.parametrize('code', [
    "open('/tmp/guide.txt').read(1024)",
    "open('/tmp/guide.txt','rb').read(1024)",
    "with open('/tmp/guide.txt') as f:\n f.read(10)",
    "n=10\nwith open('/tmp/guide.txt') as f:\n f.read(n)",
    "import io; io.StringIO('example').read(20)",
    "text='/etc/shadow'; print(text.replace('shadow','hostname'))",
    "from __future__ import annotations\nx: open('/etc/shadow').read() = 1",
])
def test_normal_reads_and_literal_processing_remain_allowed(code):
    assert explain(event(code))['decision'] == 'allow'


@pytest.mark.parametrize('code', [
    "p='/tmp/a'\nclass C:\n p='/etc/shadow'\n open(p).read()",
    "try:\n pass\nexcept type(open('/etc/shadow').read()):\n pass",
    "def f(*args: open('/etc/shadow').read()):\n pass",
    "def f(**kwargs: open('/etc/shadow').read()):\n pass",
    "a=['/tmp/a']\nb=['/etc/shadow']\nfor p in a+b:\n open(p).read()",
])
def test_accesses_in_class_handlers_and_sequence_concatenation(code):
    sample = event(code)
    assert explain(sample)['decision'] == 'deny'
    assert '/etc/shadow' in rules.target_texts('path_guard', sample)


@pytest.mark.parametrize('code', [
    "p='/tmp/a'\ndef f():\n global p\n p='/etc/shadow'\np='/tmp/a'\nf()\nopen(p).read()",
    "paths=['/tmp/a']\n[paths.append('/etc/shadow') for _ in [1]]\nfor p in paths:\n open(p).read()",
    "paths=['/tmp/a']\ntransform(paths)\nfor p in paths:\n open(p).read()",
    "p='/tmp/a'\ng=(open(p).read() for _ in [1])\np='/etc/shadow'\nlist(g)",
])
def test_deferred_and_mutable_bindings_require_review(code):
    sample = event(code)
    assert explain(sample)['decision'] == 'ask'
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)


def test_aggregate_path_work_budget_reports_incomplete(monkeypatch):
    monkeypatch.setattr(path_analysis, '_MAX_PATH_WORK_CHARS', 30)
    # An ordinary literal join fits, repeated accesses exhaust the aggregate
    # work budget rather than producing a large set of expanded path evidence.
    sample = event("p='/tmp/'+'guide'\n" + "open(p).read()\n" * 10)
    assert explain(sample)['decision'] == 'ask'
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)


@pytest.mark.parametrize('code', [
    "d={}\nd[open('/etc/shadow').read()] = 1",
    "d={}\nd[open('/etc/shadow').read()] += 1",
    "for d[open('/etc/shadow').read()] in [1]:\n pass",
])
def test_assignment_targets_are_evaluated_expressions(code):
    assert explain(event(code))['decision'] == 'deny'


@pytest.mark.parametrize('code', [
    "p='/tmp/a'\nx=(p:='/etc/shadow') if condition else (p:='/tmp/a')\nopen(p).read()",
    "p='/tmp/a'\nx=condition and (p:='/etc/shadow') or (p:='/tmp/a')\nopen(p).read()",
    "p='/tmp/a'\nx=(p:='/etc/shadow')=='other'==(p:='/tmp/a')\nopen(p).read()",
    "p='/tmp/a'\nx={(p:='/etc/shadow'):open(p).read(),(p:='/tmp/a'):0}",
    "p='/tmp/a'\ntry:\n p='/etc/shadow'\n raise Exception()\n p='/tmp/a'\nfinally:\n open(p).read()",
    "p='/tmp/a'\nmatch condition:\n case 1:\n  p='/etc/shadow'\n  open(p).read()",
])
def test_unmodeled_expression_and_control_flow_cannot_prove_safety(code):
    sample = event(code)
    assert explain(sample)['decision'] in {'ask', 'deny'}
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)


@pytest.mark.parametrize('code', [
    "from os import open as f\ndef g():\n from unrelated import x as f\nf('/etc/shadow', 0)",
    "open('/etc/shadow').read()\nfrom unrelated import x as open",
    "from pathlib import Path as P\ndef f(P):\n P('/tmp/a').read_text()",
    "from pathlib import Path as P\ndef P(path):\n return make_reader(path)\nP('/tmp/a').read_text()",
])
def test_ambiguous_or_shadowed_imports_are_not_trusted_constructors(code):
    sample = event(code)
    assert explain(sample)['decision'] == 'ask'
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)


def test_configured_paths_and_tool_mapping_survive_shared_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(path_guard, '_cached', None)
    secret = tmp_path / 'CK_CUSTOM_SECRET'
    config = tmp_path / 'paths.json'
    config.write_text(json.dumps({'protectedPaths': [{'pattern': str(secret), 'reason': 'custom policy'}],
                                 'pathGuard': {'bashLikeTools': ['custom_runner']}}))
    loaded = path_guard.load_protected_paths(config)
    sample = {'toolName': 'custom_runner', 'params': {'input': 'cat ' + str(secret)}}
    assert guard_before_tool_call(sample)['matched'] == str(secret)
    assert rules.target_texts('path_guard', sample) == [str(secret)]
    assert path_guard.load_protected_paths() is loaded
    path_guard.reset_path_guard_cache()
    assert path_guard.load_protected_paths()['_source'] is None


def test_unrelated_local_import_names_are_not_path_boundaries():
    code = "def f():\n import math as local\ndef g():\n import json as local\nprint('example')"
    assert explain(event(code))['decision'] == 'allow'


@pytest.mark.parametrize('code', [
    "import os; os.open('shadow', 0, dir_fd=directory)",
    "import os; os.open('shadow', 0, **options)",
    "import os; os.rename('shadow', '/tmp/result', src_dir_fd=directory)",
    "import os; os.fchdir(directory); open('shadow').read()",
    "import subprocess; subprocess.run(['cat','shadow'], cwd='/etc')",
])
def test_descriptor_and_child_process_directory_contexts_require_review(code):
    sample = event(code)
    assert explain(sample)['decision'] == 'ask'
    with pytest.raises(ValueError, match='incomplete'):
        rules.target_texts('path_guard', sample)
