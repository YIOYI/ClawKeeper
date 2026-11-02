"""Bounded path operands, shared by enforcement and learning.

Python tracks literal strings, path joins, assignments and visible loop values.
It never imports or executes candidate code. Unknown file operands remain facts
with an explicit boundary; unrelated strings never become file accesses.
"""

from __future__ import annotations

import ast
import itertools
import os
from pathlib import Path

from clawkeeper_core.command_facts import PathAccess

_MAX_VALUES = 32
_MAX_VALUE_CHARS = 100_000
_SHORT_OPTION_PREFIX = 2
_FILE_FUNCTIONS = {
    'open', 'builtins.open', 'io.open', 'os.open', 'os.remove', 'os.unlink',
    'os.rmdir', 'os.mkdir', 'os.makedirs', 'os.listdir', 'os.scandir',
    'os.stat', 'os.lstat', 'os.chmod', 'os.chown', 'os.chdir',
}
_TWO_PATH_FUNCTIONS = {'os.rename', 'os.replace', 'shutil.copy', 'shutil.copy2',
                       'shutil.copyfile', 'shutil.move'}
_PATH_METHODS = {'read_text', 'read_bytes', 'write_text', 'write_bytes', 'open',
                 'unlink', 'rmdir', 'mkdir', 'iterdir', 'stat', 'lstat', 'chmod',
                 'touch', 'rename', 'replace'}


def python_paths(tree, aliases, location):  # noqa: PLR0915 -- bounded AST transfer functions share lexical state
    """Return path accesses; resource limits come from the enclosing AST parser."""
    out = []

    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + '.' + node.attr
        return ''

    def combine(left, right, join):
        if left is None or right is None or len(left) * len(right) > _MAX_VALUES:
            return None
        values = {join(a, b) for a, b in itertools.product(left, right)}
        return values if all(len(v) <= _MAX_VALUE_CHARS for v in values) else None

    def value(node, env):  # noqa: PLR0911, PLR0912 -- finite expression vocabulary
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            parts = [value(n, env) for n in node.elts]
            if any(p is None for p in parts):
                return None
            merged = set().union(*parts)
            return merged if len(merged) <= _MAX_VALUES else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            join = os.path.join if isinstance(node.op, ast.Div) else lambda a, b: a + b
            return combine(value(node.left, env), value(node.right, env), join)
        if isinstance(node, ast.Subscript) and dotted(node.value) == 'os.environ':
            if isinstance(node.slice, ast.Constant) and node.slice.value == 'HOME':
                return {str(Path.home())}
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name in {'Path.home', 'pathlib.Path.home', 'os.path.expanduser'}:
                if name.endswith('.home'):
                    return {str(Path.home())}
                arg = value(node.args[0], env) if node.args else None
                return {os.path.expanduser(v) for v in arg} if arg is not None else None
            if name in {'Path', 'pathlib.Path', 'str', 'os.path.join'}:
                values = {''}
                for arg in node.args:
                    values = combine(values, value(arg, env), os.path.join)
                return values
            if name in {'open', 'builtins.open', 'io.open'}:
                return value(node.args[0], env) if node.args else None
            if isinstance(node.func, ast.Attribute) and node.func.attr in {'expanduser', 'resolve', 'absolute'}:
                values = value(node.func.value, env)
                return {os.path.expanduser(v) for v in values} if values is not None else None
        return None

    def record(operand, operation, env, node):
        values = value(operand, env)
        for path in sorted(values) if values else [None]:
            out.append(PathAccess(path, operation, location(node)))

    def scan(expression, env):
        for node in ast.walk(expression):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if name in _FILE_FUNCTIONS | _TWO_PATH_FUNCTIONS:
                args = list(node.args)
                if not args:
                    args = [k.value for k in node.keywords if k.arg in {'file', 'path'}]
                if args:
                    record(args[0], name, env, node)
                else:
                    out.append(PathAccess(None, name, location(node)))
                if name in _TWO_PATH_FUNCTIONS and len(args) > 1:
                    record(args[1], name, env, node)
            elif isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in _PATH_METHODS:
                    record(node.func.value, '.' + method, env, node)
                    if method in {'rename', 'replace'} and node.args:
                        record(node.args[0], '.' + method, env, node)
                elif method in {'read', 'read_file'} and node.args:
                    # Includes configparser.read(path); read() on a handle is
                    # already covered at open(), without treating data as a path.
                    record(node.args[0], '.' + method, env, node)

    def bind(target, values, env):
        if isinstance(target, ast.Name):
            env[target.id] = values
        elif isinstance(target, (ast.List, ast.Tuple)):
            for item in target.elts:
                bind(item, None, env)

    def statements(body, env):  # noqa: PLR0912 -- bounded statement transfer
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if stmt.value is not None:
                    scan(stmt.value, env)
                    values = value(stmt.value, env)
                    for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                        bind(target, values, env)
            elif isinstance(stmt, ast.AugAssign):
                scan(stmt.value, env)
                values = combine(value(stmt.target, env), value(stmt.value, env), lambda a, b: a + b) if isinstance(stmt.op, ast.Add) else None
                bind(stmt.target, values, env)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = dict(env)
                for arg in (*stmt.args.posonlyargs, *stmt.args.args, *stmt.args.kwonlyargs):
                    local[arg.arg] = None
                statements(stmt.body, local)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                scan(stmt.iter, env)
                local = dict(env)
                bind(stmt.target, value(stmt.iter, env), local)
                statements(stmt.body, local)
                merge(env, env.copy(), local)
                statements(stmt.orelse, env)
            elif isinstance(stmt, (ast.If, ast.While)):
                scan(stmt.test, env)
                left, right = dict(env), dict(env)
                statements(stmt.body, left)
                statements(stmt.orelse, right)
                merge(env, left, right)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    scan(item.context_expr, env)
                    if item.optional_vars:
                        bind(item.optional_vars, value(item.context_expr, env), env)
                statements(stmt.body, env)
            elif isinstance(stmt, ast.Try):
                branches = [stmt.body, *[h.body for h in stmt.handlers]]
                for branch in branches:
                    local = dict(env)
                    statements(branch, local)
                    merge(env, env.copy(), local)
                statements(stmt.orelse, env)
                statements(stmt.finalbody, env)
            else:
                scan(stmt, env)

    def merge(env, left, right):
        for key in left.keys() | right.keys():
            a, b = left.get(key), right.get(key)
            env[key] = a | b if a is not None and b is not None and len(a | b) <= _MAX_VALUES else None

    statements(tree.body, {})
    return out


_FILE_PROGRAMS = {
    'cat', 'head', 'tail', 'less', 'more', 'xxd', 'od', 'base64', 'cp', 'mv',
    'rm', 'rmdir', 'mkdir', 'touch', 'tee', 'ls', 'stat', 'chmod', 'chown',
    'find', 'du', 'wc', 'sort', 'uniq', 'diff', 'cmp', 'readlink', 'realpath',
    'cd', 'pushd', 'popd', 'install', 'truncate', 'shred', 'ln', 'file',
}


def shell_paths(call):
    """Use parsed operand roles, never re-scan an inline script or output data."""
    paths = []
    if call.kind != 'process':
        return paths
    if '/' in call.program.raw:
        paths.append(PathAccess(call.program.raw.strip('\"\''), 'execute', call.origin))
    for index, arg in enumerate(call.arguments):
        if (index and call.arguments[index - 1].value in {'-o', '--output', '--output-document', '--body-file'}
                and call.executable in {'curl', 'wget', 'gh'}):
            paths.append(PathAccess(arg.value, call.executable, arg.origin or call.origin))
            continue
        if arg.role == 'option' and arg.value:
            option, separator, operand = arg.value.partition('=')
            if separator and operand.startswith(('~/', '/', './', '../')):
                known = option in {'--output', '--output-document', '--target-directory', '--reference', '--file'} and call.executable in _FILE_PROGRAMS | {'curl', 'wget'}
                paths.append(PathAccess(operand, call.executable, arg.origin or call.origin,
                                        'access' if known else 'uncertain'))
            elif len(arg.value) > _SHORT_OPTION_PREFIX and arg.value[_SHORT_OPTION_PREFIX:].startswith(('~/', '/', './', '../')):
                # Attached short-option operands need program-specific option
                # knowledge. Preserve the boundary instead of dropping a path.
                paths.append(PathAccess(arg.value[_SHORT_OPTION_PREFIX:], call.executable, arg.origin or call.origin, 'uncertain'))
            continue
        if arg.role in {'data', 'code', 'metadata', 'module', 'option'}:
            continue
        if arg.role == 'script' or call.executable in _FILE_PROGRAMS:
            paths.append(PathAccess(arg.value, call.executable, arg.origin or call.origin))
        elif arg.value and (arg.value.startswith(('~/', '/', './', '../'))):
            paths.append(PathAccess(arg.value, call.executable, arg.origin or call.origin, 'uncertain'))
    for redirect in call.redirects:
        if redirect.kind == 'file' and redirect.operator not in {'>&', '<&'}:
            paths.append(PathAccess(redirect.target.value, redirect.operator, redirect.origin))
    return paths
