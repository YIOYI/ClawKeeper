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
_MAX_PATH_WORK_CHARS = 1_000_000
_SHORT_OPTION_PREFIX = 2
_MUTATING_METHODS = {'append', 'extend', 'insert', 'clear', 'pop', 'remove', 'sort',
                     'reverse', 'update', 'add', 'discard', 'difference_update',
                     'intersection_update', 'symmetric_difference_update'}
_FILE_FUNCTIONS = {
    'open', 'builtins.open', 'io.open', 'os.open', 'os.remove', 'os.unlink',
    'os.rmdir', 'os.mkdir', 'os.makedirs', 'os.listdir', 'os.scandir',
    'os.stat', 'os.lstat', 'os.chmod', 'os.chown', 'os.chdir', 'os.fchdir',
}
_TWO_PATH_FUNCTIONS = {'os.rename', 'os.replace', 'shutil.copy', 'shutil.copy2',
                       'shutil.copyfile', 'shutil.move'}
_PATH_METHODS = {'read_text', 'read_bytes', 'write_text', 'write_bytes', 'open',
                 'unlink', 'rmdir', 'mkdir', 'iterdir', 'stat', 'lstat', 'chmod',
                 'touch', 'rename', 'replace'}
_PURE_CALLS = {'print', 'builtins.print', 'str', 'repr', 'len', 'sorted', 'list', 'tuple', 'set',
               'Path', 'pathlib.Path', 'os.path.join'}
_PROCESS_CALLS = {'subprocess.run', 'subprocess.call', 'subprocess.check_call',
                  'subprocess.check_output', 'subprocess.Popen'}


class _SequenceValues(frozenset):
    """Possible elements of a sequence, distinct from possible string values."""


def python_paths(tree, aliases, location):  # noqa: PLR0915 -- bounded AST transfer functions share lexical state
    """Return path accesses; resource limits come from the enclosing AST parser."""
    out = []
    remaining_chars = _MAX_PATH_WORK_CHARS
    import_bindings = {}
    path_imports = _FILE_FUNCTIONS | _TWO_PATH_FUNCTIONS | {'os', 'io', 'shutil', 'pathlib', 'pathlib.Path'}
    called_names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for node in ast.walk(tree):
        bindings = []
        if isinstance(node, ast.Import):
            bindings = [(n.asname or n.name.split('.')[0], n.name if n.asname else n.name.split('.')[0]) for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            bindings = [(n.asname or n.name, node.module + '.' + n.name) for n in node.names]
        for name, qualified in bindings:
            import_bindings.setdefault(name, set()).add(qualified)
            if ((len(import_bindings[name]) > 1 and import_bindings[name] & path_imports)
                    or (name == 'open' and name in called_names and qualified not in _FILE_FUNCTIONS)):
                out.append(PathAccess(None, 'ambiguous import binding', location(node)))
    dynamic_bindings = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal))
                        for name in node.names}
    deferred_annotations = any(isinstance(s, ast.ImportFrom) and s.module == '__future__'
                               and any(n.name == 'annotations' for n in s.names) for s in tree.body)

    def charge(size):
        nonlocal remaining_chars
        remaining_chars -= size
        if remaining_chars < 0:
            raise ValueError('path association character budget exhausted')

    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + '.' + node.attr
        return ''

    def shadowed_operation(node, env):
        name = dotted(node)
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        supported = name in _FILE_FUNCTIONS | _TWO_PATH_FUNCTIONS | _PURE_CALLS or name in {
            'Path.home', 'pathlib.Path.home', 'os.path.expanduser'}
        return supported and isinstance(root, ast.Name) and root.id in env

    def combine(left, right, join):
        if left is None or right is None or len(left) * len(right) > _MAX_VALUES:
            return None
        if isinstance(left, _SequenceValues) or isinstance(right, _SequenceValues):
            return None
        charge(sum(len(a) + len(b) for a, b in itertools.product(left, right)))
        values = {join(a, b) for a, b in itertools.product(left, right)}
        return values if all(len(v) <= _MAX_VALUE_CHARS for v in values) else None

    def value(node, env):  # noqa: PLR0911, PLR0912 -- finite expression vocabulary
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return None if node.id in dynamic_bindings else env.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            parts = [value(n, env) for n in node.elts]
            if any(p is None for p in parts):
                return None
            merged = set().union(*parts)
            return _SequenceValues(merged) if len(merged) <= _MAX_VALUES else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            left, right = value(node.left, env), value(node.right, env)
            if isinstance(node.op, ast.Add) and isinstance(left, _SequenceValues) and isinstance(right, _SequenceValues):
                return _SequenceValues(left | right) if len(left | right) <= _MAX_VALUES else None
            join = os.path.join if isinstance(node.op, ast.Div) else lambda a, b: a + b
            return combine(left, right, join)
        if isinstance(node, ast.Subscript) and dotted(node.value) == 'os.environ':
            if isinstance(node.slice, ast.Constant) and node.slice.value == 'HOME':
                return {str(Path.home())}
        if isinstance(node, ast.Call):
            if shadowed_operation(node.func, env):
                return None
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
        charge(sum(len(v) for v in values) if values else 0)
        for path in sorted(values) if values else [None]:
            out.append(PathAccess(path, operation, location(node)))

    def operand(node, index, names):
        return node.args[index] if index < len(node.args) else next(
            (k.value for k in node.keywords if k.arg in names), None)

    def invalidate(names, env):
        # The string-set abstraction does not model mutable containers. Drop
        # aliases as well rather than retaining an obsolete set of loop values.
        previous = [env[n] for n in names if n in env and env[n] is not None]
        for name in list(env):
            if name in names or any(env[name] is p for p in previous):
                env[name] = None

    def modified_names(nodes, *, bindings=True):
        names = set()
        for root in nodes:
            for node in ast.walk(root):
                if bindings and isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr in _MUTATING_METHODS and isinstance(node.func.value, ast.Name)):
                    names.add(node.func.value.id)
                elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name):
                    names.add(node.value.id)
        return names

    def scan(node, env):  # noqa: PLR0912 -- expression scopes and file operand signatures
        if node is None:
            return
        if isinstance(node, (ast.IfExp, ast.BoolOp, ast.Compare, ast.Dict)) and any(
            isinstance(child, ast.NamedExpr) for child in ast.walk(node)
        ):
            # Short-circuit/reordered expression assignments need a CFG. Do
            # not turn the last syntactic assignment into a proved value.
            out.append(PathAccess(None, 'conditional expression binding', location(node)))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            local = dict.fromkeys(env) if isinstance(node, ast.GeneratorExp) else dict(env)
            for generator in node.generators:
                scan(generator.iter, local)
                bind(generator.target, value(generator.iter, local), local)
                for condition in generator.ifs:
                    scan(condition, local)
            for expression in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
                scan(expression, local)
            # Walrus assignments escape a comprehension's target scope.
            changed = {n.target.id for n in ast.walk(node) if isinstance(n, ast.NamedExpr)
                       and isinstance(n.target, ast.Name)}
            invalidate(changed | modified_names([node], bindings=False), env)
            return
        if isinstance(node, ast.Lambda):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                scan(default, env)
            local = dict.fromkeys(env)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                local[arg.arg] = None
            scan(node.body, local)
            return
        for child in ast.iter_child_nodes(node):
            scan(child, env)
        if isinstance(node, ast.NamedExpr):
            bind(node.target, value(node.value, env), env)
        if not isinstance(node, ast.Call):
            return
        if shadowed_operation(node.func, env):
            out.append(PathAccess(None, 'shadowed operation binding', location(node)))
            return
        name = dotted(node.func)
        if name in _FILE_FUNCTIONS | _TWO_PATH_FUNCTIONS:
            if any(k.arg is None or (k.arg in {'dir_fd', 'src_dir_fd', 'dst_dir_fd'}
                   and not (isinstance(k.value, ast.Constant) and k.value.value is None)) for k in node.keywords):
                out.append(PathAccess(None, 'unresolved file-operation directory', location(node)))
            record(operand(node, 0, {'src'} if name in _TWO_PATH_FUNCTIONS else {'file', 'path'}), name, env, node)
            if name in _TWO_PATH_FUNCTIONS:
                record(operand(node, 1, {'dst'}), name, env, node)
        elif name in _PROCESS_CALLS and any(k.arg == 'cwd' and not (
            isinstance(k.value, ast.Constant) and k.value.value is None) for k in node.keywords
        ):
            out.append(PathAccess(None, 'subprocess working directory', location(node)))
        elif isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in _MUTATING_METHODS and isinstance(node.func.value, ast.Name):
                invalidate({node.func.value.id}, env)
            if method == 'replace' and (len(node.args) >= _SHORT_OPTION_PREFIX or any(k.arg in {'old', 'new'} for k in node.keywords)):
                return  # str.replace(old, new) transforms text, unlike Path.replace(target).
            if method in _PATH_METHODS:
                record(node.func.value, '.' + method, env, node)
                if method in {'rename', 'replace'}:
                    record(operand(node, 0, {'target'}), '.' + method, env, node)
            elif method == 'read':
                argument = operand(node, 0, {'filenames'})
                # read(size) on a known open handle, and numeric read sizes,
                # are not filenames. ConfigParser.read(filenames) is a path.
                if argument is not None and value(node.func.value, env) is None and not (
                    isinstance(argument, ast.Constant) and isinstance(argument.value, (int, type(None)))
                ):
                    record(argument, '.read', env, node)
        if name not in _PURE_CALLS:
            # Unknown callees may mutate a visible sequence passed by reference.
            arguments = [*node.args, *(k.value for k in node.keywords)]
            if isinstance(node.func, ast.Attribute) and node.func.attr not in {'copy', 'count', 'index'}:
                arguments.append(node.func.value)
            invalidate({arg.id for arg in arguments if isinstance(arg, ast.Name)
                        and isinstance(env.get(arg.id), _SequenceValues)}, env)

    def bind(target, values, env):
        if isinstance(target, ast.Name):
            env[target.id] = values
        elif isinstance(target, (ast.List, ast.Tuple)):
            for item in target.elts:
                bind(item, None, env)

    def statements(body, env, *, function=False):  # noqa: PLR0912, PLR0915 -- bounded statement transfer
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.AnnAssign) and not deferred_annotations and not function:
                    scan(stmt.annotation, env)
                if stmt.value is not None:
                    scan(stmt.value, env)
                    values = value(stmt.value, env)
                    for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                        scan(target, env)
                        bind(target, values, env)
                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            invalidate({target.value.id}, env)
            elif isinstance(stmt, ast.AugAssign):
                scan(stmt.target, env)
                scan(stmt.value, env)
                values = combine(value(stmt.target, env), value(stmt.value, env), lambda a, b: a + b) if isinstance(stmt.op, ast.Add) else None
                bind(stmt.target, values, env)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for expression in (*stmt.decorator_list, *stmt.args.defaults, *stmt.args.kw_defaults):
                    scan(expression, env)
                arguments = [*stmt.args.posonlyargs, *stmt.args.args, *stmt.args.kwonlyargs,
                             *([stmt.args.vararg] if stmt.args.vararg else []),
                             *([stmt.args.kwarg] if stmt.args.kwarg else [])]
                if not deferred_annotations:
                    for expression in [stmt.returns, *(a.annotation for a in arguments)]:
                        scan(expression, env)
                # Free variables are looked up when called, not when defined.
                # Their definition-time values cannot prove a future access safe.
                local = dict.fromkeys(env)
                for arg in (*stmt.args.posonlyargs, *stmt.args.args, *stmt.args.kwonlyargs):
                    local[arg.arg] = None
                statements(stmt.body, local, function=True)
                env[stmt.name] = None
            elif isinstance(stmt, ast.ClassDef):
                for expression in [*stmt.bases, *stmt.decorator_list, *(k.value for k in stmt.keywords)]:
                    scan(expression, env)
                statements(stmt.body, dict(env))
                env[stmt.name] = None
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                scan(stmt.iter, env)
                local = dict(env)
                changed = modified_names(stmt.body)
                invalidate(changed, local)
                bind(stmt.target, value(stmt.iter, local), local)
                scan(stmt.target, local)
                statements(stmt.body, local, function=function)
                invalidate(changed, local)
                merge(env, env.copy(), local)
                statements(stmt.orelse, env, function=function)
            elif isinstance(stmt, ast.While):
                changed = modified_names(stmt.body)
                local = dict(env)
                invalidate(changed, local)
                scan(stmt.test, local)
                statements(stmt.body, local, function=function)
                invalidate(changed, local)
                merge(env, env.copy(), local)
                statements(stmt.orelse, env, function=function)
            elif isinstance(stmt, ast.If):
                scan(stmt.test, env)
                left, right = dict(env), dict(env)
                statements(stmt.body, left, function=function)
                statements(stmt.orelse, right, function=function)
                merge(env, left, right)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    scan(item.context_expr, env)
                    if item.optional_vars:
                        scan(item.optional_vars, env)
                        bind(item.optional_vars, value(item.context_expr, env), env)
                statements(stmt.body, env, function=function)
            elif isinstance(stmt, ast.Try):
                branches = [stmt.body, *[h.body for h in stmt.handlers]]
                changed = modified_names(stmt.body)
                for index, branch in enumerate(branches):
                    local = dict(env)
                    if index:
                        invalidate(changed, local)
                        scan(stmt.handlers[index - 1].type, local)
                    statements(branch, local, function=function)
                    merge(env, env.copy(), local)
                statements(stmt.orelse, env, function=function)
                invalidate(modified_names([*stmt.body, *stmt.orelse, *stmt.handlers]), env)
                statements(stmt.finalbody, env, function=function)
            elif isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                scan(stmt, env)
                invalidate(modified_names(body), env)
                return
            elif isinstance(stmt, (ast.Match, ast.TryStar)):
                out.append(PathAccess(None, 'unsupported Python control flow', location(stmt)))
                scan(stmt, env)
                invalidate(modified_names([stmt]), env)
            else:
                scan(stmt, env)

    def merge(env, left, right):
        for key in left.keys() | right.keys():
            a, b = left.get(key), right.get(key)
            merged = a | b if a is not None and b is not None and len(a | b) <= _MAX_VALUES else None
            env[key] = _SequenceValues(merged) if merged is not None and isinstance(a, _SequenceValues) and isinstance(b, _SequenceValues) else merged

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
