"""Read every tracked file and record structural evidence without importing it.

This is an audit inventory, not a substitute for semantic review. Source-level
observations and manually reviewed files must be identified separately.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = {'.py', '.rs', '.cc', '.cpp', '.cu', '.h', '.hpp', '.jl', '.ts',
          '.js', '.m', '.sh', '.ps1', '.bat'}


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return name(node.value) + '.' + node.attr
    return ''


def summarize_python(text, path):
    tree = ast.parse(text, filename=path)
    imports, functions, classes, flags = [], [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend({'module': x.name, 'line': node.lineno, 'level': 0}
                           for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append({'module': node.module or '', 'level': node.level,
                            'names': [x.name for x in node.names], 'line': node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = collections.Counter(name(x.func) for x in ast.walk(node)
                                        if isinstance(x, ast.Call))
            writes = sorted({name(t) for x in ast.walk(node)
                             if isinstance(x, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                             for t in (x.targets if isinstance(x, ast.Assign) else [x.target])
                             if name(t).startswith('self.')})
            doc = ast.get_docstring(node) or ''
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            normalized = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
            functions.append({'name': node.name, 'line': node.lineno,
                              'end': node.end_lineno, 'doc': doc.split('\n')[0],
                              'calls': dict(calls), 'self_writes': writes,
                              'body_hash': hashlib.sha256(normalized.encode()).hexdigest()})
        elif isinstance(node, ast.ClassDef):
            classes.append({'name': node.name, 'line': node.lineno,
                            'bases': [ast.unparse(b) for b in node.bases],
                            'methods': [x.name for x in node.body if isinstance(x, ast.FunctionDef)]})
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None or name(node.type) in ('Exception', 'BaseException'):
                flags.append({'kind': 'broad_exception', 'line': node.lineno,
                              'body': ast.unparse(ast.Module(body=node.body, type_ignores=[]))[:220]})
        elif isinstance(node, ast.Call):
            call = name(node.func)
            if call in ('eval', 'exec', 'pickle.load', 'pickle.loads', 'torch.load'):
                flags.append({'kind': call, 'line': node.lineno})
            if call.endswith('.add_argument'):
                if any(isinstance(x, ast.Constant) and x.value == '--tag' for x in node.args):
                    flags.append({'kind': 'tag_argument', 'line': node.lineno,
                                  'arguments': ast.unparse(node)})
    return {'doc': (ast.get_docstring(tree) or '').split('\n')[0],
            'imports': imports, 'classes': classes, 'functions': functions, 'flags': flags}


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    records = []
    for path in sorted(x for x in tracked if x):
        p = ROOT / path
        if not p.is_file():
            records.append({'path': path, 'status': 'missing'})
            continue
        raw = p.read_bytes()
        rec = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
               'kind': p.suffix or 'extensionless', 'status': 'bytes_read'}
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            rec['encoding'] = 'binary_or_non_utf8'
            records.append(rec)
            continue
        rec['lines'] = len(text.splitlines())
        if p.suffix == '.py':
            try:
                rec.update(summarize_python(text, path))
                rec['status'] = 'ast_read'
            except SyntaxError as exc:
                rec['parse_error'] = f'{exc.lineno}: {exc.msg}'
        elif p.suffix == '.ipynb':
            try:
                cells = json.loads(text)['cells']
                code = [''.join(c.get('source', [])) for c in cells if c.get('cell_type') == 'code']
                rec['notebook_code_cells'] = len(code)
                rec['notebook_code_lines'] = sum(len(s.splitlines()) for s in code)
                rec['notebook_headings'] = [''.join(c.get('source', [])).split('\n')[0]
                                            for c in cells if c.get('cell_type') == 'markdown']
            except (ValueError, KeyError) as exc:
                rec['parse_error'] = str(exc)
        elif p.suffix in SOURCE:
            rec['source_read'] = True
        records.append(rec)

    modules = {r['path'][:-3].replace('/', '.').removesuffix('.__init__'): r['path']
               for r in records if r['path'].endswith('.py')}
    edges = []
    for rec in records:
        path = rec['path']
        current = path[:-3].replace('/', '.')
        package = current.removesuffix('.__init__') if current.endswith('.__init__') else current.rpartition('.')[0]
        for imp in rec.get('imports', []):
            level = imp['level']
            if level:
                parent = package.split('.')
                if level > 1:
                    parent = parent[:-(level - 1)]
                target = '.'.join([*parent, *([imp['module']] if imp['module'] else [])])
            else:
                target = imp['module']
            candidates = [target, *(target + '.' + n for n in imp.get('names', []))]
            for cand in candidates:
                if cand in modules:
                    edges.append({'from': path, 'to': modules[cand], 'line': imp['line']})

    duplicates = collections.defaultdict(list)
    for rec in records:
        for fn in rec.get('functions', []):
            if fn['end'] - fn['line'] >= 8:
                duplicates[fn['body_hash']].append({'path': rec['path'], 'name': fn['name'],
                                                   'line': fn['line'], 'lines': fn['end']-fn['line']+1})
    duplicate_groups = sorted((v for v in duplicates.values() if len(v) > 1),
                              key=lambda v: len(v) * v[0]['lines'], reverse=True)
    output = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Tracked files, current working-tree bytes; AST inventory is not manual semantic certification.',
              'files': records, 'import_edges': edges, 'exact_body_duplicates': duplicate_groups}
    (OUT / 'inventory.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'head': output['head'], 'files': len(records),
                      'python_ast': sum(r['status'] == 'ast_read' for r in records),
                      'parse_errors': [(r['path'],r['parse_error']) for r in records if 'parse_error' in r],
                      'import_edges': len(edges), 'duplicate_groups': len(duplicate_groups)}, indent=2))


if __name__ == '__main__':
    main()
